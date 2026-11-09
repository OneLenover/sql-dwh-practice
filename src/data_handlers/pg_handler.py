from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import Engine
from sqlalchemy.types import (
    BIGINT,
    BOOLEAN,
    DATE,
    DOUBLE_PRECISION,
    TEXT,
    TIMESTAMP,
    Interval,
)

from src.data_handlers.config import PostgresConfig

logger = logging.getLogger(__name__)


class PostgresHandler:
    def __init__(
        self,
        config: PostgresConfig,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_pre_ping: bool = True,
        echo: bool = False,
    ) -> None:
        self.config = config
        self.engine: Engine = create_engine(
            config.build_url(),
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            echo=echo,
        )

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> PostgresHandler:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def table_exists(self, table_name: str, schema: str = "public") -> bool:
        return inspect(self.engine).has_table(table_name, schema=schema)

    def truncate_table(self, table_name: str, schema: str = "public") -> None:
        with self.engine.begin() as connection:
            connection.execute(text(f'TRUNCATE TABLE "{schema}"."{table_name}"'))

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str = "public",
        mode: str = "append",
        chunksize: int = 100_000,
        dtype: Mapping[str, Any] | None = None,
        index: bool = False,
    ) -> int:
        """Загружает DataFrame через COPY.

        mode="replace" пересоздаёт таблицу по схеме датафрейма,
        mode="append" дозаписывает (создаёт таблицу, если её нет),
        mode="fail" падает, если таблица уже существует.
        Возвращает реальное число записанных строк.
        """
        if df.empty:
            logger.warning("DataFrame для %s пустой; загрузка пропущена.", table_name)
            return 0

        if mode not in {"append", "replace", "fail"}:
            raise ValueError("mode должен быть 'append', 'replace' или 'fail'")

        prepared_df = self._prepare_dataframe(df)
        sa_types = dtype or self._infer_sqlalchemy_types(prepared_df)

        exists = self.table_exists(table_name, schema=schema)
        if exists and mode == "fail":
            raise ValueError(f"Таблица {schema}.{table_name} уже существует")

        if mode == "replace" or not exists:
            # Создаём таблицу по схеме датафрейма (без данных).
            prepared_df.head(0).to_sql(
                name=table_name,
                con=self.engine,
                schema=schema,
                if_exists="replace",
                index=index,
                dtype=sa_types,
            )

        columns = ", ".join(f'"{c}"' for c in prepared_df.columns)
        copy_sql = f'COPY "{schema}"."{table_name}" ({columns}) FROM STDIN'

        total = 0
        with self.engine.raw_connection() as raw:
            try:
                with raw.cursor() as cursor:
                    for start in range(0, len(prepared_df), chunksize):
                        chunk = self._to_python_objects(
                            prepared_df.iloc[start : start + chunksize]
                        )
                        with cursor.copy(copy_sql) as copy:
                            for row in chunk.itertuples(index=False, name=None):
                                copy.write_row(row)
                        total += len(chunk)
                raw.commit()
            except Exception:
                raw.rollback()
                raise

        logger.info("Загружено %d строк в %s.%s", total, schema, table_name)
        return total

    def replace_preserving_schema(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str = "public",
        chunksize: int = 100_000,
        dtype: Mapping[str, Any] | None = None,
    ) -> int:
        if not self.table_exists(table_name, schema=schema):
            raise ValueError(
                f"Таблица {schema}.{table_name} не существует; "
                "сначала создайте её или используйте load_dataframe(mode='replace')."
            )

        self.truncate_table(table_name=table_name, schema=schema)
        return self.load_dataframe(
            df=df,
            table_name=table_name,
            schema=schema,
            mode="append",
            chunksize=chunksize,
            dtype=dtype,
        )

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for column in result.columns:
            if pd.api.types.is_datetime64_any_dtype(result[column]):
                result[column] = pd.to_datetime(result[column], utc=True)
            elif pd.api.types.is_bool_dtype(result[column]):
                result[column] = result[column].astype("boolean")

        return result

    @staticmethod
    def _to_python_objects(df: pd.DataFrame) -> pd.DataFrame:
        """NaN / NA / NaT -> None, значения -> python-объекты (для COPY)."""
        result = pd.DataFrame(index=df.index)
        for column in df.columns:
            series = df[column]
            result[column] = series.astype(object).where(series.notna(), None)
        return result

    @staticmethod
    def _infer_sqlalchemy_types(df: pd.DataFrame) -> dict[str, Any]:
        inferred: dict[str, Any] = {}

        for column in df.columns:
            series = df[column]

            if pd.api.types.is_bool_dtype(series):
                inferred[column] = BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(series):
                inferred[column] = TIMESTAMP(timezone=True)
            elif pd.api.types.is_timedelta64_dtype(series):
                inferred[column] = Interval
            elif pd.api.types.is_integer_dtype(series):
                inferred[column] = BIGINT
            elif pd.api.types.is_float_dtype(series):
                inferred[column] = DOUBLE_PRECISION
            elif pd.api.types.is_object_dtype(series):
                sample = series.dropna()
                first = sample.iloc[0] if not sample.empty else None

                if isinstance(first, UUID):
                    inferred[column] = PG_UUID(as_uuid=True)
                elif isinstance(first, datetime):
                    inferred[column] = TIMESTAMP(timezone=True)
                elif isinstance(first, date):
                    inferred[column] = DATE
                else:
                    inferred[column] = TEXT
            else:
                inferred[column] = TEXT

        return inferred