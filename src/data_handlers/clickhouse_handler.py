from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import clickhouse_connect
import pandas as pd
from clickhouse_connect.driver import Client

from src.data_handlers.config import ClickhouseConfig

logger = logging.getLogger(__name__)


class ClickHouseHandler:
    def __init__(
        self,
        config: ClickhouseConfig,
        pool_mgr: Any | None = None,
    ) -> None:
        self.config = config
        self._client: Client | None = None
        self._pool_mgr = pool_mgr

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
                secure=self.config.secure,
                connect_timeout=self.config.connect_timeout,
                pool_mgr=self._pool_mgr,
            )

        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> ClickHouseHandler:
        _ = self.client
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def execute(self, query: str, parameters: Mapping[str, Any] | None = None) -> None:
        self.client.command(query, parameters=parameters)

    def query_df(
        self, query: str, parameters: Mapping[str, Any] | None = None
    ) -> pd.DataFrame:
        return self.client.query_df(query, parameters=parameters)

    def table_exists(self, table_name: str, database: str | None = None) -> bool:
        database = database or self.config.database

        result = self.client.query(
            """
            SELECT 1
            FROM system.tables
            WHERE database = {database:String}
              AND name = {table:String}
            LIMIT 1
            """,
            parameters={"database": database, "table": table_name},
        )

        return len(result.result_rows) > 0

    def truncate_table(self, table_name: str, database: str | None = None) -> None:
        database = database or self.config.database
        self.execute(f'TRUNCATE TABLE "{database}"."{table_name}"')

    def drop_table(self, table_name: str, database: str | None = None) -> None:
        database = database or self.config.database
        self.execute(f'DROP TABLE IF EXISTS "{database}"."{table_name}"')

    def create_table(
        self,
        table_name: str,
        ddl: str,
        database: str | None = None,
        if_not_exists: bool = True,
    ) -> None:
        database = database or self.config.database
        if_not_exists_sql = "IF NOT EXISTS " if if_not_exists else ""

        query = f'CREATE TABLE {if_not_exists_sql}"{database}"."{table_name}" ({ddl})'
        self.execute(query)

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        database: str | None = None,
        mode: str = "append",
        chunk_size: int = 100_000,
        column_types: Mapping[str, str] | None = None,
    ) -> int:
        if df.empty:
            logger.warning("DataFrame для %s пустой; загрузка пропущена.", table_name)
            return 0

        if mode not in {"append", "replace"}:
            raise ValueError("mode должен быть 'append' или 'replace'")

        database = database or self.config.database

        if mode == "replace":
            self.truncate_table(table_name, database=database)

        columns = list(df.columns)

        total_rows = 0

        for start in range(0, len(df), chunk_size):
            # Подготовка по чанкам: не копируем весь датафрейм целиком.
            chunk = self._prepare_dataframe(df.iloc[start : start + chunk_size])

            self.client.insert_df(
                table=f'"{database}"."{table_name}"',
                df=chunk,
                column_names=columns,
                column_type_names=(
                    [column_types[column] for column in columns]
                    if column_types
                    else None
                ),
            )

            total_rows += len(chunk)

        logger.info(
            "Загружено %d строк в %s.%s",
            total_rows,
            database,
            table_name,
        )

        return total_rows

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for column in result.columns:
            if pd.api.types.is_datetime64_any_dtype(result[column]):
                result[column] = pd.to_datetime(result[column], utc=True)

            elif pd.api.types.is_object_dtype(result[column]):
                result[column] = result[column].where(result[column].notna(), None)

        return result