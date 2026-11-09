from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError

from src.data_handlers.config import MongoConfig

logger = logging.getLogger(__name__)


class MongoHandler:
    def __init__(
        self,
        config: MongoConfig,
        server_selection_timeout_ms: int = 10_000,
        connect_timeout_ms: int = 10_000,
    ) -> None:
        self.config = config
        self._client: MongoClient | None = None

        self._client_kwargs = {
            "serverSelectionTimeoutMS": server_selection_timeout_ms,
            "connectTimeoutMS": connect_timeout_ms,
        }

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            self._client = MongoClient(
                self.config.build_uri(),
                **self._client_kwargs,
            )

        return self._client

    @property
    def database(self):
        return self.client[self.config.database]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> MongoHandler:
        _ = self.client
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def get_collection(self, collection_name: str) -> Collection:
        return self.database[collection_name]

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.database.list_collection_names()

    def create_indexes(
        self,
        collection_name: str,
        indexes: list[tuple[str, int]] | list[list[tuple[str, int]]],
        unique: bool = False,
    ) -> None:
        collection = self.get_collection(collection_name)

        if isinstance(indexes[0], tuple):
            keys = [indexes]
        else:
            keys = indexes

        collection.create_indexes(
            [pymongo_index for pymongo_index in self._build_index_models(keys, unique)]
        )

    @staticmethod
    def _build_index_models(
        keys: list[list[tuple[str, int]]],
        unique: bool,
    ):
        from pymongo import IndexModel

        return [IndexModel(key, unique=unique) for key in keys]

    def drop_collection(self, collection_name: str) -> None:
        self.database[collection_name].drop()

    def load_dataframe(
        self,
        df: pd.DataFrame,
        collection_name: str,
        mode: str = "append",
        batch_size: int = 10_000,
        ordered: bool = False,
        index_fields: list[str] | list[list[str]] | None = None,
        unique_index: bool = False,
    ) -> int:
        if df.empty:
            logger.warning(
                "DataFrame для коллекции %s пустой; загрузка пропущена.",
                collection_name,
            )
            return 0

        if mode not in {"append", "replace"}:
            raise ValueError("mode должен быть 'append' или 'replace'")

        if mode == "replace":
            self.drop_collection(collection_name)

        if index_fields:
            self.create_indexes(
                collection_name=collection_name,
                indexes=index_fields,
                unique=unique_index,
            )

        collection = self.get_collection(collection_name)

        inserted_count = 0

        for start in range(0, len(df), batch_size):
            # Документы строим по батчам, а не для всего датафрейма сразу.
            batch = self._dataframe_to_documents(df.iloc[start : start + batch_size])

            try:
                result = collection.insert_many(
                    documents=batch,
                    ordered=ordered,
                )
                inserted_count += len(result.inserted_ids)

            except BulkWriteError as error:
                inserted = error.details.get("nInserted", 0)
                inserted_count += inserted

                logger.error(
                    "Bulk insert в %s частично завершился ошибкой: %s",
                    collection_name,
                    error.details,
                )
                raise

        logger.info(
            "Загружено %d документов в %s.%s",
            inserted_count,
            self.config.database,
            collection_name,
        )

        return inserted_count

    def find_dataframe(
        self,
        collection_name: str,
        query: Mapping[str, Any] | None = None,
        projection: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        collection = self.get_collection(collection_name)
        cursor = collection.find(query or {}, projection)

        if limit is not None:
            cursor = cursor.limit(limit)

        return pd.DataFrame(list(cursor))

    @staticmethod
    def _dataframe_to_documents(df: pd.DataFrame) -> list[dict[str, Any]]:
        result = df.copy()

        for column in result.columns:
            if pd.api.types.is_datetime64_any_dtype(result[column]):
                result[column] = pd.to_datetime(result[column], utc=True)

            elif pd.api.types.is_object_dtype(result[column]):
                result[column] = result[column].where(result[column].notna(), None)

        result = result.replace({np.nan: None})
        return result.to_dict(orient="records")