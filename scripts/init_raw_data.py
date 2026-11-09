import os
import logging
import pandas as pd
import numpy as np
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from src.data_handlers.pg_handler import PostgresHandler
from src.data_handlers.clickhouse_handler import ClickHouseHandler
from src.data_handlers.mongodb_handler import MongoHandler
from src.data_handlers.redis_handler import RedisHandler

from src.data_handlers.config import (
    PostgresConfig,
    ClickhouseConfig,
    MongoConfig,
    RedisConfig,
)

from src.data_generation.clickhouse import (
    gen_delivery_price_history_dataframe,
    gen_products_price_history_dataframe,
    generate_interaction_history_dataframe,
    generate_orders_history_dataframe,
)
from src.data_generation.mongodb import gen_competitors_products_price_history_dataframe
from src.data_generation.pg_ml import (
    gen_daily_revenue_predict,
    iter_shops_daily_orders_predict,
)
from src.data_generation.pg_prod import (
    generate_products_dataframe,
    generate_shops_dataframe,
    generate_users_dataframe,
)
from src.data_generation.redis import (
    gen_users_cities_dataframe,
    gen_users_clusters_dataframe,
)

UNIQUE_USERS_NUM = int(os.getenv("UNIQUE_USERS_NUM", "10000"))
UNIQUE_SHOPS_NUM = int(os.getenv("UNIQUE_SHOPS_NUM", "300"))
NUM_UNIQUE_INTERACTIONS = int(os.getenv("NUM_UNIQUE_INTERACTIONS", "1000000"))
INTERACTIONS_BATCH_SIZE = int(os.getenv("INTERACTIONS_BATCH_SIZE", "2000000"))
ML_DAYS_PER_CHUNK = int(os.getenv("ML_DAYS_PER_CHUNK", "14"))

REQUIRED_ENV = [
    "POSTGRES_MAIN_USER",
    "POSTGRES_MAIN_PASSWORD",
    "POSTGRES_ML_USER",
    "POSTGRES_ML_PASSWORD",
    "CLICKHOUSE_USER",
    "CLICKHOUSE_PASSWORD",
    "MONGO_USER",
    "MONGO_PASSWORD",
    "REDIS_PASSWORD",
]

# Ключ сортировки для таблиц ClickHouse (источник, поэтому без партиций).
CLICKHOUSE_ORDER_BY = {
    "interactions": "(datetime, interaction_id)",
    "orders_history": "(order_id, datetime)",
    "products_price_history": "(product_id, effective_from)",
    "delivery_price_history": "(object_id, effective_from)",
}


def check_env() -> None:
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise SystemExit(
            "Не заданы переменные окружения: "
            + ", ".join(missing)
            + ". Скопируйте .env.example в .env и заполните."
        )


def init_databases_connections() -> dict:
    connections = {
        "pg_main": PostgresHandler(
            PostgresConfig(
                host="localhost",
                port=5432,
                dbname=os.getenv("POSTGRES_MAIN_USER"),
                user=os.getenv("POSTGRES_MAIN_USER"),
                password=os.getenv("POSTGRES_MAIN_PASSWORD"),
            )
        ),
        "pg_ml": PostgresHandler(
            PostgresConfig(
                host="localhost",
                port=5433,
                dbname=os.getenv("POSTGRES_ML_USER"),
                user=os.getenv("POSTGRES_ML_USER"),
                password=os.getenv("POSTGRES_ML_PASSWORD"),
            )
        ),
        "clickhouse": ClickHouseHandler(
            ClickhouseConfig(
                host="localhost",
                port=8123,
                username=os.getenv("CLICKHOUSE_USER"),
                password=os.getenv("CLICKHOUSE_PASSWORD"),
                database="default",
            )
        ),
        "mongodb": MongoHandler(
            MongoConfig(
                host="localhost",
                port=27017,
                username=os.getenv("MONGO_USER"),
                password=os.getenv("MONGO_PASSWORD"),
                database="shop",
                auth_source="admin",
            )
        ),
        "redis": RedisHandler(
            RedisConfig(
                host="localhost",
                port=6379,
                password=os.getenv("REDIS_PASSWORD"),
            )
        )
    }

    try:
        for name in ("pg_main", "pg_ml"):
            with connections[name].engine.connect():
                logging.info("%s: подключение успешно", name)

        if not connections["clickhouse"].client.ping():
            raise Exception("Clickhouse: подключение не удалось")

        logging.info("clickhouse: подключение успешно")

        connections["mongodb"].client.admin.command("ping")
        logging.info("mongodb: подключение успешно")

        connections["redis"].client.ping()
        logging.info("redis: подключение успешно")

    except Exception:
        for handler in connections.values():
            handler.close()
        raise

    return connections

def get_clickhouse_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        ch_type = "Bool"
    elif pd.api.types.is_integer_dtype(series):
        ch_type = "Int64"

    elif pd.api.types.is_float_dtype(series):
        ch_type = "Float64"

    elif pd.api.types.is_datetime64_any_dtype(series):
        ch_type = "DateTime64(3, 'UTC')"

    else:
        sample = series.dropna()

        if sample.empty:
            ch_type = "String"
        else:
            value = sample.iloc[0]

            if isinstance(value, UUID):
                ch_type = "UUID"
            elif isinstance(value, datetime):
                ch_type = "DateTime64(3, 'UTC')"
            elif isinstance(value, date):
                ch_type = "Date"
            elif isinstance(value, Decimal):
                ch_type = "Decimal(18, 4)"
            elif isinstance(value, (bool, np.bool_)):
                ch_type = "Bool"
            elif isinstance(value, (int, np.integer)):
                ch_type = "Int64"
            elif isinstance(value, (float, np.floating)):
                ch_type = "Float64"
            else:
                ch_type = "String"

    if series.isna().any():
        ch_type = f"Nullable({ch_type})"

    return ch_type

def create_clickhouse_table(
    handler: ClickHouseHandler,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    if handler.table_exists(table_name):
        return

    columns_ddl = ",\n".join(
        f"`{column}` {get_clickhouse_column_type(df[column])}" for column in df.columns
    )
    order_by = CLICKHOUSE_ORDER_BY.get(table_name, "tuple()")

    handler.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `default`.`{table_name}`
        (
            {columns_ddl}
        )
        ENGINE = MergeTree
        ORDER BY {order_by}
        """
    )


def clear_redis_pattern(handler: RedisHandler, pattern: str) -> None:
    pipe = handler.client.pipeline(transaction=False)
    count = 0

    for key in handler.client.scan_iter(match=pattern, count=5000):
        pipe.delete(key)
        count += 1

        if count % 5000 == 0:
            pipe.execute()
            pipe = handler.client.pipeline(transaction=False)

    pipe.execute()


def load_pg_batches(handler: PostgresHandler, table_name: str, batches) -> int:
    """Первая пачка пересоздаёт таблицу, остальные дозаписываются."""
    total = 0
    for i, df in enumerate(batches):
        total += handler.load_dataframe(
            df=df, table_name=table_name, mode="replace" if i == 0 else "append"
        )
    return total


def load_clickhouse_batches(
    handler: ClickHouseHandler, table_name: str, batches
) -> int:
    """Первая пачка создаёт таблицу и очищает её, остальные дозаписываются."""
    total = 0
    for i, df in enumerate(batches):
        if i == 0:
            create_clickhouse_table(handler, df, table_name)
            handler.truncate_table(table_name)
        total += handler.load_dataframe(df=df, table_name=table_name, mode="append")
    return total


def iter_interactions_and_orders(products_df, users_df):
    """Пачками генерирует interactions и выведенный из них orders_history."""
    offset = 0
    while offset < NUM_UNIQUE_INTERACTIONS:
        batch_size = min(INTERACTIONS_BATCH_SIZE, NUM_UNIQUE_INTERACTIONS - offset)
        interactions = generate_interaction_history_dataframe(
            products_df, users_df, batch_size, id_offset=offset
        )
        orders = generate_orders_history_dataframe(interactions)
        offset += len(interactions)
        yield interactions, orders


def load_data_to_databases(connections: dict) -> dict[str, dict[str, int]]:
    """Генерирует данные по одной таблице и сразу заливает в источник."""
    loaded: dict[str, dict[str, int]] = {
        "pg_main": {},
        "pg_ml": {},
        "clickhouse": {},
        "mongodb": {},
        "redis": {},
    }

    # --- Базовые справочники (небольшие, живут в памяти до конца) ---
    products_df = generate_products_dataframe(
        pd.read_csv(PROJECT_ROOT / "data" / "products.csv", sep=";")
    )
    shops_df = generate_shops_dataframe(UNIQUE_SHOPS_NUM)
    users_df = generate_users_dataframe(UNIQUE_USERS_NUM)

    # 1. PostgreSQL: основной источник
    pg_main = connections["pg_main"]
    for name, df in (("products", products_df), ("shops", shops_df), ("users", users_df)):
        loaded["pg_main"][name] = load_pg_batches(pg_main, name, [df])

    # 2. Redis
    redis = connections["redis"]
    users_cities_df = gen_users_cities_dataframe(users_df.copy())
    users_clusters_df = gen_users_clusters_dataframe(users_df.copy())

    # Удаляем только ключи, относящиеся к этим наборам данных.
    clear_redis_pattern(redis, "user:*:city")
    clear_redis_pattern(redis, "user:*:cluster")
    loaded["redis"]["users_cities"] = redis.load_dataframe_as_hashes(
        df=users_cities_df,
        key_template="user:{user_id}:city",
        key_fields=["user_id"],
        value_fields=["city"],
    )
    loaded["redis"]["users_clusters"] = redis.load_dataframe_as_hashes(
        df=users_clusters_df,
        key_template="user:{user_id}:cluster",
        key_fields=["user_id"],
        value_fields=["price_cluster"],
    )
    del users_cities_df, users_clusters_df

    # 3. MongoDB
    competitors_df = gen_competitors_products_price_history_dataframe(products_df.copy())
    loaded["mongodb"]["competitors_products_price_history"] = connections[
        "mongodb"
    ].load_dataframe(
        df=competitors_df,
        collection_name="competitors_products_price_history",
        mode="replace",
    )
    del competitors_df

    # 4. ClickHouse: небольшие таблицы цен
    ch = connections["clickhouse"]
    for name, df in (
        ("products_price_history", gen_products_price_history_dataframe(products_df.copy())),
        ("delivery_price_history", gen_delivery_price_history_dataframe(shops_df.copy())),
    ):
        loaded["clickhouse"][name] = load_clickhouse_batches(ch, name, [df])
        del df

    # 5. ClickHouse: interactions + orders_history (большие, пачками)
    interactions_total = orders_total = 0
    first = True
    for interactions, orders in iter_interactions_and_orders(products_df, users_df):
        if first:
            for name, df in (("interactions", interactions), ("orders_history", orders)):
                create_clickhouse_table(ch, df, name)
                ch.truncate_table(name)
            first = False
        interactions_total += ch.load_dataframe(interactions, "interactions", mode="append")
        orders_total += ch.load_dataframe(orders, "orders_history", mode="append")
        del interactions, orders
    loaded["clickhouse"]["interactions"] = interactions_total
    loaded["clickhouse"]["orders_history"] = orders_total

    # 6. PostgreSQL ML: прогнозы (большие, пачками по creation_date)
    pg_ml = connections["pg_ml"]
    completed_total = all_total = 0
    for i, (completed_df, all_df) in enumerate(
        iter_shops_daily_orders_predict(shops_df, days_per_chunk=ML_DAYS_PER_CHUNK)
    ):
        mode = "replace" if i == 0 else "append"
        completed_total += pg_ml.load_dataframe(
            completed_df, "shops_daily_completed_orders_predict", mode=mode
        )
        all_total += pg_ml.load_dataframe(
            all_df, "shops_daily_all_orders_predict", mode=mode
        )
        del completed_df, all_df
    loaded["pg_ml"]["shops_daily_completed_orders_predict"] = completed_total
    loaded["pg_ml"]["shops_daily_all_orders_predict"] = all_total
    loaded["pg_ml"]["daily_revenue_predict"] = load_pg_batches(
        pg_ml, "daily_revenue_predict", [gen_daily_revenue_predict()]
    )

    return loaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_env()

    connections = init_databases_connections()

    try:
        print(load_data_to_databases(connections))
    finally:
        for handler in connections.values():
            handler.close()
