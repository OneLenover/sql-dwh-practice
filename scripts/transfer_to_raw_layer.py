from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

from scripts.init_raw_data import CLICKHOUSE_ORDER_BY, check_env
from src.data_handlers.pyspark_handler import JdbcConfig, PySparkHandler

logger = logging.getLogger(__name__)

POSTGRES_DRIVER = "org.postgresql.Driver"
CLICKHOUSE_DRIVER = "com.clickhouse.jdbc.ClickHouseDriver"


# Оба ключа создаются генератором источников и монотонно возрастают
# в пределах одной генерации набора данных.
INCREMENTAL_COLUMNS = {
    "interactions": "interaction_id",
    "orders_history": "order_history_id",
}

# ClickHouse UUID читаем как String: иначе JDBC возвращает UUID как OTHER,
# который Spark JDBC не может отобразить в собственный тип.

CLICKHOUSE_READ_QUERIES = {
    "interactions": """
        SELECT
            interaction_id,
            toString(user_id) AS user_id,
            product_id,
            datetime,
            interaction_type,
            toString(order_id) AS order_id
        FROM default.interactions
    """,

    "orders_history": """
        SELECT
            order_history_id,
            toString(user_id) AS user_id,
            product_id,
            datetime,
            toString(order_id) AS order_id,
            toString(shop_id) AS shop_id,
            stage
        FROM default.orders_history
    """,
}


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Не задана переменная окружения {name}")
    return value


def build_mongodb_uri(
    host: str,
    port: int,
    username: str,
    password: str,
    database: str,
    collection: str,
    auth_source: str = "admin",
) -> str:
    user_encoded = quote_plus(username)
    password_encoded = quote_plus(password)
    return (
        f"mongodb://{user_encoded}:{password_encoded}@{host}:{port}/"
        f"{database}.{collection}?authSource={auth_source}"
    )



def create_clickhouse_table_in_dwh(
    spark_handler: PySparkHandler,
    dwh_connection: JdbcConfig,
    df: SparkDataFrame,
    table_name: str,
    database: str = "raw_layer",
) -> None:
    """Создаёт таблицу или добавляет отсутствующие колонки в существующую."""

    target = spark_handler.quote_table(database, table_name)

    if spark_handler.clickhouse_table_exists(
        dwh_connection,
        database,
        table_name,
    ):
        # Получаем текущую структуру таблицы DWH.
        existing_schema = spark_handler.read_jdbc_query(
            dwh_connection,
            f"""
                SELECT name
                FROM system.columns
                WHERE database = '{database}'
                  AND table = '{table_name}'
            """,
        )

        existing_columns = {
            row["name"] for row in existing_schema.collect()
        }

        # Добавляем новые колонки, которых ещё нет в DWH.
        for field in df.schema.fields:
            if field.name in existing_columns:
                continue

            column_name = spark_handler.quote_identifier(field.name)

            column_type = spark_handler.clickhouse_type(
                field.dataType,
                nullable=field.nullable,
            )

            spark_handler.execute_jdbc(
                dwh_connection,
                f"""
                    ALTER TABLE {target}
                    ADD COLUMN IF NOT EXISTS
                    {column_name} {column_type}
                """,
            )

            logger.info(
                "Добавлена колонка %s в %s.%s (%s)",
                field.name,
                database,
                table_name,
                column_type,
            )

        return

    # Если таблицы ещё нет, создаём её по схеме Spark DataFrame.
    columns_ddl = ",\n".join(
        f"{spark_handler.quote_identifier(field.name)} "
        f"{spark_handler.clickhouse_type(field.dataType, nullable=field.nullable)}"
        for field in df.schema.fields
    )

    order_by = CLICKHOUSE_ORDER_BY.get(table_name, "tuple()")

    sql = f"""
        CREATE TABLE IF NOT EXISTS {target}
        (
            {columns_ddl}
        )
        ENGINE = MergeTree
        ORDER BY {order_by}
        SETTINGS allow_nullable_key = 1
    """

    spark_handler.execute_jdbc(dwh_connection, sql)

    logger.info(
        "Создана таблица %s.%s",
        database,
        table_name,
    )


def load_dataframe_to_raw_layer(
    spark_handler: PySparkHandler,
    dwh_connection: JdbcConfig,
    df: SparkDataFrame,
    table_name: str,
    mode: str,
    database: str = "raw_layer",
    incremental_column: str | None = None,
) -> int:
    """Загружает Spark DataFrame в raw_layer.

    replace: сохраняет структуру таблицы и очищает данные через TRUNCATE.
    append: загружает только строки с ключом больше максимального ключа в DWH.
    """
    if mode not in {"replace", "append"}:
        raise ValueError(f"Неизвестный режим загрузки: {mode}")

    df = spark_handler.normalize_for_clickhouse(df)
    if not df.columns:
        logger.warning("Таблица %s не содержит колонок — пропускаем", table_name)
        return 0

    table_exists = spark_handler.clickhouse_table_exists(
        dwh_connection, database, table_name
    )
    create_clickhouse_table_in_dwh(
        spark_handler=spark_handler,
        dwh_connection=dwh_connection,
        df=df,
        table_name=table_name,
        database=database,
    )

    target = spark_handler.quote_table(database, table_name)

    if mode == "replace":
        # TRUNCATE, в отличие от overwrite, не удаляет и не пересоздаёт таблицу.
        spark_handler.execute_jdbc(
            dwh_connection,
            f"TRUNCATE TABLE {target}",
        )

    else:
        if not incremental_column:
            raise RuntimeError(
                f"Для append-таблицы {table_name} не задан инкрементальный ключ. "
                "Укажите в .env RAW_INCREMENTAL_COLUMN_* уникальную возрастающую "
                "числовую колонку. Без неё повторный запуск создаст дубликаты."
            )
        if incremental_column not in df.columns:
            raise ValueError(
                f"В {table_name} нет колонки {incremental_column!r}. "
                f"Доступные колонки: {df.columns}"
            )

        if table_exists:
            existing_count, last_value = spark_handler.clickhouse_table_state(
                dwh_connection,
                database,
                table_name,
                incremental_column,
            )
            if existing_count > 0:
                if last_value is None:
                    raise RuntimeError(
                        f"В {database}.{table_name} есть строки, но max({incremental_column}) = NULL"
                    )
                # Фильтр выполняется Spark. Для JDBC источников Spark обычно
                # передаёт такой предикат в исходную БД (predicate pushdown).
                df = df.filter(F.col(f"`{incremental_column}`") > F.lit(last_value))

    rows_to_load = df.count()
    if rows_to_load == 0:
        logger.info("Для %s новых строк нет", table_name)
        return 0

    # Ни toPandas(), ни pandas DataFrame здесь нет: запись остаётся распределённой.
    spark_handler.write_jdbc(
        df=df,
        connection=dwh_connection,
        table=f"{database}.{table_name}",
        batch_size=10_000,
        num_partitions=4,
    )

    logger.info("В raw_layer.%s загружено строк: %s", table_name, rows_to_load)
    return rows_to_load


def transfer_data_to_raw_layer() -> dict[str, int]:
    check_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Адреса источников задаются через окружение.
    # Локальный запуск сохраняет localhost и опубликованные на хост порты,
    # а Airflow в Docker использует DNS-имена сервисов и внутренние порты.
    pg_main_host = os.getenv("POSTGRES_MAIN_HOST", "localhost")
    pg_main_port = int(os.getenv("POSTGRES_MAIN_PORT", "5432"))
    pg_ml_host = os.getenv("POSTGRES_ML_HOST", "localhost")
    pg_ml_port = int(os.getenv("POSTGRES_ML_PORT", "5433"))

    clickhouse_source_host = os.getenv("CLICKHOUSE_SOURCE_HOST", "localhost")
    clickhouse_source_port = int(os.getenv("CLICKHOUSE_SOURCE_PORT", "8123"))
    clickhouse_dwh_host = os.getenv("CLICKHOUSE_DWH_HOST", "localhost")
    clickhouse_dwh_port = int(os.getenv("CLICKHOUSE_DWH_PORT", "8124"))

    mongo_host = os.getenv("MONGO_HOST", "localhost")
    mongo_port = int(os.getenv("MONGO_PORT", "27017"))
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    # Источники PostgreSQL.
    pg_main = JdbcConfig(
        url=(
            f"jdbc:postgresql://{pg_main_host}:{pg_main_port}/"
            f"{os.getenv('POSTGRES_MAIN_DB', required_env('POSTGRES_MAIN_USER'))}"
        ),
        driver=POSTGRES_DRIVER,
        user=required_env("POSTGRES_MAIN_USER"),
        password=required_env("POSTGRES_MAIN_PASSWORD"),
    )
    pg_ml = JdbcConfig(
        url=(
            f"jdbc:postgresql://{pg_ml_host}:{pg_ml_port}/"
            f"{os.getenv('POSTGRES_ML_DB', required_env('POSTGRES_ML_USER'))}"
        ),
        driver=POSTGRES_DRIVER,
        user=required_env("POSTGRES_ML_USER"),
        password=required_env("POSTGRES_ML_PASSWORD"),
    )

    # Исходный ClickHouse и отдельный ClickHouse-инстанс DWH.
    clickhouse_src = JdbcConfig(
        url=f"jdbc:clickhouse:http://{clickhouse_source_host}:{clickhouse_source_port}/default",
        driver=CLICKHOUSE_DRIVER,
        user=required_env("CLICKHOUSE_USER"),
        password=required_env("CLICKHOUSE_PASSWORD"),
    )
    clickhouse_dwh = JdbcConfig(
        url=f"jdbc:clickhouse:http://{clickhouse_dwh_host}:{clickhouse_dwh_port}/default",
        driver=CLICKHOUSE_DRIVER,
        user=required_env("CLICKHOUSE_USER"),
        password=required_env("CLICKHOUSE_PASSWORD"),
    )

    mongo_uri = build_mongodb_uri(
        host=mongo_host,
        port=mongo_port,
        username=required_env("MONGO_USER"),
        password=required_env("MONGO_PASSWORD"),
        database="shop",
        collection="competitors_products_price_history",
        auth_source="admin",
    )
    redis_password = required_env("REDIS_PASSWORD")

    spark_config = {
        "spark.sql.shuffle.partitions": os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"),
        "spark.sql.session.timeZone": "UTC",
        "spark.redis.host": redis_host,
        "spark.redis.port": str(redis_port),
    }

    loaded_stats: dict[str, int] = {}

    # Весь перенос выполняется через один SparkSession.
    with PySparkHandler(config=spark_config) as spark_handler:
        # Инициализируем три базы DWH.
        for database in ("raw_layer", "stage_layer", "data_marts"):
            spark_handler.execute_jdbc(
                clickhouse_dwh,
                f"CREATE DATABASE IF NOT EXISTS {spark_handler.quote_identifier(database)}",
            )

        # 1–3. PostgreSQL main: products, shops, users.
        for table_name in ("products", "shops", "users"):
            logger.info("Чтение PostgreSQL main: %s", table_name)
            source_df = spark_handler.read_from_postgres(
                pg_main,
                table_name,
                schema="public",
            )
            loaded_stats[table_name] = load_dataframe_to_raw_layer(
                spark_handler,
                clickhouse_dwh,
                source_df,
                table_name,
                mode="replace",
            )

        # 4–7. Исходный ClickHouse.
        clickhouse_jobs = (
            ("interactions", "append"),
            ("orders_history", "append"),
            ("products_price_history", "replace"),
            ("delivery_price_history", "replace"),
        )
        for table_name, mode in clickhouse_jobs:
            logger.info("Чтение ClickHouse source: %s (mode=%s)", table_name, mode)

            if table_name in CLICKHOUSE_READ_QUERIES:
                source_df = spark_handler.read_jdbc_query(
                    clickhouse_src,
                    CLICKHOUSE_READ_QUERIES[table_name],
                )
            else:
                source_df = spark_handler.read_from_clickhouse(
                    clickhouse_src,
                    table_name,
                    database="default",
                )

            loaded_stats[table_name] = load_dataframe_to_raw_layer(
                spark_handler,
                clickhouse_dwh,
                source_df,
                table_name,
                mode=mode,
                incremental_column=INCREMENTAL_COLUMNS.get(table_name),
            )

        # 8. MongoDB.
        mongo_table = "competitors_products_price_history"
        logger.info("Чтение MongoDB: %s", mongo_table)
        mongo_df = spark_handler.read_from_mongodb(mongo_uri)
        # MongoDB ObjectId в raw_layer сохраняем строковым представлением.
        if "_id" in mongo_df.columns:
            mongo_df = mongo_df.withColumn("_id", F.col("_id").cast("string"))
        loaded_stats[mongo_table] = load_dataframe_to_raw_layer(
            spark_handler,
            clickhouse_dwh,
            mongo_df,
            mongo_table,
            mode="replace",
        )

        # 9–11. PostgreSQL ML: прогнозы.
        ml_tables = (
            "shops_daily_completed_orders_predict",
            "shops_daily_all_orders_predict",
            "daily_revenue_predict",
        )
        for table_name in ml_tables:
            logger.info("Чтение PostgreSQL ML: %s", table_name)
            source_df = spark_handler.read_from_postgres(
                pg_ml,
                table_name,
                schema="public",
            )
            loaded_stats[table_name] = load_dataframe_to_raw_layer(
                spark_handler,
                clickhouse_dwh,
                source_df,
                table_name,
                mode="replace",
            )

        # 12–13. Redis Hash.
        redis_jobs = (
            ("users_cities", "user:*:city"),
            ("users_clusters", "user:*:cluster"),
        )
        for table_name, key_pattern in redis_jobs:
            logger.info("Чтение Redis: %s, pattern=%s", table_name, key_pattern)
            redis_df = spark_handler.read_from_redis_hashes(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                key_pattern=key_pattern,
            )

            if redis_df.limit(1).count() == 0:
                logger.warning("В Redis не найдены данные для %s", table_name)
                loaded_stats[table_name] = 0
                continue

            loaded_stats[table_name] = load_dataframe_to_raw_layer(
                spark_handler,
                clickhouse_dwh,
                redis_df,
                table_name,
                mode="replace",
            )

    logger.info("Перенос в raw_layer завершён. Статистика: %s", loaded_stats)
    return loaded_stats


if __name__ == "__main__":
    transfer_data_to_raw_layer()
