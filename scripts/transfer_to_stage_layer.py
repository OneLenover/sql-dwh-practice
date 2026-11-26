
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from pyspark.sql import functions as F

from src.data_handlers.pyspark_handler import JdbcConfig, PySparkHandler


logger = logging.getLogger(__name__)

CLICKHOUSE_DRIVER = "com.clickhouse.jdbc.ClickHouseDriver"

STAGE_DATABASE = "stage_layer"
STAGE_TABLE = "competitors_products_price_history"


def required_env(name: str) -> str:
    """Возвращает обязательную переменную окружения."""
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Не задана переменная окружения {name}"
        )

    return value


def get_clickhouse_dwh_config() -> JdbcConfig:
    """Создаёт подключение к ClickHouse DWH."""

    host = os.getenv("CLICKHOUSE_DWH_HOST", "localhost")
    port = int(os.getenv("CLICKHOUSE_DWH_PORT", "8124"))

    return JdbcConfig(
        url=f"jdbc:clickhouse:http://{host}:{port}/default",
        driver=CLICKHOUSE_DRIVER,
        user=required_env("CLICKHOUSE_USER"),
        password=required_env("CLICKHOUSE_PASSWORD"),
    )


def build_competitor_prices_stage() -> dict[str, int]:
    """
    Создаёт Stage-витрину с историей цен конкурентов.

    Источник:
        raw_layer.competitors_products_price_history

    Результат:
        stage_layer.competitors_products_price_history

    Витрина обновляется полностью при каждом запуске.
    """

    logger.info("Начинаем построение истории цен конкурентов")

    dwh_connection = get_clickhouse_dwh_config()

    spark_config = {
        "spark.sql.shuffle.partitions": os.getenv(
            "SPARK_SHUFFLE_PARTITIONS", "8"
        ),
        "spark.sql.session.timeZone": "UTC",
    }

    with PySparkHandler(
        app_name="DWH_Stage_Competitor_Prices",
        config=spark_config,
    ) as spark_handler:
        
        # 1. Читаем исходные данные из Raw Layer

        logger.info(
            "Читаем raw_layer.%s",
            STAGE_TABLE,
        )

        raw_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name=STAGE_TABLE,
            database="raw_layer",
        )

        # 2. Приводим данные к нужной структуре

        stage_df = raw_df.select(
            F.col("parse_id").cast("string").alias("parse_id"),
            F.col("company_name").cast("string").alias("company_name"),
            F.col("product_id").cast("int").alias("product_id"),
            F.col("datetime").cast("timestamp").alias("datetime"),
            F.col("price").cast("double").alias("price"),
            F.col("cost").cast("int").alias("cost"),
            F.col("value").cast("int").alias("value"),
        )

        # 3. Проверяем корректность данных

        # У каждой строки должен быть заполнен ровно один
        # из трёх показателей: price, cost или value.

        measure_count = (
            F.col("price").isNotNull().cast("int")
            + F.col("cost").isNotNull().cast("int")
            + F.col("value").isNotNull().cast("int")
        )

        # Эти поля обязательны для идентификации записи.
        missing_key = (
            F.col("parse_id").isNull()
            | F.col("company_name").isNull()
            | F.col("product_id").isNull()
            | F.col("datetime").isNull()
        )

        validation = stage_df.agg(
            F.count("*").alias("row_count"),

            F.count(
                F.when(measure_count != 1, F.lit(1))
            ).alias("invalid_measure_rows"),

            F.count(
                F.when(missing_key, F.lit(1))
            ).alias("invalid_key_rows"),
        ).first()

        rows_to_load = int(validation["row_count"])
        invalid_measure_rows = int(validation["invalid_measure_rows"])
        invalid_key_rows = int(validation["invalid_key_rows"])

        if rows_to_load == 0:
            raise RuntimeError(
                "В Raw Layer нет данных для построения витрины"
            )

        if invalid_measure_rows > 0:
            raise ValueError(
                "Найдены строки, в которых заполнено не ровно одно "
                "из полей price, cost, value. "
                f"Количество: {invalid_measure_rows}"
            )

        if invalid_key_rows > 0:
            raise ValueError(
                "Найдены строки с незаполненными обязательными полями. "
                f"Количество: {invalid_key_rows}"
            )

        logger.info(
            "Проверка пройдена. Строк для загрузки: %s",
            rows_to_load,
        )

        # 4. Создаём Stage Layer и таблицу

        target = spark_handler.quote_table(
            STAGE_DATABASE,
            STAGE_TABLE,
        )

        spark_handler.execute_jdbc(
            dwh_connection,
            f"CREATE DATABASE IF NOT EXISTS "
            f"{spark_handler.quote_identifier(STAGE_DATABASE)}",
        )

        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {target}
            (
                `parse_id` String,
                `company_name` String,
                `product_id` Int32,
                `datetime` DateTime64(6, 'UTC'),
                `price` Nullable(Float64),
                `cost` Nullable(Int32),
                `value` Nullable(Int32)
            )
            ENGINE = MergeTree
            ORDER BY (`company_name`, `product_id`, `datetime`)
        """

        spark_handler.execute_jdbc(
            dwh_connection,
            create_table_sql,
        )

        # 5. Полностью обновляем таблицу

        # write_jdbc поддерживает append.
        # Поэтому перед записью очищаем старое содержимое таблицы.

        logger.info(
            "Очищаем таблицу %s.%s",
            STAGE_DATABASE,
            STAGE_TABLE,
        )

        spark_handler.execute_jdbc(
            dwh_connection,
            f"TRUNCATE TABLE {target}",
        )

        # Записываем Spark DataFrame в ClickHouse.
        spark_handler.write_jdbc(
            df=stage_df,
            connection=dwh_connection,
            table=f"{STAGE_DATABASE}.{STAGE_TABLE}",
            batch_size=10_000,
            num_partitions=4,
        )

        # 6. Проверяем результат загрузки

        result = spark_handler.read_jdbc_query(
            dwh_connection,
            f"SELECT count() AS row_count FROM {target}",
        ).first()

        loaded_rows = int(result["row_count"])

        if loaded_rows != rows_to_load:
            raise RuntimeError(
                "Количество загруженных строк не совпадает "
                f"с исходным: ожидалось {rows_to_load}, "
                f"загружено {loaded_rows}"
            )

        logger.info(
            "Stage Layer успешно обновлён. Загружено строк: %s",
            loaded_rows,
        )

        return {
            "source_rows": rows_to_load,
            "loaded_rows": loaded_rows,
        }

# 2. Витрины с актуальными ML-прогнозами

def _load_stage_dataframe(
    spark_handler: PySparkHandler,
    dwh_connection: JdbcConfig,
    df,
    table_name: str,
    order_by: str,
) -> int:
    """
    Полностью обновляет таблицу Stage Layer данными из Spark DataFrame.

    Если таблица ещё не существует, создаёт её по схеме DataFrame.
    """

    df = spark_handler.normalize_for_clickhouse(df)

    rows_to_load = df.count()

    if rows_to_load == 0:
        raise RuntimeError(
            f"Нет строк для загрузки в stage_layer.{table_name}"
        )

    spark_handler.execute_jdbc(
        dwh_connection,
        "CREATE DATABASE IF NOT EXISTS `stage_layer`",
    )

    target = spark_handler.quote_table(
        "stage_layer",
        table_name,
    )

    # Создаём таблицу только при первом запуске.
    if not spark_handler.clickhouse_table_exists(
        dwh_connection,
        "stage_layer",
        table_name,
    ):
        columns_ddl = ",\n".join(
            f"{spark_handler.quote_identifier(field.name)} "
            f"{spark_handler.clickhouse_type(field.dataType, nullable=field.nullable)}"
            for field in df.schema.fields
        )

        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {target}
            (
                {columns_ddl}
            )
            ENGINE = MergeTree
            ORDER BY {order_by}
            SETTINGS allow_nullable_key = 1
        """

        spark_handler.execute_jdbc(
            dwh_connection,
            create_table_sql,
        )

    # Удаляем результат предыдущего построения витрины.
    spark_handler.execute_jdbc(
        dwh_connection,
        f"TRUNCATE TABLE {target}",
    )

    # Записываем актуальный результат.
    spark_handler.write_jdbc(
        df=df,
        connection=dwh_connection,
        table=f"stage_layer.{table_name}",
        batch_size=10_000,
        num_partitions=4,
    )

    # Проверяем количество строк после загрузки.
    result = spark_handler.read_jdbc_query(
        dwh_connection,
        f"SELECT count() AS row_count FROM {target}",
    ).first()

    loaded_rows = int(result["row_count"])

    if loaded_rows != rows_to_load:
        raise RuntimeError(
            f"Ошибка загрузки stage_layer.{table_name}: "
            f"ожидалось {rows_to_load}, загружено {loaded_rows}"
        )

    logger.info(
        "stage_layer.%s обновлена. Строк: %s",
        table_name,
        loaded_rows,
    )

    return loaded_rows


def _build_latest_orders_forecast(
    table_name: str,
    app_name: str,
) -> dict[str, int | str]:
    """
    Оставляет только прогноз с максимальной creation_date.

    Используется для completed orders и all orders.
    """

    dwh_connection = get_clickhouse_dwh_config()

    spark_config = {
        "spark.sql.shuffle.partitions": os.getenv(
            "SPARK_SHUFFLE_PARTITIONS", "8"
        ),
        "spark.sql.session.timeZone": "UTC",
    }

    with PySparkHandler(
        app_name=app_name,
        config=spark_config,
    ) as spark_handler:

        logger.info("Читаем raw_layer.%s", table_name)

        source_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name=table_name,
            database="raw_layer",
        )

        # Находим дату создания последнего прогноза.
        latest_row = source_df.agg(
            F.max("creation_date").alias("latest_creation_date")
        ).first()

        latest_creation_date = latest_row["latest_creation_date"]

        if latest_creation_date is None:
            raise RuntimeError(
                f"В raw_layer.{table_name} нет корректной creation_date"
            )

        logger.info(
            "Последний прогноз %s создан: %s",
            table_name,
            latest_creation_date,
        )

        # Оставляем все строки последнего прогноза,
        # а не одну строку с максимальной датой.
        stage_df = source_df.filter(
            F.col("creation_date") == F.lit(latest_creation_date)
        )

        loaded_rows = _load_stage_dataframe(
            spark_handler=spark_handler,
            dwh_connection=dwh_connection,
            df=stage_df,
            table_name=table_name,
            order_by="(creation_date, shop_id, forecast_date)",
        )

        return {
            "latest_creation_date": latest_creation_date.isoformat(),
            "loaded_rows": loaded_rows,
        }


def build_latest_completed_orders_stage() -> dict[str, int | str]:
    """Строит витрину последнего прогноза завершённых заказов."""

    return _build_latest_orders_forecast(
        table_name="shops_daily_completed_orders_predict",
        app_name="DWH_Stage_Completed_Orders",
    )


def build_latest_all_orders_stage() -> dict[str, int | str]:
    """Строит витрину последнего прогноза всех заказов."""

    return _build_latest_orders_forecast(
        table_name="shops_daily_all_orders_predict",
        app_name="DWH_Stage_All_Orders",
    )


def build_latest_revenue_stage() -> dict[str, int | str]:
    """
    Строит витрину последних прогнозов выручки.

    Для каждой модели выбирается собственная максимальная creation_date.
    """

    table_name = "daily_revenue_predict"

    dwh_connection = get_clickhouse_dwh_config()

    spark_config = {
        "spark.sql.shuffle.partitions": os.getenv(
            "SPARK_SHUFFLE_PARTITIONS", "8"
        ),
        "spark.sql.session.timeZone": "UTC",
    }

    with PySparkHandler(
        app_name="DWH_Stage_Revenue",
        config=spark_config,
    ) as spark_handler:

        logger.info("Читаем raw_layer.%s", table_name)

        source_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name=table_name,
            database="raw_layer",
        )

        # Для корректного сопоставления у модели должно быть имя.
        if source_df.filter(
            F.col("model_name").isNull()
        ).limit(1).count() > 0:
            raise ValueError(
                "В daily_revenue_predict найдены строки без model_name"
            )

        # Находим максимальную дату создания отдельно для каждой модели.
        latest_per_model = (
            source_df
            .groupBy("model_name")
            .agg(
                F.max("creation_date").alias("_latest_creation_date")
            )
        )

        latest_models = latest_per_model.collect()

        if not latest_models:
            raise RuntimeError(
                "В raw_layer.daily_revenue_predict нет данных"
            )

        if any(
            row["_latest_creation_date"] is None
            for row in latest_models
        ):
            raise ValueError(
                "Для одной из моделей не найдена creation_date"
            )

        logger.info(
            "Последние прогнозы выручки: %s",
            {
                row["model_name"]: row["_latest_creation_date"]
                for row in latest_models
            },
        )

        # Присоединяем найденные даты к исходным строкам.
        # Так сохраняются ВСЕ дни прогноза для каждой модели.
        stage_df = (
            source_df.alias("src")
            .join(
                latest_per_model.alias("latest"),
                (
                    F.col("src.model_name")
                    == F.col("latest.model_name")
                )
                & (
                    F.col("src.creation_date")
                    == F.col("latest._latest_creation_date")
                ),
                how="inner",
            )
            .select(
                *[
                    F.col(f"src.`{column}`").alias(column)
                    for column in source_df.columns
                ]
            )
        )

        loaded_rows = _load_stage_dataframe(
            spark_handler=spark_handler,
            dwh_connection=dwh_connection,
            df=stage_df,
            table_name=table_name,
            order_by="(model_name, creation_date, forecast_date)",
        )

        return {
            "models_count": len(latest_models),
            "loaded_rows": loaded_rows,
        }


# 3. Таблица количества интеракций по типам за каждый день

def build_daily_interactions_stage() -> dict[str, int]:
    """
    Строит витрину ежедневного количества интеракций по типам.

    Источники:
        raw_layer.interactions
        raw_layer.orders_history

    Результат:
        stage_layer.daily_interactions

    Статусы completed и delivered из orders_history
    объединяются в тип completed.
    """

    table_name = "daily_interactions"

    dwh_connection = get_clickhouse_dwh_config()

    spark_config = {
        "spark.sql.shuffle.partitions": os.getenv(
            "SPARK_SHUFFLE_PARTITIONS", "8"
        ),
        "spark.sql.session.timeZone": "UTC",
    }

    with PySparkHandler(
        app_name="DWH_Stage_Daily_Interactions",
        config=spark_config,
    ) as spark_handler:

        # 1. Читаем исходные данные
        logger.info("Читаем raw_layer.interactions")

        interactions_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="interactions",
            database="raw_layer",
        )

        logger.info("Читаем raw_layer.orders_history")

        orders_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="orders_history",
            database="raw_layer",
        )

        # 2. Проверяем обязательные поля
        invalid_interactions = interactions_df.filter(
            F.col("datetime").isNull()
            | F.col("interaction_type").isNull()
        ).limit(1).count()

        if invalid_interactions > 0:
            raise ValueError(
                "В raw_layer.interactions найдены строки "
                "без datetime или interaction_type"
            )

        # Нас интересуют только успешно завершённые заказы.
        completed_orders = orders_df.filter(
            F.col("stage").isin("completed", "delivered")
        )

        invalid_completed = completed_orders.filter(
            F.col("datetime").isNull()
        ).limit(1).count()

        if invalid_completed > 0:
            raise ValueError(
                "В завершённых заказах отсутствует datetime"
            )

        # 3. Преобразуем обычные интеракции
        interactions_events = interactions_df.select(
            F.to_date("datetime").alias("date"),
            F.when(
                F.col("interaction_type") == "delivered",
                F.lit("completed"),
            ).otherwise(
                F.col("interaction_type")
            ).alias("interaction_name"),
        )

        # 4. Преобразуем завершённые заказы
        completed_events = completed_orders.select(
            F.to_date("datetime").alias("date"),
            F.lit("completed").alias("interaction_name"),
        )

        # 5. Объединяем события двух источников

        # unionByName объединяет DataFrame по именам колонок.
        # После объединения считаем события вместе.

        all_events = interactions_events.unionByName(
            completed_events
        )

        # 6. Группируем по дате и типу интеракции
        stage_df = (
            all_events
            .groupBy("date", "interaction_name")
            .agg(
                F.count("*").alias("interaction_count")
            )
        )

        # 7. Проверяем, что данные существуют
        total_events = (
            stage_df
            .agg(
                F.sum("interaction_count").alias("total_events")
            )
            .first()["total_events"]
        )

        if total_events is None or total_events == 0:
            raise RuntimeError(
                "Не найдено событий для построения витрины"
            )

        total_events = int(total_events)

        logger.info(
            "Количество событий для агрегации: %s",
            total_events,
        )

        # 8. Загружаем результат в Stage Layer
        loaded_rows = _load_stage_dataframe(
            spark_handler=spark_handler,
            dwh_connection=dwh_connection,
            df=stage_df,
            table_name=table_name,
            order_by="(date, interaction_name)",
        )

        logger.info(
            "Daily interactions Stage успешно обновлена. "
            "Групп: %s, событий: %s",
            loaded_rows,
            total_events,
        )

        return {
            "total_events": total_events,
            "loaded_groups": loaded_rows,
        }


# 4. Полная таблица заказов

def build_full_orders_stage() -> dict[str, int]:
    """
    Строит полную таблицу заказов.

    Источники:
        raw_layer.orders_history
        raw_layer.shops
        raw_layer.products
        raw_layer.products_price_history
        raw_layer.delivery_price_history
        raw_layer.users_cities
        raw_layer.users_clusters

    Результат:
        stage_layer.full_orders

    Одна строка соответствует одному заказу.
    Все временные отметки переводятся в локальное время магазина.
    """

    from pyspark.sql.window import Window

    table_name = "full_orders"

    dwh_connection = get_clickhouse_dwh_config()

    spark_config = {
        "spark.sql.shuffle.partitions": os.getenv(
            "SPARK_SHUFFLE_PARTITIONS", "8"
        ),
        "spark.sql.session.timeZone": "UTC",
    }

    stage_names = [
        "created",
        "accepted",
        "collecting_order",
        "sent_to_shop",
        "ready",
        "waiting_pickup",
        "sent_via_delivery",
        "completed",
        "delivered",
        "canceled",
        "lost",
    ]

    def latest_price_as_of(
        orders_df,
        prices_df,
        order_entity_column: str,
        price_entity_column: str,
        result_column: str,
    ):
        """
        Для каждого заказа находит последнюю цену,
        действовавшую на момент создания заказа.
        """

        order_rows = orders_df.select(
            F.col("order_id").alias("_order_id"),
            F.col("created_at_utc").alias("_created_at_utc"),
            F.col(order_entity_column)
            .cast("string")
            .alias("_entity_id"),
        ).alias("o")

        price_rows = (
            prices_df
            .filter(
                F.col(price_entity_column).isNotNull()
                & F.col("effective_from").isNotNull()
                & F.col("price").isNotNull()
            )
            .select(
                F.col(price_entity_column)
                .cast("string")
                .alias("_price_entity_id"),
                F.col("effective_from").alias("_effective_from"),
                F.col("price").cast("long").alias("_price"),
            )
            .alias("p")
        )

        candidates = order_rows.join(
            price_rows,
            (
                F.col("o._entity_id")
                == F.col("p._price_entity_id")
            )
            & (
                F.col("p._effective_from")
                <= F.col("o._created_at_utc")
            ),
            how="left",
        )

        price_window = (
            Window
            .partitionBy(F.col("o._order_id"))
            .orderBy(
                F.col("p._effective_from").desc_nulls_last(),
                F.col("p._price").desc_nulls_last(),
            )
        )

        result = (
            candidates
            .withColumn(
                "_price_rank",
                F.row_number().over(price_window),
            )
            .filter(F.col("_price_rank") == 1)
            .select(
                F.col("o._order_id").alias("order_id"),
                F.col("p._price").alias(result_column),
            )
        )

        return result

    with PySparkHandler(
        app_name="DWH_Stage_Full_Orders",
        config=spark_config,
    ) as spark_handler:

        # 1. Читаем исходные таблицы
        logger.info("Читаем raw_layer.orders_history")

        orders_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="orders_history",
            database="raw_layer",
        )

        logger.info("Читаем raw_layer.shops")

        shops_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="shops",
            database="raw_layer",
        )

        logger.info("Читаем raw_layer.products")

        products_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="products",
            database="raw_layer",
        )

        logger.info("Читаем raw_layer.products_price_history")

        product_prices_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="products_price_history",
            database="raw_layer",
        )

        logger.info("Читаем raw_layer.delivery_price_history")

        delivery_prices_df = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="delivery_price_history",
            database="raw_layer",
        )

        logger.info("Читаем пользовательские справочники Redis")

        users_cities_raw = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="users_cities",
            database="raw_layer",
        )

        users_clusters_raw = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="users_clusters",
            database="raw_layer",
        )

        # 2. Проверяем исходную историю заказов
        invalid_events = orders_df.filter(
            F.col("order_id").isNull()
            | F.col("user_id").isNull()
            | F.col("product_id").isNull()
            | F.col("shop_id").isNull()
            | F.col("datetime").isNull()
            | F.col("stage").isNull()
            | F.col("order_history_id").isNull()
        ).limit(1).count()

        if invalid_events > 0:
            raise ValueError(
                "В raw_layer.orders_history найдены события "
                "с незаполненными обязательными полями"
            )

        unknown_stages = [
            row["stage"]
            for row in (
                orders_df
                .filter(~F.col("stage").isin(*stage_names))
                .select("stage")
                .distinct()
                .limit(10)
                .collect()
            )
        ]

        if unknown_stages:
            raise ValueError(
                f"Обнаружены неизвестные стадии заказов: {unknown_stages}"
            )

        # Проверяем, что идентификаторы не меняются
        # между событиями одного заказа.
        identity_stats = (
            orders_df
            .groupBy("order_id")
            .agg(
                F.countDistinct("user_id").alias("users"),
                F.countDistinct("product_id").alias("products"),
                F.countDistinct("shop_id").alias("shops"),
            )
        )

        invalid_orders = identity_stats.filter(
            (F.col("users") != 1)
            | (F.col("products") != 1)
            | (F.col("shops") != 1)
        ).limit(1).count()

        if invalid_orders > 0:
            raise ValueError(
                "У некоторых заказов меняются идентификаторы "
                "пользователя, товара или магазина"
            )

        # 3. Собираем события в одну строку на заказ
        logger.info("Агрегируем историю стадий заказов")

        stage_aggregations = [
            F.max(
                F.when(
                    F.col("stage") == stage,
                    F.col("datetime"),
                )
            ).alias(f"{stage}_at_utc")
            for stage in stage_names
        ]

        order_summary = (
            orders_df
            .groupBy("order_id")
            .agg(
                F.max("user_id").alias("user_id"),
                F.max("product_id").alias("product_id"),
                F.max("shop_id").alias("shop_id"),
                *stage_aggregations,
            )
        )

        # Для определения последней стадии выбираем
        # событие с максимальным datetime.
        last_event_window = (
            Window
            .partitionBy("order_id")
            .orderBy(
                F.col("datetime").desc(),
                F.col("order_history_id").desc(),
            )
        )

        last_event = (
            orders_df
            .withColumn(
                "_row_number",
                F.row_number().over(last_event_window),
            )
            .filter(F.col("_row_number") == 1)
            .select(
                "order_id",
                F.col("stage").alias("last_stage"),
                F.col("datetime").alias("last_stage_at_utc"),
            )
        )

        order_summary = order_summary.join(
            last_event,
            on="order_id",
            how="inner",
        )

        orders_count = order_summary.count()

        # Дата создания обязательна: она используется
        # для выбора ценовых записей.
        if (
            order_summary
            .filter(F.col("created_at_utc").isNull())
            .limit(1)
            .count()
            > 0
        ):
            raise ValueError(
                "У некоторых заказов отсутствует стадия created"
            )

        # 4. Добавляем информацию о магазинах
        shop_lookup = shops_df.select(
            F.col("shop_id").cast("string").alias("shop_id"),
            F.col("city").alias("shop_city"),
            F.col("timezone").alias("shop_timezone"),
        )

        duplicate_shops = (
            shop_lookup
            .groupBy("shop_id")
            .count()
            .filter(F.col("count") > 1)
            .limit(1)
            .count()
        )

        if duplicate_shops > 0:
            raise ValueError(
                "В raw_layer.shops найдены повторяющиеся shop_id"
            )

        order_base = order_summary.join(
            shop_lookup,
            on="shop_id",
            how="left",
        )

        # 5. Добавляем информацию о товарах
        product_lookup = products_df.select(
            F.col("product_id").cast("long").alias("product_id"),
            F.col("prime_cost").cast("long").alias(
                "product_prime_cost"
            ),
        )

        duplicate_products = (
            product_lookup
            .groupBy("product_id")
            .count()
            .filter(F.col("count") > 1)
            .limit(1)
            .count()
        )

        if duplicate_products > 0:
            raise ValueError(
                "В raw_layer.products найдены повторяющиеся product_id"
            )

        order_base = order_base.join(
            product_lookup,
            on="product_id",
            how="left",
        )

        # 6. Восстанавливаем user_id из Redis-ключей
        users_cities = users_cities_raw.select(
            F.regexp_extract(
                F.col("_redis_key"),
                r"^user:(.*):city$",
                1,
            ).alias("user_id"),
            F.col("city").alias("user_city"),
        )

        users_clusters = users_clusters_raw.select(
            F.regexp_extract(
                F.col("_redis_key"),
                r"^user:(.*):cluster$",
                1,
            ).alias("user_id"),
            F.col("price_cluster"),
        )

        # Пустой результат regexp_extract означает,
        # что формат ключа Redis не соответствует ожидаемому.
        invalid_city_keys = users_cities.filter(
            (F.col("user_id") == "")
            | F.col("user_city").isNull()
        ).limit(1).count()

        invalid_cluster_keys = users_clusters.filter(
            (F.col("user_id") == "")
            | F.col("price_cluster").isNull()
        ).limit(1).count()

        if invalid_city_keys or invalid_cluster_keys:
            raise ValueError(
                "Не удалось восстановить пользовательские "
                "справочники из Redis-ключей"
            )

        for mapping_df, mapping_name in (
            (users_cities, "users_cities"),
            (users_clusters, "users_clusters"),
        ):
            duplicates = (
                mapping_df
                .groupBy("user_id")
                .count()
                .filter(F.col("count") > 1)
                .limit(1)
                .count()
            )

            if duplicates > 0:
                raise ValueError(
                    f"В {mapping_name} найдены повторяющиеся user_id"
                )

        order_base = (
            order_base
            .join(users_cities, on="user_id", how="left")
            .join(users_clusters, on="user_id", how="left")
        )

        # 7. Восстанавливаем цену товара на момент заказа
        logger.info("Определяем цену товара на момент создания заказа")

        product_price_by_order = latest_price_as_of(
            orders_df=order_base,
            prices_df=product_prices_df,
            order_entity_column="product_id",
            price_entity_column="product_id",
            result_column="product_price_paid",
        )

        order_base = order_base.join(
            product_price_by_order,
            on="order_id",
            how="left",
        )

        # 8. Восстанавливаем цену доставки
        logger.info("Определяем цену доставки")

        shop_delivery_prices = delivery_prices_df.filter(
            F.col("object_type") == "shop"
        )

        city_delivery_prices = delivery_prices_df.filter(
            F.col("object_type") == "city"
        )

        # Сначала ищем цену конкретного магазина.
        shop_delivery_by_order = latest_price_as_of(
            orders_df=order_base,
            prices_df=shop_delivery_prices,
            order_entity_column="shop_id",
            price_entity_column="object_id",
            result_column="shop_delivery_price",
        )

        # Дополнительно ищем цену города магазина.
        city_delivery_by_order = latest_price_as_of(
            orders_df=order_base,
            prices_df=city_delivery_prices,
            order_entity_column="shop_city",
            price_entity_column="object_id",
            result_column="city_delivery_price",
        )

        order_base = (
            order_base
            .join(
                shop_delivery_by_order,
                on="order_id",
                how="left",
            )
            .join(
                city_delivery_by_order,
                on="order_id",
                how="left",
            )
            .withColumn(
                "delivery_price_paid",
                F.coalesce(
                    F.col("shop_delivery_price"),
                    F.col("city_delivery_price"),
                ),
            )
            .drop(
                "shop_delivery_price",
                "city_delivery_price",
            )
        )

        # 9. Определяем обобщённое состояние заказа
        order_base = order_base.withColumn(
            "order_status",
            F.when(
                F.col("last_stage").isin("completed", "delivered"),
                F.lit("completed"),
            )
            .when(
                F.col("last_stage").isin("canceled", "lost"),
                F.lit("canceled_or_lost"),
            )
            .otherwise(
                F.lit("in_progress")
            ),
        )

        # 10. Проверяем полноту итоговых данных
        required_fields = [
            "shop_city",
            "shop_timezone",
            "product_prime_cost",
            "user_city",
            "price_cluster",
            "created_at_utc",
            "product_price_paid",
            "delivery_price_paid",
        ]

        missing_stats = (
            order_base
            .agg(*[
                F.sum(
                    F.when(F.col(field).isNull(), 1).otherwise(0)
                ).alias(field)
                for field in required_fields
            ])
            .first()
            .asDict()
        )

        missing_stats = {
            name: int(count or 0)
            for name, count in missing_stats.items()
        }

        invalid_fields = {
            name: count
            for name, count in missing_stats.items()
            if count > 0
        }

        if invalid_fields:
            raise ValueError(
                "Не удалось заполнить обязательные поля заказов: "
                f"{invalid_fields}"
            )

        # 11. Переводим временные отметки в локальное время
        utc_timestamp_columns = [
            f"{stage}_at_utc"
            for stage in stage_names
        ] + ["last_stage_at_utc"]

        for utc_column in utc_timestamp_columns:
            local_column = utc_column.removesuffix("_utc")

            order_base = (
                order_base
                .withColumn(
                    local_column,
                    F.expr(
                        f"from_utc_timestamp("
                        f"`{utc_column}`, `shop_timezone`)"
                    ),
                )
                .drop(utc_column)
            )

        # 12. Формируем итоговую структуру таблицы
        stage_df = order_base.select(
            "order_id",
            "user_id",
            "product_id",
            "shop_id",

            "created_at",
            "accepted_at",
            "collecting_order_at",
            "sent_to_shop_at",
            "ready_at",
            "waiting_pickup_at",
            "sent_via_delivery_at",
            "completed_at",
            "delivered_at",
            "canceled_at",
            "lost_at",

            "order_status",
            "last_stage",
            "last_stage_at",

            "product_price_paid",
            "delivery_price_paid",

            "price_cluster",
            "user_city",
            "shop_city",
            "product_prime_cost",
            "shop_timezone",
        )

        # 13. Загружаем результат в Stage Layer
        loaded_rows = _load_stage_dataframe(
            spark_handler=spark_handler,
            dwh_connection=dwh_connection,
            df=stage_df,
            table_name=table_name,
            order_by="order_id",
        )

        if loaded_rows != orders_count:
            raise RuntimeError(
                "Количество строк в full_orders не совпадает "
                f"с числом заказов: {loaded_rows} != {orders_count}"
            )

        status_counts = (
            stage_df
            .groupBy("order_status")
            .count()
            .collect()
        )

        stats = {
            "orders": loaded_rows,
        }

        for row in status_counts:
            stats[row["order_status"]] = int(row["count"])

        logger.info(
            "Full Orders Stage успешно обновлена: %s",
            stats,
        )

        return stats