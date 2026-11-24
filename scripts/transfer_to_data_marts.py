from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from scripts.transfer_to_stage_layer import get_clickhouse_dwh_config
from src.data_handlers.pyspark_handler import JdbcConfig, PySparkHandler

logger = logging.getLogger(__name__)

DATA_MARTS_DATABASE = "data_marts"


def _spark_config() -> dict[str, str]:
    return {
        "spark.sql.shuffle.partitions": os.getenv(
            "SPARK_SHUFFLE_PARTITIONS", "8"
        ),
        # Внутри DWH храним timestamps как UTC или как локальное wall-clock
        # время после явного from_utc_timestamp. Это исключает неявные сдвиги.
        "spark.sql.session.timeZone": "UTC",
    }


def _load_data_mart_dataframe(
    spark_handler: PySparkHandler,
    dwh_connection: JdbcConfig,
    df: DataFrame,
    table_name: str,
    order_by: str,
) -> int:
    """Полностью обновляет таблицу data_marts данными Spark DataFrame.

    Если набор колонок динамический (например, pivot по товарам),
    таблица пересоздаётся при изменении состава или порядка колонок.
    """

    df = spark_handler.normalize_for_clickhouse(df)
    rows_to_load = df.count()

    if rows_to_load == 0:
        raise RuntimeError(
            f"Нет строк для загрузки в {DATA_MARTS_DATABASE}.{table_name}"
        )

    spark_handler.execute_jdbc(
        dwh_connection,
        f"CREATE DATABASE IF NOT EXISTS "
        f"{spark_handler.quote_identifier(DATA_MARTS_DATABASE)}",
    )

    target = spark_handler.quote_table(DATA_MARTS_DATABASE, table_name)
    table_exists = spark_handler.clickhouse_table_exists(
        dwh_connection,
        DATA_MARTS_DATABASE,
        table_name,
    )

    if table_exists:
        existing_columns = [
            row["name"]
            for row in spark_handler.read_jdbc_query(
                dwh_connection,
                "SELECT name "
                "FROM system.columns "
                f"WHERE database = '{DATA_MARTS_DATABASE}' "
                f"AND table = '{table_name}' "
                "ORDER BY position",
            ).collect()
        ]

        if existing_columns != df.columns:
            logger.warning(
                "Схема %s.%s изменилась. Пересоздаём таблицу. "
                "Старые колонки: %s; новые: %s",
                DATA_MARTS_DATABASE,
                table_name,
                existing_columns,
                df.columns,
            )
            spark_handler.execute_jdbc(
                dwh_connection,
                f"DROP TABLE IF EXISTS {target}",
            )
            table_exists = False

    if not table_exists:
        columns_ddl = ",\n".join(
            f"{spark_handler.quote_identifier(field.name)} "
            f"{spark_handler.clickhouse_type(field.dataType, nullable=field.nullable)}"
            for field in df.schema.fields
        )

        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {target}
            (
                {columns_ddl}
            )
            ENGINE = MergeTree
            ORDER BY {order_by}
            SETTINGS allow_nullable_key = 1
        """
        spark_handler.execute_jdbc(dwh_connection, create_sql)

    spark_handler.execute_jdbc(
        dwh_connection,
        f"TRUNCATE TABLE {target}",
    )

    spark_handler.write_jdbc(
        df=df,
        connection=dwh_connection,
        table=f"{DATA_MARTS_DATABASE}.{table_name}",
        batch_size=10_000,
        num_partitions=4,
    )

    result = spark_handler.read_jdbc_query(
        dwh_connection,
        f"SELECT count() AS row_count FROM {target}",
    ).first()
    loaded_rows = int(result["row_count"])

    if loaded_rows != rows_to_load:
        raise RuntimeError(
            f"Ошибка загрузки {DATA_MARTS_DATABASE}.{table_name}: "
            f"ожидалось {rows_to_load}, загружено {loaded_rows}"
        )

    logger.info(
        "%s.%s успешно обновлена. Строк: %s",
        DATA_MARTS_DATABASE,
        table_name,
        loaded_rows,
    )
    return loaded_rows


def _read_raw_table(
    spark_handler: PySparkHandler,
    dwh_connection: JdbcConfig,
    table_name: str,
) -> DataFrame:
    return spark_handler.read_from_clickhouse(
        connection=dwh_connection,
        table_name=table_name,
        database="raw_layer",
    )


def build_daily_competitor_price_difference_mart() -> dict[str, int | str]:
    """Средняя разница цены нашей компании и каждого конкурента за день.

    Для конкурента используем price или value. Поле cost не является
    продажной ценой и в сравнение не попадает.

    Разница = цена конкурента - наша цена.
    """

    table_name = "daily_competitor_price_difference"
    reporting_timezone = os.getenv(
        "DWH_REPORTING_TIMEZONE", "Europe/Moscow"
    )
    dwh_connection = get_clickhouse_dwh_config()

    with PySparkHandler(
        app_name="DWH_Mart_Daily_Competitor_Price_Difference",
        config=_spark_config(),
    ) as spark_handler:
        logger.info("Читаем Stage историю цен конкурентов")
        competitors_raw = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="competitors_products_price_history",
            database="stage_layer",
        )

        logger.info("Читаем Raw историю собственных цен")
        own_prices_raw = _read_raw_table(
            spark_handler,
            dwh_connection,
            "products_price_history",
        )

        competitors = (
            competitors_raw
            .filter(
                F.coalesce(
                    F.col("price").cast("double"),
                    F.col("value").cast("double"),
                ).isNotNull()
            )
            .select(
                F.col("parse_id").cast("string").alias("parse_id"),
                F.col("company_name").cast("string").alias("company_name"),
                F.col("product_id").cast("long").alias("product_id"),
                F.col("datetime").cast("timestamp").alias("competitor_datetime"),
                F.coalesce(
                    F.col("price").cast("double"),
                    F.col("value").cast("double"),
                ).alias("competitor_price"),
            )
        )

        own_prices = (
            own_prices_raw
            .filter(
                F.col("product_id").isNotNull()
                & F.col("effective_from").isNotNull()
                & F.col("price").isNotNull()
            )
            .select(
                F.col("product_id").cast("long").alias("product_id"),
                F.col("effective_from").cast("timestamp").alias(
                    "own_effective_from"
                ),
                F.col("price").cast("double").alias("own_price"),
            )
        )

        invalid_competitor_rows = competitors.filter(
            F.col("parse_id").isNull()
            | F.col("company_name").isNull()
            | F.col("product_id").isNull()
            | F.col("competitor_datetime").isNull()
        ).limit(1).count()
        if invalid_competitor_rows:
            raise ValueError(
                "В истории цен конкурентов найдены строки без ключевых полей"
            )

        c = competitors.alias("c")
        p = own_prices.alias("p")
        candidates = c.join(
            p,
            (F.col("c.product_id") == F.col("p.product_id"))
            & (
                F.col("p.own_effective_from")
                <= F.col("c.competitor_datetime")
            ),
            how="left",
        )

        latest_own_price_window = (
            Window
            .partitionBy(
                F.col("c.parse_id"),
                F.col("c.company_name"),
                F.col("c.product_id"),
                F.col("c.competitor_datetime"),
            )
            .orderBy(F.col("p.own_effective_from").desc_nulls_last())
        )

        matched = (
            candidates
            .withColumn(
                "_price_rank",
                F.row_number().over(latest_own_price_window),
            )
            .filter(F.col("_price_rank") == 1)
            .select(
                F.col("c.company_name").alias("company_name"),
                F.col("c.product_id").alias("product_id"),
                F.col("c.competitor_datetime").alias("competitor_datetime"),
                F.col("c.competitor_price").alias("competitor_price"),
                F.col("p.own_price").alias("own_price"),
            )
        )

        missing_own_prices = matched.filter(
            F.col("own_price").isNull()
        ).count()
        if missing_own_prices:
            raise ValueError(
                "Не найдена собственная цена на момент парсинга "
                f"для {missing_own_prices} записей конкурентов"
            )

        local_datetime = F.from_utc_timestamp(
            F.col("competitor_datetime"),
            reporting_timezone,
        )

        compared = (
            matched
            .withColumn("date", F.to_date(local_datetime))
            .withColumn(
                "price_difference",
                F.col("competitor_price") - F.col("own_price"),
            )
            .withColumn(
                "absolute_price_difference",
                F.abs(F.col("price_difference")),
            )
        )

        mart_df = (
            compared
            .groupBy("date", "company_name")
            .agg(
                F.countDistinct("product_id").alias("compared_products"),
                F.avg("own_price").alias("avg_our_price"),
                F.avg("competitor_price").alias("avg_competitor_price"),
                F.avg("price_difference").alias("avg_price_difference"),
                F.avg("absolute_price_difference").alias(
                    "avg_absolute_price_difference"
                ),
            )
            .withColumn(
                "reporting_timezone",
                F.lit(reporting_timezone),
            )
            .select(
                "date",
                "company_name",
                "compared_products",
                "avg_our_price",
                "avg_competitor_price",
                "avg_price_difference",
                "avg_absolute_price_difference",
                "reporting_timezone",
            )
        )

        loaded_rows = _load_data_mart_dataframe(
            spark_handler,
            dwh_connection,
            mart_df,
            table_name,
            order_by="(date, company_name)",
        )

        return {
            "loaded_rows": loaded_rows,
            "reporting_timezone": reporting_timezone,
        }


def build_monthly_orders_by_segment_city_mart() -> dict[str, int]:
    """Количество всех созданных заказов по сегменту и городу магазина."""

    table_name = "monthly_orders_by_segment_city"
    dwh_connection = get_clickhouse_dwh_config()

    with PySparkHandler(
        app_name="DWH_Mart_Monthly_Orders_By_Segment_City",
        config=_spark_config(),
    ) as spark_handler:
        orders = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="full_orders",
            database="stage_layer",
        )

        invalid_rows = orders.filter(
            F.col("order_id").isNull()
            | F.col("created_at").isNull()
            | F.col("price_cluster").isNull()
            | F.col("shop_city").isNull()
        ).limit(1).count()
        if invalid_rows:
            raise ValueError(
                "В stage_layer.full_orders найдены заказы без даты, "
                "сегмента или города магазина"
            )

        mart_df = (
            orders
            .withColumn(
                "month_start",
                F.to_date(F.date_trunc("month", F.col("created_at"))),
            )
            .groupBy("month_start", "price_cluster", "shop_city")
            .agg(
                F.countDistinct("order_id").alias("orders_count")
            )
            .select(
                "month_start",
                "price_cluster",
                "shop_city",
                "orders_count",
            )
        )

        loaded_rows = _load_data_mart_dataframe(
            spark_handler,
            dwh_connection,
            mart_df,
            table_name,
            order_by="(month_start, price_cluster, shop_city)",
        )

        return {"loaded_groups": loaded_rows}


def build_annual_weekly_conversion_mart() -> dict[str, int]:
    """Средняя недельная конверсия сайт-интеракций в завершённые заказы.

    Сначала вычисляется завершённые заказы / сайт-интеракции за неделю,
    затем недельные коэффициенты усредняются в пределах календарного года.
    """

    table_name = "annual_weekly_conversion"
    dwh_connection = get_clickhouse_dwh_config()
    website_interaction_types = [
        "click",
        "add_to_cart",
        "purchased",
        "paid",
    ]

    with PySparkHandler(
        app_name="DWH_Mart_Annual_Weekly_Conversion",
        config=_spark_config(),
    ) as spark_handler:
        interactions = _read_raw_table(
            spark_handler,
            dwh_connection,
            "interactions",
        )
        users_cities_raw = _read_raw_table(
            spark_handler,
            dwh_connection,
            "users_cities",
        )
        shops_raw = _read_raw_table(
            spark_handler,
            dwh_connection,
            "shops",
        )
        orders = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="full_orders",
            database="stage_layer",
        )

        invalid_interactions = interactions.filter(
            F.col("user_id").isNull()
            | F.col("datetime").isNull()
            | F.col("interaction_type").isNull()
        ).limit(1).count()
        if invalid_interactions:
            raise ValueError(
                "В raw_layer.interactions найдены записи без user_id, "
                "datetime или interaction_type"
            )

        unknown_types = [
            row["interaction_type"]
            for row in (
                interactions
                .filter(
                    ~F.col("interaction_type").isin(
                        *website_interaction_types
                    )
                )
                .select("interaction_type")
                .distinct()
                .limit(10)
                .collect()
            )
        ]
        if unknown_types:
            raise ValueError(
                f"Обнаружены неизвестные типы сайт-интеракций: {unknown_types}"
            )

        users_cities = users_cities_raw.select(
            F.regexp_extract(
                F.col("_redis_key"),
                r"^user:(.*):city$",
                1,
            ).alias("user_id"),
            F.col("city").alias("user_city"),
        )


        city_timezones = (
            shops_raw
            .filter(
                F.col("city").isNotNull()
                & F.col("timezone").isNotNull()
            )
            .groupBy("city")
            .agg(
                F.countDistinct("timezone").alias("timezone_count"),
                F.first("timezone", ignorenulls=True).alias("user_timezone"),
            )
        )

        ambiguous_cities = city_timezones.filter(
            F.col("timezone_count") > 1
        ).limit(10).collect()
        if ambiguous_cities:
            raise ValueError(
                "Для одного города найдено несколько часовых поясов: "
                f"{[row['city'] for row in ambiguous_cities]}"
            )

        users_city_alias = users_cities.alias("u")
        city_timezone_alias = city_timezones.select(
            "city", "user_timezone"
        ).alias("ct")

        user_lookup = (
            users_city_alias
            .join(
                city_timezone_alias,
                F.col("u.user_city") == F.col("ct.city"),
                how="left",
            )
            .select(
                F.col("u.user_id").alias("user_id"),
                F.col("u.user_city").alias("user_city"),
                F.col("ct.user_timezone").alias("user_timezone"),
            )
        )

        duplicate_users = (
            user_lookup
            .groupBy("user_id")
            .count()
            .filter(F.col("count") > 1)
            .limit(1)
            .count()
        )
        if duplicate_users:
            raise ValueError(
                "В users_cities найдено несколько записей на пользователя"
            )

        interaction_events = (
            interactions
            .filter(
                F.col("interaction_type").isin(*website_interaction_types)
            )
            .join(user_lookup, on="user_id", how="left")
        )

        missing_user_timezones = interaction_events.filter(
            F.col("user_timezone").isNull()
        ).limit(1).count()
        if missing_user_timezones:
            raise ValueError(
                "Для части пользовательских интеракций не удалось "
                "определить часовой пояс по городу пользователя"
            )

        interaction_weekly = (
            interaction_events
            .withColumn(
                "local_datetime",
                F.expr("from_utc_timestamp(datetime, user_timezone)"),
            )
            .withColumn(
                "week_start",
                F.to_date(F.date_trunc("week", F.col("local_datetime"))),
            )
            .groupBy("week_start")
            .agg(
                F.count("*").alias("interaction_count")
            )
        )

        completed_orders = (
            orders
            .filter(F.col("order_status") == "completed")
            .withColumn(
                "completed_local_datetime",
                F.coalesce(
                    F.col("delivered_at"),
                    F.col("completed_at"),
                ),
            )
        )

        missing_completion_dates = completed_orders.filter(
            F.col("completed_local_datetime").isNull()
        ).limit(1).count()
        if missing_completion_dates:
            raise ValueError(
                "У завершённых заказов отсутствует дата завершения"
            )

        completed_weekly = (
            completed_orders
            .withColumn(
                "week_start",
                F.to_date(
                    F.date_trunc(
                        "week",
                        F.col("completed_local_datetime"),
                    )
                ),
            )
            .groupBy("week_start")
            .agg(
                F.countDistinct("order_id").alias("completed_orders")
            )
        )

        weekly = (
            interaction_weekly
            .join(completed_weekly, on="week_start", how="full_outer")
            .withColumn(
                "interaction_count",
                F.coalesce(F.col("interaction_count"), F.lit(0)),
            )
            .withColumn(
                "completed_orders",
                F.coalesce(F.col("completed_orders"), F.lit(0)),
            )
            .filter(F.col("interaction_count") > 0)
            .withColumn(
                "weekly_conversion",
                F.col("completed_orders").cast("double")
                / F.col("interaction_count").cast("double"),
            )
            .withColumn("year", F.year("week_start"))
        )

        mart_df = (
            weekly
            .groupBy("year")
            .agg(
                F.avg("weekly_conversion").alias("avg_weekly_conversion"),
                F.count("*").alias("weeks_count"),
                F.avg("interaction_count").alias("avg_weekly_interactions"),
                F.avg("completed_orders").alias(
                    "avg_weekly_completed_orders"
                ),
                F.sum("interaction_count").alias("total_interactions"),
                F.sum("completed_orders").alias("total_completed_orders"),
            )
            .select(
                "year",
                "avg_weekly_conversion",
                "weeks_count",
                "avg_weekly_interactions",
                "avg_weekly_completed_orders",
                "total_interactions",
                "total_completed_orders",
            )
        )

        loaded_rows = _load_data_mart_dataframe(
            spark_handler,
            dwh_connection,
            mart_df,
            table_name,
            order_by="year",
        )
        return {"years": loaded_rows}


def build_user_product_orders_4m_mart() -> dict[str, int | str]:
    """Матрица заказов пользователей по товарам за последние четыре месяца.

    Период отсчитывается от максимальной локальной даты создания заказа
    в Stage Layer. В матрице присутствуют все product_id из Raw products.
    """

    table_name = "user_product_orders_4m"
    dwh_connection = get_clickhouse_dwh_config()

    with PySparkHandler(
        app_name="DWH_Mart_User_Product_Orders_4M",
        config=_spark_config(),
    ) as spark_handler:
        orders = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="full_orders",
            database="stage_layer",
        )
        products = _read_raw_table(
            spark_handler,
            dwh_connection,
            "products",
        )

        max_order_date = (
            orders
            .agg(F.max(F.to_date("created_at")).alias("max_order_date"))
            .first()["max_order_date"]
        )
        if max_order_date is None:
            raise RuntimeError("В stage_layer.full_orders нет дат создания")

        period_start_date = (
            spark_handler.spark.range(1)
            .select(
                F.add_months(
                    F.lit(max_order_date), -4
                ).alias("period_start")
            )
            .first()["period_start"]
        )

        recent_orders = orders.filter(
            F.to_date("created_at") >= F.lit(period_start_date)
        )

        if recent_orders.limit(1).count() == 0:
            raise RuntimeError(
                "За последние четыре месяца данных нет"
            )

        product_ids = [
            int(row["product_id"])
            for row in (
                products
                .filter(F.col("product_id").isNotNull())
                .select(F.col("product_id").cast("long").alias("product_id"))
                .distinct()
                .orderBy("product_id")
                .collect()
            )
        ]
        if not product_ids:
            raise RuntimeError("В raw_layer.products не найдены товары")

        pivot_values = [str(product_id) for product_id in product_ids]

        user_product_counts = (
            recent_orders
            .groupBy("user_id", "product_id")
            .agg(
                F.countDistinct("order_id").alias("order_count")
            )
            .withColumn(
                "product_key",
                F.col("product_id").cast("string"),
            )
        )

        wide_df = (
            user_product_counts
            .groupBy("user_id")
            .pivot("product_key", pivot_values)
            .agg(F.sum("order_count"))
            .fillna(0, subset=pivot_values)
        )

        new_column_names = ["customer_id"] + [
            f"product_{product_id}" for product_id in product_ids
        ]
        mart_df = wide_df.toDF(*new_column_names)

        loaded_rows = _load_data_mart_dataframe(
            spark_handler,
            dwh_connection,
            mart_df,
            table_name,
            order_by="customer_id",
        )

        return {
            "customers": loaded_rows,
            "products": len(product_ids),
            "period_start": str(period_start_date),
            "period_end": str(max_order_date),
        }


def build_ml_actual_comparison_mart() -> dict[str, int]:
    """Сравнивает прогнозы количества заказов и выручки с фактом.

    completed fact: заказы, завершённые/доставленные в этот локальный день.
    all fact: заказы, созданные в этот локальный день.
    revenue fact: для завершённых заказов price + delivery - prime_cost.
    """

    orders_table = "ml_orders_comparison"
    revenue_table = "ml_revenue_comparison"
    dwh_connection = get_clickhouse_dwh_config()

    with PySparkHandler(
        app_name="DWH_Mart_ML_Actual_Comparison",
        config=_spark_config(),
    ) as spark_handler:
        orders = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="full_orders",
            database="stage_layer",
        )
        completed_forecasts_raw = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="shops_daily_completed_orders_predict",
            database="stage_layer",
        )
        all_forecasts_raw = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="shops_daily_all_orders_predict",
            database="stage_layer",
        )
        revenue_forecasts_raw = spark_handler.read_from_clickhouse(
            connection=dwh_connection,
            table_name="daily_revenue_predict",
            database="stage_layer",
        )

        # Факт всех заказов — по локальной дате created_at магазина.
        fact_all = (
            orders
            .filter(
                F.col("order_id").isNotNull()
                & F.col("created_at").isNotNull()
            )
            .withColumn("date", F.to_date("created_at"))
            .groupBy("shop_id", "date")
            .agg(
                F.countDistinct("order_id").alias("fact_all_orders")
            )
        )

        # Факт завершённых заказов — по локальной дате завершения/доставки.
        completed = (
            orders
            .filter(F.col("order_status") == "completed")
            .withColumn(
                "completed_local_datetime",
                F.coalesce(
                    F.col("delivered_at"),
                    F.col("completed_at"),
                ),
            )
        )

        fact_completed = (
            completed
            .filter(F.col("completed_local_datetime").isNotNull())
            .withColumn(
                "date",
                F.to_date("completed_local_datetime"),
            )
            .groupBy("shop_id", "date")
            .agg(
                F.countDistinct("order_id").alias("fact_completed_orders")
            )
        )

        predicted_completed = completed_forecasts_raw.select(
            F.col("shop_id").cast("string").alias("shop_id"),
            F.col("forecast_date").cast("date").alias("date"),
            F.col("predict_orders").cast("long").alias(
                "predicted_completed_orders"
            ),
        )

        predicted_all = all_forecasts_raw.select(
            F.col("shop_id").cast("string").alias("shop_id"),
            F.col("forecast_date").cast("date").alias("date"),
            F.col("predict_orders").cast("long").alias(
                "predicted_all_orders"
            ),
        )

        orders_comparison = (
            predicted_completed
            .join(
                predicted_all,
                on=["shop_id", "date"],
                how="full_outer",
            )
            .join(
                fact_all,
                on=["shop_id", "date"],
                how="full_outer",
            )
            .join(
                fact_completed,
                on=["shop_id", "date"],
                how="full_outer",
            )
            .withColumn(
                "fact_all_orders",
                F.coalesce(F.col("fact_all_orders"), F.lit(0).cast("long")),
            )
            .withColumn(
                "fact_completed_orders",
                F.coalesce(
                    F.col("fact_completed_orders"),
                    F.lit(0).cast("long"),
                ),
            )
            .withColumn(
                "completed_error",
                F.col("predicted_completed_orders")
                - F.col("fact_completed_orders"),
            )
            .withColumn(
                "all_orders_error",
                F.col("predicted_all_orders")
                - F.col("fact_all_orders"),
            )
            .select(
                "date",
                "shop_id",
                "predicted_completed_orders",
                "fact_completed_orders",
                "completed_error",
                "predicted_all_orders",
                "fact_all_orders",
                "all_orders_error",
            )
        )

        loaded_orders_rows = _load_data_mart_dataframe(
            spark_handler,
            dwh_connection,
            orders_comparison,
            orders_table,
            order_by="(date, shop_id)",
        )

        # Доход признаём по завершённым заказам. Все три ценовых поля
        # сформированы на этапе Stage и относятся к одному заказу.
        fact_revenue = (
            completed
            .filter(
                F.col("completed_local_datetime").isNotNull()
                & F.col("product_price_paid").isNotNull()
                & F.col("delivery_price_paid").isNotNull()
                & F.col("product_prime_cost").isNotNull()
            )
            .withColumn(
                "date",
                F.to_date("completed_local_datetime"),
            )
            .withColumn(
                "order_revenue",
                F.col("product_price_paid").cast("double")
                + F.col("delivery_price_paid").cast("double")
                - F.col("product_prime_cost").cast("double"),
            )
            .groupBy("date")
            .agg(
                F.sum("order_revenue").alias("fact_revenue")
            )
        )

        predicted_revenue = revenue_forecasts_raw.select(
            F.col("forecast_date").cast("date").alias("date"),
            F.col("model_name").cast("string").alias("model_name"),
            F.col("predict_revenue").cast("double").alias(
                "predicted_revenue"
            ),
        )

        revenue_comparison = (
            predicted_revenue
            .join(fact_revenue, on="date", how="full_outer")
            .withColumn(
                "fact_revenue",
                F.coalesce(F.col("fact_revenue"), F.lit(0.0)),
            )
            .withColumn(
                "revenue_error",
                F.col("predicted_revenue") - F.col("fact_revenue"),
            )
            .withColumn(
                "absolute_percentage_error",
                F.when(
                    F.col("fact_revenue") != 0,
                    F.abs(F.col("revenue_error"))
                    / F.abs(F.col("fact_revenue")) * F.lit(100.0),
                ),
            )
            .select(
                "date",
                "model_name",
                "predicted_revenue",
                "fact_revenue",
                "revenue_error",
                "absolute_percentage_error",
            )
        )

        loaded_revenue_rows = _load_data_mart_dataframe(
            spark_handler,
            dwh_connection,
            revenue_comparison,
            revenue_table,
            order_by="(date, model_name)",
        )

        return {
            "orders_comparison_rows": loaded_orders_rows,
            "revenue_comparison_rows": loaded_revenue_rows,
        }
