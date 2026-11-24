from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


@dag(
    dag_id="monthly_orders_by_segment_city_mart",
    description="Количество заказов по ценовым сегментам и городам магазинов за месяц",
    schedule="0 14 * * *",
    start_date=pendulum.datetime(2026, 11, 16, tz="Europe/Amsterdam"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    dagrun_timeout=timedelta(hours=3),
    default_args={
        "owner": "data-engineering",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["dwh", "data-marts", "pyspark", "orders"],
)
def monthly_orders_by_segment_city_mart():
    @task(task_id="build_monthly_orders_by_segment_city")
    def build():
        from scripts.transfer_to_data_marts import (
            build_monthly_orders_by_segment_city_mart,
        )

        stats = build_monthly_orders_by_segment_city_mart()
        logger.info("Monthly orders by segment/city mart: %s", stats)
        return stats

    build()


monthly_orders_by_segment_city_mart()
