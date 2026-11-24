from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


@dag(
    dag_id="user_product_orders_4m_mart",
    description="Матрица заказов пользователей по товарам за последние четыре месяца",
    schedule="0 16 * * *",
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
    tags=["dwh", "data-marts", "pyspark", "user-products"],
)
def user_product_orders_4m_mart():
    @task(task_id="build_user_product_orders_4m")
    def build():
        from scripts.transfer_to_data_marts import (
            build_user_product_orders_4m_mart,
        )

        stats = build_user_product_orders_4m_mart()
        logger.info("User product orders 4m mart: %s", stats)
        return stats

    build()


user_product_orders_4m_mart()
