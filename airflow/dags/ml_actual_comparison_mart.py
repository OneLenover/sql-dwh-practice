from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


@dag(
    dag_id="ml_actual_comparison_mart",
    description="Сравнение прогнозов заказов и выручки с фактическими значениями",
    schedule="0 17 * * *",
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
    tags=["dwh", "data-marts", "pyspark", "ml"],
)
def ml_actual_comparison_mart():
    @task(task_id="build_ml_actual_comparison")
    def build():
        from scripts.transfer_to_data_marts import (
            build_ml_actual_comparison_mart,
        )

        stats = build_ml_actual_comparison_mart()
        logger.info("ML actual comparison mart: %s", stats)
        return stats

    build()


ml_actual_comparison_mart()
