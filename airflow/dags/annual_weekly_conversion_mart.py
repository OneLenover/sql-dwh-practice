from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


@dag(
    dag_id="annual_weekly_conversion_mart",
    description="Средняя недельная конверсия сайт-интеракций в завершённые заказы за год",
    schedule="0 15 * * *",
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
    tags=["dwh", "data-marts", "pyspark", "conversion"],
)
def annual_weekly_conversion_mart():
    @task(task_id="build_annual_weekly_conversion")
    def build():
        from scripts.transfer_to_data_marts import (
            build_annual_weekly_conversion_mart,
        )

        stats = build_annual_weekly_conversion_mart()
        logger.info("Annual weekly conversion mart: %s", stats)
        return stats

    build()


annual_weekly_conversion_mart()
