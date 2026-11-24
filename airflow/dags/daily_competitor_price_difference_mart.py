from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


@dag(
    dag_id="daily_competitor_price_difference_mart",
    description="Средняя ежедневная разница собственных и конкурентных цен",
    schedule="0 13 * * *",
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
    tags=["dwh", "data-marts", "pyspark", "pricing"],
)
def daily_competitor_price_difference_mart():
    @task(task_id="build_daily_competitor_price_difference")
    def build():
        from scripts.transfer_to_data_marts import (
            build_daily_competitor_price_difference_mart,
        )

        stats = build_daily_competitor_price_difference_mart()
        logger.info("Daily competitor price difference mart: %s", stats)
        return stats

    build()


daily_competitor_price_difference_mart()
