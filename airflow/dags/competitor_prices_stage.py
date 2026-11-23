
from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task


logger = logging.getLogger(__name__)


@dag(
    dag_id="competitor_prices_stage",
    description=(
        "Создание Stage-витрины с историей цен конкурентов"
    ),
    schedule="0 10 * * *",
    start_date=pendulum.datetime(
        2026, 11, 16, tz="Europe/Amsterdam"
    ),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    dagrun_timeout=timedelta(hours=2),
    default_args={
        "owner": "data-engineering",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["dwh", "stage-layer", "pyspark"],
)
def competitor_prices_stage():

    @task(task_id="build_competitor_prices_stage")
    def build_stage() -> dict[str, int]:

        # Импортируем модуль во время выполнения задачи.
        from scripts.transfer_to_stage_layer import (
            build_competitor_prices_stage,
        )

        stats = build_competitor_prices_stage()

        logger.info(
            "Competitor prices Stage refresh completed: %s",
            stats,
        )

        return stats

    build_stage()


competitor_prices_stage()