
from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task


logger = logging.getLogger(__name__)


@dag(
    dag_id="daily_interactions_stage",
    description="Ежедневное количество интеракций по типам",
    schedule="0 11 * * *",
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
def daily_interactions_stage():

    @task(task_id="build_daily_interactions")
    def build():
        from scripts.transfer_to_stage_layer import (
            build_daily_interactions_stage,
        )

        stats = build_daily_interactions_stage()

        logger.info(
            "Daily interactions Stage completed: %s",
            stats,
        )

        return stats

    build()


daily_interactions_stage()