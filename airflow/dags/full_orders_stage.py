from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task


logger = logging.getLogger(__name__)


@dag(
    dag_id="full_orders_stage",
    description="Полная таблица заказов",
    schedule="0 12 * * *",
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
def full_orders_stage():

    @task(task_id="build_full_orders")
    def build():
        from scripts.transfer_to_stage_layer import (
            build_full_orders_stage,
        )

        stats = build_full_orders_stage()

        logger.info(
            "Full Orders Stage completed: %s",
            stats,
        )

        return stats

    build()


full_orders_stage()