
from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task


logger = logging.getLogger(__name__)


DEFAULT_ARGS = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


@dag(
    dag_id="completed_orders_stage",
    description="Актуальный прогноз завершённых заказов",
    schedule="0 10 * * *",
    start_date=pendulum.datetime(
        2026, 11, 16, tz="Europe/Amsterdam"
    ),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    dagrun_timeout=timedelta(hours=2),
    default_args=DEFAULT_ARGS,
    tags=["dwh", "stage-layer", "ml"],
)
def completed_orders_stage():

    @task(task_id="build_latest_completed_orders")
    def build():
        from scripts.transfer_to_stage_layer import (
            build_latest_completed_orders_stage,
        )

        stats = build_latest_completed_orders_stage()

        logger.info("Completed orders Stage: %s", stats)

        return stats

    build()


@dag(
    dag_id="all_orders_stage",
    description="Актуальный прогноз всех заказов",
    schedule="20 10 * * *",
    start_date=pendulum.datetime(
        2026, 11, 16, tz="Europe/Amsterdam"
    ),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    dagrun_timeout=timedelta(hours=2),
    default_args=DEFAULT_ARGS,
    tags=["dwh", "stage-layer", "ml"],
)
def all_orders_stage():

    @task(task_id="build_latest_all_orders")
    def build():
        from scripts.transfer_to_stage_layer import (
            build_latest_all_orders_stage,
        )

        stats = build_latest_all_orders_stage()

        logger.info("All orders Stage: %s", stats)

        return stats

    build()


@dag(
    dag_id="revenue_stage",
    description="Актуальные прогнозы выручки по моделям",
    schedule="40 10 * * *",
    start_date=pendulum.datetime(
        2026, 11, 16, tz="Europe/Amsterdam"
    ),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    dagrun_timeout=timedelta(hours=2),
    default_args=DEFAULT_ARGS,
    tags=["dwh", "stage-layer", "ml"],
)
def revenue_stage():

    @task(task_id="build_latest_revenue")
    def build():
        from scripts.transfer_to_stage_layer import (
            build_latest_revenue_stage,
        )

        stats = build_latest_revenue_stage()

        logger.info("Revenue Stage: %s", stats)

        return stats

    build()


completed_orders_stage()
all_orders_stage()
revenue_stage()
