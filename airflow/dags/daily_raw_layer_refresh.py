from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


@dag(
    dag_id="daily_raw_layer_refresh",
    description="Ежедневный перенос данных из источников в ClickHouse raw_layer через PySpark",
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2026, 11, 16, tz="Europe/Amsterdam"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    dagrun_timeout=timedelta(hours=6),
    default_args={
        "owner": "data-engineering",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["dwh", "raw-layer", "pyspark"],
)
def daily_raw_layer_refresh():
    @task(task_id="transfer_data_to_raw_layer")
    def transfer_data() -> dict[str, int]:
        # Import at task runtime, not while the DAG processor parses the DAG.
        from scripts.transfer_to_raw_layer import transfer_data_to_raw_layer

        stats = transfer_data_to_raw_layer()
        logger.info("Raw Layer refresh completed. Loaded rows: %s", stats)
        return stats

    transfer_data()


daily_raw_layer_refresh()
