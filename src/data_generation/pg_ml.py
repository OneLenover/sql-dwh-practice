from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.data_generation.constants import (
    AMPLIFY_COEF_MAX,
    AMPLIFY_COEF_MIN,
    DAILY_PREDICT_START_DATE,
    DAILY_REVENUE_PREDICT_START_DATE,
    LAST_WORKING_DATE,
    MAX_DAILY_ORDERS,
    MAX_DAILY_REVENUE,
    MIN_DAILY_ORDERS,
    MIN_DAILY_REVENUE,
)
from src.data_generation.utils import normal_distribution_sequence


def gen_shops_daily_completed_orders_predict(shops_df: pd.DataFrame) -> pd.DataFrame:
    shops_daily_orders_predict = []
    shops_df = shops_df[["shop_id"]]
    cur_date = DAILY_PREDICT_START_DATE
    end_date = LAST_WORKING_DATE.date()
    while cur_date <= end_date:
        dates = pd.DataFrame(
            {"forecast_date": pd.date_range(cur_date, cur_date + timedelta(days=7))}
        )
        cur_daily_orders_predict = shops_df.merge(dates, how="cross")
        cur_daily_orders_predict["predict_orders"] = normal_distribution_sequence(
            MIN_DAILY_ORDERS, MAX_DAILY_ORDERS, len(cur_daily_orders_predict)
        )
        cur_daily_orders_predict["creation_date"] = cur_date
        shops_daily_orders_predict.append(cur_daily_orders_predict)
        cur_date += timedelta(days=1)
    return pd.concat(shops_daily_orders_predict, ignore_index=True)


def iter_shops_daily_orders_predict(
    shops_df: pd.DataFrame, days_per_chunk: int = 14, horizon_days: int = 8
):
    """Потоково генерирует прогнозы заказов пачками по creation_date.

    Отдаёт пары (completed_chunk, all_chunk). Прогноз "all" в каждой строке
    не меньше прогноза "completed". forecast_date и creation_date — python date.
    """
    shop_ids = shops_df["shop_id"].to_numpy()
    n_shops = len(shop_ids)
    offsets = np.tile(np.tile(np.arange(horizon_days), n_shops), 1)

    cur = np.datetime64(DAILY_PREDICT_START_DATE, "D")
    end = np.datetime64(LAST_WORKING_DATE.date(), "D")

    while cur <= end:
        chunk_end = min(cur + np.timedelta64(days_per_chunk - 1, "D"), end)
        creation = np.arange(cur, chunk_end + np.timedelta64(1, "D"), dtype="datetime64[D]")
        n_days = len(creation)

        creation_col = np.repeat(creation, n_shops * horizon_days)
        forecast_col = creation_col + np.tile(offsets, n_days).astype("timedelta64[D]")
        shop_col = np.tile(np.repeat(shop_ids, horizon_days), n_days)

        n = len(creation_col)
        completed = np.array(
            normal_distribution_sequence(MIN_DAILY_ORDERS, MAX_DAILY_ORDERS, n)
        )
        all_orders = (
            completed * np.random.uniform(AMPLIFY_COEF_MIN, AMPLIFY_COEF_MAX, n)
        ).astype(int)

        base = {
            "shop_id": shop_col,
            "forecast_date": forecast_col.astype(object),
            "creation_date": creation_col.astype(object),
        }
        completed_df = pd.DataFrame({**base, "predict_orders": completed})
        all_df = pd.DataFrame({**base, "predict_orders": all_orders})
        cols = ["shop_id", "forecast_date", "predict_orders", "creation_date"]
        yield completed_df[cols], all_df[cols]

        cur = chunk_end + np.timedelta64(1, "D")


def gen_shops_daily_all_orders_predict(
    shops_daily_completed_orders_predict_df: pd.DataFrame,
) -> pd.DataFrame:
    shops_daily_completed_orders_predict_df["predict_orders"] = (
        shops_daily_completed_orders_predict_df["predict_orders"]
        * np.random.uniform(
            AMPLIFY_COEF_MIN,
            AMPLIFY_COEF_MAX,
            (len(shops_daily_completed_orders_predict_df),),
        )
    ).astype(int)
    return shops_daily_completed_orders_predict_df


def _gen_daily_revenue_predict(
    start_date: date, end_date: date, predict_horizon: int, next_predict: int
) -> pd.DataFrame:
    daily_predict = []
    cur_date = start_date

    while cur_date <= end_date:
        cur_daily_predict = pd.DataFrame(
            {
                "forecast_date": pd.date_range(
                    cur_date, cur_date + timedelta(days=predict_horizon), freq="d"
                ).date
            }
        )
        cur_daily_predict["predict_revenue"] = normal_distribution_sequence(
            MIN_DAILY_REVENUE, MAX_DAILY_REVENUE, len(cur_daily_predict)
        )
        cur_daily_predict["creation_date"] = cur_date
        daily_predict.append(cur_daily_predict)
        cur_date += timedelta(days=next_predict)
    return pd.concat(daily_predict, ignore_index=True)


def gen_daily_revenue_predict() -> pd.DataFrame:
    daily_predict1 = _gen_daily_revenue_predict(
        DAILY_REVENUE_PREDICT_START_DATE, LAST_WORKING_DATE.date(), 31, 15
    )
    daily_predict1["model_name"] = "catboost"
    daily_predict2 = _gen_daily_revenue_predict(
        DAILY_REVENUE_PREDICT_START_DATE, LAST_WORKING_DATE.date(), 93, 31
    )
    daily_predict2["model_name"] = "TRex"
    return pd.concat([daily_predict1, daily_predict2], ignore_index=True)