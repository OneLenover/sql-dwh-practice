import uuid
from datetime import timedelta

import pandas as pd

from src.data_generation.constants import (
    COMPETITORS,
    LAST_WORKING_DATE,
    MAX_PRICE,
    MIN_PRICE,
    OPENING_DATE,
)
from src.data_generation.utils import normal_distribution_sequence


def gen_competitors_products_price_history_dataframe(
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    products_df = products_df[["product_id"]]
    products_df["company_name"] = [COMPETITORS for _ in range(len(products_df))]
    products_df = products_df.explode("company_name", ignore_index=True)

    updates_dt = []
    update_dt = OPENING_DATE
    while update_dt < LAST_WORKING_DATE:
        updates_dt.append(update_dt)
        update_dt += timedelta(days=31)
    parse_ids = [uuid.uuid4() for _ in range(len(updates_dt))]

    products_df["datetime"] = [updates_dt for _ in range(len(products_df))]
    products_df["parse_id"] = [parse_ids for _ in range(len(products_df))]
    products_df = products_df.explode(["parse_id", "datetime"], ignore_index=True)
    products_df["parse_id"] = products_df["parse_id"].astype(str)
    products_df["price"] = normal_distribution_sequence(
        MIN_PRICE, MAX_PRICE, len(products_df)
    )
    products_df["cost"] = None
    products_df["value"] = None
    products_df["tmp"] = normal_distribution_sequence(0, 2, len(products_df))
    products_df.loc[products_df["tmp"] == 0, "cost"] = products_df.loc[
        products_df["tmp"] == 0, "price"
    ]
    products_df.loc[products_df["tmp"] == 2, "value"] = products_df.loc[
        products_df["tmp"] == 2, "price"
    ]
    products_df.loc[products_df["tmp"] != 1, "price"] = None
    return products_df[["parse_id", "company_name", "product_id", "datetime", "price", "cost", "value"]]
