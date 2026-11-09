import uuid
from datetime import timedelta
from random import randint

import numpy as np
import pandas as pd

from src.data_generation.constants import (
    DEAD_END_STAGES,
    LAST_WORKING_DATE,
    MAX_PRICE,
    MAX_PRICE_UPDATES,
    MIN_PRICE,
    MIN_PRICE_UPDATES,
    OPENING_DATE,
    ORDER_STATES,
    ORDER_STATES_PROB,
)
from src.data_generation.utils import (
    add_random_timedelta,
    normal_distribution_sequence,
    random_datetime,
    random_datetimes,
)


def generate_interaction_history_dataframe(
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    size: int,
    id_offset: int = 0,
) -> pd.DataFrame:
    """Генерирует пачку интеракций (~size строк).

    id_offset сдвигает interaction_id, чтобы id были уникальны между пачками.
    Размер приблизительный: на один клик в среднем приходится 1.525 строк.
    """
    rng = np.random.default_rng()

    user_ids = users_df["user_id"].to_numpy()
    registration_dates = pd.to_datetime(users_df["registration_date"]).to_numpy(
        dtype="datetime64[ns]"
    )

    product_ids = products_df["product_id"].to_numpy()
    last_working_date = LAST_WORKING_DATE

    n_clicks = max(int(size / 1.525), 1)
    last_ns = np.datetime64(last_working_date, "ns")

    user_idx = rng.integers(0, len(user_ids), size=n_clicks)
    product_idx = rng.integers(0, len(product_ids), size=n_clicks)

    click_user_ids = user_ids[user_idx]
    click_product_ids = product_ids[product_idx]
    click_dt = random_datetimes(
        registration_dates[user_idx],
        last_working_date,
        n_clicks,
    )

    has_cart = rng.random(n_clicks) < 0.4
    cart_idx = np.flatnonzero(has_cart)
    cart_user_ids = click_user_ids[cart_idx]
    cart_product_ids = click_product_ids[cart_idx]
    cart_dt = np.minimum(
        add_random_timedelta(
            click_dt[cart_idx],
            max_days=14,
            max_hours=23,
            max_minutes=59,
            max_seconds=59,
        ),
        last_ns,
    )

    has_purchase = rng.random(len(cart_idx)) < 0.25
    purchase_idx = cart_idx[has_purchase]
    purchase_user_ids = cart_user_ids[has_purchase]
    purchase_product_ids = cart_product_ids[has_purchase]
    purchase_dt = np.minimum(
        add_random_timedelta(
            cart_dt[has_purchase],
            max_days=30,
            max_hours=23,
            max_minutes=59,
            max_seconds=59,
        ),
        last_ns,
    )
    purchase_order_ids = np.array([uuid.uuid4() for _ in purchase_idx], dtype=object)

    has_paid = rng.random(len(purchase_idx)) < 0.25
    paid_user_ids = purchase_user_ids[has_paid]
    paid_product_ids = purchase_product_ids[has_paid]
    paid_dt = np.minimum(
        add_random_timedelta(
            purchase_dt[has_paid],
            max_minutes=59,
            max_seconds=59,
        ),
        last_ns,
    )
    paid_order_ids = purchase_order_ids[has_paid]

    frames = [
        pd.DataFrame(
            {
                "user_id": click_user_ids,
                "product_id": click_product_ids,
                "datetime": click_dt,
                "interaction_type": "click",
                "order_id": None,
            }
        ),
        pd.DataFrame(
            {
                "user_id": cart_user_ids,
                "product_id": cart_product_ids,
                "datetime": cart_dt,
                "interaction_type": "add_to_cart",
                "order_id": None,
            }
        ),
        pd.DataFrame(
            {
                "user_id": purchase_user_ids,
                "product_id": purchase_product_ids,
                "datetime": purchase_dt,
                "interaction_type": "purchased",
                "order_id": purchase_order_ids,
            }
        ),
        pd.DataFrame(
            {
                "user_id": paid_user_ids,
                "product_id": paid_product_ids,
                "datetime": paid_dt,
                "interaction_type": "paid",
                "order_id": paid_order_ids,
            }
        ),
    ]

    interaction_history = pd.concat(frames, ignore_index=True)

    interaction_history.insert(
        0,
        "interaction_id",
        np.arange(id_offset, id_offset + len(interaction_history), dtype="int64"),
    )

    return interaction_history[
        [
            "interaction_id",
            "user_id",
            "product_id",
            "datetime",
            "interaction_type",
            "order_id",
        ]
    ]


def generate_orders_history_dataframe(interactions_df: pd.DataFrame) -> pd.DataFrame:
    # NB: раньше здесь на каждую строку выбирался случайный shop_id, который
    # потом выбрасывался (в orders_history такой колонки нет). Убрано.
    rng = np.random.default_rng()

    purchased_orders_df = interactions_df[
        interactions_df["interaction_type"] == "purchased"
    ]
    paid_orders_df = interactions_df[interactions_df["interaction_type"] == "paid"]
    canceled_orders_df = purchased_orders_df[
        ~purchased_orders_df.order_id.isin(paid_orders_df.order_id.unique())
    ]

    purchased_orders_df = purchased_orders_df[
        ["user_id", "product_id", "datetime", "order_id"]
    ]
    purchased_orders_df["stage"] = "created"
    paid_orders_df = paid_orders_df[["user_id", "product_id", "datetime", "order_id"]]
    paid_orders_df["stage"] = "accepted"
    paid_orders_df["datetime"] = add_random_timedelta(
        paid_orders_df["datetime"],
        max_minutes=59,
        max_seconds=59,
    )
    canceled_orders_df = canceled_orders_df[
        ["user_id", "product_id", "datetime", "order_id"]
    ]
    canceled_orders_df["datetime"] += timedelta(hours=1)
    canceled_orders_df["stage"] = "canceled"
    orders_history_df = [purchased_orders_df, paid_orders_df, canceled_orders_df]

    cur_df = paid_orders_df.copy()
    for gen_arr, probs in zip(ORDER_STATES, ORDER_STATES_PROB):
        cur_df["stage"] = rng.choice(gen_arr, size=len(cur_df), p=probs)
        cur_df["datetime"] = add_random_timedelta(
            cur_df["datetime"],
            max_days=0,
            max_hours=23,
            max_minutes=59,
            max_seconds=59,
        )
        orders_history_df.append(cur_df)
        cur_df = cur_df[~cur_df.stage.isin(DEAD_END_STAGES)]

    orders_history_df = pd.concat(orders_history_df, ignore_index=True)
    orders_history_df.loc[
        (orders_history_df.stage == "completed")
        & (
            orders_history_df.order_id.isin(
                orders_history_df[
                    orders_history_df["stage"] == "sent_via_delivery"
                ].order_id.unique()
            )
        ),
        "stage",
    ] = "delivered"
    return orders_history_df


def gen_products_price_history_dataframe(products_df: pd.DataFrame) -> pd.DataFrame:
    products_df = products_df[["product_id", "prime_cost"]]
    products_df["updates"] = normal_distribution_sequence(
        MIN_PRICE_UPDATES, MAX_PRICE_UPDATES, len(products_df)
    )
    products_df["effective_from"] = products_df["updates"].apply(
        lambda x: (
            [OPENING_DATE]
            + [random_datetime(OPENING_DATE, LAST_WORKING_DATE) for _ in range(x)]
        )
    )
    products_df = products_df.explode("effective_from")
    products_df["price"] = normal_distribution_sequence(
        MIN_PRICE, MAX_PRICE, len(products_df)
    )
    products_df["price"] = products_df["price"] + products_df["prime_cost"]
    return products_df[["product_id", "effective_from", "price"]]


def gen_delivery_price_history_dataframe(shops_df: pd.DataFrame) -> pd.DataFrame:
    city_df = shops_df[["city"]].drop_duplicates()
    city_df["object_type"] = "city"
    city_df = city_df.rename({"city": "object_id"}, axis=1)
    city_dates = [OPENING_DATE] + [
        random_datetime(OPENING_DATE, LAST_WORKING_DATE)
        for _ in range(randint(MIN_PRICE_UPDATES, MAX_PRICE_UPDATES))
    ]
    city_df["effective_from"] = [city_dates for _ in range(len(city_df))]
    city_df = city_df.explode("effective_from")

    shops_df = shops_df.rename({"shop_id": "object_id"}, axis=1)[
        ["object_id", "open_date", "closing_date"]
    ]
    shops_df["object_type"] = "shop"
    shops_df["updates"] = normal_distribution_sequence(
        MIN_PRICE_UPDATES, MAX_PRICE_UPDATES, len(shops_df)
    )
    shops_df["effective_from"] = shops_df["updates"].apply(
        lambda x: [random_datetime(OPENING_DATE, LAST_WORKING_DATE) for _ in range(x)]
    )
    shops_df = shops_df.explode("effective_from")
    shops_df = shops_df[
        (shops_df.open_date <= shops_df.effective_from.dt.date)
        & (
            (shops_df.closing_date >= shops_df.effective_from.dt.date)
            | (shops_df.closing_date.isna())
        )
    ]
    shops_df = shops_df[["object_id", "object_type", "effective_from"]]

    shops_df = pd.concat([city_df, shops_df], ignore_index=True)
    shops_df["price"] = normal_distribution_sequence(
        MIN_PRICE, MAX_PRICE, len(shops_df)
    )

    shops_df["object_id"] = shops_df["object_id"].map(
        lambda value: str(value) if pd.notna(value) else None
    )
    return shops_df