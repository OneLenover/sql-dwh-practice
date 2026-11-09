import uuid
from datetime import date, datetime, timedelta
from random import choice

import numpy as np
import pandas as pd

from src.data_generation.constants import (
    DAYS_IN_YEAR,
    END_BIRTHDAY_DATE,
    GENDERS,
    LAST_WORKING_DATE,
    MAX_PRICE,
    MIN_PRICE,
    OPENING_DATE,
    RUSSIAN_TIMEZONES,
    START_BIRTHDAY_DATE,
)
from src.data_generation.utils import normal_distribution_sequence, random_datetime


def _generate_closing_date(opening_date: date) -> datetime:
    d1 = datetime.fromisoformat(opening_date.isoformat()) + timedelta(days=DAYS_IN_YEAR)
    d2 = LAST_WORKING_DATE + timedelta(days=DAYS_IN_YEAR * 10)
    return random_datetime(d1, d2).date()


def generate_shops_dataframe(num_unique_shops: int) -> pd.DataFrame:
    rng = np.random.default_rng()

    unique_shops = [uuid.uuid4() for _ in range(num_unique_shops)]
    shops_opening_date = [
        random_datetime(OPENING_DATE, LAST_WORKING_DATE).date()
        for _ in range(num_unique_shops)
    ]
    shops_closing_date = [_generate_closing_date(i) for i in shops_opening_date]
    shops_timezone = rng.choice(RUSSIAN_TIMEZONES, num_unique_shops)
    shops_city = [i.split(sep="/")[1] for i in shops_timezone]
    shops_only_picking = rng.choice([0, 1], size=num_unique_shops, p=[0.9, 0.1])
    shops_removed = rng.choice([0, 1], size=num_unique_shops, p=[0.85, 0.15])

    df = pd.DataFrame(
        {
            "shop_id": unique_shops,
            "city": shops_city,
            "timezone": shops_timezone,
            "is_only_picking": shops_only_picking,
            "is_removed": shops_removed,
            "open_date": shops_opening_date,
            "closing_date": shops_closing_date,
        }
    )
    df = df.astype({"is_only_picking": bool, "is_removed": bool})
    df.loc[df.closing_date >= LAST_WORKING_DATE.date(), "closing_date"] = None
    return df


def _generate_registration_datetime(birthday: date) -> datetime:
    d1 = max(
        OPENING_DATE,
        datetime.fromisoformat(birthday.isoformat()) + timedelta(days=DAYS_IN_YEAR),
    )
    d2 = LAST_WORKING_DATE
    return random_datetime(d1, d2)


def generate_users_dataframe(num_unique_users: int) -> pd.DataFrame:
    start_birthday_date = date.fromisoformat(START_BIRTHDAY_DATE).toordinal()
    end_birthday_date = date.fromisoformat(END_BIRTHDAY_DATE).toordinal()

    unique_users = [uuid.uuid4() for _ in range(num_unique_users)]
    users_birthday = list(
        map(
            date.fromordinal,
            normal_distribution_sequence(
                start_birthday_date,
                end_birthday_date,
                size=num_unique_users
            ),
        )
    )
    user_registration_date = [
        _generate_registration_datetime(i) for i in users_birthday
    ]
    use_gender = [choice(GENDERS) for _ in range(num_unique_users)]
    return pd.DataFrame(
        {
            "user_id": unique_users,
            "gender": use_gender,
            "birthday": users_birthday,
            "registration_date": user_registration_date,
        }
    )


def generate_products_dataframe(products_base_df: pd.DataFrame) -> pd.DataFrame:
    products_base_df["prime_cost"] = normal_distribution_sequence(
        MIN_PRICE, MAX_PRICE, len(products_base_df)
    )
    return products_base_df
