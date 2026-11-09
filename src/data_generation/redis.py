import numpy as np
import pandas as pd

from src.data_generation.constants import RUSSIAN_TIMEZONES, USER_CLUSTERS
from src.data_generation.utils import normal_distribution_sequence


def gen_users_cities_dataframe(users_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng()
    users_df["city"] = list(
        map(lambda x: x.split(sep="/")[1], rng.choice(RUSSIAN_TIMEZONES, len(users_df)))
    )
    return users_df[["user_id", "city"]]


def gen_users_clusters_dataframe(users_df: pd.DataFrame) -> pd.DataFrame:
    users_df["price_cluster"] = list(
        map(
            lambda x: USER_CLUSTERS[x],
            normal_distribution_sequence(0, len(USER_CLUSTERS) - 1, len(users_df)),
        )
    )
    return users_df[["user_id", "price_cluster"]]