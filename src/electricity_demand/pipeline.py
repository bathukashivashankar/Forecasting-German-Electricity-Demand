from typing import Tuple

import pandas as pd

from .data import (
    load_opsd_raw,
    filter_germany_load,
    resample_to_weekly,
)

import numpy as np
import pandas as pd

from .data import (
    load_opsd_raw,
    filter_germany_load,
    resample_to_weekly,
    load_berlin_temperature_daily,
    resample_temperature_to_weekly,
)

def make_weekly_dataset() -> pd.DataFrame:
    """
    Full pipeline to create the weekly modelling dataset.
    """
    raw = load_opsd_raw()
    load = filter_germany_load(raw)
    weekly = resample_to_weekly(load)
    return weekly


def train_test_split_weekly(
    df: pd.DataFrame, test_weeks: int = 104
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split weekly data into train and test sets using the last `test_weeks` as test.
    """
    df = df.sort_index()
    train = df.iloc[:-test_weeks].copy()
    test = df.iloc[-test_weeks:].copy()
    return train, test

from .data import (
    load_opsd_raw,
    filter_germany_load,
    resample_to_weekly,
    load_berlin_temperature_daily,
    resample_temperature_to_weekly,
)

def make_weekly_with_temperature() -> pd.DataFrame:
    """
    Create weekly dataset with load and Berlin temperature features.
    """
    # Weekly load
    raw = load_opsd_raw()
    load = filter_germany_load(raw)
    weekly_load = resample_to_weekly(load)  # column: load_gw

    # Weekly temperature
    temp_daily = load_berlin_temperature_daily()
    weekly_temp = resample_temperature_to_weekly(temp_daily)

    # Align on common weekly index (inner join)
    weekly = weekly_load.join(weekly_temp, how="inner")
    return weekly



def make_weekly_with_temperature() -> pd.DataFrame:
    """
    Weekly dataset with load and Berlin temperature.
    """
    raw = load_opsd_raw()
    load = filter_germany_load(raw)
    weekly_load = resample_to_weekly(load)  # load_gw

    temp_daily = load_berlin_temperature_daily()
    weekly_temp = resample_temperature_to_weekly(temp_daily)

    weekly = weekly_load.join(weekly_temp, how="inner")
    return weekly


def make_weekly_feature_dataset(
    max_lag: int = 4,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Create a supervised learning dataset for weekly forecasting:

    - Target is load_gw at t + horizon.
    - Features: lagged load, calendar features, temperature at time t.
    """
    weekly = make_weekly_with_temperature().copy()

    # Calendar features
    weekly["weekofyear"] = weekly.index.isocalendar().week.astype(int)
    weekly["year"] = weekly.index.year
    weekly["month"] = weekly.index.month

    # Lag features of load
    for lag in range(1, max_lag + 1):
        weekly[f"load_lag_{lag}"] = weekly["load_gw"].shift(lag)

    # Target: future load (horizon weeks ahead)
    weekly["target"] = weekly["load_gw"].shift(-horizon)

    # Drop rows with NaNs (from lagging/leading)
    weekly = weekly.dropna()

    return weekly

from typing import Tuple

def train_test_split_features(df: pd.DataFrame, test_weeks: int = 104) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a feature dataset with 'target' column into train/test
    using the last `test_weeks` rows as test.
    """
    df = df.sort_index()
    train = df.iloc[:-test_weeks].copy()
    test = df.iloc[-test_weeks:].copy()
    return train, test

from .data import (
    load_opsd_raw,
    filter_germany_load,
    resample_to_daily,
    # other imports...
)


def make_daily_dataset() -> pd.DataFrame:
    """
    Create the daily German load dataset in GW.
    """
    raw = load_opsd_raw()
    load = filter_germany_load(raw)
    daily = resample_to_daily(load)  # column: load_gw, index: daily UTC
    return daily