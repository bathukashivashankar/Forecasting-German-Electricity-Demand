import numpy as np
import pandas as pd


def mean_forecast(train: pd.Series, h: int) -> pd.Series:
    """
    Forecast all future values as the mean of the training data.
    """
    mean_val = train.mean()
    index = pd.date_range(
        start=train.index[-1] + train.index.freq, periods=h, freq=train.index.freq
    )
    return pd.Series(mean_val, index=index, name="mean")


def naive_forecast(train: pd.Series, h: int) -> pd.Series:
    """
    Forecast all future values as the last observed value.
    """
    last_val = train.iloc[-1]
    index = pd.date_range(
        start=train.index[-1] + train.index.freq, periods=h, freq=train.index.freq
    )
    return pd.Series(last_val, index=index, name="naive")


def seasonal_naive_forecast(train: pd.Series, h: int, season_length: int = 52) -> pd.Series:
    """
    Seasonal naive forecast: for weekly data, repeat the last observed season.

    For a horizon h:
    - Take the last `season_length` values from the training data.
    - Repeat them enough times to cover h steps.
    - Trim to length h.
    """
    if len(train) < season_length:
        # Not enough history: fall back to naive forecast
        return naive_forecast(train, h)

    # Last full season from the end of training data
    last_season = train.iloc[-season_length:].to_numpy()

    # Repeat this pattern to cover the forecast horizon
    reps = int(np.ceil(h / season_length))
    pattern = np.tile(last_season, reps)[:h]

    index = pd.date_range(
        start=train.index[-1] + train.index.freq, periods=h, freq=train.index.freq
    )
    return pd.Series(pattern, index=index, name="seasonal_naive")


def drift_forecast(train: pd.Series, h: int) -> pd.Series:
    """
    Drift method: last value plus a linear trend projected forward.
    Formula from Hyndman:
    y_hat_{T+h} = y_T + h * (y_T - y_1) / (T - 1)
    """
    y1 = train.iloc[0]
    yT = train.iloc[-1]
    T = len(train)
    index = pd.date_range(
        start=train.index[-1] + train.index.freq, periods=h, freq=train.index.freq
    )
    drift = (yT - y1) / (T - 1)
    steps = np.arange(1, h + 1)
    values = yT + drift * steps
    return pd.Series(values, index=index, name="drift")