from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def make_sliding_windows(
    series: pd.Series,
    window_size: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Convert a univariate time series into sliding windows for LSTM.

    - series: daily load_gw (indexed by date)
    - window_size: number of past days used as input
    - horizon: forecast horizon in days (default: 1-step ahead)

    Returns:
        X: shape (n_samples, window_size, 1)
        y: shape (n_samples,)
        scaler: fitted MinMaxScaler for inverse-transform later
    """
    values = series.values.reshape(-1, 1)

    # Scale to [0,1] for stable LSTM training
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values)

    X_list, y_list = [], []
    for i in range(len(values_scaled) - window_size - horizon + 1):
        X_list.append(values_scaled[i : i + window_size])
        y_list.append(values_scaled[i + window_size + horizon - 1, 0])

    X = np.array(X_list)  # (samples, timesteps, features=1)
    y = np.array(y_list)  # (samples,)
    return X, y, scaler


def split_sequences_time_based(
    X: np.ndarray, y: np.ndarray, test_fraction: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into train/test by a fraction of the timeline.

    Assumes X is ordered in time already.
    """
    n_samples = X.shape[0]
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_test

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, X_test, y_train, y_test