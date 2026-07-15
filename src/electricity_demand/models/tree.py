from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_feature_target_arrays(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy()
    y = df["target"].to_numpy()
    return X, y


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestRegressor:
    """
    Basic Random Forest for regression.
    """
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"MAE": mae, "RMSE": rmse, "Bias": bias}