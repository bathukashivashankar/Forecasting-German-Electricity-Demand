import itertools
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima(
    y_train: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> Any:
    """
    Fit a SARIMA model (no exogenous variables) to weekly data.
    """
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    return result


def sarima_grid_search_aic(
    y_train: pd.Series,
    p_range=range(0, 7),
    d_range=range(0, 3),
    q_range=range(0, 7),
    P_range=range(0, 2),
    D_range=range(0, 2),
    Q_range=range(0, 2),
    m: int = 52,
) -> Dict[str, Any]:
    """
    Grid search over SARIMA hyperparameters using AIC as selection criterion.

    Returns a dict with best_order, best_seasonal_order, best_aic, best_model.
    """
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None

    for p, d, q in itertools.product(p_range, d_range, q_range):
        for P, D, Q in itertools.product(P_range, D_range, Q_range):
            order = (p, d, q)
            seasonal_order = (P, D, Q, m)
            try:
                model = SARIMAX(
                    y_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)
                aic = result.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_model = result
            except Exception:
                # Some parameter combos may fail to converge; skip them
                continue

    return {
        "best_order": best_order,
        "best_seasonal_order": best_seasonal_order,
        "best_aic": best_aic,
        "best_model": best_model,
    }