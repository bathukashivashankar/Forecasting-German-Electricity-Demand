import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.electricity_demand.pipeline import make_weekly_dataset, train_test_split_weekly
from src.electricity_demand.models.sarimax import sarima_grid_search_aic
from src.electricity_demand.evaluation import plot_residual_acf_pacf


FORECAST_DIR_PART3 = pathlib.Path("outputs") / "forecasts" / "part3"
METRIC_DIR_PART3 = pathlib.Path("outputs") / "metrics" / "part3"
FIG_DIR_PART3 = pathlib.Path("outputs") / "figures" / "part3"


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def bias(y_true, y_pred):
    return np.mean(y_pred - y_true)


def main():
    FORECAST_DIR_PART3.mkdir(parents=True, exist_ok=True)
    METRIC_DIR_PART3.mkdir(parents=True, exist_ok=True)
    FIG_DIR_PART3.mkdir(parents=True, exist_ok=True)

    weekly = make_weekly_dataset()
    train, test = train_test_split_weekly(weekly, test_weeks=104)
    y_train = train["load_gw"]
    y_test = test["load_gw"]
    h = len(y_test)

    print("Starting SARIMA grid search (may take some time)...")
    search_result = sarima_grid_search_aic(
        y_train,
        p_range=range(0, 4),
        d_range=range(0, 2),
        q_range=range(0, 4),
        P_range=range(0, 2),
        D_range=range(0, 2),
        Q_range=range(0, 2),
        m=52,
    )

    best_order = search_result["best_order"]
    best_seasonal_order = search_result["best_seasonal_order"]
    best_aic = search_result["best_aic"]
    best_model = search_result["best_model"]

    print("Best SARIMA order:", best_order)
    print("Best seasonal order:", best_seasonal_order)
    print("Best AIC:", best_aic)

    meta = pd.DataFrame(
        {
            "best_order": [best_order],
            "best_seasonal_order": [best_seasonal_order],
            "best_aic": [best_aic],
        }
    )
    meta.to_csv(METRIC_DIR_PART3 / "sarima_best_model.csv", index=False)

    residuals = best_model.resid
    plot_residual_acf_pacf(residuals, name="sarima_weekly", fig_dir=FIG_DIR_PART3)

    sarima_forecast = best_model.get_forecast(steps=h)
    y_pred = sarima_forecast.predicted_mean
    conf_int = sarima_forecast.conf_int(alpha=0.05)

    y_pred.index = y_test.index
    conf_int.index = y_test.index

    forecasts_df = pd.DataFrame(
        {
            "actual": y_test,
            "sarima": y_pred,
            "lower_95": conf_int.iloc[:, 0],
            "upper_95": conf_int.iloc[:, 1],
        }
    )

    forecast_path = FORECAST_DIR_PART3 / "sarima_weekly.csv"
    forecasts_df.to_csv(forecast_path)
    print(f"Saved SARIMA forecasts to {forecast_path}")

    metrics = pd.DataFrame(
        [
            {
                "model": "sarima",
                "MAE": mae(y_test, y_pred),
                "RMSE": rmse(y_test, y_pred),
                "Bias": bias(y_test, y_pred),
            }
        ]
    )
    metrics_path = METRIC_DIR_PART3 / "sarima_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved SARIMA metrics to {metrics_path}")

    x_dates = forecasts_df.index.to_pydatetime()

    plt.figure(figsize=(12, 6))
    plt.plot(x_dates, forecasts_df["actual"], label="Actual", color="black", linewidth=2)
    plt.plot(x_dates, forecasts_df["sarima"], label="SARIMA forecast", color="tab:blue", linewidth=2)

    plt.fill_between(
        x_dates,
        forecasts_df["lower_95"].astype(float).values,
        forecasts_df["upper_95"].astype(float).values,
        color="tab:blue",
        alpha=0.2,
        label="95% prediction interval",
    )

    plt.title("Weekly load – 2-year SARIMA forecast with 95% prediction intervals")
    plt.xlabel("Date")
    plt.ylabel("Load (GW)")
    plt.legend()
    plt.tight_layout()

    fig_path = FIG_DIR_PART3 / "sarima_weekly_2y_ci.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SARIMA forecast plot with CI to {fig_path}")


if __name__ == "__main__":
    main()