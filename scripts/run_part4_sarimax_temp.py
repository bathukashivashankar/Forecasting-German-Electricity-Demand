import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.electricity_demand.pipeline import train_test_split_weekly, make_weekly_with_temperature
from src.electricity_demand.evaluation import plot_residual_acf_pacf


FORECAST_DIR_PART4 = pathlib.Path("outputs") / "forecasts" / "part4"
METRIC_DIR_PART4 = pathlib.Path("outputs") / "metrics" / "part4"
FIG_DIR_PART4 = pathlib.Path("outputs") / "figures" / "part4"


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def bias(y_true, y_pred):
    return np.mean(y_pred - y_true)


def main():
    FORECAST_DIR_PART4.mkdir(parents=True, exist_ok=True)
    METRIC_DIR_PART4.mkdir(parents=True, exist_ok=True)
    FIG_DIR_PART4.mkdir(parents=True, exist_ok=True)

    weekly = make_weekly_with_temperature()
    train, test = train_test_split_weekly(weekly, test_weeks=104)

    y_train = train["load_gw"]
    y_test = test["load_gw"]
    exog_train = train[["temp_mean", "temp_min", "temp_max"]]
    exog_test = test[["temp_mean", "temp_min", "temp_max"]]
    h = len(y_test)

    # Use the SARIMA orders found in Part 3 as a starting point
    order = (0, 1, 3)
    seasonal_order = (0, 1, 1, 52)

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    print("Fitted SARIMAX with temperature.")

    residuals = result.resid
    plot_residual_acf_pacf(residuals, name="sarimax_temp_weekly", fig_dir=FIG_DIR_PART4)

    # Forecast last 2 years (conditional on realised temperature)
    forecast_res = result.get_forecast(steps=h, exog=exog_test)
    y_pred = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    y_pred.index = y_test.index
    conf_int.index = y_test.index

    forecasts_df = pd.DataFrame(
        {
            "actual": y_test,
            "sarimax_temp": y_pred,
            "lower_95": conf_int.iloc[:, 0],
            "upper_95": conf_int.iloc[:, 1],
        }
    )
    forecast_path = FORECAST_DIR_PART4 / "sarimax_temp_weekly.csv"
    forecasts_df.to_csv(forecast_path)
    print(f"Saved SARIMAX+temp forecasts to {forecast_path}")

    metrics = pd.DataFrame(
        [
            {
                "model": "sarimax_temp",
                "MAE": mae(y_test, y_pred),
                "RMSE": rmse(y_test, y_pred),
                "Bias": bias(y_test, y_pred),
            }
        ]
    )
    metrics_path = METRIC_DIR_PART4 / "sarimax_temp_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved SARIMAX+temp metrics to {metrics_path}")

    # === NEW: Plot actual vs SARIMAX+temp forecast with 95% prediction intervals ===
    x_dates = forecasts_df.index.to_pydatetime()

    plt.figure(figsize=(12, 6))
    plt.plot(x_dates, forecasts_df["actual"], label="Actual", color="black", linewidth=2)
    plt.plot(
        x_dates,
        forecasts_df["sarimax_temp"],
        label="SARIMAX+temp forecast",
        color="tab:green",
        linewidth=2,
    )

    plt.fill_between(
        x_dates,
        forecasts_df["lower_95"].astype(float).values,
        forecasts_df["upper_95"].astype(float).values,
        color="tab:green",
        alpha=0.2,
        label="95% prediction interval",
    )

    plt.title("Weekly load – 2-year SARIMAX (with temperature) forecast with 95% prediction intervals")
    plt.xlabel("Date")
    plt.ylabel("Load (GW)")
    plt.legend()
    plt.tight_layout()

    fig_path = FIG_DIR_PART4 / "sarimax_temp_weekly_2y_ci.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SARIMAX+temp forecast plot with CI to {fig_path}")


if __name__ == "__main__":
    main()