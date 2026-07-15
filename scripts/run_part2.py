import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # NEW

from src.electricity_demand.pipeline import make_weekly_dataset, train_test_split_weekly
from src.electricity_demand.models.benchmarks import (
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    drift_forecast,
)


FORECAST_DIR_PART2 = pathlib.Path("outputs") / "forecasts" / "part2"
METRIC_DIR_PART2 = pathlib.Path("outputs") / "metrics" / "part2"
FIGURE_DIR_PART2 = pathlib.Path("outputs") / "figures" / "part2"  # NEW


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def bias(y_true, y_pred):
    return np.mean(y_pred - y_true)


def main():
    FORECAST_DIR_PART2.mkdir(parents=True, exist_ok=True)
    METRIC_DIR_PART2.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR_PART2.mkdir(parents=True, exist_ok=True)  # NEW

    weekly = make_weekly_dataset()
    train, test = train_test_split_weekly(weekly, test_weeks=104)  # 2-year horizon
    h = len(test)

    y_train = train["load_gw"]
    y_test = test["load_gw"]

    # Generate forecasts
    fc_mean = mean_forecast(y_train, h)
    fc_naive = naive_forecast(y_train, h)
    fc_seasonal_naive = seasonal_naive_forecast(y_train, h, season_length=52)
    fc_drift = drift_forecast(y_train, h)

    # Combine into one DataFrame aligned with test index
    forecasts = pd.DataFrame(
        {
            "actual": y_test,
            "mean": fc_mean.reindex(y_test.index),
            "naive": fc_naive.reindex(y_test.index),
            "seasonal_naive": fc_seasonal_naive.reindex(y_test.index),
            "drift": fc_drift.reindex(y_test.index),
        }
    )

    # Save forecasts
    forecast_path = FORECAST_DIR_PART2 / "benchmarks_weekly.csv"
    forecasts.to_csv(forecast_path)
    print(f"Saved benchmark forecasts to {forecast_path}")

    # Compute metrics
    rows = []
    for model_name in ["mean", "naive", "seasonal_naive", "drift"]:
        y_pred = forecasts[model_name]
        rows.append(
            {
                "model": model_name,
                "MAE": mae(y_test, y_pred),
                "RMSE": rmse(y_test, y_pred),
                "Bias": bias(y_test, y_pred),
            }
        )
    metrics_df = pd.DataFrame(rows)
    metric_path = METRIC_DIR_PART2 / "benchmarks_metrics.csv"
    metrics_df.to_csv(metric_path, index=False)
    print(f"Saved benchmark metrics to {metric_path}")

    # === NEW: Plot actual vs benchmark forecasts on 2-year horizon ===
    plt.figure(figsize=(12, 6))
    plt.plot(forecasts.index, forecasts["actual"], label="Actual", color="black", linewidth=2)
    plt.plot(forecasts.index, forecasts["mean"], label="Mean", linestyle="--")
    plt.plot(forecasts.index, forecasts["naive"], label="Naive", linestyle="--")
    plt.plot(forecasts.index, forecasts["seasonal_naive"], label="Seasonal naive", linestyle="--")
    plt.plot(forecasts.index, forecasts["drift"], label="Drift", linestyle="--")

    plt.title("Weekly load – 2-year test horizon with benchmark forecasts")
    plt.xlabel("Date")
    plt.ylabel("Load (GW)")
    plt.legend()
    plt.tight_layout()

    fig_path = FIGURE_DIR_PART2 / "benchmarks_weekly_2y.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved benchmark comparison plot to {fig_path}")


if __name__ == "__main__":
    main()