import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.electricity_demand.pipeline import (
    make_weekly_feature_dataset,
    train_test_split_features,
)
from src.electricity_demand.models.tree import (
    get_feature_target_arrays,
    train_random_forest,
    compute_metrics,
)


FORECAST_DIR_PART5 = pathlib.Path("outputs") / "forecasts" / "part5"
METRIC_DIR_PART5 = pathlib.Path("outputs") / "metrics" / "part5"
FIG_DIR_PART5 = pathlib.Path("outputs") / "figures" / "part5"


def main():
    FORECAST_DIR_PART5.mkdir(parents=True, exist_ok=True)
    METRIC_DIR_PART5.mkdir(parents=True, exist_ok=True)
    FIG_DIR_PART5.mkdir(parents=True, exist_ok=True)

    # 1. Build feature dataset
    data = make_weekly_feature_dataset(max_lag=4, horizon=1)

    # 2. Split into train and test (last 104 weeks)
    train_df, test_df = train_test_split_features(data, test_weeks=104)

    # 3. Define feature columns
    feature_cols = [
        "load_gw",
        "temp_mean",
        "temp_min",
        "temp_max",
        "weekofyear",
        "month",
        "year",
        "load_lag_1",
        "load_lag_2",
        "load_lag_3",
        "load_lag_4",
    ]

    X_train, y_train = get_feature_target_arrays(train_df, feature_cols)
    X_test, y_test = get_feature_target_arrays(test_df, feature_cols)

    # 4. Train Random Forest
    rf = train_random_forest(X_train, y_train)

    # 5. Predict test period
    y_pred = rf.predict(X_test)

    # 6. Build forecast DataFrame
    forecasts = pd.DataFrame(
        {
            "utc_timestamp": test_df.index,
            "actual": y_test,
            "rf_weekly": y_pred,
        }
    ).set_index("utc_timestamp")

    forecast_path = FORECAST_DIR_PART5 / "rf_weekly.csv"
    forecasts.to_csv(forecast_path)
    print(f"Saved tree-based forecasts to {forecast_path}")

    # 7. Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    metrics_df = pd.DataFrame(
        [
            {
                "model": "rf_weekly",
                **metrics,
            }
        ]
    )
    metrics_path = METRIC_DIR_PART5 / "rf_weekly_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved tree-based metrics to {metrics_path}")

    # 8. Plot actual vs Random Forest forecast on 2-year weekly horizon
    x_dates = forecasts.index.to_pydatetime()

    plt.figure(figsize=(12, 6))
    plt.plot(x_dates, forecasts["actual"], label="Actual", color="black", linewidth=2)
    plt.plot(
        x_dates,
        forecasts["rf_weekly"],
        label="Random Forest forecast",
        color="tab:orange",
        linewidth=2,
    )

    plt.title("Weekly load – 2-year Random Forest forecast (feature-based model)")
    plt.xlabel("Date")
    plt.ylabel("Load (GW)")
    plt.legend()
    plt.tight_layout()

    fig_path = FIG_DIR_PART5 / "rf_weekly_2y.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Random Forest forecast plot to {fig_path}")


if __name__ == "__main__":
    main()