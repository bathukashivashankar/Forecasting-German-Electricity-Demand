import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras

from src.electricity_demand.pipeline import make_daily_dataset
from src.electricity_demand.models.lstm_data import (
    make_sliding_windows,
    split_sequences_time_based,
)
from src.electricity_demand.models.lstm_model import build_univariate_lstm

FORECAST_DIR_PART6 = pathlib.Path("outputs") / "forecasts" / "part6"
METRIC_DIR_PART6 = pathlib.Path("outputs") / "metrics" / "part6"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))


def main():
    FORECAST_DIR_PART6.mkdir(parents=True, exist_ok=True)
    METRIC_DIR_PART6.mkdir(parents=True, exist_ok=True)

    daily = make_daily_dataset()
    series = daily["load_gw"]

    window_size = 30  # past 30 days -> next day
    X, y, scaler = make_sliding_windows(series, window_size=window_size, horizon=1)
    X_train, X_test, y_train, y_test = split_sequences_time_based(X, y, test_fraction=0.2)

    # Build model
    model = build_univariate_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    # Train
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        shuffle=False,  # keep time order
        callbacks=callbacks,
    )

    # Predict on test set (still in scaled space)
    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse scale y_true and y_pred to get back to GW
    # y and predictions are scaled; need to invert using scaler
    # Build arrays with correct shape for inverse_transform
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Metrics in original units
    mae_val = float(mean_absolute_error(y_test_inv, y_pred_inv))
    rmse_val = rmse(y_test_inv, y_pred_inv)
    bias_val = bias(y_test_inv, y_pred_inv)

    print("LSTM MAE:", mae_val)
    print("LSTM RMSE:", rmse_val)
    print("LSTM Bias:", bias_val)

    # Match test timestamps (last len(y_test_inv) days of the series)
    test_index = series.index[-len(y_test_inv) :]

    forecasts_df = pd.DataFrame(
        {
            "utc_timestamp": test_index,
            "actual": y_test_inv,
            "lstm_daily": y_pred_inv,
        }
    ).set_index("utc_timestamp")

    forecast_path = FORECAST_DIR_PART6 / "lstm_daily.csv"
    forecasts_df.to_csv(forecast_path)
    print(f"Saved LSTM forecasts to {forecast_path}")

    metrics_df = pd.DataFrame(
        [
            {
                "model": "lstm_daily",
                "MAE": mae_val,
                "RMSE": rmse_val,
                "Bias": bias_val,
            }
        ]
    )
    metrics_path = METRIC_DIR_PART6 / "lstm_daily_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved LSTM metrics to {metrics_path}")


if __name__ == "__main__":
    main()