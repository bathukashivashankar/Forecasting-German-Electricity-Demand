from src.electricity_demand.pipeline import make_daily_dataset
from src.electricity_demand.models.lstm_data import (
    make_sliding_windows,
    split_sequences_time_based,
)


def main():
    daily = make_daily_dataset()
    print("Daily shape:", daily.shape)

    window_size = 30  # past 30 days to predict next day
    X, y, scaler = make_sliding_windows(daily["load_gw"], window_size=window_size)
    print("X shape:", X.shape)  # (samples, 30, 1)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = split_sequences_time_based(X, y, test_fraction=0.2)
    print("Train samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])


if __name__ == "__main__":
    main()