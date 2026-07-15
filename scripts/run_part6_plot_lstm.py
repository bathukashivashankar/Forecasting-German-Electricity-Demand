import pathlib

import pandas as pd
import matplotlib.pyplot as plt


FORECAST_DIR_PART6 = pathlib.Path("outputs") / "forecasts" / "part6"
FIG_DIR_PART6 = pathlib.Path("outputs") / "figures" / "part6"


def main():
    FIG_DIR_PART6.mkdir(parents=True, exist_ok=True)

    forecast_path = FORECAST_DIR_PART6 / "lstm_daily.csv"
    # utc_timestamp as index, parse_dates to handle the timezone-aware timestamps
    forecasts = pd.read_csv(
        forecast_path,
        parse_dates=["utc_timestamp"],
    ).set_index("utc_timestamp")

    # Columns from your file
    actual_col = "actual"
    pred_col = "lstm_daily"

    if actual_col not in forecasts.columns or pred_col not in forecasts.columns:
        raise ValueError(
            f"Expected columns '{actual_col}' and '{pred_col}' in {forecast_path}, "
            f"found {forecasts.columns.tolist()}"
        )

    x_dates = forecasts.index.to_pydatetime()

    plt.figure(figsize=(12, 6))
    plt.plot(x_dates, forecasts[actual_col], label="Actual", color="black", linewidth=1.5)
    plt.plot(x_dates, forecasts[pred_col], label="LSTM forecast", color="tab:red", linewidth=1.5)

    plt.title("Daily load – 2-year LSTM forecast vs actual")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Load (GW)")
    plt.legend()
    plt.tight_layout()

    fig_path = FIG_DIR_PART6 / "lstm_daily_2y.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved LSTM forecast plot to {fig_path}")


if __name__ == "__main__":
    main()