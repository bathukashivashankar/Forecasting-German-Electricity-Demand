import pathlib
import pandas as pd


METRIC_ROOT = pathlib.Path("outputs") / "metrics"
OUTPUT_COMPARISON = METRIC_ROOT / "model_comparison.csv"


def load_and_tag(path: pathlib.Path, source_label: str) -> pd.DataFrame:
    """Load a metrics CSV and add a source_file column."""
    df = pd.read_csv(path)
    df["source_file"] = source_label
    return df


def main():
    rows = []

    # Part 2 – benchmark metrics (mean, naive, seasonal_naive, drift)
    part2_path = METRIC_ROOT / "part2" / "benchmarks_metrics.csv"
    if part2_path.exists():
        df_part2 = load_and_tag(part2_path, "part2_benchmarks_metrics")
        rows.append(df_part2)

    # Part 3 – SARIMA metrics
    part3_path = METRIC_ROOT / "part3" / "sarima_metrics.csv"
    if part3_path.exists():
        df_part3 = load_and_tag(part3_path, "part3_sarima_metrics")
        rows.append(df_part3)

    # Part 4 – SARIMAX+temperature metrics
    part4_path = METRIC_ROOT / "part4" / "sarimax_temp_metrics.csv"
    if part4_path.exists():
        df_part4 = load_and_tag(part4_path, "part4_sarimax_temp_metrics")
        rows.append(df_part4)

    # Part 5 – Random Forest weekly metrics
    part5_path = METRIC_ROOT / "part5" / "rf_weekly_metrics.csv"
    if part5_path.exists():
        df_part5 = load_and_tag(part5_path, "part5_rf_weekly_metrics")
        rows.append(df_part5)

    # Part 6 – LSTM daily metrics
    part6_path = METRIC_ROOT / "part6" / "lstm_daily_metrics.csv"
    if part6_path.exists():
        df_part6 = load_and_tag(part6_path, "part6_lstm_daily_metrics")
        rows.append(df_part6)

    if not rows:
        print("No metrics files found; model_comparison.csv not created.")
        return

    comparison_df = pd.concat(rows, ignore_index=True)

    # Optional: ensure consistent column order
    expected_cols = ["model", "MAE", "RMSE", "MASE", "Bias", "source_file"]
    cols_present = [c for c in expected_cols if c in comparison_df.columns]
    comparison_df = comparison_df[cols_present]

    OUTPUT_COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_COMPARISON, index=False)
    print(f"Saved combined model comparison to {OUTPUT_COMPARISON}")


if __name__ == "__main__":
    main()