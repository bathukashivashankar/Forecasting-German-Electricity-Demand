import pathlib

import pandas as pd
import matplotlib.pyplot as plt


METRIC_ROOT = pathlib.Path("outputs") / "metrics"
FIG_ROOT = pathlib.Path("outputs") / "figures"


def main():
    comparison_path = METRIC_ROOT / "model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"{comparison_path} not found. Run make_model_comparison.py first.")

    df = pd.read_csv(comparison_path)

    # Optional: rename model labels to be more readable
    name_map = {
        "rf_weekly": "RandomForest",
        "lstm_daily": "LSTM",
        "sarimax_temp": "SARIMAX+Temp",
    }
    df["model"] = df["model"].replace(name_map)

    # Sort by RMSE (ascending) so best models appear first
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE")
    else:
        raise ValueError("RMSE column not found in model_comparison.csv")

    models = df["model"].tolist()
    rmse_values = df["RMSE"].tolist()

    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, rmse_values, color="tab:blue", alpha=0.8)

    plt.title("Model comparison – RMSE on 2-year test horizon")
    plt.ylabel("RMSE")
    plt.xlabel("Model")

    # Annotate bars with RMSE values
    for bar, value in zip(bars, rmse_values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    fig_path = FIG_ROOT / "model_comparison_rmse.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overall model comparison plot to {fig_path}")


if __name__ == "__main__":
    main()