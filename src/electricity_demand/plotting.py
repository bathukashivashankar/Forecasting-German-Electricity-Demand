import pathlib
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR_PART1 = pathlib.Path("outputs") / "figures" / "part1"


def plot_daily_series(daily: pd.DataFrame, save: bool = True) -> None:
    FIG_DIR_PART1.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    daily["load_gw"].plot()
    plt.title("Daily German Electricity Load (GW)")
    plt.ylabel("Load (GW)")
    plt.xlabel("Date")
    plt.tight_layout()
    if save:
        out_path = FIG_DIR_PART1 / "daily_load.png"
        plt.savefig(out_path)
        print(f"Saved {out_path}")
    plt.close()


def plot_weekly_series(weekly: pd.DataFrame, save: bool = True) -> None:
    FIG_DIR_PART1.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    weekly["load_gw"].plot()
    plt.title("Weekly German Electricity Load (GW)")
    plt.ylabel("Load (GW)")
    plt.xlabel("Date")
    plt.tight_layout()
    if save:
        out_path = FIG_DIR_PART1 / "weekly_load.png"
        plt.savefig(out_path)
        print(f"Saved {out_path}")
    plt.close()