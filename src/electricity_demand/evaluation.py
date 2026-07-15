import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

FIG_DIR_PART1 = pathlib.Path("outputs") / "figures" / "part1"
METRIC_DIR_PART1 = pathlib.Path("outputs") / "metrics" / "part1"


def adf_test(series: pd.Series, name: str = "load_gw") -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller test and save results to CSV.
    """
    METRIC_DIR_PART1.mkdir(parents=True, exist_ok=True)
    result = adfuller(series.dropna(), autolag="AIC")  # standard ADF setup [web:107][web:104].
    out = pd.DataFrame(
        {
            "statistic": [result[0]],
            "p_value": [result[1]],
            "used_lags": [result[2]],
            "n_obs": [result[3]],
        },
        index=[name],
    )
    out_path = METRIC_DIR_PART1 / f"adf_{name}.csv"
    out.to_csv(out_path)
    print(f"Saved ADF results to {out_path}")
    return out


def plot_acf_pacf(series: pd.Series, name: str = "load_gw") -> None:
    """
    Plot ACF and PACF for the series and save to part1 figures.
    """
    FIG_DIR_PART1.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(series.dropna(), ax=axes[0], lags=50)
    plot_pacf(series.dropna(), ax=axes[1], lags=50)
    axes[0].set_title(f"ACF of {name}")
    axes[1].set_title(f"PACF of {name}")
    plt.tight_layout()
    out_path = FIG_DIR_PART1 / f"acf_pacf_{name}.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()


def plot_residual_acf_pacf(residuals: pd.Series, name: str, fig_dir: pathlib.Path) -> None:
    """
    Plot ACF and PACF of model residuals and save to given folder.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(residuals.dropna(), ax=axes[0], lags=50)
    plot_pacf(residuals.dropna(), ax=axes[1], lags=50)
    axes[0].set_title(f"ACF of residuals: {name}")
    axes[1].set_title(f"PACF of residuals: {name}")
    plt.tight_layout()
    out_path = fig_dir / f"residual_acf_pacf_{name}.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()