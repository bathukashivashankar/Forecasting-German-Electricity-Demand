import pathlib
import pandas as pd

DATA_DIR = pathlib.Path("data")


def load_opsd_raw(path: pathlib.Path | None = None) -> pd.DataFrame:
    if path is None:
        path = DATA_DIR / "raw" / "time_series_60min_singleindex.csv"
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    return df


def filter_germany_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only German load from 2015-01-01 onwards.
    """
    # Use the observed German load series from OPSD
    load_col = "DE_load_actual_entsoe_transparency"

    out = (
        df[["utc_timestamp", load_col]]
        .rename(columns={load_col: "load_mw"})
        .set_index("utc_timestamp")
        .loc["2015-01-01":]
        .dropna()
    )
    return out


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.resample("D").mean()
    daily["load_gw"] = daily["load_mw"] / 1000.0
    return daily[["load_gw"]]


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    weekly = df.resample("W").mean()
    weekly["load_gw"] = weekly["load_mw"] / 1000.0
    return weekly[["load_gw"]]

def load_berlin_temperature_daily(path: pathlib.Path | None = None) -> pd.DataFrame:
    """
    Load daily Berlin temperature from Open-Meteo CSV.
    """
    if path is None:
        path = DATA_DIR / "raw" / "berlin_daily_temperature.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    # Make timezone-aware in UTC to match OPSD load index
    df.index = df.index.tz_localize("UTC")
    return df


def resample_temperature_to_weekly(temp_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily Berlin temperature to weekly features.
    """
    weekly = temp_daily.resample("W").agg(
        {
            "temp_mean": "mean",
            "temp_min": "min",
            "temp_max": "max",
        }
    )
    return weekly