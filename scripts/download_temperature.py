import pathlib
import requests
import pandas as pd

RAW_DIR = pathlib.Path("data") / "raw"

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Berlin coordinates
    latitude = 52.52
    longitude = 13.41

    # Match OPSD period: 2015-01-01 to 2020-10-06
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2015-01-01",
        "end_date": "2020-10-06",
        "daily": ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max"],
        "timezone": "UTC",
    }

    url = "https://archive-api.open-meteo.com/v1/archive"
    print(f"Requesting daily temperature for Berlin from {url} ...")
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    # Convert to DataFrame
    dates = pd.to_datetime(data["daily"]["time"])
    df = pd.DataFrame(
        {
            "date": dates,
            "temp_mean": data["daily"]["temperature_2m_mean"],
            "temp_min": data["daily"]["temperature_2m_min"],
            "temp_max": data["daily"]["temperature_2m_max"],
        }
    ).set_index("date")

    out_path = RAW_DIR / "berlin_daily_temperature.csv"
    df.to_csv(out_path)
    print(f"Saved Berlin daily temperature to {out_path}")


if __name__ == "__main__":
    main()