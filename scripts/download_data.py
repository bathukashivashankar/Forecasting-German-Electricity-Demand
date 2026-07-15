import pathlib
import requests

OPS_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
RAW_DIR = pathlib.Path("data") / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "time_series_60min_singleindex.csv"

    print(f"Downloading OPSD data from {OPS_URL} ...")
    resp = requests.get(OPS_URL)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"Saved OPSD data to {out_path}")


if __name__ == "__main__":
    main()