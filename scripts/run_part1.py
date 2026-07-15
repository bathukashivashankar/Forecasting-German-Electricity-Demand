from src.electricity_demand.data import (
    load_opsd_raw,
    filter_germany_load,
    resample_to_daily,
    resample_to_weekly,
)
from src.electricity_demand.plotting import (
    plot_daily_series,
    plot_weekly_series,
)
from src.electricity_demand.evaluation import (
    adf_test,
    plot_acf_pacf,
)


def main():
    # 1. Load and prepare data
    raw = load_opsd_raw()
    load = filter_germany_load(raw)

    # 2. Daily and weekly resampling
    daily = resample_to_daily(load)
    weekly = resample_to_weekly(load)

    # 3. Initial plots (EDA)
    plot_daily_series(daily)
    plot_weekly_series(weekly)

    # 4. Stationarity tests on weekly series
    adf_test(weekly["load_gw"], name="weekly_load_gw")
    plot_acf_pacf(weekly["load_gw"], name="weekly_load_gw")

    # 5. Stationarity tests on differenced weekly series
    weekly_diff = weekly["load_gw"].diff().dropna()
    adf_test(weekly_diff, name="weekly_load_gw_diff")
    plot_acf_pacf(weekly_diff, name="weekly_load_gw_diff")


if __name__ == "__main__":
    main()