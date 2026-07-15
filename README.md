# Forecasting German Electricity Demand

This repository contains a reproducible time-series forecasting project for modelling and forecasting **German electricity demand** using benchmark, classical statistical, feature-based machine-learning, and neural-network approaches.

The project starts from raw German electricity load data and temperature data, prepares weekly and daily forecasting datasets, and compares multiple models on a common holdout period. The final workflow includes benchmark forecasts, SARIMA, SARIMAX with temperature, a weekly Random Forest model, and a daily LSTM model.

## Project aim

The aim of this project is to forecast German electricity demand and compare the accuracy, interpretability, and practical usefulness of different forecasting approaches.

The main research questions are:

1. How well do simple benchmark methods forecast German electricity demand?
2. Does a SARIMA model improve on the benchmark forecasts?
3. Does SARIMAX with temperature improve on SARIMA?
4. Do engineered lagged and calendar features improve forecast accuracy in a feature-based model?
5. Does an LSTM model justify its additional complexity?
6. Which model is most appropriate for operational forecasting?

## Data

The project uses raw German electricity load data together with daily Berlin temperature data.

Main raw files:

```text
data/raw/time_series_60min_singleindex.csv
data/raw/berlin_daily_temperature.csv
```

The electricity load data are transformed into daily and weekly demand series inside the project workflow. Weekly data are used for benchmark, SARIMA, SARIMAX, and Random Forest modelling. Daily data are used for the LSTM model.

## Repository structure

```text
electricity-demand-forecasting-shiva/
│
├── data/
│   ├── raw/
│   │   ├── berlin_daily_temperature.csv
│   │   └── time_series_60min_singleindex.csv
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   └── part6_lstm_daily_colab.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── model_comparison_rmse.png
│   │   ├── part1/
│   │   │   ├── acf_pacf_weekly_load_gw.png
│   │   │   ├── acf_pacf_weekly_load_gw_diff.png
│   │   │   ├── daily_load.png
│   │   │   └── weekly_load.png
│   │   ├── part2/
│   │   │   └── benchmarks_weekly_2y.png
│   │   ├── part3/
│   │   │   ├── residual_acf_pacf_sarima_weekly.png
│   │   │   └── sarima_weekly_2y_ci.png
│   │   ├── part4/
│   │   │   ├── residual_acf_pacf_sarimax_temp_weekly.png
│   │   │   └── sarimax_temp_weekly_2y_ci.png
│   │   ├── part5/
│   │   │   └── rf_weekly_2y.png
│   │   └── part6/
│   │       └── lstm_daily_2y.png
│   ├── forecasts/
│   │   ├── part2/
│   │   │   └── benchmarks_weekly.csv
│   │   ├── part3/
│   │   │   └── sarima_weekly.csv
│   │   ├── part4/
│   │   │   └── sarimax_temp_weekly.csv
│   │   ├── part5/
│   │   │   └── rf_weekly.csv
│   │   └── part6/
│   │       └── lstm_daily.csv
│   └── metrics/
│       ├── model_comparison.csv
│       ├── part1/
│       │   ├── adf_weekly_load_gw.csv
│       │   └── adf_weekly_load_gw_diff.csv
│       ├── part2/
│       │   └── benchmarks_metrics.csv
│       ├── part3/
│       │   ├── sarima_best_model.csv
│       │   └── sarima_metrics.csv
│       ├── part4/
│       │   └── sarimax_temp_metrics.csv
│       ├── part5/
│       │   └── rf_weekly_metrics.csv
│       └── part6/
│           └── lstm_daily_metrics.csv
│
├── reports/
│   └── 24071919_Report.pdf
│
├── scripts/
│   ├── debug_columns.py
│   ├── download_data.py
│   ├── download_temperature.py
│   ├── make_model_comparison.py
│   ├── preview_lstm_data.py
│   ├── run_model_comparison_plot.py
│   ├── run_part1.py
│   ├── run_part2.py
│   ├── run_part3.py
│   ├── run_part4_sarimax_temp.py
│   ├── run_part5_tree_weekly.py
│   ├── run_part6_lstm_daily.py
│   ├── run_part6_plot_lstm.py
│   └── run_pipeline.py
│
├── src/
│   └── electricity_demand/
│       ├── data.py
│       ├── evaluation.py
│       ├── pipeline.py
│       ├── plotting.py
│       └── models/
│           ├── benchmarks.py
│           ├── lstm_data.py
│           ├── lstm_model.py
│           ├── sarimax.py
│           └── tree.py
│
├── environment.yml
├── README.md
└── requirements.txt
```

## Pipeline overview

The main workflow is coordinated through:

```text
scripts/run_pipeline.py
```

This script runs the project parts in sequence and regenerates the main forecasting outputs and figures.

The pipeline currently performs the following stages:

1. Part 1: exploratory analysis and stationarity checks.
2. Part 2: benchmark forecasting.
3. Part 3: SARIMA grid search, fitting, diagnostics, and forecast generation.
4. Part 4: SARIMAX with temperature covariates.
5. Part 5: weekly Random Forest forecasting.
6. Part 6: LSTM forecast plotting from exported Colab results.
7. Final model comparison table and RMSE comparison plot.

## Main scripts

### Data download

```bash
python scripts/download_data.py
python scripts/download_temperature.py
```

### End-to-end pipeline

```bash
python scripts/run_pipeline.py
```

### Individual project parts

```bash
python scripts/run_part1.py
python scripts/run_part2.py
python scripts/run_part3.py
python scripts/run_part4_sarimax_temp.py
python scripts/run_part5_tree_weekly.py
python scripts/run_part6_plot_lstm.py
```

### Comparison outputs

```bash
python scripts/make_model_comparison.py
python scripts/run_model_comparison_plot.py
```

## Models

The project compares several forecasting approaches.

### Benchmark models

The benchmark methods include:

```text
Mean forecast
Naive forecast
Seasonal naive forecast
Drift forecast
```

These provide simple reference points for judging whether more advanced models are actually worthwhile.

### SARIMA model

The statistical forecasting stage uses a seasonal ARIMA model for weekly data. The selected model is stored through the project outputs in:

```text
outputs/metrics/part3/sarima_best_model.csv
outputs/metrics/part3/sarima_metrics.csv
outputs/forecasts/part3/sarima_weekly.csv
outputs/figures/part3/sarima_weekly_2y_ci.png
```

### SARIMAX with temperature

The SARIMAX stage extends the weekly model by including temperature as an exogenous predictor.

Related outputs include:

```text
outputs/metrics/part4/sarimax_temp_metrics.csv
outputs/forecasts/part4/sarimax_temp_weekly.csv
outputs/figures/part4/sarimax_temp_weekly_2y_ci.png
```

### Feature-based machine-learning model

The project includes a weekly Random Forest model in:

```text
src/electricity_demand/models/tree.py
```

Outputs include:

```text
outputs/metrics/part5/rf_weekly_metrics.csv
outputs/forecasts/part5/rf_weekly.csv
outputs/figures/part5/rf_weekly_2y.png
```

### Neural model

The project also includes an LSTM-based forecasting stage in:

```text
src/electricity_demand/models/lstm_data.py
src/electricity_demand/models/lstm_model.py
scripts/run_part6_lstm_daily.py
```

The final LSTM training and evaluation were completed in Google Colab due to local TensorFlow compatibility issues, and the exported outputs are stored in:

```text
outputs/metrics/part6/lstm_daily_metrics.csv
outputs/forecasts/part6/lstm_daily.csv
outputs/figures/part6/lstm_daily_2y.png
notebooks/part6_lstm_daily_colab.ipynb
```

## Evaluation

All models are evaluated on a holdout test period so that performance comparisons are fair and directly comparable.

The main evaluation metrics used in the project are:

```text
MAE
RMSE
Bias
```

The overall model comparison is saved in:

```text
outputs/metrics/model_comparison.csv
outputs/figures/model_comparison_rmse.png
```

## Figures and outputs

### Part 1: exploratory analysis and stationarity

```text
outputs/figures/part1/daily_load.png
outputs/figures/part1/weekly_load.png
outputs/figures/part1/acf_pacf_weekly_load_gw.png
outputs/figures/part1/acf_pacf_weekly_load_gw_diff.png
outputs/metrics/part1/adf_weekly_load_gw.csv
outputs/metrics/part1/adf_weekly_load_gw_diff.csv
```

### Part 2: benchmark forecasting

```text
outputs/figures/part2/benchmarks_weekly_2y.png
outputs/forecasts/part2/benchmarks_weekly.csv
outputs/metrics/part2/benchmarks_metrics.csv
```

### Part 3: SARIMA diagnostics and forecast intervals

```text
outputs/figures/part3/residual_acf_pacf_sarima_weekly.png
outputs/figures/part3/sarima_weekly_2y_ci.png
outputs/forecasts/part3/sarima_weekly.csv
outputs/metrics/part3/sarima_best_model.csv
outputs/metrics/part3/sarima_metrics.csv
```

### Part 4: SARIMAX with temperature

```text
outputs/figures/part4/residual_acf_pacf_sarimax_temp_weekly.png
outputs/figures/part4/sarimax_temp_weekly_2y_ci.png
outputs/forecasts/part4/sarimax_temp_weekly.csv
outputs/metrics/part4/sarimax_temp_metrics.csv
```

### Part 5: Random Forest forecasting

```text
outputs/figures/part5/rf_weekly_2y.png
outputs/forecasts/part5/rf_weekly.csv
outputs/metrics/part5/rf_weekly_metrics.csv
```

### Part 6: LSTM forecasting

```text
outputs/figures/part6/lstm_daily_2y.png
outputs/forecasts/part6/lstm_daily.csv
outputs/metrics/part6/lstm_daily_metrics.csv
```

### Final comparison

```text
outputs/figures/model_comparison_rmse.png
outputs/metrics/model_comparison.csv
```

## Installation

Create and activate a virtual environment.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Reproducing the analysis

A typical workflow from a fresh clone is:

```bash
python scripts/download_data.py
python scripts/download_temperature.py
python scripts/run_pipeline.py
```

If the LSTM must be retrained, use the Colab notebook:

```text
notebooks/part6_lstm_daily_colab.ipynb
```

and then place the exported forecast and metrics files back into:

```text
outputs/forecasts/part6/
outputs/metrics/part6/
```

## Report

The written report is stored in:

```text
reports/report.md
```

The report discusses model behaviour, forecast accuracy, comparison against the seasonal naive benchmark, and operational model selection.

## Good practice

This project follows standard forecasting good practice:

- Uses a time-based train-test split rather than a random split.
- Keeps reusable logic inside `src/`.
- Saves figures, forecasts, and metrics systematically under `outputs/`.
- Compares advanced models against strong benchmark forecasts.
- Includes both classical statistical and machine-learning approaches.
- Separates raw data, source code, scripts, outputs, notebooks, and report files clearly.

## Expected contents of the submission

The final submission includes:

```text
README.md
requirements.txt
source code in src/
pipeline and utility scripts in scripts/
raw data references in data/
figures, forecasts, and metrics in outputs/
report in reports/
```
