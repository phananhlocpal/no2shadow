# Shadow labels for NO<sub>2</sub> prediction intervals under sensor outages

Code for the paper *Calibrating NO<sub>2</sub> prediction intervals for stations under sensor outage with shadow labels when retrospective labels are unavailable* (Loc Phan, Nguyen Tat Thanh University; manuscript under review).

When a monitoring station stops reporting, its prediction interval must be calibrated with residuals borrowed from elsewhere: the station's own labels are missing during the outage, and every other label becomes usable only after a 72 h validation delay. Residuals of online stations come from the right period but from forecasts with full context; archived residuals of past outages have the right information state but come from an earlier period. The **shadow label source** builds residuals that match both: at every forecast origin it hides the recent history of stations that are still online, forecasts them again with the trained model, and keeps the residuals once their labels have matured.

This repository contains the data preparation, the forecaster, the outage simulator, the shadow forecasts, the calibration with four label sources, the online conformal updaters, and the additional analyses reported in the paper.

## Repository layout

```
no2shadow/
  data.py                download and prepare the three NO2 panels; chronological splits
  model.py               multi-station quantile forecaster (GRU encoders + missingness-aware attention)
  train.py               training with pinball loss and early stopping
  outages.py             simulated outage schedules (random, level-dependent, past-level, shared-cause)
  forecast.py            base forecasts and causal shadow forecasts under each outage schedule
  simple_forecasters.py  persistence and seasonal forecasters (no training)
  calibration.py         interval calibration with archival, online, shadow and oracle residuals
  updaters.py            SAOCP, FACI and NExConformal fed with online or shadow labels
  audit.py               residual distributions, sample-size checks, horizon and bootstrap sensitivity
  review.py              level-conditional coverage, stronger controls, TV sensitivity, cost, case study
  scenarios.py           summary of the outage-mechanism runs
configs/                 training hyperparameters and data sources (URLs and SHA-256)
scripts/reproduce.sh     the full pipeline, step by step
tests/                   unit tests on synthetic data
```

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/phananhlocpal/no2.git
cd no2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The online conformal updaters (`no2shadow.updaters`) use the authors' implementation of SAOCP, FACI and NExConformal at a fixed commit. Install it only if you need that step:

```bash
pip install -r requirements-updaters.txt
```

## Data

All data are public. `python -m no2shadow.data` downloads them into `data/raw/` and writes hourly panels to `data/processed/`; missing values stay missing.

| Network | Stations | Unit | Source |
|---|---|---|---|
| Beijing | 12 | µg/m³ | Beijing Multi-Site Air Quality, UCI Machine Learning Repository |
| London | 17 | µg/m³ | KDD Cup 2018, Monash Time Series Forecasting Repository (Zenodo) |
| Los Angeles | 13 | ppb | EPA AQS pre-generated hourly NO<sub>2</sub> files, Los Angeles County |

Los Angeles monitors are selected from 2023 only (at least 80% of hours observed, lowest POC per site) and kept fixed for 2024 and 2025. URLs and checksums are in `configs/data_sources.json`. The 2025 year is the held-out set; prepare it with `--holdout`.

Splits are chronological. Beijing and London: 40% training, 10% validation, then calibration history, tuning and the last 20% as the evaluation period. Los Angeles: training on 2023, validation on January to June 2024, evaluation on October to December 2024, and 2025 as held-out data.

## Reproducing the paper

`scripts/reproduce.sh` runs every step in order. The steps are:

| Step | Command | Output |
|---|---|---|
| Data | `python -m no2shadow.data --holdout` | `data/processed/` |
| Forecaster | `python -m no2shadow.train` | `models/<dataset>/seed_<seed>/checkpoint.pt` |
| Shadow forecasts | `python -m no2shadow.forecast` (and the variants in the script) | `results/forecasts*/` |
| Calibration | `python -m no2shadow.calibration` | `results/calibration*/` |
| Online updaters | `python -m no2shadow.updaters` | `results/updaters*/` |
| Additional analyses | `python -m no2shadow.audit`, `review`, `scenarios` | `results/audit/`, `results/review/`, `results/calibration_scenarios/` |

Each module documents its options (environment variables) at the top of the file. Steps that write a file skip it if it already exists, so an interrupted run can be resumed.

Main settings used in the paper: forecast horizon 1 to 6 h, four forecast origins per day, nominal coverage 80%, label delay 72 h, rolling window 30 days, outages of 72 h starting with daily probability 0.10 per station, sixteen hiding lengths (6 to 96 h), model seeds 2025 (used while building the method) and 2026, 2027 (used for reporting).

Training and forecasting use a GPU when one is available (`AQ_DEVICE=cpu` forces the CPU); calibration and the analyses run on the CPU.

## Tests

```bash
python -m pytest tests
```

The tests use synthetic panels and check, among other things, that no information after the forecast origin is used, that labels enter calibration only after the 72 h delay, and that held-out data do not leak into earlier periods. `tests/test_updaters.py` needs the optional updater dependencies.

## Citation

If you use this code, please cite the paper (the reference will be updated after publication):

```bibtex
@article{phan_shadow_no2,
  author  = {Phan, Loc},
  title   = {Calibrating {NO2} prediction intervals for stations under sensor outage with shadow labels when retrospective labels are unavailable},
  journal = {Manuscript under review},
  year    = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`). The datasets keep the licenses of their providers.

## Contact

Loc Phan, Nguyen Tat Thanh University, Ho Chi Minh City, Vietnam (2500022622@nttu.edu.vn).
