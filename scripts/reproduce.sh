#!/usr/bin/env bash
# Full pipeline behind the paper. Run from the repository root. Each step skips files that already exist.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

# 1. Data (Beijing, London, EPA Los Angeles 2023-2024; --holdout adds EPA 2025)
python -m no2shadow.data --holdout

# 2. Forecaster: three datasets x three seeds
python -m no2shadow.train

# 3. Base and shadow forecasts
#    main schedules (31751-31755) for seeds 2025-2027, level-dependent schedules for seed 2025
python -m no2shadow.forecast
#    independent schedules (31756-31760) for the reporting seeds
AQ_SHADOW_OUT=results/forecasts_fresh AQ_SHADOW_SEEDS=2026,2027 AQ_SHADOW_MNAR_SEEDS= \
AQ_SHADOW_MCAR_SEEDS=31756,31757,31758,31759,31760 python -m no2shadow.forecast
#    outage mechanisms: selection on the past 24 h level and shared-cause outages
AQ_SHADOW_OUT=results/forecasts_scenarios AQ_SHADOW_SEEDS=2025,2026 AQ_SHADOW_MNAR_SEEDS= \
AQ_SHADOW_MCAR_SEEDS=51751,51752,51753 AQ_SHADOW_SCEN_SEEDS=51751,51752,51753 \
AQ_SHADOW_SCENARIOS=pre:1,pre:2,corr:0.6,corr:0.9 python -m no2shadow.forecast
#    Los Angeles 2025 held-out year (lock date 2025-01-01)
AQ_SHADOW_OUT=results/forecasts_holdout AQ_SHADOW_DATASETS=epa_la AQ_SHADOW_HOLDOUT=2025-01-01 \
AQ_SHADOW_SEEDS=2026,2027 AQ_SHADOW_MNAR_SEEDS= AQ_SHADOW_MCAR_SEEDS=31756,31757,31758,31759,31760 python -m no2shadow.forecast
#    forecasters without training
AQ_SHADOW_MCAR_SEEDS=31756,31757,31758 python -m no2shadow.simple_forecasters

# 4. Calibration with the four label sources
python -m no2shadow.calibration
AQ_EVAL_SRC=results/forecasts_fresh AQ_EVAL_OUT=results/calibration_fresh AQ_EVAL_TAG=fresh python -m no2shadow.calibration
AQ_EVAL_SRC=results/forecasts_scenarios AQ_EVAL_OUT=results/calibration_scenarios AQ_EVAL_TAG=scenarios \
AQ_EVAL_PERIODS=evaluation python -m no2shadow.calibration
AQ_EVAL_SRC=results/forecasts_holdout AQ_EVAL_OUT=results/calibration_holdout AQ_EVAL_TAG=holdout \
AQ_EVAL_PERIODS=holdout python -m no2shadow.calibration
for kind in persistence seasonal; do
  AQ_EVAL_SRC=results/forecasts_$kind AQ_EVAL_OUT=results/calibration_$kind AQ_EVAL_TAG=$kind python -m no2shadow.calibration
done

# 5. Online conformal updaters (requires the optional dependencies, see README)
python -m no2shadow.updaters
AQ_UPD_SRC=results/forecasts_holdout AQ_UPD_OUT=results/updaters_holdout AQ_UPD_PERIOD=holdout python -m no2shadow.updaters

# 6. Additional analyses
python -m no2shadow.scenarios
python -m no2shadow.audit
python -m no2shadow.review
