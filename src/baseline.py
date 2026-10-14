from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


DATA_DIR = Path("data/raw")


# ============================================================
# LOAD DATA
# ============================================================

training_files = sorted(DATA_DIR.glob("training_*.parquet"))

print("Loading training data...")

dfs = []

for file in training_files:
    df = pd.read_parquet(file)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# We are predicting taxi-OUT, so use departures
departures = data[data["PHASE_mvt"] == "DEP"].copy()

print(f"Departure rows: {len(departures):,}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train = departures[
    departures["MVT_TIME_UTC_mvt"] < "2025-11-01"
].copy()

validation = departures[
    departures["MVT_TIME_UTC_mvt"] >= "2025-11-01"
].copy()

print(f"Training rows:   {len(train):,}")
print(f"Validation rows: {len(validation):,}")


# ============================================================
# GLOBAL MEDIAN BASELINE
# ============================================================

global_median = train["TAXITIME_SEC_mvt"].median()

print(f"\nTraining median taxi time: {global_median:.0f} sec")

# Predict the same value for every validation flight
validation["prediction"] = global_median

rmse = root_mean_squared_error(
    validation["TAXITIME_SEC_mvt"],
    validation["prediction"],
)

print(f"Global median RMSE: {rmse:.2f} sec")


# ============================================================
# AIRPORT MEDIAN BASELINE
# ============================================================

airport_medians = (
    train
    .groupby("ADEP_mvt")["TAXITIME_SEC_mvt"]
    .median()
)

print("\nAirport medians from training data:")
print(airport_medians.round().astype(int))

validation["airport_prediction"] = (
    validation["ADEP_mvt"]
    .map(airport_medians)
    .fillna(global_median)
)

airport_rmse = root_mean_squared_error(
    validation["TAXITIME_SEC_mvt"],
    validation["airport_prediction"],
)

print(f"\nAirport median RMSE: {airport_rmse:.2f} sec")
print(f"Improvement: {rmse - airport_rmse:.2f} sec")


# ============================================================
# AIRPORT + RUNWAY + STAND MEDIAN BASELINE
# ============================================================

route_medians = (
    train
    .groupby(["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"])
    ["TAXITIME_SEC_mvt"]
    .median()
)

validation["route_prediction"] = validation.set_index(
    ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]
).index.map(route_medians)

# If a particular stand/runway combination was never seen
# during training, fall back to the airport median.
validation["route_prediction"] = validation["route_prediction"].fillna(
    validation["airport_prediction"]
)

route_rmse = root_mean_squared_error(
    validation["TAXITIME_SEC_mvt"],
    validation["route_prediction"],
)

print(f"\nAirport + runway + stand RMSE: {route_rmse:.2f} sec")
print(f"Improvement over airport baseline: {airport_rmse - route_rmse:.2f} sec")
print(f"Improvement over global baseline: {rmse - route_rmse:.2f} sec")