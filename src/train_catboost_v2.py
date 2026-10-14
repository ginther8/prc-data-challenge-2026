from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error


DATA_DIR = Path("data/raw")


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

training_files = sorted(DATA_DIR.glob("training_*.parquet"))

dfs = []

for file in training_files:
    df = pd.read_parquet(file)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print(f"Total movements: {len(data):,}")


# ============================================================
# CONGESTION FEATURE
# ============================================================

print("Creating 15-minute airport traffic feature...")

# A departure is physically at ADEP.
# An arrival is physically at ADES.
data["movement_airport"] = data["ADEP_mvt"]

data.loc[
    data["PHASE_mvt"] == "ARR",
    "movement_airport"
] = data.loc[
    data["PHASE_mvt"] == "ARR",
    "ADES_mvt"
]

# Sort movements chronologically within each airport
data = data.sort_values(
    ["movement_airport", "MVT_TIME_UTC_mvt"]
).copy()

# Count movements at the same airport in the previous 15 minutes.
#
# closed="left" means the current movement itself is not counted.
traffic_15min = (
    data
    .set_index("MVT_TIME_UTC_mvt")
    .groupby("movement_airport")["MVT_ID_mvt"]
    .rolling("15min", closed="left")
    .count()
    .reset_index(name="traffic_last_15min")
)

# IMPORTANT:
# Assign directly rather than merging on airport + timestamp,
# because multiple movements can have the same timestamp.
data["traffic_last_15min"] = (
    traffic_15min["traffic_last_15min"].to_numpy()
)

data["traffic_last_15min"] = (
    data["traffic_last_15min"]
    .fillna(0)
)


# ============================================================
# KEEP DEPARTURES
# ============================================================

# Taxi-out prediction = departures only
departures = data[
    data["PHASE_mvt"] == "DEP"
].copy()

print(f"Departure rows: {len(departures):,}")

# Adding a feature must not create or remove flights
assert len(departures) == 2_085_047, (
    f"Unexpected departure count: {len(departures):,}"
)

print("\n15-minute traffic statistics:")
print(
    departures["traffic_last_15min"]
    .describe()
    .round(1)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

departures["hour"] = departures["MVT_TIME_UTC_mvt"].dt.hour
departures["day_of_week"] = departures["MVT_TIME_UTC_mvt"].dt.dayofweek
departures["month"] = departures["MVT_TIME_UTC_mvt"].dt.month


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt",
    "hour",
    "day_of_week",
    "month",
    "traffic_last_15min",
]

CATEGORICAL_FEATURES = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt",
]

for column in CATEGORICAL_FEATURES:
    departures[column] = (
        departures[column]
        .fillna("UNKNOWN")
        .astype(str)
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train = departures[
    departures["MVT_TIME_UTC_mvt"] < "2025-11-01"
].copy()

validation = departures[
    departures["MVT_TIME_UTC_mvt"] >= "2025-11-01"
].copy()

print(f"\nTraining rows:   {len(train):,}")
print(f"Validation rows: {len(validation):,}")

# These should be identical to V1
assert len(train) == 1_757_038
assert len(validation) == 328_009


# ============================================================
# X AND y
# ============================================================

X_train = train[FEATURES]
y_train = train["TAXITIME_SEC_mvt"]

X_validation = validation[FEATURES]
y_validation = validation["TAXITIME_SEC_mvt"]


# ============================================================
# MODEL
# ============================================================

model = CatBoostRegressor(
    iterations=1000,
    depth=8,
    learning_rate=0.1,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    use_best_model=True,
)

print("\nTraining CatBoost...")

model.fit(
    X_train,
    y_train,
    cat_features=CATEGORICAL_FEATURES,
    eval_set=(X_validation, y_validation),
)


# ============================================================
# EVALUATION
# ============================================================

predictions = model.predict(X_validation)

rmse = root_mean_squared_error(
    y_validation,
    predictions,
)

print(f"\nCatBoost v2 RMSE: {rmse:.2f} sec")
print("CatBoost v1 RMSE: 424.85 sec")
print(f"Improvement over v1: {424.85 - rmse:.2f} sec")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.get_feature_importance(),
})

importance = importance.sort_values(
    "importance",
    ascending=False,
)

print("\nFeature importance:")
print(importance.to_string(index=False))