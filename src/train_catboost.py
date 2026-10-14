from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error


DATA_DIR = Path("data/raw")

print("Loading training data...")

training_files = sorted(DATA_DIR.glob("training_*.parquet"))

dfs = []

for file in training_files:
    df = pd.read_parquet(file)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Taxi-out = departures only
departures = data[data["PHASE_mvt"] == "DEP"].copy()

print(f"Departure rows: {len(departures):,}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

departures["hour"] = departures["MVT_TIME_UTC_mvt"].dt.hour
departures["day_of_week"] = departures["MVT_TIME_UTC_mvt"].dt.dayofweek
departures["month"] = departures["MVT_TIME_UTC_mvt"].dt.month

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
    departures[column] = departures[column].fillna("UNKNOWN").astype(str)


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

predictions = model.predict(X_validation)

rmse = root_mean_squared_error(
    y_validation,
    predictions,
)

print(f"\nCatBoost v1 RMSE: {rmse:.2f} sec")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.get_feature_importance(),
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance:")
print(importance.to_string(index=False))