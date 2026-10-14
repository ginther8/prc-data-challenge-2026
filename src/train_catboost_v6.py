from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error


DATA_DIR = Path("data/raw")

V2_BENCHMARK_RMSE = 422.62
V5_BENCHMARK_RMSE = 422.39

# One V5 categorical feature per experiment, added on top of V2.
EXPERIMENTS = [
    ("A", "WK_TBL_CAT_flt"),
    ("B", "FLIGHT_TYPE_flt"),
    ("C", "AIRCRAFT_OPERATOR_flt"),
    ("D", "MARKET_SEGMENT_flt"),
]


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
# BASE FEATURES (IDENTICAL TO V2)
# ============================================================

V2_FEATURES = [
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

V2_CATEGORICAL_FEATURES = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt",
]

ABLATION_COLUMNS = [feature for _, feature in EXPERIMENTS]

# Identical to V2 handling, applied to the V2 categoricals and to the
# ablation columns so every experiment sees the same preprocessing.
ALL_CATEGORICAL_FEATURES = [
    *V2_CATEGORICAL_FEATURES,
    *ABLATION_COLUMNS,
]

for column in ALL_CATEGORICAL_FEATURES:
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

# These should be identical to V1 and V2
assert len(train) == 1_757_038, (
    f"Unexpected training count: {len(train):,}"
)
assert len(validation) == 328_009, (
    f"Unexpected validation count: {len(validation):,}"
)

y_train = train["TAXITIME_SEC_mvt"]
y_validation = validation["TAXITIME_SEC_mvt"]


# ============================================================
# SAFETY CHECKS
# ============================================================

# Guard rails: the target and post-departure fields must never be
# used as inputs, and every experiment must add exactly one column.
FORBIDDEN = {
    "TAXITIME_SEC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "ARVT_1_flt",
    "AOBT_3_flt",
    "ARVT_3_flt",
    "MVT_TIME_UTC_mvt",
}

for _, added in EXPERIMENTS:
    features = [*V2_FEATURES, added]
    categoricals = [*V2_CATEGORICAL_FEATURES, added]

    assert not (FORBIDDEN & set(features)), (
        f"Forbidden column in features: {added}"
    )
    assert features[:len(V2_FEATURES)] == V2_FEATURES, (
        f"V2 features not preserved: {added}"
    )
    assert len(features) == len(V2_FEATURES) + 1, (
        f"Experiment must add exactly one feature: {added}"
    )
    assert set(categoricals) <= set(features), (
        f"Categoricals must be a subset of features: {added}"
    )
    assert added in ABLATION_COLUMNS, (
        f"Unknown ablation column: {added}"
    )

print("\nAll safety checks passed. Starting 4 experiments.")


# ============================================================
# EXPERIMENTS
# ============================================================

results = []

for label, added in EXPERIMENTS:
    print("\n" + "=" * 60)
    print(f"EXPERIMENT {label}: V2 + {added}")
    print("=" * 60)

    features = [*V2_FEATURES, added]
    categoricals = [*V2_CATEGORICAL_FEATURES, added]

    X_train = train[features]
    X_validation = validation[features]

    # Unchanged from V2 so only the added feature differs
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

    model.fit(
        X_train,
        y_train,
        cat_features=categoricals,
        eval_set=(X_validation, y_validation),
    )

    predictions = model.predict(X_validation)

    rmse = root_mean_squared_error(
        y_validation,
        predictions,
    )

    best_iteration = model.get_best_iteration()

    results.append({
        "experiment": label,
        "added_feature": added,
        "rmse": rmse,
        "best_iteration": best_iteration,
        "vs_v2": V2_BENCHMARK_RMSE - rmse,
        "vs_v5": V5_BENCHMARK_RMSE - rmse,
    })

    print(f"\nExperiment {label} RMSE: {rmse:.2f} sec")
    print(f"Best iteration: {best_iteration}")

    if rmse < V2_BENCHMARK_RMSE:
        print(
            f"Improvement over v2: "
            f"{V2_BENCHMARK_RMSE - rmse:.2f} sec"
        )
    else:
        print(
            f"Regression vs v2: "
            f"{rmse - V2_BENCHMARK_RMSE:.2f} sec"
        )

    if rmse < V5_BENCHMARK_RMSE:
        print(
            f"Improvement over v5: "
            f"{V5_BENCHMARK_RMSE - rmse:.2f} sec"
        )
    else:
        print(
            f"Regression vs v5: "
            f"{rmse - V5_BENCHMARK_RMSE:.2f} sec"
        )

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.get_feature_importance(),
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print(f"\nExperiment {label} feature importance:")
    print(importance.to_string(index=False))

    del model


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

print("\n" + "=" * 60)
print("V6 ABLATION SUMMARY (sorted by RMSE, lowest first)")
print("=" * 60)
print(
    f"{'Experiment':<34}"
    f"{'RMSE':>9}"
    f"{'Best Iter':>12}"
    f"{'vs V2':>9}"
    f"{'vs V5':>9}"
)
print("-" * 73)

for row in summary.itertuples():
    print(
        f"{'V2 + ' + row.added_feature:<34}"
        f"{row.rmse:>9.2f}"
        f"{row.best_iteration:>12d}"
        f"{row.vs_v2:>+9.2f}"
        f"{row.vs_v5:>+9.2f}"
    )

print("-" * 73)
print(f"{'V2 benchmark':<34}{V2_BENCHMARK_RMSE:>9.2f}")
print(f"{'V5 benchmark':<34}{V5_BENCHMARK_RMSE:>9.2f}")
