"""Seasonal diagnostic for V7.

Answers one question: does V7 perform materially worse on January/July-like
periods than on the existing Nov-Dec validation?

This is a DIAGNOSTIC, not a production model. No V8, no feature changes,
no hyperparameter changes. V7 features and CatBoost config are reproduced
exactly from train_catboost_v7.py.
"""

from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error


DATA_DIR = Path("data/raw")

TARGET = "TAXITIME_SEC_mvt"

OFFICIAL_LEADERBOARD_RMSE = 572.1223


# ============================================================
# FEATURE DEFINITIONS (EXACTLY V7)
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
    "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt",
]

CATEGORICAL_FEATURES = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt",
    "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt",
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading 2025 training data...")

training_files = sorted(DATA_DIR.glob("training_*.parquet"))

print(f"files found: {len(training_files)}")

dfs = []

for file in training_files:
    df = pd.read_parquet(file)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print(f"Total movements: {len(data):,}")


# ============================================================
# CONGESTION FEATURE (IDENTICAL TO V7)
# ============================================================

print("Creating 15-minute airport traffic feature...")

data["movement_airport"] = data["ADEP_mvt"]

data.loc[
    data["PHASE_mvt"] == "ARR",
    "movement_airport"
] = data.loc[
    data["PHASE_mvt"] == "ARR",
    "ADES_mvt"
]

data = data.sort_values(
    ["movement_airport", "MVT_TIME_UTC_mvt"]
).copy()

traffic_15min = (
    data
    .set_index("MVT_TIME_UTC_mvt")
    .groupby("movement_airport")["MVT_ID_mvt"]
    .rolling("15min", closed="left")
    .count()
    .reset_index(name="traffic_last_15min")
)

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

departures = data[
    data["PHASE_mvt"] == "DEP"
].copy()

print(f"Departure rows: {len(departures):,}")

assert len(departures) == 2_085_047, (
    f"Unexpected departure count: {len(departures):,}"
)


# ============================================================
# FEATURE ENGINEERING (IDENTICAL TO V7)
# ============================================================

departures["hour"] = departures["MVT_TIME_UTC_mvt"].dt.hour
departures["day_of_week"] = departures["MVT_TIME_UTC_mvt"].dt.dayofweek
departures["month"] = departures["MVT_TIME_UTC_mvt"].dt.month

for column in CATEGORICAL_FEATURES:
    departures[column] = (
        departures[column]
        .fillna("UNKNOWN")
        .astype(str)
    )

# helper columns for period selection only, never features
departures["year"] = departures["MVT_TIME_UTC_mvt"].dt.year

assert set(FEATURES).isdisjoint({"year"}), "year must not be a feature"


# ============================================================
# PERIOD DEFINITIONS
# ============================================================

is_2025 = departures["year"] == 2025

TESTS = [
    (
        "TEST A - Existing benchmark (Nov-Dec 2025)",
        departures[is_2025 & (departures["month"] <= 10)],
        departures[is_2025 & (departures["month"] >= 11)],
    ),
    (
        "TEST B - January holdout (January 2025)",
        departures[is_2025 & (departures["month"] != 1)],
        departures[is_2025 & (departures["month"] == 1)],
    ),
    (
        "TEST C - July holdout (July 2025)",
        departures[is_2025 & (departures["month"] != 7)],
        departures[is_2025 & (departures["month"] == 7)],
    ),
]


# ============================================================
# RUN TESTS
# ============================================================

results = []

for label, train, validation in TESTS:

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    # ---- integrity checks before training ----
    assert len(train) > 0, f"{label}: empty training set"
    assert len(validation) > 0, f"{label}: empty validation set"

    train_ids = set(train["MVT_ID_mvt"].to_numpy())
    val_ids = set(validation["MVT_ID_mvt"].to_numpy())
    overlap = train_ids & val_ids

    # No validation target row may appear in training.
    assert len(overlap) == 0, (
        f"{label}: {len(overlap):,} validation IDs leaked into training"
    )

    y_train = train[TARGET]
    y_validation = validation[TARGET]

    print(f"Training rows            : {len(train):,}")
    print(f"Validation rows          : {len(validation):,}")
    print(f"ID overlap train/valid   : {len(overlap):,}")
    print(f"Train months             : {sorted(train['month'].unique())}")
    print(f"Validation months        : {sorted(validation['month'].unique())}")
    print(
        f"Validation target mean   : {y_validation.mean():.2f}  "
        f"std: {y_validation.std():.2f}"
    )
    print(
        f"Train target mean        : {y_train.mean():.2f}  "
        f"std: {y_train.std():.2f}"
    )

    X_train = train[FEATURES]
    X_validation = validation[FEATURES]

    # ---- V7 CatBoost configuration, unchanged ----
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

    print(f"\nTraining {label.split(' - ')[0]}...")

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

    best_iteration = model.get_best_iteration()

    print()
    print(f"RMSE                     : {rmse:.2f} sec")
    print(f"Best iteration           : {best_iteration}")
    print()
    print("--- actual (validation target) ---")
    print(f"  mean   : {y_validation.mean():.2f}")
    print(f"  std    : {y_validation.std():.2f}")
    print(f"  median : {y_validation.median():.2f}")
    print("--- prediction ---")
    print(f"  mean   : {predictions.mean():.2f}")
    print(f"  std    : {predictions.std():.2f}")
    print(f"  median : {float(pd.Series(predictions).median()):.2f}")

    # ---- per-airport RMSE ----
    scored = pd.DataFrame({
        "ADEP_mvt": validation["ADEP_mvt"].to_numpy(),
        "actual": y_validation.to_numpy(),
        "pred": predictions,
    })

    per_airport = (
        scored
        .groupby("ADEP_mvt", observed=True)
        .apply(
            lambda g: pd.Series({
                "n": len(g),
                "actual_mean": g["actual"].mean(),
                "actual_std": g["actual"].std(),
                "pred_mean": g["pred"].mean(),
                "rmse": root_mean_squared_error(g["actual"], g["pred"]),
            }),
            include_groups=False,
        )
        .sort_values("rmse", ascending=False)
    )

    print()
    print(f"--- RMSE by airport ({label.split(' - ')[0]}) ---")
    print(
        f"{'airport':10s} {'n':>9s} {'rmse':>9s} "
        f"{'actual_mean':>12s} {'actual_std':>11s} {'pred_mean':>10s}"
    )
    print("-" * 66)

    for airport, row in per_airport.iterrows():
        print(
            f"{airport:10s} {int(row['n']):>9,} {row['rmse']:>9.2f} "
            f"{row['actual_mean']:>12.2f} {row['actual_std']:>11.2f} "
            f"{row['pred_mean']:>10.2f}"
        )

    results.append({
        "label": label,
        "test": label.split(" - ")[0],
        "n_train": len(train),
        "n_val": len(validation),
        "val_target_mean": y_validation.mean(),
        "val_target_std": y_validation.std(),
        "val_target_median": y_validation.median(),
        "rmse": rmse,
        "best_iteration": best_iteration,
        "pred_mean": predictions.mean(),
        "pred_std": predictions.std(),
        "pred_median": float(pd.Series(predictions).median()),
    })

    per_airport.to_csv(
        Path("reports") / f"per_airport_{label.split(' ')[1].lower()}.csv"
    )

    del model


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary = pd.DataFrame(results)

print()
print(
    f"{'test':8s} {'n_train':>10s} {'n_val':>9s} {'rmse':>9s} "
    f"{'best_it':>8s} {'act_mean':>9s} {'act_std':>8s} "
    f"{'pred_mean':>10s} {'pred_std':>9s}"
)
print("-" * 92)

for row in summary.itertuples():
    print(
        f"{row.test:8s} {row.n_train:>10,} {row.n_val:>9,} {row.rmse:>9.2f} "
        f"{row.best_iteration:>8d} {row.val_target_mean:>9.2f} "
        f"{row.val_target_std:>8.2f} {row.pred_mean:>10.2f} "
        f"{row.pred_std:>9.2f}"
    )

print()
print("-" * 92)
print(f"{'Nov-Dec RMSE (TEST A, existing benchmark)':<44}{results[0]['rmse']:>10.2f}")
print(f"{'January RMSE (TEST B, January holdout)':<44}{results[1]['rmse']:>10.2f}")
print(f"{'July RMSE (TEST C, July holdout)':<44}{results[2]['rmse']:>10.2f}")
print(f"{'Official leaderboard RMSE':<44}{OFFICIAL_LEADERBOARD_RMSE:>10.4f}")
print("-" * 92)

print()
print("Factual differences vs the Nov-Dec benchmark:")
print(
    f"  January - NovDec : {results[1]['rmse'] - results[0]['rmse']:+.2f} sec"
)
print(
    f"  July    - NovDec : {results[2]['rmse'] - results[0]['rmse']:+.2f} sec"
)
print(
    f"  Leaderboard - NovDec : "
    f"{OFFICIAL_LEADERBOARD_RMSE - results[0]['rmse']:+.2f} sec"
)
print(
    f"  Leaderboard - July   : "
    f"{OFFICIAL_LEADERBOARD_RMSE - results[2]['rmse']:+.2f} sec"
)
