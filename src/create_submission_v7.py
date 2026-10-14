from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


DATA_DIR = Path("data/raw")
SUBMISSIONS_DIR = Path("submissions")

TEAM_NAME = "vigorous-glove"
SUBMISSION_FILENAME = f"{TEAM_NAME}_v1.parquet"

# V7 best model used 207 as its best iteration (0-indexed), which means
# 208 trees. Retraining on all 2025 rows has no eval_set, so early
# stopping and use_best_model are deliberately NOT used here.
N_TREES = 208


# ============================================================
# FEATURE DEFINITIONS (MUST MATCH train_catboost_v7.py EXACTLY)
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

TARGET = "TAXITIME_SEC_mvt"


# ============================================================
# TRAFFIC FEATURE
# ============================================================

def add_traffic_feature(frame):
    """Count ALL movements at the same physical airport in the
    preceding 15 minutes, excluding the current movement.

    Identical methodology to V2 / V7:
      DEP -> ADEP_mvt, ARR -> ADES_mvt
      rolling("15min", closed="left") per airport
      assigned positionally, never merged on airport + timestamp

    Uses only MVT_TIME_UTC_mvt. Never touches TAXITIME_SEC_mvt or
    BLOCK_TIME_UTC_mvt.
    """
    rows_before = len(frame)

    frame["movement_airport"] = frame["ADEP_mvt"]

    frame.loc[
        frame["PHASE_mvt"] == "ARR",
        "movement_airport"
    ] = frame.loc[
        frame["PHASE_mvt"] == "ARR",
        "ADES_mvt"
    ]

    frame = frame.sort_values(
        ["movement_airport", "MVT_TIME_UTC_mvt"]
    ).copy()

    traffic_15min = (
        frame
        .set_index("MVT_TIME_UTC_mvt")
        .groupby("movement_airport")["MVT_ID_mvt"]
        .rolling("15min", closed="left")
        .count()
        .reset_index(name="traffic_last_15min")
    )

    frame["traffic_last_15min"] = (
        traffic_15min["traffic_last_15min"].to_numpy()
    )

    frame["traffic_last_15min"] = (
        frame["traffic_last_15min"]
        .fillna(0)
    )

    # One row per original movement must be preserved.
    assert len(frame) == rows_before, (
        f"Traffic feature changed row count: "
        f"{rows_before} -> {len(frame)}"
    )

    return frame


def add_time_features(frame):
    frame["hour"] = frame["MVT_TIME_UTC_mvt"].dt.hour
    frame["day_of_week"] = frame["MVT_TIME_UTC_mvt"].dt.dayofweek
    frame["month"] = frame["MVT_TIME_UTC_mvt"].dt.month

    return frame


def apply_categorical_handling(frame):
    """Same handling as V7: fill with UNKNOWN, then cast to string."""
    for column in CATEGORICAL_FEATURES:
        frame[column] = (
            frame[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    return frame


# ============================================================
# LOAD SUBMISSION TEMPLATE FIRST
# ============================================================

print("Loading submission template...")

template = pd.read_parquet(DATA_DIR / "submitting.parquet")

print(f"Template rows: {len(template):,}")
print(f"Template columns: {list(template.columns)}")

TEMPLATE_COLUMNS = list(template.columns)
TEMPLATE_ID_COLUMN = "MVT_ID_mvt"

assert TEMPLATE_ID_COLUMN in TEMPLATE_COLUMNS, (
    f"Template has no {TEMPLATE_ID_COLUMN} column"
)
assert TARGET in TEMPLATE_COLUMNS, (
    f"Template has no {TARGET} column: {TEMPLATE_COLUMNS}"
)

# Refuse to overwrite an existing submission.
destination = SUBMISSIONS_DIR / SUBMISSION_FILENAME

if destination.exists():
    raise SystemExit(
        f"STOP: {destination} already exists. "
        f"Not overwriting. Choose a new version manually."
    )


# ============================================================
# LOAD 2025 TRAINING DATA
# ============================================================

print("Loading 2025 training data...")

training_files = sorted(DATA_DIR.glob("training_*.parquet"))

print(f"Training files found: {len(training_files)}")

dfs = []

for file in training_files:
    df = pd.read_parquet(file)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print(f"Total 2025 movements: {len(data):,}")


# ============================================================
# TRAINING TRAFFIC FEATURE
# ============================================================

print("Creating 15-minute traffic feature on 2025 data...")

data = add_traffic_feature(data)

# Taxi-out prediction = departures only, target must be known.
train = data[
    (data["PHASE_mvt"] == "DEP")
    & data[TARGET].notna()
].copy()

train = add_time_features(train)
train = apply_categorical_handling(train)

print(f"2025 training departure rows: {len(train):,}")

assert len(train) == 2_085_047, (
    f"Unexpected 2025 training departure count: {len(train):,}"
)


# ============================================================
# LOAD AND PREPARE RANKING DATA
# ============================================================

print("\nLoading ranking data...")

ranking = pd.read_parquet(DATA_DIR / "ranking.parquet")

print(f"Ranking rows: {len(ranking):,}")

assert len(ranking) == 689_534, (
    f"Unexpected ranking row count: {len(ranking):,}"
)

# The target must NOT be available for ranking departures.
ranking_dep_target = ranking.loc[
    ranking["PHASE_mvt"] == "DEP", TARGET
].notna().sum()

assert ranking_dep_target == 0, (
    f"Leakage: {ranking_dep_target} ranking departures already have a target"
)

print("Creating 15-minute traffic feature on ranking movements...")
print("(computed from ranking movements only, no 2025 rows mixed in)")

ranking = add_traffic_feature(ranking)

ranking_dep = ranking[ranking["PHASE_mvt"] == "DEP"].copy()

ranking_dep = add_time_features(ranking_dep)
ranking_dep = apply_categorical_handling(ranking_dep)

print(f"Ranking departure rows: {len(ranking_dep):,}")


# ============================================================
# MODEL
# ============================================================

# V7 configuration, with the iteration count fixed to the V7 best
# iteration and no early stopping / use_best_model.
model = CatBoostRegressor(
    iterations=N_TREES,
    depth=8,
    learning_rate=0.1,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
)

print(f"\nTraining CatBoost on all 2025 departures ({N_TREES} trees)...")

model.fit(
    train[FEATURES],
    train[TARGET],
    cat_features=CATEGORICAL_FEATURES,
)

print(f"Built trees: {model.tree_count_}")


# ============================================================
# PREDICT
# ============================================================

print("\nPredicting ranking departures...")

predictions = model.predict(ranking_dep[FEATURES])

print(f"Generated predictions: {len(predictions):,}")


# ============================================================
# BUILD SUBMISSION USING submitting.parquet AS THE TEMPLATE
# ============================================================

# Index predictions by movement id, then reindex to the template order
# so the submission follows submitting.parquet exactly.
prediction_lookup = pd.Series(
    predictions,
    index=ranking_dep["MVT_ID_mvt"].to_numpy(),
)

template_ids = template[TEMPLATE_ID_COLUMN].to_numpy()

submission = pd.DataFrame({
    TEMPLATE_ID_COLUMN: template_ids,
})

submission[TARGET] = prediction_lookup.reindex(template_ids).to_numpy()

# Column order must match the template exactly.
submission = submission[TEMPLATE_COLUMNS]


# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)

failures = []


def check(label, value, expected=None, ok=None):
    if ok is None:
        ok = value == expected

    status = "PASS" if ok else "FAIL"
    suffix = ""
    if expected is not None:
        suffix = f" (expected {expected:,})" if isinstance(expected, int) else f" (expected {expected})"

    print(f"[{status}] {label}: {value:,}{suffix}" if isinstance(value, int)
          else f"[{status}] {label}: {value}{suffix}")

    if not ok:
        failures.append(label)


check("2025 training departure rows", len(train), 2_085_047)
check("ranking rows", len(ranking), 689_534)
check("ranking departure rows", len(ranking_dep), 344_841)
check("submission template rows", len(template), 344_841)
check("generated predictions", len(predictions), 344_841)
check("final submission rows", len(submission), 344_841)

check(
    "duplicate MVT_ID_mvt in submission",
    int(submission[TEMPLATE_ID_COLUMN].duplicated().sum()),
    0,
)
check(
    "missing prediction count",
    int(submission[TARGET].isna().sum()),
    0,
)
check(
    "infinite prediction count",
    int(np.isinf(submission[TARGET].to_numpy(dtype=float)).sum()),
    0,
)

# Every template id must have exactly one prediction, no extras, none missing.
check(
    "template ids == submission ids (order and set)",
    bool((submission[TEMPLATE_ID_COLUMN].to_numpy() == template_ids).all()),
    True,
)
check(
    "no extra ids in submission",
    int(len(set(submission[TEMPLATE_ID_COLUMN]) - set(template_ids))),
    0,
)
check(
    "no missing ids in submission",
    int(len(set(template_ids) - set(submission[TEMPLATE_ID_COLUMN]))),
    0,
)
check(
    "output columns match template",
    list(submission.columns) == TEMPLATE_COLUMNS,
    True,
)

print()
print("Prediction distribution (seconds):")
print(
    submission[TARGET]
    .describe()
    .round(2)
    .to_string()
)

print()
print(f"  min    = {submission[TARGET].min():.2f}")
print(f"  max    = {submission[TARGET].max():.2f}")
print(f"  mean   = {submission[TARGET].mean():.2f}")
print(f"  median = {submission[TARGET].median():.2f}")
print(f"  std    = {submission[TARGET].std():.2f}")

print()
print(f"2025 train target mean   = {train[TARGET].mean():.2f}")
print(f"2025 train target median = {train[TARGET].median():.2f}")
print(f"2025 train target std    = {train[TARGET].std():.2f}")


# ============================================================
# SAVE
# ============================================================

if failures:
    raise SystemExit(
        f"STOP: {len(failures)} validation check(s) failed: {failures}. "
        f"Submission file NOT written."
    )

print("\nAll validation checks passed.")

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

if destination.exists():
    raise SystemExit(f"STOP: {destination} appeared during run. Not overwriting.")

submission.to_parquet(destination, index=False)

print(f"Saved submission: {destination}")
print(f"Rows: {len(submission):,}")
print(f"Columns: {list(submission.columns)}")
