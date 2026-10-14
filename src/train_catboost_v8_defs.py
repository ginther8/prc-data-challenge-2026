"""Single source of truth for the V8 candidate.

Imported by both train_catboost_v8.py and create_submission_v8.py so the
feature list and the feature construction can never drift between the model
that was trained and the submission that is built.

V8 = the exact feature set and feature construction validated in
src/experiment_v7_vs_t4.py (V7's 12 features + the 5 T4 deltas = 17).

  * No clock / time-of-day features. Rejected: the apparent gain was ~95%
    attributable to LIRF extreme rows, and -2.32 (noise) once LIRF was removed.
  * No H1 flight-history features. Rejected: fold-inconsistent (+6.93 JAN,
    -12.21 JUL) and worse (+8.78) once LIRF was removed.
  * No BLOCK_TIME_UTC_mvt and no TAXITIME_SEC_mvt as a feature.

ITERATION COUNT, fixed from Jan/Jul 2025 validation evidence only:
    JAN 2025  best_iter = 365   RMSE 404.8803102
    JUL 2025  best_iter = 266   RMSE 513.7559738
V7 used iterations = best_iter + 1 (208 trees for best_iter 207), so these
correspond to 366 and 267 trees. The midpoint is 316.5, and V8 uses 316 trees.
2026 ranking data was not used to select this value.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "v8_catboost.cbm"
SUBMISSIONS_DIR = Path("submissions")

TEAM_NAME = "vigorous-glove"
SUBMISSION_FILENAME = f"{TEAM_NAME}_v2.parquet"

TARGET = "TAXITIME_SEC_mvt"
MINUEND = "MVT_TIME_UTC_mvt"
FORBIDDEN = ["BLOCK_TIME_UTC_mvt", TARGET]

EXPECTED_RANKING_DEPARTURES = 344_841
EXPECTED_TRAINING_DEPARTURES = 2_085_047

# Validated T4 scores on the 2025 holdout folds.
VALIDATED_T4 = {"JAN": 404.8803102, "JUL": 513.7559738}
VALIDATED_T4_BEST_ITER = {"JAN": 365, "JUL": 266}

# ---- V7 features, verbatim from experiment_v7_vs_t4.py ----
V7_FEATURES = [
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

V7_CATEGORICALS = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt",
    "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt",
]

# (feature name, subtrahend column). MVT_TIME_UTC_mvt is the minuend for all.
T4_SPEC = [
    ("d_AOBT3", "AOBT_3_flt"),
    ("d_EOBT1", "EOBT_1_flt"),
    ("d_IOBT", "IOBT_flt"),
    ("d_LOBT", "LOBT_flt"),
    ("d_SCHED", "SCHED_TIME_UTC_mvt"),
]
T4_FEATURES = [name for name, _ in T4_SPEC]
T4_SOURCE_COLS = [MINUEND] + [col for _, col in T4_SPEC]

FEATURES = V7_FEATURES + T4_FEATURES

# Fixed iteration count (see module docstring).
N_TREES = 316

# CatBoost hyper-parameters, identical to the validated T4 experiment.
CB_PARAMS = dict(
    depth=8,
    learning_rate=0.1,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
)

assert len(FEATURES) == 17, f"expected 17 V8 features, got {len(FEATURES)}"
assert not (set(FEATURES) & set(FORBIDDEN)), "forbidden column in FEATURES"
assert not any(f.startswith("clock_") for f in FEATURES), \
    "clock features must not appear in V8"
assert not any(f.startswith("hist_") for f in FEATURES), \
    "H1 history features must not appear in V8"


# ============================================================
# FEATURE CONSTRUCTION - identical to the validated T4 experiment
# ============================================================

def add_traffic_feature(frame):
    """Count ALL movements at the same physical airport in the preceding
    15 minutes, excluding the current movement.

    Byte-for-byte the V2 / V7 / T4 construction:
      DEP -> ADEP_mvt, ARR -> ADES_mvt
      rolling("15min", closed="left") per airport
      assigned positionally, never merged on airport + timestamp
    """
    rows_before = len(frame)

    frame["movement_airport"] = frame["ADEP_mvt"]
    frame.loc[frame["PHASE_mvt"] == "ARR", "movement_airport"] = frame.loc[
        frame["PHASE_mvt"] == "ARR", "ADES_mvt"
    ]
    frame = frame.sort_values(["movement_airport", "MVT_TIME_UTC_mvt"]).copy()

    traffic_15min = (
        frame
        .set_index("MVT_TIME_UTC_mvt")
        .groupby("movement_airport")["MVT_ID_mvt"]
        .rolling("15min", closed="left")
        .count()
        .reset_index(name="traffic_last_15min")
    )
    frame["traffic_last_15min"] = traffic_15min["traffic_last_15min"].to_numpy()
    frame["traffic_last_15min"] = frame["traffic_last_15min"].fillna(0)

    assert len(frame) == rows_before, (
        f"traffic feature changed row count: {rows_before} -> {len(frame)}"
    )
    return frame


def add_time_features(frame):
    frame["hour"] = frame[MINUEND].dt.hour
    frame["day_of_week"] = frame[MINUEND].dt.dayofweek
    frame["month"] = frame[MINUEND].dt.month
    return frame


def add_t4_features(frame):
    """The five T4 deltas, in seconds.

    All six source columns are timestamp[us, tz=UTC], so the subtraction yields
    timedelta64[us] and .dt.total_seconds() gives seconds directly. The
    object-dtype branch is defensive only and is not taken on this data.
    """
    frame[MINUEND + "_ts"] = pd.to_datetime(frame[MINUEND], utc=True)
    for name, column in T4_SPEC:
        if frame[column].dtype != "object":
            frame[name] = (frame[MINUEND] - frame[column]).dt.total_seconds()
        else:
            frame[name] = (
                frame[MINUEND + "_ts"]
                - pd.to_datetime(frame[column], utc=True)
            ).dt.total_seconds()
    assert MINUEND + "_ts" not in FEATURES
    return frame


def apply_categorical_handling(frame):
    """Same handling as V7: fill with UNKNOWN, then cast to string."""
    for column in V7_CATEGORICALS:
        frame[column] = frame[column].fillna("UNKNOWN").astype(str)
    return frame


def verify_t4_deltas(frame):
    """Independent integer-epoch recheck of the five T4 deltas.

    The parquet timestamps are MICROSECONDS, so this recomputes each delta from
    raw epoch microseconds and requires exact agreement with the
    timedelta-based construction. Guards against a silent unit error.
    """
    import numpy as np
    for name, column in T4_SPEC:
        d = frame[name]
        ok = d.notna()
        a = frame.loc[ok, MINUEND].astype("int64").to_numpy() // 1_000_000
        b = frame.loc[ok, column].astype("int64").to_numpy() // 1_000_000
        manual = (a - b).astype("float64")
        got = d[ok].to_numpy()
        same = int((manual == got).sum())
        assert same == int(ok.sum()), (
            f"{name}: only {same}/{int(ok.sum())} match on the epoch path")
        print(f"  {name:9s} compared={int(ok.sum()):>9,} exact={same:>9,} "
              f"max|diff|={np.abs(manual - got).max():.3e}")
