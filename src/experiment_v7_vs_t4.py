"""V7 vs V7+T4 on the January and July 2025 folds.

T4 adds exactly five numeric features to the V7 feature set. Every V7 feature,
every V7 hyper-parameter and the V7 traffic feature are reproduced unchanged,
so the only difference between the two arms is the presence of T4.

T4 definitions (all in seconds, UTC):
    d_AOBT3 = MVT_TIME_UTC_mvt - AOBT_3_flt
    d_EOBT1 = MVT_TIME_UTC_mvt - EOBT_1_flt
    d_IOBT  = MVT_TIME_UTC_mvt - IOBT_flt
    d_LOBT  = MVT_TIME_UTC_mvt - LOBT_flt
    d_SCHED = MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt

All six source columns are timestamp[us, tz=UTC]. Subtraction therefore yields
timedelta64[us] and .dt.total_seconds() gives seconds directly. No .asi8 /
.view("int64") epoch arithmetic is used anywhere, because the parquet
resolution is MICROSECONDS, not nanoseconds.

T4 uses neither BLOCK_TIME_UTC_mvt nor TAXITIME_SEC_mvt; this is asserted
below and re-checked in the leakage diagnostic, which also reports how often
d_AOBT3 coincides with the target.

Usage:
    python src/experiment_v7_vs_t4.py                 # full run, both folds
    python src/experiment_v7_vs_t4.py --smoke         # 5 iterations, plumbing only
    python src/experiment_v7_vs_t4.py --dry-run       # data prep only, NO training
    python src/experiment_v7_vs_t4.py --folds JUL     # one fold
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

DATA_DIR = Path("data/raw")

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true",
                    help="5 iterations, to check plumbing only")
parser.add_argument("--dry-run", action="store_true",
                    help="prepare data and report, but never call model.fit")
parser.add_argument("--folds", nargs="+", default=["JAN", "JUL"],
                    choices=["JAN", "JUL"])
args = parser.parse_args()

# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# Copied verbatim from src/train_catboost_v7.py. Do not edit V7 to match this.
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
T4_SOURCES = ["MVT_TIME_UTC_mvt"] + [col for _, col in T4_SPEC]

MINUEND = "MVT_TIME_UTC_mvt"
TARGET = "TAXITIME_SEC_mvt"
FORBIDDEN = ["BLOCK_TIME_UTC_mvt", TARGET]

# Identical to V7.
CB_PARAMS = dict(
    iterations=1000,
    depth=8,
    learning_rate=0.1,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    use_best_model=True,
)

# (label, first day of holdout month, first day of next month)
FOLDS = {
    "JAN": ("2025-01-01", "2025-02-01"),
    "JUL": ("2025-07-01", "2025-08-01"),
}

# Known V7 scores on these folds, from the historical-feature experiment.
V7_KNOWN = {"JAN": 519.90, "JUL": 630.63}


def check_no_leakage(feature_list, label):
    """T4 must not read the withheld target or the on-block time."""
    bad = [f for f in feature_list if f in FORBIDDEN]
    assert not bad, f"{label} uses forbidden column(s): {bad}"


check_no_leakage(V7_FEATURES, "V7_FEATURES")
check_no_leakage(V7_FEATURES + T4_FEATURES, "V7+T4_FEATURES")
check_no_leakage(T4_SOURCES, "T4_SOURCES")
assert len(set(V7_FEATURES) & set(T4_FEATURES)) == 0, "T4 name collides with V7"

print("=" * 78)
print("V7 vs V7+T4  -  January and July 2025 holdout folds")
print("=" * 78)
print(f"V7 features      : {len(V7_FEATURES)}")
print(f"T4 features      : {len(T4_FEATURES)}  {T4_FEATURES}")
print(f"T4 sources       : {T4_SOURCES}")
print(f"forbidden check  : no feature uses {FORBIDDEN}  [OK]")
print(f"folds            : {args.folds}")
if args.smoke:
    print("MODE             : SMOKE (5 iterations) - numbers are meaningless")
if args.dry_run:
    print("MODE             : DRY RUN - model.fit is never called")
print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

training_files = sorted(DATA_DIR.glob("training_*.parquet"))
print(f"  files: {len(training_files)}")

data = pd.concat(
    [pd.read_parquet(f) for f in training_files],
    ignore_index=True,
)

print(f"Total movements: {len(data):,}")

# Required columns must all exist before anything else is attempted.
for col in T4_SOURCES + [MINUEND, "PHASE_mvt", "MVT_ID_mvt", TARGET]:
    assert col in data.columns, f"missing column {col}"
print("All T4 source columns present.")


# ============================================================
# CONGESTION FEATURE  (identical to V7)
# ============================================================

print("\nCreating 15-minute airport traffic feature...")

data["movement_airport"] = data["ADEP_mvt"]
data.loc[
    data["PHASE_mvt"] == "ARR", "movement_airport"
] = data.loc[
    data["PHASE_mvt"] == "ARR", "ADES_mvt"
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

data["traffic_last_15min"] = traffic_15min["traffic_last_15min"].to_numpy()
data["traffic_last_15min"] = data["traffic_last_15min"].fillna(0)


# ============================================================
# KEEP DEPARTURES
# ============================================================

departures = data[data["PHASE_mvt"] == "DEP"].copy()

print(f"Departure rows: {len(departures):,}")

assert len(departures) == 2_085_047, (
    f"Unexpected departure count: {len(departures):,}"
)


# ============================================================
# TIME FEATURES  (identical to V7)
# ============================================================

departures["hour"] = departures[MINUEND].dt.hour
departures["day_of_week"] = departures[MINUEND].dt.dayofweek
departures["month"] = departures[MINUEND].dt.month


# ============================================================
# T4 FEATURES
# ============================================================

print("\nCreating the five T4 features (seconds)...")

departures[MINUEND + "_ts"] = pd.to_datetime(
    departures[MINUEND], utc=True
)

for name, column in T4_SPEC:
    if departures[column].dtype != "object":
        # already a proper tz-aware timestamp
        departures[name] = (
            departures[MINUEND] - departures[column]
        ).dt.total_seconds()
    else:
        departures[name] = (
            departures[MINUEND + "_ts"]
            - pd.to_datetime(departures[column], utc=True)
        ).dt.total_seconds()

# The helper column is not a feature and must not leak into any frame.
assert MINUEND + "_ts" not in V7_FEATURES + T4_FEATURES

print("T4 feature summary over all 2025 departures:")
for name, _ in T4_SPEC:
    s = departures[name]
    print(
        f"  {name:9s} dtype={str(s.dtype):8s} "
        f"nan={int(s.isna().sum()):7,} "
        f"({100 * s.isna().mean():5.2f}%)  "
        f"min={s.min():11,.0f}  p50={s.median():9,.0f}  max={s.max():11,.0f}"
    )

# Exactly three of the five share one null mask; d_SCHED is always present.
null_masks = {name: departures[name].isna() for name, _ in T4_SPEC}
grouped = {}
for name, mask in null_masks.items():
    grouped.setdefault(tuple(mask.to_numpy()), []).append(name)
print(f"\n  distinct null patterns among the five: {len(grouped)}")
for names in grouped.values():
    print(f"    {names}  -> {int(null_masks[names[0]].sum()):,} nulls")


# ============================================================
# CATEGORICAL MISSING VALUE HANDLING  (identical to V7)
# ============================================================

for column in V7_CATEGORICALS:
    departures[column] = (
        departures[column]
        .fillna("UNKNOWN")
        .astype(str)
    )


# ============================================================
# LEAKAGE DIAGNOSTIC FOR T4
# ============================================================

print("\n" + "=" * 78)
print("T4 LEAKAGE DIAGNOSTIC")
print("=" * 78)
print("d_AOBT3 = off-block - AOBT_3(on-block). AOBT_3_flt is minute-rounded,")
print("so wherever it is the on-block time of THIS departure, d_AOBT3 is the")
print("target to within 30s. Quantified below so any RMSE gain is readable.")

block = pd.to_datetime(departures["BLOCK_TIME_UTC_mvt"], utc=True)
target = departures[TARGET].astype("float64")
onblock_match = (departures["AOBT_3_flt"] == block.dt.round("min"))

print(f"  rows where AOBT_3 == round(BLOCK_TIME,'min'): "
      f"{int(onblock_match.sum()):,} ({100 * onblock_match.mean():.2f}%)")

sub = onblock_match & departures["d_AOBT3"].notna() & target.notna()
err = (departures.loc[sub, "d_AOBT3"] - target[sub]).abs()
print(f"  on those rows |d_AOBT3 - TAXITIME|: median={err.median():.0f}s  "
      f"p99={err.quantile(.99):.0f}s  max={err.max():.0f}s")

for thr in (31, 60, 300, 900):
    frac = 100 * (departures["d_AOBT3"].notna() & target.notna()
                  & ((departures["d_AOBT3"] - target).abs() < thr)).sum() / len(departures)
    print(f"  |d_AOBT3 - TAXITIME| < {thr:4d}s on {frac:6.2f}% of departures")

print("\n  Pearson r against the target (all 2025 departures):")
for name, _ in T4_SPEC:
    m = departures[name].notna() & target.notna()
    r = np.corrcoef(departures.loc[m, name], target[m])[0, 1]
    print(f"    {name:9s} r={r:+.4f}  r^2={r * r:.4f}")

print("\n  Interpret: a Jan/July improvement is only trustworthy as genuine")
print("  signal if it survives the d_AOBT3 contamination noted above.")
print()


# ============================================================
# FIT ONE ARM
# ============================================================

def evaluate(model, X, y, tag):
    pred = model.predict(X)
    rmse = root_mean_squared_error(y, pred)
    mae = np.abs(y.to_numpy() - pred).mean()
    print(f"    {tag:8s} RMSE={rmse:8.2f}   MAE={mae:7.2f}   "
          f"best_iter={model.get_best_iteration()}")
    return rmse, mae, pred


def run_fold(fold):
    lo, hi = FOLDS[fold]
    holdout = (departures[MINUEND] >= lo) & (departures[MINUEND] < hi)

    tr = departures[~holdout]
    va = departures[holdout]

    print(f"\n{'-' * 78}")
    print(f"FOLD {fold}: holdout {lo} .. {hi}")
    print(f"{'-' * 78}")
    print(f"  train rows: {len(tr):,}   validation rows: {len(va):,}")

    y_tr = tr[TARGET]
    y_va = va[TARGET]

    results = {}

    # Both arms always run: the whole point is an exact A/B.
    arms = [("V7", V7_FEATURES), ("V7+T4", V7_FEATURES + T4_FEATURES)]

    for tag, features in arms:
        print(f"  arm {tag}  ({len(features)} features)")
        if args.dry_run:
            print("    dry run: fit skipped")
            continue

        params = dict(CB_PARAMS)
        if args.smoke:
            params["iterations"] = 5
            params["early_stopping_rounds"] = 5
            params["verbose"] = 0

        t0 = time.time()
        model = CatBoostRegressor(**params)
        model.fit(
            tr[features],
            y_tr,
            cat_features=V7_CATEGORICALS,
            eval_set=(va[features], y_va),
        )
        elapsed = time.time() - t0

        rmse, mae, pred = evaluate(model, va[features], y_va, tag)
        print(f"    fit time: {elapsed / 60:.1f} min")
        results[tag] = dict(rmse=rmse, mae=mae, pred=pred, model=model,
                            features=features)

    return results


# ============================================================
# RUN
# ============================================================

if args.dry_run:
    print("=" * 78)
    print("DRY RUN: verifying fold construction without training")
    print("=" * 78)
    for fold in args.folds:
        lo, hi = FOLDS[fold]
        holdout = (departures[MINUEND] >= lo) & (departures[MINUEND] < hi)
        tr, va = departures[~holdout], departures[holdout]
        print(f"\n{fold}: train={len(tr):,}  validation={len(va):,}")
        print(f"  validation month values: "
              f"{sorted(va['month'].unique())}")
        print(f"  train months: {sorted(tr['month'].unique())}")
        assert lo[:7] not in set(tr["month"]), "holdout month leaked into train"
        print(f"  validation RMSE check only (no fit) - features present: "
              f"{all(c in va.columns for c in V7_FEATURES + T4_FEATURES)}")
    print("\nDry run complete. No model was trained.")
else:
    all_results = {}
    for fold in args.folds:
        all_results[fold] = run_fold(fold)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'fold':6s} {'V7':>10s} {'V7+T4':>10s} {'delta':>10s} "
          f"{'V7 known':>10s} {'match':>8s}")
    for fold, res in all_results.items():
        if "V7" not in res or "V7+T4" not in res:
            continue
        a, b = res["V7"]["rmse"], res["V7+T4"]["rmse"]
        known = V7_KNOWN[fold]
        print(f"{fold:6s} {a:10.2f} {b:10.2f} {b - a:+10.2f} "
              f"{known:10.2f} {abs(a - known) < 0.05:8.4f}")

    print("\nFeature importance, V7+T4 arm:")
    for fold, res in all_results.items():
        if "V7+T4" not in res:
            continue
        imp = pd.DataFrame({
            "feature": res["V7+T4"]["features"],
            "importance": res["V7+T4"]["model"].get_feature_importance(),
        }).sort_values("importance", ascending=False)
        t4_imp = imp[imp["feature"].isin(T4_FEATURES)]
        print(f"\n  {fold}: T4 total importance share = "
              f"{100 * t4_imp['importance'].sum() / imp['importance'].sum():.1f}%")
        print(t4_imp.to_string(index=False))
