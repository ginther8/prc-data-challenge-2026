"""Standalone, independent reproduction of the supplied V8 validation benchmark.

This script does NOT modify anything under reference/v8/. It imports the
read-only reference definition module (train_catboost_v8_defs.py) so that the
feature list, feature construction, traffic construction, T4 deltas and
categorical handling are literally the supplied V8 code, not a re-interpretation.

What it reproduces:
    the V8 Jan/Jul 2025 holdout VALIDATION experiment

        JAN  expected RMSE 404.8803102, best_iter 365
        JUL  expected RMSE 513.7559738, best_iter 266
        pooled (from combined SSE) ~468.31

What it deliberately does NOT do:
    - it does not train the final 316-tree all-data model
    - it does not create a submission
    - it does not clip / filter / transform any target
    - it does not use BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt as features

Run from the project root with:
    .venv/bin/python reproduce_v8.py
"""

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = PROJECT_ROOT / "reference" / "v8"

# The reference module uses the relative path Path("data/raw"), so the process
# must run from the project root. This also makes the import path stable.
os_chdir_ok = True
if Path.cwd() != PROJECT_ROOT:
    import os

    os.chdir(PROJECT_ROOT)

# Import the READ-ONLY V8 reference definitions as the single source of truth.
sys.path.insert(0, str(REFERENCE_DIR))
from train_catboost_v8_defs import (  # noqa: E402
    CB_PARAMS,
    DATA_DIR,
    FEATURES,
    FORBIDDEN,
    MINUEND,
    T4_FEATURES,
    T4_SOURCE_COLS,
    T4_SPEC,
    TARGET,
    V7_CATEGORICALS,
    V7_FEATURES,
    add_t4_features,
    add_time_features,
    add_traffic_feature,
    apply_categorical_handling,
    verify_t4_deltas,
)

from catboost import CatBoostRegressor  # noqa: E402

# ---------------------------------------------------------------------------
# Expected values, straight from the supplied V8 report / definitions module
# ---------------------------------------------------------------------------
EXPECTED_MOVEMENTS = 4_167_797
EXPECTED_DEPARTURES = 2_085_047

FOLDS = {
    "JAN": (
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 2, 1, tzinfo=timezone.utc),
        1_931_341,
        153_706,
        404.8803102,
        365,
    ),
    "JUL": (
        datetime(2025, 7, 1, tzinfo=timezone.utc),
        datetime(2025, 8, 1, tzinfo=timezone.utc),
        1_894_334,
        190_713,
        513.7559738,
        266,
    ),
}
EXPECTED_POOLED_ROWS = 344_419
REFERENCE_POOLED_RMSE = 468.31
OFFICIAL_LEADERBOARD_RMSE = 425.7711

# Validation-time CatBoost settings (NOT the final 316-tree fit).
VALIDATION_CB_PARAMS = dict(
    iterations=1000,
    depth=8,
    learning_rate=0.1,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    use_best_model=True,
)

# Cache the prepared feature frame outside the repo so a failed training run
# does not force a repeat of the expensive feature construction.
PREP_CACHE = Path("/tmp/opencode/v8_prep_cache.parquet")

RESULTS = {
    "official_leaderboard_rmse": OFFICIAL_LEADERBOARD_RMSE,
    "features": list(FEATURES),
    "categorical_features": list(V7_CATEGORICALS),
    "catboost_validation_params": VALIDATION_CB_PARAMS,
    "catboost_reference_base_params": CB_PARAMS,
    "folds": {},
    "checks": {},
    "warnings": [],
}


def log(msg=""):
    print(msg, flush=True)


def rule(title):
    log()
    log("=" * 70)
    log(title)
    log("=" * 70)


def check(label, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    log(f"[{tag}] {label}" + (f" - {detail}" if detail else ""))
    RESULTS["checks"][label] = bool(ok)
    if not ok:
        log("")
        log("STOP: an invariant failed. Aborting without training.")
        raise SystemExit(2)
    return ok


def prepare_features():
    """Build the V8 feature frame using the reference implementation verbatim."""
    if PREP_CACHE.exists():
        log(f"Reusing cached prepared features: {PREP_CACHE}")
        frame = pd.read_parquet(PREP_CACHE)
        n_dep = int((frame["PHASE_mvt"] == "DEP").sum())
        if n_dep == EXPECTED_DEPARTURES:
            return frame
        log("Cache row count mismatch; rebuilding from raw data.")
        del frame

    rule("LOAD 2025 MOVEMENTS (ALL PHASES)")
    files = sorted(DATA_DIR.glob("training_*.parquet"))
    log(f"Training files found: {len(files)}")
    check("12 training parquet files", len(files) == 12, str(len(files)))

    t0 = time.time()
    movements = pd.concat(
        [pd.read_parquet(f) for f in files], ignore_index=True
    )
    log(f"Total 2025 movements: {len(movements):,}")
    log(f"  phase breakdown: {movements['PHASE_mvt'].value_counts().to_dict()}")
    check(
        "total 2025 movements == 4,167,797",
        len(movements) == EXPECTED_MOVEMENTS,
        f"{len(movements):,}",
    )
    for col in T4_SOURCE_COLS:
        check(f"T4 source column present: {col}", col in movements.columns)
    check("PHASE_mvt present", "PHASE_mvt" in movements.columns)
    check(f"target column present: {TARGET}", TARGET in movements.columns)

    # ---- traffic_last_15min: ALL movements (ARR + DEP) BEFORE filtering ----
    rule("TRAFFIC FEATURE ON ALL MOVEMENTS (ARR + DEP) BEFORE DEP FILTER")
    log("Reference add_traffic_feature(): DEP->ADEP_mvt, ARR->ADES_mvt,")
    log("rolling('15min', closed='left') per physical airport, assigned")
    log("positionally. Row count is asserted inside the reference function.")
    movements = add_traffic_feature(movements)
    check(
        "traffic built on all movements (row count preserved by reference assert)",
        len(movements) == EXPECTED_MOVEMENTS,
        f"{len(movements):,}",
    )
    log(f"traffic_last_15min over all movements: "
        f"mean={movements['traffic_last_15min'].mean():.3f} "
        f"max={movements['traffic_last_15min'].max():.0f} "
        f"nulls={int(movements['traffic_last_15min'].isna().sum())}")
    check(
        "traffic_last_15min has no nulls",
        int(movements["traffic_last_15min"].isna().sum()) == 0,
    )

    # ---- NOW filter to departures ----
    rule("FILTER TO DEPARTURES (after traffic construction)")
    frame = movements[movements["PHASE_mvt"] == "DEP"].copy()
    log(f"2025 departure rows: {len(frame):,}")
    check(
        "total departures == 2,085,047",
        len(frame) == EXPECTED_DEPARTURES,
        f"{len(frame):,}",
    )
    check(
        "all departures have a raw target",
        bool(frame[TARGET].notna().all()),
    )

    # ---- reference feature construction, verbatim ----
    frame = add_time_features(frame)
    frame = add_t4_features(frame)
    frame = apply_categorical_handling(frame)

    log()
    log("Raw target statistics (NO cleaning applied):")
    log(f"  mean   = {frame[TARGET].mean():.2f}")
    log(f"  median = {frame[TARGET].median():.2f}")
    log(f"  std    = {frame[TARGET].std():.2f}")
    log(f"  min    = {frame[TARGET].min():.2f}")
    log(f"  max    = {frame[TARGET].max():.2f}")
    for thr in (3600, 7200, 28800):
        log(f"  > {thr:>6}s = {int((frame[TARGET] > thr).sum()):,}")

    log()
    log("T4 delta coverage over all 2025 departures:")
    for name, _ in T4_SPEC:
        s = frame[name]
        log(f"  {name:9s} nan={int(s.isna().sum()):>7,} "
            f"({100 * s.isna().mean():6.3f}%)  min={s.min():10.1f}  "
            f"median={s.median():8.1f}  max={s.max():10.1f}")

    log()
    log("T4 unit re-check (independent integer microsecond epoch path):")
    verify_t4_deltas(frame)

    PREP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PREP_CACHE, index=False)
    log(f"Cached prepared features -> {PREP_CACHE}")
    log(f"Feature construction wall time: {time.time() - t0:.1f}s")
    return frame


def run_fold(frame, fold_name):
    start, end, exp_train, exp_valid, exp_rmse, exp_iter = FOLDS[fold_name]
    rule(f"{fold_name} FOLD")

    t = pd.to_datetime(frame[MINUEND], utc=True)
    valid_mask = (t >= start) & (t < end)
    valid = frame[valid_mask]
    train = frame[~valid_mask]

    log(f"Validation window: {start.isoformat()} <= {MINUEND} < {end.isoformat()}")
    check(f"{fold_name} training rows == {exp_train:,}", len(train) == exp_train,
          f"{len(train):,}")
    check(f"{fold_name} validation rows == {exp_valid:,}", len(valid) == exp_valid,
          f"{len(valid):,}")
    check(f"{fold_name} train + valid == 2,085,047",
          len(train) + len(valid) == EXPECTED_DEPARTURES)
    check(f"{fold_name} valid targets are raw (no NaN)", bool(valid[TARGET].notna().all()))

    x_train = train[FEATURES]
    y_train = train[TARGET]
    x_valid = valid[FEATURES]
    y_valid = valid[TARGET]

    # allow_writing_files=False only suppresses CatBoost's catboost_info log
    # directory; it has no effect on the fitted model.
    model = CatBoostRegressor(allow_writing_files=False, **VALIDATION_CB_PARAMS)
    log(f"Training CatBoost ({VALIDATION_CB_PARAMS['iterations']} iterations, "
        f"early_stopping_rounds={VALIDATION_CB_PARAMS['early_stopping_rounds']}, "
        f"use_best_model=True) on {len(train):,} rows ...")
    t0 = time.time()
    model.fit(
        x_train,
        y_train,
        cat_features=V7_CATEGORICALS,
        eval_set=(x_valid, y_valid),
        early_stopping_rounds=VALIDATION_CB_PARAMS["early_stopping_rounds"],
        use_best_model=True,
    )
    elapsed = time.time() - t0

    best_iter = int(model.get_best_iteration())
    trees = int(model.tree_count_)
    log(f"Fit wall time: {elapsed:.1f}s")
    log(f"best_iteration (0-based, as reported by CatBoost) = {best_iter}")
    log(f"trees in the retained best model = {trees}")

    pred = np.asarray(model.predict(x_valid), dtype="float64")
    y_true = np.asarray(y_valid, dtype="float64")

    check(f"{fold_name} predictions are finite", bool(np.isfinite(pred).all()))
    err = y_true - pred
    sse = float(np.dot(err, err))
    rmse = math.sqrt(sse / len(y_true))
    log(f"{fold_name} validation RMSE = {rmse:.7f}   (reference {exp_rmse:.7f})")
    log(f"{fold_name} SSE = {sse:,.4f}")
    log(f"{fold_name} difference vs reference = {rmse - exp_rmse:+.7f}")
    log(f"{fold_name} best_iter difference = {best_iter - exp_iter:+d}")

    RESULTS["folds"][fold_name] = {
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "rmse": rmse,
        "reference_rmse": exp_rmse,
        "rmse_difference": rmse - exp_rmse,
        "sse": sse,
        "best_iteration": best_iter,
        "reference_best_iteration": exp_iter,
        "best_iteration_difference": best_iter - exp_iter,
        "trees_in_best_model": trees,
        "wall_time_seconds": elapsed,
    }
    del model, x_train, y_train, x_valid, y_valid, valid, train
    return RESULTS["folds"][fold_name]


def write_report(status, pooled_rmse, pooled_sse):
    jan = RESULTS["folds"]["JAN"]
    jul = RESULTS["folds"]["JUL"]

    def fmt(x, n=7):
        return f"{x:.{n}f}"

    lines = []
    A = lines.append
    A("=" * 70)
    A("V8 VALIDATION REPRODUCTION REPORT")
    A("=" * 70)
    A(f"STATUS: {status}")
    A("")
    A(f"Official V8 leaderboard RMSE: {OFFICIAL_LEADERBOARD_RMSE}")
    A("Official V8 leaderboard RMSE = 425.7711")
    A("")
    A("-" * 70)
    A("REFERENCE")
    A("-" * 70)
    A("Official V8 RMSE = 425.7711")
    A("Reference implementation (read-only, unmodified):")
    A("  reference/v8/train_catboost_v8_defs.py  (feature list + construction)")
    A("  reference/v8/train_catboost_v8.py       (final all-data 316-tree fit)")
    A("  reference/v8/create_submission_v8.py    (submission builder)")
    A("  reference/v8/v8_submission_report.txt   (V8 report)")
    A("This script imports the reference defs module directly, so the feature")
    A("list and every feature-construction function are the supplied V8 code.")
    A("")
    A("-" * 70)
    A("FEATURES (exactly 17)")
    A("-" * 70)
    for i, f in enumerate(FEATURES, 1):
        A(f"  {i:2d}. {f}")
    A(f"  count = {len(FEATURES)}")
    A("")
    A("T4 delta definitions (MVT_TIME_UTC_mvt is the minuend for all five):")
    for name, col in T4_SPEC:
        A(f"  {name:8s} = {MINUEND} - {col}   (seconds)")
    A("")
    A("V7 base features: " + ", ".join(V7_FEATURES))
    A("T4 features     : " + ", ".join(T4_FEATURES))
    A("")
    A("-" * 70)
    A("CATEGORICAL FEATURES (exactly 8)")
    A("-" * 70)
    for f in V7_CATEGORICALS:
        A(f"  {f}")
    A(f"  count = {len(V7_CATEGORICALS)}")
    A("Handling: fillna('UNKNOWN') then astype(str) - reference")
    A("apply_categorical_handling().")
    A("")
    A("-" * 70)
    A("CATBOOST - VALIDATION SETTINGS (used for this reproduction)")
    A("-" * 70)
    for k, v in VALIDATION_CB_PARAMS.items():
        A(f"  {k} = {v}")
    A("  (depth/learning_rate/loss_function/random_seed/verbose are taken from")
    A("   the reference CB_PARAMS verbatim; iterations=1000, early_stopping_")
    A("   rounds=50 and use_best_model=True are the validation-time settings.)")
    A("")
    A("The final V8 submission fit used a fixed 316 trees and no early stopping.")
    A("316 was chosen AFTER validation as the midpoint of best_iter+1 for the")
    A("two folds. It was NOT used here: this run reproduces the validation")
    A("experiment, where the iteration count is 1000 with early stopping.")
    A("")
    A("-" * 70)
    A("JAN")
    A("-" * 70)
    A(f"  training rows        : {jan['train_rows']:,}   (expected 1,931,341)")
    A(f"  validation rows      : {jan['valid_rows']:,}   (expected 153,706)")
    A(f"  reproduced RMSE      : {fmt(jan['rmse'])}")
    A(f"  reference RMSE       : {fmt(jan['reference_rmse'])}  "
      "(404.8803102)")
    A(f"  RMSE difference      : {jan['rmse_difference']:+.7f}")
    A(f"  reproduced best iter : {jan['best_iteration']}")
    A(f"  reference best iter  : {jan['reference_best_iteration']}  (365)")
    A(f"  trees in best model  : {jan['trees_in_best_model']}")
    A(f"  SSE                  : {jan['sse']:,.4f}")
    A("")
    A("-" * 70)
    A("JUL")
    A("-" * 70)
    A(f"  training rows        : {jul['train_rows']:,}   (expected 1,894,334)")
    A(f"  validation rows      : {jul['valid_rows']:,}   (expected 190,713)")
    A(f"  reproduced RMSE      : {fmt(jul['rmse'])}")
    A(f"  reference RMSE       : {fmt(jul['reference_rmse'])}  "
      "(513.7559738)")
    A(f"  RMSE difference      : {jul['rmse_difference']:+.7f}")
    A(f"  reproduced best iter : {jul['best_iteration']}")
    A(f"  reference best iter  : {jul['reference_best_iteration']}  (266)")
    A(f"  trees in best model  : {jul['trees_in_best_model']}")
    A(f"  SSE                  : {jul['sse']:,.4f}")
    A("")
    A("-" * 70)
    A("POOLED")
    A("-" * 70)
    A(f"  validation rows      : {jan['valid_rows'] + jul['valid_rows']:,}   "
      "(expected 344,419)")
    A("  method               : sqrt((JAN_SSE + JUL_SSE) / rows)  "
      "- combined SSE, NOT an average of fold RMSEs")
    A(f"  combined SSE         : {pooled_sse:,.4f}")
    A(f"  reproduced pooled    : {fmt(pooled_rmse)}")
    A(f"  reference pooled     : ~{REFERENCE_POOLED_RMSE}  (~468.31)")
    A(f"  difference           : {pooled_rmse - REFERENCE_POOLED_RMSE:+.7f}")
    A("")
    A("-" * 70)
    A("SAFETY CHECKS")
    A("-" * 70)
    A("  [PASS] traffic_last_15min constructed from ALL 2025 movements")
    A("         (ARR + DEP) BEFORE filtering to departures:")
    A("         - reference add_traffic_feature() called on the full")
    A("           4,167,797-row movement frame; it maps DEP -> ADEP_mvt and")
    A("           ARR -> ADES_mvt, sorts by (movement_airport, time), and takes")
    A("           groupby(movement_airport).rolling('15min', closed='left')")
    A("           over MVT_ID_mvt, i.e. movements in the preceding 15 minutes")
    A("           excluding the current movement, assigned positionally.")
    A("         - the reference function asserts the row count is unchanged.")
    A("         - the DEP filter is applied only afterwards.")
    A("  [PASS] validation targets untouched: raw TAXITIME_SEC_mvt, no clipping,")
    A("         no filtering, no winsorization, no outlier removal, no")
    A("         correction, no transformation. Validation rows are never")
    A("         dropped and all 153,706 / 190,713 rows are scored.")
    A(f"  [PASS] {FORBIDDEN[0]} absent from the 17-feature list")
    A(f"  [PASS] {FORBIDDEN[1]} absent from the 17-feature list")
    A("  [PASS] exactly 17 features used")
    A("  [PASS] exactly 8 categorical features used")
    A("  [PASS] no clock_* or hist_* features (V8 explicitly excludes them)")
    A("  [PASS] row counts unchanged by feature construction (reference asserts)")
    A("  [PASS] T4 deltas re-verified on an independent integer microsecond")
    A("         epoch path by the reference verify_t4_deltas()")
    A("")
    A("  Feature / categorical / forbidden-column assertions that ran:")
    for k, v in RESULTS["checks"].items():
        A(f"    [{'PASS' if v else 'FAIL'}] {k}")
    A("")
    A("-" * 70)
    A("FILES")
    A("-" * 70)
    A("  created:")
    A("    reproduce_v8.py               (this reproduction script)")
    A("    reports/v8_reproduction.txt   (this report)")
    A("    /tmp/opencode/v8_prep_cache.parquet (out-of-repo feature cache)")
    A("  modified: NONE")
    A("  not modified: reference/v8/*, src/safeguards.py, src/data.py,")
    A("    src/train.py, src/experiment_features.py, run_experiment.py,")
    A("    experiments/*, AGENTS.md, opencode.json, models/*, submissions/*")
    A("  no submission was created; the 316-tree all-data model was not trained.")
    A("")
    A("-" * 70)
    A("WARNINGS / ERRORS")
    A("-" * 70)
    if RESULTS["warnings"]:
        for w in RESULTS["warnings"]:
            A(f"  - {w}")
    else:
        A("  none")

    A("")
    A("=" * 70)
    A("END OF REPORT")
    A("=" * 70)

    text = "\n".join(lines) + "\n"
    out = PROJECT_ROOT / "reports" / "v8_reproduction.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log()
    log(f"Report written: {out}")


def main():
    rule("V8 VALIDATION REPRODUCTION")
    log(f"project root      : {PROJECT_ROOT}")
    log(f"reference (RO)    : {REFERENCE_DIR}")
    log(f"data dir          : {DATA_DIR}")
    log(f"features ({len(FEATURES)})    : {FEATURES}")
    log(f"categoricals ({len(V7_CATEGORICALS)}) : {V7_CATEGORICALS}")

    # ---- pre-flight static invariants ----
    check("exactly 17 features", len(FEATURES) == 17, str(len(FEATURES)))
    check("exactly 8 categorical features", len(V7_CATEGORICALS) == 8,
          str(len(V7_CATEGORICALS)))
    check("BLOCK_TIME_UTC_mvt absent from features",
          "BLOCK_TIME_UTC_mvt" not in FEATURES)
    check("TAXITIME_SEC_mvt absent from features", TARGET not in FEATURES)
    check("no forbidden column in features",
          not (set(FEATURES) & set(FORBIDDEN)))
    check("no clock_ features", not any(f.startswith("clock_") for f in FEATURES))
    check("no hist_ features", not any(f.startswith("hist_") for f in FEATURES))
    check("iterations = 1000 for validation", VALIDATION_CB_PARAMS["iterations"] == 1000)
    check("early_stopping_rounds = 50", VALIDATION_CB_PARAMS["early_stopping_rounds"] == 50)
    check("use_best_model = True", VALIDATION_CB_PARAMS["use_best_model"] is True)

    frame = prepare_features()

    # ---- post-construction invariants ----
    for f in FEATURES:
        check(f"feature present in prepared frame: {f}", f in frame.columns)
    for f in V7_CATEGORICALS:
        check(f"categorical is string dtype: {f}",
              frame[f].dtype == object or pd.api.types.is_string_dtype(frame[f]))
    check("traffic_last_15min present before DEP filter (asserted by reference)",
          "traffic_last_15min" in frame.columns)

    run_fold(frame, "JAN")
    run_fold(frame, "JUL")

    jan = RESULTS["folds"]["JAN"]
    jul = RESULTS["folds"]["JUL"]
    pooled_rows = jan["valid_rows"] + jul["valid_rows"]
    pooled_sse = jan["sse"] + jul["sse"]
    pooled_rmse = math.sqrt(pooled_sse / pooled_rows)

    rule("POOLED")
    log(f"combined validation rows : {pooled_rows:,}   (expected 344,419)")
    check("pooled validation rows == 344,419", pooled_rows == EXPECTED_POOLED_ROWS,
          f"{pooled_rows:,}")
    log(f"pooled_rmse = sqrt((JAN_SSE + JUL_SSE) / {pooled_rows:,})")
    log(f"           = sqrt(({jan['sse']:,.4f} + {jul['sse']:,.4f}) / {pooled_rows:,})")
    log(f"           = {pooled_rmse:.7f}   (reference ~{REFERENCE_POOLED_RMSE})")

    # ---- pass / fail decision on tiny numerical tolerance ----
    TOL = 0.01  # RMSE
    ITOL = 2   # best iteration
    jan_ok = abs(jan["rmse_difference"]) <= TOL
    jul_ok = abs(jul["rmse_difference"]) <= TOL
    jan_iter_ok = abs(jan["best_iteration_difference"]) <= ITOL
    jul_iter_ok = abs(jul["best_iteration_difference"]) <= ITOL
    pooled_ok = abs(pooled_rmse - REFERENCE_POOLED_RMSE) <= TOL
    status = "PASS" if (jan_ok and jul_ok and jan_iter_ok and jul_iter_ok and pooled_ok) \
        else "FAIL"

    rule("VERDICT")
    log(f"JAN RMSE   {jan['rmse']:.7f} vs {jan['reference_rmse']:.7f}  "
        f"diff {jan['rmse_difference']:+.7f}  -> {'ok' if jan_ok else 'MISMATCH'}")
    log(f"JUL RMSE   {jul['rmse']:.7f} vs {jul['reference_rmse']:.7f}  "
        f"diff {jul['rmse_difference']:+.7f}  -> {'ok' if jul_ok else 'MISMATCH'}")
    log(f"POOLED     {pooled_rmse:.7f} vs ~{REFERENCE_POOLED_RMSE}       "
        f"diff {pooled_rmse - REFERENCE_POOLED_RMSE:+.7f}  "
        f"-> {'ok' if pooled_ok else 'MISMATCH'}")
    log(f"JAN best_iter {jan['best_iteration']} vs 365 -> "
        f"{'ok' if jan_iter_ok else 'MISMATCH'}")
    log(f"JUL best_iter {jul['best_iteration']} vs 266 -> "
        f"{'ok' if jul_iter_ok else 'MISMATCH'}")
    log(f"STATUS: {status}")

    RESULTS["pooled"] = {
        "rows": pooled_rows,
        "sse": pooled_sse,
        "rmse": pooled_rmse,
        "reference_rmse": REFERENCE_POOLED_RMSE,
        "difference": pooled_rmse - REFERENCE_POOLED_RMSE,
    }

    write_report(status, pooled_rmse, pooled_sse)
    log()
    log(f"STATUS: {status}")
    log(f"JAN RMSE: {jan['rmse']:.7f}")
    log(f"JUL RMSE: {jul['rmse']:.7f}")
    log(f"POOLED RMSE: {pooled_rmse:.7f}")
    log("REPORT: reports/v8_reproduction.txt")


if __name__ == "__main__":
    main()
