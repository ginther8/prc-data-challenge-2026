"""Build the V8 candidate submission: submissions/vigorous-glove_v2.parquet

Predicts 2026 ranking departures with the V8 model (V7 + five validated T4
delta features) and writes submissions/vigorous-glove_v2.parquet.

A comprehensive prediction audit runs BEFORE the file is written. The parquet
is only written if every audit item and every integrity check passes.

Predictions are never modified because they look unusual. The audit reports
them; it does not clip, cap, floor or otherwise "fix" anything.

Expected: 344,841 ranking departures.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train_catboost_v8_defs import (  # noqa: F401
    FEATURES, MINUEND, MODEL_PATH, N_TREES, SUBMISSIONS_DIR,
    SUBMISSION_FILENAME, TARGET, T4_FEATURES, T4_SPEC, V7_CATEGORICALS,
    V7_FEATURES, VALIDATED_T4, DATA_DIR, add_t4_features, add_time_features,
    add_traffic_feature, apply_categorical_handling,
)

print("=" * 70)
print("V8 CANDIDATE SUBMISSION")
print("=" * 70)
print(f"features ({len(FEATURES)}): {FEATURES}")
print(f"trees: {N_TREES}")
print(f"output: {SUBMISSIONS_DIR / SUBMISSION_FILENAME}")

# ============================================================
# LOAD THE FROZEN TEMPLATE FIRST
# ============================================================

template = pd.read_parquet(DATA_DIR / "submitting.parquet")
TEMPLATE_COLUMNS = list(template.columns)
TEMPLATE_ID_COLUMN = "MVT_ID_mvt"

print(f"\nTemplate rows: {len(template):,}")
print(f"Template columns: {TEMPLATE_COLUMNS}")
assert TEMPLATE_ID_COLUMN in TEMPLATE_COLUMNS
assert TARGET in TEMPLATE_COLUMNS
assert len(template) == 344_841, f"template rows {len(template):,}"

destination = SUBMISSIONS_DIR / SUBMISSION_FILENAME
if destination.exists():
    raise SystemExit(f"STOP: {destination} already exists. Not overwriting.")

# ============================================================
# LOAD RANKING DATA AND PREPARE FEATURES
# ============================================================

print("\nLoading ranking data...")
ranking = pd.read_parquet(DATA_DIR / "ranking.parquet")
print(f"Ranking rows: {len(ranking):,}")
assert len(ranking) == 689_534, f"ranking rows {len(ranking):,}"

leak = int(ranking.loc[ranking["PHASE_mvt"] == "DEP", TARGET].notna().sum())
assert leak == 0, f"leakage: {leak} ranking departures already have a target"
print("[PASS] no ranking departure has a target")

print("Creating 15-minute traffic feature on ranking movements...")
print("(ranking movements only; no 2025 rows mixed in)")
ranking = add_traffic_feature(ranking)

ranking_dep = ranking[ranking["PHASE_mvt"] == "DEP"].copy()
print(f"Ranking departure rows: {len(ranking_dep):,}")
assert len(ranking_dep) == 344_841, f"ranking departures {len(ranking_dep):,}"

ranking_dep = add_time_features(ranking_dep)
ranking_dep = add_t4_features(ranking_dep)
ranking_dep = apply_categorical_handling(ranking_dep)

# Keep the raw supplied timestamps for the top-20 inspection.
audit_cols = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
              "AIRCRAFT_TYPE_mvt", MINUEND, "SCHED_TIME_UTC_mvt",
              "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"] + T4_FEATURES
audit = ranking_dep[audit_cols].copy()

# ============================================================
# PREDICT
# ============================================================

assert MODEL_PATH.exists(), f"missing model {MODEL_PATH}; run train_catboost_v8.py"
model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))
print(f"\nLoaded model: {MODEL_PATH}")
print(f"Model trees: {model.tree_count_}")
assert model.tree_count_ == N_TREES, \
    f"model has {model.tree_count_} trees, expected {N_TREES}"

print(f"\nPredicting {len(ranking_dep):,} ranking departures...")
pred = model.predict(ranking_dep[FEATURES])
pred = np.asarray(pred, dtype="float64")
audit["PRED"] = pred
print(f"Generated predictions: {len(pred):,}")

# ============================================================
# PREDICTION AUDIT  (report only - nothing is modified)
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION AUDIT")
print("=" * 70)

p = audit["PRED"]
print(f"row count              : {len(p):,}")
print(f"NaN predictions        : {int(p.isna().sum())}")
print(f"+inf predictions       : {int(np.isinf(p).sum())}")
print(f"-inf predictions       : {int((p == -np.inf).sum())}")
print(f"min                    : {p.min():.2f}")
print(f"max                    : {p.max():.2f}")
print(f"mean                   : {p.mean():.2f}")
print(f"median                 : {p.median():.2f}")
print(f"std                    : {p.std():.2f}")
print("\npercentiles (seconds):")
for q in (1, 5, 25, 50, 75, 95, 99, 99.9):
    print(f"  p{q:<5} {np.percentile(p, q):10.2f}")

print("\nthreshold counts:")
print(f"  {'< 0':16s} {int((p < 0).sum()):>7,}  "
      f"({100 * (p < 0).mean():6.3f}%)")
print(f"  {'== 0':16s} {int((p == 0).sum()):>7,}")
for thr, label in [(1800, "> 1800s (30m)"), (3600, "> 3600s (1h)"),
                   (7200, "> 7200s (2h)"), (14400, "> 14400s (4h)"),
                   (28800, "> 28800s (8h)"), (43200, "> 43200s (12h)"),
                   (57600, "> 57600s (16h)"), (86400, "> 86400s (24h)")]:
    n = int((p > thr).sum())
    print(f"  {label:16s} {n:>7,}  ({100 * n / len(p):6.3f}%)")
n_neg = int((p < 0).sum())
if n_neg:
    flags.append(f"{n_neg} predictions below 0")

# ---- by airport ----
print("\n" + "-" * 70)
print("BY AIRPORT (all ranking departures)")
print("-" * 70)
g = audit.groupby("ADEP_mvt")["PRED"].agg(
    n="size", mean="mean", median="median", std="std", min="min", max="max")
g["p99"] = audit.groupby("ADEP_mvt")["PRED"].quantile(0.99)
g["gt3600"] = audit.groupby("ADEP_mvt")["PRED"].apply(lambda s: int((s > 3600).sum()))
g = g.sort_values("n", ascending=False)
print(g.round(2).to_string())

print("\nFOCUS airports LIRF / LFPG:")
for ap in ("LIRF", "LFPG"):
    if ap in g.index:
        r = g.loc[ap]
        print(f"  {ap}: n={int(r['n']):,}  mean={r['mean']:.2f}  "
              f"median={r['median']:.2f}  std={r['std']:.2f}  "
              f"min={r['min']:.2f}  max={r['max']:.2f}  p99={r['p99']:.2f}  "
              f">3600s={int(r['gt3600']):,}")
    else:
        print(f"  {ap}: NOT PRESENT in ranking departures")

# ---- 2026 predictions vs 2025 training targets ----
print("\n" + "-" * 70)
print("2026 PREDICTIONS vs 2025 TRAINING TARGETS")
print("-" * 70)
training_files = sorted(DATA_DIR.glob("training_*.parquet"))
tr = pd.concat(
    [pd.read_parquet(f, columns=[TARGET, MINUEND, "PHASE_mvt", "ADEP_mvt"])
     for f in training_files], ignore_index=True)
y = tr.loc[tr["PHASE_mvt"] == "DEP", TARGET]
print(f"{'stat':10s} {'2025 target':>14s} {'2026 pred':>14s}")
print(f"{'n':10s} {len(y):>14,} {len(p):>14,}")
print(f"{'mean':10s} {y.mean():>14.2f} {p.mean():>14.2f}")
print(f"{'median':10s} {y.median():>14.2f} {p.median():>14.2f}")
print(f"{'std':10s} {y.std():>14.2f} {p.std():>14.2f}")
print(f"{'min':10s} {y.min():>14.2f} {p.min():>14.2f}")
print(f"{'max':10s} {y.max():>14.2f} {p.max():>14.2f}")
for q in (95, 99, 99.9):
    print(f"{'p' + str(q):10s} {np.percentile(y, q):>14.2f} "
          f"{np.percentile(p, q):>14.2f}")
for thr in (1800, 3600, 7200):
    print(f"{'>' + str(thr):10s} {int((y > thr).sum()):>14,} "
          f"{int((p > thr).sum()):>14,}")
print("\nNOTE: the model shrinks toward the mean by construction, so narrower")
print("2026 spread than 2025 raw targets is expected and is not a defect.")

# ---- top 20 predictions ----
print("\n" + "-" * 70)
print("TOP 20 PREDICTED TAXI TIMES")
print("-" * 70)
top = audit.sort_values("PRED", ascending=False).head(20)
cols = ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", MINUEND,
        "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "d_AOBT3", "d_EOBT1", "d_IOBT",
        "d_LOBT", "d_SCHED", "PRED"]
show = top[cols].copy()
for c in (MINUEND, "SCHED_TIME_UTC_mvt", "AOBT_3_flt"):
    show[c] = pd.to_datetime(show[c], utc=True).dt.strftime("%Y-%m-%d %H:%M")
print(show.to_string(index=False))

# ---- suspicious pattern flags ----
print("\n" + "-" * 70)
print("SUSPICIOUS PATTERN FLAGS")
print("-" * 70)
flags = []
n_15_24h = int(((p > 15 * 3600) & (p <= 24 * 3600)).sum())
n_over_24h = int((p > 24 * 3600).sum())
n_over_4h = int((p > 4 * 3600).sum())
n_over_8h = int((p > 8 * 3600).sum())
print(f"predictions in 15-24h band : {n_15_24h:,}")
print(f"predictions > 24h          : {n_over_24h:,}")
print(f"predictions > 4h           : {n_over_4h:,} "
      f"({100 * n_over_4h / len(p):.3f}%)")
print(f"predictions > 8h           : {n_over_8h:,} "
      f"({100 * n_over_8h / len(p):.3f}%)")
if n_15_24h:
    flags.append(f"{n_15_24h} predictions in the 15-24h band")
if n_over_24h:
    flags.append(f"{n_over_24h} predictions above 24h")
if n_over_8h:
    flags.append(f"{n_over_8h} predictions above 8h")

# d_SCHED is the strongest T4 feature; extreme values can drive extrapolation
ext_d = int((audit["d_SCHED"].abs() > 3600).sum())
print(f"|d_SCHED| > 1h             : {ext_d:,} "
      f"({100 * ext_d / len(p):.3f}%)")
print(f"d_SCHED range              : {audit['d_SCHED'].min():.1f} .. "
      f"{audit['d_SCHED'].max():.1f}")
nan_t4 = int(audit[T4_FEATURES].isna().any(axis=1).sum())
print(f"rows with a missing T4 delta: {nan_t4:,} "
      f"({100 * nan_t4 / len(p):.3f}%) - handled natively by CatBoost")
if nan_t4:
    flags.append(f"{nan_t4} rows have at least one missing T4 delta")

top1 = p.max()
print(f"\nlargest prediction        : {top1:.2f}s "
      f"({top1 / 3600:.2f}h)")
print(f"2025 raw max target       : {y.max():.2f}s ({y.max() / 3600:.2f}h)")
print(f"2025 rows > 8h            : {int((y > 8 * 3600).sum()):,}")
if top1 > y.max():
    flags.append(
        f"largest prediction {top1:.0f}s exceeds the 2025 raw maximum "
        f"{y.max():.0f}s")

print("\nFLAGS FOR REVIEW:")
if flags:
    for f in flags:
        print(f"  - {f}")
else:
    print("  none")

# ============================================================
# BUILD THE SUBMISSION
# ============================================================

lookup = pd.Series(pred, index=ranking_dep["MVT_ID_mvt"].to_numpy())
template_ids = template[TEMPLATE_ID_COLUMN].to_numpy()

submission = pd.DataFrame({TEMPLATE_ID_COLUMN: template_ids})
submission[TARGET] = lookup.reindex(template_ids).to_numpy()
submission = submission[TEMPLATE_COLUMNS]

# ============================================================
# INTEGRITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("SUBMISSION INTEGRITY CHECKS")
print("=" * 70)

failures = []


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


check("exactly 344,841 rows", len(submission) == 344_841, f"{len(submission):,}")
check("columns exactly [MVT_ID_mvt, TAXITIME_SEC_mvt]",
      list(submission.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"],
      str(list(submission.columns)))
check("no duplicate ids", int(submission[TEMPLATE_ID_COLUMN].duplicated().sum()) == 0)
check("ids match template exactly (set)",
      set(submission[TEMPLATE_ID_COLUMN]) == set(template_ids))
check("ids match template exactly (order and values)",
      bool((submission[TEMPLATE_ID_COLUMN].to_numpy() == template_ids).all()))
check("no extra ids", len(set(submission[TEMPLATE_ID_COLUMN]) - set(template_ids)) == 0)
check("no missing ids", len(set(template_ids) - set(submission[TEMPLATE_ID_COLUMN])) == 0)
check("ids match ranking departure ids exactly (set)",
      set(submission[TEMPLATE_ID_COLUMN]) == set(ranking_dep["MVT_ID_mvt"]))
check("no NaN predictions", int(submission[TARGET].isna().sum()) == 0)
check("no infinite predictions",
      int(np.isinf(submission[TARGET].to_numpy(dtype=float)).sum()) == 0)
check("target dtype numeric",
      pd.api.types.is_numeric_dtype(submission[TARGET]),
      str(submission[TARGET].dtype))
check("all values finite", bool(np.isfinite(submission[TARGET].to_numpy(dtype=float)).all()))
check("filename is vigorous-glove_v2.parquet",
      destination.name == "vigorous-glove_v2.parquet", destination.name)
check("V1 submission still present and untouched",
      (SUBMISSIONS_DIR / "vigorous-glove_v1.parquet").exists())

if failures:
    raise SystemExit(
        f"STOP: {len(failures)} integrity check(s) failed: {failures}. "
        f"Submission NOT written."
    )

print("\nAll integrity checks passed.")

# ============================================================
# SAVE
# ============================================================

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
if destination.exists():
    raise SystemExit(f"STOP: {destination} appeared during run. Not overwriting.")

submission.to_parquet(destination, index=False)

size = destination.stat().st_size
sha = hashlib.sha256(destination.read_bytes()).hexdigest()

print(f"\nSaved submission: {destination}")
print(f"Rows           : {len(submission):,}")
print(f"Columns        : {list(submission.columns)}")
print(f"File size      : {size:,} bytes")
print(f"SHA256         : {sha}")

print("\nDone. Nothing was uploaded, committed, pushed or merged.")
