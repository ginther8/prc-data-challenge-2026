"""T4 vs T4 + raw clock features, on the January and July 2025 folds.

Extends src/experiment_v7_vs_t4.py with an extra arm that adds time-of-day
("clock") features derived from the SAME five timestamps the T4 deltas use.
The T4 deltas are kept, never replaced.

V7 features, the traffic feature, the CatBoost configuration, the seed, the
folds and the validation methodology are identical to
src/experiment_v7_vs_t4.py. A drift guard below asserts that the definition
blocks are still byte-identical to that file.

SAFETY: no feature, in any arm, uses BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt.
RAW LABELS: validation targets are never cleaned, clipped or filtered.

Usage:
    python src/experiment_t4_clocks.py --dry-run   # definitions + checks only
    python src/experiment_t4_clocks.py --smoke     # 5 iterations, plumbing
    python src/experiment_t4_clocks.py             # full run
    python src/experiment_t4_clocks.py --with-cyc  # also add the cyclic arm
"""

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

DATA_DIR = Path("data/raw")
SOURCE_SCRIPT = Path("src/experiment_v7_vs_t4.py")
# Predictions are cached so a reporting change never costs another full refit.
CACHE = Path(os.environ.get("TEMP", ".")) / "opencode" / "clocks_results.pkl"

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="run every definition and safety check, never call fit")
parser.add_argument("--smoke", action="store_true", help="5 iterations per arm")
parser.add_argument("--with-cyc", action="store_true",
                    help="add the cyclic (sin/cos) clock arm as well")
parser.add_argument("--report-only", action="store_true",
                    help="reload cached predictions and reprint the reports "
                         "without refitting any model")
parser.add_argument("--folds", nargs="+", default=["JAN", "JUL"], choices=["JAN", "JUL"])
args = parser.parse_args()

# ============================================================
# DEFINITIONS - duplicated from experiment_v7_vs_t4.py on purpose,
# then verified against it by the drift guard below.
# ============================================================

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

T4_SPEC = [
    ("d_AOBT3", "AOBT_3_flt"),
    ("d_EOBT1", "EOBT_1_flt"),
    ("d_IOBT", "IOBT_flt"),
    ("d_LOBT", "LOBT_flt"),
    ("d_SCHED", "SCHED_TIME_UTC_mvt"),
]
T4_FEATURES = [name for name, _ in T4_SPEC]

MINUEND = "MVT_TIME_UTC_mvt"
TARGET = "TAXITIME_SEC_mvt"
FORBIDDEN = ["BLOCK_TIME_UTC_mvt", TARGET]

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

FOLDS = {
    "JAN": ("2025-01-01", "2025-02-01"),
    "JUL": ("2025-07-01", "2025-08-01"),
}

# Exact T4 arm results from the completed run of experiment_v7_vs_t4.py,
# used only as a cross-check that this script reproduces the same baseline.
T4_KNOWN = {"JAN": 404.8803102, "JUL": 513.7559738}

# ============================================================
# CLOCK FEATURE DEFINITIONS
# ============================================================
#
# One entry per T4 timestamp source. tag -> short name, column -> the supplied
# timestamp the time-of-day is read from.
#
# For a source timestamp S with hour H and minute M (both UTC, as stored):
#
#   clock_tod_<tag> = H * 60 + M          minute-of-day, integer 0..1439
#   clock_sin_<tag> = sin(2*pi*tod/1440)  cyclic encoding, -1..1   [Tier B]
#   clock_cos_<tag> = cos(2*pi*tod/1440)  cyclic encoding, -1..1   [Tier B]
#
# Tier A (default) = clock_tod_* only.
# Tier B (--with-cyc) = clock_tod_* + clock_sin_* + clock_cos_*.
#
# Rationale for the cyclic tier: a raw integer minute-of-day is a circle, but a
# tree that splits on it treats 00:00 (0) and 23:59 (1439) as maximally far
# apart, which is wrong. sin/cos places midnight correctly next to 23:59.
#
# These are ADDED to the T4 deltas, never substituted for them.
#
# Note on timezone: every supplied timestamp is stored tz=UTC, so these are UTC
# clock features. That matches V7's existing "hour", which is also UTC-derived.
# Local airport time would need a timezone table the repo does not contain, so
# it is deliberately not attempted; this is recorded as a known limitation.

CLOCK_SPEC = [
    ("SCHED", "SCHED_TIME_UTC_mvt"),
    ("AOBT3", "AOBT_3_flt"),
    ("EOBT1", "EOBT_1_flt"),
    ("IOBT", "IOBT_flt"),
    ("LOBT", "LOBT_flt"),
]

TOD_FEATURES = [f"clock_tod_{tag}" for tag, _ in CLOCK_SPEC]
CYC_FEATURES = [f"clock_{fn}_{tag}" for tag, _ in CLOCK_SPEC
                for fn in ("sin", "cos")]
CLOCK_SOURCES = [col for _, col in CLOCK_SPEC]

# ============================================================
# DRIFT GUARD
# ============================================================

print("=" * 78)
print("EXPERIMENT: T4 vs T4 + CLOCKS   (January / July 2025 holdout folds)")
print("=" * 78)

_v7_block = """V7_FEATURES = [
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
]"""
_t4_block = """T4_SPEC = [
    ("d_AOBT3", "AOBT_3_flt"),
    ("d_EOBT1", "EOBT_1_flt"),
    ("d_IOBT", "IOBT_flt"),
    ("d_LOBT", "LOBT_flt"),
    ("d_SCHED", "SCHED_TIME_UTC_mvt"),
]"""
_src = SOURCE_SCRIPT.read_text(encoding="utf-8")
assert _v7_block in _src, f"V7_FEATURES drifted from {SOURCE_SCRIPT}"
assert _t4_block in _src, f"T4_SPEC drifted from {SOURCE_SCRIPT}"
print("drift guard  : V7_FEATURES and T4_SPEC are byte-identical to "
      f"{SOURCE_SCRIPT}  [OK]")

# ============================================================
# SAFETY CHECKS (run before any model is fitted)
# ============================================================

ARMS = [("T4", V7_FEATURES + T4_FEATURES),
        ("T4+clocks", V7_FEATURES + T4_FEATURES + TOD_FEATURES)]
if args.with_cyc:
    ARMS.append(("T4+clocks_cyc", V7_FEATURES + T4_FEATURES
                 + TOD_FEATURES + CYC_FEATURES))

print("\n" + "=" * 78)
print("SAFETY / UNIT CHECKS")
print("=" * 78)

# 1. forbidden columns
for label, feats in ARMS:
    bad = [f for f in feats if f in FORBIDDEN]
    assert not bad, f"{label} uses forbidden column(s): {bad}"
bad = [c for c in CLOCK_SOURCES if c in FORBIDDEN]
assert not bad, f"clock source uses forbidden column(s): {bad}"
print(f"[PASS] no feature in any arm uses {FORBIDDEN}")

# 2. no name collisions
for label, feats in ARMS:
    assert len(set(feats)) == len(feats), f"{label} has duplicate feature names"
assert not (set(TOD_FEATURES + CYC_FEATURES) & set(V7_FEATURES + T4_FEATURES))
print("[PASS] no duplicate or colliding feature names")

# 3. clock sources present and correctly typed in EVERY file
_files = sorted(DATA_DIR.glob("training_*.parquet")) + [
    DATA_DIR / "ranking.parquet"]
_needed = {MINUEND} | set(CLOCK_SOURCES) | set(c for _, c in T4_SPEC)
for f in _files:
    sch = pq.read_schema(f)
    for col in sorted(_needed):
        assert col in sch.names, f"{f.name}: missing {col}"
        assert str(sch.field(col).type) == "timestamp[us, tz=UTC]", (
            f"{f.name}:{col} dtype is {sch.field(col).type}, "
            f"expected timestamp[us, tz=UTC]")
print(f"[PASS] all {len(_needed)} timestamp sources are timestamp[us, tz=UTC] "
      f"in all {len(_files)} files")

# 4. no clock source is BLOCK_TIME or the target
assert not (set(CLOCK_SOURCES) & set(FORBIDDEN))
print("[PASS] clock sources are SCHED_TIME_UTC_mvt + the four *_flt timestamps "
      "only; BLOCK_TIME_UTC_mvt is not among them")

print("\nfeature counts:")
for label, feats in ARMS:
    print(f"  {label:15s} {len(feats):3d} features "
          f"(V7 {len(V7_FEATURES)} + T4 {len(T4_FEATURES)} + "
          f"clock {len(feats) - len(V7_FEATURES) - len(T4_FEATURES)})")
print()


# ============================================================
# LOAD
# ============================================================

print("Loading training data...")
data = pd.concat([pd.read_parquet(f) for f in sorted(DATA_DIR.glob("training_*.parquet"))],
                 ignore_index=True)
print(f"Total movements: {len(data):,}")


# ============================================================
# TRAFFIC FEATURE (identical to V7)
# ============================================================

print("Creating 15-minute airport traffic feature...")
data["movement_airport"] = data["ADEP_mvt"]
data.loc[data["PHASE_mvt"] == "ARR", "movement_airport"] = \
    data.loc[data["PHASE_mvt"] == "ARR", "ADES_mvt"]
data = data.sort_values(["movement_airport", "MVT_TIME_UTC_mvt"]).copy()
_traffic = (data.set_index("MVT_TIME_UTC_mvt")
            .groupby("movement_airport")["MVT_ID_mvt"]
            .rolling("15min", closed="left").count()
            .reset_index(name="traffic_last_15min"))
data["traffic_last_15min"] = _traffic["traffic_last_15min"].to_numpy()
data["traffic_last_15min"] = data["traffic_last_15min"].fillna(0)

departures = data[data["PHASE_mvt"] == "DEP"].copy()
assert len(departures) == 2_085_047, f"departures {len(departures):,}"
print(f"Departure rows: {len(departures):,}")


# ============================================================
# TIME FEATURES (identical to V7)
# ============================================================

departures["hour"] = departures[MINUEND].dt.hour
departures["day_of_week"] = departures[MINUEND].dt.dayofweek
departures["month"] = departures[MINUEND].dt.month


# ============================================================
# T4 DELTAS
# ============================================================

for name, col in T4_SPEC:
    departures[name] = (departures[MINUEND] - departures[col]).dt.total_seconds()


# ============================================================
# CLOCK FEATURES
# ============================================================

print("\nCreating clock features...")

for tag, col in CLOCK_SPEC:
    # .dt.hour / .dt.minute return float64 with NaN preserved for NaT, so a
    # missing source timestamp yields a missing clock feature. Nothing is
    # filled with 0.
    h = departures[col].dt.hour
    m = departures[col].dt.minute
    tod = h * 60.0 + m
    departures[f"clock_tod_{tag}"] = tod
    ang = 2.0 * np.pi * tod / 1440.0
    departures[f"clock_sin_{tag}"] = np.sin(ang)
    departures[f"clock_cos_{tag}"] = np.cos(ang)

# ============================================================
# UNIT / RANGE / MISSINGNESS CHECKS ON THE REAL DATA
# ============================================================

print("\n" + "-" * 78)
print("Clock feature units, range and missingness over all 2025 departures")
print("-" * 78)
print(f"{'feature':18s} {'dtype':10s} {'nans':>8s} {'nan%':>7s} {'min':>10s} "
      f"{'p50':>10s} {'max':>10s}")
for f in TOD_FEATURES + CYC_FEATURES:
    s = departures[f]
    print(f"{f:18s} {str(s.dtype):10s} {int(s.isna().sum()):8,} "
          f"{100 * s.isna().mean():7.3f} {s.min():10,.4f} "
          f"{s.median():10,.4f} {s.max():10,.4f}")

# range assertions
for tag, col in CLOCK_SPEC:
    tod = departures[f"clock_tod_{tag}"]
    v = tod.dropna()
    assert v.between(0, 1439).all(), f"clock_tod_{tag} out of 0..1439"
    assert np.allclose(v, np.round(v)), f"clock_tod_{tag} is not integral"
    # NaN mask must equal the source NaN mask exactly, and never 0-filled
    assert (tod.isna() == departures[col].isna()).all(), (
        f"clock_tod_{tag} null mask differs from {col}")
    # Independent recomputation from raw epoch microseconds. seconds since epoch
    # mod 86400 = seconds since UTC midnight; //60 = minute of day. This also
    # proves the tz-aware .dt.hour path is not silently shifting the clock.
    ok = departures[col].notna()
    secs = departures.loc[ok, col].astype("int64").to_numpy() // 1_000_000
    manual_tod = (secs % 86400) // 60
    got_tod = tod[ok].to_numpy()
    assert (manual_tod == got_tod).all(), f"clock_tod_{tag} epoch mismatch"
    _mid = int((got_tod == 0).sum())
    for fn in ("sin", "cos"):
        c = departures[f"clock_{fn}_{tag}"]
        cv = c.dropna()
        assert cv.between(-1.0, 1.0).all(), f"clock_{fn}_{tag} outside [-1,1]"
        assert (c.isna() == departures[col].isna()).all(), (
            f"clock_{fn}_{tag} null mask differs from {col}")
    print(f"  clock_tod_{tag:6s} epoch-verified on {int(ok.sum()):,} rows; "
          f"exact-midnight values={_mid}")
print("\n[PASS] every clock_tod_* is integral, within 0..1439, and reproduced "
      "exactly from raw epoch microseconds")
print("[PASS] every clock_sin_*/clock_cos_* is within [-1, 1]")
print("[PASS] every clock null mask is EXACTLY its source column's null mask")
print("[PASS] no clock feature is zero-filled; missing stays missing")

# null pattern grouping
_groups = {}
for tag, col in CLOCK_SPEC:
    _groups.setdefault(tuple(departures[col].isna().to_numpy()), []).append(tag)
print(f"\n  distinct null patterns among the 5 clock sources: {len(_groups)}")
for tags in _groups.values():
    n = int(departures[[c for t, c in CLOCK_SPEC if t in tags][0]].isna().sum())
    print(f"    {tags} -> {n:,} nulls")

# T4 deltas unchanged, verified against integer microsecond epoch arithmetic
print("\n" + "-" * 78)
print("T4 delta unit re-check (independent integer microsecond epoch path)")
print("-" * 78)
for name, col in T4_SPEC:
    vec = departures[name].to_numpy()
    ok = departures[col].notna().to_numpy()
    ua = departures.loc[ok, MINUEND].astype("int64").to_numpy()
    ub = departures.loc[ok, col].astype("int64").to_numpy()
    manual = (ua - ub) / 1_000_000.0
    got = vec[ok]
    assert (manual == got).all(), f"{name} mismatch"
    print(f"  {name:9s} compared={ok.sum():9,}  exact={int((manual == got).sum()):9,}  "
          f"max|diff|={np.abs(manual - got).max():.3e}")
print("[PASS] all five T4 deltas reproduced exactly, in seconds")

# raw labels untouched
print("\n" + "-" * 78)
print("RAW LABEL CHECK (no cleaning, clipping or filtering)")
print("-" * 78)
for fold, (lo, hi) in FOLDS.items():
    h = (departures[MINUEND] >= lo) & (departures[MINUEND] < hi)
    y = departures.loc[h, TARGET]
    assert y.notna().all()
    print(f"  {fold}: n={len(y):,}  mean={y.mean():,.1f}  median={y.median():,.1f}  "
          f"min={y.min():,}  max={y.max():,}  >3600s={int((y > 3600).sum()):,}  "
          f">7200s={int((y > 7200).sum()):,}")
print("[PASS] validation targets used raw; anomalies retained")

# ============================================================
# CATEGORICAL HANDLING (identical to V7)
# ============================================================

for col in V7_CATEGORICALS:
    departures[col] = departures[col].fillna("UNKNOWN").astype(str)

# T4 and clock features must stay numeric; V7's own categoricals are of course
# categorical and are left exactly as V7 defines them.
for f in T4_FEATURES + TOD_FEATURES + CYC_FEATURES:
    assert f not in V7_CATEGORICALS, f"{f} wrongly declared categorical"
    assert pd.api.types.is_numeric_dtype(departures[f]), (
        f"{f} is not numeric: {departures[f].dtype}")
print(f"[PASS] all {len(T4_FEATURES)} T4 and {len(TOD_FEATURES + CYC_FEATURES)} "
      f"clock features remain numeric and non-categorical")


# ============================================================
# FOLD CHECK
# ============================================================

print("\n" + "-" * 78)
print("FOLD CONSTRUCTION")
print("-" * 78)
for fold in args.folds:
    lo, hi = FOLDS[fold]
    h = (departures[MINUEND] >= lo) & (departures[MINUEND] < hi)
    tr, va = departures[~h], departures[h]
    assert int(lo[5:7]) not in set(tr["month"]), "holdout month leaked into train"
    print(f"  {fold}: train={len(tr):,}  validation={len(va):,}  "
          f"val months={sorted(va['month'].unique())}")
print("[PASS] no holdout month appears in its own training set")

if args.dry_run:
    print("\n" + "=" * 78)
    print("DRY RUN COMPLETE - all definitions and safety checks passed.")
    print("No model was trained. Awaiting approval for the full run.")
    print("=" * 78)
    raise SystemExit(0)

if args.report_only:
    if not CACHE.exists():
        raise SystemExit(f"--report-only but no cache at {CACHE}")
    with open(CACHE, "rb") as fh:
        results = pickle.load(fh)
    print(f"[report-only] loaded {sum(len(v) for v in results.values())} "
          f"cached arm/fold fits from {CACHE}")

# ============================================================
# FIT
# ============================================================

def rmse_of(y, pred):
    return root_mean_squared_error(y, pred)


def split_report(y, pred, mask_has_flight):
    """RMSE on rows with and without a flight-level record."""
    out = {}
    for label, sel in [("with_flight_data", mask_has_flight),
                       ("no_flight_data", ~mask_has_flight)]:
        if sel.sum() == 0:
            out[label] = (np.nan, 0)
        else:
            out[label] = (rmse_of(y[sel], pred[sel]), int(sel.sum()))
    return out


def per_airport(y, pred, airports, min_rows=30):
    # np.asarray, not .to_numpy(): these arrive as ndarray from run_fold but
    # as Series elsewhere, and the reporter must not care which.
    d = pd.DataFrame({"a": np.asarray(airports), "y": np.asarray(y),
                      "p": np.asarray(pred)})
    g = d.groupby("a").apply(
        lambda t: pd.Series({
            "n": len(t),
            "rmse": float(np.sqrt(((t["y"] - t["p"]) ** 2).mean())),
            "sse": float(((t["y"] - t["p"]) ** 2).sum()),
        }), include_groups=False)
    return g[g["n"] >= min_rows].sort_values("rmse", ascending=False)


if not args.report_only:
    results = {}
    for fold in args.folds:
        lo, hi = FOLDS[fold]
        h = (departures[MINUEND] >= lo) & (departures[MINUEND] < hi)
        tr, va = departures[~h], departures[h]
        y_tr, y_va = tr[TARGET], va[TARGET]
        # "flight data present" == the flight-level timestamp record exists
        has_flight = va["AOBT_3_flt"].notna().to_numpy()

        print(f"\n{'-' * 78}")
        print(f"FOLD {fold}: holdout {lo} .. {hi}")
        print(f"{'-' * 78}")
        print(f"  train={len(tr):,}  validation={len(va):,}  "
              f"with_flight_data={int(has_flight.sum()):,}  "
              f"no_flight_data={int((~has_flight).sum()):,}")

        for label, feats in ARMS:
            print(f"  arm {label}  ({len(feats)} features)", flush=True)
            params = dict(CB_PARAMS)
            if args.smoke:
                params.update(iterations=5, early_stopping_rounds=5, verbose=0)
            t0 = time.time()
            model = CatBoostRegressor(**params)
            model.fit(tr[feats], y_tr, cat_features=V7_CATEGORICALS,
                      eval_set=(va[feats], y_va))
            el = time.time() - t0
            pred = model.predict(va[feats])
            r = rmse_of(y_va, pred)
            bi = model.get_best_iteration()
            print(f"    {label:14s} RMSE={r:8.2f}  "
                  f"MAE={np.abs(y_va.to_numpy() - pred).mean():7.2f}"
                  f"  best_iter={bi:4d}  fit={el / 60:.1f} min", flush=True)
            results.setdefault(label, {})[fold] = dict(
                rmse=r, pred=pred, y=y_va.to_numpy(),
                sse=float(((y_va.to_numpy() - pred) ** 2).sum()),
                n=len(y_va), best_iter=bi, seconds=el, has_flight=has_flight,
                airports=va["ADEP_mvt"].to_numpy(),
                importance=pd.Series(model.get_feature_importance(), index=feats))

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\n[cache] wrote {sum(len(v) for v in results.values())} "
          f"arm/fold fits to {CACHE}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"{'arm':16s} {'Jan':>9s} {'Jul':>9s} {'pooled':>9s} {'dJan':>9s} "
      f"{'dJul':>9s} {'dPooled':>9s}")
base = results.get("T4", {})
# Report whatever was actually run/cached, in declared order, rather than
# whatever the current flags happen to define - otherwise --report-only would
# silently hide an arm that is sitting in the cache.
_ordered = [lab for lab, _ in ARMS if lab in results]
_ordered += [lab for lab in results if lab not in _ordered]
for label in _ordered:
    r = results[label]
    jan, jul = r["JAN"]["rmse"], r["JUL"]["rmse"]
    # pooled from combined SSE and row counts, NOT an average of the two RMSEs
    sse = r["JAN"]["sse"] + r["JUL"]["sse"]
    n = r["JAN"]["n"] + r["JUL"]["n"]
    pooled = float(np.sqrt(sse / n))
    # cross-check against the concatenated-array definition
    cat_y = np.concatenate([r["JAN"]["y"], r["JUL"]["y"]])
    cat_p = np.concatenate([r["JAN"]["pred"], r["JUL"]["pred"]])
    assert abs(pooled - rmse_of(cat_y, cat_p)) < 1e-9, "pooled RMSE mismatch"
    dj = jan - base["JAN"]["rmse"] if base else np.nan
    dl = jul - base["JUL"]["rmse"] if base else np.nan
    dp = pooled - float(np.sqrt((base["JAN"]["sse"] + base["JUL"]["sse"])
                                / (base["JAN"]["n"] + base["JUL"]["n"]))) if base else np.nan
    print(f"{label:16s} {jan:9.2f} {jul:9.2f} {pooled:9.2f} {dj:+9.2f} {dl:+9.2f} {dp:+9.2f}")

print("\nT4 cross-check against the completed experiment_v7_vs_t4.py run:")
for fold in args.folds:
    got = results["T4"][fold]["rmse"]
    print(f"  {fold}: this run {got:.4f}   prior run {T4_KNOWN[fold]:.4f}   "
          f"diff {got - T4_KNOWN[fold]:+.4f}")

print("\nFLIGHT-DATA vs NO-FLIGHT-DATA RMSE")
print(f"{'arm':16s} {'fold':5s} {'with_flight':>12s} {'n':>9s} "
      f"{'no_flight':>11s} {'n':>7s}")
for label in _ordered:
    for fold in args.folds:
        r = results[label][fold]
        s = split_report(r["y"], r["pred"], r["has_flight"])
        print(f"{label:16s} {fold:5s} {s['with_flight_data'][0]:12.2f} "
              f"{s['with_flight_data'][1]:9,} {s['no_flight_data'][0]:11.2f} "
              f"{s['no_flight_data'][1]:7,}")

print("\nPER-AIRPORT (validation rows, airports with >=30 rows)")
for label in _ordered:
    for fold in args.folds:
        r = results[label][fold]
        g = per_airport(r["y"], r["pred"], r["airports"])
        gbase = None
        if base and fold in base:
            gb = per_airport(base[fold]["y"], base[fold]["pred"],
                             base[fold]["airports"])
            gbase = gb["rmse"]
        print(f"\n  {label} {fold}: {len(g)} airports qualify, "
              f"median RMSE={g['rmse'].median():.1f}, "
              f"worst={g['rmse'].max():.1f}, best={g['rmse'].min():.1f}")
        show = g.head(12).copy()
        if gbase is not None:
            show["T4"] = gbase.reindex(show.index)
            show["delta"] = show["rmse"] - show["T4"]
        print(show.round(2).to_string())

print("\nBEST ITERATION / EARLY STOPPING / RUNTIME")
for label in _ordered:
    bits = "  ".join(
        f"{f}: iter={results[label][f]['best_iter']:4d} "
        f"fit={results[label][f]['seconds'] / 60:5.1f}m"
        for f in args.folds)
    print(f"  {label:16s} {bits}")

print("\nFEATURE IMPORTANCE")
for label in _ordered:
    for fold in args.folds:
        imp = results[label][fold]["importance"].sort_values(ascending=False)
        clock = imp[[i for i in imp.index if i.startswith("clock_")]]
        tot = 100 * clock.sum() / imp.sum()
        print(f"\n  {label} {fold}: clock share {tot:.1f}%  "
              f"(T4 share {100 * imp[[i for i in imp.index if i.startswith('d_')]].sum() / imp.sum():.1f}%)")
        print(clock.round(4).to_string() if len(clock) else "  (no clock features)")

print("\nDone. No V8, no submission, V7 untouched.")
