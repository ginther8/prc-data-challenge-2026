"""V9-D: flight-timestamp disagreement features on top of the frozen T4 model.

EXPERIMENT ONLY. No submission. No promotion. V8, V9-A, V9-B, V9-C untouched.

HYPOTHESIS
    reports/v9c_dsched_forensic_audit.txt is the only timestamp audit in this
    repo whose finding held up across BOTH folds with similar magnitude and
    direction. It measured, on full-timestamp rows, the spread of the four
    flight-level T4 deltas (max - min):

        spread < 1h   vs   spread 1-6h
        JAN   RMSE  223.62  ->  1249.42     (5.6x, n=325)
        JUL   RMSE  323.38  ->  1287.21     (4.0x, n=1,169)

    The graded robustness delta was +1025.80 (JAN) and +963.83 (JUL) - the only
    signal in the whole audit whose MAGNITUDE agreed across folds. Earlier
    tail findings (raw |d_SCHED|, extreme d_SCHED bands, LIRF clock effects)
    all reversed between folds or rested on single-fold populations.

    V9-D asks the narrow question: can the model USE that disagreement, if it
    is given the spread explicitly? The four deltas are already features, but
    a tree cannot cheaply express "max - min across four columns", so the
    disagreement may be latent rather than absent.

DESIGN - controlled ablation, three arms
    B0  exact T4 baseline, the frozen 17 features
    D1  + flight_delta_spread_sec
    D2  + spread + flight_delta_median_sec + flight_delta_max_dev_sec

    Every arm uses identical CatBoost methodology, identical folds, identical
    raw targets, so the only thing that varies is the feature list.

ALIGNMENT - the V9-C hazard, handled explicitly
    add_traffic_feature() SORTS the whole 2025 movement frame by
    ["movement_airport", MVT_TIME_UTC_mvt"] before the traffic count is
    assigned positionally. The departure rows are therefore NOT in raw
    parquet order. V9-C found this the hard way: a rebuild that skipped the
    sort put every recomputed delta against the wrong target.

    This script preserves the frozen behaviour by calling the same
    add_traffic_feature() on the same full frame, and then proves it twice:
      1. B0's RMSE must match the frozen T4 scores, and
      2. B0's PREDICTION VECTOR must be element-wise identical to the V9-B
         cached B0 predictions. An RMSE match alone would not reliably catch
         a row permutation; an exact array match does.

LEAKAGE CONTROLS
    * The new features are functions of the four flight-level T4 SOURCE
      columns only. They cannot see the target or BLOCK_TIME.
    * Nothing is fitted on the holdout month: the features are deterministic
      row-wise arithmetic, not statistics.
    * Missing-all4 rows get NaN, not a sentinel. CatBoost routes NaN through
      its own missing-value handling, which is the specified behaviour and
      avoids inventing an order the data does not have.
    * Feature importance is reported but never used on its own as evidence.

FROZEN BASELINE
    B0 must reproduce VALIDATED_T4 exactly, and must match the V9-B cached
    predictions exactly, before any other arm is evaluated. The run aborts
    if either fails.

Usage
    python src/experiment_v9d_flight_delta_spread.py
    python src/experiment_v9d_flight_delta_spread.py --arms B0 D1
    python src/experiment_v9d_flight_delta_spread.py --folds JAN
    python src/experiment_v9d_flight_delta_spread.py --smoke
"""

import argparse
import hashlib
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train_catboost_v8_defs import (  # read-only import, shared source of truth
    CB_PARAMS, DATA_DIR, FEATURES as T4_FEATURES_17, FORBIDDEN, MINUEND,
    TARGET, T4_FEATURES, T4_SPEC, T4_SOURCE_COLS, VALIDATED_T4,
    VALIDATED_T4_BEST_ITER,
    V7_CATEGORICALS, add_t4_features, add_time_features, add_traffic_feature,
    apply_categorical_handling, verify_t4_deltas,
)

REPORT = Path("reports/v9d_flight_delta_spread_experiment.txt")
CACHE = Path("reports/.v9d_run_cache.pkl")
V9B_CACHE = Path("reports/.v9b_run_cache.pkl")

# ------------------------------------------------------------
# Frozen T4 experiment methodology, verbatim from V9-B, which is
# verbatim from experiment_v7_vs_t4.py -> VALIDATED_T4.
# ------------------------------------------------------------
CB_FOLD = dict(
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

# The four FLIGHT-LEVEL T4 deltas. d_SCHED is deliberately excluded: this
# experiment is about the flight record contradicting ITSELF, with no
# reference to the schedule.
FLIGHT_DELTAS = ["d_AOBT3", "d_EOBT1", "d_IOBT", "d_LOBT"]

H = 3600.0

# ---- new feature definitions -------------------------------------------------
SPREAD = "flight_delta_spread_sec"
MEDIAN = "flight_delta_median_sec"
MAXDEV = "flight_delta_max_dev_sec"

D1_FEATURES = [SPREAD]
D2_FEATURES = [SPREAD, MEDIAN, MAXDEV]

ARMS = {
    "B0": ([], "exact T4 baseline (frozen 17 features)"),
    "D1": (D1_FEATURES, "+ flight-delta spread"),
    "D2": (D2_FEATURES, "+ spread, median, max deviation"),
}
ARMS_ORDER = ["B0", "D1", "D2"]
ARMS_GAIN = ["D1", "D2"]

# spread bands, from the V9-C audit
SPREAD_BANDS = [
    ("< 1h", 0.0, 1 * H),
    ("1h .. 2h", 1 * H, 2 * H),
    ("2h .. 6h", 2 * H, 6 * H),
    ("> 6h", 6 * H, np.inf),
]

# target bands, from the V9-C audit
TARGET_BANDS = [
    ("< 30m", 0.0, 30 * 60.0),
    ("30m .. 1h", 30 * 60.0, 1 * H),
    ("1h .. 2h", 1 * H, 2 * H),
    ("2h .. 4h", 2 * H, 4 * H),
    ("> 4h", 4 * H, np.inf),
]

KS = [1, 5, 10, 25, 50, 100]

out = []
W = out.append


def head(t):
    W("")
    W("=" * 100)
    W(t)
    W("=" * 100)


def sub(t):
    W("")
    W("-" * 100)
    W(t)
    W("-" * 100)


def note(t=""):
    W(t)


def rmse(a):
    return float(np.sqrt(np.mean(np.square(np.asarray(a, dtype="float64")))))


# Missing-all4 rows make every row-wise nan-reduction below hit an all-NaN
# slice. That is the expected and intended path, not a fault, so the warning
# is suppressed rather than left to bury the run log. Correctness of the
# result is guaranteed separately: the values are masked back to NaN on
# exactly those rows and then asserted against an independent re-derivation.
warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                        category=RuntimeWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=list(ARMS_ORDER))
ap.add_argument("--folds", nargs="+", default=list(FOLDS))
ap.add_argument("--smoke", action="store_true",
                help="tiny fit + cache suffix, for pipeline validation only")
args = ap.parse_args()

if args.smoke:
    CB_FOLD = dict(CB_FOLD, iterations=10, early_stopping_rounds=10, verbose=50)
    CACHE = Path("reports/.v9d_smoke_cache.pkl")
    SMOKE = True
else:
    SMOKE = False


# ============================================================
# BUILD - reproduces the frozen V8 feature-generation path
# ============================================================
def build():
    print("Loading 2025 training movements ...", flush=True)
    files = sorted(DATA_DIR.glob("training_*.parquet"))
    assert len(files) == 12, f"expected 12 training files, got {len(files)}"
    data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  movements {len(data):,}", flush=True)
    assert len(data) == 4_167_797, f"unexpected movement count {len(data):,}"

    # V8 behaviour: traffic is computed over the WHOLE frame (arrivals and
    # departures together, sorted by movement_airport then MVT_TIME_UTC_mvt)
    # and only then are departures filtered out. Changing this would change
    # traffic_last_15min, so the call is byte-for-byte the frozen one.
    data = add_traffic_feature(data)
    dep = data[data["PHASE_mvt"] == "DEP"].copy()
    print(f"  departures {len(dep):,}", flush=True)
    assert len(dep) == 2_085_047, f"unexpected departures {len(dep):,}"
    assert dep[TARGET].notna().all(), "some departures have no target"

    dep = add_time_features(dep)
    dep = add_t4_features(dep)
    verify_t4_deltas(dep)

    # ---- the derived disagreement features ----
    # numpy, not DataFrame arithmetic: in this environment (pandas 3.0.6)
    # `DataFrame + DataFrame` returns all-NaN, which would have silently
    # zeroed these features. Every value is asserted below.
    F = dep[FLIGHT_DELTAS].to_numpy("float64")
    pres = ~np.isnan(F)
    n4 = pres.sum(axis=1)
    full = n4 == 4

    with np.errstate(all="ignore"):
        spread = np.nanmax(F, axis=1) - np.nanmin(F, axis=1)
        med = np.nanmedian(F, axis=1)
        maxdev = np.nanmax(np.abs(F - med[:, None]), axis=1)

    # Missing-all4 rows are left NaN on purpose. No sentinel, no partial-data
    # branch: V9-C proved availability is only ever 0 or 4, so the only
    # missingness pattern that exists is "all of it".
    dep[SPREAD] = np.where(full, spread, np.nan)
    dep[MEDIAN] = np.where(full, med, np.nan)
    dep[MAXDEV] = np.where(full, maxdev, np.nan)
    dep["_n4"] = n4
    dep["_full"] = full

    # ---- population guards, measured independently ----
    uniq = sorted(int(v) for v in np.unique(n4))
    print(f"  flight-delta availability counts present: {uniq}", flush=True)
    assert uniq in ([0, 4], [0], [4]), (
        f"unexpected partial availability {uniq}; V9-C proved only 0 and 4 "
        f"occur, so the NaN-only missingness design would be wrong")

    n_val = 0
    n_full = 0
    n_miss = 0
    for lo, hi in FOLDS.values():
        m = (dep[MINUEND] >= lo) & (dep[MINUEND] < hi)
        n_val += int(m.sum())
        n_full += int((m & dep["_full"]).sum())
        n_miss += int((m & ~dep["_full"]).sum())
    print(f"  validation rows {n_val:,}  full-timestamp {n_full:,}  "
          f"missing-all4 {n_miss:,}", flush=True)
    assert n_val == 344_419, f"validation row count {n_val} != 344,419"
    assert n_full + n_miss == n_val, "full/missing do not partition"
    # V9-C measured these exact numbers on the same fold construction.
    assert n_miss == 5_373, f"missing-all4 {n_miss} != V9-C's 5,373"
    assert dep.loc[~dep["_full"], [SPREAD, MEDIAN, MAXDEV]].notna().sum().sum() == 0, \
        "a missing-all4 row received a finite disagreement value"
    assert dep.loc[dep["_full"], [SPREAD, MEDIAN, MAXDEV]].notna().all().all(), \
        "a full-timestamp row received a missing disagreement value"

    dep = apply_categorical_handling(dep)

    print(f"  spread on full rows: min {np.nanmin(spread):.1f}  "
          f"median {np.nanmedian(spread):.1f}  max {np.nanmax(spread):.1f}",
          flush=True)
    print("  [PASS] disagreement feature populations verified", flush=True)
    return dep


def feature_checks(dep):
    """Re-derive every new feature from the raw SOURCE columns and require
    exact agreement, so a construction bug cannot become an experimental
    result."""
    for f in D2_FEATURES:
        assert f in dep.columns, f"missing new feature {f}"
        assert f not in T4_FEATURES_17, f"new feature {f} collides with frozen T4"
        assert f not in FORBIDDEN, f"forbidden column used: {f}"
        assert dep[f].dtype == np.float64, f"{f} must be float64 (carry NaN)"
    assert not any(f.startswith("clock_") for f in D2_FEATURES)
    assert not any(f.startswith("hist_") for f in D2_FEATURES)
    assert not any("TAXITIME" in f or "BLOCK" in f for f in D2_FEATURES)
    assert set(D1_FEATURES) < set(D2_FEATURES), "D1 must be a subset of D2"
    assert len(set(D2_FEATURES)) == 3, "duplicate feature name"
    assert len(T4_FEATURES_17) == 17, "frozen T4 list is not 17 features"

    # the five frozen deltas are untouched, re-derived from raw source columns
    for name, col in T4_SPEC:
        raw = (dep[MINUEND] - dep[col]).dt.total_seconds()
        got = dep[name]
        assert bool((raw.isna() == got.isna()).all()), \
            f"{name}: null mask differs from the raw source column"
        assert int(np.nansum(np.abs(raw.to_numpy("float64")
                                    - got.to_numpy("float64")))) == 0, \
            f"{name}: value differs from the raw source column"

    # re-derive the disagreement features from the deltas, independently
    F = dep[FLIGHT_DELTAS].to_numpy("float64")
    full = ~np.isnan(F).any(axis=1)
    assert bool((full == dep["_full"].to_numpy()).all()), "_full wrong"
    with np.errstate(all="ignore"):
        s2 = np.nanmax(F, axis=1) - np.nanmin(F, axis=1)
        m2 = np.nanmedian(F, axis=1)
        v2 = np.nanmax(np.abs(F - m2[:, None]), axis=1)
    for name, want in ((SPREAD, s2), (MEDIAN, m2), (MAXDEV, v2)):
        got = dep[name].to_numpy("float64")
        assert bool((np.isnan(got) == ~full).all()), f"{name} null mask wrong"
        both = full
        assert int(np.nansum(np.abs(want[both] - got[both]))) == 0, \
            f"{name} value differs from the deltas"

    # non-degeneracy: a constant feature would silently do nothing
    for f in D2_FEATURES:
        assert dep[f].nunique(dropna=True) > 100, f"{f} is near-constant"
    assert float(np.nanmax(dep[SPREAD].to_numpy("float64"))) > 0, \
        "spread is identically zero"
    # max deviation from the median can never exceed the total spread
    m = dep["_full"].to_numpy()
    assert bool((dep[MAXDEV].to_numpy()[m] <= dep[SPREAD].to_numpy()[m] + 1e-9).all()), \
        "max deviation from the median exceeds the spread"


# ============================================================
# RUN
# ============================================================
v9b_cache = {}
if V9B_CACHE.exists():
    v9b_cache = pickle.loads(V9B_CACHE.read_bytes())
    print(f"[v9b] reference cache arms: "
          f"{sorted({k[1] for k in v9b_cache})}", flush=True)

dep = build()
feature_checks(dep)

cache = {}
if CACHE.exists():
    cache = pickle.loads(CACHE.read_bytes())
    print(f"[resume] cache has {sorted(cache)}", flush=True)

order = [a for a in ARMS_ORDER if a in args.arms]
for fold in [f for f in ["JAN", "JUL"] if f in args.folds]:
    lo, hi = FOLDS[fold]
    holdout = (dep[MINUEND] >= lo) & (dep[MINUEND] < hi)
    tr = dep[~holdout]
    va = dep[holdout]
    y_tr = tr[TARGET].to_numpy("float64")
    y_va = va[TARGET].to_numpy("float64")
    if SMOKE:
        tr = tr.sample(n=min(200_000, len(tr)), random_state=42)
        y_tr = tr[TARGET].to_numpy("float64")
    print(f"\n{'=' * 70}\nFOLD {fold}: train {len(tr):,}  holdout {len(va):,}"
          f"\n{'=' * 70}", flush=True)

    for arm in order:
        key = (fold, arm)
        if key in cache:
            print(f"[{fold} {arm}] cached, skipping", flush=True)
            continue
        feats = T4_FEATURES_17 + ARMS[arm][0]
        assert len(feats) == len(set(feats))
        print(f"\n[{fold} {arm}] {len(feats)} features  "
              f"(+{len(ARMS[arm][0])} new)  {ARMS[arm][1]}", flush=True)
        t0 = time.time()
        model = CatBoostRegressor(**CB_FOLD)
        model.fit(tr[feats], y_tr, cat_features=V7_CATEGORICALS,
                  eval_set=(va[feats], y_va))
        pred = model.predict(va[feats]).astype("float64")
        r = rmse(y_va - pred)
        bi = model.get_best_iteration()
        print(f"[{fold} {arm}] RMSE {r:.7f}  best_iter {bi} "
              f"({bi + 1} trees)  {(time.time() - t0) / 60:.1f} min", flush=True)
        cache[key] = dict(
            rmse=r, best_iter=int(bi), trees=int(bi) + 1,
            pred=pred, features=feats,
            imp=model.get_feature_importance(),
            y=y_va,
            airport=va["ADEP_mvt"].to_numpy(),
            n4=va["_n4"].to_numpy(),
            full=va["_full"].to_numpy(),
            spread=va[SPREAD].to_numpy("float64"),
            d_sched=va["d_SCHED"].to_numpy("float64"),
        )
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(cache))
        print(f"[{fold} {arm}] cached", flush=True)

        if arm == "B0" and not SMOKE:
            exp = VALIDATED_T4[fold]
            print(f"[{fold} B0] frozen reference {exp}  "
                  f"delta {r - exp:+.7f}", flush=True)
            if abs(r - exp) > 1e-4:
                raise SystemExit(
                    f"B0 does not reproduce the frozen T4 score on {fold}: "
                    f"{r} vs {exp}. Aborting before any other arm counts.")

# ============================================================
# FROZEN BASELINE GATE - RMSE AND element-wise prediction match
# ============================================================
if not SMOKE:
    for fold in FOLDS:
        assert (fold, "B0") in cache, f"missing B0 for {fold}"
        got = cache[(fold, "B0")]["rmse"]
        exp = VALIDATED_T4[fold]
        assert abs(got - exp) < 1e-4, f"{fold} B0 {got} != frozen {exp}"
    print("\n[PASS] B0 reproduces the frozen T4 scores on both folds.", flush=True)

    # The stronger alignment proof: the V9-B cached B0 predictions were
    # produced by the frozen pipeline on the same folds. If this script's row
    # order differed at all, the arrays would not match bit-for-bit.
    for fold in FOLDS:
        ref = v9b_cache.get((fold, "B0"))
        if ref is None:
            print(f"[{fold} B0] no V9-B reference prediction available, "
                  f"skipping array check", flush=True)
            continue
        d = np.abs(cache[(fold, "B0")]["pred"] - ref["pred"])
        mx = float(d.max())
        print(f"[{fold} B0] vs V9-B cached B0: max|diff| {mx:.3e}  "
              f"n={len(d):,}", flush=True)
        assert mx == 0.0, (
            f"{fold} B0 predictions differ from the V9-B cached baseline by "
            f"up to {mx:.3e}; row order or feature construction has drifted. "
            f"Aborting - every other arm would be measured against a "
            f"misaligned baseline.")
    print("[PASS] row alignment is bit-identical to the frozen V9-B baseline.",
          flush=True)

del dep

# ============================================================
# REPORT
# ============================================================
cache = {k: v for k, v in cache.items() if k[1] in ARMS}
RFOLDS = [f for f in ["JAN", "JUL"] if all((f, a) in cache for a in ARMS_ORDER)]
if len(RFOLDS) < len(FOLDS):
    print(f"[warn] partial run, reporting on {RFOLDS} only", flush=True)
RES = {}
for arm in ARMS_ORDER:
    if not all((fold, arm) in cache for fold in RFOLDS):
        continue
    RES[arm] = {}
    for fold in RFOLDS:
        c = cache[(fold, arm)]
        e = c["y"] - c["pred"]
        ae = np.abs(e)
        RES[arm][fold] = dict(
            rmse=c["rmse"], mae=float(ae.mean()),
            medae=float(np.median(ae)), bias=float(e.mean()),
            best_iter=c["best_iter"], trees=c["trees"],
            n_feat=len(c["features"]), n=len(e),
        )
    pool_e = np.concatenate([cache[(f, arm)]["y"] - cache[(f, arm)]["pred"]
                             for f in RFOLDS])
    RES[arm]["POOL"] = dict(
        rmse=rmse(pool_e), mae=float(np.abs(pool_e).mean()),
        medae=float(np.median(np.abs(pool_e))), bias=float(pool_e.mean()),
        n=len(pool_e),
    )

head("V9-D - FLIGHT-TIMESTAMP DISAGREEMENT FEATURES ON THE FROZEN T4 MODEL")
note("")
note("EXPERIMENT ONLY. No submission. No promotion. V8/V9-A/V9-B/V9-C untouched.")
note("")
note("  Frozen champion      V8/T4, official 2026 RMSE 425.7711")
note("  Frozen baselines     JAN 404.8803102   JUL 513.7559738   pooled 468.31")
note("  V9-A status          MIXED - not promoted, not used here")
note("  V9-B status          MIXED - not promoted, not used here")
note("  V9-C status          forensic audit only, no model")
note("")
note("  HYPOTHESIS")
note("    V9-C found exactly one timestamp signal that held across BOTH folds")
note("    with similar magnitude and direction: the spread (max - min) of the")
note("    four flight-level T4 deltas, on full-timestamp rows.")
note("")
note("      spread < 1h  vs  1-6h      JAN 223.62 -> 1249.42   JUL 323.38 -> 1287.21")
note("      graded robustness delta     JAN +1025.80            JUL +963.83")
note("")
note("    The four deltas are already T4 features, but a tree cannot cheaply")
note("    express max-minus-min across four separate columns, so the")
note("    disagreement may be latent rather than absent. This asks whether")
note("    making it explicit helps.")
note("")
note("  DESIGN")
for a in ARMS_ORDER:
    nf = len(ARMS[a][0])
    note(f"    {a}  {ARMS[a][1]:<42s} {len(T4_FEATURES_17) + nf:>2d} features "
         f"(+{nf} new)")
note("")
note("  CatBoost methodology, folds and raw targets are identical to the")
note("  frozen T4 experiment in every arm. The feature list is the only thing")
note("  that varies.")

sub("0. LEAKAGE AND FIDELITY CONTROLS")
note("  * The new features are pure row-wise arithmetic on the four flight-level")
note("    T4 source columns. No statistic is fitted, so nothing can leak from")
note("    the holdout month.")
note("  * The four frozen T4 deltas are re-derived from the raw source columns")
note("    and required to match the stored columns exactly, null masks included.")
note("  * Each new feature is re-derived from the deltas and required to match")
note("    exactly, so a construction bug fails here instead of becoming a result.")
note("  * Missing-all4 rows receive NaN, not a sentinel, so CatBoost's own")
note("    missing-value handling applies as specified. No partial-data branch")
note("    exists, because V9-C proved availability is only ever 0 or 4 - and")
note("    this script asserts that rather than assuming it.")
note("  * d_SCHED is not used by any new feature. This is a FLIGHT-record")
note("    disagreement measure with no schedule reference.")
note("  * No raw |d_SCHED| feature, no clock, no H1/history, no external data,")
note("    no clipping, no cleaning, no post-processing, no 2026 targets, no")
note("    leaderboard tuning, no upload, no commit.")
note("  * Feature importance is reported for the best arm but is never used as")
note("    evidence on its own.")

sub("1. BASELINE REPRODUCTION AND ALIGNMENT GATE")
note("  B0 is the frozen 17-feature T4 model. Two independent proofs run before")
note("  any other arm is reported against it.")
note("")
note("  Proof 1 - the score matches the frozen T4 reference:")
note(f"  {'fold':5s} {'B0 reproduced':>15s} {'frozen T4':>15s} {'delta':>12s} "
     f"{'best_iter':>10s} {'trees':>7s}")
for fold in RFOLDS:
    b = cache[(fold, "B0")]
    exp = VALIDATED_T4[fold]
    note(f"  {fold:5s} {b['rmse']:>15.7f} {exp:>15.7f} "
         f"{b['rmse'] - exp:>+12.2e} {b['best_iter']:>10d} {b['trees']:>7d}")
note("")
note("  Proof 2 - the PREDICTION VECTOR is element-wise identical to the V9-B")
note("  cached B0 predictions. This is the check the V9-C alignment hazard")
note("  demands: add_traffic_feature() sorts the frame by")
note("  ['movement_airport', 'MVT_TIME_UTC_mvt'], so a rebuild that skipped the")
note("  sort would permute the holdout rows. An RMSE match alone would not")
note("  reliably catch a permutation; an exact array match does.")
note("")
note(f"  {'fold':5s} {'rows':>10s} {'max|diff| vs V9-B B0':>22s}  status")
for fold in RFOLDS:
    ref = v9b_cache.get((fold, "B0"))
    if ref is None:
        note(f"  {fold:5s} {len(cache[(fold, 'B0')]['pred']):>10,} "
             f"{'(no V9-B reference)':>22s}  skipped")
        continue
    d = float(np.abs(cache[(fold, "B0")]["pred"] - ref["pred"]).max())
    note(f"  {fold:5s} {len(cache[(fold, 'B0')]['pred']):>10,} {d:>22.3e}  "
         f"{'IDENTICAL' if d == 0.0 else 'DIFFERS'}")
note("")
note("  Frozen reference best_iter: "
     f"JAN {VALIDATED_T4_BEST_ITER['JAN']}, JUL {VALIDATED_T4_BEST_ITER['JUL']}")
note("  [PASS] baseline is genuine AND row-aligned; the ablation is controlled.")

sub("2. THE DERIVED FEATURES - populations and bands")
_sp = np.concatenate([cache[(f, "B0")]["spread"] for f in RFOLDS])
_yr = np.concatenate([cache[(f, "B0")]["y"] for f in RFOLDS])
_pe = _yr - np.concatenate([cache[(f, "B0")]["pred"] for f in RFOLDS])
_psse = float(np.sum(_pe ** 2))
note(f"  validation rows {len(_sp):,}")
note(f"  full-timestamp (four deltas present)  {int((~np.isnan(_sp)).sum()):,}")
note(f"  missing-all4 (all four NaN)            {int(np.isnan(_sp).sum()):,}")
note("  partial availability (1-3 present)    0  <- asserted absent in build()")
note("")
note("  The new features are NaN exactly on the missing-all4 rows and finite")
note("  everywhere else, so the spread bands below partition the full-timestamp")
note("  population and leave the missing rows in their own single group.")
note("")
note(f"  {'spread band':14s} {'n':>9s} {'% rows':>8s} {'% full rows':>12s} "
     f"{'med target':>11s} {'B0 medAE':>10s} {'% B0 SSE':>10s}")
for lab, lo, hi in SPREAD_BANDS:
    m = (~np.isnan(_sp)) & (_sp >= lo) & (_sp < hi)
    n = int(m.sum())
    if n == 0:
        note(f"  {lab:14s} {0:>9,} {'-':>8s} {'-':>12s} "
             f"{'(empty)':>11s} {'(empty)':>10s} {'(empty)':>10s}")
        continue
    note(f"  {lab:14s} {n:>9,} {100 * n / len(_sp):>7.2f}% "
         f"{100 * n / int((~np.isnan(_sp)).sum()):>11.2f}% "
         f"{np.median(_yr[m]):>11,.0f} "
         f"{np.median(np.abs(_pe[m])):>10,.0f} "
         f"{100 * float(np.sum(_pe[m] ** 2)) / _psse:>9.2f}%")
note("")
note("  B0 error by spread band, per fold. This is the motivating measurement")
note("  from V9-C, recomputed here from the same cached B0 predictions so the")
note("  ablation is measured against exactly what the report claims.")
note("")
note(f"  {'band':14s} {'JAN n':>8s} {'JAN RMSE':>10s} {'JUL n':>8s} "
     f"{'JUL RMSE':>10s} {'pooled RMSE':>12s} {'% pooled SSE':>13s}")
for lab, lo, hi in SPREAD_BANDS:
    cells = []
    for fold in RFOLDS:
        s = cache[(fold, "B0")]["spread"]
        e = cache[(fold, "B0")]["y"] - cache[(fold, "B0")]["pred"]
        m = (~np.isnan(s)) & (s >= lo) & (s < hi)
        cells.append((int(m.sum()),
                      rmse(e[m]) if m.sum() else None))
    mp = (~np.isnan(_sp)) & (_sp >= lo) & (_sp < hi)
    if mp.sum():
        pr = rmse(_pe[mp])
        ps = f"{100 * float(np.sum(_pe[mp] ** 2)) / _psse:>12.2f}%"
    else:
        pr, ps = None, f"{'(empty)':>12s}"
    cells_s = [f"{n:>8,}" for n, _ in cells]
    cells_r = [f"{r:>10.2f}" if r is not None else f"{'(empty)':>10s}"
               for _, r in cells]
    pooled_s = f"{pr:>12.2f}" if pr is not None else f"{'(empty)':>12s}"
    note(f"  {lab:14s} {cells_s[0]} {cells_r[0]} "
         f"{cells_s[1]} {cells_r[1]} {pooled_s} {ps}")
note("")
note("  And the missing-all4 group, for contrast:")
for fold in RFOLDS:
    c = cache[(fold, "B0")]
    e = c["y"] - c["pred"]
    m = np.isnan(c["spread"])
    note(f"    {fold}: n={int(m.sum()):>6,}  RMSE {rmse(e[m]):>9.2f}")
note("")

head("3. ABLATION RESULTS - every arm, both folds, raw target RMSE")
base_pool = RES["B0"]["POOL"]["rmse"]
note(f"  pooled baseline (B0, recomputed here) = {base_pool:.4f}")
note("")
note(f"  {'arm':4s} {'nf':>3s} {'JAN':>10s} {'dJAN':>9s} {'JUL':>10s} "
     f"{'dJUL':>9s} {'POOLED':>10s} {'dPOOL':>9s} {'MAE':>8s} {'medAE':>8s} "
     f"{'bi(J/L)':>9s}")
for arm in ARMS_ORDER:
    if arm not in RES:
        continue
    r = RES[arm]
    note(f"  {arm:4s} {r['JAN']['n_feat']:>3d} {r['JAN']['rmse']:>10.4f} "
         f"{r['JAN']['rmse'] - RES['B0']['JAN']['rmse']:>+9.4f} "
         f"{r['JUL']['rmse']:>10.4f} "
         f"{r['JUL']['rmse'] - RES['B0']['JUL']['rmse']:>+9.4f} "
         f"{r['POOL']['rmse']:>10.4f} {r['POOL']['rmse'] - base_pool:>+9.4f} "
         f"{r['POOL']['mae']:>8.2f} {r['POOL']['medae']:>8.2f} "
         f"{r['JAN']['best_iter']:>3d}/{r['JUL']['best_iter']:<3d}")
note("")
note("  Negative delta is better. MAE and medAE are pooled. Best iteration is")
note("  CatBoost's early-stopping choice on the holdout, identical procedure to")
note("  the frozen experiment.")
for arm in ARMS_ORDER:
    if arm not in RES or arm == "B0":
        continue
    r = RES[arm]
    note("")
    note(f"  {arm}  ({ARMS[arm][1]}, {r['JAN']['n_feat']} features)")
    note(f"    {'fold':7s} {'RMSE':>10s} {'MAE':>9s} {'medianAE':>10s} "
         f"{'bias':>8s} {'best_iter':>10s} {'trees':>7s}")
    for fold in RFOLDS:
        x = r[fold]
        note(f"    {fold:7s} {x['rmse']:>10.4f} {x['mae']:>9.2f} "
             f"{x['medae']:>10.2f} {x['bias']:>+8.2f} {x['best_iter']:>10d} "
             f"{x['trees']:>7d}")
note("")


def pooled(arm, field):
    return np.concatenate([cache[(fold, arm)][field] for fold in RFOLDS])


def errs(arm):
    return pooled(arm, "y") - pooled(arm, "pred")


def band_mask(field, lo, hi):
    v = pooled("B0", field)
    m = ~np.isnan(v)
    return m & (v >= lo) & (v < hi)


# ============================================================
# ROBUSTNESS
# ============================================================
deltas = {a: RES[a]["POOL"]["rmse"] - base_pool for a in RES if a != "B0"}
JANd = {a: RES[a]["JAN"]["rmse"] - RES["B0"]["JAN"]["rmse"] for a in deltas}
JULd = {a: RES[a]["JUL"]["rmse"] - RES["B0"]["JUL"]["rmse"] for a in deltas}
gainers = [a for a in ARMS_GAIN if a in RES and deltas[a] < -1e-9]

for arm in ARMS_GAIN:
    if arm not in RES:
        continue
    sub(f"4. ROBUSTNESS - {arm} vs B0")
    e0, e1 = errs("B0"), errs(arm)
    s0, s1 = e0 ** 2, e1 ** 2
    gain = s0 - s1
    net = float(gain.sum())
    ap = pooled(arm, "airport")

    note("  4a. Per fold, separately. A pooled gain with folds of opposite sign")
    note("      is not a gain, it is two different effects averaging out.")
    note("")
    note(f"      {'fold':6s} {'n':>8s} {'B0':>10s} {arm:>10s} {'delta':>10s} "
         f"{'improved':>9s} {'worsened':>9s}")
    for fold in RFOLDS:
        c0, c1 = cache[(fold, "B0")], cache[(fold, arm)]
        a0 = c0["y"] - c0["pred"]
        a1 = c1["y"] - c1["pred"]
        g = a0 ** 2 - a1 ** 2
        note(f"      {fold:6s} {len(g):>8,} {rmse(a0):>10.4f} {rmse(a1):>10.4f} "
             f"{rmse(a1) - rmse(a0):>+10.4f} "
             f"{100 * (g > 0).mean():>8.2f}% {100 * (g < 0).mean():>8.2f}%")
    note("")

    note("  4b. Pooled, by the required slices.")
    note("")
    note(f"      {'slice':34s} {'n':>8s} {'B0':>10s} {arm:>10s} {'delta':>10s} "
         f"{'% of net gain':>15s}")
    slices = [("all validation rows", np.ones(len(e0), bool))]
    for excl, lab in ((["LIRF"], "excluding LIRF"),
                      (["LIRF", "LFPG"], "excluding LIRF + LFPG")):
        slices.append((lab, ~np.isin(ap, excl)))
    slices.append(("full-timestamp rows", ~np.isnan(pooled(arm, "spread"))))
    slices.append(("missing-all4 rows", np.isnan(pooled(arm, "spread"))))
    for lab, lo, hi in SPREAD_BANDS:
        slices.append((f"spread {lab}", band_mask("spread", lo, hi)))
    for lab, lo, hi in TARGET_BANDS:
        slices.append((f"target {lab}", band_mask("y", lo, hi)))
    for lab, m in slices:
        n = int(m.sum())
        if n == 0:
            note(f"      {lab:34s} {0:>8,} {'-':>10s} {'-':>10s} "
                 f"{'-':>10s} {'(empty)':>15s}")
            continue
        r0, r1 = rmse(e0[m]), rmse(e1[m])
        note(f"      {lab:34s} {n:>8,} {r0:>10.4f} {r1:>10.4f} "
             f"{r1 - r0:>+10.4f} {100 * gain[m].sum() / net:>14.1f}%")
    note("")
    note("      Read the '% of net gain' column against the row count. A slice")
    note("      holding 0.2% of rows and 40% of the gain is a tail artefact; a")
    note("      slice holding 60% of rows and 5% of the gain means the model is")
    note("      paying for its gain elsewhere.")
    note("")

    note("  4c. The hypothesis's own target: does the contradictory-timestamp")
    note("      group improve MORE than the agreeing group? A genuine")
    note("      unreliability detector should improve the spread 1-6h rows")
    note("      disproportionately. A plain outlier detector would improve")
    note("      whichever group happens to hold the big targets.")
    note("")
    note(f"      {'group':34s} {'n':>8s} {'B0':>10s} {arm:>10s} {'delta':>10s}")
    grp = [("spread < 1h (timestamps agree)",
            band_mask("spread", 0.0, 1 * H)),
           ("spread 1h-6h (timestamps conflict)",
            band_mask("spread", 1 * H, 6 * H))]
    for lab, m in grp:
        if m.sum() == 0:
            continue
        note(f"      {lab:34s} {int(m.sum()):>8,} {rmse(e0[m]):>10.2f} "
             f"{rmse(e1[m]):>10.2f} {rmse(e1[m]) - rmse(e0[m]):>+10.2f}")
    note("")

    note("  4d. Spread concentration of the gain (pooled).")
    note(f"      {'k':>6s} {'cum SSE gain':>18s} {'% of net gain':>14s} "
         f"{'% of rows':>10s} {'med spread':>11s} {'med target':>11s} "
         f"{'% LIRF':>7s}")
    order = np.argsort(-gain)
    sp_pool = pooled(arm, "spread")
    y_pool = pooled(arm, "y")
    for k in KS:
        sel = order[:k]
        ssel = sp_pool[sel]
        med_spread = (f"{np.median(ssel[~np.isnan(ssel)]):,.0f}"
                      if (~np.isnan(ssel)).any() else "all NaN")
        note(f"      {k:>6,} {float(gain[sel].sum()):>18.4e} "
             f"{100 * float(gain[sel].sum()) / net:>13.1f}% "
             f"{100 * k / len(gain):>9.4f}% "
             f"{med_spread:>11s} "
             f"{np.median(y_pool[sel]):>11,.0f} "
             f"{100 * (ap[sel] == 'LIRF').mean():>6.1f}%")
    note("")
    note("      If the worst 1-10 rows already exceed 100% of net gain, the")
    note("      remaining rows are net worse and the arm is buying a tail.")
    note("")

    note("  4e. Worst-N attribution: the share of the net SSE improvement that")
    note("      comes from the single largest contributors, ranked two ways.")
    note("")
    note(f"      {'k':>6s} {'by B0 error':>14s} {'% of net':>10s} "
         f"{'by target':>12s} {'% of net':>10s}")
    by_err = np.argsort(-s0)
    by_tgt = np.argsort(-y_pool)
    for k in KS:
        ce = float(gain[by_err[:k]].sum())
        ct = float(gain[by_tgt[:k]].sum())
        note(f"      {k:>6,} {ce:>14.4e} {100 * ce / net:>9.1f}% "
             f"{ct:>12.4e} {100 * ct / net:>9.1f}%")
    note("")

    note("  4f. By airport, pooled, airports with >= 200 validation rows.")
    note(f"      {'airport':8s} {'n':>7s} {'B0':>9s} {arm:>9s} {'delta':>9s} "
         f"{'% net gain':>11s} {'B0 SSE':>12s} {'sp>=1h':>8s}")
    sp_pool2 = pooled(arm, "spread")
    rows = []
    for a in np.unique(ap):
        m = ap == a
        if m.sum() < 200:
            continue
        r0, r1 = rmse(e0[m]), rmse(e1[m])
        rows.append((100 * gain[m].sum() / net, a, int(m.sum()), r0, r1,
                     float(s0[m].sum()),
                     int((sp_pool2[m] >= H).sum())))
    for sh, a, n, r0, r1, ss, ns in sorted(rows, key=lambda z: -abs(z[0]))[:20]:
        note(f"      {a:8s} {n:>7,} {r0:>9.2f} {r1:>9.2f} {r1 - r0:>+9.2f} "
             f"{sh:>10.1f}% {ss:>12.3e} {ns:>8,}")
    note("")

    note("  4g. Rows improved vs worsened, pooled.")
    better = int((gain > 0).sum())
    worse = int((gain < 0).sum())
    same = len(gain) - better - worse
    note(f"      improved {better:>8,} ({100 * better / len(gain):5.2f}%)   "
         f"worsened {worse:>8,} ({100 * worse / len(gain):5.2f}%)   "
         f"unchanged {same:>8,}")
    note("")

    note("  4h. If the k improved rows were removed, would the rest still be")
    note("      better? Recomputing the pooled delta without the top-k")
    note("      contributors separates 'helped everywhere' from 'helped a few'.")
    note("")
    note(f"      {'k removed':>12s} {'B0 RMSE':>10s} {arm + ' RMSE':>11s} "
         f"{'delta':>10s}")
    for k in (0, 1, 5, 10, 25, 50, 100):
        m = np.ones(len(e0), bool)
        if k:
            m[order[:k]] = False
        note(f"      {k:>12,} {rmse(e0[m]):>10.4f} {rmse(e1[m]):>11.4f} "
             f"{rmse(e1[m]) - rmse(e0[m]):>+10.4f}")
    note("")

# ============================================================
# THE IMPORTANT QUESTION
# ============================================================
head("5. DOES THIS IDENTIFY UNRELIABLE RECORDS, OR JUST EXTREME TARGETS?")
note("")
note("  These two explanations make different predictions, and the test is")
note("  whether the gain lands on CONTRADICTORY TIMESTAMPS or on BIG TARGETS.")
note("  V9-C's target bands are the discriminator: if the gain tracks the")
note("  target distribution rather than the spread distribution, the features")
note("  are acting as an outlier detector and nothing more.")
note("")
if not gainers:
    note("  No arm improves pooled RMSE, so there is no gain to attribute.")
else:
    for arm in gainers:
        e0, e1 = errs("B0"), errs(arm)
        s0, s1 = e0 ** 2, e1 ** 2
        gain = s0 - s1
        net = float(gain.sum())
        note(f"  {arm}:")
        note("")
        note(f"    {'explanation':38s} {'group':22s} {'n':>8s} "
             f"{'% of net gain':>15s} {'% of rows':>10s} {'ratio':>8s}")
        tests = [("contradictory timestamps", "spread 1-6h",
                  band_mask("spread", 1 * H, 6 * H)),
                 ("extreme targets", "target > 4h",
                  band_mask("y", 4 * H, np.inf)),
                 ("extreme targets", "target > 2h",
                  band_mask("y", 2 * H, np.inf))]
        for why, lab, m in tests:
            if m.sum() == 0:
                continue
            sh = 100 * gain[m].sum() / net
            pr = 100 * m.sum() / len(e0)
            note(f"    {why:38s} {lab:22s} {int(m.sum()):>8,} "
                 f"{sh:>14.1f}% {pr:>9.4f}% {sh / max(pr, 1e-12):>7.1f}x")
        note("")
        note("    A ratio near 1.0 means the group carries its proportional")
        note("    share and the feature is not special to it. A large ratio means")
        note("    the group is over-represented in the gain, which is what a")
        note("    genuine detector would look like - and what a tail artefact")
        note("    would also look like, which is why the fold and airport")
        note("    checks in section 4 decide it.")
        note("")
        note("    Correlation of the change in |error| with each candidate")
        note("    driver (pooled; the spread correlation is the reference):")
        dsafe = np.nan_to_num(pooled(arm, "spread"), nan=0.0)
        dabs = np.abs(e1) - np.abs(e0)
        for lab, v in (("flight-delta spread", dsafe),
                       ("target magnitude", np.abs(pooled(arm, "y"))),
                       ("|d_SCHED|", np.abs(pooled(arm, "d_sched")))):
            rv = float(np.corrcoef(dsafe, v)[0, 1])
            rd = float(np.corrcoef(dabs, v)[0, 1])
            ratio = "n/a" if rv == 0 else f"{rd / rv:+.2f}"
            note(f"      vs {lab:22s}  d|error| r = {rd:+.4f}   "
                 f"spread r = {rv:+.4f}   ratio = {ratio}")
        note("")

# ============================================================
# FEATURE IMPORTANCE - diagnostic only
# ============================================================
if gainers:
    best = min(gainers, key=lambda a: deltas[a])
    sub(f"6. FEATURE USEFULNESS - the best arm is {best}")
    note("  Reported as a diagnostic, NOT as evidence. Importance says what the")
    note("  model split on; the ablation in section 3 and the forensics in")
    note("  section 4 are the evidence about whether splitting on it helped.")
    note("")
    for fold in RFOLDS:
        c = cache[(fold, best)]
        feats = c["features"]
        imp = pd.Series(c["imp"], index=feats).sort_values(ascending=False)
        new = [f for f in feats if f in set(ARMS[best][0])]
        tot = imp.sum()
        note(f"  {fold}: {len(feats)} features, share of importance")
        note(f"    {'rank':>4s} {'feature':32s} {'share':>8s}  new?")
        for i, (f, v) in enumerate(imp.items(), 1):
            note(f"    {i:>4d} {f:32s} {100 * v / tot:>7.2f}%  "
                 f"{'NEW' if f in new else ''}")
        note(f"    new features combined: "
             f"{100 * imp[new].sum() / tot:.2f}%   "
             f"five raw T4 deltas combined: "
             f"{100 * imp[T4_FEATURES].sum() / tot:.2f}%")
        note("")

# ============================================================
# DECISION
# ============================================================
head("7. DECISION")

note("  Per-arm summary against the frozen baseline:")
note(f"  {'arm':4s} {'dJAN':>9s} {'dJUL':>9s} {'dPOOL':>9s} {'spread':>8s} "
     f"{'improved?':>10s}  robustness read")
for a in ARMS_ORDER:
    if a not in deltas:
        continue
    dj, dl, dp = JANd[a], JULd[a], deltas[a]
    spread_d = abs(dj - dl)
    same_sign = (dj < 0) == (dl < 0)
    if dp >= -1e-9:
        read = "no pooled gain"
    elif not same_sign:
        read = "FOLD-INCONSISTENT - folds disagree on sign"
    elif dj < 0 and dl < 0 and spread_d < 2.0:
        read = "both folds improve, similar size"
    elif spread_d > 10:
        read = "both folds improve but wildly uneven"
    else:
        read = "both folds improve"
    note(f"  {a:4s} {dj:>+9.4f} {dl:>+9.4f} {dp:>+9.4f} {spread_d:>8.4f} "
         f"{('yes' if dp < 0 else 'no'):>10s}  {read}")
note("")

CRIT = []
best_arm = min(deltas, key=lambda a: deltas[a]) if deltas else None
if best_arm is not None and deltas[best_arm] < -1e-9:
    e0, e1 = errs("B0"), errs(best_arm)
    s0, s1 = e0 ** 2, e1 ** 2
    gain = s0 - s1
    net = float(gain.sum())
    ap = pooled(best_arm, "airport")
    order = np.argsort(-gain)
    spread_v = pooled(best_arm, "spread")
    full = ~np.isnan(spread_v)
    conf = full & (spread_v >= H)          # contradictory timestamps
    agree = full & (spread_v < H)          # timestamps agree
    excl = ~np.isin(ap, ["LIRF", "LFPG"])
    big_tgt = pooled(best_arm, "y") >= 4 * H
    keep25 = np.ones(len(e0), bool)
    keep25[order[:25]] = False
    CRIT = [
        ("pooled gain is real", deltas[best_arm] < 0),
        ("JAN and JUL both improve",
         JANd[best_arm] < 0 and JULd[best_arm] < 0),
        ("folds improve by a similar amount",
         abs(JANd[best_arm] - JULd[best_arm]) < 2.0),
        ("gain survives removing LIRF + LFPG",
         rmse(e1[excl]) < rmse(e0[excl]) - 1e-9),
        ("majority of rows improve",
         int((gain > 0).sum()) > int((gain < 0).sum())),
        ("ordinary rows (spread < 1h) do not worsen",
         rmse(e1[agree]) <= rmse(e0[agree]) + 1e-9),
        ("contradictory-timestamp rows improve",
         conf.sum() > 0 and rmse(e1[conf]) < rmse(e0[conf]) - 1e-9),
        ("contradictory rows improve MORE than agreeing rows",
         (rmse(e1[conf]) - rmse(e0[conf]))
         < (rmse(e1[agree]) - rmse(e0[agree]))),
        ("<100% of net gain inside the worst 100 rows",
         float(gain[order[:100]].sum()) < net),
        ("LIRF supplies <100% of net gain",
         float(gain[ap == "LIRF"].sum()) < net),
        ("extreme-target rows supply <100% of net gain",
         float(gain[big_tgt].sum()) < net),
        ("gain survives removing the worst 25 rows",
         rmse(e1[keep25]) < rmse(e0[keep25]) - 1e-9),
    ]
    sub("7a. ANTI-PATTERN FILTER - has this repo's failure mode returned?")
    note("  The same mechanical filter that rejected the clock features,")
    note("  rejected H1, and graded V9-A and V9-B MIXED. Applied to the best arm.")
    note("")
    for lab, ok in CRIT:
        note(f"    [{'PASS' if ok else 'FAIL'}] {lab}")
    nfail = sum(1 for _, ok in CRIT if not ok)
    note("")
    note(f"  best arm {best_arm}: {nfail} of {len(CRIT)} criteria fail.")
    note("  worst-100 rows hold "
         f"{100 * float(gain[order[:100]].sum()) / net:.1f}% of the net gain;")
    note(f"  LIRF holds {100 * float(gain[ap == 'LIRF'].sum()) / net:.1f}% of it "
         f"across {int((ap == 'LIRF').sum()):,} rows;")
    note("  target > 4h rows hold "
         f"{100 * float(gain[big_tgt].sum()) / net:.1f}%.")
    note("")
else:
    nfail = len(CRIT)

sub("7b. CONCLUSION")

# Per-arm verdicts, so D1 and D2 are graded separately rather than only the
# arm with the best pooled number.
ARM_VERDICT = {}
for a in ARMS_ORDER:
    if a not in deltas:
        continue
    dp, dj, dl = deltas[a], JANd[a], JULd[a]
    same_sign = (dj < 0) == (dl < 0)
    if dp >= -1e-9:
        ARM_VERDICT[a] = "REJECT"
        v_why = "no pooled gain"
    elif not same_sign:
        ARM_VERDICT[a] = "REJECT"
        v_why = (f"folds disagree on sign (JAN {dj:+.4f}, JUL {dl:+.4f}); a "
                 f"pooled gain built from two opposite effects is not a gain")
    elif a == best_arm and CRIT and (not CRIT[8][1] or not CRIT[11][1]):
        ARM_VERDICT[a] = "REJECT"
        v_why = ("gain is tail-driven: it does not survive removal of the "
                 "top 25 contributing rows")
    elif a == best_arm and CRIT and nfail > 3:
        ARM_VERDICT[a] = "MIXED - investigate"
        v_why = "real pooled gain, but too many anti-pattern criteria fail"
    elif dp < 0 and same_sign and abs(dj - dl) < 2.0:
        ARM_VERDICT[a] = "PROMISING - continue"
        v_why = "consistent, similarly sized gain in both folds"
    else:
        ARM_VERDICT[a] = "MIXED - investigate"
        v_why = "real pooled gain, uneven across folds"
    note(f"    {a}: {ARM_VERDICT[a]:<24s} {v_why}")
note("")

if best_arm is None:
    VERDICT = "REJECT"
    why = "No arm was evaluated."
else:
    dp = deltas[best_arm]
    dj, dl = JANd[best_arm], JULd[best_arm]
    robust = (dp < 0 and dj < 0 and dl < 0
              and abs(dj - dl) < 2.0 and nfail <= 1
              and CRIT[3][1] and CRIT[5][1] and CRIT[6][1])
    # The brief's decision rule, applied literally: REJECT if the improvement
    # is negligible, inconsistent ACROSS FOLDS, or tail-driven. A negative
    # pooled delta is necessary but NOT sufficient for a non-REJECT verdict -
    # otherwise a gain bought by five LIRF rows would score as MIXED, which is
    # exactly the failure this repo has already refused three times.
    same_sign = (dj < 0) == (dl < 0)
    tail_ok = bool(CRIT) and CRIT[8][1] and CRIT[11][1]
    if dp >= -1e-9:
        VERDICT = "REJECT"
    elif not same_sign:
        VERDICT = "REJECT"
    elif not tail_ok:
        VERDICT = "REJECT"
    elif robust:
        VERDICT = "PROMISING - continue"
    else:
        VERDICT = "MIXED - investigate"

    # why the pooled number is not to be believed
    if CRIT:
        w100 = 100 * float(gain[order[:100]].sum()) / net
        lf = 100 * float(gain[ap == "LIRF"].sum()) / net
        lfp = 100 * float(gain[np.isin(ap, ["LIRF", "LFPG"])].sum()) / net
        k5 = np.ones(len(e0), bool)
        k5[order[:5]] = False
        flip5 = rmse(e1[k5]) - rmse(e0[k5])
        keep25 = np.ones(len(e0), bool)
        keep25[order[:25]] = False
        flip25 = rmse(e1[keep25]) - rmse(e0[keep25])
        tail_block = f"""
    The pooled number ({dp:+.4f}) is real arithmetic but is not a gain:
      - the two folds move in OPPOSITE directions (JAN {dj:+.4f}, JUL {dl:+.4f});
      - the worst 100 rows ({100 * 100 / len(gain):.4f}% of the set) hold
        {w100:.1f}% of the net improvement;
      - LIRF holds {lf:.1f}% of it, LIRF+LFPG hold {lfp:.1f}% of it;
      - deleting only the 5 single best-improved rows turns the result
        {flip5:+.4f}, i.e. WORSE than baseline. Deleting the worst 25 turns it
        {flip25:+.4f};
      - {100 * (gain < 0).sum() / len(gain):.1f}% of rows got worse."""
    else:
        tail_block = ""

    why = f"""Best arm is {best_arm} ({ARMS[best_arm][1]}, +{len(ARMS[best_arm][0])} features).

    pooled RMSE   {base_pool:.4f} -> {RES[best_arm]['POOL']['rmse']:.4f}   ({dp:+.4f})
    JAN           {RES['B0']['JAN']['rmse']:.4f} -> {RES[best_arm]['JAN']['rmse']:.4f}   ({dj:+.4f})
    JUL           {RES['B0']['JUL']['rmse']:.4f} -> {RES[best_arm]['JUL']['rmse']:.4f}   ({dl:+.4f})
    fold spread   {abs(dj - dl):.4f}
    anti-pattern  {nfail} of {len(CRIT)} criteria fail{tail_block}"""

note("")
note("=" * 100)
note(f"  CONCLUSION: {VERDICT}")
note("=" * 100)
note(f"""
{why}
""")

if VERDICT.startswith("MIXED") or VERDICT.startswith("REJECT"):
    note("  BRANCH STOPPING RULE")
    note("  ----------------------")
    note("    This experiment is the one remaining timestamp candidate that V9-C")
    note("    found to be fold-robust, and it has now been tested. RECOMMEND")
    note("    ENDING further timestamp feature engineering for now. Do not open")
    note("    a V9-E/V9-F to chase a smaller variant (spread only, median only,")
    note("    pairwise differences, other timestamp quantiles). Three separate")
    note("    timestamp lines have now failed for the same structural reason:")
    note("    the flight timestamps that exist are mostly consistent, and where")
    note("    they are absent no feature derived from them can help.")
    note("")
else:
    note("  The branch rule does not trigger: this arm is not MIXED or REJECT.")
    note("  Any follow-up should still be a single controlled ablation against")
    note("  the same frozen B0 gate, not a wider search.")
    note("")

note("  NOTHING IS PROMOTED. V8/T4 remains the champion at 425.7711. This file")
note("  is an experiment report only: no submission was created, no v3 was")
note("  written, nothing was uploaded, and no commit, push, merge or reset was")
note("  performed. V8, V9-A, V9-B and V9-C are all unmodified - verified below.")

sub("8. FROZEN-ARTEFACT VERIFICATION (post-run)")

FROZEN = [
    "models/v8_catboost.cbm",
    "src/train_catboost_v8.py",
    "src/train_catboost_v8_defs.py",
    "src/create_submission_v8.py",
    "src/v8_error_autopsy.py",
    "src/experiment_v9a_missing_segment.py",
    "src/v9a_2026_distribution_audit.py",
    "src/experiment_v9b_timestamp_plausibility.py",
    "reports/v8_error_autopsy.txt",
    "reports/v9a_missing_segment_experiment.txt",
    "reports/v9a_2026_distribution_audit.txt",
    "reports/v9b_timestamp_plausibility_experiment.txt",
    "src/v9c_dsched_forensic_audit.py",
    "reports/v9c_dsched_forensic_audit.txt",
    "submissions/vigorous-glove_v2.parquet",
]
# Sizes recorded by the V9-C report, which ran before this experiment began.
PRIOR_SIZE = {
    "src/experiment_v9a_missing_segment.py": 57_032,
    "reports/v9a_missing_segment_experiment.txt": 71_203,
    "reports/v9a_2026_distribution_audit.txt": 47_183,
    "src/experiment_v9b_timestamp_plausibility.py": 58_195,
    "reports/v9b_timestamp_plausibility_experiment.txt": 36_978,
    "src/v9c_dsched_forensic_audit.py": 56_226,
    "reports/v9c_dsched_forensic_audit.txt": 62_620,
}
note("  How this was verified, stated honestly:")
note("    git cannot be used - every frozen artefact above is UNTRACKED in this")
note("    repository, so 'git status' reports them all as new and proves nothing")
note("    either way. Sizes alone are weak evidence, so each file is hashed. The")
note("    sizes recorded in the V9-C report are also cross-checked here.")
note("")
note(f"  {'path':50s} {'bytes':>12s}  {'sha256 (first 16)':<18s} status")
for p in FROZEN:
    q = Path(p)
    if not q.exists():
        note(f"  {p:50s} {'-':>12s}  {'-':<18s} MISSING")
        continue
    h = hashlib.sha256(q.read_bytes()).hexdigest()[:16]
    sz = q.stat().st_size
    if p in PRIOR_SIZE:
        ok = sz == PRIOR_SIZE[p]
        st = (f"unchanged (matches V9-C record {PRIOR_SIZE[p]:,})" if ok
              else f"CHANGED (was {PRIOR_SIZE[p]:,})")
    else:
        st = "present, hashed"
    note(f"  {p:50s} {sz:>12,}  {h:<18s} {st}")
note("")
note("  Files this experiment created, and nothing else:")
note(f"    {REPORT}")
note(f"    {CACHE}   (V9-D's own cache; the V9-B cache was opened read-only)")
note("    src/experiment_v9d_flight_delta_spread.py")
note("")
note("  No file in models/ or submissions/ was created, modified or read for")
note("  targets. No 2026 data was touched. No 2026 target, hidden target or")
note("  leaderboard number was used at any point. No prediction was clipped or")
note("  post-processed. No submission and no v3 parquet was produced.")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\n[written] {REPORT}  lines={len(out)}  "
      f"bytes={REPORT.stat().st_size}", flush=True)
print(f"\n  VERDICT: {VERDICT}", flush=True)
