"""V9-A: dedicated segment model for the missing-flight-data regime.

ONE controlled experiment. Nothing in V8 is modified. No submission is written.
No 2026 data is touched. No leaderboard information is used.

HYPOTHESIS (from reports/v8_error_autopsy.txt, section 10c item #1)
    Rows where AOBT_3_flt, EOBT_1_flt, IOBT_flt and LOBT_flt are ALL missing
    are ~1.6% of validation rows but hold ~61% of pooled SSE. Those rows reach
    the T4 model as four NaN deltas plus a lone d_SCHED, so the model has
    almost nothing to work with. This experiment asks whether a CatBoost model
    trained ONLY on training-fold rows in that same regime, using only the
    features that actually exist there, beats the T4 fallback on that regime.

DESIGN
    baseline  : the exact validated T4 predictions, loaded from the frozen
                cache written by experiment_t4_clocks.py. NOT refitted.
    hybrid    : V9-A = cached T4 on every ordinary row, segment model on the
                missing-flight rows. Verified elementwise to be bit-identical
                to cached T4 on every ordinary row.
    segment   : features = the 12 non-T4 V7 operational/categorical features
                plus d_SCHED. The four flight-level deltas are excluded because
                they do not exist in this regime. No new features, no
                external data.
    target    : raw TAXITIME_SEC_mvt, no clipping, no cleaning, no filtering.
    score     : raw RMSE, exact JAN and JUL definitions from the T4 experiment.

LEAKAGE HYGIENE
    The segment model never sees the validation month.
      * training rows for a fold = all 2025 departures outside that fold's
        holdout month, restricted to the missing-flight condition.
      * the iteration count is NOT chosen on the validation fold. It is chosen
        by early stopping on a time-ordered inner split carved out of the
        TRAINING fold only, then the model is refit on all training-fold
        segment rows at that fixed count.
      * a fixed 316-tree variant (the frozen V8 count) is also reported as a
        sensitivity check so the result cannot be an artefact of how the
        iteration count was chosen.
    The inner split is the last 15% of the training fold by time, so it is a
    forward-in-time check, matching how a real submission is served.

Usage:
    python src/experiment_v9a_missing_segment.py
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train_catboost_v8_defs import (  # noqa: F401  (read-only import)
    DATA_DIR, FEATURES as T4_FULL_FEATURES, FORBIDDEN, MINUEND, TARGET,
    T4_FEATURES, T4_SPEC, V7_CATEGORICALS, V7_FEATURES, N_TREES,
    add_t4_features, add_time_features, add_traffic_feature,
    apply_categorical_handling, verify_t4_deltas,
)

# ============================================================
# CONSTANTS - frozen references, identical to the T4 experiment
# ============================================================

CACHE = Path(os.environ.get("TEMP", ".")) / "opencode" / "clocks_results.pkl"
REPORT = Path("reports/v9a_missing_segment_experiment.txt")

FOLDS = {"JAN": ("2025-01-01", "2025-02-01"),
         "JUL": ("2025-07-01", "2025-08-01")}
T4_REF = {"JAN": 404.8803102, "JUL": 513.7559738}
T4_REF_POOLED = 468.31
EXPECTED_N = {"JAN": (1_931_341, 153_706), "JUL": (1_894_334, 190_713)}

# The four flight-level timestamps. Missing-all-four is the segment condition.
FLIGHT_TS = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]
assert [c for _, c in T4_SPEC[:4]] == FLIGHT_TS

# Segment feature set: the 12 non-T4 V7 features + the only surviving T4 delta.
SEG_FEATURES = V7_FEATURES + ["d_SCHED"]
SEG_CATEGORICALS = V7_CATEGORICALS

# Same hyper-parameters as the validated T4 experiment. Only the iteration
# budget differs, and that is fixed in advance per fold from the inner split.
SEG_BASE_PARAMS = dict(
    depth=8,
    learning_rate=0.1,
    loss_function="RMSE",
    random_seed=42,
    verbose=200,
)

INNER_FRACTION = 0.15          # last 15% of the training fold by time
SENSITIVITY_TREES = N_TREES    # 316, the frozen V8 count

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


def rse(se):
    return float(np.sqrt(np.asarray(se, dtype="float64").mean()))


# ============================================================
# 0. GUARDS
# ============================================================

assert not (set(SEG_FEATURES) & set(FORBIDDEN)), "forbidden column in segment features"
assert not (set(SEG_FEATURES) & set(T4_FEATURES[:4])), \
    "the four flight-level deltas must NOT be segment features"
assert "d_SCHED" in SEG_FEATURES, "d_SCHED must be available to the segment model"
assert len(SEG_FEATURES) == 13 == len(T4_FULL_FEATURES) - 4

head("V9-A EXPERIMENT - DEDICATED SEGMENT MODEL FOR THE MISSING-FLIGHT-DATA "
     "REGIME")
note("")
note("This script MODIFIES NOTHING. V8, its model file and its submission")
note("builder are untouched. No submission is created. No 2026 data is read.")
note("No leaderboard information is used anywhere in this experiment.")
note("")
note("  segment condition : AOBT_3_flt, EOBT_1_flt, IOBT_flt and LOBT_flt are")
note("                      ALL missing (verified to be one shared null mask)")
note("  segment features  : 12 non-T4 V7 operational/categorical features")
note("                      + d_SCHED  = 13")
note("  segment training  : training-fold rows in the same condition only")
note("  baseline          : exact cached T4 predictions, not refitted")
note("  target / score    : raw TAXITIME_SEC_mvt, raw RMSE, no clipping anywhere")

# ============================================================
# 1. BASELINE PROVENANCE
# ============================================================

head("1. BASELINE PROVENANCE - the cached predictions ARE the exact T4 "
     "predictions")

assert CACHE.exists(), f"missing cache {CACHE}"
cache = pickle.load(open(CACHE, "rb"))
assert "T4" in cache, f"no T4 arm in {CACHE}"
t4 = cache["T4"]

prov = {}
for fold, ref in T4_REF.items():
    got = t4[fold]["rmse"]
    prov[fold] = abs(got - ref)
    note(f"  {fold}: cached RMSE {got:.7f}   frozen T4 reference {ref:.7f}   "
         f"diff {got - ref:+.7f}")
    assert abs(got - ref) < 1e-6, f"{fold} cached T4 RMSE drifted"
note("  [PASS] cached T4 RMSE equals the frozen T4 reference on both folds")
note(f"  cache: {CACHE}")
note("  producer: experiment_t4_clocks.py, T4 arm, which reproduced")
note("            experiment_v7_vs_t4.py exactly and was itself run twice")
note("            with identical results. No T4 model is refitted here.")

# ============================================================
# 2. REBUILD THE FEATURE FRAME (no T4 model is fitted)
# ============================================================

note("")
note("Rebuilding the feature frame with the drift-guarded V8/T4 definitions "
     "from train_catboost_v8_defs.py (no fit)...")

LOAD = list(dict.fromkeys(
    [MINUEND] + [c for _, c in T4_SPEC] + [
        "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
        "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "WK_TBL_CAT_flt",
        "FLIGHT_TYPE_flt", TARGET]))

files = sorted(DATA_DIR.glob("training_*.parquet"))
assert len(files) == 12, f"expected 12 training files, got {len(files)}"
data = pd.concat([pd.read_parquet(f, columns=LOAD) for f in files],
                 ignore_index=True)
note(f"  total movements: {len(data):,}")
data = add_traffic_feature(data)
dep = data[data["PHASE_mvt"] == "DEP"].copy()
del data
dep = add_time_features(dep)
dep = add_t4_features(dep)
dep = apply_categorical_handling(dep)
note(f"  departure rows  : {len(dep):,}")
assert len(dep) == 2_085_047, f"unexpected departures {len(dep):,}"

note("")
note("  T4 delta unit re-check on the full 2025 departure set:")
verify_t4_deltas(dep)
note("  [PASS] deltas reproduced exactly, in seconds")

miss4 = dep[FLIGHT_TS].isna()
pat = miss4.astype(np.int8).drop_duplicates()
n_pat = pat.shape[0]
note("")
note(f"  distinct missingness patterns among {FLIGHT_TS}: {n_pat}")
note("  " + pat.to_string().replace("\n", "\n  ").strip())
# exactly two patterns, each constant across the four columns, means the four
# timestamps share ONE null mask and 'all four missing' is unambiguous
assert n_pat == 2, "unexpected number of missingness patterns"
assert (pat.nunique(axis=1) == 1).all(), \
    "the four flight timestamps do NOT share one null mask"
assert int(miss4.sum(axis=1).max()) in (0, 4), "a partial-missing row exists"
seg_all = miss4.all(axis=1).to_numpy()
note(f"  rows with all four flight timestamps missing, all 2025 departures  : "
     f"{int(seg_all.sum()):,} ({100 * seg_all.mean():.3f}%)")
note(f"  d_SCHED nulls in that regime                                     : "
     f"{int(dep.loc[seg_all, 'd_SCHED'].isna().sum()):,}")
note(f"  d_SCHED nulls overall                                            : "
     f"{int(dep['d_SCHED'].isna().sum()):,}")
note("  [PASS] the four flight timestamps share exactly one null mask, so")
note("         'all four missing' is unambiguous, and d_SCHED always exists")
note("         in this regime, as the experiment design requires.")

# ============================================================
# 3. ALIGN THE CACHED PREDICTIONS AND SPLIT EVERY FOLD
# ============================================================

V, HOLD = {}, {}
for fold in ("JAN", "JUL"):
    lo, hi = FOLDS[fold]
    h = (dep[MINUEND] >= lo) & (dep[MINUEND] < hi)
    hn = h.to_numpy()
    va = dep[hn]
    assert (len(dep[~h]), len(va)) == EXPECTED_N[fold], f"{fold} row counts"
    y_new = va[TARGET].to_numpy()
    assert np.array_equal(y_new, t4[fold]["y"]), (
        f"{fold}: row order does not match the cache - ABORT")
    assert np.array_equal(va["AOBT_3_flt"].notna().to_numpy(),
                          t4[fold]["has_flight"]), f"{fold} has_flight mask"

    seg = seg_all[hn]
    f = va.copy()
    f["FOLD"] = fold
    f["T4_PRED"] = t4[fold]["pred"]
    f["Y"] = f[TARGET]
    f["SEG"] = seg
    V[fold] = f
    HOLD[fold] = hn
    note(f"  {fold}: val {len(va):,}  seg {int(seg.sum()):,} "
         f"({100 * seg.mean():.3f}%)   train {int((~hn).sum()):,}  "
         f"seg {int(seg_all[~hn].sum()):,}")
note("  [PASS] cached predictions align row-for-row with the rebuilt frame,")
note("         proved by exact target equality and exact has_flight equality")
note("         rather than assumed. Original row order preserved everywhere.")

# ============================================================
# 4. TRAIN THE SEGMENT MODEL PER FOLD  (training-fold data only)
# ============================================================

head("4. SEGMENT MODEL - training-fold data only, iteration count chosen on an "
     "inner split of the training fold")

models = {}
for fold in ("JAN", "JUL"):
    lo, hi = FOLDS[fold]
    hn = HOLD[fold]
    tr_mask = ~hn
    seg_tr_mask = tr_mask & seg_all
    tr_seg = dep[seg_tr_mask].copy()
    n_seg_tr = len(tr_seg)
    note("")
    sub(f"4.{fold} - segment training set")
    note(f"  training-fold departures            : {int(tr_mask.sum()):,}")
    note(f"  of those, missing-flight-data rows  : {n_seg_tr:,} "
         f"({100 * n_seg_tr / int(tr_mask.sum()):.3f}%)")
    assert lo[:7] not in set(tr_seg["month"]), "holdout month leaked into train"
    assert n_seg_tr > 5000, f"{fold}: too few segment training rows"
    note(f"  segment training months             : "
         f"{sorted(tr_seg['month'].unique())}")
    note(f"  validation months                   : "
         f"{sorted(V[fold]['month'].unique())}")
    note(f"  target on segment training rows: mean={tr_seg[TARGET].mean():,.1f}  "
         f"median={tr_seg[TARGET].median():,.1f}  max={tr_seg[TARGET].max():,.1f}")
    note(f"  |d_SCHED| median on segment train rows: "
         f"{tr_seg['d_SCHED'].abs().median():,.1f}s")

    # ---- inner forward-in-time split, training fold only -----------------
    cut = tr_seg[MINUEND].quantile(1.0 - INNER_FRACTION)
    inner_fit = tr_seg[tr_seg[MINUEND] < cut]
    inner_es = tr_seg[tr_seg[MINUEND] >= cut]
    note(f"  inner early-stopping split (training fold only, forward in time)")
    note(f"    cut at {pd.Timestamp(cut)}")
    note(f"    fit rows {len(inner_fit):,}   early-stopping rows {len(inner_es):,}")
    assert len(inner_es) > 500, f"{fold}: inner early-stopping set too small"

    t0 = time.time()
    m_es = CatBoostRegressor(iterations=1000, early_stopping_rounds=50,
                             use_best_model=True, **SEG_BASE_PARAMS)
    m_es.fit(inner_fit[SEG_FEATURES], inner_fit[TARGET],
             cat_features=SEG_CATEGORICALS,
             eval_set=(inner_es[SEG_FEATURES], inner_es[TARGET]))
    best = m_es.get_best_iteration()
    el_es = time.time() - t0
    note(f"  stage 1 (inner split)  best_iter={best}  -> {best + 1} trees   "
         f"{el_es / 60:.1f} min")

    # ---- refit on ALL training-fold segment rows at a fixed count ---------
    t0 = time.time()
    m_main = CatBoostRegressor(iterations=best + 1, **SEG_BASE_PARAMS)
    m_main.fit(tr_seg[SEG_FEATURES], tr_seg[TARGET],
               cat_features=SEG_CATEGORICALS)
    el_main = time.time() - t0
    assert m_main.tree_count_ == best + 1
    note(f"  stage 2 (PRIMARY)      refit on all {n_seg_tr:,} training-fold "
         f"segment rows at {best + 1} trees   {el_main / 60:.1f} min")

    # ---- sensitivity: the frozen V8 iteration count ----------------------
    t0 = time.time()
    m_sens = CatBoostRegressor(iterations=SENSITIVITY_TREES, **SEG_BASE_PARAMS)
    m_sens.fit(tr_seg[SEG_FEATURES], tr_seg[TARGET],
               cat_features=SEG_CATEGORICALS)
    el_sens = time.time() - t0
    note(f"  sensitivity arm       fixed {SENSITIVITY_TREES} trees "
         f"(frozen V8 count)   {el_sens / 60:.1f} min")

    models[fold] = dict(main=m_main, sens=m_sens, n_seg_tr=n_seg_tr,
                        best=best, cut=pd.Timestamp(cut))

    imp = pd.Series(m_main.get_feature_importance(), index=SEG_FEATURES)
    note("")
    note(f"  segment feature importance ({fold}):")
    for k, v in imp.sort_values(ascending=False).items():
        note(f"    {k:20s} {v:8.4f}  {100 * v / imp.sum():5.2f}%")

note("")
note("  [PASS] the segment model for each fold is fitted only on training-fold")
note("         rows in the same missing-flight regime")
note("  [PASS] the holdout month is absent from every segment training set")
note("  [PASS] the iteration count comes from an inner split of the TRAINING")
note("         fold, never from the validation fold")

# ============================================================
# 5. BUILD THE V9-A HYBRID AND PROVE ORDINARY ROWS ARE UNCHANGED
# ============================================================

head("5. V9-A HYBRID - routing, and the proof that ordinary rows are untouched")

IDENT = {}
for fold in ("JAN", "JUL"):
    f = V[fold]
    seg = f["SEG"].to_numpy()
    t4p = f["T4_PRED"].to_numpy()
    m = models[fold]
    p_seg = m["main"].predict(f.loc[seg, SEG_FEATURES])
    p_seg_s = m["sens"].predict(f.loc[seg, SEG_FEATURES])
    hyb = t4p.copy()
    hyb[seg] = p_seg
    hyb_s = t4p.copy()
    hyb_s[seg] = p_seg_s

    ordinary = ~seg
    assert np.array_equal(hyb[ordinary], t4p[ordinary]), \
        f"{fold}: ordinary predictions changed"
    assert np.array_equal(hyb_s[ordinary], t4p[ordinary]), \
        f"{fold}: ordinary predictions changed in the sensitivity arm"
    IDENT[fold] = (int(ordinary.sum()), int(seg.sum()),
                   float(np.abs(hyb[ordinary] - t4p[ordinary]).max()))

    note("")
    sub(f"5.{fold} - routing")
    note(f"  validation rows                          : {len(f):,}")
    note(f"  routed to the segment model (all 4 miss) : {int(seg.sum()):,} "
         f"({100 * seg.mean():.3f}%)")
    note(f"  kept on the exact cached T4 prediction   : {int(ordinary.sum()):,} "
         f"({100 * ordinary.mean():.3f}%)")
    note(f"  segment trees (primary / sensitivity)    : {m['best'] + 1} / "
         f"{SENSITIVITY_TREES}")
    note(f"  [PASS] ordinary rows: {IDENT[fold][0]:,} of "
         f"{IDENT[fold][0]:,} predictions are BIT-IDENTICAL to the cached T4")
    note(f"         prediction (max|diff| = {IDENT[fold][2]:.1e})")
    note(f"  [PASS] the same holds for the sensitivity arm")

    f["PRED"] = hyb
    f["PRED_S"] = hyb_s
    f["ERR_T4"] = f["T4_PRED"] - f["Y"]
    f["AE_T4"] = f["ERR_T4"].abs()
    f["ERR_V9"] = f["PRED"] - f["Y"]
    f["AE_V9"] = f["ERR_V9"].abs()
    f["ERR_S"] = f["PRED_S"] - f["Y"]
    f["AE_S"] = f["ERR_S"].abs()
    f["SE"] = f["ERR_T4"] ** 2
    f["SE_V9"] = f["ERR_V9"] ** 2
    f["SE_S"] = f["ERR_S"] ** 2

# ============================================================
# 6. HEADLINE
# ============================================================

head("6. HEADLINE - raw RMSE, exact T4 definitions, no exclusions")

HEAD = {}
for fold in ("JAN", "JUL"):
    f = V[fold]
    HEAD[fold] = dict(
        n=len(f), t4=rse(f["SE"]), v9a=rse(f["SE_V9"]), sens=rse(f["SE_S"]),
        check=abs(rse(f["SE"]) - T4_REF[fold]),
        d=rse(f["SE_V9"]) - rse(f["SE"]),
        ds=rse(f["SE_S"]) - rse(f["SE"]),
        sse_b=float(f["SE"].sum()), sse_a=float(f["SE_V9"].sum()),
        sse_s=float(f["SE_S"].sum()),
        n_seg=int(f["SEG"].sum()))

note("")
note(f"  {'fold':5s} {'n':>9s} {'T4 baseline':>14s} {'V9-A':>14s} {'delta':>10s} "
     f"{'sens(316)':>11s} {'sens delta':>11s} {'|b-T4ref|':>11s}")
for fold in ("JAN", "JUL"):
    r = HEAD[fold]
    note(f"  {fold:5s} {r['n']:>9,} {r['t4']:>14.7f} {r['v9a']:>14.7f} "
         f"{r['d']:>+10.4f} {r['sens']:>11.4f} {r['ds']:>+11.4f} "
         f"{r['check']:>11.2e}")
for fold in ("JAN", "JUL"):
    assert HEAD[fold]["check"] < 1e-6, f"{fold} baseline drifted from T4 reference"
note("")
note("  [PASS] both baseline columns equal the frozen T4 references exactly")
note("         (404.8803102 / 513.7559738), so the comparison is against the")
note("         exact validated T4 model, not a re-fit approximation")

sse_b = sum(HEAD[f]["sse_b"] for f in HEAD)
sse_a = sum(HEAD[f]["sse_a"] for f in HEAD)
sse_s = sum(HEAD[f]["sse_s"] for f in HEAD)
n_all = sum(HEAD[f]["n"] for f in HEAD)
POOL = dict(b=np.sqrt(sse_b / n_all), a=np.sqrt(sse_a / n_all),
            s=np.sqrt(sse_s / n_all), n=n_all)
note("")
note(f"  pooled Jan+Jul baseline T4 : {POOL['b']:.7f}  "
     f"(frozen reference {T4_REF_POOLED})")
note(f"  pooled Jan+Jul V9-A        : {POOL['a']:.7f}")
note(f"  pooled delta               : {POOL['a'] - POOL['b']:+.4f}")
note(f"  pooled sensitivity (316)   : {POOL['s']:.7f}   "
     f"delta {POOL['s'] - POOL['b']:+.4f}")
note(f"  pooled rows                : {POOL['n']:,}")
assert abs(POOL["b"] - T4_REF_POOLED) < 0.01, "pooled baseline drifted"
note("  pooled RMSE is from the combined SSE and row count, not an average of "
     "the two fold RMSEs")

# ============================================================
# 7. SEGMENT vs ORDINARY
# ============================================================

head("7. SEGMENT AND ORDINARY BREAKDOWN, PER FOLD")

for fold in ("JAN", "JUL"):
    f = V[fold]
    s = f[f["SEG"]]
    o = f[~f["SEG"]]
    sub(f"7.{fold} - SEGMENT (all four flight timestamps missing), "
        f"n = {len(s):,}")
    pairs = [
        ("segment RMSE", rse(s["SE"]), rse(s["SE_V9"])),
        ("segment MAE", float(s["AE_T4"].mean()), float(s["AE_V9"].mean())),
        ("segment median abs error", float(s["AE_T4"].median()),
         float(s["AE_V9"].median())),
        ("segment bias (pred - target)", float(s["ERR_T4"].mean()),
         float(s["ERR_V9"].mean())),
        ("segment SSE", float(s["SE"].sum()), float(s["SE_V9"].sum())),
        ("segment RMSE, 316-tree sensitivity", rse(s["SE"]), rse(s["SE_S"])),
        ("segment MAE, 316-tree sensitivity", float(s["AE_T4"].mean()),
         float(s["AE_S"].mean())),
        ("segment bias, 316-tree sensitivity", float(s["ERR_T4"].mean()),
         float(s["ERR_S"].mean())),
    ]
    note(f"  {'metric':40s} {'T4 before':>15s} {'V9-A after':>15s} "
         f"{'delta':>15s}")
    for lab, tb, av in pairs:
        note(f"  {lab:40s} {tb:>15,.4f} {av:>15,.4f} {av - tb:>+15,.4f}")
    note("")
    note(f"  segment share of this fold's rows    : {100 * len(s) / len(f):.3f}%")
    note(f"  segment share of this fold's SSE before: "
         f"{100 * s['SE'].sum() / f['SE'].sum():.2f}%")
    note(f"  segment share of this fold's SSE after : "
         f"{100 * s['SE_V9'].sum() / f['SE_V9'].sum():.2f}%")
    note(f"  target median {s['Y'].median():,.1f}s   max {s['Y'].max():,.1f}s   "
         f"mean {s['Y'].mean():,.1f}s")
    note(f"  T4 pred median before {s['T4_PRED'].median():,.1f}s   after "
         f"{s['PRED'].median():,.1f}s")

    sub(f"7.{fold} - ORDINARY / non-segment (must be identical), "
        f"n = {len(o):,}")
    rb, ra = rse(o["SE"]), rse(o["SE_V9"])
    note(f"  ordinary RMSE before : {rb:.7f}")
    note(f"  ordinary RMSE after  : {ra:.7f}")
    note(f"  delta                : {ra - rb:+.12f}")
    note(f"  ordinary MAE before  : {o['AE_T4'].mean():.7f}")
    note(f"  ordinary MAE after   : {o['AE_V9'].mean():.7f}")
    note(f"  ordinary SSE before  : {o['SE'].sum():.6e}")
    note(f"  ordinary SSE after   : {o['SE_V9'].sum():.6e}")
    assert ra == rb, f"{fold}: ordinary RMSE changed"
    assert float(o["SE"].sum()) == float(o["SE_V9"].sum())
    assert np.array_equal(o["T4_PRED"].to_numpy(), o["PRED"].to_numpy())
    note(f"  [PASS] ordinary RMSE, MAE, SSE and every individual prediction are "
         f"identical before and after")

# ============================================================
# 8. WORST 20 SEGMENT ERRORS
# ============================================================

head("8. WORST 20 SEGMENT ERRORS, BEFORE AND AFTER")

WCOLS = ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt",
         "AIRCRAFT_TYPE_mvt", "FLIGHT_TYPE_flt", "traffic_last_15min",
         "d_SCHED", "Y"]

for fold in ("JAN", "JUL"):
    s = V[fold][V[fold]["SEG"]]
    sub(f"8.{fold} - worst 20 segment rows by |error| under T4 (before)")
    w = s.sort_values("AE_T4", ascending=False).head(20)
    d = w[WCOLS + ["T4_PRED", "ERR_T4", "AE_T4"]].copy()
    d.columns = WCOLS + ["pred_before", "err_before", "ae_before"]
    note(d.round(1).to_string(index=False))
    note("")
    note("  the SAME 20 rows under V9-A (ordering preserved, so this is a "
         "row-for-row comparison):")
    d2 = w[WCOLS + ["PRED", "ERR_V9", "AE_V9"]].copy()
    d2.columns = WCOLS + ["pred_after", "err_after", "ae_after"]
    note(d2.round(1).to_string(index=False))
    note("")
    note(f"  top-20 |err| sum   before {s.nlargest(20, 'AE_T4')['AE_T4'].sum():,.1f}"
         f"   after {s.nlargest(20, 'AE_V9')['AE_V9'].sum():,.1f}")
    note(f"  worst single |err| before {s['AE_T4'].max():,.1f}   after "
         f"{s['AE_V9'].max():,.1f}   (316-tree arm {s['AE_S'].max():,.1f})")
    note(f"  these 20 rows' share of segment SSE: before "
         f"{100 * s.nlargest(20, 'AE_T4')['SE'].sum() / s['SE'].sum():.2f}%   "
         f"after {100 * s.nlargest(20, 'AE_V9')['SE'].sum() / s['SE_V9'].sum():.2f}%")

    sub(f"8.{fold} - the 20 largest errors SEEN AFTER (any new offenders)")
    w3 = s.sort_values("AE_V9", ascending=False).head(20)
    d3 = w3[WCOLS + ["T4_PRED", "ERR_T4", "AE_T4", "PRED", "ERR_V9",
                     "AE_V9"]].copy()
    d3.columns = WCOLS + ["pred_before", "err_before", "ae_before",
                          "pred_after", "err_after", "ae_after"]
    note(d3.round(1).to_string(index=False))
    note("")
    note(f"  airports of the 20 worst AFTER: "
         f"{dict(w3['ADEP_mvt'].value_counts().head(6))}")

# pooled segment frame, unique integer index
allseg = pd.concat([V["JAN"][V["JAN"]["SEG"]], V["JUL"][V["JUL"]["SEG"]]],
                   ignore_index=True)
allrows = pd.concat([V["JAN"], V["JUL"]], ignore_index=True)

# ============================================================
# 9. SSE CONCENTRATION OF THE IMPROVEMENT
# ============================================================

head("9. SSE CONCENTRATION - is the gain broad, or a handful of anomalies?")

KS = [1, 5, 10, 25, 50, 100]
CONC = {}
for tag, frame in (("JAN", V["JAN"][V["JAN"]["SEG"]]),
                   ("JUL", V["JUL"][V["JUL"]["SEG"]]),
                   ("POOLED Jan+Jul", allseg)):
    tot_before = float(frame["SE"].sum())
    tot_after = float(frame["SE_V9"].sum())
    tot_gain = tot_before - tot_after
    CONC[tag] = dict(before=tot_before, after=tot_after, gain=tot_gain,
                     n=len(frame))
    sub(f"9.{tag} - segment SSE {tot_before:.4e} -> {tot_after:.4e}   "
        f"net gain {tot_gain:+.4e} ({100 * tot_gain / tot_before:+.2f}%)")
    note(f"  rows in this segment: {len(frame):,}")

    # (a) concentration of the gain itself
    ga = frame.copy()
    ga["GAIN"] = ga["SE"] - ga["SE_V9"]
    ga = ga.sort_values("GAIN", key=lambda c: c.abs(), ascending=False)
    note("")
    note("  (a) largest |SSE change| per row - where the gain sits:")
    note(f"  {'k':>5s} {'% of seg rows':>13s} {'% of net gain':>14s} "
         f"{'cum % net gain':>15s} {'SSE drop from these k':>22s}")
    cum = 0.0
    for k in KS:
        if k > len(ga):
            continue
        kg = ga.head(k)
        cum += float(kg["GAIN"].sum())
        note(f"  {k:>5d} {100 * k / len(ga):>12.2f}% "
             f"{100 * float(kg['GAIN'].sum()) / tot_gain:>13.2f}% "
             f"{100 * cum / tot_gain:>14.2f}% "
             f"{float(kg['GAIN'].sum()):>22.4e}")

    # (b) were the pre-existing offenders fixed?
    gb = frame.sort_values("SE", ascending=False)
    note("")
    note("  (b) worst rows by PRE-EXISTING squared error - did the anomalies "
         "improve?")
    note(f"  {'k':>5s} {'% of seg rows':>13s} {'SSE before':>14s} "
         f"{'SSE after':>14s} {'SSE drop':>14s} {'% of net gain':>14s} "
         f"{'cum % net gain':>15s}")
    cum = 0.0
    for k in KS:
        if k > len(gb):
            continue
        kg = gb.head(k)
        drop = float(kg["SE"].sum() - kg["SE_V9"].sum())
        cum += drop
        note(f"  {k:>5d} {100 * k / len(gb):>12.2f}% "
             f"{float(kg['SE'].sum()):>14.4e} "
             f"{float(kg['SE_V9'].sum()):>14.4e} {drop:>14.4e} "
             f"{100 * drop / tot_gain:>13.2f}% {100 * cum / tot_gain:>14.2f}%")

    # (c) is the headline robust to dropping the tail? (DIAGNOSTIC only)
    note("")
    note("  (c) tail-robustness of the segment RMSE "
         "(DIAGNOSTIC, never a headline):")
    note(f"  {'excluded':>22s} {'n left':>9s} {'RMSE before':>13s} "
         f"{'RMSE after':>13s} {'delta':>11s}")
    for k in KS:
        if k >= len(frame):
            continue
        keep = frame.index[~frame.index.isin(gb.head(k).index)]
        gk = frame.loc[keep]
        rb, ra = rse(gk["SE"]), rse(gk["SE_V9"])
        note(f"  {'worst ' + str(k):>22s} {len(gk):>9,} {rb:>13,.2f} "
             f"{ra:>13,.2f} {ra - rb:>+11,.2f}")

    # (d) improved vs worsened
    imp_n = int((frame["SE_V9"] < frame["SE"]).sum())
    wor_n = int((frame["SE_V9"] > frame["SE"]).sum())
    tie_n = len(frame) - imp_n - wor_n
    note("")
    note(f"  rows improved  : {imp_n:>7,} ({100 * imp_n / len(frame):6.2f}%)")
    note(f"  rows worsened  : {wor_n:>7,} ({100 * wor_n / len(frame):6.2f}%)")
    note(f"  rows unchanged : {tie_n:>7,} ({100 * tie_n / len(frame):6.2f}%)")
    note(f"  median |err|   : before {float(np.median(frame['AE_T4'])):,.2f}   "
         f"after {float(np.median(frame['AE_V9'])):,.2f}")
    q = frame["AE_T4"] - frame["AE_V9"]
    note(f"  median improvement in |err| (s) : {float(q.median()):,.2f}")
    note(f"  median deterioration in |err| where worse (s): "
         f"{float(-q[frame['AE_V9'] > frame['AE_T4']].median()) if wor_n else 0:,.2f}")
    note(f"  max deterioration in |err| (s) : {float(-q.min()):,.2f}")
    note(f"  SSE on rows that IMPROVED  : {float(frame.loc[frame['SE_V9'] < frame['SE'], 'SE'].sum()):.4e}"
         f" -> {float(frame.loc[frame['SE_V9'] < frame['SE'], 'SE_V9'].sum()):.4e}")
    note(f"  SSE on rows that WORSENED : "
         f"{float(frame.loc[frame['SE_V9'] > frame['SE'], 'SE'].sum()):.4e}"
         f" -> {float(frame.loc[frame['SE_V9'] > frame['SE'], 'SE_V9'].sum()):.4e}")
    CONC[tag].update(imp=imp_n, wor=wor_n, tie=tie_n,
                     share100=float((gb.head(100)["SE"]
                                     - gb.head(100)["SE_V9"]).sum()) / tot_gain)

sub("9. FOLD-LEVEL SSE ACCOUNTING (identity check)")
for fold in ("JAN", "JUL"):
    f = V[fold]
    ds_total = float(f["SE"].sum()) - float(f["SE_V9"].sum())
    ds_seg = float(f.loc[f["SEG"], "SE"].sum() - f.loc[f["SEG"], "SE_V9"].sum())
    ds_ord = float(f.loc[~f["SEG"], "SE"].sum() - f.loc[~f["SEG"], "SE_V9"].sum())
    note(f"  {fold}: fold SSE change {ds_total:+.6e}   segment {ds_seg:+.6e}   "
         f"ordinary {ds_ord:+.1f}")
    assert ds_ord == 0.0, f"{fold}: ordinary SSE moved"
    assert abs(ds_total - ds_seg) < 1e-6
note("  [PASS] the whole fold-level change comes from the segment; the ordinary")
note("         contribution is exactly zero by construction and by measurement")

# ============================================================
# 10. BY AIRPORT
# ============================================================

head("10. MISSING-DATA RESULTS BY AIRPORT")

FOCUS = ["LIRF", "LFPG"]

sub("10.1 - every airport with at least one segment row, segment rows only")
rows = []
for fold in ("JAN", "JUL"):
    s = V[fold][V[fold]["SEG"]]
    n_fold_seg = int(V[fold]["SEG"].sum())
    for ap, g in s.groupby("ADEP_mvt", observed=True):
        rows.append(dict(
            fold=fold, airport=ap, n=len(g),
            pct_seg_rows=100 * len(g) / n_fold_seg,
            rmse_b=rse(g["SE"]), rmse_a=rse(g["SE_V9"]), rmse_s=rse(g["SE_S"]),
            mae_b=float(g["AE_T4"].mean()), mae_a=float(g["AE_V9"].mean()),
            bias_b=float(g["ERR_T4"].mean()), bias_a=float(g["ERR_V9"].mean()),
            sse_b=float(g["SE"].sum()), sse_a=float(g["SE_V9"].sum()),
            pct_fold_sse_b=100 * g["SE"].sum() / V[fold]["SE"].sum(),
            pct_fold_sse_a=100 * g["SE_V9"].sum() / V[fold]["SE_V9"].sum(),
            n_worse=int((g["SE_V9"] > g["SE"]).sum()),
            n_better=int((g["SE_V9"] < g["SE"]).sum())))
d = pd.DataFrame(rows)
d["d_sse"] = d["sse_a"] - d["sse_b"]
for fold in ("JAN", "JUL"):
    dd = d[d["fold"] == fold].sort_values("sse_b", ascending=False)
    note("")
    note(f"  --- {fold}: segment rows by departure airport, worst SSE first ---")
    note(dd[["airport", "n", "pct_seg_rows", "rmse_b", "rmse_a", "rmse_s",
             "mae_b", "mae_a", "bias_b", "bias_a", "pct_fold_sse_b",
             "pct_fold_sse_a", "d_sse", "n_better", "n_worse"]]
         .round(2).to_string(index=False))

sub("10.2 - FOCUS airports: LIRF and LFPG")
rows = []
for fold in ("JAN", "JUL"):
    f = V[fold]
    s = f[f["SEG"]]
    for ap in FOCUS:
        g = s[s["ADEP_mvt"] == ap]
        allap = f[f["ADEP_mvt"] == ap]
        if not len(g) or not len(allap):
            continue
        oap = allap[~allap["SEG"]]
        rows.append(dict(
            fold=fold, airport=ap, n_seg=len(g), n_airport=len(allap),
            pct_seg_of_airport=100 * len(g) / len(allap),
            rmse_b=rse(g["SE"]), rmse_a=rse(g["SE_V9"]), rmse_s=rse(g["SE_S"]),
            mae_b=float(g["AE_T4"].mean()), mae_a=float(g["AE_V9"].mean()),
            medae_b=float(g["AE_T4"].median()), medae_a=float(g["AE_V9"].median()),
            bias_b=float(g["ERR_T4"].mean()), bias_a=float(g["ERR_V9"].mean()),
            seg_pct_fold_sse_b=100 * g["SE"].sum() / f["SE"].sum(),
            seg_pct_fold_sse_a=100 * g["SE_V9"].sum() / f["SE_V9"].sum(),
            seg_pct_ap_sse_b=100 * g["SE"].sum() / allap["SE"].sum(),
            seg_pct_ap_sse_a=100 * g["SE_V9"].sum() / allap["SE_V9"].sum(),
            ord_rmse_b=rse(oap["SE"]), ord_rmse_a=rse(oap["SE_V9"]),
            tgt_med=float(g["Y"].median()), tgt_max=float(g["Y"].max()),
            dsched_med=float(g["d_SCHED"].abs().median()),
            n_better=int((g["SE_V9"] < g["SE"]).sum()),
            n_worse=int((g["SE_V9"] > g["SE"]).sum())))
note(pd.DataFrame(rows).round(2).to_string(index=False))
note("")
note("  ord_rmse_b / ord_rmse_a are equal by construction: ordinary rows never")
note("  leave the cached T4 model. seg_pct_* show how much of the airport's own")
note("  error is the missing-flight segment, before and after.")

sub("10.3 - pooled segment rows at the FOCUS airports")
rows = []
for ap in FOCUS:
    g = allseg[allseg["ADEP_mvt"] == ap]
    if not len(g):
        continue
    rows.append(dict(
        airport=ap, n=len(g), n_jan=int((g["FOLD"] == "JAN").sum()),
        n_jul=int((g["FOLD"] == "JUL").sum()),
        rmse_b=rse(g["SE"]), rmse_a=rse(g["SE_V9"]), rmse_s=rse(g["SE_S"]),
        mae_b=float(g["AE_T4"].mean()), mae_a=float(g["AE_V9"].mean()),
        medae_b=float(g["AE_T4"].median()), medae_a=float(g["AE_V9"].median()),
        bias_b=float(g["ERR_T4"].mean()), bias_a=float(g["ERR_V9"].mean()),
        pct_seg_sse_b=100 * g["SE"].sum() / allseg["SE"].sum(),
        pct_seg_sse_a=100 * g["SE_V9"].sum() / allseg["SE_V9"].sum(),
        n_better=int((g["SE_V9"] < g["SE"]).sum()),
        n_worse=int((g["SE_V9"] > g["SE"]).sum()),
        sse_b=float(g["SE"].sum()), sse_a=float(g["SE_V9"].sum())))
note(pd.DataFrame(rows).round(2).to_string(index=False))
note("")
note(f"  ALL segment rows pooled: n={len(allseg):,}   "
     f"SSE before {allseg['SE'].sum():.4e}   after {allseg['SE_V9'].sum():.4e}")
note(f"  share of pooled validation SSE held by the segment: before "
     f"{100 * allseg['SE'].sum() / allrows['SE'].sum():.2f}%   after "
     f"{100 * allseg['SE_V9'].sum() / allrows['SE_V9'].sum():.2f}%")
note(f"  share of pooled validation rows: {100 * len(allseg) / len(allrows):.3f}%")
note(f"  (autopsy reference: 1.56% of rows / 61% of SSE before)")

# ============================================================
# 11. WHY / WHERE THE CHANGE COMES FROM
# ============================================================

head("11. DIAGNOSTIC BREAKDOWNS OF THE SEGMENT POPULATION")

sub("11.1 - segment rows by TRUE target size (the target is never a feature)")
EDGES = [900, 1800, 3600, 7200, 14400, 28800, 86400]
LABELS = ["<15min", "15-30min", "30-60min", "1-2h", "2-4h", "4-8h", "8-15h",
          "15-24h", ">24h"]
rows = []
idx = np.digitize(allseg["Y"].to_numpy(), EDGES, right=False)
for i, lab in enumerate(LABELS):
    m = idx == i
    if m.sum() == 0:
        continue
    rows.append(dict(band=lab, n=int(m.sum()),
                     pct_rows=100 * m.sum() / len(allseg),
                     rmse_b=rse(allseg.loc[m, "SE"]),
                     rmse_a=rse(allseg.loc[m, "SE_V9"]),
                     mae_b=float(allseg.loc[m, "AE_T4"].mean()),
                     mae_a=float(allseg.loc[m, "AE_V9"].mean()),
                     bias_b=float(allseg.loc[m, "ERR_T4"].mean()),
                     bias_a=float(allseg.loc[m, "ERR_V9"].mean()),
                     pct_seg_sse_b=100 * allseg.loc[m, "SE"].sum() / allseg["SE"].sum(),
                     pct_seg_sse_a=100 * allseg.loc[m, "SE_V9"].sum() / allseg["SE_V9"].sum()))
note("  POOLED Jan+Jul segment rows by true target size:")
note(pd.DataFrame(rows).round(2).to_string(index=False))

sub("11.2 - segment rows by |d_SCHED| (the only surviving T4 signal)")
rows = []
g = allseg.copy()
g["absd"] = g["d_SCHED"].abs()
for lo, hi, lab in [(-1e18, 600, "<=10min"), (600, 3600, "10min-1h"),
                    (3600, 10800, "1-3h"), (10800, 21600, "3-6h"),
                    (21600, 43200, "6-12h"), (43200, 1e18, ">12h")]:
    m = (g["absd"] > lo) & (g["absd"] <= hi)
    if m.sum() == 0:
        continue
    rows.append(dict(band=lab, n=int(m.sum()), pct_rows=100 * m.sum() / len(g),
                     rmse_b=rse(g.loc[m, "SE"]), rmse_a=rse(g.loc[m, "SE_V9"]),
                     mae_b=float(g.loc[m, "AE_T4"].mean()),
                     mae_a=float(g.loc[m, "AE_V9"].mean()),
                     tgt_med=float(g.loc[m, "Y"].median()),
                     pred_b_med=float(g.loc[m, "T4_PRED"].median()),
                     pred_a_med=float(g.loc[m, "PRED"].median()),
                     pct_seg_sse_b=100 * g.loc[m, "SE"].sum() / g["SE"].sum(),
                     pct_seg_sse_a=100 * g.loc[m, "SE_V9"].sum() / g["SE_V9"].sum()))
note(pd.DataFrame(rows).round(2).to_string(index=False))

sub("11.3 - train vs validation segment target distribution (shift check)")
rows = []
for fold in ("JAN", "JUL"):
    trs = dep[(~HOLD[fold]) & seg_all][TARGET]
    vas = V[fold][V[fold]["SEG"]]["Y"]
    rows.append(dict(
        fold=fold, n_train=len(trs), n_val=len(vas),
        mean_tr=trs.mean(), mean_va=vas.mean(),
        med_tr=trs.median(), med_va=vas.median(),
        p90_tr=trs.quantile(.90), p90_va=vas.quantile(.90),
        p99_tr=trs.quantile(.99), p99_va=vas.quantile(.99),
        max_tr=trs.max(), max_va=vas.max(),
        frac_gt2h_tr=100 * (trs > 7200).mean(),
        frac_gt2h_va=100 * (vas > 7200).mean()))
note(pd.DataFrame(rows).round(2).to_string(index=False))
note("")
note("  The JAN training fold contains no month-1 rows, so 'month' == 1 is")
note("  unseen for the JAN segment model, exactly as it already is for the JAN")
note("  T4 model. This is a property of the existing methodology, not something")
note("  V9-A introduces.")

sub("11.4 - the 316-tree sensitivity arm, side by side")
rows = []
for tag, g in (("JAN", V["JAN"][V["JAN"]["SEG"]]),
               ("JUL", V["JUL"][V["JUL"]["SEG"]]),
               ("POOLED", allseg)):
    rows.append(dict(
        scope=tag, n=len(g), rmse_t4=rse(g["SE"]),
        rmse_primary=rse(g["SE_V9"]), rmse_316=rse(g["SE_S"]),
        mae_t4=float(g["AE_T4"].mean()), mae_primary=float(g["AE_V9"].mean()),
        mae_316=float(g["AE_S"].mean()),
        bias_t4=float(g["ERR_T4"].mean()), bias_primary=float(g["ERR_V9"].mean()),
        bias_316=float(g["ERR_S"].mean())))
note(pd.DataFrame(rows).round(2).to_string(index=False))
note("")
note("  full-fold RMSE with each arm:")
for fold in ("JAN", "JUL"):
    f = V[fold]
    note(f"    {fold}: baseline {rse(f['SE']):.4f}   primary "
         f"{rse(f['SE_V9']):.4f}   316-tree {rse(f['SE_S']):.4f}")
note(f"    POOLED: baseline {POOL['b']:.4f}   primary {POOL['a']:.4f}   "
     f"316-tree {POOL['s']:.4f}")

# ============================================================
# 12. DECISION EVIDENCE
# ============================================================

head("12. DECISION FORENSICS - how fragile is the gain?")

sub("12.1 - pooled RMSE with selected airports' segment rows left on exact T4")
note("  'held on T4' means: pretend the routing did not fire for those rows, so")
note("  their contribution is the cached T4 squared error instead of the segment")
note("  model's. Ordinary rows are identical either way, so this is a clean")
note("  counterfactual on the same rows.")


def pooled_with_mask(frame, revert):
    """frame = allrows (JAN then JUL, original order). revert = bool mask."""
    se = np.where(revert, frame["SE"].to_numpy(), frame["SE_V9"].to_numpy())
    return float(np.sqrt(se.sum() / len(frame)))


base_pool = float(np.sqrt(allrows["SE"].sum() / len(allrows)))
cands = [("(none - this is V9-A)", np.zeros(len(allrows), bool))]
for ap in FOCUS:
    cands.append((f"{ap} segment rows held on T4",
                  (allrows["SEG"] & (allrows["ADEP_mvt"] == ap)).to_numpy()))
cands.append(("LIRF + LFPG segment rows held on T4",
              (allrows["SEG"]
               & allrows["ADEP_mvt"].isin(FOCUS)).to_numpy()))
cands.append(("ALL segment rows held on T4 (= plain T4)",
              allrows["SEG"].to_numpy()))
note("")
note(f"  {'scenario':44s} {'pooled RMSE':>13s} {'delta vs T4':>13s} "
     f"{'delta vs V9-A':>14s}")
for lab, m in cands:
    p = pooled_with_mask(allrows, m)
    note(f"  {lab:44s} {p:>13.4f} {p - base_pool:>+13.4f} {p - POOL['a']:>+14.4f}")
note("")
note("  per fold, LIRF segment rows held on T4:")
for fold in ("JAN", "JUL"):
    f = V[fold]
    m = (f["SEG"] & (f["ADEP_mvt"] == "LIRF")).to_numpy()
    se = np.where(m, f["SE"].to_numpy(), f["SE_V9"].to_numpy())
    p = float(np.sqrt(se.mean()))
    note(f"    {fold}: V9-A {rse(f['SE_V9']):9.4f}   LIRF-reverted {p:9.4f}   "
         f"({p - rse(f['SE_V9']):+8.4f})   plain T4 {rse(f['SE']):9.4f}")

sub("12.2 - stress test: what if the LIRF missing-flight tail is WORSE in a "
    "future period than in 2025?")
lirlf = allseg["ADEP_mvt"] == "LIRF"
lirlf_all = (allrows["SEG"] & (allrows["ADEP_mvt"] == "LIRF")).to_numpy()
lirlf_after = float(allseg.loc[lirlf, "SE_V9"].sum())
lirlf_before = float(allseg.loc[lirlf, "SE"].sum())
rest_after = float(allrows["SE_V9"].sum()) - lirlf_after
rest_seg_after = float(allseg.loc[~lirlf, "SE_V9"].sum())
rest_seg_before = float(allseg.loc[~lirlf, "SE"].sum())
note(f"  LIRF segment rows: {int(lirlf.sum()):,} of {len(allseg):,} "
     f"({100 * lirlf.mean():.1f}% of the segment, "
     f"{100 * lirlf.mean() * len(allseg) / len(allrows):.3f}% of all validation rows)")
note(f"  LIRF segment SSE: before {lirlf_before:.4e}   after {lirlf_after:.4e}   "
     f"({100 * (lirlf_after / lirlf_before - 1):+.1f}%)")
note(f"  non-LIRF SEGMENT SSE: before {rest_seg_before:.4e}   after "
     f"{rest_seg_after:.4e}   ({100 * (rest_seg_after / rest_seg_before - 1):+.1f}%)")
note(f"  => every missing-data row outside LIRF is collectively "
     f"{rest_seg_after - rest_seg_before:+.4e} SSE worse, while ordinary rows")
note(f"     are unchanged by construction. The entire net gain is LIRF.")
note("")
note("  counterfactual: keep the V9-A prediction everywhere EXCEPT scale the")
note("  LIRF segment squared error by k. Break-even k is the point at which")
note("  V9-A stops beating plain T4.")
note("")
note(f"  {'k':>6s} {'LIRF seg SSE':>14s} {'pooled RMSE':>13s} {'vs T4':>10s} "
     f"{'vs V9-A':>10s}")
kk = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
breakeven = None
for k in kk:
    p = float(np.sqrt((rest_after + lirlf_after * k) / len(allrows)))
    note(f"  {k:>6.2f} {lirlf_after * k:>14.4e} {p:>13.4f} {p - base_pool:>+10.4f} "
         f"{p - POOL['a']:>+10.4f}")
    if p > base_pool and breakeven is None:
        lo_k, hi_k = kk[kk.index(k) - 1], k
        for _ in range(40):
            mid = 0.5 * (lo_k + hi_k)
            pm = float(np.sqrt((rest_after + lirlf_after * mid) / len(allrows)))
            if pm > base_pool:
                hi_k = mid
            else:
                lo_k = mid
        breakeven = 0.5 * (lo_k + hi_k)
note("")
note(f"  break-even: V9-A stops beating T4 once the LIRF missing-flight tail is "
     f"{breakeven:.2f}x")
note(f"  worse than 2025. Below that multiplier V9-A is ahead on pooled RMSE.")

sub("12.3 - how many rows carry the gain?")
g = allseg.copy()
g["GAIN"] = g["SE"] - g["SE_V9"]
tot = float(g["GAIN"].sum())
g = g.sort_values("GAIN", ascending=False)
cum = np.cumsum(g["GAIN"].to_numpy())
note(f"  total segment SSE gain: {tot:.4e} over {len(g):,} rows")
for frac in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
    k = max(1, int(round(frac * len(g))))
    note(f"    the top {k:>5,} rows ({100 * k / len(g):5.2f}% of the segment, "
         f"{100 * k / len(allrows):6.4f}% of all validation rows) carry "
         f"{100 * cum[k - 1] / tot:6.2f}% of the gain")
note("")
note(f"  (a share above 100% means the rows OUTSIDE that top-k gave squared")
note(f"  error BACK, so the true gain is even more concentrated than the table")
note(f"  suggests.)")
n_full = int(np.argmax(cum >= tot * 0.999)) + 1
note("")
note(f"  {n_full} rows ({100 * n_full / len(g):.2f}% of the segment, "
     f"{100 * n_full / len(allrows):.4f}% of all validation rows) account for")
note(f"  99.9% of the entire improvement. The other "
     f"{len(g) - n_full:,} segment rows are collectively a net LOSS of "
     f"{tot - cum[n_full - 1]:+.4e} SSE.")

sub("12.4 - pre-existing anti-pattern check (the repo's own filter for the "
    "rejected clock and H1 experiments")
note("  train_catboost_v8_defs.py records that the clock feature set was")
note("  rejected because 'the apparent gain was ~95% attributable to LIRF")
note("  extreme rows, and -2.32 (noise) once LIRF was removed', and that H1 was")
note("  rejected as 'fold-inconsistent (+6.93 JAN, -12.21 JUL) and worse (+8.78)")
note("  once LIRF was removed'. V9-A must be read against that same filter.")
note("")
lirlf_gain = lirlf_before - lirlf_after
rest_gain = tot - lirlf_gain
note(f"  share of the pooled segment gain attributable to LIRF : "
     f"{100 * lirlf_gain / tot:.2f}%")
note(f"  pooled segment gain outside LIRF                     : {rest_gain:+.4e}")
note(f"  as a pooled RMSE, keeping LIRF on T4 and V9-A elsewhere: "
     f"{pooled_with_mask(allrows, lirlf_all):.4f} "
     f"(vs plain T4 {base_pool:.4f}, vs full V9-A {POOL['a']:.4f})")
note("")
note(f"  fold spread of the improvement: JAN {HEAD['JAN']['d']:+.4f}   "
     f"JUL {HEAD['JUL']['d']:+.4f}   ratio "
     f"{abs(HEAD['JUL']['d'] / HEAD['JAN']['d']):.1f}x")
note(f"  segment RMSE reduction: JAN "
     f"{100 * (1 - rse(V['JAN'].loc[V['JAN']['SEG'], 'SE_V9']) / rse(V['JAN'].loc[V['JAN']['SEG'], 'SE'])):+.1f}%   "
     f"JUL "
     f"{100 * (1 - rse(V['JUL'].loc[V['JUL']['SEG'], 'SE_V9']) / rse(V['JUL'].loc[V['JUL']['SEG'], 'SE'])):+.1f}%")
note("")
note("  anti-pattern verdict:")
_lirf_conc = abs(lirlf_gain / tot) * 100
_spread = abs(HEAD["JUL"]["d"] / HEAD["JAN"]["d"])
_anti = []
_anti.append(("gain is LIRF-concentrated", _lirf_conc >= 50.0, f"{_lirf_conc:.1f}%"))
_anti.append(("fold improvement spread >= 5x", _spread >= 5.0, f"{_spread:.1f}x"))
_anti.append(("median missing-data |error| improved",
              float(allseg['AE_V9'].median()) < float(allseg['AE_T4'].median()),
              f"{float(allseg['AE_T4'].median()):.2f} -> "
              f"{float(allseg['AE_V9'].median()):.2f}"))
_anti.append(("majority of missing-data rows improved",
              CONC['POOLED Jan+Jul']['imp'] > CONC['POOLED Jan+Jul']['wor'],
              f"{100 * CONC['POOLED Jan+Jul']['imp'] / len(allseg):.2f}% vs "
              f"{100 * CONC['POOLED Jan+Jul']['wor'] / len(allseg):.2f}%"))
_anti.append(("both focus airports improved",
              all(float(allseg.loc[allseg['ADEP_mvt'] == ap, 'SE_V9'].sum())
                  < float(allseg.loc[allseg['ADEP_mvt'] == ap, 'SE'].sum())
                  and rse(allseg.loc[allseg['ADEP_mvt'] == ap, 'AE_T4'])
                  <= rse(allseg.loc[allseg['ADEP_mvt'] == ap, 'AE_V9'])
                  for ap in FOCUS if (allseg['ADEP_mvt'] == ap).sum() > 20),
              "see 10.3"))
_anti.append((">=90% of gain outside the worst 100 segment rows",
              abs(CONC['POOLED Jan+Jul']['share100']) <= 90.0,
              f"{100 * CONC['POOLED Jan+Jul']['share100']:.1f}%"))
for lab, ok, val in _anti:
    note(f"    [{'TRIGGERED' if ok else 'clear    '}] {lab:52s} {val}")
note("")
note("  (A 'TRIGGERED' row is the reason clock and H1 were rejected. This is a")
note("   diagnostic comparison, not an automatic verdict.)")

head("13. DECISION - machine-checked evidence flags")

ev = []


def flag(label, ok):
    ev.append((label, bool(ok)))


flag("baseline is the exact validated T4 on both folds "
     f"(max drift {max(prov.values()):.1e})",
     max(prov.values()) < 1e-6 and all(HEAD[f]["check"] < 1e-6 for f in HEAD))
flag(f"ordinary predictions bit-identical on both folds "
     f"({IDENT['JAN'][0]:,} + {IDENT['JUL'][0]:,} rows, max|diff| 0.0)",
     all(IDENT[f][2] == 0.0 for f in IDENT))
flag(f"JAN segment SSE reduced ({CONC['JAN']['gain']:+.4e})",
     CONC["JAN"]["gain"] > 0)
flag(f"JUL segment SSE reduced ({CONC['JUL']['gain']:+.4e})",
     CONC["JUL"]["gain"] > 0)
flag(f"JAN fold RMSE improved ({HEAD['JAN']['d']:+.4f})", HEAD["JAN"]["d"] < 0)
flag(f"JUL fold RMSE improved ({HEAD['JUL']['d']:+.4f})", HEAD["JUL"]["d"] < 0)
flag(f"pooled RMSE improved ({POOL['a'] - POOL['b']:+.4f})",
     POOL["a"] - POOL["b"] < 0)
flag(f"316-tree sensitivity also improves pooled "
     f"({POOL['s'] - POOL['b']:+.4f})", POOL["s"] - POOL["b"] < 0)
flag(f"worst 100 segment rows ({100 * 100 / len(allseg):.1f}% of the segment) "
     f"carry {100 * CONC['POOLED Jan+Jul']['share100']:.1f}% of the net gain",
     CONC["POOLED Jan+Jul"]["share100"] < 0.90)
flag(f"more missing-data rows improved ({100 * CONC['POOLED Jan+Jul']['imp'] / len(allseg):.2f}%) "
     f"than worsened ({100 * CONC['POOLED Jan+Jul']['wor'] / len(allseg):.2f}%)",
     CONC["POOLED Jan+Jul"]["imp"] > CONC["POOLED Jan+Jul"]["wor"])
for ap in FOCUS:
    g = allseg[allseg["ADEP_mvt"] == ap]
    if len(g) < 20:
        continue
    flag(f"{ap} segment SSE reduced "
         f"({float(g['SE'].sum()) - float(g['SE_V9'].sum()):+.4e})",
         float(g["SE_V9"].sum()) < float(g["SE"].sum()))
flag(f"pooled segment MAE moved "
     f"{float(allseg['AE_V9'].mean()) - float(allseg['AE_T4'].mean()):+.2f}s",
     True)
for label, ok in ev:
    note(f"  [{'PASS' if ok else 'FAIL'}] {label}")
n_pass = sum(1 for _, ok in ev if ok)
note("")
note(f"  evidence checks: {n_pass} pass, {len(ev) - n_pass} fail, of {len(ev)}")

# ============================================================
# 14. HAND-WRITTEN DECISION
# ============================================================

W("")
W("=" * 100)
W("14. DECISION")
W("=" * 100)
W(f"""
DECISION: MIXED - INVESTIGATE

The structural hypothesis is CONFIRMED; the measured benefit is NOT robust.
V9-A is not promoted to V9, and no submission is built.

WHAT IS ESTABLISHED (high confidence)
  * The routing mechanism is real and safe. Ordinary rows are provably
    untouched: 152,271 JAN + 186,775 JUL predictions are BIT-IDENTICAL to the
    cached exact T4 prediction (max|diff| = 0.0), and the ordinary RMSE, MAE and
    SSE are identical to the last digit. The entire measured change is
    attributable to moving 5,373 rows (1.560% of validation) onto a model
    trained only on training-fold rows in the same regime.
  * Leakage hygiene is clean. The holdout month is absent from both segment
    training sets, and the iteration count comes from a forward-in-time inner
    split of the TRAINING fold (189 trees JAN, 254 trees JUL), never from the
    validation fold. A fixed 316-tree arm (the frozen V8 count) reproduces the
    result to within 0.4 pooled RMSE, so nothing here is an artefact of the
    iteration count.
  * Both folds improve, neither is harmed.
        JAN  404.8803 -> 400.0235  (-4.8568,  -1.20%)
        JUL  513.7560 -> 404.7432  (-109.0128, -21.22%)
        pooled 468.3054 -> 402.6437 (-65.6617, -14.02%)
  * The autopsy's concentration claim is reproduced exactly: the segment is
    1.560% of pooled validation rows and 61.01% of pooled SSE. A dedicated
    model for this regime is the right structural fix.

WHY IT IS NOT 'PROMISING'
  1. The improvement is NOT broad. The worst 100 segment rows - 0.029% of all
     validation rows - carry 102.92% of the net gain. Strip the worst 100 and
     the segment gets WORSE: RMSE 755.60 -> 824.55, and excluding the worst 50
     still leaves it worse (930.26 -> 1004.67). Only when the worst 25 are also
     removed does the remainder turn positive (1129.66 -> 1096.63).
  2. More missing-data rows get WORSE than better, on both folds: JAN 43.28%
     improved / 56.72% worsened, JUL 44.95% / 55.05%, pooled 44.50% / 55.50%.
     Median absolute error worsens on both folds (JAN 262.75 -> 294.89, JUL
     181.34 -> 206.82, pooled 196.50 -> 226.52). Pooled segment MAE improves
     (647.30 -> 596.81) but only because the huge tail moved.
  3. The gain is 100% LIRF. LIRF segment SSE 3.3838e10 -> 1.3840e10, i.e.
     101.5% of the pooled gain from 397 of 5,373 segment rows (7.4% of the
     segment). The 4,976 non-LIRF segment rows are collectively 3.02e08 SSE
     WORSE (+2.5%), and the concentration is not subtle: LFPG MAE
     480.75 -> 552.82 with 343 rows improved vs 537 worsened, and EHAM, EDDF,
     LEBL, LEMD and (in JAN) EGLL all worsen. LFPG is the other autopsy focus
     airport; its pooled segment SSE is flat (1.04898e10 -> 1.04769e10, -0.12%)
     while its MAE and median absolute error both get worse, so it contributes
     nothing to the gain and is a small net loss on any ordinary-error measure.
  4. The fold spread is 22.4x (-4.86 vs -109.01). The JAN improvement is
     negligible AND its segment MAE and median absolute error both worsened.
     An improvement that is 22x larger in one month than another is not a
     robust result, it is a bet on which airports happened to be pathological
     in that month.
  5. This is the SAME signature the repo already rejected twice.
     train_catboost_v8_defs.py records that the clock features were killed
     because "~95% attributable to LIRF extreme rows, and -2.32 (noise) once
     LIRF was removed", and H1 because it was "fold-inconsistent (+6.93 JAN,
     -12.21 JUL) and worse (+8.78) once LIRF was removed". Here 101.5% of the
     gain is LIRF and the fold spread is 22x. Accepting this while having
     rejected those would be an inconsistent standard. Section 12.4 applies
     that filter mechanically and it fires on 3 of 6 criteria (LIRF
     concentration, fold spread, and >100% of the gain inside the worst 100
     segment rows).
  6. The mechanism is fitted on a handful of rows and is brittle by
     construction. Section 11.2 shows the one surviving signal, |d_SCHED|,
     does nothing in the bulk (the 1-3h band, 71.7% of segment rows, RMSE
     986.96 -> 995.09, target median 1,025s) and everything in the outcome
     sits in the 6-12h band (31 rows, 5,037.75 -> 10,562.73, WORSE) and the
     >12h band (23 rows, 29,785.61 -> 15,762.33, better). 54 rows decide the
     whole experiment. The segment model also emits new large errors where T4
     was fine - EGLL 181983342 683.9 -> 11,426.7, EDDF 182240520 387.4 ->
     7,147.5, and a worst single deterioration of 32,818.6s (LIRF 182638171,
     where V9-A predicted 89,687s against a true 39,837s). Section 11.1 shows
     the extreme bands merely REDISTRIBUTE: the 8-15h band's share of segment
     SSE rises 31.49% -> 51.63% while its RMSE barely moves.
  7. The target in this regime is bimodal (median ~950-1,000s, tail at
     40,000-88,000s). In 2025 the LIRF tail happened to satisfy
     d_SCHED ~= target almost exactly (JUL: 93,535s d_SCHED vs 87,170s target;
     73,260 vs 87,361; 58,563 vs 87,186; 64,438 vs 64,437). The segment model
     learned that coincidence. Whether it repeats is not verifiable without
     truthing data, which is out of scope by instruction.

WHY IT IS NOT 'REJECT'
  * The downside is bounded and small. All 4,976 non-LIRF segment rows
    combined lose only 3.02e08 of squared error, so V9-A would cost about
    +0.93 pooled RMSE (+0.20%) if the entire LIRF tail benefit vanished and
    nothing else changed. The routing itself is not harmful.
  * Section 12.2 puts a number on the risk: V9-A stops beating T4 only once the
    LIRF missing-flight tail is {breakeven:.2f}x worse than in 2025. A 1.5x
    degradation of the tail still leaves V9-A ahead.
  * The change is free of leakage, free of extra data, and provably cannot
    disturb the 98.44% of rows that T4 handles. There is no reason it must be
    discarded.

WHAT TO INVESTIGATE NEXT (not started - no V9-B, no second experiment here)
  1. Feature-distribution-only audit of the 2026 missing-flight population
     (ranking.parquet, no targets, no leaderboard): how many such rows exist,
     and what their airport / runway / |d_SCHED| distribution looks like
     relative to 2025. This sizes the exposure without touching truthing.
  2. Decide whether the LIRF d_SCHED ~= target relationship is an operational
     fact about LIRF or a 2025 coincidence. If it cannot be established, the
     only defensible version of this idea is one whose worst case is bounded
     by construction rather than by hope.
  3. Only then decide between leaving V8 frozen, and a bounded variant.

STATUS
  V8 remains the frozen candidate. No submission was created. Nothing was
  committed, pushed, merged or reset. No 2026 data was read by this
  experiment. V9-A is NOT promoted.
""")

# ============================================================
# WRITE
# ============================================================

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print(f"\n[written] {REPORT}  lines={len(out)}  bytes={REPORT.stat().st_size:,}")
print(f"[cache]   read-only, unchanged: {CACHE}")
print("V9-A: no submission, no V8 change, no 2026 data, no promotion to V9.")
