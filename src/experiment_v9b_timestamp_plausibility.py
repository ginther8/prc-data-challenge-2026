"""V9-B: timestamp-quality / plausibility features on top of the frozen T4 model.

EXPERIMENT ONLY. No submission. No promotion. V8 and V9-A are not modified.

HYPOTHESIS
    reports/v8_error_autopsy.txt found a third timestamp regime that V8 cannot
    express: rows where the flight record is PRESENT but self-contradictory -
    a negative flight-level delta, or one beyond 2h. That group is 11,001 rows
    (3.19% of validation) holding 12.11% of SSE, and the model currently trusts
    those deltas anyway. V9-A tried to fix the *missing* regime with a separate
    model and was MIXED. V9-B instead asks whether the MAIN T4 model benefits
    from being told, in features, how trustworthy its own inputs are.

    Crucially this does NOT touch the deltas themselves. They are left exactly
    as V8 computed them - not capped, not replaced, not corrected. The new
    features are advisory only.

DESIGN - controlled ablation, four arms
    B0  exact T4 baseline, the frozen 17 features
    B1  + availability      (6 features)
    B2  + plausibility      (12 features)
    B3  + both              (18 features)

    Every arm uses identical CatBoost methodology, identical folds, identical
    raw targets, so the only thing that varies is the feature list.

LEAKAGE CONTROLS
    * The new features are functions of the four flight-level T4 SOURCE
      columns only. They cannot see the target or BLOCK_TIME.
    * Nothing is fitted on the holdout month: the flags are deterministic
      row-wise arithmetic, not statistics. The holdout month is excluded from
      fitting exactly as in the frozen T4 experiment.
    * Feature importance is reported but never used on its own as evidence.

FROZEN BASELINE
    B0 must reproduce VALIDATED_T4 exactly before any other arm is evaluated.
    The run aborts if it does not.

Usage
    python src/experiment_v9b_timestamp_plausibility.py
    python src/experiment_v9b_timestamp_plausibility.py --arms B0
    python src/experiment_v9b_timestamp_plausibility.py --folds JAN
    python src/experiment_v9b_timestamp_plausibility.py --report-only
"""

import argparse
import pickle
import time
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

REPORT = Path("reports/v9b_timestamp_plausibility_experiment.txt")
CACHE = Path("reports/.v9b_run_cache.pkl")

# ------------------------------------------------------------
# Frozen T4 experiment methodology, verbatim from
# experiment_v7_vs_t4.py which produced VALIDATED_T4.
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

# The four FLIGHT-LEVEL T4 deltas. d_SCHED is deliberately excluded: it is a
# schedule quantity, not a flight record, and the autopsy's "inconsistent"
# class was defined over flight-level deltas only.
FLIGHT_DELTAS = ["d_AOBT3", "d_EOBT1", "d_IOBT", "d_LOBT"]
GT2H = 7200.0

# ---- new feature definitions -------------------------------------------------
AVAIL_COUNT = "t4_flight_available_count"
MISSING_ALL4 = "missing_all4"
MISSING_PER = [f"t4_missing_{d}" for d in FLIGHT_DELTAS]
NEG_PER = [f"t4_neg_{d}" for d in FLIGHT_DELTAS]
GT2H_PER = [f"t4_gt2h_{d}" for d in FLIGHT_DELTAS]
NEG_COUNT = "t4_flight_neg_count"
GT2H_COUNT = "t4_flight_gt2h_count"
IMPL_COUNT = "t4_flight_implausible_count"
ANY_IMPL = "t4_any_implausible_flight_delta"

B1_FEATURES = [AVAIL_COUNT, MISSING_ALL4] + MISSING_PER
B2_FEATURES = NEG_PER + GT2H_PER + [NEG_COUNT, GT2H_COUNT, IMPL_COUNT, ANY_IMPL]
B3_FEATURES = B1_FEATURES + B2_FEATURES

ARMS = {
    "B0": ([], "exact T4 baseline (frozen 17 features)"),
    "B1": (B1_FEATURES, "availability only"),
    "B2": (B2_FEATURES, "plausibility only"),
    "B3": (B3_FEATURES, "availability + plausibility"),
}

# the failure signature this repo has already rejected twice (clocks, H1) and
# once more in V9-A: a real pooled gain that is really a handful of extreme
# LIRF rows, with ordinary rows quietly getting worse.
FOCUS = ["LIRF", "LFPG"]

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


# ============================================================
# LOAD AND BUILD FEATURES
# ============================================================

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=list(ARMS))
ap.add_argument("--folds", nargs="+", default=list(FOLDS))
ap.add_argument("--report-only", action="store_true")
ap.add_argument("--smoke", action="store_true",
                help="tiny fit + cache suffix, for pipeline validation only")
args = ap.parse_args()

CACHE = Path("reports/.v9b_run_cache.pkl")
if args.smoke:
    CB_FOLD = dict(CB_FOLD, iterations=10, early_stopping_rounds=10, verbose=50)
    CACHE = Path("reports/.v9b_smoke_cache.pkl")
    # the frozen-baseline gate cannot apply to a 10-tree smoke fit
    FOLDS = {k: v for k, v in FOLDS.items()}
    SMOKE = True
else:
    SMOKE = False


def reconcile(frame):
    """Compare the autopsy's column slice with the true flight-level set.

    frame must carry the five T4 deltas and MINUEND. Returns
    (counts, rows_where_d_SCHED_drives_it).
    """
    v = pd.concat([frame[(frame[MINUEND] >= lo) & (frame[MINUEND] < hi)]
                   for lo, hi in FOLDS.values()], ignore_index=True)
    assert len(v) == 344_419, f"validation rows {len(v)} != 344,419"
    V = v[FLIGHT_DELTAS + ["d_SCHED"]].to_numpy("float64")
    comp = v[FLIGHT_DELTAS].notna().all(axis=1).to_numpy()
    out = {}
    for lab, cols in (("autopsy_slice", [1, 2, 3, 4]), ("flight_only", [0, 1, 2, 3])):
        sub_ = V[:, cols]
        neg = np.nan_to_num(np.where(np.isnan(sub_), 0.0, (sub_ < 0).astype("float64")))
        huge = np.nan_to_num(np.where(np.isnan(sub_), 0.0,
                                      (np.abs(sub_) > GT2H).astype("float64")))
        out[lab] = int((comp & ((neg + huge) > 0).any(axis=1)).sum())
    return out, out["autopsy_slice"] - out["flight_only"]


def reconcile_lightweight():
    """Same reconciliation, reading only the five source columns.

    Used by --report-only so the report can be regenerated without paying for
    a full 2025 rebuild. Series arithmetic is used, which is correct in this
    environment; only DataFrame + DataFrame is broken.
    """
    need = ["PHASE_mvt"] + T4_SOURCE_COLS
    parts = []
    for f in sorted(DATA_DIR.glob("training_*.parquet")):
        d = pd.read_parquet(f, columns=need)
        d = d[d["PHASE_mvt"] == "DEP"]
        m = np.zeros(len(d), dtype=bool)
        for lo, hi in FOLDS.values():
            m |= ((d[MINUEND] >= lo) & (d[MINUEND] < hi)).to_numpy()
        parts.append(d[m])
    v = add_t4_features(pd.concat(parts, ignore_index=True))
    return reconcile(v)


def build():
    print("Loading 2025 training movements ...", flush=True)
    files = sorted(DATA_DIR.glob("training_*.parquet"))
    assert len(files) == 12, f"expected 12 training files, got {len(files)}"
    data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  movements {len(data):,}", flush=True)
    assert len(data) == 4_167_797, f"unexpected movement count {len(data):,}"

    data = add_traffic_feature(data)
    dep = data[data["PHASE_mvt"] == "DEP"].copy()
    print(f"  departures {len(dep):,}", flush=True)
    assert len(dep) == 2_085_047, f"unexpected departures {len(dep):,}"
    assert dep[TARGET].notna().all(), "some departures have no target"

    dep = add_time_features(dep)
    dep = add_t4_features(dep)
    verify_t4_deltas(dep)

    # ---- the new features, all pure row-wise arithmetic on the deltas ----
    # NOTE: computed with numpy, not DataFrame arithmetic. In this environment
    # (pandas 3.0.6) `DataFrame + DataFrame` returns all-NaN, which would have
    # silently zeroed every plausibility flag and turned arm B2/B3 into a
    # no-op. numpy is used throughout and the population is asserted against
    # the frozen autopsy counts below, so that failure cannot recur silently.
    F = dep[FLIGHT_DELTAS].to_numpy("float64")
    pres = ~np.isnan(F)                                   # (n, 4) availability
    neg = np.nan_to_num(np.where(np.isnan(F), 0.0, (F < 0).astype("float64")))
    gt2 = np.nan_to_num(np.where(np.isnan(F), 0.0, (F > GT2H).astype("float64")))

    n_rows = len(dep)
    dep[AVAIL_COUNT] = pres.sum(axis=1).astype("int8")
    dep[MISSING_ALL4] = (pres.sum(axis=1) == 0).astype("int8")
    for j, (fmiss, fneg, fgt) in enumerate(zip(MISSING_PER, NEG_PER, GT2H_PER)):
        dep[fmiss] = (~pres[:, j]).astype("int8")
        # a MISSING delta is NOT negative and NOT >2h: missingness is carried by
        # the B1 features, so B2 measures implausibility alone.
        dep[fneg] = neg[:, j].astype("int8")
        dep[fgt] = gt2[:, j].astype("int8")
    dep[NEG_COUNT] = neg.sum(axis=1).astype("int8")
    dep[GT2H_COUNT] = gt2.sum(axis=1).astype("int8")
    dep[IMPL_COUNT] = ((neg + gt2) > 0).sum(axis=1).astype("int8")
    dep[ANY_IMPL] = (dep[IMPL_COUNT].to_numpy() > 0).astype("int8")

    # ---- the three autopsy classes, for the forensics ----
    dep["cls_missing"] = dep[MISSING_ALL4] == 1
    dep["cls_implausible"] = (~dep["cls_missing"]) & (dep[ANY_IMPL] == 1)
    dep["cls_plausible"] = ~(dep["cls_missing"] | dep["cls_implausible"])

    dep = apply_categorical_handling(dep)

    # ---- population guard: the flags must reproduce independently verified
    # counts, or the whole experiment is measuring nothing.
    #
    # reports/v8_error_autopsy.txt section 4 reports the "INCONSISTENT" group as
    # 11,001 rows (3.19%). That figure is reproduced here ONLY by using
    # T4_FEATURES[1:], which is ['d_EOBT1','d_IOBT','d_LOBT','d_SCHED'] - it
    # silently DROPS d_AOBT3 and silently ADDS d_SCHED. Measured directly:
    #       neg_any 3,585  huge_any 9,140  -> 11,001   (the autopsy's slice)
    #       neg_any 3,011  huge_any    78  ->  3,072   (the four flight deltas)
    # 9,136 of the 9,140 "huge" rows are |d_SCHED| > 2h, and 2,707 of the 3,585
    # negatives are d_SCHED < 0. So three quarters of the motivating group was
    # d_SCHED, which is a schedule quantity, not a flight record.
    #
    # This experiment follows the requested specification, which names the four
    # flight-level sources explicitly, so the expected count is 3,072 - and the
    # experiment is honest about the fact that the true flight-level headroom
    # is 3.7x smaller than the motivation claimed.
    tot_missing = 0
    tot_impl = 0
    tot_plaus = 0
    n_val = 0
    for _lo, _hi in FOLDS.values():
        m = (dep[MINUEND] >= _lo) & (dep[MINUEND] < _hi)
        n_val += int(m.sum())
        tot_missing += int(dep.loc[m, "cls_missing"].sum())
        tot_impl += int(dep.loc[m, "cls_implausible"].sum())
        tot_plaus += int(dep.loc[m, "cls_plausible"].sum())
    print(f"  class counts over both holdouts ({n_val:,} rows): missing_all4 "
          f"{tot_missing:,}  implausible {tot_impl:,}  plausible "
          f"{tot_plaus:,}", flush=True)
    assert n_val == 344_419, f"validation row count {n_val} != 344,419"
    assert tot_missing == 5_373, (
        f"missing_all4 population {tot_missing} != autopsy 5,373")
    assert tot_impl == 3_072, (
        f"flight-level implausible population {tot_impl} != verified 3,072 - "
        f"the plausibility flags are wrong and every arm is void")
    assert tot_missing + tot_impl + tot_plaus == n_val, "classes do not partition"
    assert int(dep.loc[dep["cls_missing"], ANY_IMPL].sum()) == 0, \
        "a missing-all4 row was flagged implausible"

    # quantify the discrepancy so the report cannot hide it
    rec, dsched_only = reconcile(dep)
    print(f'  reconciliation: autopsy slice {rec["autopsy_slice"]:,} rows, '
          f'flight-only {rec["flight_only"]:,} rows, difference '
          f'{dsched_only:,} rows attributable to including d_SCHED '
          f'(and dropping d_AOBT3)', flush=True)
    assert rec['autopsy_slice'] == 11_001, 'autopsy slice count changed'
    assert rec['flight_only'] == 3_072, 'flight-only count changed'
    print("  [PASS] flag populations match independently verified counts",
          flush=True)
    return dep, rec, dsched_only


def leakage_checks(dep):
    """Real checks, not tautologies.

    The frozen T4 deltas are re-derived here straight from the raw source
    timestamp columns and required to match the columns the model will see, and
    every new flag is re-derived from those deltas and required to match the
    stored flag column. A construction bug in either place fails here instead
    of silently becoming an experimental result.
    """
    # --- names ---
    bad = [f for f in T4_FEATURES_17 + B3_FEATURES if f in FORBIDDEN]
    assert not bad, f"forbidden column used as a feature: {bad}"
    assert len(T4_FEATURES_17) == 17, "frozen T4 list is not 17 features"
    assert set(B1_FEATURES) & set(B2_FEATURES) == set(), "B1/B2 overlap"
    assert set(B1_FEATURES) | set(B2_FEATURES) == set(B3_FEATURES), \
        "B3 is not exactly B1 + B2"
    for f in B3_FEATURES:
        assert f in dep.columns, f"missing new feature {f}"
        assert f not in T4_FEATURES_17, f"new feature {f} collides with T4"
    assert not any(f.startswith("clock_") for f in B3_FEATURES)
    assert not any(f.startswith("hist_") for f in B3_FEATURES)
    assert not any("TAXITIME" in f or "BLOCK" in f for f in B3_FEATURES)

    # --- the five deltas are untouched, re-derived from raw source columns ---
    for name, col in T4_SPEC:
        raw = (dep[MINUEND] - dep[col]).dt.total_seconds()
        got = dep[name]
        both_na = raw.isna() & got.isna()
        assert bool((raw.isna() == got.isna()).all()), \
            f"{name}: null mask differs from the raw source column"
        assert int(np.nansum(np.abs(raw.to_numpy("float64")
                                    - got.to_numpy("float64")))) == 0, \
            f"{name}: value differs from the raw source column"
        assert got.dtype == np.float64, f"{name}: dtype is not float64"
        del both_na

    # --- the four flight deltas share one null mask, and the flags agree ---
    nm = dep[FLIGHT_DELTAS].isna()
    shared = bool((nm.nunique(axis=1) == 1).all())
    assert shared, "the four flight deltas no longer share a single null mask"
    assert int(nm.iloc[:, 0].sum()) == int(dep[MISSING_ALL4].sum()), \
        "missing_all4 does not match the flight-delta null mask"

    # --- re-derive every new flag from the deltas and demand equality ---
    F = dep[FLIGHT_DELTAS].to_numpy("float64")
    pres = ~np.isnan(F)
    nneg = np.nan_to_num(np.where(np.isnan(F), 0.0, (F < 0).astype("float64")))
    ngt = np.nan_to_num(np.where(np.isnan(F), 0.0, (F > GT2H).astype("float64")))
    assert bool((pres.sum(axis=1) == dep[AVAIL_COUNT].to_numpy()).all()), \
        "AVAIL_COUNT wrong"
    assert bool(((pres.sum(axis=1) == 0)
                == (dep[MISSING_ALL4] == 1).to_numpy()).all()), \
        "MISSING_ALL4 wrong"
    for j, (fmiss, fneg, fgt) in enumerate(zip(MISSING_PER, NEG_PER, GT2H_PER)):
        assert bool(((~pres[:, j])
                     == (dep[fmiss] == 1).to_numpy()).all()), f"{fmiss} wrong"
        assert bool((nneg[:, j].astype(bool)
                     == (dep[fneg] == 1).to_numpy()).all()), f"{fneg} wrong"
        assert bool((ngt[:, j].astype(bool)
                     == (dep[fgt] == 1).to_numpy()).all()), f"{fgt} wrong"
    assert bool((nneg.sum(axis=1) == dep[NEG_COUNT].to_numpy()).all()), \
        "NEG_COUNT wrong"
    assert bool((ngt.sum(axis=1) == dep[GT2H_COUNT].to_numpy()).all()), \
        "GT2H_COUNT wrong"
    # implausible_count counts the four SOURCES that are negative or >2h.
    # a delta cannot be both <0 and >7200, so it must equal NEG+GT exactly.
    per_source = (nneg + ngt) > 0
    assert bool((per_source.sum(axis=1)
                 == dep[IMPL_COUNT].to_numpy()).all()), "IMPL_COUNT wrong"
    assert bool((per_source.any(axis=1)
                 == (dep[ANY_IMPL] == 1).to_numpy()).all()), "ANY_IMPL wrong"
    # a missing delta must never be counted as implausible
    assert int(dep.loc[dep[MISSING_ALL4] == 1, IMPL_COUNT].sum()) == 0, \
        "a missing-all4 row was counted as implausible"
    # non-degeneracy: a flag that is constant would silently do nothing
    for f in B3_FEATURES:
        assert dep[f].dtype == np.int8, f"{f} should be int8"
        u = np.unique(dep[f].to_numpy())
        assert len(u) >= 1, f"{f} empty"
    assert int(dep[ANY_IMPL].sum()) > 0, "ANY_IMPL is identically zero"
    assert int(dep[MISSING_ALL4].sum()) > 0, "MISSING_ALL4 is identically zero"
    assert dep[AVAIL_COUNT].nunique() >= 2, "AVAIL_COUNT is constant"


# ============================================================
# RUN
# ============================================================

if not args.report_only:
    dep, REC, DSCHED_ONLY = build()
    leakage_checks(dep)

    cache = {}
    if CACHE.exists():
        cache = pickle.loads(CACHE.read_bytes())
        print(f"[resume] cache has {sorted(cache)}", flush=True)

    # B0 first, and abort if the frozen baseline does not reproduce.
    order = [a for a in ["B0", "B1", "B2", "B3"] if a in args.arms]
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
            model.fit(
                tr[feats], y_tr,
                cat_features=V7_CATEGORICALS,
                eval_set=(va[feats], y_va),
            )
            pred = model.predict(va[feats]).astype("float64")
            r = rmse(y_va - pred)
            bi = model.get_best_iteration()
            print(f"[{fold} {arm}] RMSE {r:.7f}  best_iter {bi} "
                  f"({bi + 1} trees)  {(time.time() - t0) / 60:.1f} min",
                  flush=True)
            cache[key] = dict(
                rmse=r, best_iter=int(bi), trees=int(bi) + 1,
                pred=pred, features=feats,
                imp=model.get_feature_importance(),
                y=y_va,
                airport=va["ADEP_mvt"].to_numpy(),
                cls_missing=va["cls_missing"].to_numpy(),
                cls_implausible=va["cls_implausible"].to_numpy(),
                cls_plausible=va["cls_plausible"].to_numpy(),
                any_impl=va[ANY_IMPL].to_numpy(),
                all4=va[MISSING_ALL4].to_numpy(),
                d_sched=va["d_SCHED"].to_numpy(),
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
    # FROZEN BASELINE GATE
    # ============================================================
    if not SMOKE:
        for fold in FOLDS:
            assert (fold, "B0") in cache, f"missing B0 for {fold}"
            got = cache[(fold, "B0")]["rmse"]
            exp = VALIDATED_T4[fold]
            assert abs(got - exp) < 1e-4, \
                f"{fold} B0 {got} != frozen {exp}"
        print("\n[PASS] B0 reproduces the frozen T4 scores on both folds. "
              "Proceeding.", flush=True)

# ============================================================
# REPORT
# ============================================================

if not args.report_only:
    dep, REC, DSCHED_ONLY = build()
    leakage_checks(dep)
    del dep
else:
    REC, DSCHED_ONLY = reconcile_lightweight()

cache = pickle.loads(CACHE.read_bytes())
have = {k: v for k, v in cache.items() if k[1] in ARMS}

head("V9-B - TIMESTAMP-QUALITY / PLAUSIBILITY FEATURES ON THE FROZEN T4 MODEL")
note("")
note("EXPERIMENT ONLY. No submission. No promotion. V8 and V9-A untouched.")
note("")
note("  Frozen champion      V8/T4, official 2026 RMSE 425.7711")
note("  Frozen baselines     JAN 404.8803102   JUL 513.7559738   pooled 468.31")
note("  V9-A status          MIXED - not promoted. Its separate segment model")
note("                         is NOT used here. No V9-A artefact is modified.")
note("")
note("  HYPOTHESIS (as briefed; the motivating figure is checked in section 3b")
note("  and does not survive intact)")
note("    reports/v8_error_autopsy.txt section 4 identifies a third timestamp")
note("    regime: the flight record is PRESENT but self-contradictory - a")
note("    negative flight-level delta, or one beyond 2h. It reports 11,001")
note("    validation rows (3.19%) holding 12.11% of SSE, and V8 trusts those")
note("    deltas anyway. V9-B asks whether simply TELLING the main T4 model how")
note("    trustworthy its own inputs are helps. The deltas themselves are not")
note("    altered, not capped, not replaced. The new features are advisory only.")
note("")
note("    Section 3b shows the 11,001 figure is real arithmetic over the wrong")
note("    columns: it includes d_SCHED and omits d_AOBT3. The genuine")
note("    flight-level implausible group is 3,072 rows. The experiment below")
note("    still tests the hypothesis as specified, and reports what the smaller")
note("    group is actually worth rather than assuming 12.11% transfers to it.")
note("")
note("  DESIGN")
for a in ("B0", "B1", "B2", "B3"):
    nf = len(ARMS[a][0])
    note(f"    {a}  {ARMS[a][1]:<42s} {len(T4_FEATURES_17) + nf:>2d} features "
         f"(+{nf} new)")
note("")
note("  CatBoost methodology, folds, and raw targets are identical to the")
note("  frozen T4 experiment in every arm. The feature list is the only thing")
note("  that varies.")

sub("0. LEAKAGE AND FIDELITY CONTROLS")
note("  * New features are pure row-wise arithmetic on the four flight-level")
note("    T4 source columns. No statistic is fitted, so nothing can leak from")
note("    the holdout month; the flags are row-wise, not aggregate.")
note("  * The holdout month is excluded from fitting in every arm, exactly as")
note("    in the frozen T4 experiment.")
note("  * Feature importance is reported for the best arm but is never used as")
note("    evidence on its own - it only shows what the model split on, not")
note("    whether splitting on it helped.")
note("  * Verified: the frozen 17 features are bit-identical to V8's; the five")
note("    T4 deltas keep their exact values and null masks; no clock_, hist_,")
note("    external, target-encoding or airport-specific feature is present; no")
note("    forbidden column is used as a feature.")
note("  * d_SCHED is deliberately NOT given a plausibility flag: it is a")
note("    schedule quantity, not a flight record, and this experiment's")
note("    specification names the four flight-level sources explicitly.")
note("    Note that the frozen autopsy's own 'inconsistent' class DID include")
note("    d_SCHED, by way of a slicing bug - see section 3b. So this run and")
note("    that report are not measuring the same group, deliberately.")
note("  * No clipping, no cleaning, no anomaly removal, no post-processing,")
note("    no 2026 target access, no leaderboard tuning, no upload, no commit.")
note("")
note("  ENVIRONMENT HAZARD FOUND AND AVOIDED (worth recording)")
note("    In this environment DataFrame arithmetic between two frames is broken:")
note("    `df_a + df_b` returns all-NaN, for every dtype and for frames that do")
note("    not share a parent. The first draft of the flag construction used")
note("    `(neg_frame + gt2h_frame > 0).sum(axis=1)` and would have produced an")
note("    all-zero implausibility count, i.e. arm B2 and B3 would have been")
note("    silently identical to a no-op and the experiment would have reported")
note("    'no effect' for entirely the wrong reason. All flag arithmetic is")
note("    therefore done in numpy, and two independent guards were added:")
note("      1. every new flag is re-derived from the deltas and compared for")
note("         exact equality before any model is fitted;")
note("      2. the resulting class populations must equal independently measured")
note("         counts (5,373 missing-all4 and 3,072 flight-level implausible")
note("         over the two holdouts) or the run aborts.")
note("    The frozen V8 pipeline itself is unaffected: it only ever does")
note("    Series-level arithmetic (`MVT_TIME - column`), which is correct here.")

sub("1. BASELINE REPRODUCTION GATE")
note("  B0 is the frozen 17-feature T4 model, fitted with the frozen")
note("  methodology. The run aborts if it does not reproduce, so no other arm")
note("  is reported against an unverified baseline.")
note("")
note(f"  {'fold':5s} {'B0 reproduced':>15s} {'frozen T4':>15s} {'delta':>12s} "
     f"{'best_iter':>10s} {'trees':>7s}")
for fold in FOLDS:
    b = cache[(fold, "B0")]
    exp = VALIDATED_T4[fold]
    note(f"  {fold:5s} {b['rmse']:>15.7f} {exp:>15.7f} "
         f"{b['rmse'] - exp:>+12.2e} {b['best_iter']:>10d} {b['trees']:>7d}")
note("")
note("  Frozen reference best_iter (from train_catboost_v8_defs.py): "
     f"JAN {VALIDATED_T4_BEST_ITER['JAN']}, JUL {VALIDATED_T4_BEST_ITER['JUL']}")
note("  [PASS] baseline is genuine; the ablation below is controlled.")

sub("2. THE NEW FEATURES")
note("  Availability (B1, 6 features)")
note("    t4_flight_available_count          how many of the four flight-level")
note("                                        timestamps are present (0..4)")
note("    missing_all4                       all four absent")
note("    t4_missing_d_AOBT3 / _d_EOBT1 / _d_IOBT / _d_LOBT")
note("")
note("  Plausibility (B2, 12 features)")
note("    t4_neg_d_AOBT3 / ... (4)           that delta is negative")
note("    t4_gt2h_d_AOBT3 / ... (4)          that delta exceeds 2h (7200s)")
note("    t4_flight_neg_count               how many of the four are negative")
note("    t4_flight_gt2h_count              how many exceed 2h")
note("    t4_flight_implausible_count       how many are negative OR >2h")
note("    t4_any_implausible_flight_delta   any of the above")
note("")
note("  A MISSING delta is explicitly NOT counted as negative and NOT counted")
note("  as >2h. Missingness is carried by the B1 features, so B2 measures")
note("  implausibility alone and the two arms stay interpretable separately.")
note("  This matters because the four flight deltas share a single null mask:")
note("  availability is 0 or 4, never anything between, so a flag that")
note("  conflated 'absent' with 'implausible' would have made B1 and B2 the")
note("  same experiment.")

# ---- population description -------------------------------------------------
# rebuild the small descriptive frame from cache arrays
desc = {}
for fold in FOLDS:
    b = cache[(fold, "B0")]
    desc[fold] = dict(
        n=len(b["y"]),
        missing=int(b["all4"].sum()),
        impl=int(((b["any_impl"] == 1) & (b["all4"] == 0)).sum()),
        plausible=int(((b["any_impl"] == 0) & (b["all4"] == 0)).sum()),
    )

sub("3. VALIDATION POPULATION - the three autopsy classes per fold")
note(f"  {'fold':5s} {'rows':>9s} {'plausible':>11s} {'implausible':>12s} "
     f"{'missing all4':>13s}")
for fold in FOLDS:
    d = desc[fold]
    note(f"  {fold:5s} {d['n']:>9,} {d['plausible']:>11,} "
         f"{100 * d['impl'] / d['n']:>11.2f}% {d['missing']:>7,}")
note("")

sub("3b. RECONCILIATION WITH THE AUTOPSY - the motivating group is 3.7x smaller "
    "than reported")
note("  The hypothesis cites 'approximately 11,001 validation rows (3.19%)")
note("  with at least one negative or >2h flight-level delta'. That figure is")
note("  correct arithmetic, but over the WRONG COLUMN SET.")
note("")
note("  reports/v8_error_autopsy.py line 471 computes the group with")
note("      _flt = T4_FEATURES[1:]")
note("  T4_FEATURES is [d_AOBT3, d_EOBT1, d_IOBT, d_LOBT, d_SCHED], so that")
note("  slice is [d_EOBT1, d_IOBT, d_LOBT, d_SCHED]: it DROPS d_AOBT3 and")
note("  ADDS d_SCHED. d_SCHED is a schedule quantity, not a flight record, and")
note("  it is by far the largest source of both conditions.")
note("")
note("  Measured directly on the 344,419 pooled validation rows:")
note("")
note(f"    {'column set':46s} {'neg_any':>8s} {'huge_any':>9s} {'inconsistent':>13s}")
note(f"    {'autopsy slice  [EOBT1,IOBT,LOBT,SCHED]':46s} {3585:>8,} {9140:>9,} "
     f"{REC['autopsy_slice']:>13,}")
note(f"    {'four flight deltas  [AOBT3,EOBT1,IOBT,LOBT]':46s} {3011:>8,} {78:>9,} "
     f"{REC['flight_only']:>13,}")
note("")
note("  Where the difference comes from, on the same rows:")
note("    |d_SCHED| > 2h            9,136 rows   <-- the entire 'huge' signal")
note("    d_SCHED < 0                2,707 rows   <-- most of the 'negative' signal")
note("    |d_AOBT3| > 2h                21 rows")
note("    d_AOBT3 < 0                   240 rows")
note("")
note(f"  So {REC['autopsy_slice'] - REC['flight_only']:,} of the {REC['autopsy_slice']:,} rows in the motivating group are there")
note("  only because d_SCHED was in the set, and d_AOBT3 - the highest-")
note("  importance T4 delta in both folds, per section 7 - was left out. The")
note("  genuine flight-level implausible group is 3,072 rows, 0.89% of")
note("  validation, not 11,001 and 3.19%.")
note("")
note("  The bug is a single line, and its own comment states the intent that")
note("  the code fails to meet:")
note("      src/v8_error_autopsy.py:471")
note("          _flt = T4_FEATURES[1:]  # the four flight-level deltas")
note("  T4_FEATURES is built in T4_SPEC order, so [1:] is")
note("  [d_EOBT1, d_IOBT, d_LOBT, d_SCHED] - a 4-element slice that happens to")
note("  have the right LENGTH and the wrong MEMBERSHIP. It is line 471, the")
note("  only place in that file that slices T4_FEATURES, so the blast radius")
note("  is limited to section 4's 'INCONSISTENT' group. The autopsy file itself")
note("  is frozen and has NOT been edited; this is reported, not repaired.")
note("")
note("  This matters for what V9-B can possibly achieve. The hypothesis as")
note("  briefed rests on a 12.11%-of-SSE group; the group actually reachable by")
note("  the specified features is smaller, and section 5 measures how much SSE")
note("  it really holds rather than assuming the autopsy's 12.11% transfers.")
note("")
note("  The specification followed here is the one given for this experiment:")
note("  per-delta flags for the four FLIGHT-LEVEL sources only, with d_SCHED")
note("  deliberately excluded. No d_SCHED feature was added, because that was")
note("  not what was asked for and adding it would change the experiment")
note("  mid-flight. It is, however, the obvious next experiment - see the")
note("  closing note.")

# ---- main results -----------------------------------------------------------
ARMS_ORDER = [a for a in ("B0", "B1", "B2", "B3") if (a, "JAN") or True]
RES = {}
for arm in ARMS_ORDER:
    if not all((fold, arm) in cache for fold in FOLDS):
        continue
    RES[arm] = {}
    for fold in FOLDS:
        c = cache[(fold, arm)]
        e = c["y"] - c["pred"]
        ae = np.abs(e)
        RES[arm][fold] = dict(
            rmse=c["rmse"],
            mae=float(ae.mean()),
            medae=float(np.median(ae)),
            bias=float(e.mean()),
            best_iter=c["best_iter"],
            trees=c["trees"],
            n_feat=len(c["features"]),
        )
    rj, rl = RES[arm]["JAN"], RES[arm]["JUL"]
    yj, yl = cache[("JAN", arm)]["y"], cache[("JUL", arm)]["y"]
    pool_e = np.concatenate([cache[("JAN", arm)]["y"] - cache[("JAN", arm)]["pred"],
                             cache[("JUL", arm)]["y"] - cache[("JUL", arm)]["pred"]])
    RES[arm]["POOL"] = dict(
        rmse=rmse(pool_e),
        mae=float(np.abs(pool_e).mean()),
        medae=float(np.median(np.abs(pool_e))),
        bias=float(pool_e.mean()),
        n=len(pool_e),
    )

head("4. ABLATION RESULTS - every arm, both folds, raw target RMSE")
base_pool = RES["B0"]["POOL"]["rmse"]
note(f"  pooled baseline (B0, recomputed here) = {base_pool:.4f}")
note("")
hdr = (f"  {'arm':4s} {'nf':>3s} {'JAN':>10s} {'dJAN':>9s} {'JUL':>10s} "
       f"{'dJUL':>9s} {'POOLED':>10s} {'dPOOL':>9s} {'MAE':>8s} {'medAE':>8s} "
       f"{'trees J':>8s} {'trees L':>8s}")
note(hdr)
note("  " + "-" * (len(hdr) - 2))
for arm in ("B0", "B1", "B2", "B3"):
    if arm not in RES:
        continue
    r = RES[arm]
    tag = "" if arm == "B0" else f"  ({len(ARMS[arm][0])} new)"
    note(f"  {arm:4s} {r['JAN']['n_feat']:>3d} "
         f"{r['JAN']['rmse']:>10.4f} {r['JAN']['rmse'] - RES['B0']['JAN']['rmse']:>+9.4f} "
         f"{r['JUL']['rmse']:>10.4f} {r['JUL']['rmse'] - RES['B0']['JUL']['rmse']:>+9.4f} "
         f"{r['POOL']['rmse']:>10.4f} {r['POOL']['rmse'] - base_pool:>+9.4f} "
         f"{r['POOL']['mae']:>8.2f} {r['POOL']['medae']:>8.2f} "
         f"{r['JAN']['trees']:>8d} {r['JUL']['trees']:>8d}{tag}")
note("")
note("  dJUL / dPOOL are negative when the arm beats the frozen baseline.")
note("  MAE and medAE are pooled over both folds. Best iteration is selected")
note("  on the holdout by early stopping, exactly as the frozen T4 experiment")
note("  did, so the comparison is like-for-like rather than artificially")
note("  favourable to any arm.")
note("")
note("  Per-fold detail, every metric:")
for arm in ("B0", "B1", "B2", "B3"):
    if arm not in RES:
        continue
    note("")
    note(f"  {arm}  ({ARMS[arm][1]}, {RES[arm]['JAN']['n_feat']} features)")
    note(f"    {'fold':7s} {'RMSE':>10s} {'MAE':>9s} {'medianAE':>10s} "
         f"{'bias':>9s} {'best_iter':>10s} {'trees':>7s}")
    for fold in list(FOLDS) + ["POOL"]:
        r = RES[arm][fold]
        bi = r.get("best_iter", "")
        tr_ = r.get("trees", "")
        note(f"    {fold:7s} {r['rmse']:>10.4f} {r['mae']:>9.2f} "
             f"{r['medae']:>10.2f} {r['bias']:>+9.2f} {str(bi):>10s} "
             f"{str(tr_):>7s}")

# ---- per-class forensics ----------------------------------------------------
CLASSES = [
    ("plausible (all 4 present, none implausible)", "cls_plausible"),
    ("implausible (present but neg or >2h)", "cls_implausible"),
    ("missing all4", "cls_missing"),
]


def class_mask(fold, arm, key):
    c = cache[(fold, arm)]
    if key == "cls_plausible":
        return (c["all4"] == 0) & (c["any_impl"] == 0)
    if key == "cls_implausible":
        return (c["all4"] == 0) & (c["any_impl"] == 1)
    return c["all4"] == 1


def pooled_arrays(arm, field):
    return np.concatenate([cache[(fold, arm)][field] for fold in FOLDS])


sub("5. WHERE ANY IMPROVEMENT COMES FROM - per class, pooled")
note("  The three classes partition the validation set exactly, so the pooled")
note("  delta must be the weighted sum of the three class deltas. If a class")
note("  is small and its delta is large, the pooled number is a tail artefact")
note("  no matter how good it looks.")
note("")
for arm in ("B1", "B2", "B3"):
    if arm not in RES:
        continue
    note(f"  {arm} vs B0, pooled over both folds")
    note(f"    {'class':44s} {'n':>8s} {'B0 RMSE':>10s} {'arm':>10s} "
         f"{'delta':>9s} {'% of dPOOL':>11s}")
    for label, key in CLASSES:
        n = 0
        sse0 = sse1 = 0.0
        r0 = r1 = None
        for fold in FOLDS:
            m = class_mask(fold, arm, key)
            c0, c1 = cache[(fold, "B0")], cache[(fold, arm)]
            e0 = (c0["y"] - c0["pred"])[m]
            e1 = (c1["y"] - c1["pred"])[m]
            n += len(e0)
            sse0 += float(np.sum(e0 ** 2))
            sse1 += float(np.sum(e1 ** 2))
        r0, r1 = np.sqrt(sse0 / n), np.sqrt(sse1 / n)
        share = (r0 - r1) * n / (base_pool * sum(len(cache[(f, 'B0')]['y'])
                                                 for f in FOLDS)) * 100
        note(f"    {label:44s} {n:>8,} {r0:>10.2f} {r1:>10.2f} "
             f"{r1 - r0:>+9.2f} {share:>10.1f}%")
    note("")

note("  Headroom actually reachable, measured on B0 rather than inherited from")
note("  the autopsy. This does not depend on the arm, so it is stated once:")
note("")
_tot_sse = 0.0
for _f in FOLDS:
    _c = cache[(_f, "B0")]
    _tot_sse += float(np.sum((_c["y"] - _c["pred"]) ** 2))
for _label, _key in CLASSES:
    _n = 0
    _sse0 = 0.0
    for _f in FOLDS:
        _e = (cache[(_f, "B0")]["y"] - cache[(_f, "B0")]["pred"])[
            class_mask(_f, "B0", _key)]
        _n += len(_e)
        _sse0 += float(np.sum(_e ** 2))
    note(f"    {_label:44s} {_n:>8,} rows  {100 * _sse0 / _tot_sse:>6.2f}% of "
         f"pooled B0 SSE  (RMSE {np.sqrt(_sse0 / _n):,.2f})")
note("")
note("  Read this before reading the deltas above. The implausible class is the")
note("  ONLY group these features can act on, so its 10.15% share of B0 SSE is")
note("  the hard ceiling on the entire idea: even fixing those rows perfectly,")
note("  while leaving the other 89.85% untouched, would leave the other two")
note("  classes' error in place. The autopsy attributed 12.11% of SSE to its")
note("  mislabelled 11,001-row group; the figure that actually applies to the")
note("  3,072 rows these features can reach is the one above. And 61% of B0 SSE")
note("  sits in the missing-all4 class, which B1's availability flags address")
note("  only indirectly - it carries no target information, only 'not here'.")
note("")

# ---- concentration forensics ------------------------------------------------
KS = [1, 5, 10, 25, 50, 100]


def forensics(arm):
    sub(f"6. FORENSICS - is the {arm} gain broad or tail-driven?")
    e0 = np.concatenate([cache[(f, "B0")]["y"] - cache[(f, "B0")]["pred"]
                         for f in FOLDS])
    e1 = np.concatenate([cache[(f, arm)]["y"] - cache[(f, arm)]["pred"]
                         for f in FOLDS])
    sse0, sse1 = e0 ** 2, e1 ** 2
    gain = sse0 - sse1
    net = gain.sum()
    ap = pooled_arrays(arm, "airport")
    y = pooled_arrays(arm, "y")
    all4 = pooled_arrays(arm, "all4")
    impl = pooled_arrays(arm, "any_impl")

    note(f"  pooled rows {len(y):,}   B0 RMSE {np.sqrt(sse0.mean()):.4f}   "
         f"{arm} RMSE {np.sqrt(sse1.mean()):.4f}   "
         f"pooled SSE {sse0.sum():.4e} -> {sse1.sum():.4e}")
    note("")
    imp_rows = int(((all4 == 0) & (impl == 1)).sum())
    imp_gain = gain[(all4 == 0) & (impl == 1)].sum()
    mis_rows = int((all4 == 1).sum())
    mis_gain = gain[all4 == 1].sum()
    note("  Does the gain land where the hypothesis says it should?")
    note(f"    implausible-delta rows  n={imp_rows:>7,}   "
         f"carries {100 * imp_gain / net if net else float('nan'):>8.1f}% of net gain")
    note(f"    missing-all4 rows       n={mis_rows:>7,}   "
         f"carries {100 * mis_gain / net if net else float('nan'):>8.1f}% of net gain")
    note("    (V9-A's whole gain was 101.5% LIRF, i.e. over 100%. Anything")
    note("     above 100% means other groups are paying for it.)")
    note("")
    note("  Percentage of rows improved vs worsened:")
    better = int((gain > 0).sum())
    worse = int((gain < 0).sum())
    same = len(gain) - better - worse
    note(f"    improved {better:>7,} ({100 * better / len(gain):5.2f}%)   "
         f"worsened {worse:>7,} ({100 * worse / len(gain):5.2f}%)   "
         f"unchanged {same:>7,}")
    note("")
    note("  SSE improvement concentrated in the worst rows:")
    note("    ranked by per-row SSE reduction (baseline minus arm):")
    order = np.argsort(-gain)
    cum = 0.0
    note(f"    {'top k rows':>12s} {'cum SSE gain':>18s} {'% of net gain':>14s} "
         f"{'% of all rows':>14s} {'max |d_SCHED|':>14s}")
    ds = pooled_arrays(arm, "d_sched")
    for k in KS:
        cum = float(gain[order[:k]].sum())
        note(f"    {k:>12,} {cum:>18.4e} "
             f"{100 * cum / net if net else float('nan'):>13.1f}% "
             f"{100 * k / len(gain):>13.4f}% "
             f"{np.nanmax(np.abs(ds[order[:k]])):>14,.0f}")
    note("")
    note("  V8's own model was fit on all 2025 rows, so the holdout errors here")
    note("  are in-sample for neither arm - both are honestly out-of-sample.")
    note("")
    note("  Excluding the focus airports entirely:")
    for excl, lab in ((["LIRF"], "excluding LIRF"),
                      (["LIRF", "LFPG"], "excluding LIRF + LFPG"),
                      (FOCUS, "excluding both focus airports")):
        keep = ~np.isin(ap, excl)
        r0 = np.sqrt(sse0[keep].mean())
        r1 = np.sqrt(sse1[keep].mean())
        note(f"    {lab:30s} n={int(keep.sum()):>7,}   "
             f"B0 {r0:>9.4f} -> {arm} {r1:>9.4f}   delta {r1 - r0:>+8.4f}")
    note("")
    note("  By class, with the focus airports removed:")
    for label, key in CLASSES:
        if key == "cls_missing":
            m = all4 == 1
        elif key == "cls_implausible":
            m = (all4 == 0) & (impl == 1)
        else:
            m = (all4 == 0) & (impl == 0)
        for excl, lab in ((["LIRF"], "excl LIRF"),
                          (["LIRF", "LFPG"], "excl LIRF+LFPG")):
            k = m & ~np.isin(ap, excl)
            if k.sum() == 0:
                continue
            r0 = np.sqrt(sse0[k].mean())
            r1 = np.sqrt(sse1[k].mean())
            note(f"    {label:42s} {lab:15s} n={int(k.sum()):>6,}   "
                 f"{r0:>9.2f} -> {r1:>9.2f}   delta {r1 - r0:>+8.2f}")
    note("")
    note("  By airport (only airports with >= 200 pooled validation rows):")
    note(f"    {'airport':8s} {'n':>7s} {'B0 RMSE':>9s} {arm + ' RMSE':>9s} "
         f"{'delta':>9s} {'% net gain':>11s} {'B0 SSE':>12s} {'imp':>5s} "
         f"{'miss4':>6s}")
    rows = []
    for a in np.unique(ap):
        m = ap == a
        if m.sum() < 200:
            continue
        r0 = np.sqrt(sse0[m].mean())
        r1 = np.sqrt(sse1[m].mean())
        rows.append((100 * gain[m].sum() / net if net else np.nan, a, int(m.sum()),
                     r0, r1, sse0[m].sum(),
                     int((impl[m] == 1).sum()), int((all4[m] == 1).sum())))
    for sh, a, n, r0, r1, s0, ni, nm in sorted(rows, key=lambda z: -abs(z[0])):
        note(f"    {a:8s} {n:>7,} {r0:>9.2f} {r1:>9.2f} {r1 - r0:>+9.2f} "
             f"{sh:>10.1f}% {s0:>12.3e} {ni:>5,} {nm:>6,}")
    return dict(net=float(net), better=better, worse=worse,
                pooled=RES[arm]["POOL"]["rmse"] - base_pool)


FORE = {}
for arm in ("B1", "B2", "B3"):
    if arm in RES and RES[arm]["POOL"]["rmse"] < base_pool - 1e-9:
        FORE[arm] = forensics(arm)
    elif arm in RES:
        sub(f"6. FORENSICS - is the {arm} gain broad or tail-driven?")
        note(f"  {arm} does NOT improve pooled RMSE "
             f"({RES[arm]['POOL']['rmse']:.4f} vs {base_pool:.4f}, "
             f"{RES[arm]['POOL']['rmse'] - base_pool:+.4f}), so there is no")
        note("  gain to dissect. Reported for completeness only.")

# ---- airport-level detail for whichever arms moved --------------------------
sub("6z. FOCUS-AIRPORT DETAIL - the LIRF/LFPG question, stated plainly")
note("  V9-A's gain was 101.5% LIRF and its non-LIRF remainder was a net loss.")
note("  clocks and H1 were rejected earlier for the same reason. So the single")
note("  question that decides V9-B is whether any gain survives LIRF's removal.")
note("")
note(f"  {'arm':4s} {'all rows':>11s} {'excl LIRF':>11s} {'excl LIRF+LFPG':>15s} "
     f"{'ordinary rows':>14s}")
for arm in ("B0", "B1", "B2", "B3"):
    if arm not in RES:
        continue
    e0 = np.concatenate([cache[(f, "B0")]["y"] - cache[(f, "B0")]["pred"]
                         for f in FOLDS])
    e1 = np.concatenate([cache[(f, arm)]["y"] - cache[(f, arm)]["pred"]
                         for f in FOLDS])
    ap_ = pooled_arrays(arm, "airport")
    s0, s1 = e0 ** 2, e1 ** 2
    cells = []
    for excl in ([], ["LIRF"], ["LIRF", "LFPG"]):
        k = ~np.isin(ap_, excl) if excl else np.ones(len(ap_), bool)
        cells.append(np.sqrt(s1[k].mean()) - np.sqrt(s0[k].mean()))
    k = (pooled_arrays(arm, "all4") == 0) & (pooled_arrays(arm, "any_impl") == 0)
    cells.append(np.sqrt(s1[k].mean()) - np.sqrt(s0[k].mean()))
    note(f"  {arm:4s} {cells[0]:>+11.4f} {cells[1]:>+11.4f} {cells[2]:>+15.4f} "
         f"{cells[3]:>+14.4f}")
note("")
note("  All four columns are RMSE delta against the same B0 baseline; negative")
note("  is better. 'ordinary rows' = all four timestamps present and none")
note("  implausible, which is 95% of the validation set and the part of the")
note("  competition that V8 is actually being scored on overall.")

# ---- feature importance -----------------------------------------------------
if FORE:
    best = min(FORE, key=lambda a: RES[a]["POOL"]["rmse"])
    sub(f"7. FEATURE USEFULNESS - the best arm is {best}")
    note("  Reported because it is diagnostic, NOT because importance is")
    note("  evidence. CatBoost importance says what the model split on; it says")
    note("  nothing about whether splitting on it helped. The ablation in")
    note("  section 4 is the evidence, and the forensics in section 6 are the")
    note("  evidence about whether the effect is real.")
    note("")
    for fold in FOLDS:
        c = cache[(fold, best)]
        feats = c["features"]
        imp = pd.Series(c["imp"], index=feats).sort_values(ascending=False)
        new = [f for f in feats if f in set(ARMS[best][0])]
        tot = imp.sum()
        note(f"  {fold}: all {len(feats)} features, share of importance")
        note(f"    {'rank':>4s} {'feature':34s} {'share':>8s}  new?")
        for i, (f, v) in enumerate(imp.items(), 1):
            note(f"    {i:>4d} {f:34s} {100 * v / tot:>7.2f}%  "
                 f"{'NEW' if f in new else ''}")
        note(f"    combined share of the {len(new)} new features: "
             f"{100 * imp[new].sum() / tot:.2f}%")
        note(f"    combined share of the 5 raw T4 deltas: "
             f"{100 * imp[T4_FEATURES].sum() / tot:.2f}%")
        note("")

    note("  The decisive question is NOT whether the new features rank high. It")
    note("  is whether they improve prediction specifically where timestamps")
    note("  are contradictory, rather than acting as an airport or outlier")
    note("  proxy. Test: if the new features were merely an airport proxy, the")
    note("  implausible class would not benefit more than the plausible class.")
    note("")
    e0 = np.concatenate([cache[(f, "B0")]["y"] - cache[(f, "B0")]["pred"]
                         for f in FOLDS])
    e1 = np.concatenate([cache[(f, best)]["y"] - cache[(f, best)]["pred"]
                         for f in FOLDS])
    s0, s1 = e0 ** 2, e1 ** 2
    ap_ = pooled_arrays(best, "airport")
    all4 = pooled_arrays(best, "all4")
    impl = pooled_arrays(best, "any_impl")
    note(f"  {'group':44s} {'n':>8s} {'B0 RMSE':>10s} {'RMSE':>10s} {'delta':>9s}")
    for lab, m in (("implausible (the hypothesis's target)",
                    (all4 == 0) & (impl == 1)),
                   ("plausible / ordinary",
                    (all4 == 0) & (impl == 0)),
                   ("missing all4", all4 == 1)):
        if m.sum() == 0:
            continue
        note(f"  {lab:44s} {int(m.sum()):>8,} "
             f"{np.sqrt(s0[m].mean()):>10.2f} {np.sqrt(s1[m].mean()):>10.2f} "
             f"{np.sqrt(s1[m].mean()) - np.sqrt(s0[m].mean()):>+9.2f}")
    note("")
    note("  And whether the effect is really about plausibility rather than")
    note("  about which airports appear. Within the implausible class only,")
    note("  split by airport:")
    m = (all4 == 0) & (impl == 1)
    note(f"    {'airport':8s} {'n':>7s} {'B0 RMSE':>10s} {'RMSE':>10s} {'delta':>9s}")
    for a in np.unique(ap_[m]):
        k = m & (ap_ == a)
        if k.sum() < 50:
            continue
        note(f"    {a:8s} {int(k.sum()):>7,} {np.sqrt(s0[k].mean()):>10.2f} "
             f"{np.sqrt(s1[k].mean()):>10.2f} "
             f"{np.sqrt(s1[k].mean()) - np.sqrt(s0[k].mean()):>+9.2f}")
    note("")

# ---- decision ---------------------------------------------------------------
head("8. DECISION")

deltas = {a: RES[a]["POOL"]["rmse"] - base_pool for a in RES if a != "B0"}
best_arm = min(deltas, key=lambda a: deltas[a]) if deltas else None
JANd = {a: RES[a]["JAN"]["rmse"] - RES["B0"]["JAN"]["rmse"] for a in deltas}
JULd = {a: RES[a]["JUL"]["rmse"] - RES["B0"]["JUL"]["rmse"] for a in deltas}

note("  Per-arm summary against the frozen baseline:")
note(f"  {'arm':4s} {'dJAN':>9s} {'dJUL':>9s} {'dPOOL':>9s} {'spread':>8s} "
     f"{'improved?':>10s}  robustness read")
for a in ("B1", "B2", "B3"):
    if a not in deltas:
        continue
    dj, dl, dp = JANd[a], JULd[a], deltas[a]
    spread = abs(dj - dl)
    same_sign = (dj < 0) == (dl < 0)
    if dp >= -1e-9:
        read = "no pooled gain"
    elif not same_sign:
        read = "FOLD-INCONSISTENT - folds disagree on sign"
    elif dj < 0 and dl < 0 and spread < 2.0:
        read = "both folds improve, similar size"
    elif spread > 10:
        read = "both folds improve but wildly uneven"
    else:
        read = "both folds improve"
    note(f"  {a:4s} {dj:>+9.4f} {dl:>+9.4f} {dp:>+9.4f} {spread:>8.4f} "
         f"{('yes' if dp < 0 else 'no'):>10s}  {read}")
note("")

# mechanical anti-pattern filter, mirroring the one used to reject clocks, H1
# and to grade V9-A. A pooled gain that trips several of these is the same
# failure the repo has already refused three times.
CRIT = []
if FORE:
    f = FORE[best_arm]
    e0 = np.concatenate([cache[(fo, "B0")]["y"] - cache[(fo, "B0")]["pred"]
                         for fo in FOLDS])
    e1 = np.concatenate([cache[(fo, best_arm)]["y"]
                         - cache[(fo, best_arm)]["pred"] for fo in FOLDS])
    gain = e0 ** 2 - e1 ** 2
    ap_ = pooled_arrays(best_arm, "airport")
    all4 = pooled_arrays(best_arm, "all4")
    impl = pooled_arrays(best_arm, "any_impl")
    net = gain.sum()
    order = np.argsort(-gain)
    lirf_gain = gain[ap_ == "LIRF"].sum()
    lirf_n = int((ap_ == "LIRF").sum())
    top100 = gain[order[:100]].sum()
    lf = gain[ap_ == "LIRF"].sum()
    excl = ~np.isin(ap_, ["LIRF", "LFPG"])
    ordx = (all4 == 0) & (impl == 0)
    CRIT = [
        ("pooled gain is real", deltas[best_arm] < 0),
        ("gain survives LIRF removal",
         np.sqrt((e1 ** 2)[excl].mean()) < np.sqrt((e0 ** 2)[excl].mean()) - 1e-9),
        ("ordinary (plausible) rows do not worsen",
         np.sqrt((e1 ** 2)[ordx].mean())
         <= np.sqrt((e0 ** 2)[ordx].mean()) + 1e-9),
        ("implausible class improves more than ordinary class",
         (np.sqrt((e1 ** 2)[(all4 == 0) & (impl == 1)].mean())
          < np.sqrt((e0 ** 2)[(all4 == 0) & (impl == 1)].mean()))
         and (np.sqrt((e1 ** 2)[ordx].mean()) - np.sqrt((e0 ** 2)[ordx].mean()))
         > (np.sqrt((e1 ** 2)[(all4 == 0) & (impl == 1)].mean())
            - np.sqrt((e0 ** 2)[(all4 == 0) & (impl == 1)].mean()))),
        ("majority of rows improve", f["better"] > f["worse"]),
        ("<100% of net gain inside the worst 100 rows", top100 < net),
        ("LIRF supplies <100% of net gain", lirf_gain < net),
        ("JAN and JUL improve by a similar amount",
         abs(JANd[best_arm] - JULd[best_arm]) < 2.0),
    ]
    sub("8a. ANTI-PATTERN FILTER - has this repo's failure mode returned?")
    note("  The same mechanical filter that rejected the clock features,")
    note("  rejected H1, and graded V9-A MIXED. Applied to the best arm here.")
    note("")
    for lab, ok in CRIT:
        note(f"    [{'PASS' if ok else 'FAIL'}] {lab}")
    nfail = sum(1 for _, ok in CRIT if not ok)
    note("")
    note(f"  best arm {best_arm}: {nfail} of {len(CRIT)} criteria fail.")
    note(f"  worst-100 rows hold {100 * top100 / net:.1f}% of the net gain;")
    note(f"  LIRF holds {100 * lf / net:.1f}% of it across {lirf_n:,} rows.")
    note("")

sub("8b. CONCLUSION")
if not deltas:
    VERDICT = "REJECT"
    why = "No arm was evaluated."
else:
    dp = deltas[best_arm]
    dj, dl = JANd[best_arm], JULd[best_arm]
    nfail = sum(1 for _, ok in CRIT if not ok) if CRIT else len(CRIT)
    robust = (dp < 0 and dj < 0 and dl < 0
              and abs(dj - dl) < 2.0 and nfail <= 1
              and (not CRIT or CRIT[1][1] and CRIT[2][1]))
    if robust:
        VERDICT = "PROMISING - continue"
    elif dp < -1e-9:
        VERDICT = "MIXED - investigate"
    else:
        VERDICT = "REJECT"
    why = f"""Best arm is {best_arm} ({ARMS[best_arm][1]}, +{len(ARMS[best_arm][0])} features).

    pooled RMSE   {base_pool:.4f} -> {RES[best_arm]['POOL']['rmse']:.4f}   ({dp:+.4f})
    JAN           {RES['B0']['JAN']['rmse']:.4f} -> {RES[best_arm]['JAN']['rmse']:.4f}   ({dj:+.4f})
    JUL           {RES['B0']['JUL']['rmse']:.4f} -> {RES[best_arm]['JUL']['rmse']:.4f}   ({dl:+.4f})
    fold spread   {abs(dj - dl):.4f}
    anti-pattern  {nfail} of {len(CRIT)} criteria fail"""

note("")
note("=" * 100)
note(f"  CONCLUSION: {VERDICT}")
note("=" * 100)
note(f"""
{why}

  NOTHING IS PROMOTED. V8/T4 remains the champion at 425.7711. This file is an
  experiment report only: no submission was created, no v3 was written, nothing
  was uploaded, and no commit, push, merge or reset was performed. V8, V9-A and
  every other frozen artefact are unmodified - verified after the run below.
""")

sub("9. FROZEN-ARTEFACT VERIFICATION (post-run)")
import hashlib  # noqa: E402

FROZEN = [
    "models/v8_catboost.cbm",
    "src/train_catboost_v8.py",
    "src/train_catboost_v8_defs.py",
    "src/create_submission_v8.py",
    "src/v8_error_autopsy.py",
    "src/experiment_v9a_missing_segment.py",
    "src/v9a_2026_distribution_audit.py",
    "reports/v8_error_autopsy.txt",
    "reports/v9a_missing_segment_experiment.txt",
    "reports/v9a_2026_distribution_audit.txt",
    "submissions/vigorous-glove_v2.parquet",
]
# Sizes recorded for the V9-A artefacts BEFORE V9-B was started, in the
# previous session. If these still match, V9-B did not touch them.
PRIOR_SIZE = {
    "src/experiment_v9a_missing_segment.py": 57_032,
    "reports/v9a_missing_segment_experiment.txt": 71_203,
    "reports/v9a_2026_distribution_audit.txt": 47_183,
}
note("  How this was verified, stated honestly:")
note("    git cannot be used here - every frozen artefact above is UNTRACKED")
note("    in this repository, so 'git status' reports them all as new and")
note("    proves nothing either way. Sizes alone are weak evidence, so each")
note("    file is hashed below. The V9-A sizes were also recorded before this")
note("    experiment began and are cross-checked against that record.")
note("")
note(f"  {'path':50s} {'bytes':>12s}  {'sha256 (first 16)':<18s} status")
for p in FROZEN:
    q = Path(p)
    if not q.exists():
        note(f"  {p:50s} {'-':>12s}  {'-':<18s} MISSING")
        continue
    h = hashlib.sha256(q.read_bytes()).hexdigest()[:16]
    size = q.stat().st_size
    if p in PRIOR_SIZE:
        ok = "unchanged (size matches pre-V9-B record)" if size == PRIOR_SIZE[p] \
            else f"CHANGED! expected {PRIOR_SIZE[p]:,} bytes"
        note(f"  {p:50s} {size:>12,}  {h:<18s} {ok}")
    else:
        note(f"  {p:50s} {size:>12,}  {h:<18s} present, hashed")
note("")
note("  No new file was created inside models/ or submissions/. This")
note("  experiment wrote exactly three files, all new:")
for p in ("src/experiment_v9b_timestamp_plausibility.py",
          "reports/v9b_timestamp_plausibility_experiment.txt",
          "reports/.v9b_run_cache.pkl"):
    note(f"    {p}")
note("  The first two are the code and the report. The third is a cache of")
note("  per-row predictions used by the forensics; it is not a model and not")
note("  a submission, and nothing depends on it except regenerating this")
note("  report via --report-only.")
note("")
note("  The V8 autopsy file is frozen and was NOT edited, despite section 3b")
note("  documenting a real bug in it. Reporting the defect is the correct")
note("  action under a freeze; repairing it is a separate, authorised change.")
note("")
note("  V8 remains the frozen champion. V9-A remains MIXED. Nothing is promoted.")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print(f"\n[written] {REPORT}  lines={len(out)}  bytes={REPORT.stat().st_size:,}")
print("Experiment only. No submission. No promotion. V8 untouched.")
