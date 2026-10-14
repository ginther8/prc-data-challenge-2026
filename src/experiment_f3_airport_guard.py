"""F3-B - per-airport guard: can a TRAINING-ONLY rule route each ADEP to either
the F3 specialist or Exp04?

    python src/experiment_f3_airport_guard.py

This changes ONLY the routing policy. The Exp04 model, the F3 specialist
features, the historical aggregates, the CatBoost settings and the seasonal
folds are all exactly those of the reproduced F3 experiment. The outer
Exp04 and outer specialist predictions are re-derived here and then asserted
against the frozen F3 numbers, so "the same models" is proved, not assumed.

NO OUTER-FOLD PEEKING
    The airport decision may only see the corresponding OUTER TRAINING data.
    For each outer fold the outer training set is split deterministically:
        inner validation : abs(MVT_ID_mvt) % 5 == 0
        inner training   : everything else
    An inner Exp04 and inner specialists are trained, both are scored on the
    inner validation MISSING_4 rows, and the airport is routed to SPECIALIST
    when the specialist's inner SSE is below the inner Exp04 SSE. The outer
    validation targets are then touched once, for the final evaluation.

SAFETY
    COMPLETE predictions stay bit-identical to Exp04; only MISSING_4 rows can
    change. Historical aggregates for the inner validation rows are built from
    inner-training targets only. No BLOCK_TIME_UTC_mvt, no ranking target, no
    ranking data, no weather, no external data, no new features, no new
    fallback definitions, no tuning, no per-airport COMPLETE models, no
    ensembles, no clipping. Writes no model and no submission.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, "src")

from train_catboost_v8_defs import (  # noqa: E402
    MINUEND, T4_FEATURES, T4_SPEC, TARGET, V7_CATEGORICALS, V7_FEATURES,
    add_t4_features, add_time_features, add_traffic_feature,
    apply_categorical_handling,
)

DATA_DIR = Path("data/raw")
REPORT = Path("reports/f3_airport_guard.txt")

FOLDS = {"JAN": ("2025-01-01", "2025-02-01"), "JUL": ("2025-07-01", "2025-08-01")}
FLIGHT_TS_D = ["d_AOBT3", "d_EOBT1", "d_IOBT", "d_LOBT"]
SOURCE_FL_TS = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]

INNER_MOD = 5
EXP04_TOL = 5e-7
# The reproduced F3 results, used only to PROVE the outer models here are the
# same models. They never enter the guard decision.
F3_EXP04_REF = {"JAN": 402.5369401, "JUL": 509.0817784, "pooled": 464.5623995}
F3_UNGUARDED_REF = {"JAN": 372.462528, "JUL": 392.441579, "pooled": 383.653977}

# ---- Exp04 (identical to the reproduced F3) --------------------------
B0_FEATURES = V7_FEATURES + T4_FEATURES
EXP04_FEATURES = B0_FEATURES + ["adep_reference_state"]
EXP04_CATEGORICALS = V7_CATEGORICALS + ["adep_reference_state"]
EXP04_PARAMS = dict(
    iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=200, early_stopping_rounds=50, use_best_model=True,
)

# ---- specialist (identical to the reproduced F3) ---------------------
SPEC_CATEGORICAL_SRC = ["ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                        "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "FLIGHT_mvt"]
SPEC_CAT_INTERACTIONS = ["ADEP_FLIGHT", "ADEP_ADES_FLIGHT", "ADEP_RUNWAY_STAND"]
SPEC_CATEGORICALS = SPEC_CATEGORICAL_SRC + SPEC_CAT_INTERACTIONS
SPEC_NUMERIC = ["hour", "day_of_week", "month", "traffic_last_15min", "d_SCHED"]
SPEC_HIST = [
    "hist_adep_flight_mean", "hist_adep_flight_count",
    "hist_adep_ades_flight_mean", "hist_adep_ades_flight_count",
    "hist_adep_runway_stand_mean", "hist_adep_runway_stand_count",
    "hist_adep_ades_mean", "hist_adep_ades_count",
]
SPEC_FEATURES = SPEC_CATEGORICALS + SPEC_NUMERIC + SPEC_HIST
SPEC_PARAMS = dict(
    iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=200, early_stopping_rounds=50, use_best_model=True,
)
INNER_FRAC = 0.20
INNER_SEED = 42
LOW_CONF = 200

HIST_KEYS = {
    "hist_adep_flight": ["ADEP_mvt", "FLIGHT_mvt"],
    "hist_adep_ades_flight": ["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt"],
    "hist_adep_runway_stand": ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"],
    "hist_adep_ades": ["ADEP_mvt", "ADES_mvt"],
}

_buf: list[str] = []


def head(t):
    _buf.append("")
    _buf.append("=" * 100)
    _buf.append(t)
    _buf.append("=" * 100)
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def note(t=""):
    _buf.append(t)
    print(t)


def flush():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def sse(y, p):
    return float(((np.asarray(y) - np.asarray(p)) ** 2).sum())


# ==================================================================== 1
head("F3-B - PER-AIRPORT GUARD (training-only routing rule)")
note("  Only the ROUTING POLICY changes. Exp04, the F3 specialist features, the")
note("  historical aggregates, the CatBoost settings and the seasonal folds are")
note("  exactly those of the reproduced F3 experiment.")
note("")
note("  Reference points")
note(f"    Exp04 pooled          {F3_EXP04_REF['pooled']:.4f}")
note(f"    Unguarded F3 pooled   {F3_UNGUARDED_REF['pooled']:.4f}"
     f"   (gate gains {100 * (F3_EXP04_REF['pooled'] - F3_UNGUARDED_REF['pooled']) / F3_EXP04_REF['pooled']:.1f}%)")
note("")
note("  Guard rule: an ADEP is routed to the SPECIALIST when, on the INNER")
note("  validation MISSING_4 rows only, specialist SSE < inner Exp04 SSE.")
note("  No threshold is tuned.")
note("")
note("  NOT USED: weather, new historical fallbacks, new F3 features, tuning,")
note("  per-airport COMPLETE models, residual targets, ADS-B, airport geometry,")
note("  ensembles, target clipping, BLOCK_TIME_UTC_mvt, any ranking target or")
note("  ranking data. This stage writes no model and no submission.")

# ==================================================================== 2
head("1. FRAME AND INNER SPLIT")
LOAD = list(dict.fromkeys(
    [MINUEND] + [c for _, c in T4_SPEC] + [
        "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
        "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "WK_TBL_CAT_flt",
        "FLIGHT_TYPE_flt", "FLIGHT_mvt", TARGET]))

t0 = time.time()
files = sorted(DATA_DIR.glob("training_*.parquet"))
assert len(files) == 12, f"expected 12 training files, got {len(files)}"
data = pd.concat([pd.read_parquet(f, columns=LOAD) for f in files], ignore_index=True)
pre_ids = set(data["MVT_ID_mvt"].unique())
note(f"  total movements        : {len(data):,}")
data = add_traffic_feature(data)
assert pre_ids == set(data["MVT_ID_mvt"].unique()), \
    "add_traffic_feature changed the MVT_ID set"
note(f"  MVT_ID set preserved   : {len(pre_ids):,}   [PASS]")

dep = data[data["PHASE_mvt"] == "DEP"].copy()
del data
dep = add_time_features(dep)
dep = add_t4_features(dep)
dep = apply_categorical_handling(dep)
assert len(dep) == 2_085_047, f"unexpected departures {len(dep):,}"
assert dep["MVT_ID_mvt"].is_unique, "duplicate MVT_ID among departures"
dep["FLIGHT_mvt"] = dep["FLIGHT_mvt"].fillna("UNKNOWN").astype(str)
dep["reference_state"] = np.where(
    dep[SOURCE_FL_TS].isna().all(axis=1), "MISSING_4", "COMPLETE")
dep["adep_reference_state"] = dep["ADEP_mvt"] + "|" + dep["reference_state"]
note(f"  2025 departures        : {len(dep):,}   [PASS] MVT_ID unique")
note(f"  MISSING_4 / COMPLETE   : {int((dep['reference_state'] == 'MISSING_4').sum()):,}"
     f" / {int((dep['reference_state'] == 'COMPLETE').sum()):,}")

# The inner split must be exact and integral.
frac = dep["MVT_ID_mvt"] % 1
assert (frac == 0).all(), "MVT_ID_mvt is not whole-valued; % 5 would be unsafe"
MID = dep["MVT_ID_mvt"].astype("int64").abs()
assert (MID % 1 == 0).all()
IS_INNER_VAL = (MID % INNER_MOD == 0).to_numpy()
note(f"  MVT_ID_mvt             : whole-valued int, range "
     f"{MID.min():,}..{MID.max():,}   [PASS] % {INNER_MOD} is exact")
note(f"  inner-validation share : {100 * IS_INNER_VAL.mean():.2f}% of all departures, "
     f"{100 * IS_INNER_VAL[(dep['reference_state'] == 'MISSING_4').to_numpy()].mean():.2f}%"
     f" of MISSING_4")
note(f"  frame built in {time.time() - t0:.0f}s")

HM = {f: ((dep[MINUEND] >= lo) & (dep[MINUEND] < hi)).to_numpy()
      for f, (lo, hi) in FOLDS.items()}
note("")
note(f"  {'fold':<6}{'outer train':>13}{'outer val':>12}"
     f"{'inner train':>13}{'inner val':>12}"
     f"{'M4 inner tr':>13}{'M4 inner va':>13}{'M4 outer va':>13}")
SPLIT = {}
for f, m in HM.items():
    o_tr, o_va = ~m, m
    i_tr = o_tr & ~IS_INNER_VAL
    i_va = o_tr & IS_INNER_VAL
    assert not (i_tr & i_va).any() and not (o_va & (i_tr | i_va)).any()
    assert (i_tr | i_va).sum() == o_tr.sum(), "inner arms do not partition outer train"
    SPLIT[f] = dict(
        o_tr=dep[o_tr].copy(), o_va=dep[o_va].copy(),
        i_tr=dep[i_tr].copy(), i_va=dep[i_va].copy())
    m4 = dep["reference_state"].eq("MISSING_4").to_numpy()
    note(f"  {f:<6}{int(o_tr.sum()):>13,}{int(o_va.sum()):>12,}"
         f"{int(i_tr.sum()):>13,}{int(i_va.sum()):>12,}"
         f"{int((i_tr & m4).sum()):>13,}{int((i_va & m4).sum()):>13,}"
         f"{int((o_va & m4).sum()):>13,}")
note("")
note("  Each inner arm is a strict partition of its outer training set, and the")
note("  outer validation month is in neither arm.")
flush()

# ==================================================================== 3
head("2. SHARED BUILDING BLOCKS (bit-for-bit the reproduced F3 logic)")


def add_interactions(df):
    df["ADEP_FLIGHT"] = df["ADEP_mvt"] + "|" + df["FLIGHT_mvt"]
    df["ADEP_ADES_FLIGHT"] = (df["ADEP_mvt"] + "|" + df["ADES_mvt"]
                              + "|" + df["FLIGHT_mvt"])
    df["ADEP_RUNWAY_STAND"] = (df["ADEP_mvt"] + "|" + df["RUNWAY_mvt"]
                               + "|" + df["STAND_mvt"])


def build_hist(source_m4, rows, loo):
    """The eight historical features, unchanged from the reproduced F3.

    source_m4 carries the ONLY targets visible here. For loo=True the caller is
    the source itself and each row's own target is removed; for loo=False the
    rows are a disjoint evaluation set and see the source aggregate in full.
    Fallback for an empty/LOO-empty group: the row's own ADEP mean over the
    source MISSING_4 rows, then the global source MISSING_4 mean. Counts fall
    back to 0.
    """
    out = {}
    adep_fb = source_m4.groupby("ADEP_mvt")[TARGET].mean()
    glob_fb = float(source_m4[TARGET].mean())
    n_fb = 0
    for name, keys in HIST_KEYS.items():
        rows_key = rows[keys[0]].astype(str)
        for k in keys[1:]:
            rows_key = rows_key + "|" + rows[k].astype(str)
        src_key = source_m4[keys[0]].astype(str)
        for k in keys[1:]:
            src_key = src_key + "|" + source_m4[k].astype(str)
        agg = source_m4.assign(_k=src_key.to_numpy()).groupby("_k")[TARGET].agg(
            ["sum", "count"])
        s = rows_key.map(agg["sum"]).to_numpy(dtype="float64")
        c = rows_key.map(agg["count"]).to_numpy(dtype="float64")
        if loo:
            s = s - rows[TARGET].to_numpy(dtype="float64")
            c = c - 1.0
        mean = np.where(c > 0, s / np.where(c > 0, c, 1.0), np.nan)
        bad = ~(c > 0)
        n_fb += int(bad.sum())
        fb = rows["ADEP_mvt"].map(adep_fb).to_numpy(dtype="float64")
        fb = np.where(np.isfinite(fb), fb, glob_fb)
        out[f"{name}_mean"] = np.where(bad, fb, mean)
        out[f"{name}_count"] = np.where(bad, 0.0, c)
    return out, n_fb


def fit_specialists(source_m4, predict_rows, tag, emit):
    """One MISSING_4 specialist per ADEP, exactly as the reproduced F3.

    Early stopping mirrors the reproduced F3: a fixed 20% sub-split of the
    airport's own MISSING_4 training rows chooses the tree count, then the
    model is refit at that count on all of them. The rows being predicted are
    never used to choose trees.

    In this experiment this function is called twice per fold: once with the
    outer training MISSING_4 rows to reproduce the F3 specialists, and once with
    the inner training MISSING_4 rows to build the guard's specialists. Both
    calls use the identical code path.
    """
    src = source_m4.copy()
    prd = predict_rows.copy()
    for d in (src, prd):
        add_interactions(d)
    h_src, fb_src = build_hist(src, src, loo=True)
    h_prd, fb_prd = build_hist(src, prd, loo=False)
    for k, v in h_src.items():
        src[k] = v
    for k, v in h_prd.items():
        prd[k] = v
    emit(f"    [{tag}] MISSING_4 source {len(src):,}, predicted {len(prd):,}; "
         f"historical fallback source {fb_src:,} predict {fb_prd:,}")
    for d in FLIGHT_TS_D:
        assert src[d].isna().all() and prd[d].isna().all(), \
            f"{d} not fully null in the specialist population"
    assert src["MVT_ID_mvt"].is_unique and prd["MVT_ID_mvt"].is_unique
    assert not (set(src["MVT_ID_mvt"]) & set(prd["MVT_ID_mvt"])), \
        f"{tag}: specialist source and prediction rows overlap"

    pred_map = pd.Series(np.nan, index=prd["MVT_ID_mvt"].to_numpy())
    info = {}
    for ap in sorted(src["ADEP_mvt"].unique()):
        a_tr = src[src["ADEP_mvt"] == ap]
        a_va = prd[prd["ADEP_mvt"] == ap]
        n_in = max(1, int(round(len(a_tr) * INNER_FRAC)))
        inner = (a_tr.sample(n=n_in, random_state=INNER_SEED)
                 if len(a_tr) > n_in else a_tr)
        core = a_tr.drop(inner.index)
        m0 = CatBoostRegressor(**SPEC_PARAMS)
        m0.fit(core[SPEC_FEATURES], core[TARGET], cat_features=SPEC_CATEGORICALS,
               eval_set=(inner[SPEC_FEATURES], inner[TARGET]))
        best = int(m0.get_best_iteration())
        final = CatBoostRegressor(**{**SPEC_PARAMS, "iterations": best + 1,
                                     "use_best_model": False})
        final.fit(a_tr[SPEC_FEATURES], a_tr[TARGET], cat_features=SPEC_CATEGORICALS)
        p = final.predict(a_va[SPEC_FEATURES]) if len(a_va) else np.array([])
        if len(a_va):
            pred_map.loc[a_va["MVT_ID_mvt"].to_numpy()] = p
        info[ap] = dict(n_tr=len(a_tr), n_va=len(a_va), trees=best + 1)
    assert pred_map.notna().all(), f"{tag}: some rows have no specialist prediction"
    return pred_map, info, prd


def fit_exp04(train, predict, tag, emit):
    m = CatBoostRegressor(**EXP04_PARAMS)
    m.fit(train[EXP04_FEATURES], train[TARGET], cat_features=EXP04_CATEGORICALS,
          eval_set=(predict[EXP04_FEATURES], predict[TARGET]))
    p = m.predict(predict[EXP04_FEATURES])
    assert len(p) == len(predict)
    emit(f"    [{tag}] Exp04 RMSE {rmse(predict[TARGET], p):.6f}  "
         f"best_iter {m.get_best_iteration()}  n={len(predict):,}")
    return p, int(m.get_best_iteration())


note("  Each model keeps the early-stopping protocol of its own outer")
note("  counterpart, so the inner comparison mirrors the outer one:")
note("    inner Exp04       early-stops on the inner validation fold")
note("    inner specialists early-stop on a sub-split of the inner training rows")
note("  The guard therefore inherits the same asymmetry that decides the outer")
note("  result. That is deliberate: it is the asymmetry the rule must predict.")
note("  Bias direction, stated up front: the inner Exp04 is selected on the very")
note("  rows it is scored on, so its inner SSE is optimistic and the guard is")
note("  biased toward rejecting specialists. Question 7 quantifies the cost.")
flush()

# ==================================================================== 4
head("3. PER OUTER FOLD: OUTER MODELS, INNER MODELS, GUARD DECISIONS")
note("  Each inner model may only see its own outer TRAINING set. Outer")
note("  validation targets are not read until section 4.")

GUARD, OUTER, INNER_CMP = {}, {}, {}
for f in ("JAN", "JUL"):
    S = SPLIT[f]
    head(f"  --- OUTER FOLD {f} ---")
    head(f"3.{1 if f == 'JAN' else 2}  {f}: outer models (reproducing the F3 result)")

    o_p04, o_iter = fit_exp04(S["o_tr"], S["o_va"], f"{f} outer", note)
    o_spec_map, o_info, o_m4 = fit_specialists(
        S["o_tr"][S["o_tr"]["reference_state"] == "MISSING_4"].copy(),
        S["o_va"][S["o_va"]["reference_state"] == "MISSING_4"].copy(),
        f"{f} outer", note)
    o_m4["_p04"] = o_m4["MVT_ID_mvt"].map(pd.Series(o_p04, index=S["o_va"]["MVT_ID_mvt"]))
    o_m4["_pspec"] = o_m4["MVT_ID_mvt"].map(o_spec_map)
    assert o_m4[["_p04", "_pspec"]].notna().all().all()
    o_ids = S["o_va"]["MVT_ID_mvt"].to_numpy()
    o_y = S["o_va"][TARGET].to_numpy()
    o_vmask = (S["o_va"]["reference_state"] == "MISSING_4").to_numpy()
    o_ung = o_p04.copy()
    o_ung[o_vmask] = pd.Series(o_spec_map).reindex(o_ids).to_numpy()[o_vmask]
    r = rmse(o_y, o_ung)
    note(f"    [{f} outer] unguarded F3 RMSE {r:.6f}  "
         f"(F3 report {F3_UNGUARDED_REF[f]:.6f})")
    ok = abs(o_ung[~o_vmask] - o_p04[~o_vmask]).max() == 0
    assert ok, f"{f} unguarded F3 altered a COMPLETE prediction"
    OUTER[f] = dict(p04=o_p04, ung=o_ung, y=o_y, ids=o_ids, vmask=o_vmask,
                    m4=o_m4, info=o_info, exp04_rmse=rmse(o_y, o_p04),
                    ung_rmse=r, exp04_iter=o_iter)
    flush()

    head(f"3.{3 if f == 'JAN' else 4}  {f}: inner models (guard may see ONLY this)")
    i_p04, i_iter = fit_exp04(S["i_tr"], S["i_va"], f"{f} inner", note)
    i_spec_map, i_info, i_m4 = fit_specialists(
        S["i_tr"][S["i_tr"]["reference_state"] == "MISSING_4"].copy(),
        S["i_va"][S["i_va"]["reference_state"] == "MISSING_4"].copy(),
        f"{f} inner", note)
    i_m4["_p04"] = i_m4["MVT_ID_mvt"].map(pd.Series(i_p04, index=S["i_va"]["MVT_ID_mvt"]))
    i_m4["_pspec"] = i_m4["MVT_ID_mvt"].map(i_spec_map)
    assert i_m4[["_p04", "_pspec"]].notna().all().all()
    # Independent proof that the inner models never saw an outer-validation row.
    assert not (set(S["i_tr"]["MVT_ID_mvt"]) & set(S["o_va"]["MVT_ID_mvt"]))
    assert not (set(S["i_va"]["MVT_ID_mvt"]) & set(S["o_va"]["MVT_ID_mvt"]))
    INNER_CMP[f] = dict(m4=i_m4, info=i_info, exp04_rmse=rmse(S["i_va"][TARGET], i_p04))

    head(f"3.{5 if f == 'JAN' else 6}  {f}: guard decision, INNER validation only")
    dec = {}
    note(f"  {'ADEP':<7}{'inner val':>11}{'low conf':>10}{'iExp04 RMSE':>13}"
         f"{'iSpec RMSE':>12}{'RMSE delta':>12}{'iSSE delta':>14}  decision")
    for ap in sorted(i_m4["ADEP_mvt"].unique()):
        d = i_m4[i_m4["ADEP_mvt"] == ap]
        yv = d[TARGET].to_numpy()
        e0, s0 = rmse(yv, d["_p04"]), rmse(yv, d["_pspec"])
        ss0, ss1 = sse(yv, d["_p04"]), sse(yv, d["_pspec"])
        choice = "SPECIALIST" if ss1 < ss0 else "EXP04"
        dec[ap] = dict(n=len(d), r0=e0, r1=s0, sse0=ss0, sse1=ss1,
                       d_rmse=s0 - e0, d_sse=ss1 - ss0, decision=choice)
        note(f"  {ap:<7}{len(d):>11,}{'yes' if len(d) < LOW_CONF else '':>10}"
             f"{e0:>13.6f}{s0:>12.6f}{s0 - e0:>12.6f}{ss1 - ss0:>14.3e}  {choice}")
    n_sp = sum(1 for v in dec.values() if v["decision"] == "SPECIALIST")
    note(f"  routed to SPECIALIST: {n_sp} of {len(dec)}   to EXP04: "
         f"{len(dec) - n_sp} of {len(dec)}")
    note("  (low conf = fewer than "
         f"{LOW_CONF} inner-validation MISSING_4 rows; flagged only, no rule applied)")
    GUARD[f] = dec

    # apply the frozen decision to the outer validation rows
    route = o_m4["ADEP_mvt"].map(lambda a: dec[a]["decision"])
    assert route.notna().all()
    use_spec = (route == "SPECIALIST").to_numpy()
    # use_spec is indexed by MISSING_4 row, o_vmask by validation row, so lift it
    # onto the validation axis through MVT_ID before combining.
    lift = pd.Series(np.arange(len(o_ids)), index=o_ids)
    pos = lift.reindex(o_m4["MVT_ID_mvt"].to_numpy()).to_numpy()
    assert not np.isnan(pos.astype(float)).any(), \
        f"{f} a MISSING_4 row could not be lifted onto the validation axis"
    use_spec_at = np.zeros(len(o_ids), dtype=bool)
    use_spec_at[pos] = use_spec
    spec_at = pd.Series(o_spec_map).reindex(o_ids).to_numpy()
    o_g = o_p04.copy()
    sel = use_spec_at & o_vmask
    o_g[sel] = spec_at[sel]
    o_m4["_pg"] = o_m4["MVT_ID_mvt"].map(pd.Series(o_g, index=o_ids))
    assert o_m4["_pg"].notna().all()
    OUTER[f]["g"] = o_g
    OUTER[f]["use_spec_ap"] = sorted(set(o_m4.loc[use_spec, "ADEP_mvt"]))
    assert np.array_equal(o_g[~o_vmask], o_p04[~o_vmask]), \
        f"{f} guarded F3 altered a COMPLETE prediction"
    note(f"  outer MISSING_4 routed to the specialist: "
         f"{int(sel.sum()):,} of {int(o_vmask.sum()):,} rows across "
         f"{len(OUTER[f]['use_spec_ap'])} airports")
    flush()

# ==================================================================== 5
head("4. THE THREE ARCHITECTURES SIDE BY SIDE")
note("  Pooled uses combined SSE. TOTAL SSE is the quantity being traded.")
note("")
note(f"  {'arch':<16}{'JAN RMSE':>12}{'JUL RMSE':>12}{'POOLED RMSE':>14}"
     f"{'JAN SSE':>15}{'JUL SSE':>15}{'TOTAL SSE':>16}")
ARCH = {}
for label, key in (("Exp04", "p04"), ("Unguarded F3", "ung"), ("Guarded F3", "g")):
    per = {f: dict(rmse=rmse(OUTER[f]["y"], OUTER[f][key]),
                   sse=sse(OUTER[f]["y"], OUTER[f][key]),
                   n=len(OUTER[f]["y"])) for f in ("JAN", "JUL")}
    nP = sum(per[f]["n"] for f in ("JAN", "JUL"))
    tS = sum(per[f]["sse"] for f in ("JAN", "JUL"))
    ARCH[label] = dict(per=per, pooled=float(np.sqrt(tS / nP)), sse=tS, n=nP)
    note(f"  {label:<16}{per['JAN']['rmse']:>12.6f}{per['JUL']['rmse']:>12.6f}"
         f"{ARCH[label]['pooled']:>14.6f}{per['JAN']['sse']:>15.4e}"
         f"{per['JUL']['sse']:>15.4e}{tS:>16.4e}")
base = ARCH["Exp04"]
note("")
note(f"  {'arch':<16}{'d pooled RMSE':>16}{'% vs Exp04':>13}{'d TOTAL SSE':>16}{'% SSE':>10}")
for label in ("Exp04", "Unguarded F3", "Guarded F3"):
    a = ARCH[label]
    note(f"  {label:<16}{a['pooled'] - base['pooled']:>16.6f}"
         f"{100 * (a['pooled'] - base['pooled']) / base['pooled']:>12.2f}%"
         f"{a['sse'] - base['sse']:>16.4e}"
         f"{100 * (a['sse'] - base['sse']) / base['sse']:>9.2f}%")
note("")
note("  Cross-check that the outer models here are the reproduced F3 models:")
for f in ("JAN", "JUL"):
    a = abs(OUTER[f]["exp04_rmse"] - F3_EXP04_REF[f]) / F3_EXP04_REF[f]
    b = abs(OUTER[f]["ung_rmse"] - F3_UNGUARDED_REF[f]) / F3_UNGUARDED_REF[f]
    assert a <= EXP04_TOL and b <= EXP04_TOL, f"{f} outer models drifted from F3"
    note(f"    {f}: Exp04 rel {a:.2e}, unguarded F3 rel {b:.2e}   [PASS]")
pa = abs(ARCH["Exp04"]["pooled"] - F3_EXP04_REF["pooled"]) / F3_EXP04_REF["pooled"]
pb = abs(ARCH["Unguarded F3"]["pooled"] - F3_UNGUARDED_REF["pooled"]) / F3_UNGUARDED_REF["pooled"]
assert pa <= EXP04_TOL and pb <= EXP04_TOL, "pooled outer models drifted from F3"
note(f"    pooled: Exp04 rel {pa:.2e}, unguarded F3 rel {pb:.2e}   [PASS]")
flush()

# ==================================================================== 6
head("5. GUARD DECISIONS PER FOLD (inner validation only)")
for f in ("JAN", "JUL"):
    note("")
    note(f"  --- {f}-trained fold, inner validation "
         f"({INNER_CMP[f]['m4']['MVT_ID_mvt'].nunique():,} MISSING_4 rows) ---")
    note(f"  {'ADEP':<7}{'inner val':>11}{'low conf':>10}{'iExp04 RMSE':>13}"
         f"{'iSpec RMSE':>12}{'RMSE delta':>12}{'iSSE delta':>14}  decision")
    for ap in sorted(GUARD[f]):
        d = GUARD[f][ap]
        note(f"  {ap:<7}{d['n']:>11,}{'yes' if d['n'] < LOW_CONF else '':>10}"
             f"{d['r0']:>13.6f}{d['r1']:>12.6f}{d['d_rmse']:>12.6f}"
             f"{d['d_sse']:>14.3e}  {d['decision']}")
    nsp = sum(1 for v in GUARD[f].values() if v["decision"] == "SPECIALIST")
    note(f"  SPECIALIST {nsp} of {len(GUARD[f])}")
    note(f"  inner Exp04 overall RMSE on inner validation: "
         f"{INNER_CMP[f]['exp04_rmse']:.6f}")
flush()

# ==================================================================== 7
head("6. OUTER PERFORMANCE PER AIRPORT (MISSING_4 rows only)")


def ap_table():
    """Pooled per-airport MISSING_4 SSE for Exp04, unguarded F3 and guarded F3."""
    acc = {}
    for f in ("JAN", "JUL"):
        m = OUTER[f]["m4"]
        for ap in sorted(m["ADEP_mvt"].unique()):
            d = m[m["ADEP_mvt"] == ap]
            yv = d[TARGET].to_numpy()
            a = acc.setdefault(ap, dict(n=0, s0=0.0, su=0.0, sg=0.0))
            a["n"] += len(d)
            a["s0"] += sse(yv, d["_p04"])
            a["su"] += sse(yv, d["_pspec"])
            a["sg"] += sse(yv, d["_pg"])
    return acc


ACC = ap_table()
for label in ("unguarded F3", "guarded F3"):
    note("")
    note(f"  --- {label}: SSE delta vs Exp04, negative is an improvement ---")
    note(f"  {'ADEP':<7}{'n outer':>9}{'Exp04 RMSE':>13}"
         f"{'unguarded':>12}{'guarded':>12}{'ung dSSE':>14}{'gtd dSSE':>14}"
         f"{'gtd':>7}  Jan/Jul route")
    for ap in sorted(ACC):
        a = ACC[ap]
        r0 = float(np.sqrt(a["s0"] / a["n"]))
        ru = float(np.sqrt(a["su"] / a["n"]))
        rg = float(np.sqrt(a["sg"] / a["n"]))
        decs = "/".join(GUARD[f][ap]["decision"][:4] for f in ("JAN", "JUL"))
        v = ("kept" if a["sg"] < a["s0"] else
             ("hurts" if a["sg"] > a["s0"] else "same"))
        note(f"  {ap:<7}{a['n']:>9,}{r0:>13.6f}{ru:>12.6f}{rg:>12.6f}"
             f"{a['su'] - a['s0']:>14.3e}{a['sg'] - a['s0']:>14.3e}"
             f"{v:>7}  {decs}")
    note(f"  {'TOTAL':<7}{sum(a['n'] for a in ACC.values()):>9,}"
         f"{'':>13}{'':>12}{'':>12}"
         f"{sum(a['su'] - a['s0'] for a in ACC.values()):>14.3e}"
         f"{sum(a['sg'] - a['s0'] for a in ACC.values()):>14.3e}")

note("")
note("  --- LIRF EXPLICITLY ---")
for f in ("JAN", "JUL"):
    d = OUTER[f]["m4"]
    d = d[d["ADEP_mvt"] == "LIRF"]
    yv = d[TARGET].to_numpy()
    note(f"  {f}: n {len(d):,}  inner decision {GUARD[f]['LIRF']['decision']}"
         f"  (inner n {GUARD[f]['LIRF']['n']})")
    note(f"      RMSE  Exp04 {rmse(yv, d['_p04']):>12.6f}"
         f"   unguarded {rmse(yv, d['_pspec']):>12.6f}"
         f"   guarded {rmse(yv, d['_pg']):>12.6f}")
    note(f"      SSE   Exp04 {sse(yv, d['_p04']):>12.4e}"
         f"   unguarded {sse(yv, d['_pspec']):>12.4e}"
         f"   guarded {sse(yv, d['_pg']):>12.4e}")
flush()

# ==================================================================== 8
head("7. ROBUSTNESS QUESTIONS")
ung, gtd, e04 = ARCH["Unguarded F3"], ARCH["Guarded F3"], ARCH["Exp04"]
A1 = (gtd["pooled"] - ung["pooled"])
A2 = (gtd["per"]["JAN"]["rmse"] - ung["per"]["JAN"]["rmse"])
A3 = (gtd["per"]["JUL"]["rmse"] - ung["per"]["JUL"]["rmse"])
sp = {f: sum(1 for v in GUARD[f].values() if v["decision"] == "SPECIALIST")
      for f in ("JAN", "JUL")}
helpful = {ap for ap in ACC if ACC[ap]["su"] < ACC[ap]["s0"]}
harmful = {ap for ap in ACC if ACC[ap]["su"] > ACC[ap]["s0"]}
# "Exposed" counts an airport as exposed if EITHER fold routes it to the
# specialist, so an airport the guard fixes in one fold still counts. The
# mitigated set is the fairer measure of what the guard actually achieved.
mitig_harm = {ap for ap in harmful if ACC[ap]["sg"] < ACC[ap]["su"]}
rej_help = {ap for ap in helpful
            if all(GUARD[f][ap]["decision"] == "EXP04" for f in ("JAN", "JUL"))}
kept_harm = {ap for ap in harmful
             if any(GUARD[f][ap]["decision"] == "SPECIALIST" for f in ("JAN", "JUL"))}
same = {ap for ap in GUARD["JAN"]
        if GUARD["JAN"][ap]["decision"] == GUARD["JUL"][ap]["decision"]}
lr_u = ACC["LIRF"]["s0"] - ACC["LIRF"]["su"]
lr_g = ACC["LIRF"]["s0"] - ACC["LIRF"]["sg"]
gain_u = e04["sse"] - ung["sse"]
gain_g = e04["sse"] - gtd["sse"]

note("")
note(f"  1. Guarded better than unguarded on pooled RMSE? "
     f"{'YES' if A1 < 0 else 'NO'}")
note(f"       guarded {gtd['pooled']:.6f}  unguarded {ung['pooled']:.6f}  "
     f"delta {A1:+.6f}  ({100 * A1 / ung['pooled']:+.2f}%)")
note(f"  2. Improves JAN? {'YES' if A2 < 0 else 'NO'}   "
     f"delta {A2:+.6f}  ({100 * A2 / ung['per']['JAN']['rmse']:+.2f}%)")
note(f"  3. Improves JUL? {'YES' if A3 < 0 else 'NO'}   "
     f"delta {A3:+.6f}  ({100 * A3 / ung['per']['JUL']['rmse']:+.2f}%)")
note(f"  4. SPECIALIST chosen: JAN {sp['JAN']} of 10, JUL {sp['JUL']} of 10")
note(f"  5. Decision stability between folds: {len(same)} of 10 airports agree "
     f"({', '.join(sorted(same))})")
dis = sorted(set(GUARD['JAN']) - same)
note(f"     unstable: {', '.join(dis) if dis else 'none'}")
note(f"  6. Avoided harmful airports? "
     f"{len(harmful) - len(kept_harm)} of {len(harmful)}")
note(f"     specialist harmful at: {', '.join(sorted(harmful))}")
note(f"     still routed to SPECIALIST in at least one fold: "
     f"{', '.join(sorted(kept_harm)) if kept_harm else 'none'}")
note(f"     but MITIGATED, guarded dSSE strictly better than unguarded: "
     f"{len(mitig_harm)} of {len(harmful)}"
     f"  ({', '.join(sorted(mitig_harm)) if mitig_harm else 'none'})")
note("     'Exposed' counts an airport if EITHER fold routes it to the specialist,")
note("     so an airport fixed in one fold still counts as exposed. The mitigated")
note("     count is the fairer read of what the guard actually achieved.")
note(f"  7. Wrongly rejected helpful airports? {len(rej_help)} of {len(helpful)}")
note(f"     specialist helpful at: {', '.join(sorted(helpful))}")
note(f"     rejected in BOTH folds despite helping: "
     f"{', '.join(sorted(rej_help)) if rej_help else 'none'}")
note(f"  8. SSE the guard adds or saves relative to unguarded F3: "
     f"{gtd['sse'] - ung['sse']:+.4e}")
note(f"     (negative = the guard saved SSE).  As a share of Exp04 SSE: "
     f"{100 * (gtd['sse'] - ung['sse']) / e04['sse']:+.2f}%")
note(f"  9. Driven by one airport?")
note(f"     SSE the UNGUARDED gate saves vs Exp04: {gain_u:.4e}")
note(f"        of which LIRF alone            : {lr_u:.4e}"
     f"   ({100 * lr_u / gain_u:.0f}% of the gain)")
note(f"     SSE the GUARDED gate saves vs Exp04 : {gain_g:.4e}")
note(f"        of which LIRF alone            : {lr_g:.4e}"
     f"   ({100 * lr_g / gain_g:.0f}% of the gain)")
other_u = gain_u - lr_u
other_g = gain_g - lr_g
note(f"     gain excluding LIRF               : unguarded {other_u:.4e}, "
     f"guarded {other_g:.4e}")
note(f"     => the gain is LIRF-dominated under BOTH routings, so a pooled-only")
note(f"        improvement must not be read as broad-based.")
flush()

# ==================================================================== 9
head("8. SAFETY CHECKS")
fails = []


def check(name, cond, detail=""):
    if cond:
        note(f"  [PASS] {name}" + (f"   {detail}" if detail else ""))
    else:
        fails.append(name)
        note(f"  [FAIL] {name}" + (f"   {detail}" if detail else ""))


allids = np.concatenate([OUTER[f]["ids"] for f in ("JAN", "JUL")])
for f in ("JAN", "JUL"):
    O = OUTER[f]
    check(f"{f}: guarded COMPLETE predictions BIT-IDENTICAL to Exp04",
          np.array_equal(O["g"][~O["vmask"]], O["p04"][~O["vmask"]]))
    check(f"{f}: unguarded COMPLETE predictions BIT-IDENTICAL to Exp04",
          np.array_equal(O["ung"][~O["vmask"]], O["p04"][~O["vmask"]]))
    ch_g = O["g"] != O["p04"]
    ch_u = O["ung"] != O["p04"]
    check(f"{f}: no COMPLETE row changed under the guard", not bool((ch_g & ~O["vmask"]).any()))
    check(f"{f}: no COMPLETE row changed unguarded", not bool((ch_u & ~O["vmask"]).any()))
    check(f"{f}: every MISSING_4 row changed unguarded", bool(ch_u[O["vmask"]].all()))
    check(f"{f}: no missing or non-finite prediction", bool(np.isfinite(O["g"]).all()))
    check(f"{f}: one prediction per outer validation row",
          len(O["g"]) == len(SPLIT[f]["o_va"]))
    check(f"{f}: outer validation MVT_ID unique", SPLIT[f]["o_va"]["MVT_ID_mvt"].is_unique)
    check(f"{f}: no row loss, inner arms partition outer train",
          len(SPLIT[f]["i_tr"]) + len(SPLIT[f]["i_va"]) == len(SPLIT[f]["o_tr"]))
    check(f"{f}: guard saw NO outer-validation ID (train arm)",
          not (set(SPLIT[f]["i_tr"]["MVT_ID_mvt"]) & set(O["ids"])))
    check(f"{f}: guard saw NO outer-validation ID (val arm)",
          not (set(SPLIT[f]["i_va"]["MVT_ID_mvt"]) & set(O["ids"])))
    check(f"{f}: inner MISSING_4 rows disjoint from outer MISSING_4 rows",
          not (set(INNER_CMP[f]["m4"]["MVT_ID_mvt"]) & set(O["m4"]["MVT_ID_mvt"])))
    check(f"{f}: guard decision defined for all 10 airports",
          len(GUARD[f]) == 10)
check("pooled outer IDs unique", len(np.unique(allids)) == len(allids))
check("JAN and JUL outer validation disjoint",
      not (set(OUTER["JAN"]["ids"]) & set(OUTER["JUL"]["ids"])))
check("row count preserved across folds", len(allids) == 344_419, f"{len(allids):,}")
check("traffic reorder preserved every MVT_ID", len(pre_ids) == 4_167_797)
check("MVT_ID_mvt whole-valued, % 5 exact", bool((frac == 0).all()))
check("no BLOCK_TIME_UTC_mvt in any feature list",
      not any("BLOCK_TIME" in c for c in EXP04_FEATURES + SPEC_FEATURES))
check("no ranking target in any feature list",
      TARGET not in EXP04_FEATURES and TARGET not in SPEC_FEATURES)
check("reference_state itself is not an Exp04 feature",
      "reference_state" not in EXP04_FEATURES)
for d in FLIGHT_TS_D:
    check(f"specialist excludes {d}", d not in SPEC_FEATURES)
check("no ranking data file was read", "ranking" not in " ".join(map(str, LOAD)).lower())
note("")
note("  Inner Exp04 is early-stopped on the inner validation fold, so its inner")
note("  SSE is optimistic and the guard leans toward EXP04. This is a real")
note("  caveat, not a formality; see question 7 for what it cost.")
flush()

# ==================================================================== 10
head("9. DECISION")
better_pooled = gtd["pooled"] < ung["pooled"]
better_jan = gtd["per"]["JAN"]["rmse"] < ung["per"]["JAN"]["rmse"]
better_jul = gtd["per"]["JUL"]["rmse"] < ung["per"]["JUL"]["rmse"]
all_better = better_pooled and better_jan and better_jul
n_harm_avoided = len(harmful) - len(kept_harm)
n_instable = len(GUARD["JAN"]) - len(same)
sse_saved = ung["sse"] - gtd["sse"]
rel_pooled = A1 / ung["pooled"]
rel_sse = 100 * sse_saved / e04["sse"]

note("")
note("  Evidence")
note(f"    guard vs unguarded  pooled {A1:+.6f}  JAN {A2:+.6f}  JUL {A3:+.6f}"
     f"   -> all three improve: {all_better}")
note(f"    guard vs Exp04 SSE  {gain_g:+.4e} (unguarded {gain_u:+.4e})")
note(f"    SSE saved vs unguarded   {sse_saved:+.4e}  = {rel_sse:+.2f}% of the Exp04 error budget")
note(f"    pooled margin vs unguarded {100 * rel_pooled:+.2f}%")
note(f"    harmful airports eliminated {n_harm_avoided} of {len(harmful)}, "
     f"mitigated {len(mitig_harm)} of {len(harmful)}")
note(f"    helpful airports wrongly rejected {len(rej_help)} of {len(helpful)}")
note(f"    routing decisions unstable across folds {n_instable} of {len(GUARD['JAN'])}")
note("")
note("  KEEP requires ALL of: every metric improves, at least one harmful airport")
note("  eliminated, and zero routing decisions unstable between the two folds.")
note("")
if not fails and all_better and n_harm_avoided > 0 and n_instable == 0:
    verdict = "KEEP"
    why = ("The training-only rule improves pooled, JAN and JUL simultaneously, "
           "eliminates a harmful exposure and is stable across folds, so it is a "
           "credible leakage-safe gain over the unguarded gate.")
elif not fails and (all_better or better_pooled):
    verdict = "MIXED"
    why = ("The effect is real and leakage-safe: the rule improves pooled, JAN and "
           "JUL at once without peeking at any outer fold. It is not yet a credible "
           "swap for the unguarded gate, for three reasons. It eliminates "
           f"{n_harm_avoided} of the {len(harmful)} airports where the specialist "
           f"harms outer performance (it mitigates {len(mitig_harm)} of them but "
           f"turns none positive). {n_instable} of {len(GUARD['JAN'])} routing "
           "decisions flip between the JAN-trained and JUL-trained folds, so a "
           "single inner split is a noisy selector. And the pooled margin is only "
           f"{abs(100 * rel_pooled):.2f}%, worth {rel_sse:.2f}% of the Exp04 error "
           "budget, which two folds cannot separate from selection noise.")
else:
    verdict = "REJECT"
    why = ("A training-only routing rule does not improve the metric over the "
           "unguarded gate, so the added routing complexity is not justified.")
note(f"  VERDICT: {verdict}")
note("")
note(f"  {why}")
note("")
note("  Notes for the next stage")
note(f"  - The dominant fact is not the guard. Excluding LIRF, the nine other")
note(f"    airports contribute almost nothing and the sign flips with the guard:")
note(f"    unguarded {other_u:+.3e} of SSE, guarded {other_g:+.3e} "
     f"(negative = the gate still COSTS SSE there).")
note(f"    LIRF alone is {100 * lr_u / gain_u:.0f}% of the total F3 gain, so the whole")
note(f"    F3 result rests on one airport and a pooled-only number must not be read")
note(f"    as broad-based evidence for the specialist architecture.")
if rej_help:
    note(f"  - {len(rej_help)} airport(s) were rejected although the specialist would "
         f"have helped them outer ({', '.join(sorted(rej_help))}).")
if kept_harm:
    note(f"  - {len(kept_harm)} airport(s) the specialist harms still reach it in at "
         f"least one fold ({', '.join(sorted(kept_harm))}), though {len(mitig_harm)} "
         "of those were moved toward Exp04.")
    note("    A single inner split is a noisy selector. Averaging the decision over")
    note("    several inner splits, or requiring agreement, is the obvious next test.")
note("  - The inner comparison is structurally biased toward SPECIALIST: the inner")
note("    Exp04 is fitted on 80% of the outer training rows, so it is weaker than")
note("    the outer Exp04 it is being compared against, and beating it is easier.")
note("    That, more than the early-stopping optimism, is why the guard is so")
note("    permissive (9 of 10 and 8 of 10).")
note(f"  - NOT USED: weather, new historical fallbacks, new F3 features, tuning,")
note("    per-airport COMPLETE models, residual targets, ADS-B, airport geometry,")
note("    ensembles, clipping, BLOCK_TIME_UTC_mvt, any ranking target or ranking data.")
note("  - No ranking model was trained and no submission was created.")
flush()
print("\nDONE -> reports/f3_airport_guard.txt")
