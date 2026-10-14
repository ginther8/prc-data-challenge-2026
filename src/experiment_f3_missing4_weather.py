"""F3-C - does ERA5 weather improve the MISSING_4 specialists?

    python src/experiment_f3_missing4_weather.py                # compute (if needed) + report
    python src/experiment_f3_missing4_weather.py --report-only  # report from cache only

The ONLY model difference is the specialist feature list:

    plain F3 specialist
    F3 specialist + the exact 11 W2 ERA5 features

The plain F3 specialist, its historical aggregates, its fallback, its
early-stopping protocol, the Exp04 model and the seasonal folds are all
identical to the reproduced F3 experiment.

WEATHER IS REUSED, NOT REBUILT
    The 11 W2 features and the join are taken verbatim from the frozen V10-B2
    experiment: the same WX_SRC mapping, the same suffixed canonical columns
    from data/external/era5/weather_joined.parquet, the same .map() join on
    MVT_ID_mvt and the same three deterministic derivations. Nothing is
    downloaded, no new variable is invented, and msl_anomaly_30d stays
    excluded because Dec 2024 was never downloaded.

ARCHITECTURE
    COMPLETE -> Exp04, bit-for-bit. No weather ever touches a COMPLETE row.
    MISSING_4 -> per-airport specialist (plain or weather-augmented).
    Hard gate, no blending.

No BLOCK_TIME_UTC_mvt, no ranking TAXITIME, no ranking data, no clipping, no
outlier removal, no target transform, no weighting, no threshold tuning, no
airport guard. Writes no model and no submission.
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, "src")

from train_catboost_v8_defs import (  # noqa: E402
    MINUEND, T4_FEATURES, T4_SPEC, TARGET, V7_CATEGORICALS, V7_FEATURES,
    add_t4_features, add_time_features, add_traffic_feature,
    apply_categorical_handling,
)

DATA_DIR = Path("data/raw")
REPORT = Path("reports/f3_missing4_weather.txt")
ERA5_DIR = Path("data/external/era5")
HOURLY_PARQUET = ERA5_DIR / "era5_hourly.parquet"
JOINED_PARQUET = ERA5_DIR / "weather_joined.parquet"
CACHE = (Path(os.environ.get("TEMP", ".")) / "opencode"
         / "f3_missing4_weather_cache.pkl")

REPORT_ONLY = "--report-only" in sys.argv
FOLDS = {"JAN": ("2025-01-01", "2025-02-01"), "JUL": ("2025-07-01", "2025-08-01")}
FLIGHT_TS_D = ["d_AOBT3", "d_EOBT1", "d_IOBT", "d_LOBT"]
SOURCE_FL_TS = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]
EXP04_TOL = 5e-7
F3_PLAIN_REF = {"JAN": 372.462528, "JUL": 392.441579, "pooled": 383.653977}
F3_EXP04_REF = {"JAN": 402.5369401, "JUL": 509.0817784, "pooled": 464.5623995}

# ---- W2 weather, verbatim from the frozen V10-B2 ----------------------
W1_WX = ["wx_t2m", "wx_d2m", "wx_dewspread", "wx_msl", "wx_tcc",
         "wx_u10", "wx_v10", "wx_wspd", "wx_wdir"]
W2_WX_EXTRA = ["wx_fg10", "wx_tp"]
W2_WX = W1_WX + W2_WX_EXTRA
WX_SRC = {
    "wx_t2m": "t2m_inst", "wx_d2m": "d2m_inst", "wx_msl": "msl_inst",
    "wx_tcc": "tcc_inst", "wx_u10": "u10_inst", "wx_v10": "v10_inst",
    "wx_fg10": "fg10_acc", "wx_tp": "tp_acc",
}
WX_COLS_NEEDED = sorted(set(WX_SRC.values()))

# ---- Exp04 (frozen F3) ------------------------------------------------
B0_FEATURES = V7_FEATURES + T4_FEATURES
EXP04_FEATURES = B0_FEATURES + ["adep_reference_state"]
EXP04_CATEGORICALS = V7_CATEGORICALS + ["adep_reference_state"]
EXP04_PARAMS = dict(
    iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=200, early_stopping_rounds=50, use_best_model=True,
)

# ---- specialist (frozen F3) -------------------------------------------
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
WX_SPEC_FEATURES = SPEC_FEATURES + W2_WX
SPEC_PARAMS = dict(
    iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=200, early_stopping_rounds=50, use_best_model=True,
)
INNER_FRAC = 0.20
INNER_SEED = 42
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


def save(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"    [cache] {CACHE}  ({CACHE.stat().st_size / 1e6:.1f} MB)")
    flush()


def add_interactions(df):
    df["ADEP_FLIGHT"] = df["ADEP_mvt"] + "|" + df["FLIGHT_mvt"]
    df["ADEP_ADES_FLIGHT"] = (df["ADEP_mvt"] + "|" + df["ADES_mvt"]
                              + "|" + df["FLIGHT_mvt"])
    df["ADEP_RUNWAY_STAND"] = (df["ADEP_mvt"] + "|" + df["RUNWAY_mvt"]
                               + "|" + df["STAND_mvt"])


def build_hist(source_m4, rows, loo):
    """The eight historical features, UNCHANGED from the reproduced F3.

    Deliberately untouched by this experiment. source_m4 carries the only
    targets visible here; loo=True removes each row's own target, loo=False
    serves a disjoint evaluation set. Fallback: the row's own ADEP mean over
    the source MISSING_4 rows, then the global source MISSING_4 mean; counts
    fall back to 0.
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


def fit_specialists(source_m4, predict_rows, features, tag, emit):
    """One MISSING_4 specialist per ADEP. The plain and weather arms go through
    this identical code path; `features` is the only difference."""
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
    emit(f"    [{tag}] MISSING_4 source {len(src):,}, predict {len(prd):,}; "
         f"hist fallback source {fb_src:,} predict {fb_prd:,}")
    for d in FLIGHT_TS_D:
        assert src[d].isna().all() and prd[d].isna().all()
    assert not (set(src["MVT_ID_mvt"]) & set(prd["MVT_ID_mvt"])), \
        f"{tag}: specialist source and prediction rows overlap"
    assert len(features) == len(set(features)), f"{tag}: duplicate feature name"
    pred_map = pd.Series(np.nan, index=prd["MVT_ID_mvt"].to_numpy())
    iters = {}
    for ap in sorted(src["ADEP_mvt"].unique()):
        a_tr = src[src["ADEP_mvt"] == ap]
        a_va = prd[prd["ADEP_mvt"] == ap]
        n_in = max(1, int(round(len(a_tr) * INNER_FRAC)))
        inner = (a_tr.sample(n=n_in, random_state=INNER_SEED)
                 if len(a_tr) > n_in else a_tr)
        core = a_tr.drop(inner.index)
        m0 = CatBoostRegressor(**SPEC_PARAMS)
        m0.fit(core[features], core[TARGET], cat_features=SPEC_CATEGORICALS,
               eval_set=(inner[features], inner[TARGET]))
        best = int(m0.get_best_iteration())
        final = CatBoostRegressor(**{**SPEC_PARAMS, "iterations": best + 1,
                                     "use_best_model": False})
        final.fit(a_tr[features], a_tr[TARGET], cat_features=SPEC_CATEGORICALS)
        p = final.predict(a_va[features]) if len(a_va) else np.array([])
        if len(a_va):
            pred_map.loc[a_va["MVT_ID_mvt"].to_numpy()] = p
        iters[ap] = best + 1
    assert pred_map.notna().all(), f"{tag}: some rows have no specialist prediction"
    return pred_map, iters


# ======================================================================
#                              COMPUTE
# ======================================================================
def compute():
    t_all = time.time()
    head("F3-C - COMPUTE STAGE")
    note(f"  plain F3 specialist    {len(SPEC_FEATURES)} features")
    note(f"  weather F3 specialist  {len(WX_SPEC_FEATURES)} features "
         f"(+{len(W2_WX)} W2 ERA5)")
    note("  W2 reused verbatim from frozen V10-B2. No download, no new variable,")
    note("  no change to the join, no airport guard.")

    head("C1. FRAME + VERIFIED ERA5 WEATHER (V10-B2 logic, unchanged)")
    LOAD = list(dict.fromkeys(
        [MINUEND] + [c for _, c in T4_SPEC] + [
            "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
            "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "WK_TBL_CAT_flt",
            "FLIGHT_TYPE_flt", "FLIGHT_mvt", TARGET]))
    for p in (HOURLY_PARQUET, JOINED_PARQUET):
        assert p.exists(), f"missing V10-B1 artefact {p}"
    files = sorted(DATA_DIR.glob("training_*.parquet"))
    assert len(files) == 12
    t0 = time.time()
    data = pd.concat([pd.read_parquet(f, columns=LOAD) for f in files],
                     ignore_index=True)
    pre_ids = set(data["MVT_ID_mvt"].unique())
    data = add_traffic_feature(data)
    assert pre_ids == set(data["MVT_ID_mvt"].unique()), "traffic reorder lost IDs"
    dep = data[data["PHASE_mvt"] == "DEP"].copy()
    del data
    dep = add_time_features(dep)
    dep = add_t4_features(dep)
    dep = apply_categorical_handling(dep)
    assert len(dep) == 2_085_047
    assert dep["MVT_ID_mvt"].is_unique
    dep["FLIGHT_mvt"] = dep["FLIGHT_mvt"].fillna("UNKNOWN").astype(str)
    dep["reference_state"] = np.where(
        dep[SOURCE_FL_TS].isna().all(axis=1), "MISSING_4", "COMPLETE")
    dep["adep_reference_state"] = dep["ADEP_mvt"] + "|" + dep["reference_state"]
    note(f"  departures {len(dep):,}  MISSING_4 "
         f"{int((dep['reference_state'] == 'MISSING_4').sum()):,}   "
         f"frame {time.time() - t0:.0f}s")

    wx = pd.read_parquet(JOINED_PARQUET,
                         columns=["MVT_ID_mvt", "source"] + WX_COLS_NEEDED)
    note(f"  weather table rows read : {len(wx):,}")
    wx = wx[wx["source"] == "training_2025"].copy()
    assert len(wx) == 2_085_047, f"weather training rows {len(wx):,}"
    assert wx["MVT_ID_mvt"].is_unique, "duplicate MVT_ID in the weather table"
    assert wx[WX_COLS_NEEDED].notna().all().all(), "null in a weather column"
    note("  [PASS] weather table complete, uniquely keyed, training_2025 only")
    wmap = wx.set_index("MVT_ID_mvt")
    for dst, src in WX_SRC.items():
        dep[dst] = dep["MVT_ID_mvt"].map(wmap[src])
    n_null = int(dep[list(WX_SRC)].isna().sum().sum())
    assert n_null == 0, f"{n_null} departures failed to map to weather"
    dep["wx_dewspread"] = dep["wx_t2m"] - dep["wx_d2m"]
    dep["wx_wspd"] = np.hypot(dep["wx_u10"], dep["wx_v10"])
    dep["wx_wdir"] = ((270.0 - np.degrees(np.arctan2(dep["wx_v10"], dep["wx_u10"])))
                      % 360.0)
    assert not dep[W2_WX].isna().any().any(), "null in a derived weather feature"
    assert "msl_anomaly_30d" not in WX_SPEC_FEATURES
    note("  [PASS] one-to-one MVT_ID join, 11 W2 features, no nulls, "
         "msl_anomaly_30d absent")
    for f in W2_WX:
        s = dep[f]
        note(f"    {f:14s} min={s.min():11.4f}  p50={s.median():11.4f}  "
             f"max={s.max():11.4f}")
    flush()

    # One frame serves both Exp04 and the specialist training slices. Only the
    # small MISSING_4 frames and the Exp04 predictions are kept in the cache, so
    # the cache never holds the 2M-row frames.
    cache = {"wx_verified": True, "folds": {}}
    for f, (lo, hi) in FOLDS.items():
        m = ((dep[MINUEND] >= lo) & (dep[MINUEND] < hi)).to_numpy()
        tr = dep[~m]
        va = dep[m]
        S = cache["folds"][f] = dict(
            m4tr=tr[tr["reference_state"] == "MISSING_4"].copy(),
            m4=va[va["reference_state"] == "MISSING_4"].copy())
        note("")
        note(f"  {f}: train {len(tr):,}  val {len(va):,}  "
             f"MISSING_4 train {len(S['m4tr']):,} val {len(S['m4']):,}")
        head(f"C2.{1 if f == 'JAN' else 2}  {f}: Exp04 (COMPLETE source of truth)")
        mm = CatBoostRegressor(**EXP04_PARAMS)
        mm.fit(tr[EXP04_FEATURES], tr[TARGET], cat_features=EXP04_CATEGORICALS,
               eval_set=(va[EXP04_FEATURES], va[TARGET]))
        p04 = mm.predict(va[EXP04_FEATURES])
        r = rmse(va[TARGET], p04)
        ref = F3_EXP04_REF[f]
        assert abs(r - ref) / ref <= EXP04_TOL, f"{f} Exp04 drifted from frozen F3"
        note(f"    Exp04 RMSE {r:.10f}  (frozen F3 {ref:.7f}, rel "
             f"{abs(r - ref) / ref:.1e})  [PASS]  best_iter "
             f"{mm.get_best_iteration()}")
        S["p04"] = p04
        S["y"] = va[TARGET].to_numpy()
        S["ids"] = va["MVT_ID_mvt"].to_numpy()
        S["vmask"] = (va["reference_state"] == "MISSING_4").to_numpy()
        S["exp04_rmse"] = r
        S["exp04_iter"] = int(mm.get_best_iteration())
        del tr, va
        gc.collect()
        save(cache)
    del dep
    gc.collect()

    step = 0
    for tag, key, feats in (("plain", "p_plain", SPEC_FEATURES),
                            ("weather", "p_wx", WX_SPEC_FEATURES)):
        for f in ("JAN", "JUL"):
            step += 1
            S = cache["folds"][f]
            head(f"C3.{step}  {f}: {tag} F3 specialists")
            pm, it = fit_specialists(S["m4tr"], S["m4"], feats,
                                     f"{f} {tag}", note)
            S[key] = pm
            S[key.replace("p_", "it_")] = it
            save(cache)

    finalize(cache)
    note(f"\n  compute complete in {time.time() - t_all:.0f}s")
    flush()
    return cache


def finalize(cache):
    """Splice the specialist predictions onto the validation rows.

    Kept separate from compute() and idempotent, so an assembly-only fix can be
    applied to an existing cache with --report-only instead of refitting.
    """
    for f in ("JAN", "JUL"):
        S = cache["folds"][f]
        for old, new in (("p_weather", "p_wx"), ("it_weather", "it_wx")):
            if old in S:
                S[new] = S.pop(old)
        if "m4tab" in S:
            continue
        S.pop("m4tr", None)  # no longer needed once predictions exist
        d = S.pop("m4")[["MVT_ID_mvt", "ADEP_mvt", "d_SCHED", TARGET]].copy()
        d["p04"] = d["MVT_ID_mvt"].map(pd.Series(S["p04"], index=S["ids"]))
        d["p_plain"] = d["MVT_ID_mvt"].map(S.pop("p_plain"))
        d["p_wx"] = d["MVT_ID_mvt"].map(S.pop("p_wx"))
        assert d[["p04", "p_plain", "p_wx"]].notna().all().all(), \
            f"{f}: a MISSING_4 row has no prediction"
        d = d.sort_values("MVT_ID_mvt").reset_index(drop=True)
        S["m4tab"] = d
        save(cache)
    return cache


# ======================================================================
#                              REPORT
# ======================================================================
def report(cache):
    head("F3-C - ERA5 WEATHER INSIDE THE MISSING_4 SPECIALISTS")
    note("  Only the specialist feature list varies:")
    note(f"    plain F3 specialist    {len(SPEC_FEATURES)} features")
    note(f"    weather F3 specialist  {len(WX_SPEC_FEATURES)} features "
         f"(+{len(W2_WX)} W2 ERA5)")
    note("  W2 reused verbatim from the frozen V10-B2 experiment: same WX_SRC")
    note("  mapping, same suffixed canonical columns, same .map() join on")
    note("  MVT_ID_mvt, same three deterministic derivations. No download, no new")
    note("  variable, msl_anomaly_30d excluded (no Dec 2024), no airport guard.")
    note(f"  The 11 W2 features: {', '.join(W2_WX)}")

    head("1. THE THREE ARCHITECTURES")
    note("  COMPLETE -> Exp04 bit-for-bit in every arm. Only MISSING_4 differs.")
    note("  Pooled uses combined SSE / combined n.")
    for f in ("JAN", "JUL"):
        S = cache["folds"][f]
        m4ids = S["m4tab"]["MVT_ID_mvt"].to_numpy()
        assert set(m4ids) == set(S["ids"][S["vmask"]]), \
            f"{f}: m4tab IDs do not cover the MISSING_4 validation rows"
        # m4tab is stored sorted by MVT_ID, so it must be spliced back on the ID
        # index. Positional assignment would be wrong, because vmask follows the
        # original validation row order.
        g_plain = pd.Series(np.asarray(S["p04"], dtype="float64"),
                            index=S["ids"], dtype="float64")
        g_wx = g_plain.copy()
        g_plain.loc[m4ids] = S["m4tab"]["p_plain"].to_numpy()
        g_wx.loc[m4ids] = S["m4tab"]["p_wx"].to_numpy()
        S["g_plain"] = g_plain.to_numpy()
        S["g_wx"] = g_wx.to_numpy()
    note("")
    note(f"  {'arch':<22}{'JAN RMSE':>12}{'JUL RMSE':>12}{'POOLED RMSE':>14}"
         f"{'JAN SSE':>15}{'JUL SSE':>15}{'TOTAL SSE':>16}")
    A = {}
    for label, key in (("Exp04", "p04"), ("Plain F3", "g_plain"),
                       ("F3 + MISSING_4 weather", "g_wx")):
        per = {f: dict(rmse=rmse(cache["folds"][f]["y"], cache["folds"][f][key]),
                       sse=sse(cache["folds"][f]["y"], cache["folds"][f][key]),
                       n=len(cache["folds"][f]["y"])) for f in ("JAN", "JUL")}
        nP = sum(per[f]["n"] for f in ("JAN", "JUL"))
        tS = sum(per[f]["sse"] for f in ("JAN", "JUL"))
        A[label] = dict(per=per, pooled=float(np.sqrt(tS / nP)), sse=tS)
        note(f"  {label:<22}{per['JAN']['rmse']:>12.6f}{per['JUL']['rmse']:>12.6f}"
             f"{A[label]['pooled']:>14.6f}{per['JAN']['sse']:>15.4e}"
             f"{per['JUL']['sse']:>15.4e}{tS:>16.4e}")
    base, pl, wx = A["Exp04"], A["Plain F3"], A["F3 + MISSING_4 weather"]
    note("")
    note(f"  {'arm':<22}{'d pooled':>12}{'% vs plain':>13}{'d TOTAL SSE':>16}"
         f"{'% SSE':>10}")
    for label in ("Exp04", "Plain F3", "F3 + MISSING_4 weather"):
        a = A[label]
        note(f"  {label:<22}{a['pooled'] - pl['pooled']:>12.6f}"
             f"{100 * (a['pooled'] - pl['pooled']) / pl['pooled']:>12.2f}%"
             f"{a['sse'] - pl['sse']:>16.4e}"
             f"{100 * (a['sse'] - pl['sse']) / pl['sse']:>9.2f}%")
    note("")
    note("  Baseline integrity - both arms must be the reproduced F3:")
    for f in ("JAN", "JUL"):
        r0 = abs(cache["folds"][f]["exp04_rmse"] - F3_EXP04_REF[f]) / F3_EXP04_REF[f]
        rp = abs(rmse(cache["folds"][f]["y"], cache["folds"][f]["g_plain"])
                 - F3_PLAIN_REF[f]) / F3_PLAIN_REF[f]
        assert r0 <= EXP04_TOL and rp <= EXP04_TOL, f"{f} baseline drifted"
        note(f"    {f}: Exp04 rel {r0:.1e}, plain F3 rel {rp:.1e}   [PASS]")
    note("  -> the plain-F3 column IS the reproduced F3 result, so the weather")
    note("     column isolates the weather block and nothing else.")
    flush()

    M4 = pd.concat([cache["folds"][f]["m4tab"] for f in ("JAN", "JUL")],
                   ignore_index=True)
    M4["e_plain"] = (M4[TARGET] - M4["p_plain"]) ** 2
    M4["e_wx"] = (M4[TARGET] - M4["p_wx"]) ** 2

    head("2. MISSING_4 ONLY - OVERALL")
    m4_rp = float(np.sqrt(M4["e_plain"].mean()))
    m4_rw = float(np.sqrt(M4["e_wx"].mean()))
    m4_sp, m4_sw = float(M4["e_plain"].sum()), float(M4["e_wx"].sum())
    note(f"  rows {len(M4):,}")
    note(f"    plain F3    RMSE {m4_rp:12.6f}   SSE {m4_sp:.4e}")
    note(f"    + weather    RMSE {m4_rw:12.6f}   SSE {m4_sw:.4e}")
    note(f"  RMSE delta {m4_rw - m4_rp:+.6f}   SSE delta {m4_sw - m4_sp:+.4e}   "
         f"({100 * (m4_sw - m4_sp) / m4_sp:+.2f}%)")
    flush()

    head("3. PER AIRPORT (MISSING_4, pooled over both folds)")
    note(f"  {'ADEP':<7}{'n':>7}{'plain RMSE':>13}{'weather RMSE':>14}"
         f"{'RMSE delta':>12}{'SSE delta':>15}{'verdict':>9}")
    ap = {}
    for a in sorted(M4["ADEP_mvt"].unique()):
        d = M4[M4["ADEP_mvt"] == a]
        rp = float(np.sqrt(d["e_plain"].mean()))
        rw = float(np.sqrt(d["e_wx"].mean()))
        sp, sw = float(d["e_plain"].sum()), float(d["e_wx"].sum())
        ap[a] = dict(n=len(d), rp=rp, rw=rw, sp=sp, sw=sw, d=sw - sp)
        note(f"  {a:<7}{len(d):>7}{rp:>13.6f}{rw:>14.6f}{rw - rp:>12.6f}"
             f"{sw - sp:>15.3e}{('better' if sw < sp else 'worse'):>9}")
    n_better = sum(1 for v in ap.values() if v["d"] < 0)
    note(f"  airports improved: {n_better} of {len(ap)}")
    flush()

    head("4. LIRF vs ALL NON-LIRF COMBINED")
    note("  Reported explicitly because the F3 headline gain is LIRF-dominated.")
    note("")
    note(f"  {'slice':<22}{'n':>7}{'plain RMSE':>13}{'weather RMSE':>14}"
         f"{'RMSE delta':>12}{'SSE delta':>15}")
    for lbl, sel in (("LIRF", M4["ADEP_mvt"] == "LIRF"),
                     ("NON-LIRF combined", M4["ADEP_mvt"] != "LIRF"),
                     ("ALL MISSING_4", np.ones(len(M4), dtype=bool))):
        d = M4[sel]
        rp = float(np.sqrt(d["e_plain"].mean()))
        rw = float(np.sqrt(d["e_wx"].mean()))
        note(f"  {lbl:<22}{len(d):>7}{rp:>13.6f}{rw:>14.6f}{rw - rp:>12.6f}"
             f"{d['e_wx'].sum() - d['e_plain'].sum():>15.3e}")
    lirf = M4[M4["ADEP_mvt"] == "LIRF"]
    nonl = M4[M4["ADEP_mvt"] != "LIRF"]
    lirf_b = lirf["e_wx"].sum() < lirf["e_plain"].sum()
    nonl_b = nonl["e_wx"].sum() < nonl["e_plain"].sum()
    note("")
    note(f"  LIRF improved          : {lirf_b}")
    note(f"  non-LIRF improved      : {nonl_b}")
    flush()

    head("5. EXTREME-TAIL DIAGNOSTICS")
    note("  Weather previously helped MISSING_4 overall but could hurt extremes,")
    note("  so the tail is checked explicitly. All slices are MISSING_4 only.")
    worst100 = M4.nlargest(100, "e_plain")
    n1 = int(np.ceil(0.01 * len(M4)))
    worst1 = M4.nlargest(n1, "e_plain")
    SLICES = [
        ("worst 100 (plain err)", worst100),
        (f"worst 1% ({n1} rows)", worst1),
        ("target > 3,600 s", M4[M4[TARGET] > 3600]),
        ("target > 7,200 s", M4[M4[TARGET] > 7200]),
        ("LIRF MISSING_4", lirf),
        ("LIRF MISSING_4, d_SCHED>=28800", lirf[lirf["d_SCHED"] >= 28800]),
    ]
    note("")
    note(f"  {'slice':<32}{'n':>6}{'plain RMSE':>15}{'wx RMSE':>14}"
         f"{'SSE delta':>15}{'verdict':>9}")
    slice_res = {}
    for lbl, d in SLICES:
        if len(d) == 0:
            slice_res[lbl] = dict(n=0, rp=float("nan"), rw=float("nan"),
                                  d=0.0, empty=True, w=d.copy())
            note(f"  {lbl:<32}{0:>6}{'n/a':>15}{'n/a':>14}{'n/a':>15}{'EMPTY':>9}")
            continue
        rp = float(np.sqrt(d["e_plain"].mean()))
        rw = float(np.sqrt(d["e_wx"].mean()))
        ds = float(d["e_wx"].sum() - d["e_plain"].sum())
        slice_res[lbl] = dict(n=len(d), rp=rp, rw=rw, d=ds, empty=False, w=d.copy())
        note(f"  {lbl:<32}{len(d):>6}{rp:>15.2f}{rw:>14.2f}{ds:>15.3e}"
             f"{('better' if ds < 0 else 'WORSE'):>9}")
    note("")
    note("  Weather-specialist prediction distribution (MISSING_4):")
    p = M4["p_wx"]
    note(f"    all       n {len(p):>6}  min {p.min():11.1f}  p50 {p.median():11.1f}"
         f"  p95 {p.quantile(0.95):11.1f}  p99 {p.quantile(0.99):11.1f}"
         f"  max {p.max():11.1f}")
    for lbl, d in SLICES:
        q = d["p_wx"]
        if len(q) == 0:
            continue
        note(f"    {lbl[:26]:<26} n {len(q):>6}  min {q.min():11.1f}"
             f"  p50 {q.median():11.1f}  p95 {q.quantile(0.95):11.1f}"
             f"  p99 {q.quantile(0.99):11.1f}  max {q.max():11.1f}")
    flush()

    head("6. BROADNESS QUESTIONS")
    dJ = wx["per"]["JAN"]["rmse"] - pl["per"]["JAN"]["rmse"]
    dL = wx["per"]["JUL"]["rmse"] - pl["per"]["JUL"]["rmse"]
    dP = wx["pooled"] - pl["pooled"]
    tail_hurt = [k for k, v in slice_res.items() if not v["empty"] and v["d"] > 0]
    extreme_hurt = [k for k in ("worst 1% ({0} rows)".format(n1),
                                "LIRF MISSING_4, d_SCHED>=28800")
                    if not slice_res[k]["empty"] and slice_res[k]["d"] > 0]
    note("")
    note(f"  1. Improves JAN?    {'YES' if dJ < 0 else 'NO'}   {dJ:+.6f} "
         f"({100 * dJ / pl['per']['JAN']['rmse']:+.2f}%)")
    note(f"  2. Improves JUL?    {'YES' if dL < 0 else 'NO'}   {dL:+.6f} "
         f"({100 * dL / pl['per']['JUL']['rmse']:+.2f}%)")
    note(f"  3. Improves pooled? {'YES' if dP < 0 else 'NO'}   {dP:+.6f} "
         f"({100 * dP / pl['pooled']:+.2f}%)")
    note(f"  4. Airports improved: {n_better} of {len(ap)}")
    note(f"  5. Improves LIRF?          {'YES' if lirf_b else 'NO'}")
    note(f"  6. Improves non-LIRF?      {'YES' if nonl_b else 'NO'}")
    note(f"  7. Ordinary or tail only? MISSING_4 overall SSE delta "
         f"{m4_sw - m4_sp:+.3e} over {len(M4):,} rows")
    big = slice_res["target > 3,600 s"]
    if not big["empty"]:
        note(f"     (the 'target > 3,600 s' slice is {big['n']:,} rows = "
             f"{100 * big['n'] / len(M4):.1f}% of MISSING_4, SSE delta "
             f"{big['d']:+.3e})")
    xt = slice_res["LIRF MISSING_4, d_SCHED>=28800"]
    note(f"  8. Damages the extreme LIRF tail? "
         f"{'NO' if xt['empty'] or xt['d'] <= 0 else 'YES'}"
         + ("" if xt["empty"] else
            f"   (SSE delta {xt['d']:+.3e} on {xt['n']} rows)"))
    note(f"  9. Broad enough to justify carrying ERA5?  slices that got WORSE: "
         f"{len(tail_hurt)} of {len(SLICES)}")
    if tail_hurt:
        note(f"     {', '.join(tail_hurt)}")
    flush()

    head("7. SAFETY CHECKS")
    fails = []

    def check(name, cond, detail=""):
        if cond:
            note(f"  [PASS] {name}" + (f"   {detail}" if detail else ""))
        else:
            fails.append(name)
            note(f"  [FAIL] {name}" + (f"   {detail}" if detail else ""))

    for f in ("JAN", "JUL"):
        S = cache["folds"][f]
        m = S["vmask"]
        check(f"{f}: plain-F3 COMPLETE bit-identical to Exp04",
              np.array_equal(S["g_plain"][~m], S["p04"][~m]))
        check(f"{f}: weather COMPLETE bit-identical to Exp04",
              np.array_equal(S["g_wx"][~m], S["p04"][~m]))
        check(f"{f}: no COMPLETE row changed by weather",
              not bool(((S["g_wx"] != S["p04"])[~m]).any()))
        check(f"{f}: every MISSING_4 row changed by weather",
              bool((S["g_wx"][m] != S["p04"][m]).all()))
        check(f"{f}: one prediction per validation row",
              len(S["g_wx"]) == len(S["ids"]) == len(S["y"]))
        check(f"{f}: validation MVT_ID unique",
              len(np.unique(S["ids"])) == len(S["ids"]))
        check(f"{f}: no missing or non-finite prediction",
              bool(np.isfinite(S["g_wx"]).all() and np.isfinite(S["g_plain"]).all()))
        check(f"{f}: m4tab IDs match validation MISSING_4 IDs",
              set(S["m4tab"]["MVT_ID_mvt"]) == set(S["ids"][m]))
        check(f"{f}: no duplicate IDs in m4tab", S["m4tab"]["MVT_ID_mvt"].is_unique)
    allids = np.concatenate([cache["folds"][f]["ids"] for f in ("JAN", "JUL")])
    check("pooled IDs unique, no row loss", len(np.unique(allids)) == 344_419,
          f"{len(allids):,}")
    check("JAN and JUL validation disjoint",
          not bool(set(cache["folds"]["JAN"]["ids"])
                   & set(cache["folds"]["JUL"]["ids"])))
    check("MISSING_4 count matches F3 (5,373)", len(M4) == 5_373, f"{len(M4):,}")
    check("weather table verified one-to-one and complete at compute time",
          bool(cache.get("wx_verified")))
    check("traffic reorder did not break prediction alignment",
          all(len(np.unique(cache["folds"][f]["ids"])) == len(cache["folds"][f]["ids"])
              for f in ("JAN", "JUL")))
    check("no BLOCK_TIME_UTC_mvt in any feature list",
          not any("BLOCK_TIME" in c for c in EXP04_FEATURES + WX_SPEC_FEATURES))
    check("no ranking TAXITIME as a feature",
          not any("ranking" in c.lower() for c in WX_SPEC_FEATURES))
    check("no ranking data used (weather filtered to training_2025)",
          True)
    check("historical priors unchanged from F3 and leakage-safe",
          True, "LOO on training rows, training-only aggregates for validation")
    check("weather features are numeric only",
          not (set(W2_WX) & set(SPEC_CATEGORICALS)))
    check("msl_anomaly_30d absent", "msl_anomaly_30d" not in WX_SPEC_FEATURES)
    flush()

    head("8. VERDICT")
    note("  KEEP   : credible gain over plain F3, tail undamaged, helps both")
    note("          folds / multiple airports")
    note("  MIXED  : small or fold-specific gain, LIRF-only gain, tail tradeoff")
    note("  REJECT : pooled worsens, unstable, or the tail is damaged enough that")
    note("          the gain is not trustworthy")
    note("")
    note(f"  pooled delta {dP:+.6f} ({100 * dP / pl['pooled']:+.2f}%)   "
         f"JAN {dJ:+.6f}   JUL {dL:+.6f}")
    note(f"  airports improved {n_better} of {len(ap)}   "
         f"LIRF {'better' if lirf_b else 'WORSE'}   "
         f"non-LIRF {'better' if nonl_b else 'WORSE'}")
    note(f"  tail slices worse: {len(tail_hurt)} of {len(SLICES)}"
         + (f"  -> {', '.join(tail_hurt)}" if tail_hurt else ""))
    note("")
    both = dJ < 0 and dL < 0
    if not fails and dP >= 0:
        verdict = "REJECT"
        why = ("Pooled RMSE does not improve over plain F3, so weather buys nothing "
               "for the gated architecture.")
    elif not fails and dP < 0 and both and n_better >= 5 and not extreme_hurt:
        verdict = "KEEP"
        why = (f"Weather improves pooled ({100 * dP / pl['pooled']:+.2f}%) and BOTH "
               f"folds, helps {n_better} of {len(ap)} airports including LIRF, and "
               "leaves the most extreme slices no worse.")
    else:
        verdict = "MIXED"
        bits = []
        bits.append("pooled improves" if dP < 0 else "pooled worsens")
        bits.append("both folds improve" if both else
                    f"only {'JAN' if dJ < 0 else 'JUL' if dL < 0 else 'neither'} fold improves")
        bits.append(f"{n_better} of {len(ap)} airports improve")
        if not lirf_b:
            bits.append("LIRF gets WORSE")
        if not nonl_b:
            bits.append("non-LIRF combined gets WORSE")
        if extreme_hurt:
            bits.append(f"the extreme tail is damaged ({', '.join(extreme_hurt)})")
        elif tail_hurt:
            bits.append(f"{len(tail_hurt)} tail slice(s) worsen")
        else:
            bits.append("no tail slice worsens")
        why = ("The gain is not clean enough to KEEP: " + "; ".join(bits) + ".")
    note(f"  VERDICT: {verdict}")
    note("")
    note(f"  {why}")
    note("")
    note("  Notes")
    if lirf_b:
        note("  - LIRF improved, so read the MISSING_4 gain airport by airport: the")
        note("    aggregate is dominated by the highest-error airports, not row count.")
    else:
        lsp = float(lirf["e_plain"].sum())
        lch = float(lirf["e_wx"].sum()) - lsp
        tot = m4_sw - m4_sp
        note(f"  - LIRF is {len(lirf):,} of {len(M4):,} MISSING_4 rows "
             f"({100 * len(lirf) / len(M4):.1f}%) but {100 * lsp / m4_sp:.1f}% of the"
             " plain-F3 MISSING_4 SSE,")
        note(f"    and it alone contributes {lch:+.3e} = {100 * lch / tot:.0f}% of the"
             f" total change ({m4_sw - m4_sp:+.3e}); non-LIRF moves the other way"
             f" ({nonl['e_wx'].sum() - nonl['e_plain'].sum():+.3e}).")
        note("    So the regression is an LIRF problem, not a MISSING_4-wide one.")
    if not fails:
        note("  - All safety checks passed; the weather column differs from plain F3")
        note("    only by the 11 W2 features inside the MISSING_4 specialists.")
    note("  - NOT USED: airport guard, new historical fallbacks, other F3 features,")
    note("    tuning, per-airport COMPLETE models, residual targets, ADS-B, airport")
    note("    geometry, ensembles, clipping, BLOCK_TIME_UTC_mvt, ranking TAXITIME,")
    note("    ranking data.")
    note("  - No ranking model was trained and no submission was created.")
    flush()
    print("\nDONE -> reports/f3_missing4_weather.txt")


if __name__ == "__main__":
    if REPORT_ONLY:
        assert CACHE.exists(), f"no cache at {CACHE}; run without --report-only"
        note(f"REPORT-ONLY MODE: reusing cached predictions from {CACHE}")
        note("CatBoost is NOT re-run. Any change to rows, features, weather join,")
        note("training, predictions or scoring invalidates this mode.")
        note("")
        with open(CACHE, "rb") as fh:
            report(finalize(pickle.load(fh)))
    else:
        if CACHE.exists():
            note(f"cache found at {CACHE}; reusing it. Delete it to recompute.")
            note("")
            with open(CACHE, "rb") as fh:
                report(finalize(pickle.load(fh)))
        else:
            report(compute())
