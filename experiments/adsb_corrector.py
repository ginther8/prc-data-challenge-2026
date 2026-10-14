"""ADS-B corrector on top of V4 - validation (JAN<->JUL) and V5 submission.

BASE      V4 = A4 on COMPLETE rows, F3 (nested, as submitted) on MISSING_4.
          Validation base: out-of-fold predictions cached by
          run_janjul_v4_cached.py (pred_A4 / pred_F3_nest).
          Ranking base   : the accepted V4 submission file (--v4-submission).
CORRECTOR one pooled CatBoost model, residual target clipped to +-1800 s (hedge), (no airport identity - coverage drifts
          between 2025 and 2026), target = y - base, features = base,
          d_AOBT3, d_SCHED, MISSING_4 flag, adsb_* movement features and a few
          differences between them. Prediction = base + correction.
VALIDATION train on JAN 2025 -> score JUL 2025, and train on JUL -> score JAN.
          Iterations chosen on an inner split (abs(MVT_ID) % 5 == 0) of the
          training month only. Pooled RMSE = combined SSE of both directions.
GATE      the submission is built only if pooled RMSE improves by >= 2.0 AND
          both directions improve (override with --force).
FINAL     corrector refit on JAN+JUL with the mean of the two inner best
          iterations, applied to the V4 ranking predictions.

Run from the project root:
    python experiments/adsb_corrector.py --v4-submission submissions/<v4 file>.parquet
    python experiments/adsb_corrector.py --validate-only
    python experiments/adsb_corrector.py --validate-only --base-oof data/processed/v7_c2f3_oof_JAN.parquet data/processed/v7_c2f3_oof_JUL.parquet
    python experiments/adsb_corrector.py --base-oof <JAN> <JUL> --base-submission submissions/vigorous-glove_v7.parquet
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
PROC = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"
FEATS_PATH = PROC / "adsb_features.parquet"
REPORT = PROJECT_ROOT / "reports" / "adsb_corrector.txt"
OUT_DIR = PROJECT_ROOT / "submissions"
TEAM = "vigorous-glove"
VERSION = 9  # v6 ours, v7 friend, v8 stacked; we use v9, friend v10+

MID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
EXPECTED_RANKING = 344_841
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
ADSB = ["adsb_matched", "adsb_ok", "adsb_est", "adsb_est_move", "adsb_gs0",
        "adsb_npts", "adsb_maxgap", "adsb_tail", "adsb_stopped",
        "adsb_prev_gap", "adsb_nhex",
        # feature v2 (adsb_features.py FEATURE_VERSION = 2)
        "adsb_lead", "adsb_dpark", "adsb_dpark_conf", "adsb_down",
        "adsb_path", "adsb_disp",
        # feature v3 (OSM stands / runway thresholds)
        "adsb_d_stand", "adsb_d_first_rwy", "geo_d_stand_rwy",
        # feature v4 (callsign match type)
        "adsb_match_type"]
WX = ["wx_tmpc", "wx_spread", "wx_frost", "wx_vsby", "wx_lowvis", "wx_sknt", "wx_age_min"]
METAR = PROJECT_ROOT / "data" / "external" / "metar" / "metar_10apt.parquet"
USE_WEATHER = False
FEATURES = ["base", "d_AOBT3", "d_SCHED", "is_m4"] + ADSB + [
    "est_minus_base", "move_minus_base", "move_minus_aobt"]
CB = dict(depth=6, learning_rate=0.05, loss_function="RMSE", random_seed=42,
          verbose=False, allow_writing_files=False)
MAX_IT, ES = 3000, 100
GATE_GAIN = 2.0
# Hedge: the corrector learns only moderate corrections. Residual targets are
# clipped to +-CLIP seconds, and applied corrections are clipped the same way,
# so a few defective labels (e.g. block time copied from the schedule) cannot
# teach it to make huge confident corrections. The V4 base keeps the tail.
CLIP = 1800.0

OUT = []


def A(s=""):
    OUT.append(str(s))
    print(s, flush=True)


def write_report():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(OUT) + "\n")


def stop(msg):
    A(f"STOPPED: {msg}")
    write_report()
    sys.exit(1)


def rmse(e):
    e = np.asarray(e, np.float64)
    return math.sqrt(float(np.dot(e, e)) / len(e))


CTX = ["ctx_arr_taxiin_30", "ctx_arr_taxiin_60", "ctx_arr_n_60", "ctx_arr_taxiin_rwy_60",
       "ctx_stand_since_arr", "ctx_dep_adsb_60", "ctx_dep_adsb_n_60", "ctx_dep_aobt_60"]
CTX_PATH = PROC / "context_features.parquet"
USE_CONTEXT = False
GATE = ["gr_overdue", "gr_overdue_rwy", "gr_active_rwy", "gr_aobt_sched_60", "gr_cadence_60"]
GATE_PATH = PROC / "gate_features.parquet"
USE_GATE = False


def load_feats(context=False):
    f = pd.read_parquet(FEATS_PATH)
    if context:
        if not CTX_PATH.exists():
            stop(f"--context needs {CTX_PATH} (run context_features.py)")
        c = pd.read_parquet(CTX_PATH)
        lack = [x for x in CTX if x not in c.columns]
        if lack:
            stop(f"context file lacks {lack}")
        f = f.merge(c[[MID] + CTX], on=MID, how="outer")
    if USE_GATE:
        if not GATE_PATH.exists():
            stop(f"--gate needs {GATE_PATH} (run gate_features.py)")
        g = pd.read_parquet(GATE_PATH)
        f = f.merge(g[[MID] + GATE], on=MID, how="left")
    return f


def add_weather(df, times=None):
    """Latest METAR at the departure airport at or before takeoff (UTC, 3 h
    tolerance). times: DataFrame MVT_ID_mvt -> MVT_TIME_UTC_mvt (tz-aware) for
    frames that do not carry the takeoff time themselves."""
    if not USE_WEATHER:
        return df
    if not METAR.exists():
        stop(f"--weather needs {METAR} (run metar_download.py)")
    w = pd.read_parquet(METAR)
    w = w.rename(columns={"station": "ADEP_mvt", "valid": "_wt"})
    w["_wt"] = pd.to_datetime(w["_wt"], utc=True)
    w = w[["ADEP_mvt", "_wt", "tmpc", "spread_c", "frost_flag", "vsby_km",
           "low_vis_flag", "sknt"]].sort_values("_wt")
    d = df.copy()
    if "MVT_TIME_UTC_mvt" not in d.columns:
        d = d.merge(times, on=MID, how="left")
    d["_T"] = pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True)
    if d["_T"].isna().any():
        stop("weather join: some rows have no takeoff time")
    d["_ord"] = np.arange(len(d))
    j = pd.merge_asof(d.sort_values("_T"), w, left_on="_T", right_on="_wt",
                      by="ADEP_mvt", direction="backward",
                      tolerance=pd.Timedelta(hours=3)).sort_values("_ord")
    for new, old in (("wx_tmpc", "tmpc"), ("wx_spread", "spread_c"), ("wx_frost", "frost_flag"),
                     ("wx_vsby", "vsby_km"), ("wx_lowvis", "low_vis_flag"), ("wx_sknt", "sknt")):
        df[new] = j[old].to_numpy()
    df["wx_age_min"] = ((j["_T"] - j["_wt"]).dt.total_seconds() / 60).to_numpy()
    filled = float(np.isfinite(df["wx_tmpc"]).mean())
    A(f"  weather joined: {filled:.1%} of {len(df):,} rows have a METAR within 3 h")
    if filled < 0.8:
        stop("weather join filled < 80% of rows - check METAR coverage / timezones")
    return df


def takeoff_times():
    import polars as pl
    return pl.read_parquet(PROC / "v8_prep_cache.parquet",
                           columns=[MID, "MVT_TIME_UTC_mvt"]).to_pandas()


def add_derived(df):
    df["is_m4"] = df["d_AOBT3"].isna().astype(float)
    df["est_minus_base"] = df["adsb_est"] - df["base"]
    df["move_minus_base"] = df["adsb_est_move"] - df["base"]
    df["move_minus_aobt"] = df["adsb_est_move"] - df["d_AOBT3"]
    return df


def load_validation(feats, base_oof=None, m4_from_v4=False):
    """Validation frame. Base = V4 (A4 / F3_nest from our benchmark cache) or,
    with base_oof, any model's honest out-of-fold predictions (parquet files
    with MVT_ID_mvt, fold, pred). Targets / timestamps always come from our
    benchmark cache, and the other model's y must match them exactly."""
    frames = []
    for fold in ("JAN", "JUL"):
        p = PROC / f"janjul_v4_cache_{fold}.parquet"
        if not p.exists():
            stop(f"missing {p} - run run_janjul_v4_cached.py first")
        c = pd.read_parquet(p)
        for col in ("pred_A4", "pred_F3_nest", "y", "state", "d_AOBT3", "d_SCHED"):
            if col not in c.columns:
                stop(f"{p.name} lacks {col}")
        m4 = c["state"].to_numpy() == "MISSING_4"
        c["base"] = np.where(m4, c["pred_F3_nest"], c["pred_A4"])
        if not np.isfinite(c["base"]).all():
            stop(f"{fold}: non-finite base predictions")
        c["fold"] = fold
        c["v4_f3"] = c["pred_F3_nest"]
        frames.append(c[[MID, "fold", "ADEP_mvt", "state", "y", "base",
                         "d_AOBT3", "d_SCHED", "v4_f3"]])
    v = pd.concat(frames, ignore_index=True)
    if base_oof:
        o = pd.concat([pd.read_parquet(p) for p in base_oof], ignore_index=True)
        for col in (MID, "fold", "pred"):
            if col not in o.columns:
                stop(f"base OOF file lacks column {col}")
        if o[MID].duplicated().any():
            stop("base OOF has duplicate MVT_ID_mvt")
        if len(o) != len(v) or set(o[MID]) != set(v[MID]):
            stop(f"base OOF covers {len(o):,} rows, expected the same {len(v):,} "
                 "Jan/Jul 2025 departures")
        v = v.merge(o[[MID, "fold", "pred"] + (["y"] if "y" in o.columns else [])]
                    .rename(columns={"fold": "fold_o", "y": "y_o"}), on=MID, how="left")
        if (v["fold"] != v["fold_o"]).any():
            stop("base OOF fold labels disagree with the benchmark folds")
        if "y_o" in v.columns and not np.allclose(v["y"], v["y_o"]):
            stop("base OOF targets differ from ours - different rows or ordering")
        if not np.isfinite(v["pred"]).all():
            stop("base OOF has non-finite predictions")
        v["base"] = v["pred"]
        if m4_from_v4:
            # external model's predictions on COMPLETE rows, our benchmark's
            # V4-style F3 (pred_F3_nest) on MISSING_4 rows - matches a ranking
            # file built with the original F3 (priors from all departures)
            m4 = (v["state"] == "MISSING_4").to_numpy()
            v.loc[m4, "base"] = v.loc[m4, "v4_f3"]
            A(f"MISSING_4 rows ({int(m4.sum()):,}) use the V4-style F3 from our benchmark")
        v = v.drop(columns=[c for c in ("fold_o", "y_o", "pred") if c in v.columns])
        A(f"base = external OOF predictions: {', '.join(str(p) for p in base_oof)}")
    else:
        A("base = V4 (A4 + F3_nest) from the benchmark cache")
    v = v.merge(feats, on=MID, how="left")
    v["adsb_matched"] = v["adsb_matched"].fillna(0)
    v["adsb_ok"] = v["adsb_ok"].fillna(0)
    v = add_weather(v, takeoff_times() if USE_WEATHER else None)
    return add_derived(v)


SECOND = "none"
ZERO_UNMATCHED = False
PREP_CACHE = PROC / "v8_prep_cache.parquet"


def correction(model, X, frame):
    """Clipped correction; optionally zero where no ADS-B track was matched."""
    c = np.clip(model.predict(X), -CLIP, CLIP)
    if ZERO_UNMATCHED:
        c = np.where(frame["adsb_matched"].to_numpy() == 1, c, 0.0)
    return c


def load_extra(feats, paths, main_ids):
    """Extra training months: base predictions from the teammate's OOF files,
    everything else (target, timestamps, state, airport) from our prep cache."""
    o = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    for col in (MID, "fold", "y", "pred"):
        if col not in o.columns:
            stop(f"extra OOF lacks column {col}")
    if o[MID].duplicated().any():
        stop("extra OOF has duplicate MVT_ID_mvt")
    if o[MID].isin(main_ids).any():
        stop("extra OOF overlaps the JAN/JUL validation rows")
    if not np.isfinite(o["pred"]).all():
        stop("extra OOF has non-finite predictions")
    if not PREP_CACHE.exists():
        stop(f"missing {PREP_CACHE} (needed for extra months)")
    import polars as pl
    cols = [MID, "ADEP_mvt", TARGET, "d_AOBT3", "d_SCHED", "d_EOBT1", "d_IOBT", "d_LOBT"]
    p = pl.read_parquet(PREP_CACHE, columns=cols).to_pandas()
    e = o.merge(p, on=MID, how="left")
    if e[TARGET].isna().any():
        stop("extra OOF contains IDs not in the 2025 prep cache")
    if not np.allclose(e["y"], e[TARGET]):
        stop("extra OOF targets differ from the prep cache - different rows?")
    m4 = e[["d_AOBT3", "d_EOBT1", "d_IOBT", "d_LOBT"]].isna().all(axis=1)
    e["state"] = np.where(m4, "MISSING_4", "COMPLETE")
    e["base"] = e["pred"]
    e = e[[MID, "fold", "ADEP_mvt", "state", "y", "base", "d_AOBT3", "d_SCHED"]]
    e = e.merge(feats, on=MID, how="left")
    e["adsb_matched"] = e["adsb_matched"].fillna(0)
    e["adsb_ok"] = e["adsb_ok"].fillna(0)
    e = add_weather(e, takeoff_times() if USE_WEATHER else None)
    e = add_derived(e)
    summary = e.groupby("fold").agg(rows=(MID, "size"), adsb_ok=("adsb_ok", "mean"))
    A(f"extra training months: {len(e):,} rows")
    for f, r in summary.iterrows():
        A(f"  {f}: {int(r['rows']):,} rows, ADS-B reaching runway {r['adsb_ok']:.1%}")
    return e


class Averaged:
    """Equal-weight average of two fitted correction models."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = []
        for m in self.models:
            if isinstance(m, Averaged):
                preds.append(m.predict(X))
            elif type(m).__name__ == "Booster":          # LightGBM
                preds.append(m.predict(X.to_numpy(dtype=np.float64)))
            else:
                preds.append(m.predict(X))
        return np.mean(preds, axis=0)


SEEDS = [42]


def _fit_lgbm(X, y, inner, seed=42):
    try:
        import lightgbm as lgb
    except ImportError:
        stop("--second-model lgbm needs LightGBM: pip install lightgbm")
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, seed=seed, verbose=-1)
    dtr = lgb.Dataset(X[~inner], y[~inner])
    dva = lgb.Dataset(X[inner], y[inner], reference=dtr)
    probe = lgb.train(params, dtr, num_boost_round=MAX_IT, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(ES, verbose=False)])
    best = int(probe.best_iteration)
    model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=max(best, 1))
    return model, best


def _fit_cat2(X, y, inner, seed=7):
    p = dict(CB)
    p.update(depth=8, learning_rate=0.03, random_seed=seed, rsm=0.6, l2_leaf_reg=10)
    probe = CatBoostRegressor(iterations=MAX_IT * 2, **p)
    probe.fit(X[~inner], y[~inner], eval_set=(X[inner], y[inner]),
              early_stopping_rounds=ES, use_best_model=True)
    best = int(probe.get_best_iteration())
    model = CatBoostRegressor(iterations=best + 1, **p)
    model.fit(X, y)
    return model, best


def fit_nested(tr):
    """One (CatBoost [+ second model]) pair per seed; all averaged with equal
    weights. Tree counts per seed chosen on the inner split only."""
    inner = (tr[MID].abs() % 5 == 0).to_numpy()
    X = tr[FEATURES]
    y = np.clip((tr["y"] - tr["base"]).to_numpy(), -CLIP, CLIP)
    cat_models, sec_models, bests = [], [], []
    for sd in SEEDS:
        p = dict(CB, random_seed=sd)
        probe = CatBoostRegressor(iterations=MAX_IT, **p)
        probe.fit(X[~inner], y[~inner], eval_set=(X[inner], y[inner]),
                  early_stopping_rounds=ES, use_best_model=True)
        best = int(probe.get_best_iteration())
        m = CatBoostRegressor(iterations=best + 1, **p)
        m.fit(X, y)
        cat_models.append(m)
        bests.append(best)
        if SECOND != "none":
            fn = _fit_lgbm if SECOND == "lgbm" else _fit_cat2
            second, best2 = fn(X.to_numpy(dtype=np.float64) if SECOND == "lgbm" else X,
                               y, inner, seed=sd if SECOND == "lgbm" else sd + 1000)
            sec_models.append(second)
            A(f"    seed {sd}: catboost best_iteration={best}, "
              f"second model ({SECOND}) best_iteration={best2}")
        elif len(SEEDS) > 1:
            A(f"    seed {sd}: catboost best_iteration={best}")
    # equal weight between the two model FAMILIES, then between seeds within each
    fam = [Averaged(cat_models)] + ([Averaged(sec_models)] if sec_models else [])
    model = fam[0] if len(fam) == 1 else Averaged(fam)
    return model, int(round(np.mean(bests)))


def breakdown(v, pred_col):
    rows = []
    for name, m in (("ALL", np.ones(len(v), bool)),
                    ("COMPLETE", (v["state"] == "COMPLETE").to_numpy()),
                    ("MISSING_4", (v["state"] == "MISSING_4").to_numpy()),
                    ("adsb_ok=1", (v["adsb_ok"] == 1).to_numpy()),
                    ("adsb_ok=1 & gs0<6", ((v["adsb_ok"] == 1) & (v["adsb_gs0"] < 6)).to_numpy()),
                    ("pushback at parking", ((v["adsb_ok"] == 1) & (v["adsb_gs0"] < 6)
                                             & (v["adsb_dpark"] < 40)).to_numpy()),
                    ("no ADS-B", (v["adsb_ok"] == 0).to_numpy())):
        if m.sum() == 0:
            continue
        rows.append((name, int(m.sum()), rmse(v["y"][m] - v["base"][m]),
                     rmse(v["y"][m] - v[pred_col][m])))
    return rows


def validate(v, extra=None):
    A("=" * 78)
    A("VALIDATION: train one month, score the other (base = OOF predictions)")
    A("=" * 78)
    v["corr_pred"] = np.nan
    bests = []
    for tr_fold, te_fold in (("JAN", "JUL"), ("JUL", "JAN")):
        tr, te = v[v["fold"] == tr_fold], v["fold"] == te_fold
        if extra is not None:
            tr = pd.concat([tr, extra[tr.columns.intersection(extra.columns)]],
                           ignore_index=True)
        t0 = time.time()
        model, best = fit_nested(tr)
        bests.append(best)
        v.loc[te, "corr_pred"] = (v.loc[te, "base"] + correction(
            model, v.loc[te, FEATURES], v.loc[te])).clip(lower=0.0)
        A(f"  train {tr_fold}{' + extra months' if extra is not None else ''} -> "
          f"score {te_fold}: best_iteration={best}  ({time.time()-t0:.0f}s, "
          f"{len(tr):,} training rows)")
    res = {}
    for fold in ("JAN", "JUL"):
        f = v["fold"] == fold
        res[fold] = (rmse(v["y"][f] - v["base"][f]), rmse(v["y"][f] - v["corr_pred"][f]))
    pb, pc = rmse(v["y"] - v["base"]), rmse(v["y"] - v["corr_pred"])
    A("")
    A(f"  {'':<22}{'rows':>9}{'base':>11}{'corrected':>11}{'delta':>9}")
    for fold in ("JAN", "JUL"):
        n = int((v["fold"] == fold).sum())
        A(f"  {fold:<22}{n:>9,}{res[fold][0]:>11.2f}{res[fold][1]:>11.2f}"
          f"{res[fold][1]-res[fold][0]:>+9.2f}")
    A(f"  {'POOLED':<22}{len(v):>9,}{pb:>11.2f}{pc:>11.2f}{pc-pb:>+9.2f}")
    A("")
    A("  subsets (both folds pooled)")
    for name, n, b, c in breakdown(v, "corr_pred"):
        A(f"  {name:<22}{n:>9,}{b:>11.2f}{c:>11.2f}{c-b:>+9.2f}")
    A("")
    A("  per airport (pooled)")
    for ap, g in v.groupby("ADEP_mvt"):
        b, c = rmse(g["y"] - g["base"]), rmse(g["y"] - g["corr_pred"])
        A(f"  {ap:<8} rows={len(g):>7,}  ADS-B ok={g['adsb_ok'].mean():>6.1%}  "
          f"base={b:>8.2f}  corrected={c:>8.2f}  delta={c-b:>+8.2f}")
    e2b = (v["y"] - v["base"]) ** 2
    e2c = (v["y"] - v["corr_pred"]) ** 2
    gain = e2b - e2c
    top20 = np.sort(gain.to_numpy())[::-1][:20].sum()
    A("")
    A(f"  net SSE reduction: {gain.sum():.4e}   share from top-20 rows: "
      f"{top20 / gain.sum():.1%}" if gain.sum() > 0 else
      f"  net SSE change: {gain.sum():.4e} (no reduction)")
    A(f"  RMSE without the 20 worst base rows: base "
      f"{math.sqrt(np.sort(e2b)[:-20].mean()):.2f}  corrected "
      f"{math.sqrt(e2c[np.argsort(e2b.to_numpy())[:-20]].mean()):.2f}")
    passed = (pb - pc >= GATE_GAIN) and all(res[f][1] < res[f][0] for f in res)
    A("")
    A(f"  GATE (pooled gain >= {GATE_GAIN} and both folds better): "
      f"{'PASS' if passed else 'FAIL'}")
    return passed, int(round(np.mean(bests)))


def ranking_frame(feats):
    cols = [MID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt",
            "EOBT_1_flt", "IOBT_flt", "LOBT_flt", "SCHED_TIME_UTC_mvt"]
    r = pq.read_table(RAW / "ranking.parquet", columns=cols).to_pandas()
    r = r[r["PHASE_mvt"] == "DEP"].copy()
    M = r["MVT_TIME_UTC_mvt"]
    r["d_AOBT3"] = (M - r["AOBT_3_flt"]).dt.total_seconds()
    r["d_SCHED"] = (M - r["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    ref_null = r[["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]].isna().sum(axis=1)
    if not ref_null.isin([0, 4]).all():
        stop("ranking reference-timestamp missingness is not only {0,4}")
    r = r.merge(feats, on=MID, how="left")
    r["adsb_matched"] = r["adsb_matched"].fillna(0)
    r["adsb_ok"] = r["adsb_ok"].fillna(0)
    r = add_weather(r)
    return r


def build_submission(v, n_iter, v4_path, extra=None):
    A("")
    A("=" * 78)
    A(f"V{VERSION} SUBMISSION  (ranking base: {v4_path})")
    A("=" * 78)
    dest = OUT_DIR / f"{TEAM}_v{VERSION}.parquet"
    if dest.exists():
        stop(f"{dest} already exists; refusing to overwrite")
    template = pd.read_parquet(RAW / "submitting.parquet")
    tcols = list(template.columns)
    if MID not in tcols or TARGET not in tcols or len(template) != EXPECTED_RANKING:
        stop(f"unexpected template: {tcols} rows={len(template)}")
    v4 = pd.read_parquet(v4_path)
    if len(v4) != EXPECTED_RANKING or v4[MID].duplicated().any():
        stop("V4 submission file has wrong row count or duplicate ids")
    if set(v4[MID]) != set(template[MID]):
        stop("V4 submission ids differ from the template")
    if not np.isfinite(v4[TARGET]).all():
        stop("V4 submission has non-finite predictions")

    feats = load_feats(USE_CONTEXT)
    r = ranking_frame(feats)
    if len(r) != EXPECTED_RANKING:
        stop(f"ranking departures {len(r):,} != {EXPECTED_RANKING:,}")
    r = r.merge(v4[[MID, TARGET]].rename(columns={TARGET: "base"}), on=MID, how="left")
    if r["base"].isna().any():
        stop("some ranking departures have no V4 base prediction")
    r = add_derived(r)
    A(f"  ranking departures with ADS-B reaching runway: {r['adsb_ok'].mean():.1%}")
    for ap, g in r.groupby("ADEP_mvt"):
        A(f"    {ap}: adsb_ok {g['adsb_ok'].mean():6.1%}   "
          f"pushback-like {((g['adsb_ok'] == 1) & (g['adsb_gs0'] < 6)).mean():6.1%}")

    vt = v if extra is None else pd.concat(
        [v, extra[v.columns.intersection(extra.columns)]], ignore_index=True)
    X_all = vt[FEATURES]
    y_all = np.clip((vt["y"] - vt["base"]).to_numpy(), -CLIP, CLIP)
    cat_models, sec_models = [], []
    inner = (vt[MID].abs() % 5 == 0).to_numpy()
    for sd in SEEDS:
        m = CatBoostRegressor(iterations=n_iter, **dict(CB, random_seed=sd))
        m.fit(X_all, y_all)
        cat_models.append(m)
        if SECOND != "none":
            fn = _fit_lgbm if SECOND == "lgbm" else _fit_cat2
            second, best2 = fn(X_all.to_numpy(dtype=np.float64) if SECOND == "lgbm" else X_all,
                               y_all, inner, seed=sd if SECOND == "lgbm" else sd + 1000)
            sec_models.append(second)
            A(f"  seed {sd}: second model ({SECOND}) for the final corrector: {best2} "
              "iterations (inner split)")
    fam = [Averaged(cat_models)] + ([Averaged(sec_models)] if sec_models else [])
    model = fam[0] if len(fam) == 1 else Averaged(fam)
    Xr = r[FEATURES]
    r["pred"] = r["base"] + correction(model, Xr, r)
    n_neg = int((r["pred"] < 0).sum())
    r["pred"] = r["pred"].clip(lower=0.0)   # taxi time cannot be negative
    A(f"  predictions floored at 0: {n_neg} rows")
    corr = r["pred"] - r["base"]
    A(f"  final corrector: JAN+JUL 2025"
      f"{' + extra months' if extra is not None else ''}, {n_iter} trees, "
      f"{len(vt):,} training rows")
    A(f"  correction on ranking: mean {corr.mean():+.1f}s  median {corr.median():+.1f}s  "
      f"p1 {corr.quantile(.01):+.0f}  p99 {corr.quantile(.99):+.0f}")
    if not np.isfinite(r["pred"]).all():
        stop("non-finite V5 predictions")

    sub = template[[MID]].merge(r[[MID, "pred"]], on=MID, how="left")
    sub[TARGET] = sub["pred"].astype(template[TARGET].dtype
                                     if template[TARGET].dtype.kind == "f" else "float64")
    sub = sub[tcols]
    checks = {
        "rows == 344,841": len(sub) == EXPECTED_RANKING,
        "columns match template": list(sub.columns) == tcols,
        "row order matches template": (sub[MID].to_numpy() == template[MID].to_numpy()).all(),
        "unique ids": not sub[MID].duplicated().any(),
        "no NaN/inf": bool(np.isfinite(sub[TARGET]).all()),
    }
    for k, ok in checks.items():
        A(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    if not all(checks.values()):
        stop("submission integrity check failed")
    A(f"  prediction min/median/max: {sub[TARGET].min():.0f} / "
      f"{sub[TARGET].median():.0f} / {sub[TARGET].max():.0f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(dest, index=False)
    A(f"  wrote {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-submission", "--base-submission", dest="v4_submission",
                    default="", help="ranking predictions of the base model")
    ap.add_argument("--m4-from-v4", action="store_true",
                    help="with --base-oof: use our V4-style F3 predictions on "
                         "MISSING_4 rows (when the ranking file's F3 is V4-style)")
    ap.add_argument("--base-oof", nargs="*", default=None,
                    help="honest Jan/Jul OOF parquet(s) of another base model "
                         "(MVT_ID_mvt, fold, pred); default = V4 benchmark cache")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="build even if the gate fails")
    ap.add_argument("--drop-features", default="",
                    help="comma-separated corrector features to leave out, e.g. adsb_npts")
    ap.add_argument("--second-model", choices=["none", "lgbm", "cat2"], default="none",
                    help="add a second tree model and average the two corrections "
                         "with EQUAL weights (lgbm = LightGBM, cat2 = CatBoost variant)")
    ap.add_argument("--extra-oof", nargs="*", default=None,
                    help="honest OOF parquet(s) of the SAME base model for extra 2025 "
                         "holdout months (MVT_ID_mvt, fold, y, pred). Used as TRAINING "
                         "rows only; validation stays JAN/JUL")
    ap.add_argument("--zero-unmatched", action="store_true",
                    help="apply no correction to flights with no ADS-B callsign match")
    ap.add_argument("--save-oof", default="",
                    help="write the corrector's honest JAN/JUL predictions (MVT_ID_mvt, fold, "
                         "state, ADEP_mvt, y, base, corr_pred) to this parquet, for blending")
    ap.add_argument("--seeds", type=int, default=1,
                    help="train the corrector with N random seeds (42, 43, ...) and "
                         "average them (equal weights)")
    ap.add_argument("--max-trees", type=int, default=None,
                    help="cap on corrector trees for early stopping (default 3000)")
    ap.add_argument("--context", action="store_true",
                    help="add airport-context features (data/processed/context_features.parquet)")
    ap.add_argument("--gate", action="store_true",
                    help="add gate-retention features (data/processed/gate_features.parquet)")
    ap.add_argument("--weather", action="store_true",
                    help="add METAR weather features (data/external/metar/metar_10apt.parquet)")
    ap.add_argument("--version", type=int, default=None,
                    help="submission version number to write (vigorous-glove_v<N>.parquet); "
                         "agree it with your teammate first")
    args = ap.parse_args()
    global VERSION, FEATURES, SECOND, USE_WEATHER, SEEDS, USE_CONTEXT, MAX_IT, USE_GATE
    if args.max_trees:
        MAX_IT = args.max_trees
    USE_CONTEXT = args.context
    if USE_CONTEXT:
        FEATURES = FEATURES + CTX
    USE_GATE = args.gate
    if USE_GATE:
        FEATURES = FEATURES + GATE
    SEEDS = [42 + i for i in range(max(1, args.seeds))]
    USE_WEATHER = args.weather
    if USE_WEATHER:
        FEATURES = FEATURES + WX
    if args.version is not None:
        VERSION = args.version
    drop = [c.strip() for c in args.drop_features.split(",") if c.strip()]
    unknown = [c for c in drop if c not in FEATURES]
    if unknown:
        stop(f"--drop-features: unknown feature(s) {unknown}")
    FEATURES = [c for c in FEATURES if c not in drop]
    SECOND = args.second_model
    global ZERO_UNMATCHED
    ZERO_UNMATCHED = args.zero_unmatched
    A(f"corrector features: {len(FEATURES)}"
      + (f"  (dropped: {', '.join(drop)})" if drop else "")
      + f"   second model: {SECOND}   seeds: {SEEDS}   max trees: {MAX_IT}")
    if not FEATS_PATH.exists():
        stop(f"missing {FEATS_PATH} - run adsb_features.py first")
    feats = load_feats(args.context)
    missing = [c for c in ADSB if c not in feats.columns]
    if missing:
        stop(f"feature file lacks {missing} - rerun the updated adsb_features.py")
    v = load_validation(feats, args.base_oof, args.m4_from_v4)
    A(f"validation rows: {len(v):,}  with ADS-B reaching runway: {v['adsb_ok'].mean():.1%}")
    extra = None
    if args.extra_oof:
        extra = load_extra(feats, args.extra_oof, set(v[MID]))
    if ZERO_UNMATCHED:
        A("correction set to 0 for flights without an ADS-B callsign match")
    passed, n_iter = validate(v, extra)
    if args.save_oof:
        cols = [MID, "fold", "state", "ADEP_mvt", "y", "base", "corr_pred"]
        Path(args.save_oof).parent.mkdir(parents=True, exist_ok=True)
        v[cols].to_parquet(args.save_oof, index=False)
        A(f"saved corrector OOF predictions: {args.save_oof}  ({len(v):,} rows)")
    write_report()
    if args.validate_only:
        A("validate-only: no submission built")
    elif not (passed or args.force):
        A("gate failed: no submission built (use --force to override)")
    else:
        if not args.v4_submission:
            stop("--v4-submission is required to build the submission")
        build_submission(v, n_iter, Path(args.v4_submission), extra)
    write_report()
    A("DONE")


if __name__ == "__main__":
    main()