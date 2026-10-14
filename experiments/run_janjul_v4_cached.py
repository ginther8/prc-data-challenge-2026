"""JAN/JUL V4 BENCHMARK - HONEST F3, PER-ROW CACHE, MONTH-OOF PRIORS ARM

PURPOSE
    1. Re-measure V4 (A4 on COMPLETE + F3 on MISSING_4) with the SAME F3
       procedure the V4 submission uses (nested inner-split iteration
       selection). The earlier benchmark (run_fast_janjul_a3_a4.py) early-
       stopped F3 on the outer JAN/JUL MISSING_4 rows, so its 344.2478 is
       optimistic on MISSING_4.
    2. Cache per-row validation predictions for every component so that all
       later analysis (A3/A4 blending, residual autopsy) needs no retraining.
    3. One cheap candidate arm: F3 with leave-one-MONTH-out priors instead of
       leave-one-out priors (see F3_MOOF below).

WHAT IS UNCHANGED (copied from run_fast_janjul_a3_a4.py / V4 submission)
    - data prep (prepare_frame), feature lists, categorical handling, to_cb
    - A3 / A4 per-ADEP COMPLETE models, nested 3000/100, CB_PARAMS from the
      V8 reference (depth 8, lr 0.1, RMSE, seed 42)
    - F3 23-feature set, per-airport specialists, PER_AIRPORT_MIN_ROWS
    - folds: JAN = 2025-01, JUL = 2025-07, train = all other 2025 months

COMPONENTS (each cached per row as soon as it is fitted)
    pred_A3        A3 per-ADEP COMPLETE, target TAXITIME_SEC_mvt           [canary]
    pred_A4        A4 per-ADEP COMPLETE, target TAXITIME - d_AOBT3        [canary]
    pred_F3_est    F3 with OUTER-validation early stopping (old benchmark) [canary, NOT honest]
    pred_F3_nest   F3 with nested inner-split selection, 1000/50 probe = the V4 SUBMISSION procedure
    pred_F3_moof   as F3_nest, but training-row priors are leave-one-MONTH-out
    pred_A4_s<k>   optional extra A4 seeds (--a4-seeds), for seed averaging

    COMPLETE rows carry A3/A4 predictions; MISSING_4 rows carry F3 predictions.
    Cells that do not apply to a row are NaN.

MONTH-OOF PRIORS (F3_MOOF)
    Source population and keys are identical to F3 (all outer-training
    departures, the same 4 key sets). For an outer-training row in month m the
    aggregate is computed from outer-training rows of every month EXCEPT m, so
    the row's own target and its same-month neighbours never enter. A group
    with no rows outside month m gets count NaN / mean NaN - the same encoding
    an unseen group gets at validation and ranking time. Validation rows keep
    the plain outer-training aggregate (unchanged). Ranking would keep the plain
    all-2025 aggregate (unchanged).

GATES
    Canary checks compare to the numbers published in reports/fast_janjul_a3_a4.txt:
      A3 COMPLETE RMSE  JAN 219.8856  JUL 317.3917
      A4 COMPLETE RMSE  JAN 222.5198  JUL 272.2238
      F3_est MISSING_4  JAN 2933.6068 JUL 1341.3579
    A3/A4 must match within 1.0 % or the run STOPS (setup differs from the
    friend's machine). F3_est is reported with a 5 % soft check only, because
    outer-early-stopping F3 was shown to jitter ~1 % run to run.

CACHE / RESUME
    data/processed/janjul_v4_cache_JAN.parquet, ..._JUL.parquet, plus
    data/processed/janjul_v4_cache_meta.json (best iterations, wall times).
    A component already present in the cache (same row IDs) is skipped.
    --report-only rebuilds the report from the cache without fitting anything.
    --force recomputes every component.

SAFETY
    BLOCK_TIME_UTC_mvt is never loaded. Outer validation targets are used only
    for scoring, except in the explicitly labelled F3_est canary. Priors never
    use validation-month targets. All splicing is by MVT_ID_mvt with asserts.

Run from the project root:
    python experiments/run_janjul_v4_cached.py                 # full run
    python experiments/run_janjul_v4_cached.py --a4-seeds 7,99 # + extra A4 seeds
    python experiments/run_janjul_v4_cached.py --report-only
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_ROOT / "reference" / "v8"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

sys.path.insert(0, str(REFERENCE_DIR))
from train_catboost_v8_defs import (  # noqa: E402
    CB_PARAMS,
    V7_CATEGORICALS,
    V7_FEATURES,
    T4_FEATURES,
)

from catboost import CatBoostRegressor  # noqa: E402

TARGET = "TAXITIME_SEC_mvt"
MINUEND = "MVT_TIME_UTC_mvt"
MID = "MVT_ID_mvt"
FORBIDDEN = "BLOCK_TIME_UTC_mvt"
EXPECTED_DEPARTURES = 2_085_047

PREP_CACHE = PROJECT_ROOT / "data" / "processed" / "v8_prep_cache.parquet"
CACHE_DIR = PROJECT_ROOT / "data" / "processed"
META_PATH = CACHE_DIR / "janjul_v4_cache_meta.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "janjul_v4_cached.txt"

FOLDS = {"JAN": "2025-01", "JUL": "2025-07"}
EXPECTED = {
    "JAN": {"valid": 153_706, "train": 1_931_341},
    "JUL": {"valid": 190_713, "train": 1_894_334},
}
CANARY = {  # from reports/fast_janjul_a3_a4.txt
    "A3": {"JAN": 219.8856, "JUL": 317.3917},       # COMPLETE-only RMSE
    "A4": {"JAN": 222.5198, "JUL": 272.2238},       # COMPLETE-only RMSE
    "F3_est": {"JAN": 2933.6068, "JUL": 1341.3579},  # MISSING_4-only RMSE
}
CANARY_HARD_TOL = 0.01
CANARY_SOFT_TOL = 0.05

REFERENCE_DELTA_COLS = ["d_AOBT3", "d_EOBT1", "d_IOBT", "d_LOBT"]
MISSING_LABEL = "MISSING_4"
COMPLETE_LABEL = "COMPLETE"
ADEP_REF_STATE = "adep_reference_state"

# ---- frozen F3 (MISSING_4 specialist), 23 features, never modified --------
SPEC_CATEGORICALS = [
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt", "FLIGHT_mvt", "ADEP_FLIGHT", "ADEP_ADES_FLIGHT",
    "ADEP_RUNWAY_STAND",
]
SPEC_NUMERIC = ["hour", "day_of_week", "month", "traffic_last_15min", "d_SCHED"]
HIST_SPECS = [
    ("hist_adep_flight_mean", "hist_adep_flight_count",
     ["ADEP_mvt", "FLIGHT_mvt"]),
    ("hist_adep_ades_flight_mean", "hist_adep_ades_flight_count",
     ["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt"]),
    ("hist_adep_runway_stand_mean", "hist_adep_runway_stand_count",
     ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]),
    ("hist_adep_ades_mean", "hist_adep_ades_count", ["ADEP_mvt", "ADES_mvt"]),
]
HIST_MEAN_COLS = [m for m, _, _ in HIST_SPECS]
HIST_COUNT_COLS = [c for _, c, _ in HIST_SPECS]
F3_FEATURES = SPEC_CATEGORICALS + SPEC_NUMERIC + HIST_MEAN_COLS + HIST_COUNT_COLS
F3_CATEGORICALS = list(SPEC_CATEGORICALS)

V8_FEATURES = list(V7_FEATURES) + list(T4_FEATURES)
V8_CATEGORICALS = list(V7_CATEGORICALS)
EXP04_FEATURES = V8_FEATURES + [ADEP_REF_STATE]
EXP04_CATEGORICALS = V8_CATEGORICALS + [ADEP_REF_STATE]

INTERACTION_SPECS = [
    ("ADEP_FLIGHT", ["ADEP_mvt", "FLIGHT_mvt"]),
    ("ADEP_ADES_FLIGHT", ["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt"]),
    ("ADEP_RUNWAY_STAND", ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]),
]

PER_AIRPORT_MIN_ROWS = 500
Y_COL = "__y"

# F3 established (outer-validation early stopping) - canary only, NOT honest.
EST_CB = dict(CB_PARAMS)
EST_CB.update(verbose=False, iterations=1000, early_stopping_rounds=50,
              use_best_model=True)

# F3 nested - exactly the V4 submission's PROBE_PARAMS (1000 / 50).
F3_NEST_CB = dict(CB_PARAMS)
F3_NEST_CB.update(verbose=False, iterations=1000, early_stopping_rounds=50,
                  use_best_model=True)

# A3/A4 nested - exactly the V4 benchmark / submission (3000 / 100).
A_NEST_CB = dict(CB_PARAMS)
A_NEST_CB.update(verbose=False, iterations=3000, early_stopping_rounds=100,
                 use_best_model=True)

OUT, CHECKS = [], {}


# ==========================================================================
# logging / report helpers
# ==========================================================================
def log(m=""):
    print(m, flush=True)


def A(m=""):
    OUT.append(str(m))


def rule(t):
    A()
    A("=" * 78)
    A(t)
    A("=" * 78)


def sub(t):
    A()
    A(t)
    A("-" * 78)


def check(label, ok, detail=""):
    CHECKS[label] = bool(ok)
    line = f"  [{'PASS' if ok else 'FAIL'}] {label}"
    if detail:
        line += f"  ({detail})"
    A(line)
    log(line)
    return bool(ok)


def write_report():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(OUT) + "\n")


def stop(reason):
    rule("STOPPED")
    A(f"  reason: {reason}")
    write_report()
    log(f"STOPPED: {reason}")
    sys.exit(1)


def sse(y, p):
    e = np.asarray(y, np.float64) - np.asarray(p, np.float64)
    return float(np.dot(e, e))


def rmse(y, p):
    return math.sqrt(sse(y, p) / len(y))


# ==========================================================================
# cache
# ==========================================================================
def cache_path(fold):
    return CACHE_DIR / f"janjul_v4_cache_{fold}.parquet"


def load_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def save_meta(meta):
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True))


def init_cache(fold, va):
    """Base frame for a fold: one row per validation departure, ID-keyed."""
    base = va.select([
        pl.col(MID), pl.lit(fold).alias("fold"), pl.col("state"),
        pl.col("ADEP_mvt"), pl.col(TARGET).cast(pl.Float64).alias("y"),
        pl.col("d_AOBT3").cast(pl.Float64).alias("d_AOBT3"),
        pl.col("d_SCHED").cast(pl.Float64).alias("d_SCHED"),
    ])
    p = cache_path(fold)
    if p.exists():
        old = pl.read_parquet(p)
        if old.height == base.height and old[MID].to_list() == base[MID].to_list():
            return old
        log(f"  cache for {fold} does not match current rows; starting fresh")
    base.write_parquet(p)
    return base


def put(fold, cache, col, values):
    """Attach a full-length prediction column (validation order) and persist."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) != cache.height:
        stop(f"{fold}: {col} has {len(values)} values for {cache.height} rows")
    cache = cache.with_columns(pl.Series(col, values))
    cache.write_parquet(cache_path(fold))
    return cache


# ==========================================================================
# data
# ==========================================================================
def prepare_frame():
    if not PREP_CACHE.exists():
        stop(f"prepared cache missing: {PREP_CACHE} "
             "(rebuild with experiments/rebuild_v8_prep_cache.py)")
    frame = pl.read_parquet(PREP_CACHE)
    if not check("training departures == 2,085,047",
                 frame.height == EXPECTED_DEPARTURES, f"{frame.height:,}"):
        stop("row count")
    if not check(f"{FORBIDDEN} absent from the feature frame",
                 FORBIDDEN not in frame.columns):
        stop("forbidden column present")
    files = sorted(DATA_DIR.glob("training_*.parquet"))
    raw = (pl.scan_parquet([str(f) for f in files])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select([MID, "FLIGHT_mvt"]).collect())
    before = frame.height
    frame = frame.join(raw, on=MID, how="left")
    if not check("FLIGHT_mvt join preserved row count", frame.height == before):
        stop("FLIGHT_mvt join")
    frame = frame.with_columns(
        pl.col("FLIGHT_mvt").cast(pl.String).fill_null("UNKNOWN").alias("FLIGHT_mvt"))
    nmiss = sum(pl.col(c).is_null().cast(pl.Int32) for c in REFERENCE_DELTA_COLS)
    frame = frame.with_columns(nmiss.alias("_nm"))
    frame = frame.with_columns(
        pl.when(pl.col("_nm") == 4).then(pl.lit(MISSING_LABEL))
        .otherwise(pl.lit(COMPLETE_LABEL)).alias("state"))
    frame = frame.with_columns(
        (pl.col("ADEP_mvt") + "|" + pl.col("state")).alias(ADEP_REF_STATE))
    for name, parts in INTERACTION_SPECS:
        frame = frame.with_columns(
            pl.concat_str([pl.col(p) for p in parts], separator="|")
            .fill_null("UNKNOWN").alias(name))
    frame = frame.with_columns(pl.col(MINUEND).dt.strftime("%Y-%m").alias("ym"))
    frame = frame.drop("_nm")
    check("MISSING_4 pattern is only {0,4}",
          set(frame["state"].unique().to_list()) <= {MISSING_LABEL, COMPLETE_LABEL})
    if not check("MVT_ID_mvt unique", frame[MID].is_unique().all()):
        stop("duplicate ids")
    return frame


# ==========================================================================
# priors
# ==========================================================================
def add_priors(source, rows, *, rows_in_source):
    """ORIGINAL F3 priors (unchanged). Training rows: leave-one-out.
    Validation rows: plain outer-training aggregate."""
    out = rows
    for mean_col, cnt_col, keys in HIST_SPECS:
        agg = (source.group_by(keys)
               .agg(pl.col(TARGET).sum().alias("_s"),
                    pl.len().cast(pl.Float64).alias("_n")))
        cols = list(keys) + ([TARGET] if rows_in_source else [])
        j = out.select(cols).join(agg, on=keys, how="left")
        s = j["_s"].to_numpy().astype(np.float64)
        n = j["_n"].to_numpy().astype(np.float64)
        ins = np.isfinite(n)
        if rows_in_source:
            cnt = np.where(ins, n - 1.0, 0.0)
            sl = np.where(ins, s - j[TARGET].to_numpy().astype(np.float64), s)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = np.where(cnt > 0, sl / cnt, np.nan)
            cnt_out = np.where(ins, cnt, np.nan)
        else:
            cnt_out = np.where(ins, n, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = np.where(ins, s / n, np.nan)
        out = out.with_columns(pl.Series(cnt_col, cnt_out), pl.Series(mean_col, mean))
    return out


def add_priors_month_oof(source, rows):
    """Leave-one-MONTH-out priors for rows that are part of `source`.
    For a row in month m: aggregate over source rows whose month != m.
    Zero remaining support -> count NaN, mean NaN (same as an unseen group at
    validation / ranking time)."""
    out = rows
    for mean_col, cnt_col, keys in HIST_SPECS:
        agg_all = (source.group_by(keys)
                   .agg(pl.col(TARGET).sum().alias("_sa"),
                        pl.len().cast(pl.Float64).alias("_na")))
        agg_mon = (source.group_by(keys + ["ym"])
                   .agg(pl.col(TARGET).sum().alias("_sm"),
                        pl.len().cast(pl.Float64).alias("_nm")))
        j = (out.select(keys + ["ym"])
             .with_row_index("_ri")
             .join(agg_all, on=keys, how="left")
             .join(agg_mon, on=keys + ["ym"], how="left")
             .sort("_ri"))
        if j.height != out.height:
            stop(f"month-OOF prior join changed row count for {mean_col}")
        sa = j["_sa"].fill_null(0.0).to_numpy().astype(np.float64)
        na = j["_na"].fill_null(0.0).to_numpy().astype(np.float64)
        sm = j["_sm"].fill_null(0.0).to_numpy().astype(np.float64)
        nm = j["_nm"].fill_null(0.0).to_numpy().astype(np.float64)
        n = na - nm
        s = sa - sm
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(n > 0, s / n, np.nan)
        cnt = np.where(n > 0, n, np.nan)
        out = out.with_columns(pl.Series(cnt_col, cnt), pl.Series(mean_col, mean))
    return out


# ==========================================================================
# model fitting
# ==========================================================================
def to_cb(frame, features, categoricals):
    df = frame.select(features).to_pandas()
    for c in categoricals:
        df[c] = df[c].astype(str)
    return df


def nested_fit(rows, features, cats, params, seed=None):
    """best_iteration on the target-free inner split abs(MVT_ID) % 5 == 0,
    then refit on ALL rows with best_iteration + 1 trees, no eval_set.
    Identical to nested_fit() / nested_iteration()+refit() in the V4 code."""
    tcol = Y_COL if Y_COL in rows.columns else TARGET
    inner_valid = rows.filter(pl.col(MID).abs() % 5 == 0)
    inner_train = rows.filter(~(pl.col(MID).abs() % 5 == 0))
    if inner_train.height == 0 or inner_valid.height == 0:
        return None, None
    probe_p = dict(params)
    if seed is not None:
        probe_p["random_seed"] = seed
    probe = CatBoostRegressor(allow_writing_files=False, **probe_p)
    probe.fit(to_cb(inner_train, features, cats),
              inner_train[tcol].to_numpy().astype(np.float64),
              cat_features=cats,
              eval_set=(to_cb(inner_valid, features, cats),
                        inner_valid[tcol].to_numpy().astype(np.float64)),
              early_stopping_rounds=probe_p["early_stopping_rounds"],
              use_best_model=True)
    best = int(probe.get_best_iteration())
    del probe
    rp = dict(probe_p)
    rp["iterations"] = best + 1
    rp.pop("early_stopping_rounds", None)
    rp.pop("use_best_model", None)
    model = CatBoostRegressor(allow_writing_files=False, **rp)
    model.fit(to_cb(rows, features, cats),
              rows[tcol].to_numpy().astype(np.float64), cat_features=cats)
    return model, best


def est_fit(train_rows, valid_rows, features, cats):
    """CANARY ONLY: outer validation is the early-stopping set (not honest)."""
    model = CatBoostRegressor(allow_writing_files=False, **EST_CB)
    model.fit(to_cb(train_rows, features, cats),
              train_rows[TARGET].to_numpy().astype(np.float64),
              cat_features=cats,
              eval_set=(to_cb(valid_rows, features, cats),
                        valid_rows[TARGET].to_numpy().astype(np.float64)),
              early_stopping_rounds=EST_CB["early_stopping_rounds"],
              use_best_model=True)
    return model, int(model.get_best_iteration())


def splice(va_ids, part_ids, part_pred, n):
    """ID-based splice of a partial prediction vector into validation order."""
    pos = {m: i for i, m in enumerate(va_ids)}
    out = np.full(n, np.nan, dtype=np.float64)
    idx = np.fromiter((pos[m] for m in part_ids), dtype=np.int64,
                      count=len(part_ids))
    if len(np.unique(idx)) != len(idx):
        stop("splice produced duplicate positions")
    out[idx] = part_pred
    return out


# ==========================================================================
# components
# ==========================================================================
def fit_f3(fold, tr, va, mode, meta):
    """mode: 'est' (canary, outer ES), 'nest' (V4 submission), 'moof'."""
    m4_tr = tr.filter(pl.col("state") == MISSING_LABEL)
    m4_va = va.filter(pl.col("state") == MISSING_LABEL)
    counts = dict(m4_tr.group_by("ADEP_mvt").agg(pl.len().alias("n")).iter_rows())
    small = sorted(a for a, n in counts.items() if n < PER_AIRPORT_MIN_ROWS)
    if small:
        stop(f"{fold} F3: airports below {PER_AIRPORT_MIN_ROWS} M4 rows {small}; "
             "global fallback not implemented in this benchmark")
    va_ids = va[MID].to_list()
    pred = np.full(va.height, np.nan, dtype=np.float64)
    iters = {}
    if mode == "moof":
        tr_fe_all = add_priors_month_oof(tr, m4_tr)
    for ap in sorted(counts):
        vai = m4_va.filter(pl.col("ADEP_mvt") == ap)
        if vai.height == 0:
            continue
        if mode == "moof":
            tri_fe = tr_fe_all.filter(pl.col("ADEP_mvt") == ap)
        else:
            tri_fe = add_priors(tr, m4_tr.filter(pl.col("ADEP_mvt") == ap),
                                rows_in_source=True)
        vai_fe = add_priors(tr, vai, rows_in_source=False)
        if mode == "est":
            mdl, best = est_fit(tri_fe, vai_fe, F3_FEATURES, F3_CATEGORICALS)
        else:
            mdl, best = nested_fit(tri_fe, F3_FEATURES, F3_CATEGORICALS, F3_NEST_CB)
            if mdl is None:
                stop(f"{fold} F3_{mode} inner split empty for {ap}")
        p = np.asarray(mdl.predict(to_cb(vai_fe, F3_FEATURES, F3_CATEGORICALS)),
                       dtype=np.float64)
        part = splice(va_ids, vai_fe[MID].to_list(), p, va.height)
        pred = np.where(np.isfinite(part), part, pred)
        iters[ap] = best
        del mdl
    m4_mask = (va["state"] == MISSING_LABEL).to_numpy()
    if not check(f"{fold} F3_{mode} predicts every MISSING_4 row",
                 bool(np.isfinite(pred[m4_mask]).all())):
        stop(f"{fold} F3_{mode} missing predictions")
    meta.setdefault(fold, {})[f"F3_{mode}_iters"] = iters
    return pred


def fit_a(fold, tr, va, residual, meta, seed=None, tag=None):
    ctr = tr.filter(pl.col("state") == COMPLETE_LABEL)
    cva = va.filter(pl.col("state") == COMPLETE_LABEL)
    if residual:
        ctr = ctr.with_columns((pl.col(TARGET) - pl.col("d_AOBT3")).alias(Y_COL))
    va_ids = va[MID].to_list()
    pred = np.full(va.height, np.nan, dtype=np.float64)
    iters = {}
    aps = (ctr.group_by("ADEP_mvt").agg(pl.len().alias("n"))
           .filter(pl.col("n") >= PER_AIRPORT_MIN_ROWS)
           .get_column("ADEP_mvt").to_list())
    for ap in sorted(aps):
        tri = ctr.filter(pl.col("ADEP_mvt") == ap)
        vai = cva.filter(pl.col("ADEP_mvt") == ap)
        if vai.height == 0:
            continue
        t0 = time.time()
        mdl, best = nested_fit(tri, EXP04_FEATURES, EXP04_CATEGORICALS,
                               A_NEST_CB, seed=seed)
        p = np.asarray(mdl.predict(to_cb(vai, EXP04_FEATURES, EXP04_CATEGORICALS)),
                       dtype=np.float64)
        if residual:
            p = p + vai["d_AOBT3"].to_numpy().astype(np.float64)
        part = splice(va_ids, vai[MID].to_list(), p, va.height)
        pred = np.where(np.isfinite(part), part, pred)
        iters[ap] = best
        log(f"    {fold} {tag} {ap}: best_iteration={best}  "
            f"wall={time.time()-t0:.0f}s")
        del mdl
    c_mask = (va["state"] == COMPLETE_LABEL).to_numpy()
    if not check(f"{fold} {tag} predicts every COMPLETE row",
                 bool(np.isfinite(pred[c_mask]).all())):
        stop(f"{fold} {tag} missing predictions")
    meta.setdefault(fold, {})[f"{tag}_iters"] = iters
    return pred


# ==========================================================================
# report (works from cache only)
# ==========================================================================
def report(caches, meta):
    rule("RESULTS (all numbers recomputed from the per-row cache)")
    folds = list(caches)
    n_all = sum(c.height for c in caches.values())

    def arr(c, col):
        return c[col].to_numpy().astype(np.float64)

    comp_cols = [c for c in ("pred_A3", "pred_A4") if all(c in caches[f].columns for f in folds)]
    seed_cols = sorted({c for f in folds for c in caches[f].columns
                        if c.startswith("pred_A4_s")})
    seed_cols = [c for c in seed_cols if all(c in caches[f].columns for f in folds)]
    if seed_cols and "pred_A4" in comp_cols:
        for f in folds:
            cols = ["pred_A4"] + seed_cols
            caches[f] = caches[f].with_columns(
                pl.mean_horizontal([pl.col(c) for c in cols]).alias("pred_A4_avg"))
        comp_cols.append("pred_A4_avg")
    m4_cols = [c for c in ("pred_F3_est", "pred_F3_nest", "pred_F3_moof")
               if all(c in caches[f].columns for f in folds)]

    sub("COMPLETE-only RMSE")
    A(f"  {'model':<16}{'JAN':>12}{'JUL':>12}{'POOLED':>12}")
    for col in comp_cols:
        row, s_tot, n_tot = [], 0.0, 0
        for f in folds:
            c = caches[f]
            m = (c["state"] == COMPLETE_LABEL).to_numpy()
            y, p = arr(c, "y")[m], arr(c, col)[m]
            row.append(rmse(y, p))
            s_tot += sse(y, p)
            n_tot += int(m.sum())
        A(f"  {col:<16}" + "".join(f"{r:>12.4f}" for r in row)
          + f"{math.sqrt(s_tot/n_tot):>12.4f}")

    sub("MISSING_4-only RMSE")
    A(f"  {'model':<16}{'JAN':>12}{'JUL':>12}{'POOLED':>12}")
    for col in m4_cols:
        row, s_tot, n_tot = [], 0.0, 0
        for f in folds:
            c = caches[f]
            m = (c["state"] == MISSING_LABEL).to_numpy()
            y, p = arr(c, "y")[m], arr(c, col)[m]
            row.append(rmse(y, p))
            s_tot += sse(y, p)
            n_tot += int(m.sum())
        A(f"  {col:<16}" + "".join(f"{r:>12.4f}" for r in row)
          + f"{math.sqrt(s_tot/n_tot):>12.4f}")
    if "pred_F3_est" in m4_cols:
        A("  NOTE pred_F3_est early-stops on the scored rows: optimistic, canary only.")

    sub("FULL-VALIDATION RMSE (COMPLETE model x MISSING_4 model), pooled = combined SSE")
    A(f"  {'COMPLETE':<14}{'MISSING_4':<14}{'JAN':>12}{'JUL':>12}{'POOLED':>12}"
      f"{'M4 SSE share':>14}")
    for cc in comp_cols:
        for mc in m4_cols:
            row, s_tot, s_m4 = [], 0.0, 0.0
            for f in folds:
                c = caches[f]
                m4 = (c["state"] == MISSING_LABEL).to_numpy()
                p = np.where(m4, arr(c, mc), arr(c, cc))
                y = arr(c, "y")
                row.append(rmse(y, p))
                s_tot += sse(y, p)
                s_m4 += sse(y[m4], p[m4])
            A(f"  {cc[5:]:<14}{mc[5:]:<14}" + "".join(f"{r:>12.4f}" for r in row)
              + f"{math.sqrt(s_tot/n_all):>12.4f}{s_m4/s_tot:>14.1%}")
    A("  V4 as submitted = A4 x F3_nest.")

    if "pred_A4" in comp_cols and "pred_F3_nest" in m4_cols:
        sub("PER-AIRPORT RMSE, V4 as submitted (A4 x F3_nest)")
        A(f"  {'airport':<8}{'JAN rows':>10}{'JAN RMSE':>11}{'JUL rows':>10}"
          f"{'JUL RMSE':>11}{'SSE share':>11}")
        tot = 0.0
        per = {}
        for f in folds:
            c = caches[f]
            m4 = (c["state"] == MISSING_LABEL).to_numpy()
            p = np.where(m4, arr(c, "pred_F3_nest"), arr(c, "pred_A4"))
            e2 = (arr(c, "y") - p) ** 2
            tot += e2.sum()
            for ap, e in zip(c["ADEP_mvt"].to_list(), e2):
                d = per.setdefault(ap, {}).setdefault(f, [0, 0.0])
                d[0] += 1
                d[1] += e
        for ap in sorted(per):
            cells, s_ap = "", 0.0
            for f in folds:
                n, s = per[ap].get(f, [0, 0.0])
                s_ap += s
                cells += f"{n:>10,}{(math.sqrt(s/n) if n else float('nan')):>11.1f}"
            A(f"  {ap:<8}{cells}{s_ap/tot:>11.1%}")

    sub("CANARIES vs reports/fast_janjul_a3_a4.txt")
    for key, col, state in (("A3", "pred_A3", COMPLETE_LABEL),
                            ("A4", "pred_A4", COMPLETE_LABEL),
                            ("F3_est", "pred_F3_est", MISSING_LABEL)):
        if not all(col in caches[f].columns for f in folds):
            continue
        for f in folds:
            c = caches[f]
            m = (c["state"] == state).to_numpy()
            r = rmse(arr(c, "y")[m], arr(c, col)[m])
            ref = CANARY[key][f]
            rel = abs(r - ref) / ref
            tol = CANARY_SOFT_TOL if key == "F3_est" else CANARY_HARD_TOL
            A(f"  {key:<7}{f}: {r:.4f} vs {ref:.4f}  rel {rel:.2%}  "
              f"({'OK' if rel <= tol else 'OUTSIDE TOLERANCE'} at {tol:.0%})")

    sub("BEST ITERATIONS")
    for f in folds:
        for k, v in sorted(meta.get(f, {}).items()):
            if k.endswith("_iters") and isinstance(v, dict):
                vals = list(v.values())
                A(f"  {f} {k[:-6]:<10} median={statistics.median(vals):.0f} "
                  f"max={max(vals)}  " + " ".join(f"{a}:{b}" for a, b in sorted(v.items())))


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute every component even if cached")
    ap.add_argument("--a4-seeds", default="",
                    help="comma-separated extra A4 seeds, e.g. 7,99")
    args = ap.parse_args()
    extra_seeds = [int(s) for s in args.a4_seeds.split(",") if s.strip()]

    t_start = time.time()
    rule("JAN/JUL V4 BENCHMARK - HONEST F3 + PER-ROW CACHE")
    A(f"  started : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    A(f"  mode    : {'REPORT-ONLY' if args.report_only else 'COMPUTE'}"
      f"{' (force)' if args.force else ''}")
    A(f"  extra A4 seeds: {extra_seeds or 'none'}")
    for k, v in (("depth", 8), ("learning_rate", 0.1),
                 ("loss_function", "RMSE"), ("random_seed", 42)):
        if CB_PARAMS.get(k) != v:
            stop(f"reference {k}={CB_PARAMS.get(k)} != expected {v}")
    meta = load_meta()

    if args.report_only:
        caches = {}
        for f in FOLDS:
            if not cache_path(f).exists():
                stop(f"no cache for {f}")
            caches[f] = pl.read_parquet(cache_path(f))
        report(caches, meta)
        write_report()
        log("DONE (report-only)")
        return

    frame = prepare_frame()
    caches = {}
    for fold, ym in FOLDS.items():
        sub(f"FOLD {fold}  validation month {ym}")
        va = frame.filter(pl.col("ym") == ym)
        tr = frame.filter(pl.col("ym") != ym)
        if not check(f"{fold} validation rows == {EXPECTED[fold]['valid']:,}",
                     va.height == EXPECTED[fold]["valid"], f"{va.height:,}"):
            stop(f"{fold} validation rows")
        if not check(f"{fold} training rows == {EXPECTED[fold]['train']:,}",
                     tr.height == EXPECTED[fold]["train"], f"{tr.height:,}"):
            stop(f"{fold} training rows")
        if not check(f"{fold} train/valid months disjoint",
                     ym not in set(tr["ym"].unique().to_list())):
            stop("fold leakage")
        cache = init_cache(fold, va)
        c_mask = (va["state"] == COMPLETE_LABEL).to_numpy()

        jobs = [
            ("pred_F3_est", lambda: fit_f3(fold, tr, va, "est", meta)),
            ("pred_F3_nest", lambda: fit_f3(fold, tr, va, "nest", meta)),
            ("pred_F3_moof", lambda: fit_f3(fold, tr, va, "moof", meta)),
            ("pred_A4", lambda: fit_a(fold, tr, va, True, meta, tag="A4")),
            ("pred_A3", lambda: fit_a(fold, tr, va, False, meta, tag="A3")),
        ]
        for s in extra_seeds:
            jobs.append((f"pred_A4_s{s}",
                         lambda s=s: fit_a(fold, tr, va, True, meta, seed=s,
                                           tag=f"A4_s{s}")))
        for col, fn in jobs:
            if col in cache.columns and not args.force:
                A(f"  {col}: cached, skipped")
                log(f"  {fold} {col}: cached, skipped")
                continue
            t0 = time.time()
            log(f"  {fold} {col}: fitting ...")
            pred = fn()
            cache = put(fold, cache, col, pred)
            meta.setdefault(fold, {})[f"{col}_wall_s"] = round(time.time() - t0, 1)
            save_meta(meta)
            A(f"  {col}: fitted in {time.time()-t0:.0f}s, cached")

            # hard canary gate right after A3/A4 so a setup mismatch stops early
            if col in ("pred_A3", "pred_A4"):
                key = col[5:]
                r = rmse(cache["y"].to_numpy()[c_mask],
                         cache[col].to_numpy()[c_mask])
                ref = CANARY[key][fold]
                if not check(f"{fold} {key} COMPLETE RMSE within "
                             f"{CANARY_HARD_TOL:.0%} of published",
                             abs(r - ref) / ref <= CANARY_HARD_TOL,
                             f"{r:.4f} vs {ref:.4f}"):
                    stop(f"{fold} {key} canary failed - setup differs from the "
                         "published run; investigate before trusting any result")
            write_report()
        caches[fold] = cache

    report(caches, meta)
    rule("SAFETY")
    for label, ok in CHECKS.items():
        A(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    A(f"  wall seconds: {time.time()-t_start:.1f}")
    write_report()
    log("DONE")


if __name__ == "__main__":
    main()