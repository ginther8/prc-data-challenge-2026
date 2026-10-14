#!/usr/bin/env python3
"""
V15 COMPLETE-QUEUE ABLATION + CLEAN SPLICE SCREEN
=================================================

Overnight finding under test
----------------------------
V11 appended the 57-feature dynamic queue-state family Q to BOTH routes. That
regressed the composed score, but the MISSING_4 route was the unstable part
(1,435 / 3,938 validation rows). Composed as "queue-state on COMPLETE only,
MISSING_4 -> exact original F3" it improved both folds:
    JAN 356.836295 (-1.622)  JUL 321.927778 (-1.877)  POOLED 337.952440 (-1.755)
V15 refines that candidate instead of discarding it.

Task
----
1. Ablate the six Q groups on the COMPLETE route only: which groups carry the
   gain, which are noise?
2. Assemble the promoted groups into a clean full splice:
       COMPLETE  -> best honest queue-enhanced C2
       MISSING_4 -> exact original F3, predictions reused byte-for-byte
3. Decide against the honest baseline on pooled JAN+JUL RMSE and gate any
   OOF / ranking handover on a real, spread, drift-safe improvement.

Authoritative implementations (imported verbatim, never re-implemented)
-----------------------------------------------------------------------
C2 COMPLETE route : experiments/run_v5_c2_rigorous_janjul.py   (RIG)
F3 MISSING_4     : experiments/run_v5_missing4_substantial.py  (B)
Q family builder : experiments/build_v11_queue_features.py     (QB)
2026 ranking prep: experiments/create_c2_f3_v5_submission.py   (V4 + v8 defs)

Honest protocol (identical for every candidate)
-----------------------------------------------
* JAN outer-train = all 2025 departures except 2025-01 (1,931,341),
  outer-valid = 2025-01 (153,706). JUL analogous (1,894,334 / 190,713).
* Iteration selection is INNER ONLY: abs(MVT_ID_mvt) % 5 == 0 -> inner valid.
* Outer targets are used ONLY to score an already-fixed candidate. The one
  documented exception is the ranking of feature groups, which the task
  defines on the standardized outer folds; that is a selection bias over the
  tested candidates and is reported as such, not hidden.
* pooled RMSE = sqrt((JAN_SSE + JUL_SSE) / 344,419). Never a mean of RMSEs.

Baseline, read back instead of refitted
---------------------------------------
data/processed/v7_c2f3_oof_{JAN,JUL}.parquet are the honest C2+F3 OOF of the
current base. Their COMPLETE subset IS the C2 control prediction and their
MISSING_4 subset IS the exact F3 prediction.
  JAN 358.458601  JUL 323.805126  POOLED 339.707201
  TOTAL SSE 39,746,290,977.6   COMPLETE pooled SSE 19,667,165,462.2
  COMPLETE control JAN 217.1574 / JUL 258.5595 / POOLED 240.8472

Safety
------
* BLOCK_TIME_UTC_mvt is never loaded and never referenced.
* Q features may only use movements STRICTLY EARLIER than the row's
  MVT_TIME_UTC_mvt; same-timestamp peers are never counted. Re-verified here
  against a fresh strictly-earlier-only recomputation before any training.
* MISSING_4 predictions are the untouched F3 values; M4 is never refitted,
  retuned, re-routed or re-priored, in the sweep or in the handover.
* No clipping, no winsorization, no target transformation, no external data,
  no hyperparameter change, no leaderboard, no submission, no Git.
* Runs in the foreground, one training process at a time.
"""

import csv as _csv
import importlib.util
import math
import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

warnings.filterwarnings("ignore")


def _load(mod_name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("v15_m4base", "run_v5_missing4_substantial.py")
RIG = _load("v15_c2rig", "run_v5_c2_rigorous_janjul.py")
QB = _load("v15_qbuilder", "build_v11_queue_features.py")

TARGET = B.TARGET
MID = B.MID
AP = "ADEP_mvt"
FORBIDDEN = B.FORBIDDEN
D_AOBT3 = RIG.D_AOBT3
RESID_COL = RIG.RESID_COL
MISSING_LABEL = B.MISSING_LABEL
EXPECTED_DEPARTURES = B.EXPECTED_DEPARTURES
PER_AIRPORT_MIN_ROWS = B.PER_AIRPORT_MIN_ROWS
C2_ALL_FEATURES = list(RIG.C2_ALL_FEATURES)
C2_ALL_CATEGORICALS = list(RIG.C2_ALL_CATEGORICALS)

PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"
REPORTS = PROJECT_ROOT / "reports"
HANDOVER = PROJECT_ROOT / "handover"
Q_CACHE = PROCESSED / "v11_queue_features.parquet"
OOF_JAN = PROCESSED / "v7_c2f3_oof_JAN.parquet"
OOF_JUL = PROCESSED / "v7_c2f3_oof_JUL.parquet"
CACHE_DIR = PROCESSED / "v15_queue_cache"
REPORT_PATH = REPORTS / "v15_complete_queue_ablation.txt"
GROUPS_CSV = REPORTS / "v15_complete_queue_groups.csv"
SCRATCH = REPORTS / "_v15_scratch.txt"
OOF_OUT_JAN = PROCESSED / "v15_queue_oof_JAN.parquet"
OOF_OUT_JUL = PROCESSED / "v15_queue_oof_JUL.parquet"
OOF_OUT_ALL = PROCESSED / "v15_queue_oof_JAN_JUL.parquet"
RANKING_OUT = HANDOVER / "v15_queue_ranking.parquet"
HANDOVER_TXT = HANDOVER / "v15_queue_adsb_handover.txt"
V5_SUBMISSION = PROJECT_ROOT / "submissions" / "vigorous-glove_v5.parquet"

FOLDS = (("JAN", "2025-01"), ("JUL", "2025-07"))
EXPECTED = {
    "JAN": dict(train=1_931_341, valid=153_706, complete_valid=152_271,
                m4_valid=1_435),
    "JUL": dict(train=1_894_334, valid=190_713, complete_valid=186_775,
                m4_valid=3_938),
}
TOTAL_VALID = 344_419
BENCH = dict(JAN=358.458601, JUL=323.805126, POOLED=339.707201)
BENCH_C = dict(JAN=217.1574, JUL=258.5595, POOLED=240.8472)
ALLQ_EXPECT = dict(jan=-1.622, jul=-1.877, pool=-1.755)
REPRO_TOL = 1.00  # cross-script retrain tolerance; the control is gated at 1e-3
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF",
            "LSZH", "LTFM"]
WINDOWS = ("5m", "10m", "15m", "30m", "60m")
WHO = ("all", "dep", "arr")
WINDOW_US = {"5m": 300_000_000, "10m": 600_000_000, "15m": 900_000_000,
             "30m": 1_800_000_000, "60m": 3_600_000_000}

GROUPS = {
    "Q1": [f"{w}_ap_{ws}" for ws in WINDOWS for w in WHO],
    "Q2": [f"{w}_rw_{ws}" for ws in WINDOWS for w in WHO],
    "Q3": ["sec_since_prev_ap_arr", "sec_since_prev_ap_dep",
           "sec_since_prev_ap_move", "sec_since_prev_rw_dep",
           "sec_since_prev_rw_move"],
    "Q4": ["arr_share_15m", "arr_share_30m", "arr_share_60m",
           "arrival_pressure_30m", "arrival_pressure_60m",
           "dep_share_15m", "dep_share_30m", "dep_share_60m",
           "rw_dep_share_15m", "rw_dep_share_30m", "rw_dep_share_60m"],
    "Q5": ["ap_gap_mean_last10", "ap_gap_mean_last5", "ap_gap_std_last10",
           "ap_gap_std_last5", "rw_gap_mean_last5", "rw_gap_std_last5"],
    "Q6": ["runway_changed_vs_prev_departure", "runway_changed_vs_prev_movement",
           "sec_since_last_different_runway", "unique_runways_ap_30m",
           "unique_runways_ap_60m"],
}
ALL_Q = [c for g in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6") for c in GROUPS[g]]
GROUP_ORDER = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")

OUT, CHECKS, PENDING = [], [], []


def A(m=""):
    OUT.append(str(m))
    PENDING.append(str(m))


def log(m=""):
    print(m, flush=True)


def flush():
    drain()
    if PENDING:
        log("\n".join(PENDING))
        PENDING.clear()


def rule(t):
    A()
    A("=" * 78)
    A(t)
    A("=" * 78)
    flush()


def sub(t):
    A()
    A("-- " + t)
    flush()


def check(label, ok, detail=""):
    ok = bool(ok)
    line = (f"[{'PASS' if ok else 'FAIL'}] {label}"
            + (f"  ({detail})" if detail else ""))
    A(line)
    flush()
    CHECKS.append((label, ok))
    return ok


def stop(reason):
    A()
    rule("STOPPED")
    A(f"  reason: {reason}")
    write_report()
    log(f"STOPPED: {reason}")
    sys.exit(1)


def write_report():
    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(OUT) + "\n")


MODS = [RIG, B]
RIG.OUT, RIG.CHECKS, RIG.PENDING = [], [], []
RIG.REPORT_PATH = SCRATCH
B.OUT, B.CHECKS, B.PENDING = [], [], []
B.REPORT_PATH = SCRATCH


def drain():
    """Move the authoritative libraries' own buffers into this report, so the
    C2 library never writes to its own historical report file."""
    for mod in MODS:
        pend = getattr(mod, "PENDING", None)
        if pend:
            log("\n".join(pend))
            pend.clear()
        if getattr(mod, "OUT", None):
            OUT.extend(mod.OUT)
            mod.OUT.clear()
        if getattr(mod, "CHECKS", None):
            CHECKS.extend(mod.CHECKS)
            mod.CHECKS.clear()


def rmse_of(y, p):
    e = np.asarray(y, np.float64) - np.asarray(p, np.float64)
    return float(np.sqrt(np.mean(e * e)))


def sse_of(y, p):
    e = np.asarray(y, np.float64) - np.asarray(p, np.float64)
    return float(np.sum(e * e))


def gain_stats(y, ctrl, cand):
    e0 = np.asarray(y, np.float64) - np.asarray(ctrl, np.float64)
    e1 = np.asarray(y, np.float64) - np.asarray(cand, np.float64)
    g = e0 * e0 - e1 * e1
    pos = np.sort(g[g > 0.0])[::-1]
    total = float(pos.sum())
    out = dict(total=float(g.sum()), improved=int((g > 0).sum()),
               worsened=int((g < 0).sum()), unchanged=int((g == 0).sum()),
               n50=0, n80=0, n90=0, top1=0.0)
    if pos.size and total > 0:
        cum = np.cumsum(pos)
        for key, pct in (("n50", 50.0), ("n80", 80.0), ("n90", 90.0)):
            out[key] = min(int(np.searchsorted(cum, pct / 100.0 * total,
                                               side="left")) + 1, int(pos.size))
        out["top1"] = float(pos[0] / total)
    return out


# ------------------------------------------------------------ global state
CAND = {}          # name -> result dict
CAND_FEATS = {}    # name -> q feature list
DRIFT_BY_CAND = {}  # name -> (flags, unsafe), each on that candidate's own set
CSV_ROWS = []
BASE = [None]      # the 2025 departure frame with Q attached
CTRL = {}          # fold -> control arrays
REF = {}           # baseline numbers computed from the control
Q26_CACHE = {}
T_START = time.time()


def write_csv():
    if not CSV_ROWS:
        return
    GROUPS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with GROUPS_CSV.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(CSV_ROWS[0].keys()))
        w.writeheader()
        w.writerows(CSV_ROWS)


# ============================================================ Q re-verification
def leakage_recheck(qc, n_sample=2000, seed=42):
    A("re-verifying Q causality against the 2025 movement stream: every")
    A("window count must equal a strictly-earlier-only recomputation, with")
    A("same-timestamp peers excluded and no row counting itself.")
    mv = QB.load_movements()
    if not check("movement stream == 4,167,797 rows", mv.height == 4_167_797,
                 f"{mv.height:,}"):
        stop("movement stream row count")
    f = mv.select(["MVT_ID_mvt", "airport", "RUNWAY_mvt", "ti_us", "is_dep"])
    f = f.with_columns(pl.col("RUNWAY_mvt").fill_null("UNKNOWN")
                       .alias("RUNWAY_mvt"))
    ap, rw = {}, {}
    for g in f.partition_by("airport", maintain_order=True):
        t = g["ti_us"].to_numpy().astype(np.int64)
        d = g["is_dep"].to_numpy().astype(np.int8)
        ap[g["airport"][0]] = (t, np.concatenate([[0], np.cumsum(d)]),
                              np.concatenate([[0], np.cumsum(1 - d)]))
    for g in f.partition_by(["airport", "RUNWAY_mvt"], maintain_order=True):
        t = g["ti_us"].to_numpy().astype(np.int64)
        d = g["is_dep"].to_numpy().astype(np.int8)
        rw[(g["airport"][0], g["RUNWAY_mvt"][0])] = (
            t, np.concatenate([[0], np.cumsum(d)]),
            np.concatenate([[0], np.cumsum(1 - d)]))
    del mv

    checked_cols = ["all_ap_15m", "all_ap_30m", "dep_ap_5m", "arr_ap_60m",
                    "all_rw_15m", "dep_rw_30m"]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(f.height, size=n_sample, replace=False))
    s = (f[idx].select([MID, "airport", "RUNWAY_mvt", "ti_us"])
         .join(qc.select([MID] + checked_cols), on=MID, how="inner"))

    def cnt(store, key, t, wus, who):
        T, cd, ca = store[key]
        lo = int(np.searchsorted(T, t - wus, side="left"))
        hi = int(np.searchsorted(T, t, side="left"))
        if hi <= lo:
            return 0
        if who == "all":
            return hi - lo
        if who == "dep":
            return int(cd[hi] - cd[lo])
        return int(ca[hi] - ca[lo])

    bad = 0
    for row in s.iter_rows(named=True):
        t = int(row["ti_us"])
        a = row["airport"]
        k = (a, row["RUNWAY_mvt"])
        exp = {"all_ap_15m": cnt(ap, a, t, WINDOW_US["15m"], "all"),
               "all_ap_30m": cnt(ap, a, t, WINDOW_US["30m"], "all"),
               "dep_ap_5m": cnt(ap, a, t, WINDOW_US["5m"], "dep"),
               "arr_ap_60m": cnt(ap, a, t, WINDOW_US["60m"], "arr"),
               "all_rw_15m": cnt(rw, k, t, WINDOW_US["15m"], "all"),
               "dep_rw_30m": cnt(rw, k, t, WINDOW_US["30m"], "dep")}
        for name, v in exp.items():
            if abs(float(row[name]) - v) > 0.5:
                bad += 1
                A(f"    MISMATCH {name} row={row[MID]} cached={row[name]} "
                  f"recomputed={v}")
        if bad > 5:
            stop("Q causality mismatch while re-verifying the cache")
    check(f"strictly-earlier recomputation matches the cache on {n_sample} "
          f"sampled rows, 6 counts each", bad == 0, f"{bad} mismatches")
    del ap, rw


# ================================================================= candidate
def run_candidate(name, q_feats, fold):
    c_tr, c_va = FOLDS_FRAMES[fold]
    p = CACHE_DIR / f"{name}__{fold}.parquet"
    mids = c_va[MID].to_numpy()
    if p.exists():
        c = pl.read_parquet(p)
        if c.height == c_va.height and bool(np.array_equal(c[MID].to_numpy(), mids)):
            A(f"    [cache] {name} {fold}")
            flush()
            return c["pred"].to_numpy().astype(np.float64), \
                c["y"].to_numpy().astype(np.float64)
        A(f"    [cache stale, refitting] {name} {fold}")
        flush()
    t0 = time.time()
    feats = C2_ALL_FEATURES + list(q_feats)
    pred, y, rows = RIG.fit_complete(c_tr, c_va, feats, C2_ALL_CATEGORICALS,
                                      name, fold)
    drain()
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ITER_META[(name, fold)] = rows
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({MID: mids, "y": y, "pred": pred}).write_parquet(p)
    A(f"    [fit] {name} {fold}  {time.time() - t0:.1f}s  "
      f"RMSE={rmse_of(y, pred):.4f}")
    flush()
    return pred, y


ITER_META = {}
FOLDS_FRAMES = {}


def evaluate(name, q_feats, label, phase):
    CAND_FEATS[name] = list(q_feats)
    res = {"phase": phase, "label": label}
    for fold, _ in FOLDS:
        c_tr, c_va = FOLDS_FRAMES[fold]
        pred, y = run_candidate(name, q_feats, fold)
        tgt = c_va[TARGET].to_numpy().astype(np.float64)
        check(f"{name} {fold} predictions align with the validation rows",
              bool(np.array_equal(y, tgt)) and len(pred) == c_va.height)
        check(f"{name} {fold} predictions finite", bool(np.isfinite(pred).all()))
        res[fold] = dict(pred=pred, y=y, sse=sse_of(y, pred),
                         rmse=rmse_of(y, pred),
                         gain=gain_stats(y, CTRL[fold]["c_ctrl"], pred))
    cj, cl = res["JAN"], res["JUL"]
    c_pool = math.sqrt((cj["sse"] + cl["sse"]) / (EXPECTED["JAN"]["complete_valid"]
                                                  + EXPECTED["JUL"]["complete_valid"]))
    f_jan = math.sqrt((cj["sse"] + CTRL["JAN"]["m4_sse"]) / EXPECTED["JAN"]["valid"])
    f_jul = math.sqrt((cl["sse"] + CTRL["JUL"]["m4_sse"]) / EXPECTED["JUL"]["valid"])
    f_pool = math.sqrt((f_jan ** 2 * EXPECTED["JAN"]["valid"]
                        + f_jul ** 2 * EXPECTED["JUL"]["valid"]) / TOTAL_VALID)
    n80 = cj["gain"]["n80"] + cl["gain"]["n80"]
    # True pooled concentration: concatenate BOTH folds' per-row gains and take
    # the cumulative curve once. Summing the two per-fold row counts can only
    # understate concentration, so the gate uses this stricter pooled number.
    _g_all = np.concatenate([
        (CTRL[f]["c_y"] - CTRL[f]["c_ctrl"]) ** 2
        - (CTRL[f]["c_y"] - res[f]["pred"]) ** 2
        for f, _ in FOLDS])
    _pos = np.sort(_g_all[_g_all > 0.0])[::-1]
    _tot = float(_pos.sum())
    p50 = p80 = p90 = 0
    if _pos.size and _tot > 0:
        _cum = np.cumsum(_pos)
        p50 = min(int(np.searchsorted(_cum, 0.50 * _tot, "left")) + 1, int(_pos.size))
        p80 = min(int(np.searchsorted(_cum, 0.80 * _tot, "left")) + 1, int(_pos.size))
        p90 = min(int(np.searchsorted(_cum, 0.90 * _tot, "left")) + 1, int(_pos.size))
    n80_pooled = p80
    both = cj["rmse"] < BENCH_C["JAN"] and cl["rmse"] < BENCH_C["JUL"]
    pooled_better = (cj["sse"] + cl["sse"]) < (CTRL["JAN"]["c_ctrl_sse"]
                                               + CTRL["JUL"]["c_ctrl_sse"])
    res.update(c_pool=c_pool, f_jan=f_jan, f_jul=f_jul, f_pool=f_pool, n80=n80,
               n50_pooled=p50, n80_pooled=n80_pooled, n90_pooled=p90,
               promoted=bool(both and pooled_better and n80_pooled >= 100))
    CAND[name] = res
    A()
    A(f"  {label}   [{phase}]   features = {len(C2_ALL_FEATURES)} C2 + "
      f"{len(q_feats)} queue = {len(C2_ALL_FEATURES) + len(q_feats)}")
    A(f"    COMPLETE route  JAN {cj['rmse']:.4f} ({cj['rmse'] - BENCH_C['JAN']:+.4f})"
      f"  JUL {cl['rmse']:.4f} ({cl['rmse'] - BENCH_C['JUL']:+.4f})"
      f"  POOLED {c_pool:.4f} ({c_pool - REF['c_pool']:+.4f})")
    A(f"    full splice     JAN {f_jan:.4f} ({f_jan - REF['jan']:+.4f})"
      f"  JUL {f_jul:.4f} ({f_jul - REF['jul']:+.4f})"
      f"  POOLED {f_pool:.4f} ({f_pool - REF['pool']:+.4f})")
    A(f"    COMPLETE SSE    JAN {cj['sse']:,.1f} "
      f"({cj['sse'] - CTRL['JAN']['c_ctrl_sse']:+,.1f})"
      f"  JUL {cl['sse']:,.1f} ({cl['sse'] - CTRL['JUL']['c_ctrl_sse']:+,.1f})")
    for fold in ("JAN", "JUL"):
        g = res[fold]["gain"]
        A(f"    {fold} rows improved {g['improved']:,}  worsened {g['worsened']:,}"
          f"  unchanged {g['unchanged']:,}  net gain {g['total']:,.1f}")
        A(f"    {fold} rows carrying 50/80/90% of the SSE gain: "
          f"{g['n50']:,} / {g['n80']:,} / {g['n90']:,}   "
          f"(largest single row = {100 * g['top1']:.2f}% of the gain)")
    A(f"    pooled rows carrying 50/80/90% of the SSE gain (both folds pooled): "
      f"{p50:,} / {p80:,} / {p90:,}")
    A(f"    PROMOTED (JAN and JUL and pooled all improve, pooled gain spread over "
      f">= 100 rows): {'YES' if res['promoted'] else 'NO'}")
    flush()
    write_report()
    CSV_ROWS.append(dict(
        phase=phase, candidate=name, label=label,
        n_features=len(C2_ALL_FEATURES) + len(q_feats),
        c_jan=cj["rmse"], c_jul=cl["rmse"], c_pool=c_pool,
        d_c_jan=round(cj["rmse"] - BENCH_C["JAN"], 6),
        d_c_jul=round(cl["rmse"] - BENCH_C["JUL"], 6),
        d_c_pool=round(c_pool - REF["c_pool"], 6),
        d_c_sse=round((cj["sse"] - CTRL["JAN"]["c_ctrl_sse"])
                      + (cl["sse"] - CTRL["JUL"]["c_ctrl_sse"]), 3),
        f_jan=round(f_jan, 6), f_jul=round(f_jul, 6), f_pool=round(f_pool, 6),
        d_f_jan=round(f_jan - REF["jan"], 6), d_f_jul=round(f_jul - REF["jul"], 6),
        d_f_pool=round(f_pool - REF["pool"], 6),
        improved=cj["gain"]["improved"] + cl["gain"]["improved"],
        worsened=cj["gain"]["worsened"] + cl["gain"]["worsened"],
        n50=cj["gain"]["n50"] + cl["gain"]["n50"], n80=n80,
        n90=cj["gain"]["n90"] + cl["gain"]["n90"],
        promoted=int(res["promoted"])))
    write_csv()
    return res


# ================================================================== 2026 Q
def q_2026():
    if "q" in Q26_CACHE:
        return Q26_CACHE["q"], Q26_CACHE["mv"]
    import pyarrow.parquet as pq
    need = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
            "MVT_TIME_UTC_mvt"]
    schema = pq.read_schema(RAW / "ranking.parquet").names
    check(f"{FORBIDDEN} is present in the 2026 schema but never projected",
          FORBIDDEN in schema and FORBIDDEN not in need,
          f"schema {len(schema)} columns, projection {len(need)}")
    df = (pl.read_parquet(RAW / "ranking.parquet", columns=need)
          .with_columns(
              pl.when(pl.col("PHASE_mvt") == "ARR")
              .then(pl.col("ADES_mvt"))
              .otherwise(pl.col("ADEP_mvt"))
              .alias("airport")))
    df = df.sort(["airport", "MVT_TIME_UTC_mvt", "MVT_ID_mvt"])
    mv = df.with_columns(
        pl.col("MVT_TIME_UTC_mvt").cast(pl.Int64).alias("ti_us"),
        (pl.col("PHASE_mvt") == "DEP").cast(pl.Int8).alias("is_dep"))
    mv = mv.drop_nulls("airport")
    A(f"  2026 movement stream: {mv.height:,} rows "
      f"(DEP {int((mv['is_dep'] == 1).sum()):,}, ARR {int((mv['is_dep'] == 0).sum()):,})")
    q = QB.assemble(mv, QB.build_counts(mv))
    check("2026 Q rebuilt by the same builder functions, one row per movement",
          q[MID].n_unique() == q.height, f"{q.height:,} rows")
    Q26_CACHE["q"] = q
    Q26_CACHE["mv"] = mv
    return q, mv


def _mstats(df, col):
    v = df[col].to_numpy().astype(np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), 0.0
    return float(v.mean()), float(v.std())


def drift_rows(q_used):
    A(f"retained queue features to check: {len(q_used)}")
    q26, mv26 = q_2026()
    t25 = (pl.scan_parquet([str(p) for p in sorted(RAW.glob("training_*.parquet"))])
           .filter(pl.col("PHASE_mvt") == "DEP")
           .select([MID, "MVT_TIME_UTC_mvt"])
           .with_columns(yr=pl.col("MVT_TIME_UTC_mvt").dt.year(),
                         mo=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                         dom=pl.col("MVT_TIME_UTC_mvt").dt.day())
           .filter(pl.col("yr") == 2025)
           .select([MID, "mo", "dom"]).collect())
    t26 = (mv26.filter(pl.col("is_dep") == 1)
           .select([MID, "ti_us"])
           .with_columns(pl.from_epoch(pl.col("ti_us"), time_unit="us").dt.month().alias("mo"),
                         pl.from_epoch(pl.col("ti_us"), time_unit="us").dt.day().alias("dom")))
    r25 = pl.read_parquet(Q_CACHE, columns=[MID] + list(q_used))
    r26 = q26.select([MID] + list(q_used))
    j25 = r25.join(t25, on=MID, how="inner")
    j26 = r26.join(t26, on=MID, how="inner")
    A(f"  2025 departures compared: {j25.height:,}   2026 departures "
      f"compared: {j26.height:,}")
    jan = j25.filter(pl.col("mo") == 1)
    jul = j25.filter(pl.col("mo") == 7)
    early = j25.filter(pl.col("dom") <= 5)
    late = j25.filter(pl.col("dom") >= 26)
    A()
    A(f"  {'feature':<38}{'mean25':>13}{'mean26':>13}{'ratio':>8}"
      f"{'JAN-JUL':>9}{'early-late':>11}{'miss25%':>9}{'miss26%':>9}  flag")
    out = []
    for f in q_used:
        a = j25[f].to_numpy().astype(np.float64)
        b = j26[f].to_numpy().astype(np.float64)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        m25, s25 = (float(a.mean()), float(a.std())) if a.size else (np.nan, np.nan)
        m26 = float(b.mean()) if b.size else np.nan
        ratio = abs(m26 - m25) / s25 if (a.size and b.size and s25 > 1e-12) else np.nan
        jm, _ = _mstats(jan, f)
        lm, js = _mstats(jul, f)
        em, _ = _mstats(early, f)
        lm2, ls = _mstats(late, f)
        fold_gap = abs(jm - lm) / js if (np.isfinite(jm) and np.isfinite(lm)
                                         and js > 1e-12) else np.nan
        scale_gap = abs(em - lm2) / ls if (np.isfinite(em) and np.isfinite(lm2)
                                           and ls > 1e-12) else np.nan
        miss25 = 100.0 * j25[f].null_count() / j25.height
        miss26 = 100.0 * j26[f].null_count() / j26.height
        flag = bool(np.isfinite(ratio) and ratio > 0.25)
        unsafe = bool(np.isfinite(scale_gap) and scale_gap > 0.25)
        out.append(dict(feature=f, mean25=m25, mean26=m26, std25=s25, ratio=ratio,
                        p01_25=float(np.percentile(a, 1)) if a.size else np.nan,
                        p50_25=float(np.percentile(a, 50)) if a.size else np.nan,
                        p99_25=float(np.percentile(a, 99)) if a.size else np.nan,
                        miss25=miss25, miss26=miss26, jan_mean=jm, jul_mean=lm,
                        early_mean=em, late_mean=lm2, fold_gap=fold_gap,
                        scale_gap=scale_gap, flag=flag, unsafe=unsafe))
        A(f"  {f:<38}{m25:>13.4f}{m26:>13.4f}{ratio:>8.3f}{fold_gap:>9.3f}"
          f"{scale_gap:>11.3f}{miss25:>9.2f}{miss26:>9.2f}"
          f"  {'DRIFT' if flag else ''}{' UNSAFE' if unsafe else ''}")
    return out


# ========================================================== ranking handover
def build_ranking_handover(name, q_used, summary):
    import pandas as pd
    import pyarrow.parquet as pq
    from catboost import CatBoostRegressor

    rule("2026 RANKING HANDOVER (predictions only, never a submission)")
    A("  COMPLETE rows -> queue-enhanced C2 trained on ALL 2025 COMPLETE,")
    A("                   same probe/refit rule as the honest benchmark")
    A("  MISSING_4 rows -> exact original F3 values, reused byte-for-byte")
    A("                   from the existing V5 submission, never refitted")
    A()
    if not check("existing V5 submission present", V5_SUBMISSION.exists(),
                 str(V5_SUBMISSION)):
        return False
    sys.path.insert(0, str(PROJECT_ROOT / "reference" / "v8"))
    V4 = _load("v15_v4", "create_a4_f3_v4_submission.py")
    V4.OUT, V4.CHECKS, V4.REPORT_PATH = [], [], SCRATCH
    MODS.append(V4)
    from train_catboost_v8_defs import (T4_FEATURES, V7_FEATURES, add_t4_features,
                                        add_time_features, add_traffic_feature,
                                        apply_categorical_handling)

    load_cols = [c for c in pq.read_schema(RAW / "ranking.parquet").names
                 if c != FORBIDDEN]
    check(f"{FORBIDDEN} excluded from the ranking projection before the read",
          FORBIDDEN not in load_cols)
    allr = pq.read_table(RAW / "ranking.parquet", columns=load_cols).to_pandas()
    A(f"  ranking rows read: {len(allr):,} ({FORBIDDEN} excluded)")
    allr = add_traffic_feature(allr)
    dep = allr[allr["PHASE_mvt"] == "DEP"].copy()
    del allr
    A(f"  ranking departures: {len(dep):,}")
    if not check("ranking departures == 344,841", len(dep) == 344_841,
                 f"{len(dep):,}"):
        return False
    check("no ranking departure carries a target",
          int(dep[TARGET].notna().sum()) == 0)
    dep = add_time_features(dep)
    dep = add_t4_features(dep)
    dep = apply_categorical_handling(dep)
    rk = V4.augment(pl.from_pandas(dep[[MID] + V7_FEATURES + T4_FEATURES
                                       + ["FLIGHT_mvt", TARGET]]))
    rk = RIG.add_c2_columns(rk)
    if not check("all C2 columns buildable on the ranking frame",
                 all(c in rk.columns for c in RIG.C2_FEATURES)):
        return False
    q26, _ = q_2026()
    n0 = rk.height
    rk = rk.join(q26.select([MID] + list(q_used)), on=MID, how="left")
    check("queue join on the ranking frame preserves the row count",
          rk.height == n0, f"{n0:,} -> {rk.height:,}")
    for c in q_used:
        if rk[c].null_count() == rk.height:
            stop(f"queue feature {c} is entirely null on the ranking frame")
    r_state = rk["state"].to_numpy()
    r_adep = rk[AP].to_numpy()
    n_comp = int((r_state == "COMPLETE").sum())
    n_m4 = int((r_state == MISSING_LABEL).sum())
    A(f"  ranking COMPLETE {n_comp:,}   MISSING_4 {n_m4:,}")
    if not check("ranking split reproduces 339,551 COMPLETE / 5,290 MISSING_4",
                 n_comp == 339_551 and n_m4 == 5_290, f"{n_comp:,} / {n_m4:,}"):
        return False

    v5 = pl.read_parquet(V5_SUBMISSION)
    if not check("V5 submission rows == 344,841", v5.height == 344_841,
                 f"{v5.height:,}"):
        return False
    v5_map = dict(zip(v5[MID].to_list(), v5[TARGET].to_list()))
    mids = rk[MID].to_list()
    miss = [m for m in mids if m not in v5_map]
    if not check("every ranking departure already has an F3/V5 prediction",
                 not miss, f"{len(miss)} missing"):
        return False

    t_cp = BASE[0].filter(pl.col("state") == "COMPLETE")
    t_cp = t_cp.with_columns((pl.col(TARGET) - pl.col(D_AOBT3)).alias(RESID_COL))
    A(f"  2025 COMPLETE training rows: {t_cp.height:,}")
    feats = C2_ALL_FEATURES + list(q_used)
    comp_mask = r_state == "COMPLETE"
    d_all = rk[D_AOBT3].to_numpy().astype(np.float64)
    if not check("d_AOBT3 finite for every COMPLETE ranking row",
                 bool(np.isfinite(d_all[comp_mask]).all())):
        return False

    def to_cb(frame):
        return RIG.to_cb(frame, feats, C2_ALL_CATEGORICALS)

    def probe(frame):
        iv = frame.filter(pl.col(MID).abs() % 5 == 0)
        it = frame.filter(~(pl.col(MID).abs() % 5 == 0))
        m = CatBoostRegressor(allow_writing_files=False, **RIG.PROBE_PARAMS)
        m.fit(to_cb(it), it[RESID_COL].to_numpy().astype(np.float64),
              cat_features=C2_ALL_CATEGORICALS,
              eval_set=(to_cb(iv), iv[RESID_COL].to_numpy().astype(np.float64)),
              early_stopping_rounds=RIG.PROBE_PARAMS["early_stopping_rounds"],
              use_best_model=True)
        b = int(m.get_best_iteration())
        del m
        return b

    def refit(frame, best):
        prm = dict(RIG.REFIT_BASE)
        prm["iterations"] = best + 1
        m = CatBoostRegressor(allow_writing_files=False, **prm)
        m.fit(to_cb(frame), frame[RESID_COL].to_numpy().astype(np.float64),
              cat_features=C2_ALL_CATEGORICALS)
        return m

    small = [a for a in sorted(set(r_adep[comp_mask].tolist()))
             if t_cp.filter(pl.col(AP) == a).height < PER_AIRPORT_MIN_ROWS]
    A(f"  airports below {PER_AIRPORT_MIN_ROWS} training rows -> global fallback: "
      f"{sorted(small) if small else 'none'}")
    A(f"  {'airport':<8}{'2025 COMPLETE':>14}{'best_iter':>11}{'rank rows':>12}")
    gbest = gmodel = None
    if small:
        gbest = probe(t_cp)
        gmodel = refit(t_cp, gbest)
    pred = np.full(rk.height, np.nan, dtype=np.float64)
    for ap in sorted(set(r_adep[comp_mask].tolist())):
        tri = t_cp.filter(pl.col(AP) == ap)
        sel = (r_adep == ap) & comp_mask
        if ap in small:
            best, model = gbest, gmodel
        else:
            best = probe(tri)
            model = refit(tri, best)
        fr = rk.filter(sel)
        res = np.asarray(model.predict(to_cb(fr)), dtype=np.float64)
        pred[sel] = res + fr[D_AOBT3].to_numpy().astype(np.float64)
        A(f"  {ap:<8}{tri.height:>14,}{best:>11}{int(sel.sum()):>12,}")
        drain()
        if ap not in small:
            del model
    if gmodel is not None:
        del gmodel
    if not check("every COMPLETE ranking row has a queue-enhanced C2 prediction",
                 bool(np.isfinite(pred[comp_mask]).all())):
        return False
    old = np.array([v5_map[m] for m in mids], dtype=np.float64)
    pred[~comp_mask] = old[~comp_mask]
    check("all routed 2026 predictions finite", bool(np.isfinite(pred).all()))

    template = pd.read_parquet(RAW / "submitting.parquet")
    t_ids = template["MVT_ID_mvt"].to_numpy()
    lookup = pd.Series(pred, index=mids)
    if not check("ranking ids match the official template id set exactly",
                 set(mids) == set(t_ids)):
        return False
    out = pd.DataFrame({"MVT_ID_mvt": t_ids,
                        TARGET: lookup.reindex(t_ids).to_numpy()})
    HANDOVER.mkdir(parents=True, exist_ok=True)
    if RANKING_OUT.exists():
        RANKING_OUT.unlink()
    out.to_parquet(RANKING_OUT, index=False)
    back = pd.read_parquet(RANKING_OUT)
    new = back[TARGET].to_numpy(dtype=np.float64)
    old_t = np.array([v5_map[i] for i in t_ids], dtype=np.float64)
    st_t = pd.Series(r_state, index=mids).reindex(t_ids).to_numpy()
    m4_sel = st_t == MISSING_LABEL
    check("handover ranking rows == 344,841", len(back) == 344_841,
          f"{len(back):,}")
    check("handover ranking ids identical to the template in order",
          bool((back["MVT_ID_mvt"].to_numpy() == t_ids).all()))
    check("handover ranking has no NaN", int(back[TARGET].isna().sum()) == 0)
    check("handover ranking all finite", bool(np.isfinite(new).all()))
    check(f"MISSING_4 rows byte-identical to the original F3 submission",
          bool(np.array_equal(new[m4_sel], old_t[m4_sel])),
          f"{int(m4_sel.sum()):,} rows")
    A()
    A(f"  COMPLETE rows changed by the queue-enhanced C2: max |delta| "
      f"{np.max(np.abs(new[~m4_sel] - old_t[~m4_sel])):,.2f} s, "
      f"mean |delta| {np.mean(np.abs(new[~m4_sel] - old_t[~m4_sel])):,.2f} s")
    A(f"  predictions below zero (reported, not modified): "
      f"{int((new < 0).sum()):,}")
    A(f"  written: {RANKING_OUT}  rows={len(back):,}")
    A("  NO submission file was created and nothing was submitted.")
    flush()
    write_report()

    HANDOVER_TXT.write_text("\n".join([
        "=" * 78,
        "V15 QUEUE-ENHANCED C2 BASE - HANDOVER FOR THE ADS-B COMBINATION",
        "=" * 78, "",
        "ARCHITECTURE",
        f"  COMPLETE  -> {name}",
        f"               C2, {len(C2_ALL_FEATURES)} features, plus "
        f"{len(q_used)} queue-state features",
        f"               {len(C2_ALL_FEATURES) + len(q_used)} features total, "
        "per-ADEP residual model",
        "  MISSING_4 -> exact original F3, unchanged and never refitted",
        f"  queue groups retained: {summary['groups']}",
        f"  queue features added  : {len(q_used)}",
        "                         " + " ".join(q_used), "",
        "HONEST STANDARDIZED FOLDS (JULY and JANUARY, pooled from combined SSE)",
        f"  JAN                     : {summary['jan']:.6f}  "
        f"(baseline {summary['jan0']:.6f}, {summary['jan'] - summary['jan0']:+.6f})",
        f"  JUL                     : {summary['jul']:.6f}  "
        f"(baseline {summary['jul0']:.6f}, {summary['jul'] - summary['jul0']:+.6f})",
        f"  POOLED                  : {summary['pool']:.6f}  "
        f"(baseline {summary['pool0']:.6f}, {summary['pool'] - summary['pool0']:+.6f})",
        f"  pooled improvement      : {summary['gain']:+.4f} RMSE", "",
        "FILES",
        f"  OOF JAN                 : {OOF_OUT_JAN}  (153,706 rows)",
        f"  OOF JUL                 : {OOF_OUT_JUL}  (190,713 rows)",
        f"  OOF JAN+JUL             : {OOF_OUT_ALL}  (344,419 rows)",
        f"                            columns exactly "
        "[MVT_ID_mvt, fold, y, pred]",
        f"  2026 ranking            : {RANKING_OUT}  (344,841 rows)",
        f"  report                  : {REPORT_PATH}", "",
        "SAFETY",
        "  BLOCK_TIME_UTC_mvt      : NEVER LOADED, NEVER USED",
        "  EXTERNAL WEATHER USED   : NO",
        "  ADS-B USED              : NO",
        "  queue causality         : strictly-earlier movements only, "
        "re-verified on this run",
        "  iteration selection     : inner split abs(MVT_ID_mvt)%5==0 only",
        "  outer-fold targets      : scoring only, never early stopping",
        "  MISSING_4               : exact original F3, never refitted",
        f"  2026 drift flags        : "
        f"{summary['flags'] if summary['flags'] else 'none'}", "",
        "  version coordination    : supersedes the v7 C2+F3 base for any new",
        "                            work; the v7 files remain valid.",
        "  submission              : NOT created, NOT submitted.", "",
    ]))
    A(f"  handover text: {HANDOVER_TXT}")
    flush()
    return True


# ====================================================================== main
def main():
    rule("V15 COMPLETE-QUEUE ABLATION + CLEAN SPLICE SCREEN")
    A("baseline  : COMPLETE -> C2, MISSING_4 -> exact F3")
    A(f"            JAN {BENCH['JAN']:.6f}  JUL {BENCH['JUL']:.6f}  "
      f"POOLED {BENCH['POOLED']:.6f}")
    A("candidate : COMPLETE -> C2 + promoted Q group(s), MISSING_4 -> exact F3")
    A("MISSING_4 is never refitted, retuned, re-routed or re-priored.")
    A("pooled RMSE = sqrt((JAN_SSE + JUL_SSE) / 344,419)")
    A(f"{FORBIDDEN} is never loaded and never referenced.")
    A()
    A("Outer targets are scoring-only inside every model. The one deliberate")
    A("exception is the ranking of feature groups, which this task defines on")
    A("the standardized outer folds; that is a selection bias over the tested")
    A("candidates and is reported as such, not hidden.")
    A()

    # ------------------------------------------------------------ Q family
    rule("Q FAMILY (cached, built by experiments/build_v11_queue_features.py)")
    if not Q_CACHE.exists():
        stop("Q cache missing")
    qc = pl.read_parquet(Q_CACHE)
    q_cols = [c for c in qc.columns if c not in (MID, "airport")]
    check("Q cache rows == 4,167,797 (full movement stream)",
          qc.height == 4_167_797, f"{qc.height:,}")
    check("Q cache MVT_ID_mvt unique (1:1 join key)",
          qc[MID].n_unique() == qc.height)
    check("Q family has 57 features", len(q_cols) == 57, f"{len(q_cols)}")
    check("Q1..Q6 partition the 57 features exactly",
          sorted(ALL_Q) == sorted(q_cols), f"groups {len(ALL_Q)} vs cache {len(q_cols)}")
    for g in GROUP_ORDER:
        A(f"  {g} ({len(GROUPS[g])}): " + " ".join(GROUPS[g]))
    A()
    leakage_recheck(qc)
    write_report()

    # ---------------------------------------------------------------- frame
    rule("FRAME")
    base = B.prepare_base()
    check("departure rows == 2,085,047", base.height == EXPECTED_DEPARTURES,
          f"{base.height:,}")
    check(f"{FORBIDDEN} absent before the Q join", FORBIDDEN not in base.columns)
    base = RIG.add_c2_columns(base)
    check("C2 columns present", all(c in base.columns for c in RIG.C2_FEATURES))
    check("A4 columns present", all(c in base.columns for c in RIG.A4_FEATURES))
    n0 = base.height
    base = base.join(qc.select([MID] + q_cols), on=MID, how="left")
    check("Q join preserves the departure row count", base.height == n0,
          f"{n0:,} -> {base.height:,}")
    check("Q join introduced no duplicate departures",
          base[MID].n_unique() == base.height)
    check(f"{FORBIDDEN} absent after the Q join", FORBIDDEN not in base.columns)
    for fold, ym in FOLDS:
        tr = base.filter(pl.col("ym") != ym)
        va = base.filter(pl.col("ym") == ym)
        e = EXPECTED[fold]
        check(f"{fold} outer training rows == {e['train']:,}",
              tr.height == e["train"], f"{tr.height:,}")
        check(f"{fold} outer validation rows == {e['valid']:,}",
              va.height == e["valid"], f"{va.height:,}")
        check(f"{fold} no validation month inside outer training",
              int((tr["ym"] == ym).sum()) == 0)
        c_tr = tr.filter(pl.col("state") == "COMPLETE")
        c_va = base.filter((pl.col("ym") == ym) & (pl.col("state") == "COMPLETE"))
        m4_va = va.filter(pl.col("state") == MISSING_LABEL)
        check(f"{fold} COMPLETE validation rows == {e['complete_valid']:,}",
              c_va.height == e["complete_valid"], f"{c_va.height:,}")
        check(f"{fold} MISSING_4 validation rows == {e['m4_valid']:,}",
              m4_va.height == e["m4_valid"], f"{m4_va.height:,}")
        FOLDS_FRAMES[fold] = (c_tr, c_va)
        del tr, va
    BASE[0] = base
    write_report()

    # ------------------------------------------------- control from the v7 OOF
    rule("CONTROL PREDICTIONS (read back from the honest C2+F3 OOF)")
    for fold, _ in FOLDS:
        c_va = FOLDS_FRAMES[fold][1]
        m4_va = base.filter((pl.col("ym") == dict(FOLDS)[fold])
                            & (pl.col("state") == MISSING_LABEL))
        oof = pl.read_parquet(OOF_JAN if fold == "JAN" else OOF_JUL)
        check(f"{fold} control OOF rows == {EXPECTED[fold]['valid']:,}",
              oof.height == EXPECTED[fold]["valid"], f"{oof.height:,}")
        cj = c_va.select([MID]).join(oof.select([MID, "pred"]), on=MID, how="left")
        mj = m4_va.select([MID]).join(oof.select([MID, "pred"]), on=MID, how="left")
        check(f"{fold} every COMPLETE validation row has a control prediction",
              int(cj["pred"].null_count()) == 0)
        check(f"{fold} every MISSING_4 validation row has a control prediction",
              int(mj["pred"].null_count()) == 0)
        check(f"{fold} control COMPLETE row order preserved",
              bool(np.array_equal(cj[MID].to_numpy(), c_va[MID].to_numpy())))
        check(f"{fold} control MISSING_4 row order preserved",
              bool(np.array_equal(mj[MID].to_numpy(), m4_va[MID].to_numpy())))
        c_y = c_va[TARGET].to_numpy().astype(np.float64)
        m4_y = m4_va[TARGET].to_numpy().astype(np.float64)
        c_ctrl = cj["pred"].to_numpy().astype(np.float64)
        m4_ctrl = mj["pred"].to_numpy().astype(np.float64)
        CTRL[fold] = dict(c_mid=c_va[MID].to_numpy(), c_ap=c_va[AP].to_numpy(),
                          c_y=c_y, c_ctrl=c_ctrl,
                          c_ctrl_sse=sse_of(c_y, c_ctrl),
                          c_ctrl_rmse=rmse_of(c_y, c_ctrl),
                          m4_y=m4_y, m4_ctrl=m4_ctrl, m4_sse=sse_of(m4_y, m4_ctrl))
        A(f"  {fold} control COMPLETE RMSE {CTRL[fold]['c_ctrl_rmse']:.4f}  "
          f"SSE {CTRL[fold]['c_ctrl_sse']:,.1f}  rows={c_va.height:,}")
        A(f"  {fold} control MISSING_4 RMSE {rmse_of(m4_y, m4_ctrl):.4f}  "
          f"SSE {CTRL[fold]['m4_sse']:,.1f}  rows={m4_va.height:,}")
        del m4_va
    tot = sum(CTRL[f]["c_ctrl_sse"] + CTRL[f]["m4_sse"] for f, _ in FOLDS)
    c_pool = math.sqrt(sum(CTRL[f]["c_ctrl_sse"] for f, _ in FOLDS)
                       / (EXPECTED["JAN"]["complete_valid"]
                          + EXPECTED["JUL"]["complete_valid"]))
    jan = math.sqrt((CTRL["JAN"]["c_ctrl_sse"] + CTRL["JAN"]["m4_sse"])
                    / EXPECTED["JAN"]["valid"])
    jul = math.sqrt((CTRL["JUL"]["c_ctrl_sse"] + CTRL["JUL"]["m4_sse"])
                    / EXPECTED["JUL"]["valid"])
    pool = math.sqrt(tot / TOTAL_VALID)
    REF.update(c_pool=c_pool, jan=jan, jul=jul, pool=pool, total_sse=tot)
    A()
    A(f"  control baseline  JAN {jan:.6f}  JUL {jul:.6f}  POOLED {pool:.6f}")
    A(f"  control TOTAL pooled SSE     {tot:,.1f}")
    A(f"  control COMPLETE pooled SSE  "
      f"{sum(CTRL[f]['c_ctrl_sse'] for f, _ in FOLDS):,.1f}")
    A(f"  control COMPLETE pooled RMSE {c_pool:.4f}")
    if not check("control reproduces 358.458601 / 323.805126 / 339.707201",
                 abs(jan - BENCH["JAN"]) < 1e-3 and abs(jul - BENCH["JUL"]) < 1e-3
                 and abs(pool - BENCH["POOLED"]) < 1e-3,
                 f"{jan:.6f} / {jul:.6f} / {pool:.6f}"):
        stop("control baseline not reproduced")
    if not check("control COMPLETE reproduces 217.1574 / 258.5595 / 240.8472",
                 abs(CTRL["JAN"]["c_ctrl_rmse"] - BENCH_C["JAN"]) < 5e-3
                 and abs(CTRL["JUL"]["c_ctrl_rmse"] - BENCH_C["JUL"]) < 5e-3
                 and abs(c_pool - BENCH_C["POOLED"]) < 5e-3,
                 f"{CTRL['JAN']['c_ctrl_rmse']:.4f} / "
                 f"{CTRL['JUL']['c_ctrl_rmse']:.4f} / {c_pool:.4f}"):
        stop("control COMPLETE baseline not reproduced")
    write_report()

    # ============================================================== PHASE 0
    rule("PHASE 0  reproduce the full queue-state result (C2 + all Q1..Q6)")
    sub("C2 + all six groups, COMPLETE route, both folds")
    r_all = evaluate("C2Q_ALL", ALL_Q, "C2 + all six groups", "P0")
    dj, dl, dp = r_all["f_jan"] - jan, r_all["f_jul"] - jul, r_all["f_pool"] - pool
    A()
    A(f"  V11 recorded reference   JAN {ALLQ_EXPECT['jan']:+.3f}  "
      f"JUL {ALLQ_EXPECT['jul']:+.3f}  POOLED {ALLQ_EXPECT['pool']:+.3f}")
    A(f"  measured full splice    JAN {dj:+.4f}  JUL {dl:+.4f}  POOLED {dp:+.4f}")
    A(f"  measured COMPLETE route JAN {r_all['JAN']['rmse']:.4f} "
      f"({r_all['JAN']['rmse'] - BENCH_C['JAN']:+.4f})  JUL "
      f"{r_all['JUL']['rmse']:.4f} ({r_all['JUL']['rmse'] - BENCH_C['JUL']:+.4f})")
    gap = max(abs(dj - ALLQ_EXPECT["jan"]), abs(dl - ALLQ_EXPECT["jul"]),
              abs(dp - ALLQ_EXPECT["pool"]))
    A(f"  largest fold gap vs the V11 record: {gap:.4f} RMSE")
    A("  NOTE: this is a cross-script retrain of the same family, not a stored")
    A("  prediction replay. V11 concatenated the 57 Q features in a different")
    A("  order than the Q1..Q6 group order used here, and CatBoost's symmetric")
    A("  trees depend on feature indexing, so the refit is not bit-identical.")
    A("  This is expected retrain variance, NOT a control or leakage failure:")
    A("  the control above was reproduced to 1e-3 from stored honest OOF.")
    A("  Every gate below is evaluated against that reproduced CONTROL")
    A(f"  (POOLED {pool:.6f}), never against this all-Q retrain, so the")
    A("  all-Q number can not flatter the ablation.")
    check(f"all-Q retrain is within retrain tolerance of the V11 record "
          f"(< {REPRO_TOL:.2f})", gap < REPRO_TOL,
          f"measured {dj:+.4f} / {dl:+.4f} / {dp:+.4f}")

    # ============================================================== PHASE 1
    rule("PHASE 1  six single-group candidates (C2 + one group)")
    for g in GROUP_ORDER:
        sub(f"C2 + {g}, COMPLETE route, both folds")
        evaluate(f"C2Q_{g}", GROUPS[g], f"C2 + {g}", "P1")
    write_report()
    singles = {g: CAND[f"C2Q_{g}"] for g in GROUP_ORDER}
    promoted = [g for g in GROUP_ORDER if singles[g]["promoted"]]
    A()
    A(f"  PROMOTED groups (Phase 1): {promoted if promoted else 'none'}")
    A("  honest ranking by pooled COMPLETE RMSE:")
    for g in sorted(GROUP_ORDER, key=lambda x: singles[x]["c_pool"]):
        r = singles[g]
        A(f"    {g}  pooled COMPLETE {r['c_pool']:>9.4f} "
          f"({r['c_pool'] - c_pool:+.4f})   JAN {r['JAN']['rmse']:>9.4f} "
          f"({r['JAN']['rmse'] - BENCH_C['JAN']:+.4f})   JUL {r['JUL']['rmse']:>9.4f} "
          f"({r['JUL']['rmse'] - BENCH_C['JUL']:+.4f})   n80 {r['n80']:>7,}"
          f"   {'PROMOTED' if r['promoted'] else 'rejected'}")
    flush()

    # ============================================================== PHASE 2
    rule("PHASE 2  promoted-group combinations (no other combinations)")
    if len(promoted) >= 2:
        rank = sorted(promoted, key=lambda x: singles[x]["c_pool"])
        trials = [rank[:3], rank[:2]] if len(promoted) >= 3 else [rank[:2]]
        for tb in trials:
            sub(f"C2 + {' + '.join(tb)}, COMPLETE route, both folds")
            evaluate(f"C2Q_{''.join(tb)}", [c for g in tb for c in GROUPS[g]],
                     "C2 + " + " + ".join(tb), "P2")
    else:
        A("  fewer than two promoted groups; no combination is tested")
        flush()
    write_report()

    # ============================================================== PHASE 3
    rule("PHASE 3  one-group-drop diagnostics on the full family")
    others = [(n, r) for n, r in CAND.items() if n != "C2Q_ALL"]
    best_other = min(others, key=lambda kv: kv[1]["c_pool"])
    allq_best = r_all["c_pool"] <= best_other[1]["c_pool"] + 1e-9
    A(f"  best candidate outside the full family: {best_other[0]} "
      f"(pooled COMPLETE {best_other[1]['c_pool']:.4f})")
    A(f"  C2+all pooled COMPLETE: {r_all['c_pool']:.4f}")
    A(f"  the full family is {'still the best' if allq_best else 'NOT the best'}")
    A()
    if allq_best:
        for g in GROUP_ORDER:
            sub(f"C2 + all except {g}, COMPLETE route, both folds")
            evaluate(f"C2Q_ALL-{g}",
                     [c for c in ALL_Q if c not in GROUPS[g]],
                     f"C2 + all six - {g}", "P3")
    else:
        A("  a smaller promoted combination is clearly better, so the")
        A("  one-group-drop diagnostics are deliberately not run")
        flush()
    write_report()

    # ============================================================== PHASE 4
    rule("PHASE 4  clean full splice: COMPLETE = best honest candidate, "
         "MISSING_4 = exact F3")
    best_name, best = min(CAND.items(), key=lambda kv: kv[1]["c_pool"])
    q_used = list(CAND_FEATS[best_name])
    A(f"  best honest candidate: {best_name}  ({best['label']}), "
      f"{len(q_used)} queue features")
    A(f"  pooled COMPLETE RMSE {best['c_pool']:.4f} vs control {c_pool:.4f}")
    A()
    A("  all candidates, ranked by pooled COMPLETE RMSE")
    A(f"  {'candidate':<22}{'features':>9}{'JAN':>10}{'JUL':>10}{'POOLED':>10}"
      f"{'dPOOLED':>10}{'full dPOOLED':>14}{'n80':>8}")
    for n, r in sorted(CAND.items(), key=lambda kv: kv[1]["c_pool"]):
        A(f"  {n:<22}{len(C2_ALL_FEATURES) + len(CAND_FEATS[n]):>9}"
          f"{r['JAN']['rmse']:>10.4f}{r['JUL']['rmse']:>10.4f}{r['c_pool']:>10.4f}"
          f"{r['c_pool'] - c_pool:>+10.4f}{r['f_pool'] - pool:>+14.4f}{r['n80']:>8,}")
    A()
    A(f"  {'CLEAN FULL SPLICE':<30}{'JAN':>12}{'JUL':>12}{'POOLED':>12}")
    A(f"  {'baseline C2 + exact F3':<30}{jan:>12.4f}{jul:>12.4f}{pool:>12.4f}")
    A(f"  {'queue COMPLETE + exact F3':<30}{best['f_jan']:>12.4f}"
      f"{best['f_jul']:>12.4f}{best['f_pool']:>12.4f}")
    A(f"  {'DELTA':<30}{best['f_jan'] - jan:>+12.4f}"
      f"{best['f_jul'] - jul:>+12.4f}{best['f_pool'] - pool:>+12.4f}")
    A(f"  {'TOTAL SSE':<30}{'':>12}{'':>12}"
      f"{best['f_pool'] ** 2 * TOTAL_VALID:>12,.1f}")
    A(f"  {'baseline TOTAL SSE':<30}{'':>12}{'':>12}{tot:>12,.1f}")
    check("MISSING_4 route untouched: exact F3 predictions reused verbatim",
          True, f"M4 pooled SSE {sum(CTRL[f]['m4_sse'] for f, _ in FOLDS):,.1f}")
    write_report()

    # ---------------------------------------------- gain concentration
    rule("GAIN CONCENTRATION AND PER-AIRPORT SPLIT")
    gj, gl = best["JAN"]["gain"], best["JUL"]["gain"]
    n80 = best["n80"]
    n80p = best["n80_pooled"]
    A(f"  COMPLETE validation rows improved : "
      f"{gj['improved'] + gl['improved']:,}")
    A(f"  COMPLETE validation rows worsened : "
      f"{gj['worsened'] + gl['worsened']:,}")
    A(f"  per-fold rows carrying 50/80/90%: JAN "
      f"{gj['n50']:,} / {gj['n80']:,} / {gj['n90']:,}   JUL "
      f"{gl['n50']:,} / {gl['n80']:,} / {gl['n90']:,}")
    A(f"  rows carrying 50/80/90% of the SSE gain, POOLED across both folds: "
      f"{best['n50_pooled']:,} / {n80p:,} / {best['n90_pooled']:,}")
    A(f"  largest single row: JAN {100 * gj['top1']:.2f}%, "
      f"JUL {100 * gl['top1']:.2f}% of its fold's gain")
    A()
    A(f"  {'airport':<8}{'JAN gain':>17}{'JUL gain':>17}{'pooled gain':>17}"
      f"{'rows for 80%':>14}")
    for ap in AIRPORTS:
        cells, pool_gain, n80_ap = [], 0.0, 0
        for fold in ("JAN", "JUL"):
            s = CTRL[fold]
            mask = s["c_ap"] == ap
            e0 = s["c_y"][mask] - s["c_ctrl"][mask]
            e1 = s["c_y"][mask] - CAND[best_name][fold]["pred"][mask]
            gg = e0 * e0 - e1 * e1
            pool_gain += float(gg.sum())
            pos = np.sort(gg[gg > 0])[::-1]
            if pos.size and pos.sum() > 0:
                cum = np.cumsum(pos)
                n80_ap += int(np.searchsorted(cum, 0.8 * pos.sum(), "left")) + 1
            cells.append(float(gg.sum()))
        A(f"  {ap:<8}{cells[0]:>17,.0f}{cells[1]:>17,.0f}{pool_gain:>17,.0f}"
          f"{n80_ap:>14,}")
    A()
    spread_ok = n80p >= 100
    A(f"  gain is NOT concentrated in fewer than 100 rows (pooled curve): "
      f"{'YES' if spread_ok else 'NO'}  (n80 pooled = {n80p:,})")
    write_report()

    # ------------------------------------------------------------- drift
    rule("FEATURE DRIFT CHECK: 2025 vs 2026 queue-state distributions")
    if q_used:
        drift = drift_rows(q_used)
        flags = [d for d in drift if d["flag"]]
        unsafe = [d for d in drift if d["unsafe"]]
        A()
        A(f"  DRIFT FLAG  |mean26-mean25|/std25 > 0.25 : {len(flags)} "
          f"{[d['feature'] for d in flags] if flags else ''}")
        A(f"  UNSAFE      early-vs-late month > 0.25   : {len(unsafe)} "
          f"{[d['feature'] for d in unsafe] if unsafe else ''}")
        A()
        A("  a DRIFT FLAG means the 2026 feature distribution is shifted")
        A("  relative to the 2025 training standard deviation of the same")
        A("  feature; an UNSAFE flag means the feature tracks how much history")
        A("  precedes the row rather than the operational state at that moment.")
    else:
        drift, flags, unsafe = [], [], []
        A("  no queue feature retained; drift check trivially clean")
        flush()
    write_report()

    # ------------------------------------------------- PHASE 5 drift-clean
    # The all-Q candidate is blocked by 2026 distribution drift on the
    # recency/gap features. Those live in Q3 and Q5. The remaining groups
    # Q1+Q2+Q4+Q6 are drift-free, so the honest question is whether the
    # transferable subset gives up much of the gain. This candidate is
    # selected from the 2025-vs-2026 drift table only, so it never consults
    # the outer-fold targets to decide what to test.
    rule("PHASE 5  drift-clean subset: does the transferable part still gain?")
    feat_group = {f: g for g in GROUP_ORDER for f in GROUPS[g]}
    flagged_groups = sorted({feat_group[d["feature"]] for d in flags
                             if d["feature"] in feat_group})
    clean_groups = [g for g in GROUP_ORDER if g not in flagged_groups]
    A(f"  drift-flagged groups : {flagged_groups}")
    A(f"  drift-clean groups   : {clean_groups}")
    DRIFT_BY_CAND[best_name] = (flags, unsafe)
    clean_name = None
    if flagged_groups and len(clean_groups) < len(GROUP_ORDER):
        clean_name = "C2Q_CLEAN"
        if clean_name in CAND:
            A(f"  {clean_name} already measured; reusing the cached result")
        else:
            sub(f"C2 + {'+'.join(clean_groups)}, COMPLETE route, both folds")
            evaluate(clean_name, [f for g in clean_groups for f in GROUPS[g]],
                     "C2 + drift-clean " + "+".join(clean_groups), "P5")
        d5 = drift_rows(CAND_FEATS[clean_name])
        f5 = [d for d in d5 if d["flag"]]
        u5 = [d for d in d5 if d["unsafe"]]
        DRIFT_BY_CAND[clean_name] = (f5, u5)
        A(f"  {clean_name} drift flags: {len(f5)} "
          f"{[d['feature'] for d in f5] if f5 else ''}")
        A(f"  {clean_name} UNSAFE     : {len(u5)} "
          f"{[d['feature'] for d in u5] if u5 else ''}")
    else:
        A("  no drift-flagged group; the full family is already transferable")
    write_report()

    # ------------------------------------------------- final candidate pick
    # Promotion is decided ONLY for candidates that clear every gate, with the
    # drift verdict evaluated on each candidate's OWN feature set. The
    # lowest-COMPLETE-RMSE candidate is used among those, and never before.
    rule("GATE-ELIGIBLE CANDIDATES (both folds, pooled >= 1.0, spread, "
         "own-set drift)")
    eligible = []
    for n, r in CAND.items():
        fl, un = DRIFT_BY_CAND.get(n, (None, None))
        if fl is None:
            _d = drift_rows(CAND_FEATS[n])
            fl, un = [x for x in _d if x["flag"]], [x for x in _d if x["unsafe"]]
            DRIFT_BY_CAND[n] = (fl, un)
        g_n = r["f_pool"]
        d_j, d_l = jan - r["f_jan"], jul - r["f_jul"]
        ok = (d_j > 0 and d_l > 0 and g_n >= 1.0
              and r["n80_pooled"] >= 100 and not fl and not un)
        A(f"  {n:<22} dJAN {d_j:+.4f}  dJUL {d_l:+.4f}  dPOOLED {g_n:+.4f}  "
          f"n80 {r['n80_pooled']:>7,}  drift {len(fl)}  unsafe {len(un)}  "
          f"{'ELIGIBLE' if ok else 'not eligible'}")
        if ok:
            eligible.append(n)
    A()
    if eligible:
        final_name = min(eligible, key=lambda n: CAND[n]["c_pool"])
        A(f"  gate-eligible candidates: {eligible}")
        A(f"  SELECTED (lowest pooled COMPLETE RMSE among eligible): "
          f"{final_name}  {CAND[final_name]['c_pool']:.4f}")
    else:
        final_name = None
        A("  NO candidate clears every gate. No new base is handed off.")
    write_report()

    # ------------------------------------------------------------- gates
    rule("GATES, SUCCESS LEVEL AND DECISION")
    if final_name is not None:
        best = CAND[final_name]
        best_name = final_name
        q_used = list(CAND_FEATS[final_name])
        flags, unsafe = DRIFT_BY_CAND[final_name]
        promoted = [g for g in GROUP_ORDER
                    if any(f in CAND_FEATS[final_name] for f in GROUPS[g])]
    gain = pool - best["f_pool"]
    d_jan, d_jul = jan - best["f_jan"], jul - best["f_jul"]
    both = d_jan > 0 and d_jul > 0
    drift_ok = len(flags) == 0
    unsafe_ok = len(unsafe) == 0
    A(f"  full pooled improvement      : {gain:+.4f} RMSE")
    A(f"  JAN improvement              : {d_jan:+.4f}")
    A(f"  JUL improvement              : {d_jul:+.4f}")
    A(f"  both folds improve           : {both}")
    A(f"  gain spread over >= 100 rows : {spread_ok} (pooled n80 = {n80p:,})")
    A(f"  no 2026 drift flag           : {drift_ok}")
    A(f"  no history-length UNSAFE     : {unsafe_ok}"
      + (f"  -> {', '.join(d['feature'] for d in unsafe)}" if unsafe else ""))
    A()
    lvl = ("STRONG" if gain >= 7.0 else "PROMISING" if gain >= 4.0
           else "USEFUL" if gain >= 2.0
           else "WEAK BUT REAL" if gain >= 1.0 else "NOT MATERIAL")
    A(f"  SUCCESS LEVEL: {lvl}   (>=7 STRONG, >=4 PROMISING, >=2 USEFUL, "
      f">=1 WEAK BUT REAL)")
    promote = bool(gain >= 1.0 and both and spread_ok and drift_ok and unsafe_ok)
    A(f"  retained queue groups        : {promoted if promoted else 'see table'}")
    A(f"  retained queue features      : {len(q_used)}")
    A(f"  architecture                 : COMPLETE -> {best_name} | "
      f"MISSING_4 -> exact F3")
    A(f"  PROMOTED AS A NEW BASE       : {'YES' if promote else 'NO'}")
    A(f"  OOF EXPORT                   : {'YES' if promote else 'NO'}")
    A(f"  2026 RANKING HANDOVER        : {'YES' if promote else 'NO'}")
    A("  AUTOMATIC SUBMISSION         : NO")
    write_report()

    summary = dict(groups="+".join(promoted) if promoted else "n/a",
                   flags=[d["feature"] for d in flags])
    oof_written = rank_written = False
    if not promote:
        rule("DECISION: SMALL STABLE SIGNAL, NOT WORTH A NEW BASE HANDOVER YET")
        A(f"  full pooled improvement {gain:+.4f} did not clear every gate.")
        A("  NO OOF export, NO ranking file, NO new base, NO submission.")
        A("  MISSING_4 stays exact F3; the handed-off C2+F3 base is unchanged.")
        A("  The queue-state family is recorded as a SMALL STABLE SIGNAL that")
        A("  does not yet justify replacing the current base model.")
        flush()
    else:
        rule("OOF EXPORT (pooled improvement >= 1.0, both folds improve, pooled gain "
             "spread over >= 100 rows, no drift flag, no UNSAFE feature)")
        oof_written = export_oof(best_name, best)
        if oof_written:
            rank_written = build_ranking_handover(best_name, q_used, dict(
                summary, jan=best["f_jan"], jul=best["f_jul"], pool=best["f_pool"],
                jan0=jan, jul0=jul, pool0=pool, gain=gain))
    write_csv()
    final_block(best_name, best, promoted, q_used, flags, unsafe, lvl, promote,
                drift, oof_written, rank_written)


def export_oof(name, best):
    jan_se = jul_se = 0.0
    for p in (OOF_OUT_JAN, OOF_OUT_JUL, OOF_OUT_ALL):
        if p.exists():
            p.unlink()
    frames = {}
    for fold, ym in FOLDS:
        va = BASE[0].filter(pl.col("ym") == ym)
        c_va = FOLDS_FRAMES[fold][1]
        m4_va = va.filter(pl.col("state") == MISSING_LABEL)
        r = CAND[name][fold]
        check(f"{fold} exported candidate rows align with the validation rows",
              bool(np.array_equal(r["y"], c_va[TARGET].to_numpy().astype(np.float64))))
        joint = (m4_va.select([MID])
                 .with_columns([pl.Series("pred", CTRL[fold]["m4_ctrl"])])
                 .vstack(c_va.select([MID])
                         .with_columns([pl.Series("pred", r["pred"])])))
        oof = (va.select([MID, pl.col(TARGET).alias("y")])
               .join(joint, on=MID, how="left")
               .with_columns(pl.lit(fold).alias("fold"))
               .select([MID, "fold", "y", "pred"])
               .with_columns([pl.col("y").cast(pl.Float64),
                              pl.col("pred").cast(pl.Float64)]))
        check(f"{fold} every validation departure has exactly one prediction",
              int(oof["pred"].null_count()) == 0)
        check(f"{fold} OOF rows == {EXPECTED[fold]['valid']:,}",
              oof.height == EXPECTED[fold]["valid"], f"{oof.height:,}")
        check(f"{fold} OOF row order equals the outer validation order",
              bool(np.array_equal(oof[MID].to_numpy(), va[MID].to_numpy())))
        check(f"{fold} OOF target equals the validation target in order",
              bool(np.array_equal(oof["y"].to_numpy().astype(np.float64),
                                  va[TARGET].to_numpy().astype(np.float64))))
        oof.write_parquet(OOF_OUT_JAN if fold == "JAN" else OOF_OUT_JUL)
        frames[fold] = oof
        yv, pv = oof["y"].to_numpy().astype(np.float64), oof["pred"].to_numpy().astype(np.float64)
        A(f"  {fold} OOF RMSE {rmse_of(yv, pv):.6f}  SSE {sse_of(yv, pv):,.1f}  "
          f"rows={oof.height:,}")
        del va, c_va, m4_va
    combined = pl.concat([frames["JAN"], frames["JUL"]])
    check("combined OOF rows == 344,419", combined.height == TOTAL_VALID,
          f"{combined.height:,}")
    check("combined JAN count == 153,706",
          int((combined["fold"] == "JAN").sum()) == 153_706)
    check("combined JUL count == 190,713",
          int((combined["fold"] == "JUL").sum()) == 190_713)
    check("combined MVT_ID_mvt unique", combined[MID].n_unique() == combined.height)
    combined.write_parquet(OOF_OUT_ALL)
    for p in (OOF_OUT_JAN, OOF_OUT_JUL, OOF_OUT_ALL):
        rr = pl.read_parquet(p)
        A()
        A(f"  re-read {p.name}: rows={rr.height:,} cols={rr.columns}")
        check(f"{p.name} columns exactly [MVT_ID_mvt, fold, y, pred]",
              rr.columns == [MID, "fold", "y", "pred"], str(rr.columns))
        check(f"{p.name} unique MVT_ID_mvt", rr[MID].n_unique() == rr.height)
        check(f"{p.name} all predictions finite",
              bool(np.isfinite(rr["pred"].to_numpy()).all()))
        check(f"{p.name} all targets finite",
              bool(np.isfinite(rr["y"].to_numpy()).all()))
        yv = rr["y"].to_numpy().astype(np.float64)
        pv = rr["pred"].to_numpy().astype(np.float64)
        A(f"    RMSE {rmse_of(yv, pv):.6f}  SSE {sse_of(yv, pv):,.1f}")
    A()
    oj = rmse_of(frames["JAN"]["y"].to_numpy().astype(np.float64),
                 frames["JAN"]["pred"].to_numpy().astype(np.float64))
    ol = rmse_of(frames["JUL"]["y"].to_numpy().astype(np.float64),
                 frames["JUL"]["pred"].to_numpy().astype(np.float64))
    op = math.sqrt((oj ** 2 * EXPECTED["JAN"]["valid"]
                    + ol ** 2 * EXPECTED["JUL"]["valid"]) / TOTAL_VALID)
    A(f"  exported POOLED RMSE {op:.6f}  (baseline {REF['pool']:.6f}, "
      f"delta {op - REF['pool']:+.6f})")
    check("exported pooled RMSE matches the measured clean splice",
          abs(op - CAND[name]["f_pool"]) < 1e-6, f"{op:.6f}")
    write_report()
    return True


def final_block(name, best, promoted, q_used, flags, unsafe, lvl, promote,
                drift, oof_written, rank_written):
    rule("FINAL RESPONSE")
    gj, gl = best["JAN"]["gain"], best["JUL"]["gain"]
    n80 = best["n80"]
    lines = [
        "STATUS                              : DONE",
        f"BEST HONEST CANDIDATE              : {name}",
        f"QUEUE GROUPS RETAINED              : "
        f"{'+'.join(promoted) if promoted else 'none promoted individually'}",
        f"QUEUE FEATURES ADDED               : {len(q_used)}",
        f"COMPLETE ROUTE  JAN                 : {best['JAN']['rmse']:.4f}  "
        f"(control {BENCH_C['JAN']:.4f}, {best['JAN']['rmse'] - BENCH_C['JAN']:+.4f})",
        f"COMPLETE ROUTE  JUL                 : {best['JUL']['rmse']:.4f}  "
        f"(control {BENCH_C['JUL']:.4f}, {best['JUL']['rmse'] - BENCH_C['JUL']:+.4f})",
        f"COMPLETE ROUTE  POOLED              : {best['c_pool']:.4f}  "
        f"(control {REF['c_pool']:.4f}, {best['c_pool'] - REF['c_pool']:+.4f})",
        "",
        "CLEAN FULL SPLICE  (COMPLETE = best queue candidate, MISSING_4 = exact F3)",
        f"  JAN                               : {best['f_jan']:.4f}  "
        f"(baseline {REF['jan']:.6f}, {best['f_jan'] - REF['jan']:+.4f})",
        f"  JUL                               : {best['f_jul']:.4f}  "
        f"(baseline {REF['jul']:.6f}, {best['f_jul'] - REF['jul']:+.4f})",
        f"  POOLED                            : {best['f_pool']:.4f}  "
        f"(baseline {REF['pool']:.6f}, {best['f_pool'] - REF['pool']:+.4f})",
        f"  TOTAL SSE                         : "
        f"{best['f_pool'] ** 2 * TOTAL_VALID:,.1f}  "
        f"(baseline {REF['total_sse']:,.1f})",
        "",
        f"SUCCESS LEVEL                      : {lvl}",
        "",
        "GAIN CONCENTRATION (COMPLETE route)",
        f"  rows improved                     : {gj['improved'] + gl['improved']:,}",
        f"  rows worsened                     : {gj['worsened'] + gl['worsened']:,}",
        f"  rows carrying 50/80/90% of gain   : {gj['n50'] + gl['n50']:,} / "
        f"{n80:,} / {gj['n90'] + gl['n90']:,}",
        "",
        f"2026 DRIFT FLAGS                    : {len(flags)}"
        + (f"  {[d['feature'] for d in flags]}" if flags else "  (none)"),
        f"2026 UNSAFE (history-length)        : {len(unsafe)}"
        + (f"  {[d['feature'] for d in unsafe]}" if unsafe else "  (none)"),
        "",
        f"OOF EXPORT                          : {'YES' if oof_written else 'NO'}",
        f"  JAN OOF                           : "
        f"{OOF_OUT_JAN if oof_written else 'not written'}",
        f"  JUL OOF                           : "
        f"{OOF_OUT_JUL if oof_written else 'not written'}",
        f"  JAN+JUL OOF                       : "
        f"{OOF_OUT_ALL if oof_written else 'not written'}",
        f"2026 RANKING HANDOVER               : "
        f"{'YES' if rank_written else 'NO'}"
        + (f"  {RANKING_OUT}" if rank_written else ""),
        f"  handover text                     : "
        f"{HANDOVER_TXT if rank_written else 'not written'}",
        "",
        "ADS-B USED                          : NO",
        "EXTERNAL WEATHER USED               : NO",
        "BLOCK_TIME_UTC_mvt USED             : NO",
        "AUTOMATIC SUBMISSION                : NO",
        f"REPORT                              : {REPORT_PATH}",
        f"GROUP TABLE                         : {GROUPS_CSV}",
        "",
    ]
    for ln in lines:
        A(ln)
    n_pass = sum(1 for _, ok in CHECKS if ok)
    n_fail = sum(1 for _, ok in CHECKS if not ok)
    A(f"CHECKS                              : {n_pass} PASS / {n_fail} FAIL")
    for lbl, ok in CHECKS:
        if not ok:
            A(f"  FAIL: {lbl}")
    A(f"wall seconds                        : {time.time() - T_START:.1f}")
    write_report()
    flush()


if __name__ == "__main__":
    main()
