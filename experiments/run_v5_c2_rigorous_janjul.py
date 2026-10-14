#!/usr/bin/env python3
"""
V5 C2 RIGOROUS JAN/JUL BENCHMARK
================================

Question
--------
What is the real, leakage-safe proper JAN/JUL result of upgrading the COMPLETE
route from A4 to C2, while MISSING_4 stays on the exact current F3?

What is measured
----------------
FULL  = C2 on COMPLETE  +  exact F3 on MISSING_4
        compared against the established proper A4+F3 benchmark:
        JAN 359.7206   JUL 331.2519   POOLED 344.2478

COMPLETE-only
-------------
A4 on COMPLETE (re-measured here with the identical 4000-iteration budget) and
C2 on COMPLETE, so the COMPLETE-side delta is attributable purely to the 9
timestamp-consistency features.

Protocol
--------
Outer folds (fixed, scoring-only validation):
    JAN  valid = ym 2025-01, train = all other 2025 departures
    JUL  valid = ym 2025-07, train = all other 2025 departures
Expected: JAN train 1,931,341 / valid 153,706
          JUL train 1,894,334 / valid 190,713
          pooled valid 344,419

COMPLETE route (A4 / C2), one model per ADEP:
    target     = TAXITIME_SEC_mvt - d_AOBT3
    prediction = d_AOBT3 + predicted residual
    inner split inside outer training, per airport:
        abs(MVT_ID_mvt) % 5 == 0  -> inner validation
        everything else            -> inner training
    probe   : depth 8, learning_rate 0.1, RMSE, random_seed 42,
              max 4000 iterations, early_stopping_rounds 100,
              use_best_model True
    refit   : ALL outer-training rows of that airport, best_iteration + 1 trees

MISSING_4 route: exact current F3, imported unchanged from
    experiments/run_v5_missing4_substantial.py
    - 23 features, 10 categoricals
    - probe 1000 iterations / 50 early-stopping rounds, same inner split
    - refit on all outer-training rows of the airport with best_iteration + 1
    - airports below 500 training rows route to the F1-style global model
    - its own 4 historical target priors, leave-one-out on training rows

C2 features are the exact constants and the exact arithmetic used by
    experiments/run_v5_fast_feature_screen.py
Nothing about the feature formulas, categoricals, routing, fallback, loss,
weights or priors is redefined here.

Safety
------
* BLOCK_TIME_UTC_mvt is never loaded and never referenced as a feature.
* Outer-validation targets are used only for the final RMSE / SSE.
  Iteration selection, routing and refit use the deterministic inner split only.
* No clipping, no weighting, no loss change, no external data.
* No MAY / AUG / DEC. No submission. No Git. No web.
* Outer validation month is excluded from outer training.
"""

import importlib.util
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "v8"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402

warnings.filterwarnings("ignore")


def _load(mod_name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# exact F3 (MISSING_4) implementation, imported unchanged
B = _load("m4base", "run_v5_missing4_substantial.py")
# exact C2 feature constants, imported unchanged
S = _load("fastscreen", "run_v5_fast_feature_screen.py")

# ---------------------------------------------------------------- constants
TARGET = B.TARGET
MID = B.MID
AP = "ADEP_mvt"
D_AOBT3 = "d_AOBT3"
RESID_COL = "__residual"
MISSING_LABEL = B.MISSING_LABEL
FORBIDDEN = B.FORBIDDEN
PER_AIRPORT_MIN_ROWS = B.PER_AIRPORT_MIN_ROWS
EXPECTED_DEPARTURES = B.EXPECTED_DEPARTURES

# ---- F3, verbatim ---------------------------------------------------------
F3_FEATURES = list(B.F3_FEATURES)
F3_CATEGORICALS = list(B.SPEC_CATEGORICALS)
F3_PROBE_PARAMS = dict(B.PROBE_PARAMS)

# ---- A4 / C2, verbatim constants from the fast screen --------------------
A4_FEATURES = list(S.B0_FEATURES)
A4_CATEGORICALS = list(S.B0_CATEGORICALS)
C2_PAIRWISE = list(S.C2_PAIRWISE)
C2_DELTA_COLS = list(S.C2_DELTA_COLS)
C2_STATS = list(S.C2_STATS)
C2_FEATURES = list(S.C2_FEATURES)
C2_ALL_FEATURES = A4_FEATURES + C2_FEATURES
C2_ALL_CATEGORICALS = list(A4_CATEGORICALS)

# ---- CatBoost for the COMPLETE route, exactly as specified ----------------
CB_CORE = dict(depth=8, learning_rate=0.1, loss_function="RMSE", random_seed=42)
PROBE_PARAMS = dict(CB_CORE, iterations=4000, verbose=200,
                    early_stopping_rounds=100, use_best_model=True)
REFIT_BASE = dict(CB_CORE, verbose=False)

FOLDS = (("JAN", "2025-01"), ("JUL", "2025-07"))
EXPECTED = {
    "JAN": dict(train=1_931_341, valid=153_706, complete_valid=152_271,
                m4_valid=1_435),
    "JUL": dict(train=1_894_334, valid=190_713, complete_valid=186_775,
                m4_valid=3_938),
}
V4_POOLED = 344.2478
V4_JAN = 359.7206
V4_JUL = 331.2519
A4_ESTABLISHED_COMPLETE = {"JAN": 222.5198, "JUL": 272.2238}
M0_ESTABLISHED_POOLED = 1933.1430

REPORT_PATH = PROJECT_ROOT / "reports" / "v5_c2_rigorous_janjul.txt"

OUT, CHECKS, PENDING = [], [], []


def A(m=""):
    """Append to the report buffer and to the live-progress buffer."""
    OUT.append(str(m))
    PENDING.append(str(m))


def log(m=""):
    print(m, flush=True)


def flush():
    """Push buffered lines to stdout. OUT, the report buffer, is never cleared."""
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
    A()
    flush()


def check(label, ok, detail=""):
    ok = bool(ok)
    line = (f"[{'PASS' if ok else 'FAIL'}] {label}"
            + (f"  ({detail})" if detail else ""))
    A(line)
    flush()
    log(line)
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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(OUT) + "\n")


def rmse_of(y, p):
    e = np.asarray(y, np.float64) - np.asarray(p, np.float64)
    return float(np.sqrt(np.mean(e * e)))


def sse_of(y, p):
    e = np.asarray(y, np.float64) - np.asarray(p, np.float64)
    return float(np.sum(e * e))


# ------------------------------------------------------- exact C2 columns
def add_c2_columns(frame):
    """Verbatim from experiments/run_v5_fast_feature_screen.py (prepare_frame).

    C2_PAIRWISE and C2_DELTA_COLS are the imported screen constants, so the
    formulas are not redefined, only applied.
    """
    for name, a, b in C2_PAIRWISE:
        frame = frame.with_columns((pl.col(a) - pl.col(b)).alias(name))
    mat = np.column_stack([
        frame[c].to_numpy().astype(np.float64) for c in C2_DELTA_COLS])
    with np.errstate(all="ignore"):
        frame = frame.with_columns(
            pl.Series("reference_delta_min", np.nanmin(mat, axis=1)),
            pl.Series("reference_delta_max", np.nanmax(mat, axis=1)),
            pl.Series("reference_delta_range", np.nanmax(mat, axis=1)
                      - np.nanmin(mat, axis=1)),
            pl.Series("reference_delta_mean", np.nanmean(mat, axis=1)),
            pl.Series("reference_delta_std", np.nanstd(mat, axis=1)),
        )
    return frame


def to_cb(frame, features, categoricals):
    df = frame.select(features).to_pandas()
    for c in categoricals:
        df[c] = df[c].astype(str)
    return df


# --------------------------------------------------------- COMPLETE route
def fit_complete(tri, va, features, cats, label, fold):
    """One model per ADEP on the residual target, nested iteration selection,
    refit on all outer-training rows of that airport. Same routing and
    fallback rule as F3 (per-airport specialist, F1-style global below
    PER_AIRPORT_MIN_ROWS). No weighting, no clipping, no loss change.
    """
    t_start = time.time()
    tri = tri.with_columns((pl.col(TARGET) - pl.col(D_AOBT3)).alias(RESID_COL))
    yv = va[TARGET].to_numpy().astype(np.float64)
    apv = va[AP].to_numpy()
    d_base = va[D_AOBT3].to_numpy().astype(np.float64)
    pred_res = np.full(va.height, np.nan, dtype=np.float64)
    small = [a for a in sorted(set(apv.tolist()))
             if tri.filter(pl.col(AP) == a).height < PER_AIRPORT_MIN_ROWS]
    A(f"  {label} {fold} per-ADEP specialists: {len(set(apv.tolist()))} airports"
      f"  (below {PER_AIRPORT_MIN_ROWS} training rows -> global: "
      f"{sorted(small) if small else 'none'})")
    A()
    rows = []

    def _probe(frame):
        iv = frame.filter(pl.col(MID).abs() % 5 == 0)
        it = frame.filter(~(pl.col(MID).abs() % 5 == 0))
        if it.height == 0 or iv.height == 0:
            return None, it.height, iv.height
        probe = CatBoostRegressor(allow_writing_files=False, **PROBE_PARAMS)
        probe.fit(to_cb(it, features, cats),
                  it[RESID_COL].to_numpy().astype(np.float64),
                  cat_features=cats,
                  eval_set=(to_cb(iv, features, cats),
                            iv[RESID_COL].to_numpy().astype(np.float64)),
                  early_stopping_rounds=PROBE_PARAMS["early_stopping_rounds"],
                  use_best_model=True)
        best = int(probe.get_best_iteration())
        del probe
        return best, it.height, iv.height

    def _refit(frame, best):
        params = dict(REFIT_BASE)
        params["iterations"] = best + 1
        model = CatBoostRegressor(allow_writing_files=False, **params)
        model.fit(to_cb(frame, features, cats),
                  frame[RESID_COL].to_numpy().astype(np.float64),
                  cat_features=cats)
        return model

    gmodel = gbest = None
    if small:
        gbest, _, _ = _probe(tri)
        if gbest is None:
            stop(f"{label} {fold}: global fallback inner split is empty")
        gmodel = _refit(tri, gbest)
        A(f"    <GLOBAL>  best_iter={gbest}  (F1-style global)")

    for ap in sorted(set(apv.tolist()),
                     key=lambda a: (-int((apv == a).sum()), str(a))):
        mask = apv == ap
        t0 = time.time()
        vaf = va.filter(pl.col(AP) == ap)
        trif = tri.filter(pl.col(AP) == ap)
        if ap in small:
            best, kind, model = gbest, "global", gmodel
        else:
            best, it_n, iv_n = _probe(trif)
            if best is None:
                stop(f"{label} {fold}: inner split empty for {ap}")
            kind = "per-ADEP"
            model = _refit(trif, best)
        if vaf.height:
            r = np.asarray(model.predict(to_cb(vaf, features, cats)),
                           dtype=np.float64)
            pred_res[mask] = r
        else:
            r = np.zeros(0, dtype=np.float64)
        if kind == "per-ADEP":
            del model
        pred = d_base[mask] + pred_res[mask]
        e = yv[mask] - pred
        rows.append(dict(airport=ap, kind=kind, train_rows=int(trif.height),
                         valid_rows=int(mask.sum()), best_iteration=int(best),
                         over_2999=bool(best > 2999), at_3999=bool(best == 3999),
                         rmse=float(np.sqrt(np.mean(e * e))) if e.size else 0.0,
                         sse=float(np.sum(e * e))))
        A(f"    {ap:<6} {kind:<9} train={trif.height:>9,}  valid={int(mask.sum()):>7,}"
          f"  best_iter={best:>5}  {'>2999' if best > 2999 else '     '}"
          f"  RMSE={rows[-1]['rmse']:>9.2f}  {time.time() - t0:>6.1f}s")
        flush()

    if gmodel is not None:
        del gmodel
    if not check(f"{label} {fold} every COMPLETE validation row has a finite "
                 f"prediction", bool(np.isfinite(pred_res).all()),
                 f"{int((~np.isfinite(pred_res)).sum())} missing"):
        stop("non-finite COMPLETE predictions")
    pred = d_base + pred_res
    A(f"  {label} {fold} COMPLETE wall={time.time() - t_start:.1f}s")
    A()
    return pred, yv, rows


def iter_stats(rows):
    it = np.array([r["best_iteration"] for r in rows], dtype=np.int64)
    return dict(
        n=int(it.size),
        median=float(np.median(it)),
        max=int(it.max()),
        over_2999=int((it > 2999).sum()),
        at_3999=int((it == 3999).sum()),
    )


# ====================================================================== main
def main():
    t_all = time.time()
    rule("V5 C2 RIGOROUS JAN/JUL BENCHMARK")
    A("COMPLETE -> C2 (18 A4 features + 9 timestamp-consistency numeric)")
    A("MISSING_4 -> exact current F3 (23 features, nested inner split + refit)")
    A("pooled = combined SSE over both folds, never a mean of fold RMSEs")
    A("outer validation targets are scoring-only: never features, never "
      "iteration selection, never routing")
    A()
    A(f"JAN outer training rows : {EXPECTED['JAN']['train']:,}")
    A(f"JAN outer validation    : {EXPECTED['JAN']['valid']:,}")
    A(f"JUL outer training rows : {EXPECTED['JUL']['train']:,}")
    A(f"JUL outer validation    : {EXPECTED['JUL']['valid']:,}")
    A(f"pooled validation rows  : "
      f"{EXPECTED['JAN']['valid'] + EXPECTED['JUL']['valid']:,}")
    A()
    A("established proper A4+F3 benchmark  : JAN 359.7206  JUL 331.2519  "
      f"POOLED {V4_POOLED}")
    A()
    A(f"A4 features ({len(A4_FEATURES)}): " + " ".join(A4_FEATURES))
    A()
    A(f"C2 added features ({len(C2_FEATURES)}):")
    for name, a, b in C2_PAIRWISE:
        A(f"    {name:<22} = {a} - {b}")
    A(f"    NaN-safe statistics over {C2_DELTA_COLS}: "
      + " ".join(C2_STATS))
    A()
    A(f"C2 COMPLETE categoricals ({len(C2_ALL_CATEGORICALS)}), unchanged: "
      + " ".join(C2_ALL_CATEGORICALS))
    A()
    A(f"F3 MISSING_4 features ({len(F3_FEATURES)}): " + " ".join(F3_FEATURES))
    A(f"F3 MISSING_4 categoricals ({len(F3_CATEGORICALS)}): "
      + " ".join(F3_CATEGORICALS))
    A()
    A(f"F3 probe params   : {F3_PROBE_PARAMS}")
    A(f"COMPLETE probe     : {PROBE_PARAMS}")
    A(f"COMPLETE refit base: {REFIT_BASE}  iterations = best_iteration + 1")
    flush()

    rule("FRAME")
    base = B.prepare_base()
    check("departure rows == 2,085,047", base.height == EXPECTED_DEPARTURES,
          f"{base.height:,}")
    check(f"{FORBIDDEN} absent from the feature frame",
          FORBIDDEN not in base.columns)
    base = add_c2_columns(base)
    check("C2 columns all present",
          all(c in base.columns for c in C2_FEATURES))
    check("A4 features all present",
          all(c in base.columns for c in A4_FEATURES))
    check("F3 features all present (excluding fold-built hist priors)",
          all(c in base.columns
              for c in F3_FEATURES if not c.startswith("hist_")))
    check("C2 features add 9 numeric to the 18-feature A4 set",
          len(C2_ALL_FEATURES) == 27 and len(C2_FEATURES) == 9)
    check(f"{FORBIDDEN} still absent after the C2 build",
          FORBIDDEN not in base.columns)
    A("C2 null profile (expected where the reference timestamp is absent):")
    for c in C2_FEATURES:
        n = int(base[c].is_null().sum())
        A(f"    {c:<26} {n:>9,}  ({100.0 * n / base.height:6.2f}%)")
    A()
    flush()

    store = {}
    for fold, ym in FOLDS:
        rule(f"FOLD {fold}   validation = {ym}, training = all other 2025 "
             f"departures")
        tr_all = base.filter(pl.col("ym") != ym)
        va_all = base.filter(pl.col("ym") == ym)
        exp = EXPECTED[fold]
        check(f"{fold} outer training rows == {exp['train']:,}",
              tr_all.height == exp["train"], f"{tr_all.height:,}")
        check(f"{fold} outer validation rows == {exp['valid']:,}",
              va_all.height == exp["valid"], f"{va_all.height:,}")
        check(f"{fold} no outer-validation month inside outer training",
              int((tr_all["ym"] == ym).sum()) == 0)
        m4_tr = tr_all.filter(pl.col("state") == MISSING_LABEL)
        m4_va = va_all.filter(pl.col("state") == MISSING_LABEL)
        c_tr = tr_all.filter(pl.col("state") == "COMPLETE")
        c_va = va_all.filter(pl.col("state") == "COMPLETE")
        check(f"{fold} MISSING_4 validation rows == {exp['m4_valid']:,}",
              m4_va.height == exp["m4_valid"], f"{m4_va.height:,}")
        check(f"{fold} COMPLETE validation rows == {exp['complete_valid']:,}",
              c_va.height == exp["complete_valid"], f"{c_va.height:,}")
        check(f"{fold} MISSING_4 + COMPLETE == outer validation",
              m4_va.height + c_va.height == va_all.height)
        A()
        flush()

        # ---- F3 on MISSING_4, exact current implementation -------------
        sub(f"{fold}  F3 on MISSING_4  (exact current F3, unchanged)")
        t_f3 = [time.time()]

        def _f3_progress(ap, kind, tri_n, va_n, best, err, label="F3"):
            e = np.asarray(err, dtype=np.float64)
            A(f"    {ap:<6} {kind:<17} train={tri_n:>7,}  valid={va_n:>6,}"
              f"  best_iter={best:>5}  RMSE={np.sqrt(np.mean(e * e)):>9.2f}"
              f"  {time.time() - t_f3[0]:>6.1f}s")
            flush()

        m4_tr_p = B.add_priors(m4_tr, m4_tr, rows_in_source=True)
        m4_va_p = B.add_priors(m4_tr_p, m4_va, rows_in_source=False)
        m4_pred, m4_y, m4_iters, m4_cov = B.fit_predict(
            m4_tr_p, m4_va_p, F3_FEATURES, F3_CATEGORICALS,
            progress=_f3_progress)
        m4_pred = np.asarray(m4_pred, dtype=np.float64)
        m4_rmse = rmse_of(m4_y, m4_pred)
        m4_sse = sse_of(m4_y, m4_pred)
        m4_ap = m4_va[AP].to_numpy()
        A()
        A(f"  F3 {fold} MISSING_4 RMSE = {m4_rmse:.4f}  rows={m4_va.height:,}"
          f"  wall={time.time() - t_f3[0]:.1f}s")
        A()
        A("  F3 MISSING_4 per-airport:")
        A(f"    {'airport':<8}{'valid':>8}{'RMSE':>12}{'share of M4 SSE':>18}")
        for ap in sorted(set(m4_ap.tolist())):
            m = m4_ap == ap
            e = m4_y[m] - m4_pred[m]
            A(f"    {ap:<8}{int(m.sum()):>8,}{np.sqrt(np.mean(e * e)):>12.2f}"
              f"{100.0 * np.sum(e * e) / m4_sse:>17.2f}%")
        A()
        flush()

        # ---- C2 on COMPLETE (the candidate) ------------------------------
        sub(f"{fold}  C2 on COMPLETE  (A4 18 + 9 C2 numeric, per-ADEP, "
            f"4000 iterations)")
        c2_pred, c_y, c2_rows = fit_complete(
            c_tr, c_va, C2_ALL_FEATURES, C2_ALL_CATEGORICALS, "C2", fold)
        c2_rmse = rmse_of(c_y, c2_pred)
        c2_sse = sse_of(c_y, c2_pred)
        A(f"  C2 {fold} COMPLETE RMSE = {c2_rmse:.4f}  rows={c_va.height:,}")
        A()
        flush()

        # ---- A4 on COMPLETE (control, identical budget) ------------------
        sub(f"{fold}  A4 on COMPLETE  (18-feature control, identical 4000-"
            f"iteration budget)")
        a4_pred, c_y2, a4_rows = fit_complete(
            c_tr, c_va, A4_FEATURES, A4_CATEGORICALS, "A4", fold)
        a4_rmse = rmse_of(c_y2, a4_pred)
        a4_sse = sse_of(c_y2, a4_pred)
        A(f"  A4 {fold} COMPLETE RMSE = {a4_rmse:.4f}  rows={c_va.height:,}")
        A()
        flush()
        check(f"{fold} COMPLETE target vector identical for A4 and C2",
              bool(np.array_equal(c_y, c_y2)))

        store[fold] = dict(
            m4_y=m4_y, m4_pred=m4_pred, m4_ap=m4_ap, m4_iters=m4_iters,
            m4_rmse=m4_rmse, m4_sse=m4_sse, m4_rows=int(m4_va.height),
            c_y=c_y, c2_pred=c2_pred, c2_rmse=c2_rmse, c2_sse=c2_sse,
            c2_rows_rows=int(c_va.height), a4_pred=a4_pred, a4_rmse=a4_rmse,
            a4_sse=a4_sse, c2_iters=c2_rows, a4_iters=a4_rows,
            c2_stats=iter_stats(c2_rows), a4_stats=iter_stats(a4_rows),
        )
        write_report()

    # ================================================================== tables
    rule("ARCHITECTURE  |  JAN  |  JUL  |  POOLED   (pooled = combined SSE)")

    def pooled_sse(key):
        return sum(store[f][key] for f, _ in FOLDS)

    def pooled_rmse(key):
        n = sum(EXPECTED[f]["valid"] for f, _ in FOLDS)
        return math.sqrt(pooled_sse(key) / n)

    full_c2_sse = sum(store[f]["c2_sse"] + store[f]["m4_sse"]
                      for f, _ in FOLDS)
    full_a4_sse = sum(store[f]["a4_sse"] + store[f]["m4_sse"]
                      for f, _ in FOLDS)
    n_pool = sum(EXPECTED[f]["valid"] for f, _ in FOLDS)
    full_c2 = math.sqrt(full_c2_sse / n_pool)
    full_a4 = math.sqrt(full_a4_sse / n_pool)
    c2_jan = math.sqrt((store["JAN"]["c2_sse"] + store["JAN"]["m4_sse"])
                       / EXPECTED["JAN"]["valid"])
    c2_jul = math.sqrt((store["JUL"]["c2_sse"] + store["JUL"]["m4_sse"])
                       / EXPECTED["JUL"]["valid"])
    a4_jan = math.sqrt((store["JAN"]["a4_sse"] + store["JAN"]["m4_sse"])
                       / EXPECTED["JAN"]["valid"])
    a4_jul = math.sqrt((store["JUL"]["a4_sse"] + store["JUL"]["m4_sse"])
                       / EXPECTED["JUL"]["valid"])

    A()
    A("MISSING_4 is the identical F3 vector in every full row below, so every")
    A("full-row difference is attributable to the COMPLETE model only.")
    A()
    A(f"  {'architecture':<34}{'JAN':>11}{'JUL':>11}{'POOLED':>11}")
    A(f"  {'-' * 34}{'-' * 11}{'-' * 11}{'-' * 11}")
    A(f"  {'V4 A4+F3 (established benchmark)':<34}{V4_JAN:>11.4f}{V4_JUL:>11.4f}"
      f"{V4_POOLED:>11.4f}")
    A(f"  {'A4+F3 (re-measured, same F3 vector)':<34}{a4_jan:>11.4f}"
      f"{a4_jul:>11.4f}{full_a4:>11.4f}")
    A(f"  {'V5 C2+F3 (CANDIDATE)':<34}{c2_jan:>11.4f}{c2_jul:>11.4f}"
      f"{full_c2:>11.4f}")
    A()
    A("NOTE  the established 344.2478 spliced an F3 MISSING_4 vector whose")
    A("      best_iteration was selected on the outer validation month. The")
    A("      outer-validation target is forbidden for iteration selection, so")
    A("      the honest F3 vector is the nested one used here. It is why the")
    A("      re-measured A4+F3 row is the correct like-for-like control for the")
    A("      C2 delta.")
    A()
    A(f"  DELTA C2+F3 minus V4 POOLED        : {full_c2 - V4_POOLED:+.4f}")
    A(f"  DELTA C2+F3 minus A4+F3 POOLED     : {full_c2 - full_a4:+.4f}")
    A(f"  DELTA C2+F3 minus A4+F3 JAN        : {c2_jan - a4_jan:+.4f}")
    A(f"  DELTA C2+F3 minus A4+F3 JUL        : {c2_jul - a4_jul:+.4f}")
    A()
    A(f"  full C2+F3 pooled SSE              : {full_c2_sse:,.1f}  "
      f"over {n_pool:,} rows")
    A(f"  full A4+F3 pooled SSE              : {full_a4_sse:,.1f}  "
      f"over {n_pool:,} rows")
    A(f"  pooled SSE change C2 vs A4         : "
      f"{100.0 * (full_c2_sse - full_a4_sse) / full_a4_sse:+.4f}%")
    A()
    flush()

    # ------------------------------------------------------ COMPLETE-only
    rule("COMPLETE-ONLY (isolates the COMPLETE model)")
    comp_c2 = math.sqrt(pooled_sse("c2_sse") / sum(
        EXPECTED[f]["complete_valid"] for f, _ in FOLDS))
    comp_a4 = math.sqrt(pooled_sse("a4_sse") / sum(
        EXPECTED[f]["complete_valid"] for f, _ in FOLDS))
    A()
    A(f"  {'architecture':<40}{'JAN':>11}{'JUL':>11}{'POOLED':>11}")
    A(f"  {'-' * 40}{'-' * 11}{'-' * 11}{'-' * 11}")
    A(f"  {'A4 (established, 3000 iterations)':<40}"
      f"{A4_ESTABLISHED_COMPLETE['JAN']:>11.4f}"
      f"{A4_ESTABLISHED_COMPLETE['JUL']:>11.4f}{251.1209:>11.4f}")
    A(f"  {'A4 (re-measured, 4000 iterations)':<40}{store['JAN']['a4_rmse']:>11.4f}"
      f"{store['JUL']['a4_rmse']:>11.4f}{comp_a4:>11.4f}")
    A(f"  {'C2 (4000 iterations)':<40}{store['JAN']['c2_rmse']:>11.4f}"
      f"{store['JUL']['c2_rmse']:>11.4f}{comp_c2:>11.4f}")
    A()
    A(f"  COMPLETE pooled SSE  A4 : {pooled_sse('a4_sse'):,.1f}")
    A(f"  COMPLETE pooled SSE  C2 : {pooled_sse('c2_sse'):,.1f}")
    A(f"  COMPLETE pooled SSE change C2 vs A4 : "
      f"{100.0 * (pooled_sse('c2_sse') - pooled_sse('a4_sse')) / pooled_sse('a4_sse'):+.4f}%")
    A()
    A("  COMPLETE per-airport RMSE, A4 -> C2 (same fold, same budget):")
    A(f"    {'fold':<6}{'airport':<8}{'valid':>8}{'A4 RMSE':>12}{'C2 RMSE':>12}"
      f"{'delta':>11}{'C2 share':>11}")
    for fold, _ in FOLDS:
        st = store[fold]
        a4map = {r["airport"]: r for r in st["a4_iters"]}
        tot = st["c2_sse"]
        for r in st["c2_iters"]:
            a = a4map[r["airport"]]
            A(f"    {fold:<6}{r['airport']:<8}{r['valid_rows']:>8,}"
              f"{a['rmse']:>12.2f}{r['rmse']:>12.2f}"
              f"{r['rmse'] - a['rmse']:>11.2f}{100.0 * r['sse'] / tot:>10.2f}%")
    A()
    flush()

    # --------------------------------------------------- iteration distribution
    rule("BEST ITERATION DISTRIBUTION  (COMPLETE route, max 4000)")
    for label, key in (("C2", "c2_stats"), ("A4", "a4_stats")):
        A()
        A(f"  {label} per-ADEP specialists")
        for fold, _ in FOLDS:
            s = store[fold][key]
            A(f"    {fold}  n={s['n']}  median={s['median']:.1f}  max={s['max']}"
              f"  >2999={s['over_2999']}  ==3999={s['at_3999']}")
            detail = "  ".join(
                f"{r['airport']}:{r['best_iteration']}"
                + ("*" if r["over_2999"] else "")
                for r in store[fold][f"{label.lower()}_iters"])
            A(f"         {detail}")
            A(f"         * = best_iteration > 2999 (3000-iteration screen cap)")
        both_it = np.array([r["best_iteration"]
                            for fold, _ in FOLDS
                            for r in store[fold][f"{label.lower()}_iters"]])
        A(f"    BOTH n={both_it.size}  median={np.median(both_it):.1f}"
          f"  max={both_it.max()}  >2999={int((both_it > 2999).sum())}"
          f"  ==3999={int((both_it == 3999).sum())}")
    A()
    A("  C2 hits the 4000-iteration ceiling on at least one airport:"
      f" {'YES' if int(sum(store[f]['c2_stats']['at_3999'] for f, _ in FOLDS)) else 'NO'}")
    A("  C2 >2999 (screen cap) count: "
      f"{int(sum(store[f]['c2_stats']['over_2999'] for f, _ in FOLDS))}")
    A("  C2 ==3999 count: "
      f"{int(sum(store[f]['c2_stats']['at_3999'] for f, _ in FOLDS))}")
    A()
    A("  F3 MISSING_4 per-airport best_iteration (1000 cap, unchanged):")
    for fold, _ in FOLDS:
        st = store[fold]
        detail = "  ".join(f"{ap}:{best}"
                           for ap, best in st["m4_iters"].items())
        A(f"    {fold}  {detail}")
    A()
    flush()

    # ---------------------------------------------------------- consistency
    rule("CONSISTENCY CHECKS")
    m0_pool = math.sqrt(
        (sum(store[f]["m4_sse"] for f, _ in FOLDS)) / 5373.0)
    A(f"  M4 F3 pooled RMSE here        : {m0_pool:.4f}")
    A(f"  M4 F3 pooled RMSE established : {M0_ESTABLISHED_POOLED:.4f}")
    check("MISSING_4 F3 vector reproduces the established M0 pooled RMSE",
          abs(m0_pool - M0_ESTABLISHED_POOLED) < 0.01,
          f"{m0_pool:.4f} vs {M0_ESTABLISHED_POOLED:.4f}")
    A()
    flush()

    # -------------------------------------------------------------- decision
    rule("DECISION")
    jan_delta = c2_jan - V4_JAN
    jul_delta = c2_jul - V4_JUL
    pooled_delta = full_c2 - V4_POOLED
    validated = (full_c2 < V4_POOLED) and jan_delta <= 3.0 and jul_delta <= 3.0
    strong = (pooled_delta <= -5.0) and c2_jan < V4_JAN and c2_jul < V4_JUL
    A()
    A(f"  C2 JAN FULL   : {c2_jan:.4f}   (V4 {V4_JAN:.4f}, delta {jan_delta:+.4f})")
    A(f"  C2 JUL FULL   : {c2_jul:.4f}   (V4 {V4_JUL:.4f}, delta {jul_delta:+.4f})")
    A(f"  C2 POOLED FULL: {full_c2:.4f}   (V4 {V4_POOLED:.4f}, delta {pooled_delta:+.4f})")
    A()
    A(f"  VALIDATED (pooled < {V4_POOLED} and neither fold regresses > 3)"
      f" : {'YES' if validated else 'NO'}")
    A(f"  STRONG VALIDATED (pooled improves >= 5 and both folds improve)"
      f" : {'YES' if strong else 'NO'}")
    A(f"  READY FOR V5 SUBMISSION BUILD (pooled <= 339)  "
      f": {'YES' if full_c2 <= 339.0 else 'NO'}")
    A(f"  HIGH-CONFIDENCE V5 CANDIDATE (pooled <= 335)    "
      f": {'YES' if full_c2 <= 335.0 else 'NO'}")
    A()
    A("  No submission is built automatically.")
    A()
    flush()

    rule("CHECK SUMMARY")
    n_pass = sum(1 for _, ok in CHECKS if ok)
    n_fail = sum(1 for _, ok in CHECKS if not ok)
    A(f"  {n_pass} PASS / {n_fail} FAIL")
    for label, ok in CHECKS:
        if not ok:
            A(f"  FAIL: {label}")
    A()
    A(f"  total wall {time.time() - t_all:.1f}s")
    write_report()
    log(f"DONE  report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
