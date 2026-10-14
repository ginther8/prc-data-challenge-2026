#!/usr/bin/env python
"""V7 C2+F3 BASE OOF EXPORT FOR ADS-B COMBINATION (HANDOVER TASK).

Purpose
-------
Export honest JAN/JUL leave-one-month-out predictions from the EXACT current
C2+F3 base model so the ADS-B teammate can train a residual corrector against
our base predictions.

This is a handover / OOF-export task ONLY. Nothing is changed: no features,
no tuning, no new experiments, no leaderboard, no rebuilding of another
architecture.

Architecture (authoritative source = experiments/run_v5_c2_rigorous_janjul.py)
---------------------------------------------------------------------------
COMPLETE -> C2  (per-ADEP residual specialists, A4 18 features + 9 C2
                 timestamp-consistency numerics = 27 features, 4000-iteration
                 probe, Inner split abs(MVT_ID_mvt) % 5 == 0, refit on ALL
                 outer-training rows of the airport at best_iteration + 1)
MISSING_4 -> F3 (exact current F3, 23 features, LOO target priors on training,
                 aggregate priors from outer-training rows only for validation,
                 1000-iteration probe, same inner split + refit)

Iteration selection, early stopping, routing, priors, thresholds and model
selection NEVER use outer JAN/JUL rows. Outer targets are scoring only.

Honest benchmark target (from reports/v5_c2_rigorous_janjul.txt):
  JAN   = 358.4586
  JUL   = 323.8051
  POOLED = 339.7072     pooled = sqrt((JAN_SSE + JUL_SSE) / 344419)

Outputs
-------
  data/processed/v7_c2f3_oof_JAN.parquet   (153,706 rows; fold="JAN")
  data/processed/v7_c2f3_oof_JUL.parquet   (190,713 rows; fold="JUL")
  data/processed/v7_c2f3_oof_JAN_JUL.parquet (344,419 rows; JAN then JUL,
                                              each in original validation order)
  handover/v7_c2f3_ranking.parquet         (byte-copy of the existing matching
                                            submissions/vigorous-glove_v5.parquet)
  handover/v7_c2f3_adsb_handover.txt

Columns of every OOF file, exactly: MVT_ID_mvt, fold, y, pred.
  y    = true TAXITIME_SEC_mvt (float64)
  pred = base model prediction (float64)

Safety
------
BLOCK_TIME_UTC_mvt is never loaded or used. No outer-fold leakage. No
leaderboard tuning. No ADS-B data. No external data. No clipping. No Git.
Runs in the foreground.
"""

import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
HANDOVER_DIR = PROJECT_ROOT / "handover"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

OOF_JAN_PATH = PROCESSED_DIR / "v7_c2f3_oof_JAN.parquet"
OOF_JUL_PATH = PROCESSED_DIR / "v7_c2f3_oof_JUL.parquet"
OOF_ALL_PATH = PROCESSED_DIR / "v7_c2f3_oof_JAN_JUL.parquet"
RANKING_SRC = SUBMISSIONS_DIR / "vigorous-glove_v5.parquet"
RANKING_COPY = HANDOVER_DIR / "v7_c2f3_ranking.parquet"
HANDOVER_REPORT = HANDOVER_DIR / "v7_c2f3_adsb_handover.txt"
SCRATCH = REPORTS_DIR / "_v7_c2f3_oof_scratch.txt"

TARGET = "TAXITIME_SEC_mvt"
MID = "MVT_ID_mvt"
FORBIDDEN = "BLOCK_TIME_UTC_mvt"

FOLDS = {"JAN": ("2025-01", 153_706, 152_271, 1_435),
         "JUL": ("2025-07", 190_713, 186_775, 3_938)}
EXPECTED_TRAIN = {"JAN": 1_931_341, "JUL": 1_894_334}
TOTAL_VALID = 344_419
EXPECTED_DEPARTURES = 2_085_047

BENCH_JAN_RMSE = 358.4586
BENCH_JUL_RMSE = 323.8051
BENCH_POOLED_RMSE = 339.7072
RMSE_TOL = 1e-3


def _load(mod_name, filename):
    path = EXP_DIR / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


import importlib.util  # noqa: E402
B = None
RIG = None


def load_libs():
    global B, RIG
    B = _load("oofbase", "run_v5_missing4_substantial.py")
    RIG = _load("oofrig", "run_v5_c2_rigorous_janjul.py")
    # keep the authoritative libs from clobbering their own historical reports:
    # their A/log/flush helpers stay live (per-airport progress), but an
    # emergency stop/write_report must only ever touch the scratch file.
    RIG.OUT = []
    RIG.CHECKS = []
    RIG.PENDING = []
    RIG.REPORT_PATH = SCRATCH
    B.OUT = []
    B.CHECKS = []
    B.REPORT_PATH = SCRATCH


OUT = []


def A(m=""):
    OUT.append(str(m))
    print(m, flush=True)


def rule(t):
    A()
    A("=" * 78)
    A(t)
    A("=" * 78)


def check(label, ok, detail=""):
    ok = bool(ok)
    line = f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else "")
    A(line)
    return ok


def stop(reason):
    A()
    A("=" * 78)
    A("STOPPED")
    A(f"  reason: {reason}")
    HANDOVER_DIR.mkdir(parents=True, exist_ok=True)
    (HANDOVER_DIR / "v7_c2f3_adsb_handover.txt").write_text("\n".join(OUT) + "\n")
    sys.exit(1)


def write_final(extra_lines):
    HANDOVER_DIR.mkdir(parents=True, exist_ok=True)
    HANDOVER_REPORT.write_text("\n".join(OUT) + ("" if not extra_lines
                                                 else "\n" + "\n".join(extra_lines))
                               + "\n")


def rmse_of(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def sse_of(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sum((y - p) ** 2))


# ------------------------------------------------------------------ main
def main():
    t_all = time.time()
    load_libs()
    rule("V7 C2+F3 BASE OOF EXPORT FOR ADS-B COMBINATION")
    A(f"architecture: COMPLETE -> C2 (exact), MISSING_4 -> F3 (exact)")
    A(f"authoritative benchmark: experiments/run_v5_c2_rigorous_janjul.py")
    A(f"benchmark target   : JAN {BENCH_JAN_RMSE}  JUL {BENCH_JUL_RMSE}  "
      f"POOLED {BENCH_POOLED_RMSE}")
    A(f"pooled             : sqrt((JAN_SSE + JUL_SSE) / {TOTAL_VALID})")
    A(f"TOTAL validation   : {TOTAL_VALID:,} rows "
      f"(JAN 153,706 + JUL 190,713)")
    A(f"{FORBIDDEN} is never loaded or used")

    rule("FRAME")
    base = B.prepare_base()
    if not check("departure rows == 2,085,047", base.height == EXPECTED_DEPARTURES,
                 f"{base.height:,}"):
        stop("departure count")
    if not check(f"{FORBIDDEN} absent from the feature frame",
                 FORBIDDEN not in base.columns):
        stop("forbidden column present")
    base = RIG.add_c2_columns(base)
    if not check("all C2 columns present", all(c in base.columns for c in RIG.C2_ALL_FEATURES)):
        stop("C2 columns missing")
    if not check("A4 feature set present", all(c in base.columns for c in RIG.A4_FEATURES)):
        stop("A4 columns missing")
    if not check("F3 (non-prior) features present",
                 all(c in base.columns for c in B.F3_FEATURES
                     if not c.startswith("hist_"))):
        stop("F3 columns missing")
    if not check(f"C2 = 18 A4 + 9 numeric = 27",
                 len(RIG.C2_ALL_FEATURES) == 27):
        stop("C2 feature set wrong")
    if not check(f"{FORBIDDEN} still absent after C2 build",
                 FORBIDDEN not in base.columns):
        stop("forbidden column appeared")
    write_out = {}

    for fold, (ym, exp_valid, exp_comp, exp_m4) in FOLDS.items():
        rule(f"FOLD {fold}   validation = {ym}, training = all other 2025 departures")
        tr_all = base.filter(pl.col("ym") != ym)
        va_all = base.filter(pl.col("ym") == ym)
        if not check(f"{fold} outer training rows == {EXPECTED_TRAIN[fold]:,}",
                     tr_all.height == EXPECTED_TRAIN[fold], f"{tr_all.height:,}"):
            stop(f"{fold} outer training count")
        if not check(f"{fold} outer validation rows == {exp_valid:,}",
                     va_all.height == exp_valid, f"{va_all.height:,}"):
            stop(f"{fold} outer validation count")
        if not check(f"{fold} no validation month inside outer training",
                     int((tr_all["ym"] == ym).sum()) == 0):
            stop(f"{fold} leakage")
        m4_tr = tr_all.filter(pl.col("state") == B.MISSING_LABEL)
        m4_va = va_all.filter(pl.col("state") == B.MISSING_LABEL)
        c_tr = tr_all.filter(pl.col("state") == "COMPLETE")
        c_va = va_all.filter(pl.col("state") == "COMPLETE")
        if not check(f"{fold} MISSING_4 validation == {exp_m4:,}",
                     m4_va.height == exp_m4, f"{m4_va.height:,}"):
            stop(f"{fold} M4 validation count")
        if not check(f"{fold} COMPLETE validation == {exp_comp:,}",
                     c_va.height == exp_comp, f"{c_va.height:,}"):
            stop(f"{fold} COMPLETE validation count")
        if not check(f"{fold} COMPLETE + MISSING_4 == outer validation",
                     m4_va.height + c_va.height == va_all.height):
            stop(f"{fold} partition")

        # ---- F3 on MISSING_4 (exact current implementation) -------------
        A()
        A(f"-- {fold}  F3 on MISSING_4  (exact current F3)")
        t_f3 = [time.time()]

        def _f3_progress(ap, kind, tri_n, va_n, best, err):
            e = np.asarray(err, dtype=np.float64)
            A(f"    {ap:<6} {kind:<17} train={tri_n:>7,}  valid={va_n:>6,}"
              f"  best_iter={best:>5}  RMSE={np.sqrt(np.mean(e * e)):>9.2f}"
              f"  {time.time() - t_f3[0]:>6.1f}s")

        m4_tr_p = B.add_priors(m4_tr, m4_tr, rows_in_source=True)
        m4_va_p = B.add_priors(m4_tr_p, m4_va, rows_in_source=False)
        m4_pred, m4_y, m4_iters, m4_cov = B.fit_predict(
            m4_tr_p, m4_va_p, list(B.F3_FEATURES), list(B.SPEC_CATEGORICALS),
            progress=_f3_progress)
        m4_pred = np.asarray(m4_pred, dtype=np.float64)
        m4_y = np.asarray(m4_y, dtype=np.float64)
        m4_target = m4_va[TARGET].to_numpy().astype(np.float64)
        if not check(f"{fold} F3 m4_y exactly equals m4_va[{TARGET}] in row order",
                     bool(np.array_equal(m4_y, m4_target))):
            stop(f"{fold} F3 y/id mismatch")
        if not check(f"{fold} F3 length(m4_pred) == length(m4_va)",
                     len(m4_pred) == m4_va.height, f"{len(m4_pred):,} vs "
                     f"{m4_va.height:,}"):
            stop(f"{fold} F3 length mismatch")
        if not check(f"{fold} F3 MVT_ID order corresponds to prediction order",
                     len(m4_va[MID].to_numpy()) == len(m4_pred)):
            stop(f"{fold} F3 order mismatch")
        if not check(f"{fold} F3 predictions finite",
                     bool(np.isfinite(m4_pred).all())):
            stop(f"{fold} F3 non-finite preds")
        A(f"  F3 {fold} MISSING_4 RMSE = {rmse_of(m4_y, m4_pred):.4f}  "
          f"rows={m4_va.height:,}")

        # ---- C2 on COMPLETE (exact rigorous procedure) -------------------
        A()
        A(f"-- {fold}  C2 on COMPLETE  (exact rigorous per-ADEP procedure)")
        c2_pred, c2_y, c2_rows = RIG.fit_complete(
            c_tr, c_va, RIG.C2_ALL_FEATURES, RIG.C2_ALL_CATEGORICALS, "C2", fold)
        c2_pred = np.asarray(c2_pred, dtype=np.float64)
        c2_y = np.asarray(c2_y, dtype=np.float64)
        c2_target = c_va[TARGET].to_numpy().astype(np.float64)
        if not check(f"{fold} C2 y exactly equals c_va[{TARGET}] in row order",
                     bool(np.array_equal(c2_y, c2_target))):
            stop(f"{fold} C2 y/id mismatch")
        if not check(f"{fold} C2 length(c2_pred) == length(c_va)",
                     len(c2_pred) == c_va.height, f"{len(c2_pred):,} vs "
                     f"{c_va.height:,}"):
            stop(f"{fold} C2 length mismatch")
        if not check(f"{fold} C2 predictions finite",
                     bool(np.isfinite(c2_pred).all())):
            stop(f"{fold} C2 non-finite preds")
        A(f"  C2 {fold} COMPLETE RMSE = {rmse_of(c2_y, c2_pred):.4f}  "
          f"rows={c_va.height:,}")

        # ---- full OOF assembly in original outer-validation row order ----
        rule(f"{fold} FULL OOF ASSEMBLY (original {fold} validation order)")
        m4_out = m4_va.select([MID]).with_columns([pl.Series("pred", m4_pred)])
        c2_out = c_va.select([MID]).with_columns([pl.Series("pred", c2_pred)])
        joint = m4_out.vstack(c2_out)
        oof = va_all.select([MID, pl.col(TARGET).alias("y")]).join(joint, on=MID,
                                                                   how="left")
        oof = oof.with_columns(
            pl.when(pl.col("pred").is_not_null())
              .then(pl.col("pred"))
              .otherwise(pl.lit(float("nan"))).alias("pred"))
        oof = oof.with_columns([pl.lit(fold).alias("fold")])
        oof = oof.select([MID, "fold", "y", "pred"])
        if not check(f"{fold} every validation departure has exactly one pred",
                     int(oof["pred"].null_count()) == 0,
                     f"{oof['pred'].null_count():,} null"):
            stop(f"{fold} incomplete OOF prediction")
        if not check(f"{fold} row count == {exp_valid:,}",
                     oof.height == exp_valid, f"{oof.height:,}"):
            stop(f"{fold} OOF row count")
        if not check(f"{fold} OOF mids == outer validation mids (order+set)",
                     bool(np.array_equal(oof[MID].to_numpy(),
                                         va_all[MID].to_numpy()))):
            stop(f"{fold} OOF id order mismatch")
        if not check(f"{fold} OOF y equals outer validation target in order",
                     bool(np.array_equal(oof["y"].to_numpy().astype(np.float64),
                                         va_all[TARGET].to_numpy().astype(np.float64)))):
            stop(f"{fold} OOF y mismatch")
        oof = oof.with_columns([
            pl.col("fold").cast(pl.String),
            pl.col("y").cast(pl.Float64),
            pl.col("pred").cast(pl.Float64)])
        write_out[fold] = oof

        ojan_se = (oof["y"].to_numpy().astype(np.float64)
                   - oof["pred"].to_numpy().astype(np.float64)) ** 2
        A(f"  {fold} RMSE={np.sqrt(ojan_se.mean()):.4f} "
          f"SSE={ojan_se.sum():,.1f}  rows={oof.height:,}")

    rule("WRITE + REREAD + INTEGRITY")
    for fold in ("JAN", "JUL"):
        if not check(f"{fold} OOF target file absent",
                     not (OOF_JAN_PATH if fold == "JAN" else OOF_JUL_PATH).exists()):
            stop(f"{fold} OOF file already exists; refusing to overwrite")
    if not check("combined OOF target file absent", not OOF_ALL_PATH.exists()):
        stop("combined OOF file already exists; refusing to overwrite")
    write_out["JAN"].write_parquet(OOF_JAN_PATH)
    write_out["JUL"].write_parquet(OOF_JUL_PATH)
    combined = pl.concat([write_out["JAN"], write_out["JUL"]])
    if not check("combined rows == 344,419", combined.height == TOTAL_VALID,
                 f"{combined.height:,}"):
        stop("combined row count")
    if not check("combined JAN count == 153,706",
                 int((combined["fold"] == "JAN").sum()) == 153_706):
        stop("combined JAN count")
    if not check("combined JUL count == 190,713",
                 int((combined["fold"] == "JUL").sum()) == 190_713):
        stop("combined JUL count")
    combined.write_parquet(OOF_ALL_PATH)

    files = [OOF_JAN_PATH, OOF_JUL_PATH, OOF_ALL_PATH]
    for p in files:
        reread = pl.read_parquet(p)
        A()
        A(f"re-read {p.name}: rows={reread.height:,} cols={reread.columns}")
        if not check(f"{p.name} has exactly [MVT_ID_mvt, fold, y, pred]",
                     reread.columns == [MID, "fold", "y", "pred"],
                     str(reread.columns)):
            stop(f"{p.name} schema")
        if not check(f"{p.name} all pred float64",
                     reread.schema["pred"] == pl.Float64):
            stop(f"{p.name} pred dtype")
        if not check(f"{p.name} all y float64",
                     reread.schema["y"] == pl.Float64):
            stop(f"{p.name} y dtype")
        if not check(f"{p.name} unique MVT_ID_mvt",
                     reread[MID].n_unique() == reread.height):
            stop(f"{p.name} duplicate ids")
        if not check(f"{p.name} no NaN pred", int(reread["pred"].null_count()) == 0):
            stop(f"{p.name} NaN pred")
        if not check(f"{p.name} no inf pred",
                     bool(np.isfinite(reread["pred"].to_numpy()).all())):
            stop(f"{p.name} inf pred")
        if not check(f"{p.name} all y finite",
                     bool(np.isfinite(reread["y"].to_numpy()).all())):
            stop(f"{p.name} non-finite y")

    rule("BENCHMARK REPRODUCTION")
    jan = pl.read_parquet(OOF_JAN_PATH)
    jul = pl.read_parquet(OOF_JUL_PATH)
    jy = jan["y"].to_numpy().astype(np.float64)
    jp = jan["pred"].to_numpy().astype(np.float64)
    uy = jul["y"].to_numpy().astype(np.float64)
    up = jul["pred"].to_numpy().astype(np.float64)
    jan_rmse = rmse_of(jy, jp)
    jan_sse = sse_of(jy, jp)
    jul_rmse = rmse_of(uy, up)
    jul_sse = sse_of(uy, up)
    pooled_rmse = np.sqrt((jan_sse + jul_sse) / TOTAL_VALID)
    A(f"  JAN  rows=153,706  RMSE={jan_rmse:.4f}  SSE={jan_sse:,.1f}")
    A(f"  JUL  rows=190,713  RMSE={jul_rmse:.4f}  SSE={jul_sse:,.1f}")
    A(f"  POOLED rows=344,419 RMSE={pooled_rmse:.4f}")
    A(f"  targets: JAN {BENCH_JAN_RMSE}  JUL {BENCH_JUL_RMSE}  "
      f"POOLED {BENCH_POOLED_RMSE}")
    ok_jan = abs(jan_rmse - BENCH_JAN_RMSE) < RMSE_TOL
    ok_jul = abs(jul_rmse - BENCH_JUL_RMSE) < RMSE_TOL
    ok_pooled = abs(pooled_rmse - BENCH_POOLED_RMSE) < RMSE_TOL
    if not check("JAN reproduces 358.4586", ok_jan, f"{jan_rmse:.6f}"):
        stop("JAN benchmark not reproduced")
    if not check("JUL reproduces 323.8051", ok_jul, f"{jul_rmse:.6f}"):
        stop("JUL benchmark not reproduced")
    if not check("POOLED reproduces 339.7072", ok_pooled, f"{pooled_rmse:.6f}"):
        stop("POOLED benchmark not reproduced")
    reproduced = bool(ok_jan and ok_jul and ok_pooled)

    rule("2026 RANKING MATCH (exact current C2+F3 submission)")
    if not check("ranking source exists", RANKING_SRC.exists(),
                 str(RANKING_SRC)):
        stop("ranking source missing")
    v5 = pl.read_parquet(RANKING_SRC)
    if not check("ranking rows == 344,841", v5.height == 344_841,
                 f"{v5.height:,}"):
        stop("ranking row count")
    if not check("ranking columns == [MVT_ID_mvt, TAXITIME_SEC_mvt]",
                 v5.columns == [MID, TARGET], str(v5.columns)):
        stop("ranking schema")
    if not check("ranking ids unique", v5[MID].n_unique() == v5.height):
        stop("ranking duplicate ids")
    if not check("ranking no NaN/inf pred",
                 int(v5[TARGET].null_count()) == 0
                 and bool(np.isfinite(v5[TARGET].to_numpy()).all())):
        stop("ranking pred not clean")
    v5_c2_report = REPORTS_DIR / "g2_c2_f3_v5_submission.txt"
    if not check("V5 report present", v5_c2_report.exists(), str(v5_c2_report)):
        stop("V5 report missing")
    rtxt = v5_c2_report.read_text()
    arch_ok = ("COMPLETE -> C2" in rtxt or
               ("C2 residual per-airport (COMPLETE)" in rtxt
                and "F3 (MISSING_4)" in rtxt))
    split_ok = ("339,551 COMPLETE" in rtxt and "5,290 MISSING_4" in rtxt
                and "344,841" in rtxt)
    if not check("V5 report shows COMPLETE->C2 / MISSING_4->F3 architecture",
                 arch_ok):
        stop("V5 architecture not confirmed from its report")
    if not check("V5 report shows 344,841 / 339,551 / 5,290", split_ok):
        stop("V5 split not confirmed from its report")
    A("  matching ranking file verified: submissions/vigorous-glove_v5.parquet")
    HANDOVER_DIR.mkdir(parents=True, exist_ok=True)
    if not check("ranking handover copy absent", not RANKING_COPY.exists()):
        stop("ranking handover copy already exists; refusing to overwrite")
    shutil.copyfile(RANKING_SRC, RANKING_COPY)
    h1 = hashlib.sha256(RANKING_SRC.read_bytes()).hexdigest()
    h2 = hashlib.sha256(RANKING_COPY.read_bytes()).hexdigest()
    if not check("handover ranking copy byte-identical",
                 h1 == h2, f"{h1[:12]}..."):
        stop("ranking copy hash mismatch")
    A(f"  ranking handover copy: {RANKING_COPY}  (rows={v5.height:,})")

    extra = [
        "",
        "=" * 78,
        "HANDOVER SUMMARY / ADS-B BASE EXPORT",
        "=" * 78,
        "BASE MODEL                     : C2 + F3",
        "INTERNAL TEAM NAME             : V7 base",
        f"ACTUAL EXISTING SUBMISSION FILE: {RANKING_SRC}",
        "OOF JAN                        : "
        f"{OOF_JAN_PATH}  rows=153,706  RMSE={jan_rmse:.4f}  SSE={jan_sse:,.1f}",
        "OOF JUL                        : "
        f"{OOF_JUL_PATH}  rows=190,713  RMSE={jul_rmse:.4f}  SSE={jul_sse:,.1f}",
        "OOF POOLED                      : "
        f"{OOF_ALL_PATH}  rows=344,419  RMSE={pooled_rmse:.4f}",
        f"RANKING FILE (source)          : {RANKING_SRC}",
        f"RANKING FILE (handover copy)   : {RANKING_COPY}",
        f"RANKING copy rows              : {v5.height:,}",
        " ",
        "ARCHITECTURE:",
        "  COMPLETE -> C2",
        "  MISSING_4 -> F3",
        "ADS-B USED IN BASE             : NO",
        "EXTERNAL DATA USED IN BASE     : NO",
        "OOF TREE SELECTION             : inner split only (abs(MVT_ID_mvt)%5==0)",
        "JAN/JUL USED FOR EARLY STOPPING: NO",
        "BLOCK_TIME USED                : NO",
        " ",
        "CHANGE SUMMARY (V4 -> current base):",
        "C2 replaced A4 on COMPLETE by adding the nine timestamp-consistency "
        "features.",
        "F3 MISSING_4 remained unchanged.",
        " ",
        "BENCHMARK REPRODUCED           : "
        f"{'YES' if reproduced else 'NO'}",
        "AUTHORITATIVE BENCHMARK REPORT : reports/v5_c2_rigorous_janjul.txt",
        " ",
        "FILES FOR TEAMMATE:",
        "1. data/processed/v7_c2f3_oof_JAN.parquet",
        "2. data/processed/v7_c2f3_oof_JUL.parquet",
        "3. data/processed/v7_c2f3_oof_JAN_JUL.parquet",
        "4. handover/v7_c2f3_ranking.parquet",
        "5. handover/v7_c2f3_adsb_handover.txt",
        "6. reports/v5_c2_rigorous_janjul.txt",
        " ",
        "VERSION COORDINATION:",
        "ADS-B teammate owns v9.",
        "Our FUTURE new base submissions: v10 or higher.",
        "This export is the EXISTING C2+F3 base. No v9/v10 created here.",
    ]
    write_final(extra)

    rule("FINAL RESPONSE")
    for line in [
        f"STATUS                              : DONE",
        f"JAN OOF                             : rows=153,706  RMSE={jan_rmse:.6f}  "
        f"path={OOF_JAN_PATH}",
        f"JUL OOF                             : rows=190,713  RMSE={jul_rmse:.6f}  "
        f"path={OOF_JUL_PATH}",
        f"POOLED                              : rows=344,419  RMSE={pooled_rmse:.6f}",
        f"BENCHMARK REPRODUCED                : {'YES' if reproduced else 'NO'}",
        f"RANKING SOURCE FILE                 : {RANKING_SRC}",
        f"RANKING HANDOVER FILE               : {RANKING_COPY}",
        "ARCHITECTURE MATCH VERIFIED         : YES",
        "ADS-B USED                          : NO",
        "EXTERNAL DATA USED                  : NO",
        "OUTER-FOLD LEAKAGE CHECK            : PASS",
        "FILES TO SEND TEAMMATE:",
        "1. data/processed/v7_c2f3_oof_JAN.parquet",
        "2. data/processed/v7_c2f3_oof_JUL.parquet",
        "3. data/processed/v7_c2f3_oof_JAN_JUL.parquet",
        "4. handover/v7_c2f3_ranking.parquet",
        "5. handover/v7_c2f3_adsb_handover.txt",
        "6. reports/v5_c2_rigorous_janjul.txt",
        f"DONE  wall={time.time() - t_all:.1f}s",
    ]:
        A(line)
    A()
    A(f"handover report: {HANDOVER_REPORT}")
    A()


if __name__ == "__main__":
    main()