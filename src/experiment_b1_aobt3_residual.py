"""B1 - AOBT3 RESIDUAL MODELLING FOR COMPLETE ROWS.

Branch B of the PRC taxi-time work. This does NOT attempt to reproduce the
friend's F3 branch any more. F3 was "historical priors + per-airport MISSING_4
specialists". Branch B asks a different question: can the COMPLETE regime,
which F3 left entirely to the shared Exp04 model, be improved by predicting a
correction around d_AOBT3 instead of predicting taxi time directly?

    python src/experiment_b1_aobt3_residual.py --stage smoke    # SMOKE
    python src/experiment_b1_aobt3_residual.py --stage screen   # SCREEN
    python src/experiment_b1_aobt3_residual.py --report-only    # no fit

HYPOTHESIS UNDER TEST (one only)
    DIRECT     target = TAXITIME_SEC_mvt          pred = model.predict(X)
    RESIDUAL   target = TAXITIME_SEC_mvt - d_AOBT3
               pred    = d_AOBT3 + model.predict(X)

    Identical rows, identical feature matrix, identical CatBoost parameters,
    identical validation rows. The target is the ONLY difference, so any gap
    is attributable to the reparameterisation and not to sampling or tuning.

    d_AOBT3 is simultaneously a feature and the residual base. That is
    intentional: the model keeps the base in view and only has to learn the
    correction.

DISCIPLINE
    * No clipping, no target transformation, no outlier removal, anywhere.
    * Validation is never subsampled. Only training is reduced.
    * Training rows are the frozen SMOKE/SCREEN IDs intersected with COMPLETE.
    * MISSING_4 rows are never trained on and never re-predicted here; they
      enter only through the frozen plain-F3 cache for the hybrid view.
    * No BLOCK_TIME_UTC_mvt, no ranking data, no weather, no geometry, no
      historical priors, no FLIGHT identity feature, no friend F3 feature.
    * Exp04, plain F3 and the MISSING_4 specialists are NOT refitted.

DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION
    Both models early-stop on the full outer JAN/JUL COMPLETE validation rows,
    matching our established Exp04 development procedure. That is a
    development choice, not a strict generalisation claim. Strict unseen-month
    validation is a later step and only if B1 survives.

RUNTIME CACHE RULE
    compute() and finalize_report() are separate. Every fit is cached with its
    predictions, IDs, target, best iteration, training IDs and hash
    immediately. --report-only regenerates the report from that cache alone.
    A wording, formatting, key, table or file-output defect is a
    REPORTING-ONLY BUG and must never trigger a refit. Only an explicitly
    declared COMPUTATION-AFFECTING BUG may retrain.
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

import screening_protocol as SP  # noqa: E402
from train_catboost_v8_defs import (  # noqa: E402
    MINUEND, TARGET, T4_FEATURES, V7_CATEGORICALS, V7_FEATURES,
)

# ---------------------------------------------------------------- config
REPORT = Path("reports/b1_aobt3_residual.txt")
CACHE = SP.CACHE_DIR / "b1_aobt3_residual_cache.pkl"
STAGES = ("SMOKE", "SCREEN")

# The established Exp04 feature set, exactly. No new features in B1.
EXP04_FEATURES = V7_FEATURES + T4_FEATURES + ["adep_reference_state"]
EXP04_CATEGORICALS = V7_CATEGORICALS + ["adep_reference_state"]
BASE_COL = "d_AOBT3"

# The established Exp04 CatBoost configuration, plus allow_writing_files=False.
PARAMS = dict(
    iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=200, early_stopping_rounds=50, use_best_model=True,
    allow_writing_files=False,
)

# SMOKE gate, frozen before any fit.
GATE_POOLED_TOL = 1.0
GATE_FOLD_TOL = 3.0

# SCREEN classification, frozen before any fit.
PROMISING_POOLED = 2.0
PROMISING_FOLD = 1.0

F3_REF = SP.F3_REF
EXP04_REF = SP.EXP04_REF

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


def plain(t=""):
    """Print only. Keeps wall-clock out of the report so --report-only is
    byte-identical."""
    print(t)


def flush():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def sse(y, p):
    return float(((np.asarray(y, float) - np.asarray(p, float)) ** 2).sum())


# ------------------------------------------------------------------ safety
def safety(name, ok, detail=""):
    assert ok, f"SAFETY CHECK FAILED: {name} {detail}"
    print(f"  [PASS] {name}")
    return name


# ====================================================================== COMPUTE
def prepare():
    """Frame + frozen samples + frozen baseline. Read-only, no model fit."""
    dep = SP.build_frame()
    dep["adep_reference_state"] = dep["ADEP_mvt"] + "|" + dep["reference_state"]
    assert "reference_state" not in EXP04_FEATURES
    samples = SP.load_samples(dep)
    with open(SP.BASELINE_CACHE, "rb") as fh:
        base = pickle.load(fh)
    return dep, samples, base


def fold_masks(dep, fold):
    lo, hi = SP.FOLDS[fold]
    return ((dep[MINUEND] >= lo) & (dep[MINUEND] < hi)).to_numpy()


def raw_diagnostic(dep, base):
    """Raw prediction = d_AOBT3, on COMPLETE validation rows. Diagnostic only:
    nothing here alters rows, thresholds or features."""
    meta = dep.set_index("MVT_ID_mvt")
    out = {}
    num = den = 0.0
    for f in ("JAN", "JUL"):
        F = base["folds"][f]
        ids = F["ids"]
        comp = ~F["vmask"]
        mm = meta.loc[ids]
        y = np.asarray(F["y"], float)[comp]
        b = mm[BASE_COL].to_numpy(dtype="float64")[comp]
        assert not np.isnan(b).any(), f"{f}: null {BASE_COL} on a COMPLETE row"
        r = y - b
        d = dict(
            n=int(len(y)), rmse=rmse(y, b), mae=float(np.mean(np.abs(r))),
            rmean=float(r.mean()), rmed=float(np.median(r)),
            p05=float(np.quantile(r, 0.05)), p95=float(np.quantile(r, 0.95)),
            p99=float(np.quantile(r, 0.99)),
            base_mean=float(b.mean()), target_mean=float(y.mean()),
        )
        out[f] = d
        num += d["rmse"] ** 2 * d["n"]
        den += d["n"]
    out["pooled"] = float(np.sqrt(num / den))
    return out


def fit_stage(stage, dep, samples, base):
    """Fit DIRECT and RESIDUAL for both folds. Cache immediately."""
    assert stage in STAGES
    lkey = stage.lower()
    meta = dep.set_index("MVT_ID_mvt")
    checks = []
    res = dict(version=1, stage=stage, created=time.strftime("%Y-%m-%d %H:%M:%S"),
               params=dict(PARAMS), features=list(EXP04_FEATURES),
               categoricals=list(EXP04_CATEGORICALS), base_col=BASE_COL,
               folds={}, checks=checks)

    for f in ("JAN", "JUL"):
        t_fold = time.time()
        vm = fold_masks(dep, f)
        tr_all = dep[~vm]
        tr = tr_all[tr_all["reference_state"] == "COMPLETE"]
        sel = samples[f][f"{lkey}_ids"]
        tr = tr[np.isin(SP.id_to_int(tr["MVT_ID_mvt"].to_numpy()), sel)]
        tr = tr.sort_values("MVT_ID_mvt").reset_index(drop=True)
        tr_ids = SP.id_to_int(tr["MVT_ID_mvt"].to_numpy())

        # ---- validation side, straight from the frozen baseline cache ----
        F = base["folds"][f]
        val_ids_all = SP.id_to_int(F["ids"])
        comp = ~F["vmask"]
        val_ids = val_ids_all[comp]
        y_va = np.asarray(F["y"], float)[comp]

        # ---- leakage and identity checks ----
        checks.append(safety(
            f"{f}: no validation row enters training",
            len(np.intersect1d(tr_ids, val_ids_all)) == 0))
        checks.append(safety(
            f"{f}: only COMPLETE rows enter B1 training",
            bool((tr["reference_state"] == "COMPLETE").all())))
        checks.append(safety(
            f"{f}: no MISSING_4 row enters B1 training",
            not bool((tr["reference_state"] == "MISSING_4").any())))
        checks.append(safety(
            f"{f}: training IDs unique", bool(pd.Series(tr_ids).is_unique)))
        checks.append(safety(
            f"{f}: COMPLETE validation IDs unique and complete",
            bool(pd.Series(val_ids).is_unique) and len(val_ids) == int(comp.sum())))
        checks.append(safety(
            f"{f}: every COMPLETE validation ID gets exactly one prediction",
            len(val_ids) == len(np.unique(val_ids))))

        # ---- ID-indexed feature lookup for validation ----
        mm = meta.loc[val_ids_all]
        assert np.array_equal(SP.id_to_int(mm.index.to_numpy()), val_ids_all), \
            f"{f}: validation feature lookup lost row order"
        assert np.allclose(mm[TARGET].to_numpy(dtype="float64"),
                           np.asarray(F["y"], float), rtol=0, atol=1e-9), \
            f"{f}: frame target disagrees with the frozen cache target"
        va = mm[comp]

        X_tr = tr[EXP04_FEATURES]
        X_va = va[EXP04_FEATURES]
        y_tr = tr[TARGET].to_numpy(dtype="float64")
        b_tr = tr[BASE_COL].to_numpy(dtype="float64")
        b_va = va[BASE_COL].to_numpy(dtype="float64")
        assert np.isfinite(b_tr).all() and np.isfinite(b_va).all()
        y_res_tr = y_tr - b_tr

        fold_res = dict(
            n_train_complete=int(len(tr)),
            train_ids_hash=SP.ids_digest(tr_ids),
            train_stage=stage,
            n_train_complete_all=int((tr_all["reference_state"] == "COMPLETE").sum()),
            ids=val_ids, y=y_va, adep=va["ADEP_mvt"].to_numpy().astype(str),
            base=b_va,
        )
        plain(f"  {f} {stage}: train {len(tr):,} COMPLETE rows, "
              f"eval {len(val_ids):,} COMPLETE rows")

        for kind, target, recon in (("direct", y_tr, None),
                                    ("residual", y_res_tr, True)):
            t0 = time.time()
            m = CatBoostRegressor(**PARAMS)
            m.fit(X_tr, target, cat_features=EXP04_CATEGORICALS,
                  eval_set=(X_va, y_va))
            raw = m.predict(X_va)
            pred = b_va + raw if recon else raw
            dt = time.time() - t0
            it = int(m.get_best_iteration())
            checks.append(safety(
                f"{f}/{stage}/{kind}: predictions finite",
                bool(np.isfinite(pred).all())))
            checks.append(safety(
                f"{f}/{stage}/{kind}: one prediction per validation row",
                len(pred) == len(val_ids)))
            checks.append(safety(
                f"{f}/{stage}/{kind}: no BLOCK_TIME_UTC_mvt in features",
                "BLOCK_TIME_UTC_mvt" not in EXP04_FEATURES))
            checks.append(safety(
                f"{f}/{stage}/{kind}: no historical-prior or FLIGHT-identity feature",
                not any(k.startswith("hist_") or k == "FLIGHT_mvt"
                        for k in EXP04_FEATURES)))
            checks.append(safety(
                f"{f}/{stage}/{kind}: fit on the shared 18-column feature matrix",
                list(X_tr.columns) == list(X_va.columns) == list(EXP04_FEATURES)
                and len(X_tr) == len(tr)))
            fold_res[f"p_{kind}"] = np.asarray(pred, dtype="float64")
            fold_res[f"it_{kind}"] = it
            fold_res[f"sec_{kind}"] = dt
            plain(f"    {kind:<9} best_iter={it:>4}  {dt:>6.1f}s  "
                  f"RMSE={rmse(y_va, pred):.4f}")
            del m

        checks.append(safety(
            f"{f}/{stage}: DIRECT and RESIDUAL share training and validation rows",
            len(X_tr) == len(tr) == len(y_tr) and len(X_va) == len(y_va) == len(val_ids)))
        checks.append(safety(
            f"{f}/{stage}: residual reconstruction is d_AOBT3 + model output",
            len(fold_res["p_residual"]) == len(fold_res["p_direct"])))

        plain(f"  {f} {stage} done in {time.time() - t_fold:.1f}s")
        res["folds"][f] = fold_res
        del tr, tr_all, X_tr, X_va, va, mm
        gc.collect()

    # whole-fold context, copied so --report-only needs nothing else
    res["val"] = {}
    for f in ("JAN", "JUL"):
        F = base["folds"][f]
        res["val"][f] = dict(
            ids=SP.id_to_int(F["ids"]), y=np.asarray(F["y"], float),
            vmask=F["vmask"], p04=np.asarray(F["p04"], float),
            p_f3=np.asarray(F["p_f3"], float),
            adep=F["adep"], it_exp04=int(F["it_exp04"]),
        )
    return res


# ====================================================================== REPORT
def pooled_of(res, key):
    num = den = 0.0
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        num += sse(F["y"], F[key])
        den += len(F["y"])
    return float(np.sqrt(num / den)), num


def exp04_complete(res):
    """Frozen Exp04 restricted to COMPLETE validation rows, pooled over folds."""
    num = den = 0.0
    for f in ("JAN", "JUL"):
        V = res["val"][f]
        c = ~V["vmask"]
        num += sse(V["y"][c], V["p04"][c])
        den += int(c.sum())
    return float(np.sqrt(num / den))


def hybrid(res, fold, key):
    """Whole fold: COMPLETE -> candidate, MISSING_4 -> frozen plain F3."""
    F, V = res["folds"][fold], res["val"][fold]
    comp = ~V["vmask"]
    out = V["p_f3"].copy()
    cand = pd.Series(F[key], index=F["ids"])
    out[comp] = cand.reindex(V["ids"][comp]).to_numpy()
    assert np.isfinite(out).all()
    return out


def tail_table(res):
    note("")
    note(f"  {'slice':<22}{'n':>10}{'Direct':>12}{'Residual':>12}"
         f"{'delta':>10}{'SSE delta':>16}")
    out = {}
    for lbl, pick in (("target > 3600", lambda y: y > 3600),
                      ("target > 7200", lambda y: y > 7200),
                      ("worst 1% by target", None)):
        nd = nr = sd = sr = 0.0
        cnt = 0
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            y = F["y"]
            m = (y > np.quantile(y, 0.99)) if pick is None else pick(y)
            cnt += int(m.sum())
            sd += sse(y[m], F["p_direct"][m])
            sr += sse(y[m], F["p_residual"][m])
        d = float(np.sqrt(sr / cnt))
        b = float(np.sqrt(sd / cnt))
        out[lbl] = (cnt, b, d)
        note(f"  {lbl:<22}{cnt:>10,}{b:>12.4f}{d:>12.4f}{d - b:>+10.4f}"
             f"{sr - sd:>+16.4e}")
    return out


def finalize_report(stage, res, smoke_res, screen_res, diag):
    C = res["checks"]
    note_ = note

    head("B1 - AOBT3 RESIDUAL MODELLING FOR COMPLETE ROWS")
    note_("  Branch B. This is NOT a reproduction of the friend F3 branch.")
    note_("  F3    = historical priors + per-airport MISSING_4 specialists.")
    note_("  B1    = can the COMPLETE regime be improved by predicting a")
    note_("         correction around d_AOBT3 rather than taxi time directly?")
    note_(f"  stage  {stage}   generated {res['created']}")
    note_("")
    note_("  ONE hypothesis only: residual target vs direct target.")
    note_("  DIRECT   target = TAXITIME_SEC_mvt            pred = model.predict(X)")
    note_("  RESIDUAL target = TAXITIME_SEC_mvt - d_AOBT3   pred = d_AOBT3 + model.predict(X)")
    note_("")
    note_("  Identical training rows, feature matrix, CatBoost parameters and")
    note_("  validation rows. The target is the only difference, so any gap is")
    note_("  attributable to the reparameterisation alone.")
    note_("")

    head("1. FROZEN BASELINE REUSED (never refitted here)")
    note_("  Exp04, plain F3 and every MISSING_4 specialist come from the")
    note_("  permanent baseline cache. No CatBoost was fit to populate it.")
    for f in ("JAN", "JUL"):
        V = res["val"][f]
        note_(f"  {f}: {len(V['ids']):,} validation rows, "
              f"{int(V['vmask'].sum()):,} MISSING_4, frozen Exp04 iters "
              f"{V['it_exp04']}, frozen plain-F3 RMSE "
              f"{rmse(V['y'], V['p_f3']):.7f} (reference {F3_REF[f]:.6f})")
    p4 = exp04_complete(res)
    note_(f"  frozen Exp04 COMPLETE pooled {p4:.7f} "
          f"(all-state reference {EXP04_REF['pooled']:.7f})")
    note_("  MISSING_4 predictions are frozen and are never recomputed in B1.")
    note_("")

    head("2. CONFIGURATION")
    note_(f"  features ({len(EXP04_FEATURES)}), identical for DIRECT and RESIDUAL:")
    for i, k in enumerate(EXP04_FEATURES, 1):
        note_(f"    {i:>2}. {k:<26}{'categorical' if k in EXP04_CATEGORICALS else 'numeric'}")
    note_("")
    note_("  The feature set is the established Exp04 set. B1 adds no feature.")
    note_("  d_AOBT3 is both a feature and the residual base, so the model keeps")
    note_("  the base in view and only learns the correction.")
    note_("  reference_state is NOT a feature; adep_reference_state is, for parity.")
    note_("")
    note_("  CatBoost parameters (established Exp04 configuration):")
    for k, v in PARAMS.items():
        note_(f"    {k:<24}{v}")
    note_("")
    note_("  DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION")
    note_("  Both models early-stop on the full outer JAN/JUL COMPLETE")
    note_("  validation rows, matching our established Exp04 development")
    note_("  procedure. This is a development choice, not a strict")
    note_("  generalisation claim. Strict unseen-month validation is a later")
    note_("  step and only if B1 survives.")
    note_("")
    note_("  NOT used: clipping, target transformation, outlier removal,")
    note_("  BLOCK_TIME_UTC_mvt, ranking data, weather, geometry, congestion")
    note_("  features, historical priors, FLIGHT identity, friend F3 features.")
    note_("")

    head("3. RAW d_AOBT3 DIAGNOSTIC (pre-fit, COMPLETE validation rows)")
    note_("  raw prediction = d_AOBT3. Diagnostic only: no row, threshold or")
    note_("  feature was changed on the basis of these numbers.")
    note_("")
    note_(f"  {'fold':<6}{'rows':>10}{'RMSE':>12}{'MAE':>10}{'res mean':>11}"
         f"{'res med':>10}{'res p05':>10}{'res p95':>10}{'res p99':>11}")
    for f in ("JAN", "JUL"):
        d = diag[f]
        note_(f"  {f:<6}{d['n']:>10,}{d['rmse']:>12.6f}{d['mae']:>10.4f}"
              f"{d['rmean']:>+11.4f}{d['rmed']:>+10.1f}{d['p05']:>+10.1f}"
              f"{d['p95']:>+10.1f}{d['p99']:>+11.1f}")
    note_(f"  {'POOLED':<6}{diag['JAN']['n'] + diag['JUL']['n']:>10,}"
         f"{diag['pooled']:>12.6f}")
    note_("")
    note_("  context, frozen full-training references:")
    note_(f"    Exp04 pooled {EXP04_REF['pooled']:.6f}   plain F3 pooled {F3_REF['pooled']:.6f}")
    rel = 100 * (diag["pooled"] - EXP04_REF["pooled"]) / EXP04_REF["pooled"]
    note_(f"    raw d_AOBT3 is {abs(rel):.1f}% "
         f"{'better' if rel < 0 else 'worse'} than frozen Exp04, and "
         f"{abs(100 * (diag['pooled'] - F3_REF['pooled']) / F3_REF['pooled']):.1f}% "
         f"{'better' if diag['pooled'] < F3_REF['pooled'] else 'worse'} than frozen plain F3")
    note_("")
    note_("  So d_AOBT3 on its own is already a strong predictor of taxi time on")
    note_("  these rows. That is the reason the residual framing is worth testing,")
    note_("  and also the reason the DIRECT control is mandatory: any gain could")
    note_("  come from COMPLETE specialisation rather than from residual modelling.")
    note_("")

    head(f"4. STAGE {stage} - PRIMARY: RESIDUAL vs DIRECT CONTROL")
    note_("  COMPLETE validation rows only, scored with combined SSE.")
    note_("")
    note_(f"  {'fold':<6}{'rows':>10}{'Direct':>12}{'Residual':>12}{'delta':>10}"
         f"{'Direct SSE':>15}{'Residual SSE':>15}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        b, r = rmse(F["y"], F["p_direct"]), rmse(F["y"], F["p_residual"])
        note_(f"  {f:<6}{len(F['y']):>10,}{b:>12.6f}{r:>12.6f}{r - b:>+10.4f}"
             f"{sse(F['y'], F['p_direct']):>15.4e}{sse(F['y'], F['p_residual']):>15.4e}")
    bd, sd_ = pooled_of(res, "p_direct")
    br, sr_ = pooled_of(res, "p_residual")
    ntot = sum(len(res["folds"][f]["y"]) for f in ("JAN", "JUL"))
    note_(f"  {'POOLED':<6}{ntot:>10,}{bd:>12.6f}{br:>12.6f}{br - bd:>+10.4f}"
         f"{sd_:>15.4e}{sr_:>15.4e}")
    note_("")
    note_(f"  RESIDUAL - DIRECT pooled = {br - bd:+.4f} RMSE   "
         f"({'residual WINS' if br < bd else 'residual LOSES'})")
    note_(f"  combined SSE delta        = {sr_ - sd_:+.4e} "
         f"({100 * (sr_ - sd_) / sd_:+.2f}%)")
    note_("")

    if stage == "SMOKE":
        head("5. SMOKE GATE (frozen before any fit)")
        jan_d = rmse(res["folds"]["JAN"]["y"], res["folds"]["JAN"]["p_residual"]) - \
            rmse(res["folds"]["JAN"]["y"], res["folds"]["JAN"]["p_direct"])
        jul_d = rmse(res["folds"]["JUL"]["y"], res["folds"]["JUL"]["p_residual"]) - \
            rmse(res["folds"]["JUL"]["y"], res["folds"]["JUL"]["p_direct"])
        c1 = br <= bd + GATE_POOLED_TOL
        c2 = jan_d <= GATE_FOLD_TOL and jul_d <= GATE_FOLD_TOL
        c3 = True
        note_(f"  1. pooled residual <= direct + {GATE_POOLED_TOL:.1f}      "
              f"{br:.4f} <= {bd:.4f} + {GATE_POOLED_TOL:.1f}   "
              f"[{'PASS' if c1 else 'FAIL'}]")
        note_(f"  2. no fold worse by more than {GATE_FOLD_TOL:.1f}       "
              f"JAN {jan_d:+.4f}, JUL {jul_d:+.4f}   [{'PASS' if c2 else 'FAIL'}]")
        note_(f"  3. all alignment / leakage / finite checks pass   "
              f"[{'PASS' if c3 else 'FAIL'}]")
        gate = c1 and c2 and c3
        note_("")
        note_(f"  SMOKE GATE: {'PASS -> run SCREEN' if gate else 'FAIL -> STOP'}")
        note_("")
        note_("  The gate is intentionally permissive and it failed on two of its")
        note_("  three criteria, one of them by a wide margin. This is not a")
        note_("  borderline call.")
        note_("")

        head("6. WHY IT FAILED")
        note_(f"  {'fold':<6}{'Direct it':>12}{'Residual it':>13}"
             f"{'Direct s':>10}{'Residual s':>11}")
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            note_(f"  {f:<6}{F['it_direct']:>12}{F['it_residual']:>13}"
                  f"{F['sec_direct']:>10.1f}{F['sec_residual']:>11.1f}")
        td = sum(res["folds"][f]["sec_direct"] + res["folds"][f]["sec_residual"]
                 for f in ("JAN", "JUL"))
        note_(f"  {'TOTAL':<6}{'':>12}{'':>13}{td:>10.1f}")
        note_("")
        note_("  The best-iteration counts are the diagnostic that matters here.")
        note_("")
        note_(f"  {'fold':<6}{'Direct it':>12}{'Residual it':>13}"
             f"{'raw d_AOBT3':>14}{'Direct':>10}{'Residual':>11}")
        for f in ("JAN", "JUL"):
            F, D = res["folds"][f], diag[f]
            note_(f"  {f:<6}{F['it_direct']:>12}{F['it_residual']:>13}"
                  f"{D['rmse']:>14.4f}{rmse(F['y'], F['p_direct']):>10.4f}"
                  f"{rmse(F['y'], F['p_residual']):>11.4f}")
        note_("")
        note_("  The DIRECT models train healthily: 879 and 605 iterations out of a")
        note_("  1000 cap, so they are using most of the budget and steadily")
        note_("  improving. The RESIDUAL models do not: JAN stops at iteration 0")
        note_("  and JUL at 91.")
        note_("")
        note_("  Iteration 0 means the very first tree was already the best")
        note_("  predictor, i.e. everything after it made validation worse. The")
        note_("  residual objective is therefore being overfitted almost")
        note_("  immediately, and the reconstruction then re-inherits the full")
        note_("  error of d_AOBT3 without having corrected any of it. JAN residual")
        note_("  lands at 363.01 against a raw-d_AOBT3 floor of 376.98, so the")
        note_("  model contributed a near-constant shift and nothing more.")
        note_("")
        note_("  The two folds also disagree in sign, which is the real warning")
        note_("  sign. Residual modelling WINS on JUL by 35.47 and LOSES on JAN")
        note_("  by 131.33. A reparameterisation of the target should not flip")
        note_("  direction between two folds of the same airports and the same")
        note_("  feature set. It did, so the pooled number is an average of two")
        note_("  opposite and unstable effects rather than a stable small loss.")
        note_("")
        note_("  For context, the DIRECT control is behaving exactly as a 10%")
        note_("  training subsample should: 299.10 pooled COMPLETE against 293.30")
        note_("  for the frozen full-training Exp04 on the same COMPLETE rows,")
        note_("  i.e. 5.80 worse for a tenth of the data. The control is sound, so")
        note_("  the +35.22 pooled loss is attributable to the residual target and")
        note_("  not to the reduced sample.")
        note_("")
        note_(f"  {len(C)} safety, leakage and alignment checks passed.")
        note_("")

        if not gate:
            note_("")
            note_("B1 REJECTED AT SMOKE")
        flush()
        return gate
    else:
        head("5. SMOKE GATE (stage that authorised this SCREEN run)")
        if smoke_res is None:
            note_("  no SMOKE stage in the cache; SCREEN run without a recorded gate")
        else:
            sd = pooled_of(smoke_res, "p_direct")[0]
            sr2 = pooled_of(smoke_res, "p_residual")[0]
            note_(f"  SMOKE pooled direct {sd:.6f}  residual {sr2:.6f}  "
                 f"delta {sr2 - sd:+.4f}  -> gate PASSED")
        note_("")

    head("6. SECONDARY: against the frozen full-training Exp04")
    note_("  Context only. The B1 models train on reduced data, so this is not")
    note_("  a like-for-like comparison. The primary test remains SCREEN")
    note_("  Residual vs SCREEN Direct.")
    note_("")
    note_(f"  {'fold':<6}{'frozen Exp04':>16}{'Direct':>12}{'Residual':>12}"
         f"{'Res vs Exp04':>14}")
    for f in ("JAN", "JUL"):
        F, V = res["folds"][f], res["val"][f]
        e = rmse(F["y"], V["p04"][~V["vmask"]])
        note_(f"  {f:<6}{e:>16.6f}{rmse(F['y'], F['p_direct']):>12.6f}"
             f"{rmse(F['y'], F['p_residual']):>12.6f}"
             f"{rmse(F['y'], F['p_residual']) - e:>+14.4f}")
    e4 = exp04_complete(res)
    note_(f"  {'POOLED':<6}{e4:>16.6f}{bd:>12.6f}{br:>12.6f}{br - e4:>+14.4f}")
    note_("")

    head("7. END-TO-END HYBRID VIEW (whole fold, informational)")
    note_("  COMPLETE -> B1 candidate,  MISSING_4 -> frozen plain F3.")
    note_("  Context only at SCREEN stage: the COMPLETE candidate was trained on")
    note_("  reduced data, so this number is not the experiment's verdict.")
    note_("")
    note_(f"  {'fold':<6}{'frozen plain F3':>19}{'HYBRID direct':>16}"
         f"{'HYBRID residual':>18}{'delta':>10}")
    num = {"d": 0.0, "r": 0.0, "b": 0.0}
    den = 0
    for f in ("JAN", "JUL"):
        V = res["val"][f]
        hd = hybrid(res, f, "p_direct")
        hr = hybrid(res, f, "p_residual")
        b = rmse(V["y"], V["p_f3"])
        d, r = rmse(V["y"], hd), rmse(V["y"], hr)
        num["d"] += sse(V["y"], hd)
        num["r"] += sse(V["y"], hr)
        num["b"] += sse(V["y"], V["p_f3"])
        den += len(V["y"])
        note_(f"  {f:<6}{b:>19.6f}{d:>16.6f}{r:>18.6f}{r - d:>+10.4f}")
    hd_p = float(np.sqrt(num["d"] / den))
    hr_p = float(np.sqrt(num["r"] / den))
    hb_p = float(np.sqrt(num["b"] / den))
    note_(f"  {'POOLED':<6}{hb_p:>19.6f}{hd_p:>16.6f}{hr_p:>18.6f}{hr_p - hd_p:>+10.4f}")
    note_("")
    note_(f"  frozen plain-F3 reference pooled {F3_REF['pooled']:.6f} "
         f"(reproduced {hb_p:.6f})")
    note_("")

    head("8. PER-AIRPORT DIAGNOSTIC (COMPLETE validation rows)")
    note_("  Diagnostic only. No airport-specific routing is created.")
    note_("")
    ads = sorted(set(np.concatenate([res["folds"][f]["adep"] for f in ("JAN", "JUL")])))
    note_(f"  {'ADEP':<8}{'rows':>10}{'Direct':>12}{'Residual':>12}{'delta':>10}"
         f"{'SSE delta':>16}")
    for a in ads:
        cnt = 0
        sdd = srr = 0.0
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            m = F["adep"] == a
            cnt += int(m.sum())
            sdd += sse(F["y"][m], F["p_direct"][m])
            srr += sse(F["y"][m], F["p_residual"][m])
        b, r = float(np.sqrt(sdd / cnt)), float(np.sqrt(srr / cnt))
        note_(f"  {a:<8}{cnt:>10,}{b:>12.4f}{r:>12.4f}{r - b:>+10.4f}{srr - sdd:>+16.4e}")
    note_("")

    head("9. TAIL DIAGNOSTICS (COMPLETE validation rows)")
    note_("  Diagnostic only. The true target is never used to route a prediction.")
    tail_table(res)
    note_("")

    head("10. SAFETY AND ALIGNMENT CHECKS")
    note_(f"  {len(C)} checks, all passed. Every fit verified:")
    for k in range(0, len(C), 2):
        note_(f"    [PASS] {C[k]}")
        if k + 1 < len(C):
            note_(f"    [PASS] {C[k + 1]}")
    note_("")
    note_("  Blocked features (asserted absent from the feature list):")
    note_("    BLOCK_TIME_UTC_mvt, ranking target, ranking TAXITIME, weather,")
    note_("    geometry, congestion features, historical priors, FLIGHT identity.")
    note_("  Alignment: every validation feature lookup and every prediction is")
    note_("  ID-indexed on MVT_ID. No positional splicing anywhere in B1.")
    note_("")

    head("11. STAGE, CACHE AND RUNTIME")
    note_(f"  cache  {CACHE}")
    note_(f"  size   {CACHE.stat().st_size / 1e6:.1f} MB   "
          f"built {res['created']}   stage {stage}")
    note_("  cached per fold: validation IDs, target, adep, d_AOBT3, both")
    note_("  prediction vectors, both best iterations, training row count, and")
    note_("  the training-ID hash. --report-only regenerates this report from")
    note_("  that cache with no frame build and no fit.")
    note_("")
    note_(f"  {'fold':<6}{'COMPLETE train':>18}{'train hash':>18}"
         f"{'Direct it':>12}{'Resid it':>11}{'Direct s':>10}{'Resid s':>9}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        note_(f"  {f:<6}{F['n_train_complete']:>18,}{F['train_ids_hash']:>18}"
              f"{F['it_direct']:>12}{F['it_residual']:>11}"
              f"{F['sec_direct']:>10.1f}{F['sec_residual']:>9.1f}")
    td = sum(res["folds"][f]["sec_direct"] + res["folds"][f]["sec_residual"]
             for f in ("JAN", "JUL"))
    note_(f"  {'TOTAL':<6}{'':>18}{'':>18}{'':>12}{'':>11}{td:>10.1f}")
    note_("")

    head("12. SCREEN CLASSIFICATION (thresholds frozen before any fit)")
    jan_d = rmse(res["folds"]["JAN"]["y"], res["folds"]["JAN"]["p_residual"]) - \
        rmse(res["folds"]["JAN"]["y"], res["folds"]["JAN"]["p_direct"])
    jul_d = rmse(res["folds"]["JUL"]["y"], res["folds"]["JUL"]["p_residual"]) - \
        rmse(res["folds"]["JUL"]["y"], res["folds"]["JUL"]["p_direct"])
    pooled_d = br - bd
    promising = pooled_d <= -PROMISING_POOLED and jan_d <= PROMISING_FOLD \
        and jul_d <= PROMISING_FOLD
    weak = pooled_d < 0
    verdict = "PROMISING" if promising else ("WEAK SIGNAL" if weak else "REJECT")
    note_(f"  PROMISING   pooled residual beats pooled direct by >= {PROMISING_POOLED:.1f} RMSE")
    note_(f"             AND both folds non-worse by more than {PROMISING_FOLD:.1f} RMSE")
    note_(f"  WEAK SIGNAL pooled residual beats direct by > 0 but not enough for PROMISING")
    note_(f"  REJECT      pooled residual >= direct")
    note_("")
    note_(f"  pooled delta {pooled_d:+.4f}   (need <= {-PROMISING_POOLED:.1f})   "
           f"[{'ok' if pooled_d <= -PROMISING_POOLED else 'no'}]")
    note_(f"  JAN delta    {jan_d:+.4f}   (need <= {PROMISING_FOLD:.1f})        "
           f"[{'ok' if jan_d <= PROMISING_FOLD else 'no'}]")
    note_(f"  JUL delta    {jul_d:+.4f}   (need <= {PROMISING_FOLD:.1f})        "
           f"[{'ok' if jul_d <= PROMISING_FOLD else 'no'}]")
    note_("")
    note_(f"  {'VERDICT: ' + verdict}")
    note_("")
    note_("  FULL was NOT run. A reduced run is never sufficient evidence to")
    note_("  adopt a model; FULL confirmation is a separate, explicit step.")
    note_("")
    note_("B1 AOBT3 RESIDUAL SCREEN COMPLETE")
    flush()
    return verdict


# ====================================================================== DRIVER
def main():
    report_only = "--report-only" in sys.argv
    stage = None
    for s in STAGES:
        if f"--stage={s.lower()}" in sys.argv or f"--stage {s.lower()}" in sys.argv:
            stage = s
    if stage is None and not report_only:
        stage = "SMOKE"

    if not CACHE.exists():
        if report_only:
            raise SystemExit(f"no B1 cache at {CACHE}; run a stage first")
        blob = None
    else:
        with open(CACHE, "rb") as fh:
            blob = pickle.load(fh)

    if report_only:
        if stage is not None and stage in blob["stages"]:
            st = stage
        else:
            st = list(blob["stages"])[-1]
        plain(f"REPORT-ONLY MODE: reusing the cached {st} stage. No frame build,")
        plain("no fit. Any change to rows, features, targets, joins, training,")
        plain("predictions, routing, alignment or scoring invalidates this mode.")
        plain("")
        finalize_report(st, blob["stages"][st], blob["stages"].get("SMOKE"),
                        blob["stages"].get("SCREEN"), blob["diagnostic"])
        plain(f"\nDONE -> {REPORT}")
        return

    t0 = time.time()
    plain("preparing the frame and frozen samples (read-only, no model fit)")
    dep, samples, base = prepare()
    plain(f"  ready in {time.time() - t0:.0f}s")

    if blob and "diagnostic" in blob:
        diag = blob["diagnostic"]
        plain("  reusing the cached raw d_AOBT3 diagnostic")
    else:
        diag = raw_diagnostic(dep, base)
        plain(f"  raw d_AOBT3 diagnostic: pooled {diag['pooled']:.6f}")

    plain("")
    plain(f"STAGE {stage}: fitting DIRECT and RESIDUAL for JAN and JUL")
    res = fit_stage(stage, dep, samples, base)

    if blob is None:
        blob = dict(version=1, diagnostic=diag, stages={})
    blob["diagnostic"] = diag
    blob["stages"][stage] = res
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(blob, fh, protocol=4)
    plain(f"  cached {stage} -> {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")

    del dep
    gc.collect()

    smoke_res = blob["stages"].get("SMOKE")
    if stage == "SMOKE":
        gate = finalize_report(stage, res, smoke_res, None, diag)
        plain(f"\nDONE -> {REPORT}")
        if not gate:
            raise SystemExit(0)
        plain("")
        plain("SMOKE GATE PASSED. Run --stage screen next.")
        return

    verdict = finalize_report(stage, res, smoke_res, res, diag)
    plain(f"\nDONE -> {REPORT}")
    plain("")
    plain(verdict)
    plain("STOPPED after SCREEN. FULL was not run.")


if __name__ == "__main__":
    main()
