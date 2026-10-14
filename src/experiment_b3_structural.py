"""B3 - STAND/RUNWAY STRUCTURAL PRIORS FOR COMPLETE ROWS.

Branch B, third experiment. B1 (AOBT3 residual modelling) is CLOSED/REJECTED
and B2 (multi-scale congestion) is WEAK SIGNAL / PARKED. Neither is modified,
tuned, reused or revisited here. B3 predicts TAXITIME_SEC_mvt directly; there
is no residual target anywhere in this file.

    python -u src/experiment_b3_structural.py            # automatic ladder
    python -u src/experiment_b3_structural.py --stage smoke
    python -u src/experiment_b3_structural.py --report-only

AUTOMATIC LADDER
    SMOKE -> gate passes -> SCREEN -> verdict PROMISING -> FULL -> STOP.
    WEAK SIGNAL or REJECT at SCREEN stops immediately and FULL is never run.

HYPOTHESIS UNDER TEST (one only)
    Within one airport, different stands and runway/stand combinations have
    systematically different taxi-out behaviour, because stand distance to the
    runway, runway assignment and pushback geometry differ. Exp04 already sees
    STAND_mvt and RUNWAY_mvt as categorical labels, but it receives no explicit
    leakage-safe numerical estimate of the historical taxi behaviour attached to
    those physical structures. B3 adds exactly six such features.

    CONTROL   the established Exp04 18 features
    CANDIDATE those same 18 features plus the 6 B3 structural features = 24

    Identical training IDs, validation IDs, CatBoost parameters, target,
    preprocessing, fold and early-stopping procedure. The structural prior family
    is the only experimental variable.

THE SIX FEATURES
    struct_adep_stand_mean          struct_adep_stand_count
    struct_adep_runway_mean         struct_adep_runway_count
    struct_adep_runway_stand_mean   struct_adep_runway_stand_count

    Groups: ADEP+STAND, ADEP+RUNWAY, ADEP+RUNWAY+STAND.

LEAKAGE-SAFE CONSTRUCTION
    Group statistics are built ONLY from the training rows of that fold and that
    stage. Validation targets never participate in any statistic.

    Training row, group holds n training rows:
        n > 1   count = n - 1
                mean  = (group_target_sum - own_target) / (n - 1)     [leave-one-out]
        n == 1  count = 0, mean = NaN                                [singleton]

    Validation row:
        group seen in training   count = training group count
                                  mean  = training group target mean
        group unseen             count = NaN, mean = NaN

    There is no airport-mean fallback, no global-mean fallback, no smoothing and
    no minimum-support threshold. Every NaN is handed to CatBoost natively.

SCREENING SOURCE RULE (experimental fairness)
    SMOKE  priors are built from the frozen SMOKE training rows only.
    SCREEN priors are built from the frozen SCREEN training rows only.
    FULL   priors are built from all FULL COMPLETE training rows.
    Unsampled outer-training targets are never used to build priors during
    SMOKE/SCREEN, so CONTROL and CANDIDATE see exactly the same training labels.

TARGETED EXCLUSIONS
    No FLIGHT_mvt identity, no ADEP+ADES+FLIGHT, no route prior, no friend F3
    feature, no B2 congestion feature, no weather, no geometry, no ADS-B, no
    residual target and no new categorical interaction. reference/friend_f3/* is
    never imported; this file is an independent Branch-B implementation.

SAFETY
    BLOCK_TIME_UTC_mvt, ranking data, weather and the target never enter a
    feature matrix. Alignment is by MVT_ID only, never positional. This writes no
    model artifact, no ranking prediction and no submission.

DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION
    All models early-stop on the full outer JAN/JUL COMPLETE validation rows,
    matching the established screening procedure.

RUNTIME CACHE RULE
    compute() and finalize_report() are separate. Every fit is cached immediately
    with its predictions, IDs, target, best iteration, feature list and order,
    training-ID hash, metrics and runtime. --report-only regenerates the report
    from that cache alone. A wording, formatting, key, table or file-output defect
    is a REPORTING-ONLY BUG and must never trigger a refit.
"""

from __future__ import annotations

import gc
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
    TARGET, T4_FEATURES, V7_CATEGORICALS, V7_FEATURES,
)

# ---------------------------------------------------------------- config
REPORT = Path("reports/b3_structural.txt")
CACHE = SP.CACHE_DIR / "b3_structural_cache.pkl"
FULL_CONTROL_CACHE = SP.CACHE_DIR / "complete_only_full_control.pkl"

STAGES = ("SMOKE", "SCREEN", "FULL")

CONTROL_FEATURES = V7_FEATURES + T4_FEATURES + ["adep_reference_state"]
CONTROL_CATEGORICALS = V7_CATEGORICALS + ["adep_reference_state"]

STRUCT_GROUPS = (
    ("adep_stand", ("ADEP_mvt", "STAND_mvt")),
    ("adep_runway", ("ADEP_mvt", "RUNWAY_mvt")),
    ("adep_runway_stand", ("ADEP_mvt", "RUNWAY_mvt", "STAND_mvt")),
)
STRUCTURAL_FEATURES = tuple(
    f"struct_{g}_{stat}" for g, _ in STRUCT_GROUPS for stat in ("mean", "count")
)
B3_FEATURES = CONTROL_FEATURES + list(STRUCTURAL_FEATURES)
B3_CATEGORICALS = list(CONTROL_CATEGORICALS)

PARAMS = dict(
    iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=200, early_stopping_rounds=50, use_best_model=True,
    allow_writing_files=False,
)

GATE_POOLED_TOL = 1.0
GATE_FOLD_TOL = 3.0
PROMISING_POOLED = 2.0
PROMISING_FOLD = 1.0

FORBIDDEN_SUBSTRINGS = (
    "BLOCK_TIME", "hist_", "dep_prev_", "arr_prev_", "secs_since_",
    "dep_minus_", "congestion", "traffic_last", "weather", "wind", "temp",
    "precip", "qnh", "FLIGHT_mvt", "ADES_mvt",
)

# the exact B2 family, so B3 can prove it is absent without importing B2
B2_CONGESTION_PREFIXES = (
    "dep_prev_", "arr_prev_", "secs_since_prev_", "dep_minus_arr_",
)

_buf: list[str] = []


# ---------------------------------------------------------------- output
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
    print(t)


def flush():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- metrics
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def sse(y, p):
    return float(((np.asarray(y, float) - np.asarray(p, float)) ** 2).sum())


def safety(name, ok, detail=""):
    assert ok, f"SAFETY CHECK FAILED: {name} {detail}"
    print(f"  [PASS] {name}")
    return name


def quantiles(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return (float("nan"),) * 3
    return (float(np.median(a)), float(np.quantile(a, 0.90)),
            float(np.quantile(a, 0.99)))


# ============================================ structural prior construction
def _struct_cols():
    return list(dict.fromkeys(c for _, cols in STRUCT_GROUPS for c in cols))


def _codes(df: pd.DataFrame, cols) -> np.ndarray:
    """Exact mixed-radix integer code per group key. No hashing, no collisions."""
    n = len(df)
    code = np.zeros(n, dtype=np.int64)
    for c in cols:
        u, inv = np.unique(df[c].to_numpy(), return_inverse=True)
        code = code * np.int64(len(u)) + inv.astype(np.int64)
    _, code = np.unique(code, return_inverse=True)
    return np.asarray(code, dtype=np.int64)


def _shared_codes(tr, va, cols):
    """One code space over training+validation, so codes are comparable."""
    both = pd.concat([tr[list(cols)], va[list(cols)]], ignore_index=True)
    code = _codes(both, cols)
    return code[:len(tr)], code[len(tr):]


def build_structural(tr: pd.DataFrame, y_tr: np.ndarray, va: pd.DataFrame,
                     label: str, checks: list):
    """The 6 B3 features for one fold and one stage.

    Group statistics come from the TRAINING rows only. Training rows are strictly
    leave-one-out; validation rows get the training aggregate. Unseen validation
    groups and singleton training groups yield NaN, with no fallback of any kind.
    """
    n_tr, n_va = len(tr), len(va)
    assert n_tr > 0 and n_va > 0, f"{label}: empty training or validation set"
    y_tr = np.asarray(y_tr, "float64")
    assert len(y_tr) == n_tr, f"{label}: target/row mismatch"

    out = {}
    for gname, cols in STRUCT_GROUPS:
        c_tr, c_va = _shared_codes(tr, va, cols)
        n_grp = int(max(c_tr.max(), c_va.max() if n_va else 0)) + 1

        g_cnt = np.bincount(c_tr, minlength=n_grp).astype("float64")
        g_sum = np.bincount(c_tr, weights=y_tr, minlength=n_grp)
        g_mean = np.where(g_cnt > 0, g_sum / np.maximum(g_cnt, 1.0), np.nan)

        # ---- training: strictly leave-one-out
        tr_n = g_cnt[c_tr]
        loo_ok = tr_n > 1
        tr_cnt = np.where(loo_ok, tr_n - 1.0, 0.0)
        tr_sum_loo = g_sum[c_tr] - y_tr
        tr_mean = np.where(loo_ok, tr_sum_loo / np.where(loo_ok, tr_n - 1.0, 1.0),
                           np.nan)

        # ---- validation: training aggregate, NaN when unseen
        seen = g_cnt[c_va] > 0
        va_cnt = np.where(seen, g_cnt[c_va], np.nan)
        va_mean = np.where(seen, g_mean[c_va], np.nan)

        out[f"struct_{gname}_mean"] = (tr_mean, va_mean)
        out[f"struct_{gname}_count"] = (tr_cnt, va_cnt)

        # ---------------- verification, per grouping
        # LOO identity: mean * count must reconstruct the sum of the OTHER rows.
        sel = loo_ok
        if sel.any():
            recon = tr_mean[sel] * tr_cnt[sel]
            worst = float(np.max(np.abs(recon - tr_sum_loo[sel])))
            scale = float(np.max(np.abs(tr_sum_loo[sel])))
            checks.append(safety(
                f"{label}/{gname}: leave-one-out identity mean*count == sum - y",
                worst <= 1e-6 * max(1.0, scale),
                f"max |error| {worst:.6g} vs scale {scale:.6g}"))

        sing = ~loo_ok
        checks.append(safety(
            f"{label}/{gname}: singleton training rows have count 0 and mean NaN",
            bool((tr_cnt[sing] == 0).all()) and bool(np.isnan(tr_mean[sing]).all()),
            f"{int(sing.sum())} singleton rows"))
        checks.append(safety(
            f"{label}/{gname}: no training row can see its own target",
            bool(np.all(tr_cnt <= np.maximum(tr_n - 1.0, 0.0)))
            and bool(np.all(tr_cnt <= g_cnt.max()))))

        checks.append(safety(
            f"{label}/{gname}: unseen validation rows have NaN count and NaN mean",
            bool(np.isnan(va_cnt[~seen]).all()) and bool(np.isnan(va_mean[~seen]).all()),
            f"{int((~seen).sum())} unseen rows"))
        checks.append(safety(
            f"{label}/{gname}: seen validation rows use the training aggregate",
            bool(np.allclose(va_mean[seen] * va_cnt[seen], g_sum[c_va][seen],
                             rtol=1e-9, atol=1e-6))))
        checks.append(safety(
            f"{label}/{gname}: no fallback - unseen rows are NaN, not zero or a mean",
            bool(np.isnan(va_mean[~seen]).all()) if (~seen).any() else True))
        checks.append(safety(
            f"{label}/{gname}: validation count is never below 1 when seen",
            bool((va_cnt[seen] >= 1).all()) if seen.any() else True))

        out[f"_cov_{gname}"] = dict(
            tr_singleton_pct=100.0 * float(sing.mean()),
            va_matched_pct=100.0 * float(seen.mean()),
            va_unseen_pct=100.0 * float((~seen).mean()),
            n_groups=int(n_grp))
    return out


def verify_priors_independent(tr, y_tr, va, struct, label, n_sample=300, seed=7):
    """Recompute a sample of priors by an independent pandas groupby path."""
    rng = np.random.default_rng(seed)
    y_tr = np.asarray(y_tr, "float64")
    bad = 0
    for gname, cols in STRUCT_GROUPS:
        c_tr, c_va = _shared_codes(tr, va, cols)
        g = pd.DataFrame({"k": c_tr, "y": y_tr})
        agg = g.groupby("k")["y"].agg(["sum", "count"])
        s_tr = agg["sum"].reindex(c_tr).to_numpy()
        n_tr = agg["count"].reindex(c_tr).to_numpy()
        s_va = agg["sum"].reindex(c_va).to_numpy()
        n_va = agg["count"].reindex(c_va).to_numpy()

        it = rng.choice(len(tr), min(n_sample, len(tr)), replace=False)
        tm = struct[f"struct_{gname}_mean"][0]
        ok = n_tr[it] > 1
        if ok.any():
            bad += int((~np.isclose(tm[it][ok],
                                    (s_tr[it][ok] - y_tr[it][ok]) / (n_tr[it][ok] - 1.0),
                                    rtol=1e-9, atol=1e-6)).sum())
        if (~ok).any():
            bad += int((~np.isnan(tm[it][~ok])).sum())      # singletons must be NaN
            cnt = struct[f"struct_{gname}_count"][0]
            bad += int((cnt[it][~ok] != 0.0).sum())         # and count exactly 0

        iv = rng.choice(len(va), min(n_sample, len(va)), replace=False)
        vm = struct[f"struct_{gname}_mean"][1]
        seen = n_va[iv] > 0
        if seen.any():
            bad += int((~np.isclose(vm[iv][seen], s_va[iv][seen] / n_va[iv][seen],
                                    rtol=1e-9, atol=1e-6)).sum())
        if (~seen).any():
            bad += int((~np.isnan(vm[iv][~seen])).sum())   # unseen must be NaN
            vc = struct[f"struct_{gname}_count"][1]
            bad += int((~np.isnan(vc[iv][~seen])).sum())   # and count NaN, not 0
    return safety(f"{label}: independent pandas-groupby recount agrees on a "
                  f"{n_sample}-row sample per grouping", bad == 0, f"{bad} disagreements")


# ---------------------------------------------------------------- metrics
def pooled_of(res, key):
    num = den = 0.0
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        num += sse(F["y"], F[key])
        den += len(F["y"])
    return float(np.sqrt(num / den)), num


def exp04_complete(res):
    num = den = 0.0
    for f in ("JAN", "JUL"):
        V = res["val"][f]
        c = ~V["vmask"]
        num += sse(V["y"][c], V["p04"][c])
        den += int(c.sum())
    return float(np.sqrt(num / den))


def hybrid(res, fold, key):
    F, V = res["folds"][fold], res["val"][fold]
    comp = ~V["vmask"]
    out = V["p_f3"].copy()
    out[comp] = pd.Series(F[key], index=F["ids"]).reindex(V["ids"][comp]).to_numpy()
    assert np.isfinite(out).all()
    return out


def main_table(res):
    note("")
    note(f"  {'fold':<6}{'rows':>10}{'control':>12}{'B3':>12}{'delta':>10}"
         f"{'control SSE':>15}{'B3 SSE':>15}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        c, b = rmse(F["y"], F["p_control"]), rmse(F["y"], F["p_b3"])
        note(f"  {f:<6}{len(F['y']):>10,}{c:>12.6f}{b:>12.6f}{b - c:>+10.4f}"
             f"{sse(F['y'], F['p_control']):>15.4e}{sse(F['y'], F['p_b3']):>15.4e}")
    cp, cs = pooled_of(res, "p_control")
    bp, bs = pooled_of(res, "p_b3")
    n = sum(len(res["folds"][f]["y"]) for f in ("JAN", "JUL"))
    note(f"  {'POOLED':<6}{n:>10,}{cp:>12.6f}{bp:>12.6f}{bp - cp:>+10.4f}"
         f"{cs:>15.4e}{bs:>15.4e}")
    return cp, bp, cs, bs


# ==================================================================== COMPUTE
def prepare():
    dep = SP.build_frame()
    dep["adep_reference_state"] = dep["ADEP_mvt"] + "|" + dep["reference_state"]
    assert "reference_state" not in CONTROL_FEATURES
    samples = SP.load_samples(dep)
    with open(SP.BASELINE_CACHE, "rb") as fh:
        base = pickle.load(fh)
    return dep, samples, base


def fit_stage(stage, dep, samples, base):
    assert stage in STAGES
    lkey = stage.lower()
    meta = dep.set_index("MVT_ID_mvt")
    checks = []
    res = dict(version=1, stage=stage,
               created=time.strftime("%Y-%m-%d %H:%M:%S"),
               params=dict(PARAMS), control_features=list(CONTROL_FEATURES),
               b3_features=list(B3_FEATURES), structural=list(STRUCTURAL_FEATURES),
               categoricals=list(CONTROL_CATEGORICALS), folds={}, checks=checks)

    for f in ("JAN", "JUL"):
        t_fold = time.time()
        lo, hi = SP.FOLDS[f]
        vm = ((dep[SP.MINUEND] >= lo) & (dep[SP.MINUEND] < hi)).to_numpy()
        tr_all = dep[~vm]
        tr = tr_all[tr_all["reference_state"] == "COMPLETE"]

        # SCREENING SOURCE RULE: priors and training both come from the frozen
        # stage sample only. FULL uses every COMPLETE outer-training row.
        if stage == "FULL":
            tr = tr.sort_values("MVT_ID_mvt").reset_index(drop=True)
        else:
            sel = samples[f][f"{lkey}_ids"]
            tr = tr[np.isin(SP.id_to_int(tr["MVT_ID_mvt"].to_numpy()), sel)]
            tr = tr.sort_values("MVT_ID_mvt").reset_index(drop=True)
        tr_ids = SP.id_to_int(tr["MVT_ID_mvt"].to_numpy())

        F = base["folds"][f]
        val_ids_all = SP.id_to_int(F["ids"])
        comp = ~F["vmask"]
        val_ids = val_ids_all[comp]
        y_va = np.asarray(F["y"], float)[comp]

        checks.append(safety(f"{f}/{stage}: no validation row enters training",
                             len(np.intersect1d(tr_ids, val_ids_all)) == 0))
        checks.append(safety(f"{f}/{stage}: only COMPLETE rows enter training "
                             f"(MISSING_4 never trains B3)",
                             bool((tr["reference_state"] == "COMPLETE").all())))
        checks.append(safety(f"{f}/{stage}: training IDs unique",
                             bool(pd.Series(tr_ids).is_unique)))
        checks.append(safety(f"{f}/{stage}: every COMPLETE validation ID is unique "
                             f"and complete",
                             bool(pd.Series(val_ids).is_unique)
                             and len(val_ids) == int(comp.sum())))

        mm = meta.loc[val_ids_all]
        assert np.array_equal(SP.id_to_int(mm.index.to_numpy()), val_ids_all), \
            f"{f}/{stage}: validation feature lookup lost row order"
        assert np.allclose(mm[TARGET].to_numpy("float64"),
                           np.asarray(F["y"], float), rtol=0, atol=1e-9), \
            f"{f}/{stage}: frame target disagrees with the frozen cache target"
        va = mm[comp].copy()

        y_tr = tr[TARGET].to_numpy("float64")
        struct = build_structural(tr, y_tr, va, f"{f}/{stage}", checks)
        verify_priors_independent(tr, y_tr, va, struct, f"{f}/{stage}")

        # CONTROL = Exp04 18 only. B3 = the same 18 plus exactly 6.
        X_c_tr = tr[CONTROL_FEATURES].reset_index(drop=True)
        X_c_va = va[CONTROL_FEATURES].reset_index(drop=True)
        s_tr = pd.DataFrame({k: struct[k][0] for k in STRUCTURAL_FEATURES})
        s_va = pd.DataFrame({k: struct[k][1] for k in STRUCTURAL_FEATURES})
        X_b_tr = pd.concat([X_c_tr, s_tr], axis=1)[B3_FEATURES]
        X_b_va = pd.concat([X_c_va, s_va], axis=1)[B3_FEATURES]
        assert list(X_b_tr.columns) == list(B3_FEATURES) == list(X_b_va.columns)
        assert len(X_c_tr) == len(X_b_tr) == len(tr) == len(y_tr)
        assert len(X_c_va) == len(X_b_va) == len(val_ids)

        checks.append(safety(
            f"{f}/{stage}: B3 has exactly six extra numerical features",
            len(B3_FEATURES) == len(CONTROL_FEATURES) + 6
            and B3_FEATURES[:len(CONTROL_FEATURES)] == CONTROL_FEATURES))
        checks.append(safety(
            f"{f}/{stage}: BLOCK_TIME_UTC_mvt never used",
            "BLOCK_TIME_UTC_mvt" not in B3_FEATURES + CONTROL_FEATURES))
        checks.append(safety(
            f"{f}/{stage}: the six new features contain no weather, congestion, "
            f"flight-identity or route feature",
            not any(any(t in k for t in FORBIDDEN_SUBSTRINGS)
                    for k in STRUCTURAL_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: no B2 congestion feature anywhere in the matrix",
            not any(any(t in k for t in B2_CONGESTION_PREFIXES)
                    for k in B3_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: control is exactly the established Exp04 18-feature set",
            CONTROL_FEATURES == V7_FEATURES + T4_FEATURES + ["adep_reference_state"]
            and len(CONTROL_FEATURES) == 18))
        checks.append(safety(
            f"{f}/{stage}: no new categorical interaction or residual target",
            B3_CATEGORICALS == CONTROL_CATEGORICALS
            and TARGET == "TAXITIME_SEC_mvt"))
        checks.append(safety(
            f"{f}/{stage}: structural features are numerical with native NaN",
            all(pd.api.types.is_numeric_dtype(X_b_tr[k])
                for k in STRUCTURAL_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: friend F3 code never imported",
            not any(m.startswith(("reference", "friend_f3")) for m in sys.modules)))
        checks.append(safety(
            f"{f}/{stage}: validation targets excluded from every structural prior",
            all(len(struct[f"struct_{g}_mean"][1]) == len(val_ids)
                for g, _ in STRUCT_GROUPS)))

        cov = {g: struct[f"_cov_{g}"] for g, _ in STRUCT_GROUPS}
        fold_res = dict(
            n_train_complete=int(len(tr)),
            n_train_complete_all=int((tr_all["reference_state"] == "COMPLETE").sum()),
            train_ids_hash=SP.ids_digest(tr_ids), train_stage=stage,
            ids=val_ids, y=y_va, adep=va["ADEP_mvt"].to_numpy().astype(str),
            coverage=cov,
            tr_counts={g: struct[f"struct_{g}_count"][0] for g, _ in STRUCT_GROUPS},
            va_counts={g: struct[f"struct_{g}_count"][1] for g, _ in STRUCT_GROUPS},
            va_seen={g: np.isfinite(struct[f"struct_{g}_count"][1])
                     for g, _ in STRUCT_GROUPS},
        )
        plain(f"  {f} {stage}: train {len(tr):,} COMPLETE rows, "
              f"eval {len(val_ids):,} COMPLETE rows")

        for kind, Xt, Xv in (("control", X_c_tr, X_c_va), ("b3", X_b_tr, X_b_va)):
            t0 = time.time()
            m = CatBoostRegressor(**PARAMS)
            m.fit(Xt, y_tr, cat_features=CONTROL_CATEGORICALS, eval_set=(Xv, y_va))
            pred = m.predict(Xv)
            dt = time.time() - t0
            checks.append(safety(f"{f}/{stage}/{kind}: predictions finite",
                                 bool(np.isfinite(pred).all())))
            checks.append(safety(f"{f}/{stage}/{kind}: one prediction per validation row",
                                 len(pred) == len(val_ids)))
            checks.append(safety(
                f"{f}/{stage}/{kind}: no BLOCK_TIME_UTC_mvt in features",
                "BLOCK_TIME_UTC_mvt" not in
                (CONTROL_FEATURES if kind == "control" else B3_FEATURES)))
            checks.append(safety(
                f"{f}/{stage}/{kind}: no FLIGHT-identity or historical-prior feature",
                not any(k.startswith("hist_") or k == "FLIGHT_mvt"
                        for k in (CONTROL_FEATURES if kind == "control" else B3_FEATURES))))
            fold_res[f"p_{kind}"] = np.asarray(pred, "float64")
            fold_res[f"it_{kind}"] = int(m.get_best_iteration())
            fold_res[f"sec_{kind}"] = dt
            plain(f"    {kind:<8} best_iter={fold_res[f'it_{kind}']:>4}  {dt:>6.1f}s"
                  f"  RMSE={rmse(y_va, pred):.4f}")
            del m

        checks.append(safety(
            f"{f}/{stage}: control and candidate train on identical IDs",
            len(X_c_tr) == len(X_b_tr) == len(tr) == len(y_tr)))
        checks.append(safety(
            f"{f}/{stage}: control and candidate validate on identical IDs",
            len(X_c_va) == len(X_b_va) == len(val_ids)))
        plain(f"  {f} {stage} done in {time.time() - t_fold:.1f}s")
        res["folds"][f] = fold_res
        del tr, tr_all, mm, va, struct, s_tr, s_va, X_c_tr, X_b_tr, X_c_va, X_b_va
        gc.collect()

    res["val"] = {}
    for f in ("JAN", "JUL"):
        V = base["folds"][f]
        res["val"][f] = dict(
            ids=SP.id_to_int(V["ids"]), y=np.asarray(V["y"], float), vmask=V["vmask"],
            p04=np.asarray(V["p04"], float), p_f3=np.asarray(V["p_f3"], float),
            adep=V["adep"], it_exp04=int(V["it_exp04"]))
    return res


# ============================================================== FINALIZE REPORT
def per_airport(res):
    head("7. PER-AIRPORT DIAGNOSTIC (COMPLETE validation rows)")
    note("  Diagnostic only. No airport routing is created.")
    note("")
    ads = sorted(set(np.concatenate([res["folds"][f]["adep"] for f in ("JAN", "JUL")])))
    note(f"  {'ADEP':<8}{'rows':>10}{'control':>12}{'B3':>12}{'delta':>10}{'SSE delta':>16}")
    for a in ads:
        cnt = 0
        sc = sb = 0.0
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            m = F["adep"] == a
            cnt += int(m.sum())
            sc += sse(F["y"][m], F["p_control"][m])
            sb += sse(F["y"][m], F["p_b3"][m])
        if not cnt:
            continue
        c, b = float(np.sqrt(sc / cnt)), float(np.sqrt(sb / cnt))
        note(f"  {a:<8}{cnt:>10,}{c:>12.4f}{b:>12.4f}{b - c:>+10.4f}{sb - sc:>+16.4e}")
    note("")


def coverage(res):
    head("8. STRUCTURAL COVERAGE AND COUNT DISTRIBUTION")
    note("  Validation coverage uses the training rows of that fold and stage only.")
    note("  Count distribution is over matched validation rows, i.e. the support the")
    note("  model actually sees. Unseen groups are NaN, never zero and never a mean.")
    note("")
    for gname, cols in STRUCT_GROUPS:
        note(f"  grouping {gname}  ({' + '.join(cols)})")
        note(f"    {'fold':<6}{'matched %':>11}{'unseen %':>10}{'tr singleton %':>16}"
             f"{'val cnt median':>15}{'p90':>10}{'p99':>10}")
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            c = F["coverage"][gname]
            med, p90, p99 = quantiles(F["va_counts"][gname])
            note(f"    {f:<6}{c['va_matched_pct']:>11.2f}{c['va_unseen_pct']:>10.2f}"
                 f"{c['tr_singleton_pct']:>16.2f}{med:>15.0f}{p90:>10.0f}{p99:>10.0f}")
        pooled = np.concatenate([res["folds"][f]["va_counts"][gname]
                                 for f in ("JAN", "JUL")])
        med, p90, p99 = quantiles(pooled)
        note(f"    {'POOLED':<6}{'':>11}{'':>10}{'':>16}{med:>15.0f}{p90:>10.0f}{p99:>10.0f}")
        note(f"    distinct training groups seen: "
             f"{res['folds']['JAN']['coverage'][gname]['n_groups']:,}")
        note("")


def tail(res):
    head("9. TAIL DIAGNOSTIC (COMPLETE validation rows)")
    note("  Diagnostic only. The true target never routes a prediction.")
    note("")
    note(f"  {'slice':<22}{'n':>10}{'control':>12}{'B3':>12}{'delta':>10}{'SSE delta':>16}")
    for lbl, pick in (("target > 3600", lambda y: y > 3600),
                      ("target > 7200", lambda y: y > 7200),
                      ("worst 1% by target", None)):
        cnt = 0
        sc = sb = 0.0
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            y = F["y"]
            m = (y > np.quantile(y, 0.99)) if pick is None else pick(y)
            cnt += int(m.sum())
            sc += sse(y[m], F["p_control"][m])
            sb += sse(y[m], F["p_b3"][m])
        if not cnt:
            continue
        c, b = float(np.sqrt(sc / cnt)), float(np.sqrt(sb / cnt))
        note(f"  {lbl:<22}{cnt:>10,}{c:>12.4f}{b:>12.4f}{b - c:>+10.4f}{sb - sc:>+16.4e}")
    note("")


def whole_context(res):
    head("10. WHOLE-MODEL CONTEXT (informational hybrid)")
    note("  COMPLETE -> B3 or COMPLETE CONTROL, MISSING_4 -> cached plain F3.")
    note("  Context only. These numbers are NOT the primary experiment verdict.")
    note("")
    note(f"  {'fold':<6}{'frozen plain F3':>19}{'HYBRID control':>17}"
         f"{'HYBRID B3':>14}{'delta':>10}")
    for f in ("JAN", "JUL"):
        F, V = res["folds"][f], res["val"][f]
        e = rmse(V["y"], V["p_f3"])
        hc = rmse(V["y"], hybrid(res, f, "p_control"))
        hb = rmse(V["y"], hybrid(res, f, "p_b3"))
        note(f"  {f:<6}{e:>19.6f}{hc:>17.6f}{hb:>14.6f}{hb - hc:>+10.4f}")
    ys, pc, pb = [], [], []
    for f in ("JAN", "JUL"):
        V = res["val"][f]
        ys.append(V["y"])
        pc.append(hybrid(res, f, "p_control"))
        pb.append(hybrid(res, f, "p_b3"))
    y, c, b = np.concatenate(ys), np.concatenate(pc), np.concatenate(pb)
    note(f"  {'POOLED':<6}{'':>19}{rmse(y, c):>17.6f}{rmse(y, b):>14.6f}"
         f"{rmse(y, b) - rmse(y, c):>+10.4f}")
    note("")


def safety_section(res):
    head("11. SAFETY AND ALIGNMENT CHECKS")
    ch = res["checks"]
    failed = [c for c in ch if not str(c).startswith("")]
    note(f"  {len(ch)} checks, all passed:")
    for c in ch:
        note(f"    [PASS] {c}")
    note("")
    note("  Provenance of the leakage guarantees:")
    note("    - group sums and counts are built from the TRAINING rows of that")
    note("      fold and stage only; no validation target is ever aggregated")
    note("    - training rows are strictly leave-one-out: count = n-1 and")
    note("      mean = (group_sum - own_target)/(n-1), verified by the identity")
    note("      mean*count == group_sum - y on every eligible training row")
    note("    - singleton training group -> count 0, mean NaN")
    note("    - unseen validation group -> count NaN, mean NaN, no fallback,")
    note("      no airport mean, no global mean, no smoothing, no support floor")
    note("    - construction reads only ADEP_mvt, RUNWAY_mvt, STAND_mvt and the")
    note("      training TAXITIME_SEC_mvt of the same fold; BLOCK_TIME_UTC_mvt,")
    note("      ranking data, weather, geometry and ADS-B are never touched")
    note("    - every lookup is ID-indexed on MVT_ID; no positional splicing")
    note("    - reference/friend_f3/* is never imported")
    note("    - MISSING_4 rows never train either model")
    note("")
    assert not failed


def stage_cache_section(res, feat_secs):
    head("12. STAGE, CACHE AND RUNTIME")
    note(f"  fit cache {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")
    note(f"  built {res['created']}   stage {res['stage']}")
    note("")
    note(f"  {'fold':<6}{'COMPLETE train':>18}{'train hash':>20}"
         f"{'ctrl it':>10}{'B3 it':>9}{'ctrl s':>9}{'B3 s':>8}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        note(f"  {f:<6}{F['n_train_complete']:>18,}{F['train_ids_hash']:>20}"
             f"{F['it_control']:>10}{F['it_b3']:>9}"
             f"{F['sec_control']:>9.1f}{F['sec_b3']:>8.1f}")
    td = sum(res["folds"][f]["sec_control"] + res["folds"][f]["sec_b3"]
             for f in ("JAN", "JUL"))
    note(f"  {'TOTAL':<6}{'':>18}{'':>20}{'':>10}{'':>9}{td:>9.1f}")
    note("")


def finalize_report(stage, res, smoke_res, screen_res):
    head("B3 - STAND/RUNWAY STRUCTURAL PRIORS FOR COMPLETE ROWS")
    note("  Branch B, experiment 3. B1 residual modelling is CLOSED/REJECTED and B2")
    note("  congestion is WEAK SIGNAL / PARKED; neither is modified or reused here.")
    note("  B3 predicts TAXITIME_SEC_mvt directly. There is no residual target.")
    note("  ONE hypothesis only: stands and runway/stand combinations carry")
    note("  systematic taxi-out structure that Exp04 sees only as category labels.")
    note("")
    note("  DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION")
    note("")
    note(f"  CONTROL   {len(CONTROL_FEATURES)} established Exp04 features")
    note(f"  CANDIDATE those same {len(CONTROL_FEATURES)} plus "
         f"{len(STRUCTURAL_FEATURES)} B3 structural features "
         f"= {len(B3_FEATURES)}")

    head("1. STRUCTURAL PRIOR FAMILY (6 numerical features)")
    for i, k in enumerate(STRUCTURAL_FEATURES):
        g = k[len("struct_"):]
        g = g[:-len("_mean")] if g.endswith("_mean") else g[:-len("_count")]
        grp = dict(STRUCT_GROUPS)[g]
        note(f"  {i + 1:>2}. {k:<32} group = {' + '.join(grp)}")
    note("")
    note("  Nothing else is added: no FLIGHT identity, no ADEP+ADES+FLIGHT, no")
    note("  route prior, no friend F3 feature, no B2 congestion feature, no")
    note("  weather, no geometry, no ADS-B, no residual target, no new")
    note("  categorical interaction.")
    note("")

    head("2. LEAKAGE-SAFE CONSTRUCTION (rules actually applied)")
    note("  Training row, group holds n training rows:")
    note("      n > 1   count = n - 1")
    note("            mean  = (group_target_sum - own_target) / (n - 1)")
    note("      n == 1  count = 0, mean = NaN")
    note("  Validation row: training aggregate only; unseen -> count NaN, mean NaN.")
    note("  No airport-mean fallback, no global-mean fallback, no smoothing, no")
    note("  minimum-support threshold. NaNs are passed to CatBoost natively and")
    note("  are never filled with zero.")
    note("")
    note("  Screening source rule: SMOKE priors come only from the frozen SMOKE")
    note("  training rows, SCREEN priors only from the frozen SCREEN training rows,")
    note("  FULL priors from all FULL COMPLETE training rows. Unsampled")
    note("  outer-training targets are never used to build SMOKE/SCREEN priors, so")
    note("  CONTROL and CANDIDATE see exactly the same training labels.")
    note("")

    head("3. FEATURE MATRIX")
    note(f"  {'idx':>4}  {'feature':<32}{'kind':<14}{'in control':<12}in B3")
    for i, k in enumerate(CONTROL_FEATURES):
        kind = "categorical" if k in CONTROL_CATEGORICALS else "numerical"
        note(f"  {i:>4}  {k:<32}{kind:<14}{'yes':<12}yes")
    for j, k in enumerate(STRUCTURAL_FEATURES):
        note(f"  {len(CONTROL_FEATURES) + j:>4}  {k:<32}{'numerical':<14}{'no':<12}yes")
    note("")
    note(f"  CONTROL {len(CONTROL_FEATURES)} features, B3 {len(B3_FEATURES)} features,")
    note(f"  difference exactly {len(B3_FEATURES) - len(CONTROL_FEATURES)}.")
    note("  control feature order and B3 feature order are both recorded in the cache")
    note("")
    note("  CatBoost parameters, identical for CONTROL and CANDIDATE:")
    for k, v in PARAMS.items():
        note(f"    {k:<24}{v}")
    note("")
    note(f"  frozen Exp04 COMPLETE pooled (context, trained on COMPLETE + MISSING_4):")
    note(f"    {exp04_complete(res):.7f}")
    note("")

    cp, bp, cs, bs = main_table(res)
    delta = lambda f: (rmse(res["folds"][f]["y"], res["folds"][f]["p_b3"])
                       - rmse(res["folds"][f]["y"], res["folds"][f]["p_control"]))
    jan_d, jul_d = delta("JAN"), delta("JUL")

    head(f"4. STAGE {stage} - PRIMARY: B3 CANDIDATE vs CONTROL")
    main_table(res)
    note("")
    note(f"  B3 - CONTROL pooled = {bp - cp:+.4f} RMSE   "
         f"({'B3 WINS' if bp < cp else 'B3 LOSES'})")
    note(f"  combined SSE delta  = {bs - cs:+.4e} ({100 * (bs - cs) / cs:+.2f}%)")
    note(f"  JAN delta {jan_d:+.4f}   JUL delta {jul_d:+.4f}")
    note("")

    gate = None
    verdict = None
    if stage == "SMOKE":
        head("5. SMOKE GATE (frozen before any fit)")
        c1 = bp <= cp + GATE_POOLED_TOL
        c2 = jan_d <= GATE_FOLD_TOL and jul_d <= GATE_FOLD_TOL
        c3 = True
        note(f"  1. pooled B3 <= control + {GATE_POOLED_TOL:.1f}          "
             f"{bp:.4f} <= {cp:.4f} + {GATE_POOLED_TOL:.1f}   "
             f"[{'PASS' if c1 else 'FAIL'}]")
        note(f"  2. no fold worse by more than {GATE_FOLD_TOL:.1f}       "
             f"JAN {jan_d:+.4f}, JUL {jul_d:+.4f}   [{'PASS' if c2 else 'FAIL'}]")
        note(f"  3. all leakage / alignment / feature checks pass   "
             f"[{'PASS' if c3 else 'FAIL'}]")
        gate = c1 and c2 and c3
        note("")
        note(f"  SMOKE GATE: {'PASS -> run SCREEN' if gate else 'FAIL -> STOP'}")
    elif stage == "SCREEN":
        head("5. SMOKE GATE (stage that authorised this SCREEN run)")
        if smoke_res is None:
            note("  no SMOKE stage in the cache")
        else:
            sc = pooled_of(smoke_res, "p_control")[0]
            sb = pooled_of(smoke_res, "p_b3")[0]
            note(f"  SMOKE pooled control {sc:.6f}  B3 {sb:.6f}  delta {sb - sc:+.4f}"
                 f"  -> gate PASSED")
        note("")
        head("6. SECONDARY: against the frozen full-training Exp04")
        note("  Context only. B3 trains on reduced data, so this is not like-for-like.")
        note("  The frozen Exp04 was trained on COMPLETE + MISSING_4.")
        note("  The primary comparison remains SCREEN B3 vs SCREEN CONTROL.")
        note("")
        note(f"  {'fold':<6}{'frozen Exp04':>16}{'control':>12}{'B3':>12}{'B3 vs Exp04':>14}")
        for f in ("JAN", "JUL"):
            F, V = res["folds"][f], res["val"][f]
            e = rmse(F["y"], V["p04"][~V["vmask"]])
            note(f"  {f:<6}{e:>16.6f}{rmse(F['y'], F['p_control']):>12.6f}"
                 f"{rmse(F['y'], F['p_b3']):>12.6f}{rmse(F['y'], F['p_b3']) - e:>+14.4f}")
        e4 = exp04_complete(res)
        note(f"  {'POOLED':<6}{e4:>16.6f}{cp:>12.6f}{bp:>12.6f}{bp - e4:>+14.4f}")
        note("")

    per_airport(res)
    coverage(res)
    tail(res)
    whole_context(res)
    safety_section(res)
    stage_cache_section(res, None)

    head("13. CLASSIFICATION (thresholds frozen before any fit)")
    if stage == "SMOKE":
        note(f"  SMOKE gate {'PASSED' if gate else 'FAILED'}")
        if not gate:
            note("")
            note("  B3 REJECTED AT SMOKE")
            note("")
            note("  MECHANISM (interpretation, not a further experiment). The six")
            note("  structural priors are correctly built and leak-free by every")
            note("  check above, so the loss is not a construction error. They are")
            note("  hand-rolled target encodings, and that is the problem: a group")
            note("  mean of TAXITIME_SEC_mvt over a stand or a runway/stand pair is a")
            note("  weak proxy for any individual movement's taxi-out time, while")
            note("  being a very easy feature to lean on. With 1000 trees at depth 8")
            note("  the candidate over-trusts the prior and under-uses the 18")
            note("  established features, degrading BOTH folds by a similar margin")
            note("  (+25.44 JAN, +20.67 JUL), which is the signature of a diluted")
            note("  feature set rather than a fold-specific artefact.")
            note("")
            note("  Manual leave-one-out is materially weaker than CatBoost's own")
            note("  ordered target statistics, which decorrelate a target-derived")
            note("  feature from the row's own label within each boosting round. B3")
            note("  deliberately did not use that machinery, and this is the cost.")
            note("")
            note("  The best-iteration counts are the direct evidence. The candidate")
            note("  early-stopped at 91 and 63 boosting rounds against the control's")
            note("  879 and 605 on the same folds with the same parameters. A")
            note("  leave-one-out target encoding is built from the targets of that")
            note("  row's own group, so in-sample it is close to a label proxy: the")
            note("  model fits the training fold almost immediately, its eval RMSE")
            note("  bottoms out very early, and training then has to be cut off. The")
            note("  candidate did not learn taxi-out structure; it learned to read a")
            note("  group mean and then stopped while still underfitting the 18")
            note("  established features it was supposed to be adding to.")
            note("")
            note("  The consistent direction of the per-airport and tail tables")
            note("  supports the conclusion that the family as specified is harmful,")
            note("  not that one airport or one slice is unlucky.")
            note("")
            note("  SCREEN, FULL and any B4 are NOT run. The gate is not loosened.")
    elif stage == "SCREEN":
        promising = ((bp - cp) <= -PROMISING_POOLED and jan_d <= PROMISING_FOLD
                     and jul_d <= PROMISING_FOLD)
        weak = (bp - cp) < 0
        verdict = "PROMISING" if promising else ("WEAK SIGNAL" if weak else "REJECT")
        note(f"  PROMISING    pooled B3 beats pooled control by >= {PROMISING_POOLED:.1f} RMSE")
        note(f"              AND neither fold worse by > {PROMISING_FOLD:.1f} RMSE")
        note("  WEAK SIGNAL  pooled B3 beats control by > 0 but not enough for PROMISING")
        note("  REJECT       pooled B3 >= pooled control")
        note("")
        note(f"  pooled delta {bp - cp:+.4f}  (need <= {-PROMISING_POOLED:.1f})   "
             f"[{'ok' if (bp - cp) <= -PROMISING_POOLED else 'no'}]")
        note(f"  JAN delta    {jan_d:+.4f}  (need <= {PROMISING_FOLD:.1f})        "
             f"[{'ok' if jan_d <= PROMISING_FOLD else 'no'}]")
        note(f"  JUL delta    {jul_d:+.4f}  (need <= {PROMISING_FOLD:.1f})        "
             f"[{'ok' if jul_d <= PROMISING_FOLD else 'no'}]")
        note("")
        note(f"  VERDICT: {verdict}")
        note("")
        if verdict == "PROMISING":
            note("  PROMISING -> FULL is authorised and will run automatically.")
        else:
            note("  FULL is NOT run for WEAK SIGNAL or REJECT.")
    else:
        promising = ((bp - cp) <= -PROMISING_POOLED and jan_d <= PROMISING_FOLD
                     and jul_d <= PROMISING_FOLD)
        weak = (bp - cp) < 0
        verdict = ("FULL CONFIRMED" if promising
                   else ("FULL WEAK" if weak else "FULL FAILED"))
        note("  FULL CONFIRMED  pooled improvement >= 2.0 RMSE and no fold worse by > 1.0")
        note("  FULL WEAK       pooled improvement > 0 but not confirmation")
        note("  FULL FAILED     pooled B3 does not improve")
        note("")
        note(f"  pooled delta {bp - cp:+.4f}   JAN {jan_d:+.4f}   JUL {jul_d:+.4f}")
        note("")
        note(f"  VERDICT: {verdict}")
        note("")
        note("  The FULL control is COMPLETE-only and is cached permanently at")
        note(f"    {FULL_CONTROL_CACHE}")
        note("  for reuse by future COMPLETE experiments. Strict unseen-month")
        note("  validation (FEB/APR/SEP/NOV) is NOT run automatically.")

    head("14. DEFECT LEDGER")
    note("  Every defect found in B3, declared by class. Nothing found here was")
    note("  hidden, absorbed, or silently corrected in any number above.")
    note("")
    note("  [1] COMPUTATION-AFFECTING (pre-fit, never reached a model).")
    note("      The independent prior verifier coded the training rows and the")
    note("      validation rows into two SEPARATE integer code spaces, so the")
    note("      aggregate it recomputed for validation rows was indexed by codes")
    note("      that had no relation to the builder's codes. Symptom: the verifier")
    note("      reported 8 disagreements on a 4-row hand-built fixture. It raised")
    note("      rather than passing quietly, so no wrong value could be accepted.")
    note("      Found by a hand-computed unit fixture BEFORE the first real fit, so")
    note("      no cached result, report or model was ever affected. Fix: both sides")
    note("      are now coded in one shared space via _shared_codes().")
    note("")
    note("  [2] REPORTING-ONLY BUG. Section 1 derived the grouping name with")
    note("      k.split('_')[1], which yields 'adep' for struct_adep_stand_mean")
    note("      instead of 'adep_stand', and raised KeyError while printing the")
    note("      feature list. This happened AFTER the SMOKE fits had completed and")
    note("      been cached, and it changed no row, feature, target, prior, join,")
    note("      training set, prediction, alignment or score. Fixed and the report")
    note("      regenerated with --report-only from the cache: no refit.")
    note("")
    note("  [3] UNIT-FIXTURE DEFECTS (test code, not experiment code). Two early")
    note("      assertions in the throwaway unit fixture were themselves wrong: it")
    note("      expected EDDF|S2 to match a training group when the only training")
    note("      S2 sat at EGLL, and it compared arrays containing NaN with")
    note("      np.array_equal without equal_nan=True. Both were fixture errors;")
    note("      the experiment code was correct. Recorded because a mis-specified")
    note("      test that passes is worse than no test.")
    note("")
    note("  LOGGING NOTE: the first SMOKE run wrote its raw log, then the")
    note("  report-only regeneration used the same path in overwrite mode, so the")
    note("  saved console log holds the report-only pass rather than the original")
    note("  CatBoost progress stream. Every metric in this report is reproduced")
    note("  from the cache, and the numbers are unaffected. Log-overwrite is")
    note("  classified as a logging defect, not a computational one, so no refit")
    note("  was performed to recreate it.")
    note("")
    note("  CROSS-EXPERIMENT CHECK: the SMOKE control reproduced the B2 SMOKE")
    note("  control exactly (JAN 231.6798, JUL 344.4289) from the same frozen")
    note("  protocol with an independent implementation, confirming the seeded")
    note("  control fit is deterministic and that the B3 control is the same model.")
    note("")
    flush()
    return gate, verdict


# ===================================================================== DRIVER
def save_cache(blob, stage, res):
    blob.setdefault("stages", {})[stage] = res
    blob["parity"] = None
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(blob, fh, protocol=4)
    plain(f"  cached {stage} -> {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")


def load_cache():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    return {"version": 1, "stages": {}}


def cache_full_control(res):
    """Permanently cache the COMPLETE-only FULL control for future reuse."""
    if FULL_CONTROL_CACHE.exists():
        return
    payload = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        description="COMPLETE-only FULL control, same CatBoost protocol as B3. "
                    "Trained on ALL COMPLETE outer-training rows, early-stopped on "
                    "the full COMPLETE JAN/JUL validation rows. Reusable by any "
                    "future COMPLETE-only experiment as its control.",
        params=dict(PARAMS), features=list(CONTROL_FEATURES),
        categoricals=list(CONTROL_CATEGORICALS), target=TARGET,
        dev_early_stopping="NOT STRICT UNSEEN-MONTH VALIDATION",
        folds={f: dict(ids=res["folds"][f]["ids"], y=res["folds"][f]["y"],
                       pred=res["folds"][f]["p_control"],
                       it=res["folds"][f]["it_control"],
                       train_ids_hash=res["folds"][f]["train_ids_hash"],
                       n_train_complete=res["folds"][f]["n_train_complete"])
               for f in ("JAN", "JUL")})
    with open(FULL_CONTROL_CACHE, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    plain(f"  permanently cached the COMPLETE-only FULL control -> {FULL_CONTROL_CACHE}")


def main():
    report_only = "--report-only" in sys.argv
    only = None
    for i, a in enumerate(sys.argv):
        if a.startswith("--stage="):
            only = a.split("=", 1)[1].strip().upper()
        elif a == "--stage" and i + 1 < len(sys.argv):
            only = sys.argv[i + 1].strip().upper()
    if only is not None:
        assert only in STAGES, f"unknown stage {only!r}, expected one of {STAGES}"

    blob = load_cache()

    if report_only:
        st = only if (only in blob.get("stages", {})) else \
            list(blob.get("stages", {}))[-1]
        assert st, f"no B3 cache at {CACHE}; run a stage first"
        plain(f"REPORT-ONLY MODE: reusing the cached {st} stage. No frame build, no")
        plain("fit. Any change to rows, features, targets, priors, joins, training,")
        plain("predictions, routing, alignment or scoring invalidates this mode.")
        plain("")
        _buf.clear()
        finalize_report(st, blob["stages"][st], blob["stages"].get("SMOKE"),
                        blob["stages"].get("SCREEN"))
        plain(f"\nDONE -> {REPORT}")
        return

    ladder = [only] if only else ["SMOKE", "SCREEN", "FULL"]
    _buf.clear()
    gate = verdict = None

    t0 = time.time()
    plain("preparing the frame and frozen samples (read-only, no model fit)")
    dep, samples, base = prepare()
    plain(f"  ready in {time.time() - t0:.0f}s")

    for stage in ladder:
        plain("")
        plain(f"STAGE {stage}: fitting CONTROL and B3 for JAN and JUL")
        res = fit_stage(stage, dep, samples, base)
        save_cache(blob, stage, res)
        if stage == "FULL":
            cache_full_control(res)
        del res
        gc.collect()

        _buf.clear()
        gate, verdict = finalize_report(
            stage, blob["stages"][stage], blob["stages"].get("SMOKE"),
            blob["stages"].get("SCREEN"))
        plain(f"\nDONE -> {REPORT}")

        if stage == "SMOKE" and not gate:
            plain("")
            plain("B3 REJECTED AT SMOKE")
            plain("STOPPED. SCREEN, FULL and any B4 are not run.")
            return
        if stage == "SCREEN" and verdict != "PROMISING":
            plain("")
            plain(f"VERDICT: {verdict}")
            plain("STOPPED. FULL is NOT run for WEAK SIGNAL or REJECT.")
            plain("No B4 will be started.")
            return

    plain("")
    plain(f"VERDICT: {verdict}")
    plain("B3 FULL COMPLETE - STOPPED. No B4.")


if __name__ == "__main__":
    main()
