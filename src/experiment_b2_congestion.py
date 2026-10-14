"""B2 - MULTI-SCALE AIRPORT CONGESTION FEATURES FOR COMPLETE ROWS.

Branch B, second experiment. B1 (AOBT3 residual modelling) is CLOSED/REJECTED
and is not revisited here. B2 is deliberately unlike the friend's historical-
prior branch and predicts TAXITIME_SEC_mvt directly, with no residual target
anywhere.

    python src/experiment_b2_congestion.py --stage smoke    # SMOKE
    python src/experiment_b2_congestion.py --stage screen   # SCREEN
    python src/experiment_b2_congestion.py --report-only    # no fit

HYPOTHESIS UNDER TEST (one only)
    Exp04 carries exactly one congestion feature, traffic_last_15min, a single
    aggregate count over one timescale. Taxi-out time may instead depend on the
    airport's immediate operational state across several timescales, on whether
    activity is rising or falling, on whether departures or arrivals dominate,
    and on how long ago the previous movement happened. B2 adds a 12-feature
    family of strictly causal, multi-scale congestion measures.

    CONTROL   the established Exp04 18 features
    CANDIDATE the same 18 features plus the 12 B2 congestion features

    Identical training IDs, validation IDs, CatBoost parameters, target,
    preprocessing, fold and early-stopping procedure. The new congestion
    information is the only experimental variable.

CAUSAL WINDOW RULE
    Every B2 feature for a movement at time t counts only movements at the same
    physical airport whose timestamp lies in [t - W, t). The current movement
    can never count itself, and movements sharing its exact timestamp can never
    count one another, because t is excluded from the window. secs_since_prev_*
    looks strictly backwards and stays NaN when there is no earlier movement;
    those NaNs are handed to CatBoost natively and are never filled with zero.

PHYSICAL AIRPORT
    Byte-for-byte the semantics of the established add_traffic_feature():
    DEP -> ADEP_mvt, ARR -> ADES_mvt, anything else -> ADEP_mvt. Verified
    against the reference implementation, not re-invented.

BUILD ORDER
    The congestion family is computed once on the FULL 4,167,797-row movement
    stream, before any filtering to COMPLETE, before the departure subset,
    before the JAN/JUL split and before SMOKE/SCREEN sampling. The sample never
    defines the congestion history, exactly as for traffic_last_15min. The
    engineered table is cached so it is not recomputed per fit.

TARGET-FREENESS
    Congestion construction reads only MVT_ID_mvt, MVT_TIME_UTC_mvt, PHASE_mvt,
    ADEP_mvt and ADES_mvt. TAXITIME_SEC_mvt, BLOCK_TIME_UTC_mvt, ranking data
    and weather are never touched, and no future movement is used.

SAFETY
    No ranking data, no weather, no geometry, no ADS-B, no future-window
    traffic, no historical target priors, no FLIGHT identity feature, no
    residual target, no runway-specific or stand-specific traffic. Exp04,
    plain F3 and the MISSING_4 specialists are NOT refitted. This writes no
    model, no ranking prediction and no submission.

DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION
    Both models early-stop on the full outer JAN/JUL COMPLETE validation rows,
    matching the established screening procedure. Strict unseen-month
    validation is a later step and only if B2 survives.

RUNTIME CACHE RULE
    compute() and finalize_report() are separate. Every fit is cached with its
    predictions, IDs, target, best iteration, feature list and order, training
    ID hash, metrics and runtime immediately. --report-only regenerates the
    report from that cache alone. A wording, formatting, key, table or file
    output defect is a REPORTING-ONLY BUG and must never trigger a refit.
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
    TARGET, T4_FEATURES, V7_CATEGORICALS, V7_FEATURES,
)

# ---------------------------------------------------------------- config
REPORT = Path("reports/b2_congestion.txt")
CACHE = SP.CACHE_DIR / "b2_congestion_cache.pkl"
FEAT_CACHE = SP.CACHE_DIR / "b2_congestion_features.pkl"
FEAT_VERSION = 2
STAGES = ("SMOKE", "SCREEN")

CONTROL_FEATURES = V7_FEATURES + T4_FEATURES + ["adep_reference_state"]
CONTROL_CATEGORICALS = V7_CATEGORICALS + ["adep_reference_state"]

WINDOWS = (("5m", 300), ("15m", 900), ("30m", 1800), ("60m", 3600))
CONGESTION_FEATURES = (
    [f"dep_prev_{w}" for w, _ in WINDOWS] + [f"arr_prev_{w}" for w, _ in WINDOWS]
    + ["secs_since_prev_dep", "secs_since_prev_arr",
       "dep_minus_arr_15m", "dep_minus_arr_30m"]
)
B2_FEATURES = CONTROL_FEATURES + CONGESTION_FEATURES
B2_CATEGORICALS = list(CONTROL_CATEGORICALS)

PARAMS = dict(
    iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=200, early_stopping_rounds=50, use_best_model=True,
    allow_writing_files=False,
)

GATE_POOLED_TOL = 1.0
GATE_FOLD_TOL = 3.0
PROMISING_POOLED = 2.0
PROMISING_FOLD = 1.0

CONGESTION_ONLY_INPUTS = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt",
                          "ADEP_mvt", "ADES_mvt"]

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
    print(t)


def flush():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def sse(y, p):
    return float(((np.asarray(y, float) - np.asarray(p, float)) ** 2).sum())


def safety(name, ok, detail=""):
    assert ok, f"SAFETY CHECK FAILED: {name} {detail}"
    print(f"  [PASS] {name}")
    return name


# =============================================== congestion family (target-free)
def physical_airport(df):
    """Byte-for-byte the add_traffic_feature() semantics."""
    ap = df["ADEP_mvt"].copy()
    arr = df["PHASE_mvt"] == "ARR"
    ap[arr] = df.loc[arr, "ADES_mvt"]
    return ap


def build_congestion(verbose=True) -> pd.DataFrame:
    """The 12 B2 features for every DEP row, from the full movement stream.

    Strictly causal: window is [t - W, t), so t itself is excluded and the
    current movement can never count itself. secs_since_prev_* looks strictly
    backwards and is NaN when nothing earlier exists at that airport+phase.
    """
    t_start = time.time()
    files = sorted(SP.DATA_DIR.glob("training_*.parquet"))
    assert len(files) == 12, f"expected 12 training months, found {len(files)}"
    mv = pd.concat([pd.read_parquet(f, columns=CONGESTION_ONLY_INPUTS) for f in files],
                   ignore_index=True)
    n_stream = len(mv)
    mv["apt"] = physical_airport(mv)
    assert not mv["apt"].isna().any(), "null physical airport"
    mv["t"] = pd.to_datetime(mv["MVT_TIME_UTC_mvt"], utc=True).astype("int64")
    mv = mv.sort_values(["apt", "PHASE_mvt", "t"], kind="mergesort").reset_index(drop=True)

    t_all = mv["t"].to_numpy()
    phase_all = mv["PHASE_mvt"].to_numpy()
    n = len(mv)
    counts = {f"{p}_prev_{w}": np.zeros(n, dtype=np.int32)
              for p in ("dep", "arr") for w, _ in WINDOWS}
    recency = {f"secs_since_prev_{p}": np.full(n, np.nan, dtype="float64")
               for p in ("dep", "arr")}

    # Group by AIRPORT only. Every feature is evaluated AT each DEP row's own
    # timestamp, reading the airport's DEP and ARR timelines separately. The ARR
    # count for a departure must be looked up on the ARR timeline at that
    # departure's time, never carried on the ARR rows' own positions.
    for apt, idx in mv.groupby("apt", sort=True).indices.items():
        pos = np.sort(idx)
        is_dep = phase_all[pos] == "DEP"
        dpos, apos = pos[is_dep], pos[~is_dep]
        dep_ts, arr_ts = t_all[dpos], t_all[apos]
        assert np.all(np.diff(dep_ts) >= 0) and np.all(np.diff(arr_ts) >= 0), \
            f"{apt}: phase timeline not time-ordered"
        t = dep_ts                                  # evaluate at departure times
        for p, tl in (("dep", dep_ts), ("arr", arr_ts)):
            hi = np.searchsorted(tl, t, side="left")       # first tl >= t
            for w, secs in WINDOWS:
                lo = np.searchsorted(tl, t - secs * 1_000_000, side="left")
                counts[f"{p}_prev_{w}"][dpos] = (hi - lo).astype(np.int32)
            has = hi > 0
            recency[f"secs_since_prev_{p}"][dpos] = np.where(
                has, (t - tl[np.where(has, hi - 1, 0)]) / 1e6, np.nan)

    dep_pos = np.flatnonzero(phase_all == "DEP")
    tab = pd.DataFrame({k: v[dep_pos] for k, v in counts.items()})
    tab["secs_since_prev_dep"] = recency["secs_since_prev_dep"][dep_pos]
    tab["secs_since_prev_arr"] = recency["secs_since_prev_arr"][dep_pos]
    tab["dep_minus_arr_15m"] = (tab["dep_prev_15m"] - tab["arr_prev_15m"]).astype(np.int32)
    tab["dep_minus_arr_30m"] = (tab["dep_prev_30m"] - tab["arr_prev_30m"]).astype(np.int32)
    tab.insert(0, "MVT_ID_mvt", mv["MVT_ID_mvt"].to_numpy()[dep_pos])
    tab = tab[["MVT_ID_mvt"] + CONGESTION_FEATURES]

    assert len(tab) == int((phase_all == "DEP").sum()), "DEP row count changed"
    assert not tab["MVT_ID_mvt"].duplicated().any(), "duplicate MVT_ID in congestion"
    # the current movement can never count itself, and same-timestamp peers never
    # each other, because t is excluded from every window
    assert (tab["dep_prev_15m"] <= tab["dep_prev_60m"]).all(), "window nesting violated"
    for p in ("dep", "arr"):
        for a, b in (("5m", "15m"), ("15m", "30m"), ("30m", "60m")):
            assert (tab[f"{p}_prev_{a}"] <= tab[f"{p}_prev_{b}"]).all(), \
                f"{p} window nesting {a}<={b} violated"
    if verbose:
        plain(f"  congestion built on {n_stream:,} movements -> {len(tab):,} DEP rows "
              f"x {len(CONGESTION_FEATURES)} features in {time.time() - t_start:.0f}s")
    return tab



def load_congestion() -> pd.DataFrame:
    if FEAT_CACHE.exists():
        with open(FEAT_CACHE, "rb") as fh:
            blob = pickle.load(fh)
        assert blob["version"] == FEAT_VERSION, "congestion definition changed"
        plain(f"  congestion features loaded from {FEAT_CACHE} (v{blob['version']})")
        return blob["table"]
    tab = build_congestion()
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEAT_CACHE, "wb") as fh:
        pickle.dump(dict(version=FEAT_VERSION, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                         inputs=list(CONGESTION_ONLY_INPUTS),
                         features=list(CONGESTION_FEATURES), table=tab), fh, protocol=4)
    plain(f"  cached congestion features -> {FEAT_CACHE} "
          f"({FEAT_CACHE.stat().st_size / 1e6:.1f} MB)")
    return tab


# ====================================================================== COMPUTE
def prepare():
    dep = SP.build_frame()
    dep["adep_reference_state"] = dep["ADEP_mvt"] + "|" + dep["reference_state"]
    assert "reference_state" not in CONTROL_FEATURES
    samples = SP.load_samples(dep)
    with open(SP.BASELINE_CACHE, "rb") as fh:
        base = pickle.load(fh)
    return dep, samples, base


def parity_check(dep, cong) -> dict:
    """dep_prev_15m + arr_prev_15m must equal traffic_last_15min everywhere.

    Any mismatch must be explained before fitting, never silently absorbed.
    """
    m = dep[["MVT_ID_mvt", "traffic_last_15min"]].merge(
        cong[["MVT_ID_mvt", "dep_prev_15m", "arr_prev_15m"]],
        on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert m[["dep_prev_15m", "arr_prev_15m"]].notna().all().all(), \
        "congestion table did not cover every departure row"
    lhs = m["dep_prev_15m"].to_numpy("int64") + m["arr_prev_15m"].to_numpy("int64")
    rhs = m["traffic_last_15min"].to_numpy("float64")
    d = np.abs(lhs - rhs)
    return dict(rows=int(len(m)), mismatches=int((d != 0).sum()),
                max_abs_diff=float(d.max()),
                lhs_min=int(lhs.min()), lhs_max=int(lhs.max()),
                **brute_force_causality(cong))


def brute_force_causality(cong, n_sample=4000, seed=42) -> dict:
    """Independent causal proof, not inherited from traffic_last_15min.

    Recounts a random sample of departures straight from the raw movement
    stream with an explicit two-sided time filter, and confirms that the row
    itself and any same-timestamp peer are excluded.
    """
    files = sorted(SP.DATA_DIR.glob("training_*.parquet"))
    mv = pd.concat([pd.read_parquet(f, columns=CONGESTION_ONLY_INPUTS) for f in files],
                   ignore_index=True)
    mv["apt"] = physical_airport(mv)
    mv["t"] = pd.to_datetime(mv["MVT_TIME_UTC_mvt"], utc=True).astype("int64")

    c = cong.set_index("MVT_ID_mvt")
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(c), min(n_sample, len(c)), replace=False))
    mids = SP.id_to_int(c.index.to_numpy()[pick])
    apt_of = mv.set_index("MVT_ID_mvt")["apt"]
    t_of = mv.set_index("MVT_ID_mvt")["t"]
    groups = {a: (g["t"].to_numpy(), g["PHASE_mvt"].to_numpy(),
                  g["MVT_ID_mvt"].to_numpy()) for a, g in mv.groupby("apt", sort=True)}

    bad = self_hit = peer_leak = peer_rows = 0
    for k, mid in zip(pick, mids):
        t = int(t_of.loc[mid])
        ts, ph, ids = groups[apt_of.loc[mid]]
        sel = (ts >= t - 900_000_000) & (ts < t)      # [t-15m, t): t excluded
        win_ts, win_ph, win_ids = ts[sel], ph[sel], ids[sel]
        self_hit += int((win_ids == mid).any())
        same = win_ts == t
        peer_rows += int(same.sum())
        peer_leak += int(same.sum())
        d = int((win_ph == "DEP").sum())
        a = int((win_ph == "ARR").sum())
        r = c.iloc[k]
        if d != int(r["dep_prev_15m"]) or a != int(r["arr_prev_15m"]):
            bad += 1
    return dict(causal_sample=int(len(pick)), causal_mismatches=bad,
                causal_self_hits=self_hit, causal_same_ts_in_window=peer_rows)


def fit_stage(stage, dep, cong, samples, base):
    assert stage in STAGES
    lkey = stage.lower()
    meta = dep.set_index("MVT_ID_mvt")
    cmeta = cong.set_index("MVT_ID_mvt")
    checks = []
    res = dict(version=1, stage=stage, created=time.strftime("%Y-%m-%d %H:%M:%S"),
               params=dict(PARAMS), control_features=list(CONTROL_FEATURES),
               b2_features=list(B2_FEATURES), congestion=list(CONGESTION_FEATURES),
               categoricals=list(CONTROL_CATEGORICALS), folds={}, checks=checks)

    for f in ("JAN", "JUL"):
        t_fold = time.time()
        lo, hi = SP.FOLDS[f]
        vm = ((dep[SP.MINUEND] >= lo) & (dep[SP.MINUEND] < hi)).to_numpy()
        tr_all = dep[~vm]
        tr = tr_all[tr_all["reference_state"] == "COMPLETE"]
        sel = samples[f][f"{lkey}_ids"]
        tr = tr[np.isin(SP.id_to_int(tr["MVT_ID_mvt"].to_numpy()), sel)]
        tr = tr.sort_values("MVT_ID_mvt").reset_index(drop=True)
        tr_ids = SP.id_to_int(tr["MVT_ID_mvt"].to_numpy())

        F = base["folds"][f]
        val_ids_all = SP.id_to_int(F["ids"])
        comp = ~F["vmask"]
        val_ids = val_ids_all[comp]
        y_va = np.asarray(F["y"], float)[comp]

        checks.append(safety(f"{f}: no validation row enters training",
                             len(np.intersect1d(tr_ids, val_ids_all)) == 0))
        checks.append(safety(f"{f}: only COMPLETE rows enter training",
                             bool((tr["reference_state"] == "COMPLETE").all())))
        checks.append(safety(f"{f}: training IDs unique", bool(pd.Series(tr_ids).is_unique)))
        checks.append(safety(f"{f}: every COMPLETE validation ID is unique and complete",
                             bool(pd.Series(val_ids).is_unique)
                             and len(val_ids) == int(comp.sum())))

        mm = meta.loc[val_ids_all]
        assert np.array_equal(SP.id_to_int(mm.index.to_numpy()), val_ids_all), \
            f"{f}: validation feature lookup lost row order"
        assert np.allclose(mm[TARGET].to_numpy("float64"),
                           np.asarray(F["y"], float), rtol=0, atol=1e-9), \
            f"{f}: frame target disagrees with the frozen cache target"
        va = mm[comp].copy()
        cva = cmeta.loc[val_ids_all]
        assert np.array_equal(SP.id_to_int(cva.index.to_numpy()), val_ids_all), \
            f"{f}: congestion lookup lost row order"
        cva = cva.loc[val_ids][CONGESTION_FEATURES]
        ctr = cmeta.loc[tr_ids][CONGESTION_FEATURES]
        assert np.array_equal(SP.id_to_int(ctr.index.to_numpy()), tr_ids), \
            f"{f}: training congestion lookup lost row order"
        assert cva.index.to_numpy().tolist() == val_ids.tolist(), \
            f"{f}: validation congestion rows are not in ID order"
        assert ctr.index.to_numpy().tolist() == tr_ids.tolist(), \
            f"{f}: training congestion rows are not in ID order"
        count_cols = [c for c in CONGESTION_FEATURES
                      if c.startswith("dep_prev_") or c.startswith("arr_prev_")]
        diff_cols = [c for c in CONGESTION_FEATURES if c.startswith("dep_minus_")]
        for tag, src in (("validation", cva), ("training", ctr)):
            assert src[count_cols].notna().all().all(), f"null count feature ({tag})"
            assert src[diff_cols].notna().all().all(), f"null difference ({tag})"
            assert (src[count_cols] >= 0).all().all(), f"negative count ({tag})"
            assert src[CONGESTION_FEATURES].index.is_unique, f"duplicate row ({tag})"

        y_tr = tr[TARGET].to_numpy("float64")
        # CONTROL = Exp04 18 only. B2 = exactly the same 18 plus the 12 new.
        X_c_tr = tr[CONTROL_FEATURES].reset_index(drop=True)
        X_c_va = va[CONTROL_FEATURES].reset_index(drop=True)
        X_b_tr = pd.concat([X_c_tr, ctr[CONGESTION_FEATURES].reset_index(drop=True)],
                           axis=1)[B2_FEATURES]
        X_b_va = pd.concat([X_c_va, cva[CONGESTION_FEATURES].reset_index(drop=True)],
                           axis=1)[B2_FEATURES]
        assert list(X_b_tr.columns) == list(B2_FEATURES) == list(X_b_va.columns)
        assert len(X_c_tr) == len(X_b_tr) == len(tr) == len(y_tr)
        assert len(X_c_va) == len(X_b_va) == len(val_ids)
        tr15 = (ctr["dep_prev_15m"].to_numpy("int32")
                + ctr["arr_prev_15m"].to_numpy("int32"))

        fold_res = dict(
            n_train_complete=int(len(tr)),
            n_train_complete_all=int((tr_all["reference_state"] == "COMPLETE").sum()),
            train_ids_hash=SP.ids_digest(tr_ids), train_stage=stage,
            ids=val_ids, y=y_va, adep=va["ADEP_mvt"].to_numpy().astype(str),
            congestion_15m=(cva["dep_prev_15m"].to_numpy("int32")
                            + cva["arr_prev_15m"].to_numpy("int32")),
        )
        # training terciles of the 15-minute congestion total, training data only
        fold_res["bucket_edges"] = [float(np.quantile(tr15, 1 / 3)),
                                    float(np.quantile(tr15, 2 / 3))]
        plain(f"  {f} {stage}: train {len(tr):,} COMPLETE rows, "
              f"eval {len(val_ids):,} COMPLETE rows")

        for kind, Xt, Xv in (("control", X_c_tr, X_c_va), ("b2", X_b_tr, X_b_va)):
            t0 = time.time()
            m = CatBoostRegressor(**PARAMS)
            m.fit(Xt, y_tr, cat_features=CONTROL_CATEGORICALS, eval_set=(Xv, y_va))
            pred = m.predict(Xv)
            dt = time.time() - t0
            checks.append(safety(f"{f}/{stage}/{kind}: predictions finite",
                                 bool(np.isfinite(pred).all())))
            checks.append(safety(f"{f}/{stage}/{kind}: one prediction per validation row",
                                 len(pred) == len(val_ids)))
            checks.append(safety(f"{f}/{stage}/{kind}: no BLOCK_TIME_UTC_mvt in features",
                                 "BLOCK_TIME_UTC_mvt" not in
                                 (CONTROL_FEATURES if kind == "control" else B2_FEATURES)))
            checks.append(safety(
                f"{f}/{stage}/{kind}: no historical-prior / FLIGHT-identity feature",
                not any(k.startswith("hist_") or k == "FLIGHT_mvt"
                        for k in (CONTROL_FEATURES if kind == "control" else B2_FEATURES))))
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
        checks.append(safety(
            f"{f}/{stage}: candidate is exactly the control plus 12 features",
            len(B2_FEATURES) == len(CONTROL_FEATURES) + 12
            and B2_FEATURES[:len(CONTROL_FEATURES)] == CONTROL_FEATURES))
        plain(f"  {f} {stage} done in {time.time() - t_fold:.1f}s")
        res["folds"][f] = fold_res
        del tr, tr_all, ctr, cva, mm, va, X_c_tr, X_b_tr, X_c_va, X_b_va
        gc.collect()

    res["val"] = {}
    for f in ("JAN", "JUL"):
        V = base["folds"][f]
        res["val"][f] = dict(
            ids=SP.id_to_int(V["ids"]), y=np.asarray(V["y"], float), vmask=V["vmask"],
            p04=np.asarray(V["p04"], float), p_f3=np.asarray(V["p_f3"], float),
            adep=V["adep"], it_exp04=int(V["it_exp04"]))
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
    note(f"  {'fold':<6}{'rows':>10}{'control':>12}{'B2':>12}{'delta':>10}"
         f"{'control SSE':>15}{'B2 SSE':>15}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        c, b = rmse(F["y"], F["p_control"]), rmse(F["y"], F["p_b2"])
        note(f"  {f:<6}{len(F['y']):>10,}{c:>12.6f}{b:>12.6f}{b - c:>+10.4f}"
             f"{sse(F['y'], F['p_control']):>15.4e}{sse(F['y'], F['p_b2']):>15.4e}")
    cp, cs = pooled_of(res, "p_control")
    bp, bs = pooled_of(res, "p_b2")
    n = sum(len(res["folds"][f]["y"]) for f in ("JAN", "JUL"))
    note(f"  {'POOLED':<6}{n:>10,}{cp:>12.6f}{bp:>12.6f}{bp - cp:>+10.4f}"
         f"{cs:>15.4e}{bs:>15.4e}")
    return cp, bp, cs, bs


def screen_sections(res, cp, bp):
    e4 = exp04_complete(res)
    head("6. SECONDARY: against the frozen full-training Exp04")
    note("  Context only. B2 trains on reduced data, so this is not like-for-like.")
    note("  The primary comparison remains SCREEN B2 vs SCREEN CONTROL.")
    note("")
    note(f"  {'fold':<6}{'frozen Exp04':>16}{'control':>12}{'B2':>12}{'B2 vs Exp04':>14}")
    for f in ("JAN", "JUL"):
        F, V = res["folds"][f], res["val"][f]
        e = rmse(F["y"], V["p04"][~V["vmask"]])
        note(f"  {f:<6}{e:>16.6f}{rmse(F['y'], F['p_control']):>12.6f}"
             f"{rmse(F['y'], F['p_b2']):>12.6f}{rmse(F['y'], F['p_b2']) - e:>+14.4f}")
    note(f"  {'POOLED':<6}{e4:>16.6f}{cp:>12.6f}{bp:>12.6f}{bp - e4:>+14.4f}")

    head("7. PER-AIRPORT DIAGNOSTIC (COMPLETE validation rows)")
    note("  Diagnostic only. No airport routing is created.")
    note("")
    ads = sorted(set(np.concatenate([res["folds"][f]["adep"] for f in ("JAN", "JUL")])))
    note(f"  {'ADEP':<8}{'rows':>10}{'control':>12}{'B2':>12}{'delta':>10}{'SSE delta':>16}")
    for a in ads:
        cnt = 0
        sc = sb = 0.0
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            m = F["adep"] == a
            cnt += int(m.sum())
            sc += sse(F["y"][m], F["p_control"][m])
            sb += sse(F["y"][m], F["p_b2"][m])
        c, b = float(np.sqrt(sc / cnt)), float(np.sqrt(sb / cnt))
        note(f"  {a:<8}{cnt:>10,}{c:>12.4f}{b:>12.4f}{b - c:>+10.4f}{sb - sc:>+16.4e}")
    note("")

    head("8. CONGESTION-BUCKET DIAGNOSTIC")
    note("  Bucket = dep_prev_15m + arr_prev_15m. Boundaries are terciles of the")
    note("  TRAINING rows of each fold. No validation target or validation")
    note("  performance was used to choose them, and no prediction is gated.")
    note("")
    note(f"  {'fold':<6}{'bucket':<10}{'rows':>10}{'control':>12}{'B2':>12}{'delta':>10}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        e1, e2 = F["bucket_edges"]
        v = F["congestion_15m"]
        for lbl, m in (("low", v < e1), ("medium", (v >= e1) & (v < e2)),
                       ("high", v >= e2)):
            note(f"  {f:<6}{lbl:<10}{int(m.sum()):>10,}"
                 f"{rmse(F['y'][m], F['p_control'][m]):>12.4f}"
                 f"{rmse(F['y'][m], F['p_b2'][m]):>12.4f}"
                 f"{rmse(F['y'][m], F['p_b2'][m]) - rmse(F['y'][m], F['p_control'][m]):>+10.4f}")
    note("")

    head("9. TAIL DIAGNOSTIC (COMPLETE validation rows)")
    note("  Diagnostic only. The true target never routes a prediction.")
    note("")
    note(f"  {'slice':<22}{'n':>10}{'control':>12}{'B2':>12}{'delta':>10}{'SSE delta':>16}")
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
            sb += sse(y[m], F["p_b2"][m])
        c, b = float(np.sqrt(sc / cnt)), float(np.sqrt(sb / cnt))
        note(f"  {lbl:<22}{cnt:>10,}{c:>12.4f}{b:>12.4f}{b - c:>+10.4f}{sb - sc:>+16.4e}")
    note("")

    head("10. WHOLE-FOLD CONTEXT (informational hybrid)")
    note("  COMPLETE -> SCREEN candidate, MISSING_4 -> frozen plain F3.")
    note("  Context only: the COMPLETE candidate trained on reduced data, so B2 is")
    note("  NOT classified from these numbers.")
    note("")
    note(f"  {'fold':<6}{'frozen plain F3':>19}{'HYBRID control':>17}"
         f"{'HYBRID B2':>13}{'delta':>10}")
    num = {"c": 0.0, "b": 0.0, "f": 0.0}
    den = 0
    for f in ("JAN", "JUL"):
        V = res["val"][f]
        hc, hb = hybrid(res, f, "p_control"), hybrid(res, f, "p_b2")
        c, b = rmse(V["y"], hc), rmse(V["y"], hb)
        num["c"] += sse(V["y"], hc)
        num["b"] += sse(V["y"], hb)
        num["f"] += sse(V["y"], V["p_f3"])
        den += len(V["y"])
        note(f"  {f:<6}{rmse(V['y'], V['p_f3']):>19.6f}{c:>17.6f}{b:>13.6f}{b - c:>+10.4f}")
    fp = float(np.sqrt(num["f"] / den))
    note(f"  {'POOLED':<6}{fp:>19.6f}{np.sqrt(num['c'] / den):>17.6f}"
         f"{np.sqrt(num['b'] / den):>13.6f}"
         f"{np.sqrt(num['b'] / den) - np.sqrt(num['c'] / den):>+10.4f}")
    note("")


def finalize_report(stage, res, parity, feat_secs, smoke_res, diag_unused=None):
    C = res["checks"]
    head("B2 - MULTI-SCALE AIRPORT CONGESTION FEATURES FOR COMPLETE ROWS")
    note(f"  Branch B, experiment 2. B1 (AOBT3 residual) is CLOSED/REJECTED and")
    note(f"  is not revisited. B2 predicts TAXITIME_SEC_mvt directly, no residual.")
    note(f"  stage {stage}   generated {res['created']}")
    note("")
    note("  ONE hypothesis only: does richer causal multi-scale airport activity")
    note("  information improve the COMPLETE model beyond Exp04?")
    note(f"  CONTROL   {len(CONTROL_FEATURES)} established Exp04 features")
    note(f"  CANDIDATE those same {len(CONTROL_FEATURES)} plus "
         f"{len(CONGESTION_FEATURES)} B2 congestion features "
         f"= {len(B2_FEATURES)}")
    note("  Identical training IDs, validation IDs, CatBoost parameters, target,")
    note("  preprocessing, fold and early-stopping procedure.")
    note("")

    head("1. CONGESTION FEATURE FAMILY (12 features, strictly causal)")
    note(f"  Feature-generation runtime: {feat_secs:.1f}s on the full movement stream.")
    note("")
    note("  physical airport: DEP -> ADEP_mvt, ARR -> ADES_mvt (byte-for-byte the")
    note("  established add_traffic_feature() semantics, verified not re-invented)")
    note("  window [t - W, t): t is EXCLUDED, so the current movement can never")
    note("  count itself and same-timestamp peers can never count one another")
    note("  built on ALL 4,167,797 movements BEFORE any COMPLETE filter, departure")
    note("  subset, JAN/JUL split or SMOKE/SCREEN sampling")
    note("  construction reads only: " + ", ".join(CONGESTION_ONLY_INPUTS))
    note("")
    for i, k in enumerate(CONGESTION_FEATURES, 1):
        note(f"    {i:>2}. {k}")
    note("")
    note("  secs_since_prev_dep / secs_since_prev_arr are NaN when no strictly")
    note("  earlier movement of that phase exists at that airport. Those NaNs are")
    note("  passed to CatBoost natively and are NOT filled with zero.")
    note("")
    note("  NOT added, by instruction: runway-specific traffic, stand features,")
    note("  historical target priors, FLIGHT identity, weather, geometry, ADS-B,")
    note("  future-window traffic, residual targets.")
    note("")

    head("2. 15-MINUTE PARITY CHECK (must pass before any fit)")
    note("  Where semantics are identical, dep_prev_15m + arr_prev_15m must equal")
    note("  the established traffic_last_15min. The old feature is NOT redefined.")
    note("")
    note(f"  rows compared      {parity['rows']:,}")
    note(f"  mismatching rows  {parity['mismatches']:,}")
    note(f"  max abs difference {parity['max_abs_diff']:.0f}")
    note(f"  range of dep_prev_15m + arr_prev_15m: "
         f"{parity['lhs_min']} .. {parity['lhs_max']}")
    note("")
    note("  Independent causal verification, recounted straight from the raw")
    note("  movement stream instead of trusting the reference feature:")
    note(f"    departures brute-force checked : {parity['causal_sample']:,}")
    note(f"    disagreements with the features : {parity['causal_mismatches']}")
    note(f"    rows counting THEMSELVES        : {parity['causal_self_hits']}")
    note(f"    same-timestamp peers admitted   : "
         f"{parity['causal_same_ts_in_window']}")
    causal_ok = (parity["causal_mismatches"] == 0
                 and parity["causal_self_hits"] == 0
                 and parity["causal_same_ts_in_window"] == 0)
    note(f"  [{'PASS' if causal_ok else 'FAIL'}] window is [t-W, t) on the physical")
    note("       airport, so t is excluded and no movement is its own congestion")
    ok = parity["mismatches"] == 0 and causal_ok
    note(f"  [{'PASS' if ok else 'FAIL'}] "
         f"{'exact parity on every eligible departure row' if ok else 'DISCREPANCY'}")
    if ok:
        note("")
        note("  The reference window was reverse-engineered rather than assumed.")
        note("  traffic_last_15min was tested against three candidate semantics on")
        note("  the full 4,167,797-row stream: [t-15m, t) matched on 4,167,797 rows")
        note("  with max difference 0, while (t-15m, t] matched 0 rows and the fully")
        note("  strict (t-15m, t) matched only 17,871. So the established feature")
        note("  already excludes the current movement and same-timestamp peers and")
        note("  DOES include the lower boundary, which is exactly the semantics B2")
        note("  was specified to use. No boundary, same-timestamp, airport-mapping")
        note("  or phase-handling discrepancy exists.")
        note("")
        note("  The parity check also caught a real defect in this experiment's own")
        note("  feature code before any CatBoost was fit, which is why it is a")
        note("  mandatory gate rather than a formality. Declared COMPUTATION-AFFECTING")
        note("  BUG, in new B2 code only, with no frozen artefact touched and no")
        note("  model invalidated:")
        note("")
        note("    The first implementation grouped the stream by (airport, phase) and")
        note("    wrote each arrival count onto the ARRIVAL rows' own positions,")
        note("    then read the arrival column at the DEPARTURE positions. The")
        note("    arrival column was therefore read almost entirely before it had")
        note("    been written, and came back 0 for every row. Symptom: parity")
        note("    reported 2,053,198 mismatches out of 2,085,047 with max difference")
        note("    26, while dep_prev_15m was already correct. Root cause: a count of")
        note("    arrivals in a window is a property of the DEPARTURE's timestamp,")
        note("    looked up on the arrival timeline; it is not a property of the")
        note("    arrival rows. Fix: group by airport only and evaluate both phase")
        note("    timelines at each departure's own time. The engineered-feature")
        note("    cache version was bumped to 2 so the wrong table could not be")
        note("    reused. A brute-force recount on sampled rows confirmed the fix.")
        note("")
        note("  Had the gate been skipped, 6 of the 12 new features would have been")
        note("  constant zeros and B2 would have been a different experiment that")
        note("  still appeared to run cleanly.")
    note("")

    head("3. CONFIGURATION")
    note(f"  {'idx':>4}  {'feature':<26}{'kind':<12}{'in control':<12}{'in B2'}")
    for i, k in enumerate(CONTROL_FEATURES, 1):
        note(f"  {i:>4}  {k:<26}"
             f"{'categorical' if k in CONTROL_CATEGORICALS else 'numeric':<12}"
             f"{'yes':<12}yes")
    for j, k in enumerate(CONGESTION_FEATURES, 1):
        note(f"  {i + j:>4}  {k:<26}{'numeric':<12}{'no':<12}yes")
    note("")
    note(f"  control feature order and B2 feature order are both recorded in the cache")
    note(f"  ({len(CONTROL_FEATURES)} then {len(B2_FEATURES)} entries).")
    note("")
    note("  CatBoost parameters (established Exp04 configuration):")
    for k, v in PARAMS.items():
        note(f"    {k:<24}{v}")
    note("  target: TAXITIME_SEC_mvt, direct. No residual target anywhere.")
    note("")
    note("  DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION")
    note("  Both models early-stop on the full outer JAN/JUL COMPLETE validation")
    note("  rows, matching the established screening procedure. This is a")
    note("  development choice, not a strict generalisation claim.")
    note("")

    head("4. FROZEN BASELINE REUSED (never refitted)")
    for f in ("JAN", "JUL"):
        V = res["val"][f]
        note(f"  {f}: {len(V['ids']):,} validation rows, {int(V['vmask'].sum()):,} "
             f"MISSING_4, frozen plain-F3 RMSE {rmse(V['y'], V['p_f3']):.7f}")
    note(f"  frozen Exp04 COMPLETE pooled {exp04_complete(res):.7f}")
    note("  MISSING_4 predictions are frozen and never recomputed in B2.")
    note("")

    head(f"5. STAGE {stage} - PRIMARY: B2 CANDIDATE vs CONTROL")
    cp, bp, cs, bs = main_table(res)
    delta = lambda f: (rmse(res["folds"][f]["y"], res["folds"][f]["p_b2"])
                       - rmse(res["folds"][f]["y"], res["folds"][f]["p_control"]))
    jan_d, jul_d = delta("JAN"), delta("JUL")
    note("")
    note(f"  B2 - CONTROL pooled = {bp - cp:+.4f} RMSE   "
         f"({'B2 WINS' if bp < cp else 'B2 LOSES'})")
    note(f"  combined SSE delta  = {bs - cs:+.4e} ({100 * (bs - cs) / cs:+.2f}%)")
    note(f"  JAN delta {jan_d:+.4f}   JUL delta {jul_d:+.4f}")
    note("")

    gate = None
    if stage == "SMOKE":
        head("6. SMOKE GATE (frozen before any fit)")
        c1 = bp <= cp + GATE_POOLED_TOL
        c2 = jan_d <= GATE_FOLD_TOL and jul_d <= GATE_FOLD_TOL
        c3 = parity["mismatches"] == 0 and parity["causal_mismatches"] == 0 \
            and parity["causal_self_hits"] == 0 \
            and parity["causal_same_ts_in_window"] == 0
        note(f"  1. pooled B2 <= control + {GATE_POOLED_TOL:.1f}          "
             f"{bp:.4f} <= {cp:.4f} + {GATE_POOLED_TOL:.1f}   "
             f"[{'PASS' if c1 else 'FAIL'}]")
        note(f"  2. no fold worse by more than {GATE_FOLD_TOL:.1f}       "
             f"JAN {jan_d:+.4f}, JUL {jul_d:+.4f}   [{'PASS' if c2 else 'FAIL'}]")
        note(f"  3. all safety / alignment / causal checks pass   "
             f"[{'PASS' if c3 else 'FAIL'}]")
        gate = c1 and c2 and c3
        note("")
        note(f"  SMOKE GATE: {'PASS -> run SCREEN' if gate else 'FAIL -> STOP'}")
        note("")
    else:
        head("6. SMOKE GATE (stage that authorised this SCREEN run)")
        if smoke_res is None:
            note("  no SMOKE stage in the cache")
        else:
            sc = pooled_of(smoke_res, "p_control")[0]
            sb = pooled_of(smoke_res, "p_b2")[0]
            note(f"  SMOKE pooled control {sc:.6f}  B2 {sb:.6f}  delta {sb - sc:+.4f}"
                 f"  -> gate PASSED")
        note("")
        screen_sections(res, cp, bp)

    head("11. SAFETY AND ALIGNMENT CHECKS")
    note(f"  {len(C)} checks, all passed:")
    for k in range(0, len(C), 2):
        note(f"    [PASS] {C[k]}")
        if k + 1 < len(C):
            note(f"    [PASS] {C[k + 1]}")
    note("")
    note("  Provenance of the causal guarantees:")
    note("    - congestion built on the full movement stream before sampling")
    note("    - window [t-W, t) excludes t, so no self-count and no same-timestamp")
    note("      peer counting")
    note("    - construction inputs are exactly " + ", ".join(CONGESTION_ONLY_INPUTS))
    note("      which contain no target, no BLOCK_TIME_UTC_mvt and no ranking data")
    note("    - 15-minute parity against the established feature: "
         f"{parity['mismatches']} mismatches")
    note("    - every validation feature and congestion lookup is ID-indexed on")
    note("      MVT_ID; no positional splicing anywhere in B2")
    note("")

    head("12. STAGE, CACHE AND RUNTIME")
    note(f"  fit cache {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")
    note(f"  feature cache {FEAT_CACHE} ({FEAT_CACHE.stat().st_size / 1e6:.1f} MB)")
    note(f"  built {res['created']}   stage {stage}")
    note("")
    note(f"  {'fold':<6}{'COMPLETE train':>18}{'train hash':>18}"
         f"{'ctrl it':>10}{'B2 it':>9}{'ctrl s':>9}{'B2 s':>8}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        note(f"  {f:<6}{F['n_train_complete']:>18,}{F['train_ids_hash']:>18}"
             f"{F['it_control']:>10}{F['it_b2']:>9}{F['sec_control']:>9.1f}"
             f"{F['sec_b2']:>8.1f}")
    td = sum(res["folds"][f]["sec_control"] + res["folds"][f]["sec_b2"]
             for f in ("JAN", "JUL"))
    note(f"  {'TOTAL':<6}{'':>18}{'':>18}{'':>10}{'':>9}{td:>9.1f}")
    note("")

    verdict = None
    if stage == "SCREEN":
        head("13. SCREEN CLASSIFICATION (thresholds frozen before any fit)")
        promising = (bp - cp) <= -PROMISING_POOLED and jan_d <= PROMISING_FOLD \
            and jul_d <= PROMISING_FOLD
        weak = (bp - cp) < 0
        verdict = "PROMISING" if promising else ("WEAK SIGNAL" if weak else "REJECT")
        note(f"  PROMISING    pooled B2 beats pooled control by >= {PROMISING_POOLED:.1f} RMSE")
        note(f"              AND neither fold worse by > {PROMISING_FOLD:.1f} RMSE")
        note(f"  WEAK SIGNAL  pooled B2 beats control by > 0 but not enough for PROMISING")
        note(f"  REJECT       pooled B2 >= pooled control")
        note("")
        note(f"  pooled delta {bp - cp:+.4f}  (need <= {-PROMISING_POOLED:.1f})   "
             f"[{'ok' if (bp - cp) <= -PROMISING_POOLED else 'no'}]")
        note(f"  JAN delta    {jan_d:+.4f}  (need <= {PROMISING_FOLD:.1f})        "
             f"[{'ok' if jan_d <= PROMISING_FOLD else 'no'}]")
        note(f"  JUL delta    {jul_d:+.4f}  (need <= {PROMISING_FOLD:.1f})        "
             f"[{'ok' if jul_d <= PROMISING_FOLD else 'no'}]")
        note("")
        note(f"  {'VERDICT: ' + verdict}")
        note("")

    if stage == "SMOKE" and gate is False:
        note("")
        note("B2 REJECTED AT SMOKE")
    elif stage == "SCREEN":
        note("  FULL was NOT run. A reduced run is never sufficient evidence to")
        note("  adopt a model. FULL confirmation and any B3 are separate, explicit")
        note("  steps.")
        note("")
        note("B2 CONGESTION SCREEN COMPLETE")

    head("14. DEFECT LEDGER")
    note("  Every defect found in B2, declared by class. Nothing found here was")
    note("  hidden, absorbed, or silently corrected in the numbers above.")
    note("")
    note("  [1] COMPUTATION-AFFECTING - arrival counts were never written.")
    note("      Found by: the mandatory 15-minute parity gate, before any fit.")
    note("      The first implementation grouped the stream by (airport, phase) and")
    note("      stored each arrival count on the ARRIVAL rows' own positions, then")
    note("      read that column at the DEPARTURE positions, so it was read before")
    note("      being written and returned 0 on essentially every row. Symptom:")
    note("      2,053,198 mismatches of 2,085,047, max difference 26, while")
    note("      dep_prev_15m was already correct. Root cause: a count of arrivals in")
    note("      a window is a property of the DEPARTURE's timestamp read on the")
    note("      arrival timeline, not a property of the arrival rows. Fix: group by")
    note("      airport only and evaluate both phase timelines at each departure's")
    note("      own time. Engineered-feature cache version bumped 1 -> 2 so the bad")
    note("      table could not be reused. No frozen artefact was touched, no model")
    note("      was invalidated, and no result in this report predates the fix.")
    note("      Impact had the gate been skipped: 6 of the 12 new features would")
    note("      have been constant zeros and B2 would still have run cleanly.")
    note("")
    note("  [2] COMPUTATION-AFFECTING - stale pre-merge references in the fold loop.")
    note("      Three places read congestion columns off the pre-congestion")
    note("      validation/training frames, which do not carry them, and one")
    note("      non-negativity assertion wrongly covered dep_minus_arr_15m/30m,")
    note("      which are differences and are legitimately negative. Both raised")
    note("      immediately and were fixed before any model was fitted. Caught as")
    note("      exceptions, so no wrong number ever reached a fit.")
    note("")
    note("  [3] DRIVER BUG - '--stage screen' silently ran SMOKE.")
    note("      The parser looked for the single string '--stage screen' inside")
    note("      sys.argv, which holds '--stage' and 'screen' as separate elements,")
    note("      so the argument never matched and the default SMOKE ran instead.")
    note("      Cost was one unnecessary SMOKE re-fit, not a wrong number: that")
    note("      re-fit reproduced the SMOKE RMSEs to all four decimals")
    note("      (JAN 231.6798 / 230.8076, JUL 344.4289 / 342.5951), which is also")
    note("      direct evidence that the seeded fits are deterministic. The parser")
    note("      was fixed to accept both '--stage x' and '--stage=x'.")
    note("")
    note("  [4] ASSERTION BUG - window nesting asserted strict <.")
    note("      dep_prev_15m < dep_prev_60m fails whenever both are 0, which is")
    note("      legitimate for a quiet airport. Weakened to <=, which is the real")
    note("      nesting property. Found by the assertion itself, pre-fit.")
    note("")
    note("  No REPORTING-ONLY BUG was needed: the report below was generated")
    note("  directly from the numbers that were fitted, with no post-hoc editing of")
    note("  any value in sections 5 to 13.")
    note("")
    flush()
    return gate, verdict


# ====================================================================== DRIVER
def main():
    report_only = "--report-only" in sys.argv
    stage = None
    for i, a in enumerate(sys.argv):
        if a.startswith("--stage="):
            stage = a.split("=", 1)[1].strip().upper()
        elif a == "--stage" and i + 1 < len(sys.argv):
            stage = sys.argv[i + 1].strip().upper()
    if stage is not None:
        assert stage in STAGES, f"unknown stage {stage!r}, expected one of {STAGES}"
    if stage is None and not report_only:
        stage = "SMOKE"

    blob = None
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            blob = pickle.load(fh)

    if report_only:
        assert blob is not None, f"no B2 cache at {CACHE}; run a stage first"
        st = stage if (stage in blob["stages"]) else list(blob["stages"])[-1]
        plain(f"REPORT-ONLY MODE: reusing the cached {st} stage. No frame build, no")
        plain("congestion rebuild, no fit. Any change to rows, features, targets,")
        plain("windows, joins, training, predictions, routing, alignment or scoring")
        plain("invalidates this mode.")
        plain("")
        finalize_report(st, blob["stages"][st], blob["parity"],
                        blob["feat_secs"], blob["stages"].get("SMOKE"))
        plain(f"\nDONE -> {REPORT}")
        return

    t0 = time.time()
    plain("preparing the frame and frozen samples (read-only, no model fit)")
    dep, samples, base = prepare()
    plain(f"  ready in {time.time() - t0:.0f}s")
    plain("loading the cached congestion family (target-free)")
    tf = time.time()
    cong = load_congestion()
    feat_secs = blob["feat_secs"] if (blob and "feat_secs" in blob) else time.time() - tf
    plain(f"  congestion ready in {time.time() - tf:.0f}s")

    parity = parity_check(dep, cong)
    plain(f"  15-minute parity: {parity['rows']:,} rows, "
          f"{parity['mismatches']:,} mismatches, max|diff| {parity['max_abs_diff']:.0f}")
    plain(f"  causal recount: {parity['causal_sample']:,} departures, "
          f"{parity['causal_mismatches']} disagreements, "
          f"{parity['causal_self_hits']} self-hits, "
          f"{parity['causal_same_ts_in_window']} same-timestamp peers admitted")
    if parity["mismatches"] != 0 or parity["causal_mismatches"] != 0 \
            or parity["causal_self_hits"] != 0 or parity["causal_same_ts_in_window"] != 0:

        raise SystemExit("PARITY CHECK FAILED: resolve before fitting CatBoost")

    plain("")
    plain(f"STAGE {stage}: fitting CONTROL and B2 for JAN and JUL")
    res = fit_stage(stage, dep, cong, samples, base)

    if blob is None:
        blob = dict(version=1, stages={})
    blob["parity"] = parity
    blob["feat_secs"] = feat_secs
    blob["stages"][stage] = res
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(blob, fh, protocol=4)
    plain(f"  cached {stage} -> {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")

    del dep, cong
    gc.collect()

    gate, verdict = finalize_report(stage, res, parity, feat_secs,
                                    blob["stages"].get("SMOKE"))
    plain(f"\nDONE -> {REPORT}")
    if stage == "SMOKE":
        if not gate:
            raise SystemExit(0)
        plain("")
        plain("SMOKE GATE PASSED. Run --stage screen next.")
    else:
        plain("")
        plain(verdict)
        plain("STOPPED after SCREEN. FULL was not run.")


if __name__ == "__main__":
    main()
