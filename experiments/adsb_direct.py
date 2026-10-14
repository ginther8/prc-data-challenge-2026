"""ADS-B-first model: an independent LightGBM taxi-time model trained on ALL
2025 months, then averaged (equal weights) with the corrector's predictions.

WHY
    The ADS-B corrector can only learn from months that have the base model's
    honest out-of-fold predictions (JAN/JUL). This model needs no base model:
    it learns taxi time directly from the supplied timestamps + ADS-B features,
    so it can use all ~2.1M 2025 departures. It is a second model family,
    averaged with the corrector at the end (equal weights held on the board for
    other teams; tuned weights did not).

MODEL
    rows    : COMPLETE departures only (d_AOBT3 present). MISSING_4 rows keep
              the corrector's prediction (their labels are mostly schedule
              echoes this model cannot learn from 2025 alone).
    target  : y - d_AOBT3 (the anchor every strong model uses), clipped to
              +-3600 s for training only. Prediction = d_AOBT3 + model output.
    features: time (hour, day_of_week, month), traffic_last_15min, the five T4
              deltas and their differences (timestamp consistency), all ADS-B
              movement / parking / stand-distance / match-type features
              (adsb_npts left out - receiver-density drift), aircraft type,
              wake category, flight type, flight rule.
              Airport identity (ADEP, RUNWAY, STAND, ADES) is OFF by default:
              ADS-B coverage per airport changed a lot between 2025 and 2026,
              and physics (stand->runway distance) transfers better than
              identity. --with-identity switches it on for comparison.
    learner : LightGBM, tree count chosen on the inner split
              abs(MVT_ID_mvt) % 5 == 0 of the training months, refit on all.

VALIDATION (same folds as everything else)
    JAN = train on all 2025 months except January, predict January (COMPLETE)
    JUL = likewise for July. Then --blend compares
          corrector alone  vs  0.5 * corrector + 0.5 * direct  (COMPLETE rows)
    on the corrector's own honest JAN/JUL predictions. Keep only if BOTH
    months improve.

STEPS (run from the project root)
    1. corrector OOF for the setup whose ranking file you will blend with,
       e.g. the V12 setup:
         python -u experiments\\adsb_corrector.py --validate-only --second-model lgbm ^
             --drop-features adsb_npts --m4-from-v4 --base-oof <JAN> <JUL> ^
             --save-oof data\\processed\\corrector_oof_v12.parquet
    2. python -u experiments\\adsb_direct.py --train
    3. python -u experiments\\adsb_direct.py --blend --corr-oof data\\processed\\corrector_oof_v12.parquet ^
             --corr-submission submissions\\vigorous-glove_v12.parquet --validate-only
    4. same as 3 with --version N instead of --validate-only to write the submission

Inputs : data/processed/v8_prep_cache.parquet, data/processed/adsb_features.parquet
         (built with --months all), data/raw/ranking.parquet, data/raw/submitting.parquet,
         reference/v8/train_catboost_v8_defs.py
Outputs: data/processed/direct_oof_{JAN,JUL}.parquet, data/processed/direct_ranking.parquet,
         reports/adsb_direct.txt, reports/adsb_blend.txt, submissions/vigorous-glove_vN.parquet
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "v8"))

PROC = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"
PREP = PROC / "v8_prep_cache.parquet"
FEATS = PROC / "adsb_features.parquet"
RANK_PREP = PROC / "ranking_prep_cache.parquet"
OUT_RANK = PROC / "direct_ranking.parquet"
META = PROC / "direct_meta.json"
REPORT = PROJECT_ROOT / "reports" / "adsb_direct.txt"
BLEND_REPORT = PROJECT_ROOT / "reports" / "adsb_blend.txt"
SUBS = PROJECT_ROOT / "submissions"
TEAM = "vigorous-glove"

MID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
EXPECTED_RANKING = 344_841
FOLDS = {"JAN": "2025-01", "JUL": "2025-07"}
T4 = ["d_AOBT3", "d_EOBT1", "d_IOBT", "d_LOBT", "d_SCHED"]
BASE_NUM = ["hour", "day_of_week", "month", "traffic_last_15min"] + T4
CAT_ALWAYS = ["AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt"]
CAT_IDENTITY = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "ADES_mvt"]
DROP_ADSB = ["adsb_npts"]
TRAIN_CLIP = 3600.0
LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=255,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, max_bin=255, seed=42,
                  cat_smooth=20, max_cat_to_onehot=8, verbose=-1)
MAX_ROUNDS, ES = 6000, 100

OUT = []


def A(s=""):
    OUT.append(str(s))
    print(s, flush=True)


def save_report(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(OUT) + "\n")


def stop(msg, path=REPORT):
    A(f"STOPPED: {msg}")
    save_report(path)
    sys.exit(1)


def rmse(e):
    e = np.asarray(e, np.float64)
    return math.sqrt(float(np.dot(e, e)) / len(e)) if len(e) else float("nan")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def ranking_prep():
    """V8 features for the 2026 ranking departures, built exactly like the
    training prep cache (same reference functions, ranking-only stream)."""
    if RANK_PREP.exists():
        return pd.read_parquet(RANK_PREP)
    from train_catboost_v8_defs import (MINUEND, T4_FEATURES, V7_FEATURES,
                                        add_t4_features, add_time_features,
                                        add_traffic_feature,
                                        apply_categorical_handling, T4_SPEC)
    cols = sorted({MID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", MINUEND,
                   *[c for c in V7_FEATURES if c not in ("hour", "day_of_week", "month",
                                                         "traffic_last_15min")],
                   *[src for _, src in T4_SPEC]})
    frame = pq.read_table(RAW / "ranking.parquet", columns=cols).to_pandas()
    frame = add_traffic_feature(frame)
    dep = frame[frame["PHASE_mvt"] == "DEP"].copy()
    if len(dep) != EXPECTED_RANKING:
        stop(f"ranking departures {len(dep):,} != {EXPECTED_RANKING:,}")
    dep = add_time_features(dep)
    dep = add_t4_features(dep)
    dep = apply_categorical_handling(dep)
    out = dep[[MID, MINUEND] + V7_FEATURES + T4_FEATURES].reset_index(drop=True)
    out.to_parquet(RANK_PREP, index=False)
    A(f"built ranking prep cache: {len(out):,} departures -> {RANK_PREP.name}")
    return out


def add_features(df, feats, cats):
    df = df.merge(feats, on=MID, how="left")
    df["adsb_matched"] = df["adsb_matched"].fillna(0)
    df["adsb_ok"] = df["adsb_ok"].fillna(0)
    # timestamp-consistency differences (trees cannot subtract)
    df["aobt_minus_eobt"] = df["d_EOBT1"] - df["d_AOBT3"]
    df["aobt_minus_iobt"] = df["d_IOBT"] - df["d_AOBT3"]
    df["aobt_minus_sched"] = df["d_SCHED"] - df["d_AOBT3"]
    df["est_minus_aobt"] = df["adsb_est"] - df["d_AOBT3"]
    df["move_minus_aobt"] = df["adsb_est_move"] - df["d_AOBT3"]
    for c in cats:
        df[c] = df[c].astype(str)
    return df


def numeric_features(feats_cols):
    adsb = [c for c in feats_cols if c != MID and c not in DROP_ADSB]
    return BASE_NUM + adsb + ["aobt_minus_eobt", "aobt_minus_iobt", "aobt_minus_sched",
                              "est_minus_aobt", "move_minus_aobt"]


def to_matrix(df, num, cats, categories):
    X = df[num].astype("float32").copy()
    for c in cats:
        X[c] = pd.Categorical(df[c], categories=categories[c])
    return X


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def fit(Xtr, ytr, ids, n_rounds=None):
    import lightgbm as lgb
    if n_rounds is None:
        inner = (np.abs(ids) % 5 == 0)
        dtr = lgb.Dataset(Xtr[~inner], ytr[~inner], free_raw_data=True)
        dva = lgb.Dataset(Xtr[inner], ytr[inner], reference=dtr)
        probe = lgb.train(LGB_PARAMS, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(ES, verbose=False)])
        n_rounds = max(int(probe.best_iteration), 1)
        del probe
    model = lgb.train(LGB_PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=n_rounds)
    return model, n_rounds


def train(args):
    t_all = time.time()
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        stop("needs LightGBM: pip install lightgbm")
    for p in (PREP, FEATS):
        if not p.exists():
            stop(f"missing {p}")
    cats = CAT_ALWAYS + (CAT_IDENTITY if args.with_identity else [])
    A("=" * 78)
    A("ADS-B-FIRST MODEL (LightGBM, COMPLETE rows, target y - d_AOBT3)")
    A("=" * 78)
    A(f"airport identity features: {'ON' if args.with_identity else 'OFF'}   "
      f"categoricals: {cats}")
    feats = pd.read_parquet(FEATS)
    missing_feat = [c for c in ("adsb_d_stand", "adsb_match_type") if c not in feats.columns]
    if missing_feat:
        stop(f"feature file lacks {missing_feat} - rebuild with the latest adsb_features.py")
    prep = pd.read_parquet(PREP)
    prep["ym"] = prep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    months_with_adsb = set(pd.read_parquet(FEATS, columns=[MID]).merge(
        prep[[MID, "ym"]], on=MID)["ym"].unique())
    if len(months_with_adsb) < 12:
        stop(f"ADS-B features cover only {sorted(months_with_adsb)} - build with --months all")
    prep = prep[prep["d_AOBT3"].notna()].reset_index(drop=True)          # COMPLETE only
    prep = add_features(prep, feats, cats)
    num = numeric_features(feats.columns)
    rank = ranking_prep()
    rank = rank[rank["d_AOBT3"].notna()].reset_index(drop=True)
    rank = add_features(rank, feats, cats)
    categories = {c: sorted(set(prep[c]) | set(rank[c])) for c in cats}
    A(f"COMPLETE training departures: {len(prep):,}   ranking COMPLETE: {len(rank):,}   "
      f"features: {len(num)} numeric + {len(cats)} categorical")
    if args.sample < 1.0:
        prep = prep.sample(frac=args.sample, random_state=1).reset_index(drop=True)
        A(f"SMOKE TEST: training sample {args.sample:.0%} -> {len(prep):,} rows")

    y_res = np.clip((prep[TARGET] - prep["d_AOBT3"]).to_numpy(np.float64), -TRAIN_CLIP, TRAIN_CLIP)
    meta = {"with_identity": args.with_identity, "folds": {}}
    for fold, ym in FOLDS.items():
        t0 = time.time()
        tr = (prep["ym"] != ym).to_numpy()
        va = ~tr
        Xtr = to_matrix(prep[tr], num, cats, categories)
        model, n = fit(Xtr, y_res[tr], prep.loc[tr, MID].to_numpy())
        del Xtr
        Xva = to_matrix(prep[va], num, cats, categories)
        pred = prep.loc[va, "d_AOBT3"].to_numpy() + model.predict(Xva)
        yv = prep.loc[va, TARGET].to_numpy(np.float64)
        anchor = prep.loc[va, "d_AOBT3"].to_numpy()
        A(f"{fold}: trained on {tr.sum():,} rows, {n} rounds ({time.time()-t0:.0f}s) | "
          f"COMPLETE RMSE: direct {rmse(yv - pred):.2f}   (d_AOBT3 alone {rmse(yv - anchor):.2f})")
        out = pd.DataFrame({MID: prep.loc[va, MID].to_numpy(), "fold": fold,
                            "y": yv, "pred_direct": pred})
        out.to_parquet(PROC / f"direct_oof_{fold}.parquet", index=False)
        meta["folds"][fold] = {"rounds": n, "rows_train": int(tr.sum()), "rows_valid": int(va.sum())}
        if fold == "JAN":
            imp = pd.Series(model.feature_importance("gain"), index=list(num) + cats)
            A("  top features (gain): " + ", ".join(
                f"{k} {v / imp.sum():.1%}" for k, v in imp.sort_values(ascending=False).head(12).items()))
        del model
        save_report(REPORT)

    n_final = int(round(np.mean([v["rounds"] for v in meta["folds"].values()]) * 1.05))
    t0 = time.time()
    Xall = to_matrix(prep, num, cats, categories)
    model, _ = fit(Xall, y_res, prep[MID].to_numpy(), n_rounds=n_final)
    del Xall
    pr = rank["d_AOBT3"].to_numpy() + model.predict(to_matrix(rank, num, cats, categories))
    pd.DataFrame({MID: rank[MID].to_numpy(), "pred_direct": pr}).to_parquet(OUT_RANK, index=False)
    meta["final_rounds"] = n_final
    META.write_text(json.dumps(meta, indent=2))
    A(f"final model: all 2025 COMPLETE rows, {n_final} rounds ({time.time()-t0:.0f}s) -> "
      f"{OUT_RANK.name}: {len(pr):,} ranking rows, median {np.median(pr):.0f} s")
    A(f"total {time.time()-t_all:.0f}s")
    A("DONE")
    save_report(REPORT)


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------
def blend(args):
    global OUT
    OUT = []
    rep = BLEND_REPORT
    w = args.weight
    A("=" * 78)
    A(f"BLEND: corrector (weight {1 - w:.2f}) + ADS-B-first model (weight {w:.2f}) on COMPLETE rows")
    A("=" * 78)
    if not args.corr_oof or not Path(args.corr_oof).exists():
        stop("--corr-oof file missing (run adsb_corrector.py --validate-only ... --save-oof <file>)", rep)
    c = pd.read_parquet(args.corr_oof)
    d = pd.concat([pd.read_parquet(PROC / f"direct_oof_{f}.parquet") for f in FOLDS], ignore_index=True)
    v = c.merge(d[[MID, "pred_direct"]], on=MID, how="left")
    comp = (v["state"] == "COMPLETE").to_numpy()
    if v.loc[comp, "pred_direct"].isna().any():
        stop("some COMPLETE validation rows have no direct-model prediction", rep)
    adsb_ok = None
    if args.only_adsb:
        adsb_ok = pd.read_parquet(FEATS, columns=[MID, "adsb_ok"])
        v = v.merge(adsb_ok, on=MID, how="left")
        ok = (v["adsb_ok"].fillna(0) == 1).to_numpy()
        A(f"  blending only rows with an ADS-B track reaching the runway: "
          f"{(comp & ok).mean():.1%} of validation rows")
        comp = comp & ok
    v["blend"] = np.where(comp, (1 - w) * v["corr_pred"] + w * v["pred_direct"], v["corr_pred"])
    v["blend"] = v["blend"].clip(lower=0)
    y = v["y"].to_numpy()
    A(f"  {'':<16}{'rows':>9}{'corrector':>11}{'direct':>9}{'blend':>9}{'delta':>9}")
    res = {}
    for name, m in (("JAN", v["fold"] == "JAN"), ("JUL", v["fold"] == "JUL"),
                    ("POOLED", np.ones(len(v), bool))):
        m = np.asarray(m)
        rc, rb = rmse(y[m] - v["corr_pred"][m]), rmse(y[m] - v["blend"][m])
        mc = m & comp
        rd = rmse(y[mc] - v["pred_direct"][mc])
        res[name] = (rc, rb)
        A(f"  {name:<16}{m.sum():>9,}{rc:>11.2f}{rd:>9.2f}{rb:>9.2f}{rb - rc:>+9.2f}"
          + ("   (direct: COMPLETE rows only)" if name == "POOLED" else ""))
    A("")
    A("  per airport (pooled): corrector -> blend")
    for ap, g in v.groupby("ADEP_mvt") if "ADEP_mvt" in v.columns else []:
        A(f"    {ap}: {rmse(g['y'] - g['corr_pred']):8.2f} -> {rmse(g['y'] - g['blend']):8.2f}")
    gain = (y - v["corr_pred"]) ** 2 - (y - v["blend"]) ** 2
    top = np.sort(gain.to_numpy())[::-1][:20].sum()
    if gain.sum() > 0:
        A(f"  share of the gain from the top-20 rows: {top / gain.sum():.1%}")
    passed = all(res[f][1] < res[f][0] for f in ("JAN", "JUL"))
    A(f"  GATE (both months better than the corrector alone): {'PASS' if passed else 'FAIL'}")
    save_report(rep)
    if args.validate_only:
        A("validate-only: no submission built")
        save_report(rep)
        return
    if not passed and not args.force:
        A("gate failed: no submission built (use --force to override)")
        save_report(rep)
        return
    if args.version is None:
        stop("--version N is required to build a submission", rep)
    dest = SUBS / f"{TEAM}_v{args.version}.parquet"
    if dest.exists():
        stop(f"{dest} already exists; refusing to overwrite", rep)
    tpl = pd.read_parquet(RAW / "submitting.parquet")
    cs = pd.read_parquet(args.corr_submission)
    if len(cs) != EXPECTED_RANKING or set(cs[MID]) != set(tpl[MID]):
        stop("corrector submission does not match the template ids", rep)
    dr = pd.read_parquet(OUT_RANK)
    r = tpl[[MID]].merge(cs[[MID, TARGET]].rename(columns={TARGET: "corr"}), on=MID, how="left")
    r = r.merge(dr, on=MID, how="left")
    has = r["pred_direct"].notna()
    if args.only_adsb:
        r = r.merge(adsb_ok, on=MID, how="left")
        has = has & (r["adsb_ok"].fillna(0) == 1)
    r["pred"] = np.where(has, (1 - w) * r["corr"] + w * r["pred_direct"], r["corr"])
    r["pred"] = r["pred"].clip(lower=0)
    A(f"  ranking rows blended: {has.mean():.1%}, others keep the corrector")
    diff = (r["pred"] - r["corr"])[has]
    A(f"  change vs corrector: mean {diff.mean():+.1f}s  median {diff.median():+.1f}s  "
      f"p1 {diff.quantile(.01):+.0f}  p99 {diff.quantile(.99):+.0f}")
    sub = pd.DataFrame({MID: r[MID].to_numpy(), TARGET: r["pred"].to_numpy()})[list(tpl.columns)]
    checks = {
        "rows == 344,841": len(sub) == EXPECTED_RANKING,
        "columns match template": list(sub.columns) == list(tpl.columns),
        "row order matches template": (sub[MID].to_numpy() == tpl[MID].to_numpy()).all(),
        "unique ids": not sub[MID].duplicated().any(),
        "no NaN/inf": bool(np.isfinite(sub[TARGET]).all()),
    }
    for k, ok in checks.items():
        A(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    if not all(checks.values()):
        stop("integrity check failed", rep)
    A(f"  prediction min/median/max: {sub[TARGET].min():.0f} / {sub[TARGET].median():.0f} / "
      f"{sub[TARGET].max():.0f}")
    SUBS.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(dest, index=False)
    A(f"  wrote {dest}")
    save_report(rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--with-identity", action="store_true",
                    help="also use ADEP / RUNWAY / STAND / ADES as categoricals")
    ap.add_argument("--sample", type=float, default=1.0, help="smoke-test training fraction")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--corr-oof", default="")
    ap.add_argument("--corr-submission", default="")
    ap.add_argument("--weight", type=float, default=0.5,
                    help="weight of the direct model (default 0.5 = equal weights)")
    ap.add_argument("--only-adsb", action="store_true",
                    help="blend only rows whose ADS-B track reaches the runway (adsb_ok = 1); "
                         "all other rows keep the corrector's prediction")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--version", type=int, default=None)
    args = ap.parse_args()
    if args.train:
        train(args)
    if args.blend:
        blend(args)
    if not (args.train or args.blend):
        ap.print_help()


if __name__ == "__main__":
    main()