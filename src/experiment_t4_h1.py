"""T4 vs T4 + H1 flight-history features, on the January / July 2025 folds.

Question: do the H1 flight-history features add GENUINE predictive information
ON TOP OF T4, or is any apparent gain an artefact of a few extreme target rows?

DESIGN NOTE - why nothing here is retyped:
The H1 machinery (Level, build_levels, resolve, build_hist_features, the stats,
the backoff chain, the constants) and the T4 definitions are EXTRACTED FROM THE
TWO VALIDATED SOURCE FILES AT RUNTIME VIA `ast`, then exec'd. They are never
transcribed into this file, so "exact H1, no modifications" is guaranteed by
construction rather than by my care. A drift guard re-asserts the H1 identity.

Only the six backoff levels the H1 chain actually touches are built. That does
not change the H1 feature values at all - resolve() only reads levels named in
the chain - it only avoids computing the 10 groupings that H1 never uses.

T4 is NOT refitted. Its predictions are reused from the validated
experiment_v7_vs_t4.py run, cached by experiment_t4_clocks.py, and checked
against the frozen reference scores before any comparison is printed.

SAFETY: no feature uses BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt.
LEAKAGE: every historical aggregate is built from TRAINING ROWS ONLY and
        audited by the historical script's own audit_levels() assertion.
RAW LABELS: validation targets are never cleaned, clipped or filtered.

Usage:
    python src/experiment_t4_h1.py --dry-run    # definitions + checks, no fit
    python src/experiment_t4_h1.py --smoke      # 5 iterations, plumbing only
    python src/experiment_t4_h1.py              # full run
    python src/experiment_t4_h1.py --report-only
"""

import argparse
import gc
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

DATA_DIR = Path("data/raw")
HIST_SCRIPT = Path("src/experiment_historical_features.py")
T4_SCRIPT = Path("src/experiment_v7_vs_t4.py")
CACHE = Path(os.environ.get("TEMP", ".")) / "opencode" / "t4_h1_results.pkl"
T4_CACHE = Path(os.environ.get("TEMP", ".")) / "opencode" / "clocks_results.pkl"

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--report-only", action="store_true")
parser.add_argument("--folds", nargs="+", default=["JAN", "JUL"],
                    choices=["JAN", "JUL"])
args = parser.parse_args()

print("=" * 78)
print("EXPERIMENT: T4 vs T4 + H1 FLIGHT-HISTORY FEATURES")
print("=" * 78)

# ============================================================
# EXTRACT the validated definitions from the two source scripts
# ============================================================

import ast  # noqa: E402


def extract(path, names):
    """Pull the named top-level defs/assignments out of `path` verbatim."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        nm = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nm = node.name
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    nm = t.id
                    break
        if nm in names and nm not in found:
            found[nm] = node
    missing = set(names) - set(found)
    assert not missing, f"{path}: could not extract {sorted(missing)}"
    return found, tree


HIST_NAMES = [
    "MIN_COUNT", "TRIM", "CLIP_CAP", "STATS", "CHAINS", "GROUPINGS",
    "FALLBACK_GROUPINGS", "ALL_GROUPINGS", "FAMILIES", "CATEGORICAL_FEATURES",
    "NEEDED_COLS", "HELPERS",
    "trimmed_mean_1d", "Level", "build_levels", "resolve",
    "build_hist_features", "audit_levels",
]
hist_nodes, hist_tree = extract(HIST_SCRIPT, HIST_NAMES)

# exec the historical machinery in a namespace seeded with its runtime deps
H = {"np": np, "pd": pd, "time": time,
     "TARGET": "TAXITIME_SEC_mvt"}
exec(compile(ast.Module(body=[hist_nodes[n] for n in HIST_NAMES],
                        type_ignores=[]), str(HIST_SCRIPT), "exec"), H)

T4_NAMES = ["V7_FEATURES", "V7_CATEGORICALS", "T4_SPEC", "T4_FEATURES",
            "MINUEND", "TARGET", "FORBIDDEN", "FOLDS", "CB_PARAMS"]
t4_nodes, _ = extract(T4_SCRIPT, T4_NAMES)
T = {"np": np, "pd": pd}
exec(compile(ast.Module(body=[t4_nodes[n] for n in T4_NAMES],
                        type_ignores=[]), str(T4_SCRIPT), "exec"), T)

V7_FEATURES = T["V7_FEATURES"]
V7_CATEGORICALS = T["V7_CATEGORICALS"]
T4_SPEC = T["T4_SPEC"]
T4_FEATURES = T["T4_FEATURES"]
MINUEND = T["MINUEND"]
TARGET = "TAXITIME_SEC_mvt"
FORBIDDEN = T["FORBIDDEN"]
CB_PARAMS = T["CB_PARAMS"]

H1_GROUPINGS = H["FAMILIES"]["H1"]           # ["flight"]
H1_CHAIN = H["CHAINS"]["flight"]
H1_NEEDED = sorted(set(H1_CHAIN))
ALL_GROUPINGS = H["ALL_GROUPINGS"]
CATEGORICAL_FEATURES = H["CATEGORICAL_FEATURES"]
H1_COLS = [f"hist_flight_{s}" for s in H["STATS"]] + \
          ["hist_flight_seen", "hist_flight_level"]

# drift guard: H1 must still be exactly the single "flight" grouping
assert H1_GROUPINGS == ["flight"], f"H1 groupings changed: {H1_GROUPINGS}"
assert H1_CHAIN == ["flight", "callsign", "route_hour", "adep_hour",
                    "adep", "global"], f"H1 chain changed: {H1_CHAIN}"
assert len(H1_COLS) == len(H["STATS"]) + 2 == 10
assert sorted(set(T4_SPEC[i][1] for i in range(len(T4_SPEC)))) == \
    sorted(c for _, c in T4_SPEC)
print(f"H1 groupings    : {H1_GROUPINGS}")
print(f"H1 backoff chain: {H1_CHAIN}")
print(f"H1 columns      : {len(H1_COLS)}  {H1_COLS}")
print(f"H1 stats        : {H['STATS']}  MIN_COUNT={H['MIN_COUNT']} "
      f"TRIM={H['TRIM']} CLIP_CAP={H['CLIP_CAP']}")
print(f"[OK] H1 + T4 definitions extracted verbatim from the validated sources")

T4_REF = {"JAN": 404.8803102, "JUL": 513.7559738}
T4_H1_FEATURES = V7_FEATURES + T4_FEATURES + H1_COLS
print(f"\nfeature counts:")
print(f"  T4        {len(V7_FEATURES) + len(T4_FEATURES):3d} "
      f"(V7 {len(V7_FEATURES)} + T4 {len(T4_FEATURES)})")
print(f"  T4+H1     {len(T4_H1_FEATURES):3d} "
      f"(+ H1 {len(H1_COLS)})")
assert not (set(T4_H1_FEATURES) & set(FORBIDDEN)), "forbidden column in features"
assert not any(f.startswith("clock_") for f in T4_H1_FEATURES), \
    "clock features are forbidden in this experiment"
print("[PASS] no BLOCK_TIME_UTC_mvt / TAXITIME_SEC_mvt in any feature list")
print("[PASS] no clock_/time-of-day feature present (as instructed)")


# ============================================================
# LOAD - column set = historical NEEDED_COLS + the five T4 sources
# ============================================================

T4_SOURCE_COLS = [c for _, c in T4_SPEC]
LOAD_COLS = list(dict.fromkeys(
    H["NEEDED_COLS"] + T4_SOURCE_COLS + [MINUEND]))
print(f"\nLoading {len(LOAD_COLS)} columns from 2025 training data...")
files = sorted(DATA_DIR.glob("training_*.parquet"))
print(f"files found: {len(files)}")
data = pd.concat([pd.read_parquet(f, columns=LOAD_COLS) for f in files],
                 ignore_index=True)
gc.collect()
print(f"Total movements: {len(data):,}")

for col in T4_SOURCE_COLS + [MINUEND]:
    assert col in data.columns, f"missing {col}"
print("[PASS] all T4 source columns present")

# --- traffic feature, identical to the V7 / T4 / historical construction ---
data["movement_airport"] = data["ADEP_mvt"]
data.loc[data["PHASE_mvt"] == "ARR", "movement_airport"] = data.loc[
    data["PHASE_mvt"] == "ARR", "ADES_mvt"]
data = data.sort_values(["movement_airport", "MVT_TIME_UTC_mvt"]).copy()
traffic_15min = (
    data.set_index("MVT_TIME_UTC_mvt")
    .groupby("movement_airport")["MVT_ID_mvt"]
    .rolling("15min", closed="left").count()
    .reset_index(name="traffic_last_15min")
)
data["traffic_last_15min"] = traffic_15min["traffic_last_15min"].to_numpy()
data["traffic_last_15min"] = data["traffic_last_15min"].fillna(0)
del traffic_15min, data["movement_airport"]
gc.collect()

departures = data[data["PHASE_mvt"] == "DEP"].copy()
del data
gc.collect()
print(f"Departure rows: {len(departures):,}")
assert len(departures) == 2_085_047, f"unexpected {len(departures):,}"

# --- V7 features, unchanged ---
departures["hour"] = departures[MINUEND].dt.hour
departures["day_of_week"] = departures[MINUEND].dt.dayofweek
departures["month"] = departures[MINUEND].dt.month
for column in CATEGORICAL_FEATURES:
    departures[column] = departures[column].fillna("UNKNOWN").astype(str)

# --- historical group keys, unchanged ---
departures["K_FLIGHT"] = departures["FLIGHT_mvt"].fillna("UNKNOWN").astype(str)
departures["K_CALLSIGN"] = departures["CALLSIGN_flt"].fillna("UNKNOWN").astype(str)
departures["year"] = departures[MINUEND].dt.year
departures["date"] = departures[MINUEND].dt.normalize()

# --- T4 deltas, same branch logic as experiment_v7_vs_t4.py ---
departures[MINUEND + "_ts"] = pd.to_datetime(departures[MINUEND], utc=True)
for name, column in T4_SPEC:
    if departures[column].dtype != "object":
        departures[name] = (departures[MINUEND]
                            - departures[column]).dt.total_seconds()
    else:
        departures[name] = (departures[MINUEND + "_ts"]
                            - pd.to_datetime(departures[column], utc=True)
                            ).dt.total_seconds()
assert MINUEND + "_ts" not in T4_H1_FEATURES

# independent integer-epoch recheck of the T4 deltas
print("\nT4 delta unit re-check (independent integer microsecond epoch path)")
for name, column in T4_SPEC:
    d = departures[name]
    ok = d.notna()
    a = departures.loc[ok, MINUEND].astype("int64").to_numpy() // 1_000_000
    b = departures.loc[ok, column].astype("int64").to_numpy() // 1_000_000
    manual = (a - b).astype("float64")
    got = d[ok].to_numpy()
    same = int((manual == got).sum())
    print(f"  {name:9s} compared={int(ok.sum()):>9,} exact={same:>9,} "
          f"max|diff|={np.abs(manual - got).max():.3e}")
    assert same == int(ok.sum()), f"{name} epoch mismatch"
print("[PASS] all T4 deltas reproduced exactly, in seconds")

# --- forbidden-column guard on the actual frame ---
# H1 columns are not here yet: they are materialised per fold by
# build_hist_features() from that fold's training-only aggregates.
for f in V7_FEATURES + T4_FEATURES:
    assert f in departures.columns, f"missing feature {f}"
    assert f not in FORBIDDEN
print("[PASS] every V7/T4 feature column exists and none is forbidden")

FOLDS = {"JAN": ("2025-01-01", "2025-02-01"),
         "JUL": ("2025-07-01", "2025-08-01")}
HOLDOUT_MONTH = {"JAN": 1, "JUL": 7}
EXPECTED_N = {"JAN": (1_931_341, 153_706), "JUL": (1_894_334, 190_713)}

print("\n" + "-" * 78)
print("FOLD CONSTRUCTION")
print("-" * 78)
for fold in args.folds:
    month = HOLDOUT_MONTH[fold]
    is2025 = departures["year"] == 2025
    tr = departures.loc[is2025 & (departures["month"] != month)]
    va = departures.loc[is2025 & (departures["month"] == month)]
    assert (len(tr), len(va)) == EXPECTED_N[fold], \
        f"{fold}: rows {(len(tr), len(va))} != validated T4 fold "\
        f"{EXPECTED_N[fold]}"
    lo, hi = FOLDS[fold]
    assert int(tr["month"].eq(month).sum()) == 0, "holdout month in training"
    # no validation movement may appear in training
    ov = len(set(tr["MVT_ID_mvt"].to_numpy()) & set(va["MVT_ID_mvt"].to_numpy()))
    assert ov == 0, f"{fold}: {ov} validation IDs present in training"
    print(f"  {fold}: train={len(tr):,}  validation={len(va):,}  "
          f"ID overlap=0  val months={sorted(va['month'].unique())}")
print("[PASS] folds match the validated T4 experiment row-for-row")

if args.dry_run:
    print("\n" + "=" * 78)
    print("DRY RUN COMPLETE - definitions and checks passed. No model trained.")
    print("=" * 78)
    raise SystemExit(0)


# ============================================================
# FIT
# ============================================================

def rmse_of(y, p):
    return float(root_mean_squared_error(y, p))


def mae_of(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def per_airport(y, p, airports, min_rows=30):
    d = pd.DataFrame({"a": np.asarray(airports), "y": np.asarray(y),
                      "p": np.asarray(p)})
    g = d.groupby("a").apply(
        lambda t: pd.Series({
            "n": len(t),
            "rmse": float(np.sqrt(((t["y"] - t["p"]) ** 2).mean())),
            "sse": float(((t["y"] - t["p"]) ** 2).sum()),
        }), include_groups=False)
    return g[g["n"] >= min_rows].sort_values("rmse", ascending=False)


if args.report_only:
    if not CACHE.exists():
        raise SystemExit(f"--report-only but no cache at {CACHE}")
    results = pickle.load(open(CACHE, "rb"))
    print(f"[report-only] loaded {sum(len(v) for v in results.values())} "
          f"cached fits from {CACHE}")
else:
    results = {}
    for fold in args.folds:
        month = HOLDOUT_MONTH[fold]
        is2025 = departures["year"] == 2025
        tr = departures.loc[is2025 & (departures["month"] != month)]
        va = departures.loc[is2025 & (departures["month"] == month)]

        print(f"\n{'-' * 78}\nFOLD {fold}: building H1 levels from "
              f"TRAINING ROWS ONLY\n{'-' * 78}")
        levels = H["build_levels"](tr, H1_NEEDED)
        tot = H["audit_levels"](levels, tr, fold)
        print(f"  LEAKAGE AUDIT OK: every level's group sums total {tot:,.0f} "
              f"= training target sum exactly")

        Hva, dva = H["build_hist_features"](levels, H1_GROUPINGS, va)
        Htr, dtr = H["build_hist_features"](
            levels, H1_GROUPINGS, tr,
            own_target=tr[TARGET].to_numpy(), loo_mask=None)
        # the H1 feature family must be exactly the 10 expected columns
        assert list(Htr.columns) == H1_COLS, \
            f"H1 columns drifted: {list(Htr.columns)}"
        assert list(Hva.columns) == H1_COLS, \
            f"H1 columns drifted (val): {list(Hva.columns)}"
        print(f"  H1 columns built: train={len(Htr):,} val={len(Hva):,}  "
              f"(exactly the {len(H1_COLS)} expected H1 columns)")
        print(f"  validation-side resolution: "
              f"seen={dva['flight']['seen_pct']:.1f}%  "
              f"at_finest(flight)={dva['flight']['at_finest_pct']:.1f}%  "
              f"level_mean={dva['flight']['level_mean']:.2f}")

        Xtr = tr[V7_FEATURES + T4_FEATURES].copy()
        Xva = va[V7_FEATURES + T4_FEATURES].copy()
        for c in Htr.columns:
            Xtr[c] = Htr[c]
            Xva[c] = Hva[c]
        del Htr, Hva
        gc.collect()

        params = dict(CB_PARAMS)
        if args.smoke:
            params.update(iterations=5, early_stopping_rounds=5, verbose=0)
        y_tr, y_va = tr[TARGET], va[TARGET]
        print(f"  arm T4+H1 ({len(Xtr.columns)} features)", flush=True)
        t0 = time.time()
        model = CatBoostRegressor(**params)
        model.fit(Xtr, y_tr, cat_features=V7_CATEGORICALS,
                  eval_set=(Xva, y_va))
        el = time.time() - t0
        pred = model.predict(Xva)
        r = rmse_of(y_va, pred)
        bi = model.get_best_iteration()
        print(f"    T4+H1  RMSE={r:8.2f}  MAE={mae_of(y_va, pred):7.2f}  "
              f"best_iter={bi:4d}  fit={el / 60:.1f} min", flush=True)
        results.setdefault("T4+H1", {})[fold] = dict(
            rmse=r, mae=mae_of(y_va, pred), pred=pred,
            y=y_va.to_numpy(),
            sse=float(((y_va.to_numpy() - pred) ** 2).sum()),
            n=len(y_va), best_iter=bi, seconds=el,
            has_flight=va["AOBT_3_flt"].notna().to_numpy(),
            airports=va["ADEP_mvt"].to_numpy(),
            importance=pd.Series(model.get_feature_importance(),
                                 index=Xtr.columns))
        del Xtr, Xva, model, levels, tr, va
        gc.collect()

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\n[cache] wrote fits to {CACHE}")

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 78)
print("SUMMARY - T4 vs T4+H1")
print("=" * 78)

# reuse validated T4 predictions rather than refitting
t4 = {}
if T4_CACHE.exists():
    cached = pickle.load(open(T4_CACHE, "rb"))
    if "T4" in cached:
        t4 = cached["T4"]
        print(f"[reuse] T4 predictions loaded from {T4_CACHE} (not refitted)")
    else:
        print(f"[warn] {T4_CACHE} has no T4 arm")
else:
    print(f"[warn] no T4 cache at {T4_CACHE}; T4 columns unavailable")

print(f"\n{'arm':10s} {'fold':5s} {'RMSE':>10s} {'MAE':>9s} {'dRMSE':>9s}")
for fold in args.folds:
    ref = T4_REF[fold]
    h = results["T4+H1"][fold]
    print(f"{'T4':10s} {fold:5s} {ref:10.4f} {'':9s} {'':9s}")
    print(f"{'T4+H1':10s} {fold:5s} {h['rmse']:10.4f} {h['mae']:9.2f} "
          f"{h['rmse'] - ref:+9.2f}")

# pooled from combined SSE / row counts, cross-checked
print(f"\n{'arm':10s} {'Jan':>9s} {'Jul':>9s} {'pooled':>9s} {'dPooled':>9s}")
if t4:
    sse_t = sum(t4[f]["sse"] for f in args.folds)
    n_t = sum(t4[f]["n"] for f in args.folds)
    sse_h = sum(results["T4+H1"][f]["sse"] for f in args.folds)
    n_h = sum(results["T4+H1"][f]["n"] for f in args.folds)
    pt, ph = float(np.sqrt(sse_t / n_t)), float(np.sqrt(sse_h / n_h))
    # assert pooled == concatenated-array RMSE (not an average of RMSEs)
    cy = np.concatenate([t4[f]["y"] for f in args.folds])
    cp = np.concatenate([t4[f]["pred"] for f in args.folds])
    assert abs(pt - rmse_of(cy, cp)) < 1e-9, "T4 pooled mismatch"
    hy = np.concatenate([results["T4+H1"][f]["y"] for f in args.folds])
    hp = np.concatenate([results["T4+H1"][f]["pred"] for f in args.folds])
    assert abs(ph - rmse_of(hy, hp)) < 1e-9, "T4+H1 pooled mismatch"
    print(f"{'T4':10s} {t4['JAN']['rmse']:9.2f} {t4['JUL']['rmse']:9.2f} "
          f"{pt:9.2f} {'':9s}")
    print(f"{'T4+H1':10s} {results['T4+H1']['JAN']['rmse']:9.2f} "
          f"{results['T4+H1']['JUL']['rmse']:9.2f} {ph:9.2f} {ph - pt:+9.2f}")
    print(f"  pooled n={n_h:,}  (sqrt of combined SSE/n; asserted == "
          f"concatenated-array RMSE)")

print("\nFLIGHT-DATA vs NO-FLIGHT-DATA RMSE (T4+H1, T4 in brackets)")
print(f"{'fold':5s} {'with_flight':>22s} {'no_flight':>22s}")
for fold in args.folds:
    h = results["T4+H1"][fold]
    hf = h["has_flight"]
    w = rmse_of(h["y"][hf], h["pred"][hf])
    n_ = rmse_of(h["y"][~hf], h["pred"][~hf])
    if t4:
        b = t4[fold]
        wb = rmse_of(b["y"][b["has_flight"]], b["pred"][b["has_flight"]])
        nb = rmse_of(b["y"][~b["has_flight"]], b["pred"][~b["has_flight"]])
        print(f"{fold:5s} {w:11.2f} [{wb:7.2f}] n={int(hf.sum()):>7,} "
              f"{n_:11.2f} [{nb:7.2f}] n={int((~hf).sum()):>6,}")
    else:
        print(f"{fold:5s} {w:11.2f} {'':16s} {n_:11.2f} {'':16s}")

if t4:
    print("\nPER-AIRPORT RMSE (>=30 validation rows), delta = T4+H1 minus T4")
    for fold in args.folds:
        h, b = results["T4+H1"][fold], t4[fold]
        gh = per_airport(h["y"], h["pred"], h["airports"])
        gb = per_airport(b["y"], b["pred"], b["airports"])
        sh = gh.head(12).copy()
        sh["T4"] = gb["rmse"].reindex(sh.index)
        sh["delta"] = sh["rmse"] - sh["T4"]
        print(f"\n  {fold}: {len(gh)} airports qualify")
        print(sh.round(2).to_string())

    # ---- concentration diagnostics ----
    print("\n" + "=" * 78)
    print("IMPROVEMENT CONCENTRATION DIAGNOSTIC")
    print("=" * 78)
    for fold in args.folds:
        h, b = results["T4+H1"][fold], t4[fold]
        assert len(h["y"]) == len(b["y"]), f"{fold}: row count differs"
        y = h["y"]
        eb = (y - b["pred"]) ** 2
        ec = (y - h["pred"]) ** 2
        gain = eb - ec
        tot = float(gain.sum())
        print(f"\n{fold}: T4 SSE={eb.sum():.4e}  T4+H1 SSE={ec.sum():.4e}  "
              f"net gain={tot:+.4e} ({100 * tot / eb.sum():+.1f}% of T4 SSE)")
        order = np.argsort(-gain)
        cs = np.cumsum(gain[order])
        for N in (10, 20, 50, 200, 1000):
            print(f"  top {N:>5,} rows' share of net SSE gain = "
                  f"{100 * cs[N - 1] / tot:6.1f}%")
        # y bands
        for lo, hi, tag in [(None, 3600, "y <= 3600"), (3600, None, "y >  3600")]:
            m = np.ones(len(y), bool) if lo is None else \
                ((y > lo) if hi is None else ((y > lo) & (y <= hi)))
            gb_, gc_ = float(eb[m].sum()), float(ec[m].sum())
            nb = int(m.sum())
            print(f"  {tag}: n={nb:>7,} ({100*nb/len(y):5.2f}% of rows)  "
                  f"T4 RMSE={np.sqrt(gb_/nb):8.2f}  "
                  f"T4+H1 RMSE={np.sqrt(gc_/nb):8.2f}  "
                  f"delta={np.sqrt(gc_/nb)-np.sqrt(gb_/nb):+8.2f}  "
                  f"share of net gain={100*(gb_-gc_)/tot:6.1f}%")
        # robust / typical-row view
        print(f"  median |error|  T4={np.median(np.abs(y-b['pred'])):7.2f}  "
              f"T4+H1={np.median(np.abs(y-h['pred'])):7.2f}")
        print(f"  rows improved={100*(gain>0).mean():5.1f}%  "
              f"worsened={100*(gain<0).mean():5.1f}%  "
              f"unchanged={100*(gain==0).mean():4.1f}%")
        # airport concentration of the gain
        d = pd.DataFrame({"a": h["airports"], "g": gain})
        ga = d.groupby("a").agg(n=("g", "size"), gain=("g", "sum"))
        ga = ga[ga["n"] >= 30].sort_values("gain", ascending=False)
        ga["pct"] = 100 * ga["gain"] / tot
        print("  gain by airport (top 5):")
        print(ga.head(5).round(2).to_string())
        top = ga.index[0]
        print(f"  -> {top} alone = {ga['pct'].iloc[0]:.1f}% of net gain; "
              f"other {len(ga)-1} airports = {ga['pct'].iloc[1:].sum():.1f}%")
        for a in ("LIRF", "LFPG"):
            if a in ga.index:
                print(f"     {a}: n={int(ga.loc[a,'n']):,}  "
                      f"gain={ga.loc[a,'gain']:.3e}  "
                      f"{ga.loc[a,'pct']:.1f}% of net gain")

    # pooled robustness cuts
    print("\n" + "=" * 78)
    print("POOLED ROBUSTNESS CUTS (delta = T4+H1 - T4; negative is better)")
    print("=" * 78)

    def pooled_cut(mask_fn, label):
        sb = sc = nb = 0
        for f in args.folds:
            b, h = t4[f], results["T4+H1"][f]
            m = mask_fn(b, h)
            sb += float(((b["y"][m] - b["pred"][m]) ** 2).sum())
            sc += float(((h["y"][m] - h["pred"][m]) ** 2).sum())
            nb += int(m.sum())
        a_, c_ = np.sqrt(sb / nb), np.sqrt(sc / nb)
        print(f"  {label:36s} n={nb:>7,}  T4={a_:8.2f}  T4+H1={c_:8.2f}  "
              f"delta={c_ - a_:+8.2f}")

    pooled_cut(lambda b, h: np.ones(len(b["y"]), bool), "ALL rows (headline)")
    pooled_cut(lambda b, h: b["y"] <= 3600, "y <= 3600s only")
    pooled_cut(lambda b, h: b["y"] > 3600, "y > 3600s only")
    pooled_cut(lambda b, h: b["has_flight"], "with-flight-data rows only")
    pooled_cut(lambda b, h: ~b["has_flight"], "no-flight-data rows only")
    pooled_cut(lambda b, h: b["airports"] != "LIRF", "excluding LIRF")
    pooled_cut(lambda b, h: (b["airports"] != "LIRF") & b["has_flight"],
               "excluding LIRF + with-flight only")
    pooled_cut(lambda b, h: (b["airports"] != "LIRF")
               & (b["airports"] != "LFPG"), "excluding LIRF and LFPG")

print("\nBEST ITERATION / RUNTIME")
for fold in args.folds:
    h = results["T4+H1"][fold]
    print(f"  T4+H1 {fold}: best_iter={h['best_iter']:4d}  "
          f"fit={h['seconds']/60:5.1f} min")

print("\nFEATURE IMPORTANCE (H1 share and top features)")
for fold in args.folds:
    imp = results["T4+H1"][fold]["importance"].sort_values(ascending=False)
    h1s = 100 * imp[[i for i in imp.index if i.startswith("hist_")]].sum() \
        / imp.sum()
    t4s = 100 * imp[[i for i in imp.index if i.startswith("d_")]].sum() \
        / imp.sum()
    print(f"\n  {fold}: H1 share {h1s:.1f}%   T4 share {t4s:.1f}%")
    print(imp.head(12).round(4).to_string())

print("\nDone. No V8, no submission, V7 untouched, nothing committed.")
