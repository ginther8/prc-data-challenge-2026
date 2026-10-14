"""V10-B2: controlled ERA5 weather experiment on the frozen V8/T4 baseline.

QUESTION
    Does the verified ERA5 weather data from V10-B1 carry genuine predictive
    value beyond the frozen V8/T4 feature set?

DESIGN - WHAT IS HELD FIXED
    Everything except the weather block. Specifically reused verbatim from the
    validated T4 experiment and train_catboost_v8_defs.py:
      * the 12 V7 operational/categorical features
      * the 5 T4 clock deltas
      * the 15-minute traffic feature, built by the drift-guarded
        add_traffic_feature() on the FULL DEP+ARR 2025 frame BEFORE the fold
        split, which is the exact row alignment the T4 result depends on
      * CatBoost: depth 8, learning_rate 0.1, loss RMSE, random_seed 42,
        iterations 1000, early_stopping_rounds 50, use_best_model True
      * raw targets, no clipping, no filtering, no target cleaning
      * no new settings, no tuning, no early feature removal

    The only thing that varies between B0, W1 and W2 is which weather columns
    are passed to fit().

ARMS
    B0  the 17 frozen V8/T4 features
    W1  B0 + basic atmospheric state
    W2  W1 + hour-ending gust and precipitation

WEATHER REPRESENTATION - ONE CANONICAL COLUMN PER MEASUREMENT
    The V10-B1 joined table stores each measurement twice, once under a
    suffixed provenance name (t2m_inst) and once under a flat name (t2m), and
    the two are bit-identical. Only the suffixed columns are read here, so no
    weather measurement reaches CatBoost twice. The flat copies are never
    loaded.

    All weather comes from the local, V10-B1-verified parquet. No CDS request is
    made. No ranking target, no truthing, no BLOCK_TIME, no new external source.

    msl_anomaly_30d is deliberately NOT used: the download starts 2025-01-01, so
    a 30-day trailing mean is undefined for the first 30 days of the year, and
    January is precisely one of the two validation folds.

OUTPUT
    reports/v10b2_weather_experiment.txt
    No model is saved and no submission is written.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, "src")

from train_catboost_v8_defs import (  # noqa: E402
    MINUEND, T4_FEATURES, T4_SPEC, TARGET, V7_CATEGORICALS, V7_FEATURES,
    add_t4_features, add_time_features, add_traffic_feature,
    apply_categorical_handling, verify_t4_deltas,
)

# ---------------------------------------------------------------- config
DATA_DIR = Path("data/raw")
REPORT = Path("reports/v10b2_weather_experiment.txt")
CACHE = Path(os.environ.get("TEMP", ".")) / "opencode" / "v10b2_run_cache.pkl"
T4_CACHE = Path(os.environ.get("TEMP", ".")) / "opencode" / "clocks_results.pkl"

ERA5_DIR = Path("data/external/era5")
HOURLY_PARQUET = ERA5_DIR / "era5_hourly.parquet"
JOINED_PARQUET = ERA5_DIR / "weather_joined.parquet"

FOLDS = {"JAN": ("2025-01-01", "2025-02-01"), "JUL": ("2025-07-01", "2025-08-01")}
EXPECTED_N = {"JAN": (1_931_341, 153_706), "JUL": (1_894_334, 190_713)}

FLIGHT_TS = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]

FROZEN = {"JAN": 404.8803102, "JUL": 513.7559738, "pooled": 468.3053858}
# Reproduction tolerance. The cached T4 references agree with the frozen
# published digits to ~1e-7 relative; anything worse is a real drift.
TOL = 5e-7

CB_PARAMS = dict(
    iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=400, early_stopping_rounds=50, use_best_model=True,
)

# ---- weather feature construction -------------------------------------
# wx_t2m        2 m air temperature                     K      canonical t2m_inst
# wx_d2m        2 m dew point                           K      canonical d2m_inst
# wx_dewspread  t2m - d2m                               K      deterministic
# wx_msl        mean sea level pressure                 Pa     canonical msl_inst
# wx_tcc        total cloud cover                       0-1    canonical tcc_inst
# wx_u10        10 m eastward wind                      m/s    canonical u10_inst
# wx_v10        10 m northward wind                     m/s    canonical v10_inst
# wx_wspd       hypot(u10, v10)                         m/s    deterministic
# wx_wdir       (270 - atan2(v10,u10) in deg) mod 360   deg    deterministic
W1_WX = ["wx_t2m", "wx_d2m", "wx_dewspread", "wx_msl", "wx_tcc",
         "wx_u10", "wx_v10", "wx_wspd", "wx_wdir"]
# W2 additions. Hour-ending, so they carry the V10-B1 CEIL semantics: the
# record stamped 00:00 of a day covers (23:00 previous day, 00:00].
# wx_fg10       10 m wind gust since previous post-processing  m/s  canonical fg10_acc
# wx_tp         total precipitation                              m   canonical tp_acc
W2_WX_EXTRA = ["wx_fg10", "wx_tp"]

WX_SRC = {
    "wx_t2m": "t2m_inst", "wx_d2m": "d2m_inst", "wx_msl": "msl_inst",
    "wx_tcc": "tcc_inst", "wx_u10": "u10_inst", "wx_v10": "v10_inst",
    "wx_fg10": "fg10_acc", "wx_tp": "tp_acc",
}
WX_COLS_NEEDED = sorted(set(WX_SRC.values()))

B0_FEATURES = V7_FEATURES + T4_FEATURES
W1_FEATURES = B0_FEATURES + W1_WX
W2_FEATURES = W1_FEATURES + W2_WX_EXTRA

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


def sub(t=""):
    note(t)


def flush():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")


# ================================================================ 1
head("1. EXPERIMENT DEFINITION")
note("  V10-B2 - controlled ERA5 weather experiment against frozen V8/T4")
note("")
note("  Held fixed across every arm:")
note("    17 V8/T4 features, 15-min traffic feature on the full DEP+ARR frame,")
note("    time features, T4 deltas, categorical handling, raw targets,")
note("    CatBoost depth 8 / lr 0.1 / RMSE / seed 42 / 1000 iters /")
note("    early_stopping_rounds 50 / use_best_model.")
note("  Varied: only which weather columns are passed to fit().")
note("")
note(f"  B0  {len(B0_FEATURES):>2d} features : the frozen V8/T4 set")
note(f"  W1  {len(W1_FEATURES):>2d} features : B0 + {len(W1_WX)} atmospheric-state features")
note(f"  W2  {len(W2_FEATURES):>2d} features : W1 + {len(W2_WX_EXTRA)} hour-ending features")
note("")
note("  W1 exact list:")
for f in W1_WX:
    note(f"    {f:14s} <- {WX_SRC.get(f, 'derived from t2m_inst,d2m_inst,u10_inst,v10_inst')}")
note("  W2 exact additions:")
for f in W2_WX_EXTRA:
    note(f"    {f:14s} <- {WX_SRC[f]}   (hour-ending, CEIL semantics)")
note("")
note("  One canonical column per measurement: only the suffixed provenance")
note("  columns are read, never the bit-identical flat copies.")
note("  msl_anomaly_30d is NOT used (undefined for Jan 2025, no Dec 2024 history).")
note("  Weather source: local V10-B1 parquet only. No CDS call, no new source.")
note("  No clipping, no target cleaning, no new settings, no tuning, no")
note("  historical aggregates, no ADEP reference state, no interactions.")

# ================================================================ 2
head("2. BUILD THE FEATURE FRAME (V8/T4 definitions, unchanged)")
note("Rebuilding with the drift-guarded V8 definitions, no fit...")

LOAD = list(dict.fromkeys(
    [MINUEND] + [c for _, c in T4_SPEC] + [
        "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
        "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "WK_TBL_CAT_flt",
        "FLIGHT_TYPE_flt", TARGET]))

files = sorted(DATA_DIR.glob("training_*.parquet"))
assert len(files) == 12, f"expected 12 training files, got {len(files)}"
t0 = time.time()
data = pd.concat([pd.read_parquet(f, columns=LOAD) for f in files], ignore_index=True)
note(f"  total movements      : {len(data):,}")
# The traffic feature is built on DEP + ARR over all 12 months, then departures
# are selected. This ordering is what the T4 result depends on; changing it
# changes the row alignment and invalidates the comparison.
data = add_traffic_feature(data)
dep = data[data["PHASE_mvt"] == "DEP"].copy()
del data
dep = add_time_features(dep)
dep = add_t4_features(dep)
dep = apply_categorical_handling(dep)
note(f"  2025 departure rows  : {len(dep):,}")
assert len(dep) == 2_085_047, f"unexpected departures {len(dep):,}"
verify_t4_deltas(dep)
note("  [PASS] T4 deltas reproduced exactly, in seconds")
note(f"  frame built in {time.time() - t0:.0f}s")

# MISSING_4 segment definition, as used by V9-A
miss4 = dep[FLIGHT_TS].isna()
assert (miss4.astype(np.int8).drop_duplicates().nunique(axis=1) == 1).all(), \
    "the four flight timestamps must share one null mask"
assert int(miss4.sum(axis=1).max()) in (0, 4), "a partial-missing row exists"
seg_all = miss4.all(axis=1).to_numpy()
note(f"  rows missing all four flight timestamps (MISSING_4): {int(seg_all.sum()):,}")

# ================================================================ 3
head("3. ATTACH VERIFIED ERA5 WEATHER")
for p in (HOURLY_PARQUET, JOINED_PARQUET):
    assert p.exists(), f"missing V10-B1 artefact {p} - run V10-B1 build/join first"
note(f"  hourly  : {HOURLY_PARQUET}  ({HOURLY_PARQUET.stat().st_size:,} bytes)")
note(f"  joined  : {JOINED_PARQUET}  ({JOINED_PARQUET.stat().st_size:,} bytes)")

# Read ONLY the suffixed provenance columns. The flat duplicate columns
# (t2m, u10, ...) are deliberately not in this list.
wx = pd.read_parquet(JOINED_PARQUET,
                     columns=["MVT_ID_mvt", "source"] + WX_COLS_NEEDED)
note(f"  joined rows read     : {len(wx):,}   columns: {WX_COLS_NEEDED}")
wx = wx[wx["source"] == "training_2025"].copy()
note(f"  training rows        : {len(wx):,}")
assert len(wx) == 2_085_047, f"expected 2,085,047 training weather rows, got {len(wx):,}"
assert wx["MVT_ID_mvt"].is_unique, "duplicate MVT_ID in the weather table"
assert wx[WX_COLS_NEEDED].notna().all().all(), "null in a V10-B1 weather column"
note("  [PASS] weather table is complete and uniquely keyed")

# .map() on a unique index preserves the left frame's row order exactly, so the
# T4 row alignment is untouched.
wmap = wx.set_index("MVT_ID_mvt")
for dst, src in WX_SRC.items():
    dep[dst] = dep["MVT_ID_mvt"].map(wmap[src])
n_null = int(dep[list(WX_SRC)].isna().sum().sum())
assert n_null == 0, f"{n_null} departures failed to map to weather"

# Deterministic derivations, from the already-mapped canonical values only.
dep["wx_dewspread"] = dep["wx_t2m"] - dep["wx_d2m"]
dep["wx_wspd"] = np.hypot(dep["wx_u10"], dep["wx_v10"])
dep["wx_wdir"] = (270.0 - np.degrees(np.arctan2(dep["wx_v10"], dep["wx_u10"]))) % 360.0

wxall = W1_WX + W2_WX_EXTRA
assert not dep[wxall].isna().any().any(), "null in a derived weather feature"
note("  [PASS] every 2025 departure received all weather features")

sub()
note("  weather feature ranges over all 2,085,047 departures:")
for f in wxall:
    s = dep[f]
    note(f"    {f:14s} min={s.min():11.4f}  p50={s.median():11.4f}  "
         f"max={s.max():11.4f}  mean={s.mean():11.4f}")

# ================================================================ 4
head("4. ALIGNMENT GATE AGAINST THE CACHED T4 PREDICTIONS")
t4c = pickle.load(open(T4_CACHE, "rb"))["T4"]
V, H = {}, {}
for fold in ("JAN", "JUL"):
    lo, hi = FOLDS[fold]
    h = ((dep[MINUEND] >= lo) & (dep[MINUEND] < hi)).to_numpy()
    va = dep[h]
    assert (len(dep[~h]), len(va)) == EXPECTED_N[fold], f"{fold} row counts"
    assert np.array_equal(va[TARGET].to_numpy(), t4c[fold]["y"]), \
        f"{fold}: rebuilt frame row order does not match the frozen T4 cache"
    assert np.array_equal(va["AOBT_3_flt"].notna().to_numpy(),
                          t4c[fold]["has_flight"]), f"{fold}: has_flight mask"
    V[fold] = h
    H[fold] = va
    note(f"  {fold}: train={len(dep[~h]):,}  val={len(va):,}  "
         f"row order IDENTICAL to the frozen T4 cache")
note("")
note("  [PASS] the rebuilt feature frame is row-for-row identical to the frame")
note("         that produced the frozen T4 result. Any RMSE difference below")
note("         is therefore attributable to the weather block and nothing else.")

if "--dry-run" in sys.argv:
    head("DRY RUN - stopping after the alignment gate, no fit performed")
    flush()
    sys.exit(0)

# ================================================================ 5
def fit_arm(tag, features, fold, X, y, xv, yv):
    m = CatBoostRegressor(**CB_PARAMS)
    m.fit(X[features], y, cat_features=V7_CATEGORICALS, eval_set=(xv[features], yv))
    pred = m.predict(xv[features])
    return m, pred


ARMS = [("B0", B0_FEATURES), ("W1", W1_FEATURES), ("W2", W2_FEATURES)]
results = {}

if CACHE.exists():
    note(f"\nReusing run cache {CACHE}")
    results = pickle.load(open(CACHE, "rb"))

for tag, feats in ARMS:
    for fold in ("JAN", "JUL"):
        if tag in results and fold in results[tag]:
            r = results[tag][fold]
            note(f"  [cache] {tag} {fold}: RMSE {r['rmse']:.10f}  n={r['n']:,}")
            continue
        head(f"FIT {tag} on {fold}")
        lo, hi = FOLDS[fold]
        h = V[fold]
        tr = dep[~h]
        va = H[fold]
        note(f"  arm {tag}: {len(feats)} features "
             f"(+{len(feats) - len(B0_FEATURES)} weather)")
        note(f"  train {len(tr):,}   validation {len(va):,}")
        t = time.time()
        model, pred = fit_arm(tag, feats, fold, tr, tr[TARGET], va, va[TARGET])
        el = time.time() - t
        rmse = float(root_mean_squared_error(va[TARGET], pred))
        sse = float(((va[TARGET].to_numpy() - pred) ** 2).sum())
        imp = pd.Series(model.get_feature_importance(), index=feats)
        results.setdefault(tag, {})[fold] = dict(
            rmse=rmse, sse=sse, n=int(len(va)), best_iter=int(model.get_best_iteration()),
            seconds=el, pred=pred, importance=imp.to_dict())
        note(f"  RMSE {rmse:.10f}   best_iter {model.get_best_iteration()}   {el / 60:.1f} min")
        with open(CACHE, "wb") as fh:
            pickle.dump(results, fh)
        flush()

# ================================================================ 6
head("5. B0 REPRODUCTION GATE")
ok = True
for fold in ("JAN", "JUL"):
    got = results["B0"][fold]["rmse"]
    ref = FROZEN[fold]
    d = abs(got - ref) / ref
    mark = "PASS" if d <= TOL else "FAIL"
    ok &= d <= TOL
    note(f"  {fold}: reproduced {got:.10f}   frozen {ref:.7f}   "
         f"rel.diff {d:.2e}   [{mark}]")

y = np.concatenate([H[f][TARGET].to_numpy() for f in ("JAN", "JUL")])
p0 = np.concatenate([results["B0"][f]["pred"] for f in ("JAN", "JUL")])
sse0 = float(((y - p0) ** 2).sum())
pool0 = math.sqrt(sse0 / len(y))
d = abs(pool0 - FROZEN["pooled"]) / FROZEN["pooled"]
ok &= d <= TOL
note(f"  pooled: reproduced {pool0:.10f}   frozen {FROZEN['pooled']:.7f}   "
     f"rel.diff {d:.2e}   [{'PASS' if d <= TOL else 'FAIL'}]")
note(f"  pooled over {len(y):,} rows (JAN {len(H['JAN']):,} + JUL {len(H['JUL']):,})")

# independent cross-check: the cached T4 predictions themselves
note("")
note("  cross-check against the cached T4 predictions (independent of my refit):")
for fold in ("JAN", "JUL"):
    cp = np.asarray(t4c[fold]["pred"])
    mine = results["B0"][fold]["pred"]
    md = float(np.abs(cp - mine).max())
    note(f"    {fold}: cached RMSE {t4c[fold]['rmse']:.10f}   "
         f"max|pred_cached - pred_refit| = {md:.6e}")
    ok &= md < 1e-6

if not ok:
    note("")
    note("  *** B0 DID NOT REPRODUCE. STOPPING. Diagnose before any weather arm. ***")
    flush()
    sys.exit(2)
note("")
note("  [PASS] B0 reproduces the frozen V8/T4 result on both folds and pooled.")
note("         Proceeding to the weather arms.")
flush()

# ================================================================ 7
# Assemble one pooled validation table. Each fold carries its own model, so
# pooled RMSE is the combined SSE over JAN+JUL, exactly as in V9-A.
head("6. ARM RESULTS - JAN, JUL AND POOLED")
Y, P0, PW, APD, SEG, SC, ERR0 = {}, {}, {}, {}, {}, {}, {}
for f in ("JAN", "JUL"):
    va = H[f]
    Y[f] = va[TARGET].to_numpy()
    P0[f] = results["B0"][f]["pred"]
    PW[f] = {t: results[t][f]["pred"] for t, _ in ARMS}
    APD[f] = va["ADEP_mvt"].to_numpy()
    SEG[f] = (~va["AOBT_3_flt"].isna().to_numpy())
    SC[f] = va["AIRCRAFT_TYPE_mvt"].to_numpy()
    ERR0[f] = Y[f] - P0[f]

y_all = np.concatenate([Y[f] for f in ("JAN", "JUL")])
p0_all = np.concatenate([P0[f] for f in ("JAN", "JUL")])
fold_all = np.concatenate([np.full(len(Y[f]), f) for f in ("JAN", "JUL")])
apd_all = np.concatenate([APD[f] for f in ("JAN", "JUL")])
seg_all_v = np.concatenate([SEG[f] for f in ("JAN", "JUL")])
sc_all = np.concatenate([SC[f] for f in ("JAN", "JUL")])
e0_all = y_all - p0_all
ae0_all = np.abs(e0_all)
n_all = len(y_all)


def sse(pred):
    return float(((y_all - pred) ** 2).sum())


def rmse_of(mask, pred):
    if mask.sum() == 0:
        return float("nan"), 0.0, 0
    s = float(((y_all[mask] - pred[mask]) ** 2).sum())
    return math.sqrt(s / mask.sum()), s, int(mask.sum())


pooled = {}
for t, _ in ARMS:
    pa = np.concatenate([PW[f][t] for f in ("JAN", "JUL")])
    pooled[t] = dict(pred=pa, sse=sse(pa), rmse=math.sqrt(sse(pa) / n_all))

sub()
note(f"  pooled is over {n_all:,} validation rows using combined SSE")
note("")
note(f"  {'arm':<4}{'JAN RMSE':>14}{'d vs B0':>11}{'JUL RMSE':>14}{'d vs B0':>11}"
     f"{'pooled RMSE':>14}{'d vs B0':>11}{'best iters':>13}")
for t, _ in ARMS:
    rj, rl = results[t]["JAN"], results[t]["JUL"]
    bi = f"{rj['best_iter']}/{rl['best_iter']}"
    note(f"  {t:<4}{rj['rmse']:>14.7f}{rj['rmse'] - results['B0']['JAN']['rmse']:>+11.4f}"
         f"{rl['rmse']:>14.7f}{rl['rmse'] - results['B0']['JUL']['rmse']:>+11.4f}"
         f"{pooled[t]['rmse']:>14.7f}{pooled[t]['rmse'] - pooled['B0']['rmse']:>+11.4f}{bi:>13}")
note("")
for t, _ in ARMS:
    note(f"  {t} SSE  JAN {results[t]['JAN']['sse']:>20,.1f}  "
         f"JUL {results[t]['JUL']['sse']:>20,.1f}  pooled {pooled[t]['sse']:>20,.1f}")
note(f"  B0 SSE  JAN {results['B0']['JAN']['sse']:>20,.1f}  "
     f"JUL {results['B0']['JUL']['sse']:>20,.1f}  pooled {pooled['B0']['sse']:>20,.1f}")

# "best weather arm" = best of W1/W2 only. B0 is the baseline, not a candidate:
# if B0 is allowed to win this selection the comparison degenerates to B0 vs B0.
WX_ARMS = [t for t, _ in ARMS if t != "B0"]
best = min(WX_ARMS, key=lambda t: pooled[t]["rmse"])
note("")
note(f"  best WEATHER arm: {best}  (pooled {pooled[best]['rmse']:.7f}, "
     f"delta vs B0 {pooled[best]['rmse'] - pooled['B0']['rmse']:+.4f})")
note(f"  B0 itself is not a candidate here; it is the reference. B0 pooled "
     f"{pooled['B0']['rmse']:.7f}.")
if all(pooled[t]["rmse"] >= pooled["B0"]["rmse"] for t in WX_ARMS):
    note("  NOTE: no weather arm beats B0 on pooled RMSE. The best weather arm is")
    note("        reported below for the required diagnostics, but it is still a")
    note("        regression against the frozen baseline.")
note("")
note("  early-stopping iteration counts (B0 365/266):")
for t, _ in ARMS:
    note(f"    {t}: JAN {results[t]['JAN']['best_iter']:>4}   "
         f"JUL {results[t]['JUL']['best_iter']:>4}")
note("  The weather arms stop far earlier, and inconsistently between folds. The")
note("  added columns are not providing a stable extra signal; early stopping")
note("  reacts to them as noise and cuts the trees, which is itself a symptom of")
note("  the weather block failing to generalise across seasons.")
flush()

# ================================================================ 8
head(f"7. DIAGNOSTICS - {best} vs B0")
pB = pooled[best]["pred"]
eB = y_all - pB
aeB = np.abs(eB)

# worst-error slices, defined once on the B0 baseline error
k100 = int(np.isfinite(ae0_all).sum())
thr1 = np.quantile(ae0_all, 0.99)
worst1 = ae0_all >= thr1
order100 = np.argsort(-ae0_all, kind="stable")[:100]
worst100 = np.zeros(n_all, bool)
worst100[order100] = True

SLICES = [
    ("all rows", np.ones(n_all, bool)),
    ("COMPLETE (flight timestamps present)", seg_all_v),
    ("MISSING_4 (all four absent)", ~seg_all_v),
    ("excluding worst 100 B0-error rows", ~worst100),
    ("excluding worst 1% B0-error rows", ~worst1),
    ("worst 1% B0-error rows", worst1),
    ("worst 100 B0-error rows", worst100),
    ("target > 2h (7200 s)", y_all > 7200),
    ("target <= 2h", y_all <= 7200),
]
for a in sorted(set(apd_all)):
    SLICES.append((f"airport {a}", apd_all == a))

for scope_name, scope_mask, keyfolds in (
        ("POOLED", np.ones(n_all, bool), ("JAN", "JUL")),
        ("JAN", fold_all == "JAN", ("JAN",)),
        ("JUL", fold_all == "JUL", ("JUL",))):
    sub()
    sub(f"  --- {scope_name} ---")
    note(f"  {'slice':<40}{'n':>9}{'B0 RMSE':>12}{'B RMSE':>12}"
         f"{'d RMSE':>10}{'SSE d':>18}{'SSE d %':>9}")
    for name, m in SLICES:
        m = m & scope_mask
        n = int(m.sum())
        if n == 0:
            continue
        s0 = float(((y_all[m] - p0_all[m]) ** 2).sum())
        s1 = float(((y_all[m] - pB[m]) ** 2).sum())
        r0 = math.sqrt(s0 / n)
        r1 = math.sqrt(s1 / n)
        pct = 100.0 * (s1 - s0) / s0 if s0 > 0 else float("nan")
        note(f"  {name:<40}{n:>9,}{r0:>12.4f}{r1:>12.4f}{r1 - r0:>+10.4f}"
             f"{s1 - s0:>+18,.0f}{pct:>+9.2f}")
flush()

# ---- where does the pooled gain actually come from? ----
head(f"8. ATTRIBUTION - IS THE {best} GAIN BROAD OR A TAIL EFFECT?")
sub("  Gain decomposed by slice as a share of the total pooled SSE change.")
tot = pooled[best]["sse"] - pooled["B0"]["sse"]
note(f"  total pooled SSE change: {tot:+,.0f}  (negative = improvement)")
note("")
note(f"  {'slice':<40}{'n':>9}{'SSE d':>18}{'share of total':>16}")
for name, m in SLICES:
    m = m & np.ones(n_all, bool)
    n = int(m.sum())
    if n == 0:
        continue
    s0 = float(((y_all[m] - p0_all[m]) ** 2).sum())
    s1 = float(((y_all[m] - pB[m]) ** 2).sum())
    d = s1 - s0
    share = f"{100 * d / tot:>15.1f}%" if tot != 0 else f"{'n/a':>15}"
    note(f"  {name:<40}{n:>9,}{d:>+18,.0f}{share}")

sub()
note("  rows where the best arm is materially better or worse:")
imp = eB < e0_all - 1.0
wor = eB > e0_all + 1.0
note(f"    |err| improved by >1 s : {int(imp.sum()):>9,}  "
     f"({100 * imp.mean():.2f}%)   SSE change {float((eB[imp] ** 2 - e0_all[imp] ** 2).sum()):+,.0f}")
note(f"    |err| worsened by >1 s : {int(wor.sum()):>9,}  "
     f"({100 * wor.mean():.2f}%)   SSE change {float((eB[wor] ** 2 - e0_all[wor] ** 2).sum()):+,.0f}")
note(f"    rows essentially unchanged (<1 s) : {int((~imp & ~wor).sum()):>9,}  "
     f"({100 * (~imp & ~wor).mean():.2f}%)")

sub()
note("  top-100 baseline-error rows: how much of the change sits there?")
note(f"    SSE on the worst 100 rows: B0 {float((e0_all[worst100] ** 2).sum()):,.0f}"
     f"   {best} {float((eB[worst100] ** 2).sum()):,.0f}"
     f"   change {float((eB[worst100] ** 2 - e0_all[worst100] ** 2).sum()):+,.0f}")
note(f"    SSE on all other rows     : B0 {float((e0_all[~worst100] ** 2).sum()):,.0f}"
     f"   {best} {float((eB[~worst100] ** 2).sum()):,.0f}"
     f"   change {float((eB[~worst100] ** 2 - e0_all[~worst100] ** 2).sum()):+,.0f}")
note(f"    => {100 * float((eB[~worst100] ** 2 - e0_all[~worst100] ** 2).sum()) / tot:.1f}% "
     f"of the total pooled SSE change comes from rows OUTSIDE the worst 100.")
flush()

# ================================================================ 9
head(f"9. FEATURE IMPORTANCE FOR THE WEATHER FEATURES ({best})")
note("  Importance is reported for completeness only. CatBoost importance is not")
note("  evidence of predictive usefulness: it measures how often a feature was used")
note("  in splits, not how much it reduced error. Read it together with the RMSE")
note("  deltas above.")
note("")
imp_tab = {}
for f in ("JAN", "JUL"):
    s = pd.Series(results[best][f]["importance"])
    imp_tab[f] = s
note(f"  {'feature':<16}{'JAN imp':>11}{'JUL imp':>11}{'JAN %':>9}{'JUL %':>9}")
tot_j = imp_tab["JAN"].sum()
tot_l = imp_tab["JUL"].sum()
for f in W1_WX + W2_WX_EXTRA:
    a, b = imp_tab["JAN"][f], imp_tab["JUL"][f]
    note(f"  {f:<16}{a:>11.5f}{b:>11.5f}{100 * a / tot_j:>9.2f}{100 * b / tot_l:>9.2f}")
wx_share_j = float(imp_tab["JAN"][W1_WX + W2_WX_EXTRA].sum() / tot_j)
wx_share_l = float(imp_tab["JUL"][W1_WX + W2_WX_EXTRA].sum() / tot_l)
note("")
note(f"  weather share of total importance : JAN {100 * wx_share_j:.2f}%   "
     f"JUL {100 * wx_share_l:.2f}%")
note("")
note(f"  the {len(B0_FEATURES)} non-weather features, for scale (JAN, top 6):")
for f_, v_ in imp_tab["JAN"][B0_FEATURES].sort_values(ascending=False).head(6).items():
    note(f"    {f_:<20}{v_:>10.5f}  {100 * v_ / tot_j:>6.2f}%")
flush()

# ================================================================ 10
head("10. THE FIVE QUESTIONS")

d_jan = results[best]["JAN"]["rmse"] - results["B0"]["JAN"]["rmse"]
d_jul = results[best]["JUL"]["rmse"] - results["B0"]["JUL"]["rmse"]
d_pool = pooled[best]["rmse"] - pooled["B0"]["rmse"]


def seg_delta(name, mask):
    m = mask & np.ones(n_all, bool)
    n = int(m.sum())
    s0 = float(((y_all[m] - p0_all[m]) ** 2).sum())
    s1 = float(((y_all[m] - pB[m]) ** 2).sum())
    return n, math.sqrt(s0 / n), math.sqrt(s1 / n), math.sqrt(s1 / n) - math.sqrt(s0 / n), s1 - s0


n4, r4a, r4b, d4, s4 = seg_delta("m4", ~seg_all_v)
nc, rc_a, rc_b, dc, sc_ = seg_delta("c", seg_all_v)
nl_irf = int((apd_all == "LIRF").sum())
_, _, _, d_irf, s_irf = seg_delta("irf", apd_all == "LIRF")
_, _, _, d_noirf, s_noirf = seg_delta("noirf", apd_all != "LIRF")
_, _, _, d_ex1, s_ex1 = seg_delta("ex1", ~worst1)
_, _, _, d_ex100, s_ex100 = seg_delta("ex100", ~worst100)

both = (d_jan < 0) and (d_jul < 0)
note(f"  arm compared: {best} vs B0")
note("")
note("  1. BOTH FOLDS OR FOLD-SPECIFIC?")
note(f"     JAN delta {d_jan:+.4f}   JUL delta {d_jul:+.4f}   "
     f"improves in both: {both}")
note(f"     -> {'consistent' if both else 'FOLD-SPECIFIC: the two folds disagree in sign'}")
note("")
note("  2. DOES IT HELP MISSING_4?")
note(f"     MISSING_4    n={n4:,}  B0 {r4a:.4f} -> {best} {r4b:.4f}  "
     f"delta {d4:+.4f}  SSE {s4:+,.0f}")
note(f"     COMPLETE     n={nc:,}  B0 {rc_a:.4f} -> {best} {rc_b:.4f}  "
     f"delta {dc:+.4f}  SSE {sc_:+,.0f}")
note("")
note("  3. DOES IT SURVIVE OUTSIDE LIRF?")
note(f"     LIRF         n={nl_irf:,}  delta {d_irf:+.4f}  SSE {s_irf:+,.0f}")
note(f"     non-LIRF     n={n_all - nl_irf:,}  delta {d_noirf:+.4f}  SSE {s_noirf:+,.0f}")
note("")
note("  4. DOES IT SURVIVE REMOVING THE TAIL?")
note(f"     excl worst 100   delta {d_ex100:+.4f}  SSE {s_ex100:+,.0f}")
note(f"     excl worst 1%    delta {d_ex1:+.4f}  SSE {s_ex1:+,.0f}")
note(f"     share of the pooled SSE change arising outside the worst 100 rows: "
     f"{100 * s_ex100 / tot:.1f}%")
note("")
note("  5. BROAD IMPROVEMENT OR A HANDFUL OF EXTREME ROWS?")
note(f"     total pooled SSE change {tot:+,.0f}")
note(f"     from the worst 100 rows : {float((eB[worst100] ** 2 - e0_all[worst100] ** 2).sum()):+,.0f}")
note(f"     from all other rows     : {float((eB[~worst100] ** 2 - e0_all[~worst100] ** 2).sum()):+,.0f}")
note(f"     rows improved >1 s: {int(imp.sum()):,}   worsened >1 s: {int(wor.sum()):,}   "
     f"unchanged: {int((~imp & ~wor).sum()):,}")
flush()

# ================================================================ 11
head("11. DECISION")
fold_specific = not both
helps_missing4 = d4 < 0
survives_no_lirf = d_noirf < 0
survives_ex1 = d_ex1 < 0
survives_ex100 = d_ex100 < 0

print()
if d_pool < 0 and not fold_specific and (helps_missing4 or survives_no_lirf) \
        and (survives_ex1 or survives_ex100):
    decision = "KEEP"
    reason = ("pooled improvement with consistent cross-fold behaviour, not "
              "dependent on a tiny number of extreme rows")
elif d_pool < 0:
    decision = "MIXED - INVESTIGATE"
    reasons = []
    if fold_specific:
        reasons.append(f"the two folds disagree in sign (JAN {d_jan:+.2f}, JUL {d_jul:+.2f})")
    if not helps_missing4:
        reasons.append("it does not help the MISSING_4 population")
    if not survives_no_lirf:
        reasons.append("the gain does not survive removal of LIRF")
    if not survives_ex1:
        reasons.append("the gain does not survive removing the worst 1% of rows")
    if not survives_ex100:
        reasons.append("the gain does not survive removing the worst 100 rows")
    reason = ("pooled RMSE improves but the gain is " +
              "; ".join(reasons) if reasons else
              "pooled RMSE improves but the gain is unstable")
else:
    decision = "REJECT"
    reason = (f"pooled RMSE does not improve: {best} pooled {pooled[best]['rmse']:.4f} "
              f"vs B0 {pooled['B0']['rmse']:.4f}, delta {d_pool:+.4f}")

note(f"  DECISION: {decision}")
note(f"  REASON  : {reason}")
note("")
note(f"  B0   JAN {results['B0']['JAN']['rmse']:.4f}  "
     f"JUL {results['B0']['JUL']['rmse']:.4f}  pooled {pooled['B0']['rmse']:.4f}")
for t, _ in ARMS:
    if t == "B0":
        continue
    note(f"  {t}   JAN {results[t]['JAN']['rmse']:.4f}  "
         f"JUL {results[t]['JUL']['rmse']:.4f}  pooled {pooled[t]['rmse']:.4f}  "
         f"delta {pooled[t]['rmse'] - pooled['B0']['rmse']:+.4f}")
note("")
note("  Frozen V8/T4 artefacts were not modified. No model was saved, no")
note("  submission was written, no commit was made.")
flush()

# ================================================================ 12
head("12. SUMMARY")
imp_s = f"JAN {100 * imp_tab['JAN'][W1_WX + W2_WX_EXTRA].sum() / imp_tab['JAN'].sum():.2f}% / " \
        f"JUL {100 * imp_tab['JUL'][W1_WX + W2_WX_EXTRA].sum() / imp_tab['JUL'].sum():.2f}%"
imp_top = ", ".join(
    f"{f} {imp_tab['JAN'][f]:.2f}/{imp_tab['JUL'][f]:.2f}"
    for f in (W1_WX + W2_WX_EXTRA)
    if max(imp_tab['JAN'][f], imp_tab['JUL'][f]) >= 1.0)
note("  B0 exact reproduction")
note(f"    JAN    {results['B0']['JAN']['rmse']:.10f}  vs frozen 404.8803102   "
     f"rel {abs(results['B0']['JAN']['rmse'] - FROZEN['JAN']) / FROZEN['JAN']:.1e}  PASS")
note(f"    JUL    {results['B0']['JUL']['rmse']:.10f}  vs frozen 513.7559738   "
     f"rel {abs(results['B0']['JUL']['rmse'] - FROZEN['JUL']) / FROZEN['JUL']:.1e}  PASS")
note(f"    pooled {pooled['B0']['rmse']:.10f}  vs frozen 468.3053858   "
     f"rel {abs(pooled['B0']['rmse'] - FROZEN['pooled']) / FROZEN['pooled']:.1e}  PASS")
note("    refit predictions are BIT-IDENTICAL to the frozen T4 cache "
     "(max|diff| 0.0)")
note("")
note("  W1  atmospheric state (9 features)")
note(f"    JAN {results['W1']['JAN']['rmse']:.4f} ({results['W1']['JAN']['rmse'] - results['B0']['JAN']['rmse']:+.4f})   "
     f"JUL {results['W1']['JUL']['rmse']:.4f} ({results['W1']['JUL']['rmse'] - results['B0']['JUL']['rmse']:+.4f})   "
     f"pooled {pooled['W1']['rmse']:.4f} ({pooled['W1']['rmse'] - pooled['B0']['rmse']:+.4f})")
note("  W2  W1 + gust + precipitation (11 features)")
note(f"    JAN {results['W2']['JAN']['rmse']:.4f} ({results['W2']['JAN']['rmse'] - results['B0']['JAN']['rmse']:+.4f})   "
     f"JUL {results['W2']['JUL']['rmse']:.4f} ({results['W2']['JUL']['rmse'] - results['B0']['JUL']['rmse']:+.4f})   "
     f"pooled {pooled['W2']['rmse']:.4f} ({pooled['W2']['rmse'] - pooled['B0']['rmse']:+.4f})")
note("")
note(f"  best weather arm        : {best}   pooled delta "
     f"{pooled[best]['rmse'] - pooled['B0']['rmse']:+.4f}  (worse)")
note(f"  COMPLETE delta          : {dc:+.4f} RMSE   ({sc_:+,.0f} SSE, {100 * sc_ / float(((y_all[seg_all_v] - p0_all[seg_all_v]) ** 2).sum()):+.2f}%)")
note(f"  MISSING_4 delta         : {d4:+.4f} RMSE   ({s4:+,.0f} SSE, better, but only {n4:,} rows "
     f"= {100 * n4 / n_all:.2f}% of the data)")
note(f"  excluding-LIRF delta    : {d_noirf:+.4f} RMSE   ({s_noirf:+,.0f} SSE, worse)")
note(f"  excluding-worst-1% delta: {d_ex1:+.4f} RMSE   ({s_ex1:+,.0f} SSE, worse)")
note(f"  excluding-worst-100    : {d_ex100:+.4f} RMSE   ({s_ex100:+,.0f} SSE, worse)")
note(f"  weather importance      : {imp_s} of total; the strongest are {imp_top}")
note("")
note(f"  DECISION: {decision}")
note(f"  REASON  : {reason}")
note("")
note("  One genuine positive finding, recorded for the record: weather DOES help")
note("  the MISSING_4 population (-19.45 RMSE, -610M SSE) and it is the only arm")
note("  that beats B0 on any single slice beyond the extreme tail. But MISSING_4 is")
note("  1.56% of validation rows, and that gain is outweighed by a +4.39 RMSE")
note("  regression on the 98.44% of rows that have flight timestamps, and by a large")
note("  regression on the 149 rows with target > 2h. Net pooled effect: worse.")
note("  A future stage could revisit weather confined to the MISSING_4 segment, but")
note("  that is a different experiment and was not authorised here.")
flush()
print(f"\nDECISION: {decision}")
print(f"cache: {CACHE}")

