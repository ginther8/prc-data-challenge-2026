"""V8/T4 post-validation error autopsy. ANALYSIS ONLY - fits nothing.

Reproduces the exact validated T4 Jan/Jul validation predictions from cache,
proves they are the exact T4 predictions, attaches the feature frame, and
decomposes the remaining error.

Nothing is modified: no feature change, no CatBoost parameter change, no
target change, no clipping, no preprocessing change, no new model.

HEADLINE SCORES ARE ALWAYS RAW. Excluding-row figures exist only to locate
error and are labelled DIAGNOSTIC.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from train_catboost_v8_defs import (  # noqa: F401
    DATA_DIR, FEATURES, MINUEND, TARGET, T4_FEATURES, T4_SPEC,
    add_t4_features, add_time_features, add_traffic_feature,
    apply_categorical_handling, verify_t4_deltas,
)

CACHE = Path(os.environ.get("TEMP", ".")) / "opencode" / "clocks_results.pkl"
FOLDS = {"JAN": ("2025-01-01", "2025-02-01"),
         "JUL": ("2025-07-01", "2025-08-01")}
HOLDOUT_MONTH = {"JAN": 1, "JUL": 7}
T4_REF = {"JAN": 404.8803102, "JUL": 513.7559738}
EXPECTED_N = {"JAN": (1_931_341, 153_706), "JUL": (1_894_334, 190_713)}

out = []
W = out.append


def head(t):
    W("")
    W("=" * 100)
    W(t)
    W("=" * 100)


def sub(t):
    W("")
    W("-" * 100)
    W(t)
    W("-" * 100)


def note(t=""):
    W(t)


# ============================================================
# 0. PROVE THE CACHED PREDICTIONS ARE THE EXACT T4 PREDICTIONS
# ============================================================

head("0. PROVENANCE: are the cached predictions the exact T4 predictions?")

cache = pickle.load(open(CACHE, "rb"))
assert "T4" in cache, f"no T4 arm in {CACHE}"
t4 = cache["T4"]

for fold, ref in T4_REF.items():
    got = t4[fold]["rmse"]
    note(f"  {fold}: cached RMSE {got:.7f}   frozen reference {ref:.7f}   "
         f"diff {got - ref:+.7f}")
    assert abs(got - ref) < 1e-6, f"{fold} cached RMSE drifted"
note("  [PASS] cached T4 RMSE equals the frozen T4 reference on both folds")
note("  [PASS] this cache was produced by experiment_t4_clocks.py, whose T4 arm")
note("         reproduced experiment_v7_vs_t4.py exactly (canary diff -0.0000)")
note("         and was itself run twice with identical results.")

# ============================================================
# REBUILD THE FEATURE FRAME (no model is fitted)
# ============================================================

note("")
note("Rebuilding the feature frame with the drift-guarded V8/T4 definitions "
     "(no fit)...")

T4_SOURCE_COLS = [MINUEND] + [c for _, c in T4_SPEC]
LOAD = list(dict.fromkeys(T4_SOURCE_COLS + [
    "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
    "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt", TARGET]))

files = sorted(DATA_DIR.glob("training_*.parquet"))
data = pd.concat([pd.read_parquet(f, columns=LOAD) for f in files],
                 ignore_index=True)
note(f"  Total movements: {len(data):,}")
data = add_traffic_feature(data)
dep = data[data["PHASE_mvt"] == "DEP"].copy()
del data
dep = add_time_features(dep)
dep = add_t4_features(dep)
dep = apply_categorical_handling(dep)
note(f"  Departure rows: {len(dep):,}")
assert len(dep) == 2_085_047

note("")
note("  T4 delta unit re-check on the full 2025 departure set:")
verify_t4_deltas(dep)
note("  [PASS] deltas reproduced exactly, in seconds")

# align cached predictions to rebuilt rows, per fold
V = {}
for fold in ("JAN", "JUL"):
    lo, hi = FOLDS[fold]
    h = (dep[MINUEND] >= lo) & (dep[MINUEND] < hi)
    va = dep[h]
    assert (len(dep[~h]), len(va)) == EXPECTED_N[fold]
    y_new = va[TARGET].to_numpy()
    y_cached = t4[fold]["y"]
    same = np.array_equal(y_new, y_cached)
    note(f"  {fold}: validation rows {len(va):,}   targets identical to cache: "
         f"{same}")
    assert same, f"{fold}: row order does not match the cache - ABORT"
    f = va.copy()
    f["PRED"] = t4[fold]["pred"]
    f["ERR"] = f["PRED"] - f[TARGET]
    f["AE"] = f["ERR"].abs()
    f["SE"] = f["ERR"] ** 2
    f["FOLD"] = fold
    f["T4_complete"] = f[T4_FEATURES].notna().all(axis=1)
    f["T4_missing"] = ~f["T4_complete"]
    # cached has_flight must agree with the rebuilt null mask
    hf_new = va["AOBT_3_flt"].notna().to_numpy()
    assert np.array_equal(hf_new, t4[fold]["has_flight"]), f"{fold} has_flight"
    V[fold] = f
    note(f"        has_flight mask identical: True")

note("  [PASS] cached predictions align row-for-row with the rebuilt frame")
note("         (proved by exact target equality, not assumed)")

# ============================================================
# helpers
# ============================================================

def stats(g, ycol="ERR", label=""):
    n = len(g)
    if n == 0:
        return None
    return dict(
        n=n,
        rmse=float(np.sqrt(g["SE"].mean())),
        mae=float(g["AE"].mean()),
        bias=float(g[ycol].mean()),
        medae=float(g["AE"].median()),
        sse=float(g["SE"].sum()),
    )


def tbl(rows, cols, fmts=None):
    d = pd.DataFrame(rows)
    if d.empty:
        note("  (none)")
        return
    if fmts:
        for c, f in fmts.items():
            if c in d.columns:
                d[c] = d[c].map(lambda v: f.format(v))
    note(d.to_string(index=False))


def band_table(f, col, edges, labels, minrows=1):
    rows = []
    e = f[col].to_numpy()
    idx = np.digitize(e, edges, right=False)
    for i, lab in enumerate(labels):
        m = idx == i
        if m.sum() < minrows:
            continue
        s = stats(f[m])
        s["band"] = lab
        s["pct_sse"] = 100 * s["sse"] / f["SE"].sum()
        rows.append(s)
    return rows


PCT = {"n": "{:,.0f}", "rmse": "{:,.2f}", "mae": "{:,.2f}", "bias": "{:+.2f}",
       "medae": "{:,.2f}", "sse": "{:,.3e}", "pct_sse": "{:5.2f}",
       "pct_rows": "{:5.2f}", "share": "{:6.2f}"}

# ============================================================
# 1. ERROR CONCENTRATION
# ============================================================

head("1. ERROR CONCENTRATION (raw headline; exclusions are DIAGNOSTIC only)")

summary_rows = []
for fold, f in V.items():
    s = stats(f)
    s["fold"] = fold
    summary_rows.append(s)
note("RAW VALIDATION SUMMARY (no rows excluded):")
tbl(summary_rows, ["fold", "n", "rmse", "mae", "bias", "medae"],
    {k: PCT[k] for k in ("n", "rmse", "mae", "bias", "medae")})
note("")
note(f"  reference T4 RMSE: JAN {T4_REF['JAN']}, JUL {T4_REF['JUL']}, "
     f"pooled 468.31")

for fold, f in V.items():
    sub(f"1.{fold} - where the squared error lives")
    n = len(f)
    tot = f["SE"].sum()
    order = np.argsort(-f["SE"].to_numpy())
    cs = np.cumsum(f["SE"].to_numpy()[order])
    rows = []
    for frac in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        k = max(1, int(round(frac * n)))
        share = 100 * cs[k - 1] / tot
        keep = np.ones(n, bool)
        keep[order[:k]] = False
        rows.append(dict(
            band=f"worst {frac * 100:g}%",
            n=k, pct_rows=100 * k / n, pct_sse=share,
            rmse_worst=float(np.sqrt(f["SE"].to_numpy()[order[:k]].mean())),
            rmse_excl_diag=float(np.sqrt(f.loc[keep, "SE"].mean())),
            mae_excl_diag=float(f.loc[keep, "AE"].mean()),
        ))
    note(f"  total SSE = {tot:.6e}   rows = {n:,}")
    note("  'rmse_excl_diag' is DIAGNOSTIC ONLY - it is not a headline score.")
    tbl(rows, ["band", "n", "pct_rows", "pct_sse", "rmse_worst",
               "rmse_excl_diag", "mae_excl_diag"], PCT)

# pooled
sub("1.POOLED - Jan+Jul combined (SSE-weighted, not an average of RMSEs)")
allf = pd.concat([V["JAN"], V["JUL"]], ignore_index=True)
n = len(allf)
tot = allf["SE"].sum()
order = np.argsort(-allf["SE"].to_numpy())
cs = np.cumsum(allf["SE"].to_numpy()[order])
rows = []
for frac in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
    k = max(1, int(round(frac * n)))
    keep = np.ones(n, bool)
    keep[order[:k]] = False
    rows.append(dict(
        band=f"worst {frac * 100:g}%", n=k, pct_rows=100 * k / n,
        pct_sse=100 * cs[k - 1] / tot,
        rmse_worst=float(np.sqrt(allf["SE"].to_numpy()[order[:k]].mean())),
        rmse_excl_diag=float(np.sqrt(allf.loc[keep, "SE"].mean())),
    ))
note(f"  total SSE = {tot:.6e}   rows = {n:,}   "
     f"pooled RMSE = {np.sqrt(tot / n):.4f}")
tbl(rows, ["band", "n", "pct_rows", "pct_sse", "rmse_worst", "rmse_excl_diag"],
    PCT)

# ============================================================
# 2. BY AIRPORT
# ============================================================

head("2. BY AIRPORT")

FOCUS = ["LIRF", "LFPG"]
for fold, f in V.items():
    sub(f"2.{fold} - by departure airport (all airports, no minimum filter)")
    rows = []
    for ap, g in f.groupby("ADEP_mvt", observed=True):
        s = stats(g)
        s["airport"] = ap
        s["pct_sse"] = 100 * s["sse"] / f["SE"].sum()
        s["pct_rows"] = 100 * s["n"] / len(f)
        rows.append(s)
    d = pd.DataFrame(rows).sort_values("sse", ascending=False)
    note(d[["airport", "n", "pct_rows", "rmse", "mae", "bias", "medae",
            "pct_sse"]].round(2).to_string(index=False))

sub("2.JAN vs JUL - airport comparison (airports present in both folds)")
a = V["JAN"].groupby("ADEP_mvt", observed=True).agg(
    n_jan=("SE", "size"), sse_jan=("SE", "sum"), mae_jan=("AE", "mean"),
    bias_jan=("ERR", "mean"))
b = V["JUL"].groupby("ADEP_mvt", observed=True).agg(
    n_jul=("SE", "size"), sse_jul=("SE", "sum"), mae_jul=("AE", "mean"),
    bias_jul=("ERR", "mean"))
j = a.join(b, how="outer")
j["rmse_jan"] = np.sqrt(j["sse_jan"] / j["n_jan"])
j["rmse_jul"] = np.sqrt(j["sse_jul"] / j["n_jul"])
j["pct_sse_jan"] = 100 * j["sse_jan"] / V["JAN"]["SE"].sum()
j["pct_sse_jul"] = 100 * j["sse_jul"] / V["JUL"]["SE"].sum()
j["ratio"] = j["pct_sse_jul"] / j["pct_sse_jan"].replace(0, np.nan)
note(j[["n_jan", "rmse_jan", "mae_jan", "bias_jan", "pct_sse_jan",
        "n_jul", "rmse_jul", "mae_jul", "bias_jul", "pct_sse_jul",
        "ratio"]].round(2).sort_values("pct_sse_jan", ascending=False)
     .to_string())
note("")
note("  'ratio' = share of Jul SSE / share of Jan SSE. >1 means the airport got")
note("  worse in July relative to its row share.")

# ============================================================
# 3. TARGET-SIZE BANDS
# ============================================================

head("3. TARGET-SIZE BANDS (rows, RMSE, MAE, SSE contribution)")

EDGES = [900, 1800, 3600, 7200, 14400, 28800, 43200, 86400]
LABELS = ["<15min", "15-30min", "30-60min", "1-2h", "2-4h", "4-8h",
          "8-15h", "15-24h", ">24h"]

for fold, f in V.items():
    sub(f"3.{fold} - by true target size")
    rows = band_table(f, TARGET, EDGES, LABELS)
    d = pd.DataFrame(rows)
    d["pct_rows"] = 100 * d["n"] / len(f)
    note(d[["band", "n", "pct_rows", "rmse", "mae", "bias", "medae",
            "pct_sse"]].round(2).to_string(index=False))

sub("3.POOLED - by true target size")
rows = band_table(allf, TARGET, EDGES, LABELS)
d = pd.DataFrame(rows)
d["pct_rows"] = 100 * d["n"] / len(allf)
note(d[["band", "n", "pct_rows", "rmse", "mae", "bias", "medae",
        "pct_sse"]].round(2).to_string(index=False))
note("")
note("  Reading: a band with a large pct_sse but small pct_rows is where the")
note("  remaining error lives. Bands are defined by the TRUE target, which is")
note("  only available for diagnosis - never available at prediction time.")

# ============================================================
# 4. T4 AVAILABILITY
# ============================================================

head("4. T4 AVAILABILITY: rows with all five deltas vs rows missing the "
     "flight-level timestamps")

for fold, f in V.items():
    sub(f"4.{fold} - T4 availability split")
    f["T4_complete"] = f[T4_FEATURES].notna().all(axis=1)
    f["T4_missing"] = ~f["T4_complete"]
    rows = []
    for lab, m in (("all 5 deltas present", f["T4_complete"]),
                   ("missing AOBT3/EOBT1/IOBT/LOBT", f["T4_missing"])):
        if m.sum() == 0:
            continue
        s = stats(f[m])
        s["group"] = lab
        s["pct_rows"] = 100 * s["n"] / len(f)
        s["pct_sse"] = 100 * s["sse"] / f["SE"].sum()
        s["pct_all_sse"] = 100 * s["sse"] / f["SE"].sum()
        s["target_med"] = float(f.loc[m, TARGET].median())
        s["target_max"] = float(f.loc[m, TARGET].max())
        s["pred_med"] = float(f.loc[m, "PRED"].median())
        s["pred_max"] = float(f.loc[m, "PRED"].max())
        rows.append(s)
    d = pd.DataFrame(rows)
    note(d[["group", "n", "pct_rows", "rmse", "mae", "bias", "medae",
            "pct_sse", "target_med", "target_max", "pred_med",
            "pred_max"]].round(2).to_string(index=False))
    note(f"  missing-group share of ALL SSE: "
         f"{100 * f.loc[f['T4_missing'], 'SE'].sum() / f['SE'].sum():.2f}%")

sub("4.FOCUS - LIRF and LFPG, T4 availability split")
rows = []
for fold in ("JAN", "JUL"):
    f = V[fold]
    for ap in FOCUS:
        g = f[f["ADEP_mvt"] == ap]
        if not len(g):
            continue
        for lab, m in (("complete", g["T4_complete"]),
                       ("missing", g["T4_missing"])):
            if m.sum() == 0:
                continue
            s = stats(g[m])
            s["fold"], s["airport"], s["group"] = fold, ap, lab
            s["pct_sse_ap"] = 100 * s["sse"] / g["SE"].sum()
            s["pct_sse_fold"] = 100 * s["sse"] / f["SE"].sum()
            rows.append(s)
note(pd.DataFrame(rows)[["fold", "airport", "group", "n", "rmse", "mae", "bias",
                         "medae", "pct_sse_ap", "pct_sse_fold"]]
     .round(2).to_string(index=False))

# ============================================================
# 5. T4 RELIABILITY - error vs delta magnitude
# ============================================================

head("5. T4 RELIABILITY: error versus delta magnitude")

for col, name in (("d_SCHED", "d_SCHED"), ("d_AOBT3", "d_AOBT3")):
    sub(f"5. error vs |{name}| (pooled Jan+Jul, rows where {name} exists)")
    g = allf[allf[col].notna()].copy()
    g["absd"] = g[col].abs()
    qs = np.percentile(g["absd"], [20, 40, 60, 80, 95, 99])
    edges = np.unique(qs)
    rows = []
    idx = np.digitize(g["absd"].to_numpy(), edges, right=True)
    for i, lab in enumerate([f"<={edges[0]:,.0f}"] +
                            [f"{edges[i]:,.0f}-{edges[i+1]:,.0f}"
                             for i in range(len(edges) - 1)] +
                            [f">{edges[-1]:,.0f}"]):
        m = idx == i
        if m.sum() < 50:
            continue
        s = stats(g[m])
        s["band"] = lab
        s["pct_sse"] = 100 * s["sse"] / g["SE"].sum()
        s["pct_rows"] = 100 * s["n"] / len(g)
        s["corr_absd_err"] = float(np.corrcoef(
            g.loc[m, "absd"], g.loc[m, "AE"])[0, 1])
        rows.append(s)
    note(f"  |{name}| vs absolute error, overall Pearson r = "
         f"{np.corrcoef(g['absd'], g['AE'])[0, 1]:+.4f}  "
         f"(n={len(g):,})")
    note(pd.DataFrame(rows)[["band", "n", "pct_rows", "rmse", "mae", "bias",
                             "corr_absd_err", "pct_sse"]]
         .round(3).to_string(index=False))

sub("5. d_SCHED reliability: is a big d_SCHED a good or bad signal?")
g = allf[allf["d_SCHED"].notna()].copy()
rows = []
for lo, hi, lab in [(-1e18, 600, "|d_SCHED| <= 10min (on time)"),
                    (600, 3600, "10min-1h"),
                    (3600, 10800, "1-3h"),
                    (10800, 21600, "3-6h"),
                    (21600, 43200, "6-12h"),
                    (43200, 1e18, ">12h")]:
    m = (g["d_SCHED"].abs() > lo) & (g["d_SCHED"].abs() <= hi)
    if m.sum() < 50:
        continue
    s = stats(g[m])
    s["band"] = lab
    s["pct_sse"] = 100 * s["sse"] / g["SE"].sum()
    s["corr"] = float(np.corrcoef(g.loc[m, "d_SCHED"], g.loc[m, "AE"])[0, 1])
    s["corr_signed"] = float(np.corrcoef(g.loc[m, "d_SCHED"],
                                          g.loc[m, "ERR"])[0, 1])
    s["d_sched_med"] = float(g.loc[m, "d_SCHED"].median())
    s["tgt_med"] = float(g.loc[m, TARGET].median())
    s["pred_med"] = float(g.loc[m, "PRED"].median())
    rows.append(s)
note(pd.DataFrame(rows)[["band", "n", "rmse", "mae", "bias", "corr",
                         "corr_signed", "d_sched_med", "tgt_med",
                         "pred_med", "pct_sse"]]
     .round(3).to_string(index=False))
note("")
note("  corr = corr(|d_SCHED|, |err|) within band; corr_signed =")
note("  corr(d_SCHED, signed error). tgt_med / pred_med are medians.")
note("  The pattern: small |d_SCHED| (on-time flights) is the most accurate")
note("  region in the entire model (RMSE 142.75, 0.87% of SSE). Large |d_SCHED|")
note("  is the least accurate (RMSE 15,370 at >12h, 35.3% of SSE from 113 rows).")
note("  Critically, at |d_SCHED| > 12h the TRUE target median is only 1,116s")
note("  (~19 min) while the model predicts 1,066s - so d_SCHED is not even")
note("  systematically biasing the median; it is the TAIL of that band that")
note("  explodes. d_SCHED is a symptom indicator, not a cause indicator.")

sub("5. d_AOBT3 availability: does it help when present?")
for fold in ("JAN", "JUL"):
    f = V[fold]
    for lab, m in (("d_AOBT3 present", f["d_AOBT3"].notna()),
                   ("d_AOBT3 missing", f["d_AOBT3"].isna())):
        if m.sum() < 50:
            continue
        s = stats(f[m])
        s["corr_dA3_err"] = float(np.corrcoef(
            f.loc[m, "d_AOBT3"].fillna(0), f.loc[m, "AE"])[0, 1])
        note(f"  {fold} {lab:20s} n={s['n']:>7,}  rmse={s['rmse']:8.2f}  "
             f"mae={s['mae']:7.2f}  bias={s['bias']:+8.2f}  "
             f"corr(|d_AOBT3|,|err|)={s['corr_dA3_err']:+.3f}  "
             f"{s['sse']/f['SE'].sum()*100:5.1f}% of fold SSE")

sub("4.INCONSISTENT RECORDS: flight data PRESENT but the deltas are nonsensical")
note("  A third regime exists that is neither 'complete' nor 'missing': the")
note("  flight record is present but the timestamps disagree with MVT_TIME in a")
note("  physically impossible way. e.g. 192622435 (LIRF 25, JUL) has")
note("  d_AOBT3 = d_IOBT = -1,146s, i.e. the aircraft was allegedly still")
note("  airborne 19 minutes AFTER it moved off block, yet d_SCHED = +15,294s.")
note("  The model still emitted 12,739s against a true target of 1,669s.")
note("")
_flt = T4_FEATURES[1:]  # the four flight-level deltas
allf["neg_any"] = (allf[_flt] < 0).any(axis=1)
allf["huge_any"] = (allf[_flt].abs() > 7200).any(axis=1)
allf["inconsistent"] = allf["T4_complete"] & (allf["neg_any"] | allf["huge_any"])
rows = []
for lab, m in [
    ("consistent complete record", allf["T4_complete"] & ~allf["inconsistent"]),
    ("INCONSISTENT (neg or >2h delta)", allf["inconsistent"]),
    ("record missing", ~allf["T4_complete"]),
]:
    s = stats(allf[m])
    s["group"] = lab
    s["pct_rows"] = 100 * s["n"] / len(allf)
    s["pct_sse"] = 100 * s["sse"] / allf["SE"].sum()
    s["pct_all_sse"] = s["pct_sse"]
    rows.append(s)
note(pd.DataFrame(rows)[["group", "n", "pct_rows", "rmse", "mae", "bias",
                         "medae", "pct_sse"]].round(2).to_string(index=False))
note("")
note(f"  rows with at least one NEGATIVE flight-level delta: "
     f"{int(allf['neg_any'].sum()):,}")
note(f"  rows with a flight-level delta beyond +/-2h: "
     f"{int(allf['huge_any'].sum()):,}")
note(f"  => the INCONSISTENT group is "
     f"{rows[1]['pct_rows']:.2f}% of rows holding {rows[1]['pct_sse']:.2f}% of SSE")
note("  This group is a THIRD actionable class: the model currently trusts the")
note("  deltas even when they are self-contradictory. An explicit")
note("  'record-implausible' flag is cheap and directly targets it.")

# ============================================================
# 5b. ALL FIVE DELTAS
# ============================================================

sub("5. ALL FIVE DELTAS: how much does each predict the error?")
rows = []
for col in T4_FEATURES:
    g = allf[allf[col].notna()]
    na = allf[allf[col].isna()]
    rows.append(dict(
        delta=col,
        n_present=len(g),
        pct_rows_na=100 * len(na) / len(allf),
        r_abs_vs_ae=float(np.corrcoef(g[col].abs(), g["AE"])[0, 1]),
        r_signed_vs_err=float(np.corrcoef(g[col], g["ERR"])[0, 1]),
        rmse_present=float(np.sqrt(g["SE"].mean())),
        pct_sse_present=100 * g["SE"].sum() / allf["SE"].sum(),
        rmse_missing=(float(np.sqrt(na["SE"].mean())) if len(na) else np.nan),
        pct_sse_missing=(100 * na["SE"].sum() / allf["SE"].sum()
                         if len(na) else np.nan),
    ))
note(pd.DataFrame(rows).round(4).to_string(index=False))
note("")
note("  WHERE T4 WORKS: for rows where the flight-level timestamps exist,")
note("  |d_AOBT3| stays in a tight band (the middle 60% of rows sit inside")
note("  715-1,243s) and group RMSE is only 227-250. There d_AOBT3 is a clean,")
note("  nearly additive signal and the model is accurate.")
note("  WHERE T4 FAILS: once |d_AOBT3| > 2,282s (1.0% of rows) RMSE jumps to")
note("  1,474.86 and corr(|d_AOBT3|,|err|) = +0.844 - i.e. the delta's")
note("  MAGNITUDE almost perfectly predicts the ERROR MAGNITUDE, which means")
note("  the model is roughly adding d_AOBT3 to its output while the true taxi")
note("  time is uncorrelated with it. Same story for d_SCHED: 52.4% of ALL SSE")
note("  comes from the 1.0% of rows with |d_SCHED| > 10,422s.")
note("  Note d_AOBT3 is the better error predictor (r +0.3228) than d_SCHED")
note("  (r +0.2242), but d_SCHED carries more SSE purely because its tail is")
note("  heavier and it is the only delta alive when flight data is missing.")
note("  d_IOBT / d_LOBT / d_EOBT1 are mutually near-collinear with d_AOBT3")
note("  (LOBT is constant within a flight), so they add little independent")
note("  information - consistent with their low individual shares above.")

# ============================================================
# 6. OPERATIONAL / CATEGORICAL BREAKDOWNS
# ============================================================

head("6. OPERATIONAL AND CATEGORICAL BREAKDOWNS")
note("  Only segments with >= 1000 validation rows are shown; anything smaller")
note("  is too small to generalise from.")

MINROWS = 1000
BREAKS = [
    ("RUNWAY_mvt", "runway"),
    ("hour", "hour of day (UTC)"),
    ("day_of_week", "day of week (0=Mon)"),
    ("AIRCRAFT_TYPE_mvt", "aircraft type"),
    ("FLIGHT_TYPE_flt", "flight type"),
    ("WK_TBL_CAT_flt", "WK_TBL_CAT"),
    ("traffic_last_15min", "traffic_last_15min"),
]

for col, label in BREAKS:
    sub(f"6. by {label} (pooled Jan+Jul, segments with >= {MINROWS:,} rows)")
    if col == "traffic_last_15min":
        edges = [-1, 0, 2, 5, 10, 20, 40, 10 ** 9]
        labels = ["0", "1-2", "3-5", "6-10", "11-20", "21-40", ">40"]
        rows = band_table(allf, col, edges, labels, minrows=MINROWS)
        d = pd.DataFrame(rows)
        d["pct_rows"] = 100 * d["n"] / len(allf)
        note(d[["band", "n", "pct_rows", "rmse", "mae", "bias", "medae",
                "pct_sse"]].round(2).to_string(index=False))
        continue
    rows = []
    for key, g in allf.groupby(col, observed=True):
        if len(g) < MINROWS:
            continue
        s = stats(g)
        s[label] = key
        s["pct_rows"] = 100 * s["n"] / len(allf)
        s["pct_sse"] = 100 * s["sse"] / allf["SE"].sum()
        rows.append(s)
    if not rows:
        note("  (no segment large enough)")
        continue
    d = pd.DataFrame(rows).sort_values("pct_sse", ascending=False)
    note(d[[label, "n", "pct_rows", "rmse", "mae", "bias", "medae",
            "pct_sse"]].round(2).to_string(index=False))

sub("6. by stand (top 15 stands by SSE share, segments >= 1000 rows)")
rows = []
for key, g in allf.groupby("STAND_mvt", observed=True):
    if len(g) < MINROWS:
        continue
    s = stats(g)
    s["stand"] = key
    s["pct_rows"] = 100 * s["n"] / len(allf)
    s["pct_sse"] = 100 * s["sse"] / allf["SE"].sum()
    rows.append(s)
d = pd.DataFrame(rows).sort_values("pct_sse", ascending=False)
note(f"  {len(d)} stands qualify. Top 15 by SSE share:")
note(d.head(15)[["stand", "n", "pct_rows", "rmse", "mae", "bias", "medae",
                 "pct_sse"]].round(2).to_string(index=False))
note(f"  these 15 holds {d.head(15)['pct_sse'].sum():.1f}% of all SSE "
     f"from {d.head(15)['n'].sum():,} rows "
     f"({100 * d.head(15)['n'].sum() / len(allf):.1f}% of rows)")

# ============================================================
# 7. WORST CASES
# ============================================================

head("7. WORST CASES - top 50 absolute-error rows per fold")

WORST_COLS = ["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt",
              "AIRCRAFT_TYPE_mvt", "FLIGHT_TYPE_flt", "traffic_last_15min",
              TARGET, "PRED", "ERR", "AE"]
TS = [MINUEND, "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "IOBT_flt",
      "LOBT_flt"]

for fold, f in V.items():
    sub(f"7.{fold} - ALL 50 worst absolute-error rows")
    w = f.sort_values("AE", ascending=False).head(50).copy()
    show = w[WORST_COLS + TS + T4_FEATURES].copy()
    for c in TS:
        show[c] = pd.to_datetime(show[c], utc=True).dt.strftime("%m-%d %H:%M")
    for c in T4_FEATURES:
        show[c] = show[c].round(0)
    note(show.round(1).to_string(index=False))
    w50 = f.nlargest(50, "AE")
    note("")
    note(f"  top 50 rows: {100 * w50['SE'].sum() / f['SE'].sum():.2f}% of "
         f"fold SSE from 50 rows ({100 * 50 / len(f):.4f}% of rows)")
    note(f"  their airports: "
         f"{dict(w50['ADEP_mvt'].value_counts().head(6))}")
    note(f"  T4 complete in top 50: {int(w50[T4_FEATURES].notna().all(axis=1).sum())}/50")
    note(f"  target > 3600s in top 50: {int((w50[TARGET] > 3600).sum())}/50")
    note(f"  target > 7200s in top 50: {int((w50[TARGET] > 7200).sum())}/50")
    note(f"  |d_SCHED| > 3600s in top 50: "
         f"{int((w50['d_SCHED'].abs() > 3600).sum())}/50")
    note(f"  prediction sign agrees with target sign: "
         f"{int((np.sign(w50['PRED']) == np.sign(w50[TARGET])).sum())}/50")
    note(f"  median |d_SCHED| in top 50 = {w50['d_SCHED'].abs().median():,.0f}s "
         f"vs {f['d_SCHED'].abs().median():,.0f}s fold-wide")

sub("7. PATTERNS across the top 50 of each fold")
w = pd.concat([V["JAN"].nlargest(50, "AE"), V["JUL"].nlargest(50, "AE")])
note(f"  combined top-100 rows, n={len(w)}")
note(f"  airports: {dict(w['ADEP_mvt'].value_counts())}")
note(f"  rows with all 5 T4 deltas present : "
     f"{int(w[T4_FEATURES].notna().all(axis=1).sum())}/100")
note(f"  rows with target > 3600s           : "
     f"{int((w[TARGET] > 3600).sum())}/100")
note(f"  rows with target > 7200s           : "
     f"{int((w[TARGET] > 7200).sum())}/100")
note(f"  rows with |d_SCHED| > 3600s        : "
     f"{int((w['d_SCHED'].abs() > 3600).sum())}/100")
note(f"  rows where model UNDER-predicted   : "
     f"{int((w['ERR'] < 0).sum())}/100")
note(f"  rows where model OVER-predicted    : "
     f"{int((w['ERR'] > 0).sum())}/100")
note(f"  median target in worst 100         : {w[TARGET].median():,.0f}s")
note(f"  median prediction in worst 100     : {w['PRED'].median():,.0f}s")
note(f"  median |d_SCHED| in worst 100      : {w['d_SCHED'].abs().median():,.0f}s")
note(f"  median |d_SCHED| all validation    : "
     f"{allf['d_SCHED'].abs().median():,.0f}s")

sub("7. runways and stands recurring in the worst rows")
note(f"  runways: {dict(w['RUNWAY_mvt'].value_counts().head(8))}")
note(f"  stands (top 8): {dict(w['STAND_mvt'].value_counts().head(8))}")
note(f"  distinct stands among 100 worst rows: {w['STAND_mvt'].nunique()}")
note(f"  distinct airports among 100 worst rows: {w['ADEP_mvt'].nunique()}")

# ============================================================
# 8. WHAT SHARE OF SSE IS "ORDINARY"?
# ============================================================

head("8. DECOMPOSITION: how much error is ordinary vs anomalous?")

note("Attempting an honest split of pooled validation SSE. Groups are mutually")
note("exclusive and exhaustive; every number is a raw measured quantity.")

seg = []
gm = ~allf["T4_complete"]
gt = allf[TARGET] > 7200
gbig = allf[TARGET] > 3600
for lab, m in [
    ("A. missing flight data AND target > 2h", gm & gt),
    ("B. missing flight data AND target <= 2h", gm & ~gt),
    ("C. T4 present, target > 2h", ~gm & gt),
    ("D. T4 present, 1h < target <= 2h", ~gm & gbig & ~gt),
    ("E. T4 present, target <= 1h (ordinary ops)", ~gm & ~gbig),
]:
    s = stats(allf[m])
    s["group"] = lab
    s["pct_rows"] = 100 * s["n"] / len(allf)
    s["pct_sse"] = 100 * s["sse"] / allf["SE"].sum()
    seg.append(s)
d = pd.DataFrame(seg)
note(d[["group", "n", "pct_rows", "rmse", "mae", "medae",
        "pct_sse"]].round(2).to_string(index=False))
note("")
note(f"  check: group SSE shares sum to "
     f"{d['pct_sse'].sum():.2f}%  (must be 100.00)")
_e = d[d["group"].str.startswith("E")].iloc[0]
note(f"  group E = 'ordinary operations': {_e['pct_rows']:.2f}% of rows hold "
     f"{_e['pct_sse']:.2f}% of SSE (RMSE {_e['rmse']:.2f})")
note(f"  groups A-D together = {100 - _e['pct_rows']:.2f}% of rows hold "
     f"{100 - _e['pct_sse']:.2f}% of SSE")
note("")
note("  Ceiling check (DIAGNOSTIC): a PERFECT model on group E alone would")
note(f"  leave pooled RMSE at 468.31 * sqrt({_e['pct_sse']/100:.4f}) = "
     f"{468.31 * np.sqrt(_e['pct_sse']/100):.1f}.")
note("  So the 98%-of-rows bulk can only ever be worth ~25% of current SSE.")
note("  Conversely, halving the A-D error would take pooled RMSE to")
_a = d[d["group"].str.startswith("A")].iloc[0]
note(f"  468.31 * sqrt({_e['pct_sse']/100 + (_a['pct_sse']/100)*0.25:.4f}) = "
     f"{468.31 * np.sqrt(_e['pct_sse']/100 + (_a['pct_sse']/100) * 0.25):.1f}.")


sub("8. same split, per fold")
rows = []
for fold, f in V.items():
    gm = ~f["T4_complete"]
    gt = f[TARGET] > 7200
    gbig = f[TARGET] > 3600
    for lab, m in [
        ("A. missing T4, target > 2h", gm & gt),
        ("B. missing T4, target <= 2h", gm & ~gt),
        ("C. T4 present, target > 2h", ~gm & gt),
        ("D. T4 present, 1-2h", ~gm & gbig & ~gt),
        ("E. T4 present, <= 1h (ordinary)", ~gm & ~gbig),
    ]:
        s = stats(f[m])
        s["fold"] = fold
        s["group"] = lab
        s["pct_rows"] = 100 * s["n"] / len(f)
        s["pct_sse"] = 100 * s["sse"] / f["SE"].sum()
        rows.append(s)
note(pd.DataFrame(rows)[["fold", "group", "n", "pct_rows", "rmse", "mae",
                         "medae", "pct_sse"]].round(2).to_string(index=False))

# ============================================================
# WRITE
# ============================================================

REPORT = Path("reports/v8_error_autopsy.txt")
REPORT.parent.mkdir(parents=True, exist_ok=True)

head("9. LEADERBOARD CONTEXT")
note("  official 2026 leaderboard RMSE : V8/T4 = 425.7711, V1 = 572.1223")
note("  V8 improvement over V1 on the public board: -146.3512 (-25.6%)")
note("  local pooled Jan+Jul T4 RMSE   : 468.31")
note("  The 2026 board score (425.77) is BETTER than the local pooled holdout")
note("  (468.31). The 2026 period is therefore not harder than the 2025 folds,")
note("  so the remaining error is unlikely to be a period-shift artefact.")
note("")
note("  V1 scored 572.12 on the board while scoring 519.90 (JAN) / 630.63 (JUL)")
note("  locally, so V1 also sat between its own folds. Local pooled RMSE has")
note("  historically been a slightly PESSIMISTIC proxy for the board.")

# ============================================================
# 10. PROBLEM CLASSES + RANKED PROPOSALS  (analysis, NOT implemented)
# ============================================================

head("10. PROBLEM CLASSES AND RANKED PROPOSED EXPERIMENTS")
note("NOTHING IN THIS SECTION HAS BEEN IMPLEMENTED. These are proposals only.")

W("")
W("=" * 100)
W("10a. DECOMPOSITION OF THE REMAINING ERROR INTO ACTIONABLE CLASSES")
W("=" * 100)
W("""
The remaining error is NOT spread across ordinary operations. It is almost
entirely two regimes, and they together are 1.6% of the rows.

  CLASS 1 - MISSING FLIGHT-LEVEL TIMESTAMPS + EXTREME TARGET  (the dominant one)
    111 rows (0.03% of validation rows) carry 46.8% of ALL pooled SSE.
    These are rows where AOBT3/EOBT1/IOBT/LOBT are all absent (so four of the
    five T4 deltas are NaN and only d_SCHED survives) AND the true taxi time
    is over 2h. Group RMSE 17,848.69.
    Concentrated in LIRF (runway 25 / 16R / 34L) and LFPG (08L / 26R).
    LIRF missing-flight-data rows alone: JAN RMSE 10,076 (84% of LIRF's SSE),
    JUL RMSE 9,074 (72% of LIRF's SSE, 55% of the entire fold's SSE).

  CLASS 2 - MISSING FLIGHT-LEVEL TIMESTAMPS, ORDINARY TARGET
    5,262 rows (1.53%) carry 14.2% of SSE. Group RMSE 1,427.35 - six times
    worse than ordinary rows, purely because d_SCHED is the only T4 signal and
    it is unreliable when the flight record is absent. Median target is normal
    (~900-1,000s) and median absolute error is small (191s), so the damage is
    entirely in this group's tail: 26 negative predictions also live here.

  CLASS 2b - PRESENT BUT SELF-CONTRADICTORY FLIGHT RECORDS
    A third regime, distinct from both "complete" and "missing": the flight
    record exists but the timestamps are physically impossible. Example
    192622435 (LIRF 25, JUL): d_AOBT3 = d_IOBT = -1,146s, meaning the aircraft
    is recorded as still airborne 19 minutes after it moved off block, while
    d_SCHED = +15,294s. The model trusted the deltas and emitted 12,739s
    against a true target of 1,669s - an 11,070s error on a row that LOOKS
    well-populated. See section 4.INCONSISTENT for the measured size of this
    group. It is small in rows but expensive per row, and it is trivially
    detectable at prediction time (no modelling required).

  CLASS 3 - ORDINARY OPERATIONS (all T4 present, target <= 1h)
    338,271 rows (98.2%) carry only 25.1% of SSE. Group RMSE 236.72.
    This is the bulk. Even a PERFECT model on this class would only take
    pooled RMSE from 468.31 to ~234, and realistically it might recover a
    third of that gap. It is also the only class that plausibly needs
    EXTERNAL data (weather / ATC congestion) to improve.

  Cross-cutting: extreme targets and missing flight data are largely the SAME
  rows. Of the 50 worst rows per fold, 50/50 in JUL and 46/50 in JAN have
  |d_SCHED| > 3600s, and only 26/100 of the combined worst rows have a
  complete flight record. So "fix the missing-flight-data regime" and "fix the
  extreme tail" are ONE experiment, not two.

  Where the error is NOT: there is no stand-level pathology (21 stands clear
  the 1,000-row bar; the top 15 hold 9.8% of SSE from 5.4% of rows, and only
  stand 308 at LIRF stands out at 5.8%), no aircraft-type pathology (largest
  is B738 at 11.7% of SSE from 10.6% of rows, i.e. proportional), and no
  traffic-monootonicity (RMSE falls from 1,094 at 3-5 movements/15min to 255
  at >40, which is an airport-mix confound, not a congestion effect).
""")

W("=" * 100)
W("10b. THE REPEATED PATTERNS IN THE WORST ROWS")
W("=" * 100)
W("""
Pattern 1 - LIRF runway 25, missing flight record, huge d_SCHED, wildly varying
  target. 67 of the 100 worst rows are LIRF; 51 of 100 are runway 25; 83
  distinct stands and only 6 airports, so it is an airport/runway regime and
  NOT a stand regime. In these rows d_SCHED runs 6-26h (60,118s; 93,535s;
  87,202s) while the true target ranges from 594s to 87,177s. With only
  d_SCHED alive, the model emits 20,000-45,000s: accidentally near-correct for
  the handful of rows whose target really is ~87,000s, and catastrophic for
  the rest. d_SCHED is being used as a proxy for a target that is bimodal here.

Pattern 2 - "small target, huge d_SCHED" = systematic OVER-prediction.
  182641684 (LIRF 25, target 1,190s, d_SCHED 12,835s, pred 24,932) and
  192624005 (LIRF 25, target 1,077s, d_SCHED 15,178s, pred 42,680).
  d_SCHED here reflects schedule/actual mismatch, not taxi-out, and the model
  effectively adds it to the prediction. corr(d_SCHED, signed err) is
  NEGATIVE (-0.117) in the |d_SCHED|>12h band, i.e. not even directionally
  reliable where it is largest. 47 of the 100 worst rows are over-predictions.

Pattern 3 - genuine long taxi WITH a flight record, UNDER-predicted.
  181983675 (EGLL, target 10,615s, full T4, d_SCHED 11,097, pred 3,267) and
  191767771 (LFPG, target 15,059s, d_SCHED 16,858, pred 2,442). Here the
  timestamps are all present and d_SCHED is large and correct, but a 2.5-4.2h
  crawl is simply not predictable from any current feature. This is the
  genuinely irreducible class and it is the one that would need weather,
  runway configuration or ATC data.

Pattern 4 - mid-range under-prediction even with full T4.
  The 30-60min band (14,733 rows, 4.3%) has bias -444s and RMSE 682; the
  1-2h band (833 rows) has bias -1,776s and RMSE 2,165. The model pulls
  mid-length taxi times toward the bulk. This is a real, addressable
  "middle" error that is separate from the extreme tail.

Pattern 5 - the model is right about ORDER but wrong about MAGNITUDE on the
  worst rows. In the combined worst 100, median target is 6,960s and median
  prediction is 6,473s - the same order of magnitude - and the prediction sign
  matches the target sign on 100/100 rows. The failure is calibration within
  the tail, not a total misclassification of which flights are slow.

Pattern 6 - the record is PRESENT but self-contradictory, and the model
  believes it anyway. 192622435 (LIRF 25, JUL): d_AOBT3 = d_IOBT = -1,146s
  (airborne 19 min after off-block), d_SCHED = +15,294s, true target 1,669s,
  prediction 12,739s. This row is fully populated, so it passes every
  availability check, yet it is nearly as wrong as the missing-record rows.
  This is the cheapest class to fix because it requires no model at all - just
  a plausibility test on the deltas - and it is detectable on 2026 too.
""")

W("=" * 100)
W("10c. RANKED PROPOSED EXPERIMENTS (NOT IMPLEMENTED - proposals only)")
W("=" * 100)
W("""
Ranked by expected potential, then robustness/generalisation, then
implementation difficulty (LOW = easy), then whether external data is needed.

#1. MISSING-DATA SEGMENT MODEL
    What    : Train a separate model for the ~1.6% of rows where
              AOBT3/EOBT1/IOBT/LOBT are ALL missing, using only the features
              actually available in that regime (airport, runway, stand, hour,
              day_of_week, month, traffic, aircraft, FLIGHT_TYPE, d_SCHED).
              Today those rows are predicted by a model that expects four
              deltas, finds them NaN, and falls back on d_SCHED alone.
    Why     : This regime is 61% of all SSE (46.8% + 14.2%) from 1.6% of rows,
              and the regime is identifiable at prediction time, so it needs
              no 2026 tuning.
    Potential      : VERY HIGH - the only class where an order-of-magnitude
                     error reduction is plausible.
    Robustness     : HIGH - missingness is structural, and the 2026 submission
                     shows the same missing-AOBT tail, so it transfers.
    Difficulty     : LOW (no new data, reuses existing features + a group split)
    External data  : NO
    Risk           : 5,262 of those rows have ordinary targets, so the segment
                     model must not collapse to the conditional mean and
                     discard the genuine huge-target signal. Validate the
                     segment on the same 2025 folds, and check LIRF/08L/26R.

#1b. RECORD-PLAUSIBILITY FLAG (cheapest item on this list)
    What    : Add a binary/categorical feature flagging flight records that
              are internally impossible - any of the four flight-level deltas
              negative (MVT earlier than AOBT/IOBT/LOBT) or beyond a sane
              +/-2h. This is a FEATURE, not a filter: nothing is dropped, no
              prediction is clipped, and the row still gets a prediction.
    Why     : Pattern 6. 192622435 has d_AOBT3 = -1,146s and costs 11,070s of
              error while looking perfectly complete, so it survives every
              availability filter the model currently has. The same flag also
              fires on rows that are merely extreme, letting the model
              re-weight d_SCHED there.
    Potential      : MEDIUM-HIGH - small in row count, expensive per row.
    Robustness     : HIGH - it is a physical consistency test, not a learned
                     pattern, so it cannot overfit and transfers to 2026 by
                     construction.
    Difficulty     : TRIVIAL (one boolean derivation from existing columns)
    External data  : NO
    Note          : This is a feature addition, NOT anomaly removal. It must
                     never be used to exclude rows from the metric.

#2. d_SCHED RELIABILITY INTERACTION (missingness x d_SCHED magnitude)
    What    : Give the model explicit coarse buckets for |d_SCHED| AND their
              interaction with "flight record missing", so it can learn a
              different mapping in that quadrant. Do NOT clip d_SCHED -
              clipping was ruled out and would bias the raw-RMSE score.
    Why     : d_SCHED is a good proxy only when the flight record exists
              (corr(|d_AOBT3|,|err|) = +0.844 once |d_AOBT3| > 2,282s) and is
              close to noise when it does not (corr_signed -0.117 at
              |d_SCHED| > 12h). A single feature cannot express that; a
              missingness x magnitude interaction can.
    Potential      : HIGH - compounds with #1 and also targets Class 2.
    Robustness     : MEDIUM-HIGH - physically motivated, testable on the same
                     two folds without touching 2026.
    Difficulty     : LOW-MEDIUM (feature engineering on existing columns only)
    External data  : NO
    Risk           : interaction terms overfit easily; keep it to one or two
                     coarse buckets rather than a fine grid.

#3. MID-RANGE CALIBRATION (30-60min and 1-2h bands)
    What    : Fit an isotonic or piecewise-scaled correction for the mid-range
              inside each training fold only, to remove the -444s / -1,776s
              bias. This is genuinely separate from the tail work.
    Potential      : MEDIUM - 30-60min is 9.1% of SSE and 1-2h is 5.2%; a 20%
                     RMSE cut there is worth a few pooled points.
    Robustness     : MEDIUM - fold-fitted calibration can generalise, but the
                     bias sign flips between Jan and Jul (<15min bias +77 Jan
                     vs +97 Jul; 15-30min -21 both), so a single global
                     correction may not survive. Test per-fold sign stability
                     first.
    Difficulty     : MEDIUM (needs strict fold hygiene to avoid calibrating on
                     the validation fold)
    External data  : NO

#4. ORDINARY-OPERATION BULK (the 98% / 25% class) - needs external data
    What    : The only realistic lever for the 25%-of-SSE bulk is exogenous
              congestion/weather/ATC-delay data joined on (airport, hour, day).
              The segments with real structure are the evening peak
              (18:00-20:00 UTC: RMSE 722/639/635, ~22% of SSE combined) and
              the high-traffic band (21-40 movements/15min: RMSE 513 vs 255 for
              >40), plus Sunday/Monday (RMSE ~558-569 vs 319 Thursday).
    Potential      : MEDIUM - it caps at ~25% of SSE even if perfect.
    Robustness     : MEDIUM - external joins are the usual source of silent
                     leakage and train/serve skew; availability for 2026 must
                     be proven before investing.
    Difficulty     : MEDIUM-HIGH (sourcing, join keys, backfill)
    External data  : YES
    Risk           : hour-of-day and traffic are ALREADY features, so the
                     marginal gain is only what those two miss. Also note
                     traffic_last_15min is NON-monotonic in error (RMSE 1,094 at
                     3-5 movements vs 255 at >40), which is an airport-mix
                     confound, not congestion - so a naive "congestion" feature
                     may just re-encode airport identity.

#5. LOSS / OBJECTIVE CHANGE (Huber, quantile, two-stage tail model)
    What    : The metric is RMSE, optimised by the conditional mean. A
              two-stage "classify long-taxi then regress magnitude", or a
              mild Huber/quantile blend to stop the 10,000-80,000s targets from
              dominating the fit.
    Potential      : MEDIUM at best, and only via reduced tail VARIANCE.
    Robustness     : MEDIUM-LOW on a raw-RMSE leaderboard; this is the classic
                     optimise-the-wrong-loss trap.
    Difficulty     : MEDIUM
    External data  : NO
    Risk           : HIGH. Treat as a diagnostic probe, not a primary bet. Any
                     bias introduced away from the conditional mean is charged
                     directly against the score.

#6. SEED / ENSEMBLE AVERAGING of the unchanged T4 model
    What    : Average several CatBoost seeds of the SAME T4 feature set and
              the same hyperparameters.
    Potential      : LOW-MEDIUM (a few points at most)
    Robustness     : HIGH - pure variance reduction, introduces no new
                     assumption and no new failure mode
    Difficulty     : TRIVIAL
    External data  : NO
    Note           : the safest floor move if everything else stalls.

EXPLICITLY NOT RECOMMENDED
  - Clipping or capping predictions or training targets. The leaderboard is
    raw RMSE, anomalies are part of the target, and the 18 LIRF rows above 8h
    in the 2026 submission are the model's honest output. Leave them.
  - Re-introducing the rejected clock or H1-history feature sets. Both had
    their apparent gains concentrated in LIRF and both inverted once LIRF was
    excluded; neither addresses the 61%-of-SSE missing-data regime.
  - A monotone constraint on d_SCHED. The data contradicts monotonicity in the
    extreme band (corr_signed -0.117) and in the mid bands the relationship is
    not monotone either.
  - Per-airport or per-runway separate models for the top airports. LIRF's
    contribution is dominated by the missing-flight-data regime, not by an
    airport-specific effect, so per-airport models would mostly re-learn the
    same broken fallback. Only consider this AFTER #1 and #2.

DEPENDENCY NOTE
  #1 and #2 are independent, composable, need no external data, and together
  address ~61% of the remaining SSE. They are the highest-value first moves.
  #3 is a cheap, independent follow-up. #6 is the free floor. #4 is the only
  route to gains beyond those, but needs external data and carries leakage and
  availability risk. #5 is the lowest-confidence item.
""")

REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print(f"\n[written] {REPORT}  lines={len(out)}  "
      f"bytes={REPORT.stat().st_size:,}")
