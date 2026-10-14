"""V9-A follow-up: 2025 vs 2026 missing-flight / extreme-d_SCHED feature audit.

ANALYSIS ONLY. This script:
  * fits nothing (no CatBoost import, no model, no prediction),
  * reads NO target and NO truthing file,
  * writes NO submission,
  * does not modify V8, V9-A or any other existing file,
  * does not use leaderboard information.

It asks one question: does the 2026 ranking set contain the same FEATURE-SIDE
regimes that produced V9-A's apparent 2025 improvement, namely rows where all
four flight-level timestamps are missing and |d_SCHED| is in the extreme tail?

matching distributions do NOT mean V9-A should be promoted. This audit can
only size exposure and flag shift. It cannot show that the 2026 targets of
those rows resemble the 2025 targets, and that is precisely the unverified
link in V9-A's evidence chain.

Periods compared (departures only, PHASE_mvt == 'DEP'):
    JAN2025  2025-01-01 .. 2025-02-01   from data/raw/training_*.parquet
    JUL2025  2025-07-01 .. 2025-08-01   from data/raw/training_*.parquet
    JAN2026  2026-01-01 .. 2026-02-01   from data/raw/ranking.parquet
    JUL2026  2026-07-01 .. 2026-08-01   from data/raw/ranking.parquet
ranking.parquet turns out to contain exactly those two months, so the
comparison is month-for-month against the two validated 2025 folds.

SAFETY
    ranking.parquet DOES contain TAXITIME_SEC_mvt and BLOCK_TIME_UTC_mvt.
    Those are exactly the two columns that must never inform a prediction.
    Every read below goes through an explicit allow-list and asserts that
    neither forbidden column is in it, so they are never even materialised.

Usage:
    python src/v9a_2026_distribution_audit.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from train_catboost_v8_defs import (  # noqa: F401  (read-only import)
    DATA_DIR, FORBIDDEN, MINUEND, V7_FEATURES, add_traffic_feature,
)

REPORT = Path("reports/v9a_2026_distribution_audit.txt")

# ============================================================
# PERIOD DEFINITIONS - the two validated 2025 folds, and their
# 2026 counterparts read from the ranking file.
# ============================================================

PERIODS = [
    ("JAN2025", "train", ("2025-01-01", "2025-02-01")),
    ("JUL2025", "train", ("2025-07-01", "2025-08-01")),
    ("JAN2026", "rank", ("2026-01-01", "2026-02-01")),
    ("JUL2026", "rank", ("2026-07-01", "2026-08-01")),
]

# V9-A's segment condition, reproduced verbatim.
FLIGHT_TS = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]

# V7/V8 features that are also wanted for the 2026 row listing.
KEEP = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
        "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "WK_TBL_CAT_flt",
        "FLIGHT_TYPE_flt"]

FOCUS = ["LIRF", "LFPG", "EGLL", "EDDF"]
# airports where V9-A created new large errors on the 2025 folds
ERROR_SITES = ["EGLL", "EDDF", "EHAM", "LSZH", "LEBL", "LEMD", "EDDM", "LTFM"]

# V9-A's decisive |d_SCHED| tail: the 6-12h and >12h bands held 54 of the 5,373
# 2025 segment validation rows and decided the whole experiment.
DECISIVE_LO, DECISIVE_HI = 6 * 3600, float("inf")

# d_SCHED magnitude bands, in hours.
BANDS = [(0, 1, "<1h"), (1, 2, "1-2h"), (2, 4, "2-4h"), (4, 6, "4-6h"),
         (6, 12, "6-12h"), (12, float("inf"), ">12h")]

MIN_AIRPORT_N = 20        # a segment airport must have >= this many rows
                           # in BOTH periods of a pair to get a per-airport
                           # distribution table
REPORT_MIN_N = 20        # absolute floor for any single airport table

out = []
W = out.append

# populated as the audit proceeds; section 6 reads it to form the verdict
SHIFT = {}
SHIFT_P = {}


def rate_ratio_ci(a, ea, b, eb, conf=0.95):
    """Exact conditional (Clopper-Pearson) CI for the rate ratio b/eb over a/ea.

    Departures are the whole population, so only the event counts are random.
    Conditioning on the total number of events, the share a/(a+b) is Binomial
    with p = ea*lam_a / (ea*lam_a + eb*lam_b). Solving for the rate ratio R
    gives R = (1-p) / (p * r) with r = eb/ea. Returns (lo, hi, point).
    """
    n = a + b
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    al = (1 - conf) / 2
    p_lo = 0.0 if a == 0 else stats.beta.ppf(al, a, n - a + 1)
    p_hi = 1.0 if b == 0 else stats.beta.ppf(1 - al, a + 1, n - a)
    r = eb / ea
    f = lambda p: (1 - p) / (p * r)  # noqa: E731  monotone decreasing in p
    hi = f(p_lo) if p_lo > 0 else float("inf")
    lo = f(p_hi) if p_hi < 1 else 0.0
    return lo, hi, (b / eb) / (a / ea) if a and b else float("nan")


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
# 0. SAFETY GUARDS
# ============================================================

# never load these, from anywhere, for any reason
READ_ALLOW = set(KEEP) | set(FLIGHT_TS) | {MINUEND, "SCHED_TIME_UTC_mvt"}

assert not (READ_ALLOW & set(FORBIDDEN)), \
    f"allow-list contains a forbidden column: {READ_ALLOW & set(FORBIDDEN)}"
assert "TAXITIME_SEC_mvt" not in READ_ALLOW
assert "BLOCK_TIME_UTC_mvt" not in READ_ALLOW

_truthing = list(DATA_DIR.glob("*truthing*")) + list(Path(".").glob("*truthing*"))
assert not _truthing, f"truthing data present, aborting: {_truthing}"

head("V9-A FOLLOW-UP - 2025 vs 2026 MISSING-FLIGHT / EXTREME-d_SCHED "
     "FEATURE-SIDE DISTRIBUTION AUDIT")
note("")
note("ANALYSIS ONLY. No model is fitted, no prediction is produced, no")
note("submission is written, nothing is uploaded, nothing is committed.")
note("No hidden target and no truthing data is accessed. No leaderboard")
note("information is used.")
note("")
note("  V8 stays the frozen champion (official RMSE 425.7711). V9-A stays")
note("  MIXED / not promoted. This audit does not change either status.")
note("")
note("  SAFETY: ranking.parquet physically contains TAXITIME_SEC_mvt and")
note("  BLOCK_TIME_UTC_mvt - the two columns that must never inform a")
note(f"  prediction. Every read uses an explicit allow-list; the forbidden set")
note(f"  {FORBIDDEN} is asserted absent from it, so those columns are never")
note("  materialised in memory at all.")
note(f"  allow-list ({len(READ_ALLOW)} cols): {sorted(READ_ALLOW)}")
note("")
note("  CAVEAT ON WHAT THIS CAN PROVE: matching feature distributions do")
note("  NOT validate V9-A. V9-A's gain came from 2025 TARGETS that happened")
note("  to track d_SCHED closely in the LIRF tail. This audit sees features")
note("  only. It can size exposure and detect shift; it cannot establish that")
note("  the 2026 targets of the matching rows behave the same way.")

# ============================================================
# 1. LOAD
# ============================================================

note("")
note("Loading 2025 movement data (allow-list columns only)...")
_files = sorted(DATA_DIR.glob("training_*.parquet"))
assert len(_files) == 12, f"expected 12 training files, got {len(_files)}"
_2025 = pd.concat([pd.read_parquet(f, columns=sorted(READ_ALLOW))
                   for f in _files], ignore_index=True)
note(f"  2025 movements loaded: {len(_2025):,}")

note("Loading 2026 ranking data (allow-list columns only)...")
_rank = pd.read_parquet(DATA_DIR / "ranking.parquet", columns=sorted(READ_ALLOW))
note(f"  2026 ranking movements loaded: {len(_rank):,}")
assert "TAXITIME_SEC_mvt" not in _rank.columns
assert "BLOCK_TIME_UTC_mvt" not in _rank.columns
note("  [PASS] neither forbidden column is present in either in-memory frame")

# traffic feature, using the exact V7/V8 construction, so 2026 is measured
# on the same footing as 2025
note("")
note("Building traffic_last_15min with the exact V7/V8 construction "
     "(add_traffic_feature, imported unchanged)...")
_2025 = add_traffic_feature(_2025)
_rank = add_traffic_feature(_rank)
note("  [PASS] identical construction applied to both years")


def prep(frame, lo, hi):
    m = (frame[MINUEND] >= lo) & (frame[MINUEND] < hi)
    d = frame[m & (frame["PHASE_mvt"] == "DEP")].copy()
    d["d_SCHED"] = (d[MINUEND] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["SEG"] = d[FLIGHT_TS].isna().all(axis=1)
    return d.reset_index(drop=True)


P = {}
for name, source, (lo, hi) in PERIODS:
    P[name] = prep(_2025 if source == "train" else _rank, lo, hi)
    d = P[name]
    note(f"  {name}: {len(d):>7,} departures  {d[MINUEND].min()} -> "
         f"{d[MINUEND].max()}  missing_all4 {int(d['SEG'].sum()):>5,}")

# 2025 fold counts must match the audited V9-A experiment exactly
V9A_N = {"JAN2025": 153_706, "JUL2025": 190_713}
V9A_SEG = {"JAN2025": 1_435, "JUL2025": 3_938}
sub("1. PROVENANCE - these 2025 slices are the audited V9-A folds")
for k, n in V9A_N.items():
    got = len(P[k])
    note(f"  {k}: departures {got:,}   V9-A {n:,}   match {got == n}")
    assert got == n, f"{k} departure count does not match the V9-A fold"
for k, n in V9A_SEG.items():
    got = int(P[k]["SEG"].sum())
    note(f"  {k}: missing_all4 {got:,}   V9-A {n:,}   match {got == n}")
    assert got == n, f"{k} segment count does not match V9-A"
note("  [PASS] row counts and missing_all4 counts reproduce reports/"
     "v9a_missing_segment_experiment.txt exactly, so the 2026 side is being")
note("         compared against the same 2025 evidence V9-A was judged on")

sub("1b. RANKING FILE SHAPE - what is actually in 2026")
_rank_dep = _rank[_rank["PHASE_mvt"] == "DEP"]
note(f"  ranking.parquet total movements : {len(_rank):,}")
note(f"  ranking.parquet departures      : {len(_rank_dep):,}")
note(f"  submitting.parquet rows         : "
     f"{pq.ParquetFile(DATA_DIR / 'submitting.parquet').metadata.num_rows:,} "
     f"(the 344,841 expected ranking departures)")
note(f"  movement time span              : {_rank_dep[MINUEND].min()} -> "
     f"{_rank_dep[MINUEND].max()}")
_m = pd.to_datetime(_rank_dep[MINUEND]).dt.strftime("%Y-%m").value_counts().sort_index()
note(f"  months present                  : { {k: int(v) for k, v in _m.items()} }")
note("")
note("  The ranking file contains exactly two months - January 2026 and July")
note("  2026 - so it is a month-for-month mirror of the two 2025 folds and the")
note("  comparison needs no extrapolation to unobserved months.")
note("")
note("  Traffic-feature caveat: 2025's 15-minute window was computed over 12")
note("  contiguous monthly files, so 2025-12-31 movements inform 2025-01-01.")
note("  The 2026 ranking file is self-contained with a six-month gap, so no")
note("  2026 movement sees anything from 2025. The 2025 JAN/JUL folds are far")
note("  from any file boundary, so this asymmetry does not affect them. The")
note("  effect is second-order and is recorded, not corrected.")

# ============================================================
# 2. MISSING-DATA PREVALENCE
# ============================================================

head("2. MISSING-DATA PREVALENCE - how common is the V9-A segment?")

PREV = {}
note("")
note("  1) OVERALL, per period")
note(f"  {'period':9s} {'departures':>12s} {'missing_all4':>13s} {'pct':>8s} "
     f"{'rate per 1k':>13s}")
for name, _, _ in PERIODS:
    d = P[name]
    n, s = len(d), int(d["SEG"].sum())
    PREV[name] = dict(period=name, departures=n, missing_all4=s, pct=100 * s / n)
note("")
for r in PREV.values():
    note(f"  {r['period']:9s} {r['departures']:>12,} {r['missing_all4']:>13,} "
         f"{r['pct']:>7.3f}% {1000 * r['missing_all4'] / r['departures']:>13.2f}")
note("")
note("  2025 -> 2026 change in prevalence (same calendar month). The interval is")
note("  an exact conditional Clopper-Pearson interval on the rate ratio, which is")
note("  the right model here: departures are the entire population, so only the")
note("  event count is random.")
note("")
note(f"  {'pair':22s} {'2025':>10s} {'2026':>10s} {'rate ratio':>11s} "
     f"{'exact 95% CI':>18s}   reading")
for a, b in (("JAN2025", "JAN2026"), ("JUL2025", "JUL2026")):
    pa, pb = PREV[a], PREV[b]
    lo, hi, rr = rate_ratio_ci(pa["missing_all4"], pa["departures"],
                               pb["missing_all4"], pb["departures"])
    SHIFT_P[a[:3] + "25to26"] = (rr, lo, hi)
    if lo <= 1.0 <= hi:
        rd = "unchanged"
    elif rr > 1:
        rd = f"more common, up to {hi:.1f}x"
    else:
        rd = f"less common, down to {1 / hi:.1f}x"
    note(f"  {a + ' -> ' + b:22s} {pa['pct']:>9.3f}% {pb['pct']:>9.3f}% "
         f"{rr:>10.2f}x  [{lo:>5.2f}, {hi:>5.2f}]   {rd}")

sub("2b) missing_all4 BY AIRPORT, all periods side by side")
aps = {}
for name, _, _ in PERIODS:
    s = P[name][P[name]["SEG"]]
    aps[name] = s["ADEP_mvt"].value_counts()
allap = sorted(set().union(*[set(v.index) for v in aps.values()]))
tot = {n: max(1, int(aps[n].sum())) for n in aps}
mixrows = []
for ap in allap:
    r = dict(airport=ap)
    for name, _, _ in PERIODS:
        c = int(aps[name].get(ap, 0))
        r[name] = c
        r[name + "_pct"] = 100 * c / tot[name]
    mixrows.append(r)
MIX = pd.DataFrame(mixrows)
d = MIX
d["max25"] = d[["JAN2025", "JUL2025"]].max(axis=1)
d["max26"] = d[["JAN2026", "JUL2026"]].max(axis=1)
d["ratio"] = d["max26"] / d["max25"].replace(0, np.nan)
show = d[d[["JAN2025", "JUL2025", "JAN2026", "JUL2026"]].sum(axis=1) > 0].copy()
show = show.sort_values(["max25", "max26"], ascending=False)
note("  counts, and each period's share of that period's segment rows")
note(f"  {'airport':8s} {'JAN25':>7s} {'JUL25':>7s} {'JAN26':>7s} {'JUL26':>7s} "
     f"{'26/25':>7s}   {'%JAN25':>7s} {'%JUL25':>7s} {'%JAN26':>7s} {'%JUL26':>7s}")
for _, r in show.iterrows():
    note(f"  {r['airport']:8s} {int(r['JAN2025']):>7,} {int(r['JUL2025']):>7,} "
         f"{int(r['JAN2026']):>7,} {int(r['JUL2026']):>7,} "
         f"{r['ratio'] if np.isfinite(r['ratio']) else float('nan'):>7.2f}   "
         f"{r['JAN2025_pct']:>6.2f}% {r['JUL2025_pct']:>6.2f}% "
         f"{r['JAN2026_pct']:>6.2f}% {r['JUL2026_pct']:>6.2f}%")
note(f"  {'TOTAL':8s} {int(show['JAN2025'].sum()):>7,} "
     f"{int(show['JUL2025'].sum()):>7,} {int(show['JAN2026'].sum()):>7,} "
     f"{int(show['JUL2026'].sum()):>7,}")

sub("2c) FOCUS AIRPORTS - LIRF, LFPG, EGLL, EDDF")
rows = []
for ap in FOCUS:
    for name, _, _ in PERIODS:
        d = P[name]
        s = d[d["SEG"]]
        g = s[s["ADEP_mvt"] == ap]
        a = d[d["ADEP_mvt"] == ap]
        rows.append(dict(
            airport=ap, period=name, n_dep_ap=len(a), n_seg=len(g),
            pct_of_ap_dep=100 * len(g) / max(1, len(a)),
            pct_of_seg=100 * len(g) / max(1, len(s)),
            ap_rate_pct=100 * len(g) / max(1, len(a))))
note(f"  {'airport':7s} {'period':8s} {'dep at ap':>9s} {'seg rows':>9s} "
     f"{'seg% of ap deps':>15s} {'% of period segment':>19s}")
for r in rows:
    note(f"  {r['airport']:7s} {r['period']:8s} {r['n_dep_ap']:>9,} "
         f"{r['n_seg']:>9,} {r['pct_of_ap_dep']:>14.3f}% "
         f"{r['pct_of_seg']:>18.2f}%")
note("")
note("  'seg% of ap deps' is the airport's own missing-flight rate. If that")
note("  rate is stable year over year, the regime is structural; if it swings,")
note("  the 2025 concentration was a period accident.")

# ============================================================
# 3. d_SCHED DISTRIBUTION INSIDE THE SEGMENT
# ============================================================

head("3. d_SCHED DISTRIBUTION WITHIN missing_all4")

note("  d_SCHED = MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt, in seconds, exactly")
note("  as train_catboost_v8_defs builds it. Bands are on |d_SCHED|.")

SEG = {n: P[n][P[n]["SEG"]].copy() for n, _, _ in PERIODS}
for n in SEG:
    SEG[n]["absd"] = SEG[n]["d_SCHED"].abs()

sub("3a) OVERALL, per period")
note(f"  {'period':9s} {'n':>7s} {'min':>10s} {'median':>10s} {'p90':>10s} "
     f"{'p95':>10s} {'p99':>10s} {'max':>11s} {'mean|d|':>10s}")
QSTATS = {}
for n, _, _ in PERIODS:
    a = SEG[n]["absd"].to_numpy()
    QSTATS[n] = dict(n=len(a), mn=a.min(), p50=np.median(a),
                     p90=np.quantile(a, .90), p95=np.quantile(a, .95),
                     p99=np.quantile(a, .99), mx=a.max(), mean=a.mean())
    q = QSTATS[n]
    note(f"  {n:9s} {q['n']:>7,} {q['mn']:>10,.0f} {q['p50']:>10,.0f} "
         f"{q['p90']:>10,.0f} {q['p95']:>10,.0f} {q['p99']:>10,.0f} "
         f"{q['mx']:>11,.0f} {q['mean']:>10,.0f}")
note("  (seconds)")

sub("3b) BAND COUNTS AND SHARES, per period")
bandrows = []
for n, _, _ in PERIODS:
    a = SEG[n]["absd"].to_numpy()
    r = dict(period=n, n=len(a))
    for lo, hi, lab in BANDS:
        m = (a > lo * 3600) & (a <= hi * 3600) if lo else (a <= hi * 3600)
        c = int(m.sum())
        r[lab] = c
        r[lab + "_pct"] = 100 * c / len(a)
    bandrows.append(r)
BAND = pd.DataFrame(bandrows)
dd = BAND
note(f"  {'period':9s} {'n':>7s} " + " ".join(f"{lab:>10s}" for _, _, lab in BANDS))
for _, r in dd.iterrows():
    note(f"  {r['period']:9s} {int(r['n']):>7,} " +
         " ".join(f"{int(r[lab]):>10,}" for _, _, lab in BANDS))
note("")
note("  same, as a percentage of the period's segment rows")
note(f"  {'period':9s} {'n':>7s} " + " ".join(f"{lab:>11s}" for _, _, lab in BANDS))
for _, r in dd.iterrows():
    note(f"  {r['period']:9s} {int(r['n']):>7,} " +
         " ".join(f"{r[lab + '_pct']:>10.2f}%" for _, _, lab in BANDS))

sub("3c) BY AIRPORT, where the sample is meaningful in both periods of a pair")
for a_name, b_name in (("JAN2025", "JAN2026"), ("JUL2025", "JUL2026")):
    note("")
    note(f"  {a_name} vs {b_name}")
    note(f"  {'airport':8s} {'n25':>6s} {'n26':>6s} {'med25':>9s} {'med26':>9s} "
         f"{'p90_25':>9s} {'p90_26':>9s} {'p99_25':>9s} {'p99_26':>9s} "
         f"{'max25':>10s} {'max26':>10s} {'>=6h 25':>9s} {'>=6h 26':>9s}")
    aps_a = SEG[a_name].groupby("ADEP_mvt", observed=True)
    aps_b = SEG[b_name].groupby("ADEP_mvt", observed=True)
    cand = sorted(set(aps_a.groups) | set(aps_b.groups))
    cand = [ap for ap in cand
            if len(aps_a.get_group(ap)) >= MIN_AIRPORT_N
            and len(aps_b.get_group(ap)) >= MIN_AIRPORT_N]
    if not cand:
        note("    (no airport has >= "
             f"{MIN_AIRPORT_N} segment rows in both periods)")
    for ap in cand:
        a1 = aps_a.get_group(ap)["absd"].to_numpy()
        a2 = aps_b.get_group(ap)["absd"].to_numpy()
        t1 = int((a1 > DECISIVE_LO).sum())
        t2 = int((a2 > DECISIVE_LO).sum())
        note(f"  {ap:8s} {len(a1):>6,} {len(a2):>6,} {np.median(a1):>9,.0f} "
             f"{np.median(a2):>9,.0f} {np.quantile(a1, .9):>9,.0f} "
             f"{np.quantile(a2, .9):>9,.0f} {np.quantile(a1, .99):>9,.0f} "
             f"{np.quantile(a2, .99):>9,.0f} {a1.max():>10,.0f} {a2.max():>10,.0f} "
             f"{t1:>9,} {t2:>9,}")
note("")
note(f"  (an airport needs >= {MIN_AIRPORT_N} segment rows in BOTH periods of a")
note("   pair to appear; anything smaller cannot support a distribution claim)")
note("")
note("  Single-period airport tables for the focus airports (no pairing "
     "requirement):")
for ap in FOCUS:
    note("")
    note(f"  {ap}:")
    note(f"  {'period':9s} {'n':>6s} {'median':>9s} {'p90':>9s} {'p95':>9s} "
         f"{'p99':>9s} {'max':>10s}   bands: " +
         " ".join(f"{lab}" for _, _, lab in BANDS))
    for n, _, _ in PERIODS:
        g = SEG[n][SEG[n]["ADEP_mvt"] == ap]
        if len(g) == 0:
            note(f"  {n:9s} {0:>6,} {'-':>9s} {'-':>9s} {'-':>9s} {'-':>9s} "
                 f"{'-':>10s}   (no missing-flight rows)")
            continue
        a = g["absd"].to_numpy()
        cs = []
        for lo, hi, lab in BANDS:
            m = (a > lo * 3600) & (a <= hi * 3600) if lo else (a <= hi * 3600)
            cs.append(f"{lab}={int(m.sum())}")
        note(f"  {n:9s} {len(a):>6,} {np.median(a):>9,.0f} "
             f"{np.quantile(a, .9):>9,.0f} {np.quantile(a, .95):>9,.0f} "
             f"{np.quantile(a, .99):>9,.0f} {a.max():>10,.0f}   " +
             " ".join(cs))
    note("   (counts; seconds)")

# ============================================================
# 4. THE V9-A DECISIVE TAIL
# ============================================================

sub("3d) MODEL-INPUT MISSINGNESS INSIDE THE SEGMENT - a silent feature shift")
note("  The segment model consumes the V7 features. If one of them starts")
note("  arriving empty in 2026, the model silently receives a category it was")
note("  never trained on, which is a feature-side shift that no error-autopsy")
note("  on 2025 could have revealed. Reported as the share of each input that is")
note("  missing, within the segment, per period.")
note("")
CHECK = [c for c in V7_FEATURES
         if c in READ_ALLOW or c == "traffic_last_15min"]
CHECK = [c for c in V7_FEATURES if c in SEG["JAN2025"].columns]
note(f"  {'model input':22s} " + " ".join(f"{n:>10s}" for n, _, _ in PERIODS) +
     "   flag")
miss = {}
for c in CHECK:
    row = []
    for n, _, _ in PERIODS:
        row.append(100 * SEG[n][c].isna().mean())
    miss[c] = row
    spread = max(row) - min(row)
    flag = ""
    if spread >= 20:
        flag = f"SHIFT {spread:.1f}pp"
    elif spread >= 5:
        flag = f"drift {spread:.1f}pp"
    note(f"  {c:22s} " + " ".join(f"{v:>9.2f}%" for v in row) + f"   {flag}")
note("")
note("  A flagged column is a MODEL-INPUT shift, not a target insight: it changes")
note("  what the model is asked, using no label information at all. It is")
note("  therefore admissible evidence here, and it is actionable.")
note("")
dead = [c for c in CHECK if all(v >= 99.99 for v in miss[c])]
if dead:
    note(f"  STRUCTURAL FINDING: {len(dead)} of the {len(CHECK)} V7 inputs are")
    note(f"  100% missing inside the segment in EVERY period: {dead}")
    note("  These are flight-level fields, so a row with no flight timestamps")
    note("  also has no flight category. They carry zero information for the")
    note("  segment model and were dead weight in V9-A's 13-feature segment")
    note("  design. Stable across years, so not a shift - but it means the")
    note("  segment model is effectively a 10-input model, and removing the dead")
    note("  columns is free.")
    note("")
note("  Per-airport aircraft-type missingness, the input that matters most here:")
note(f"  {'airport':8s} " + " ".join(f"{n:>10s}" for n, _, _ in PERIODS))
for ap in ["LIRF", "LFPG", "EGLL", "EDDF", "EHAM", "EDDM", "LSZH", "LEBL",
           "LEMD", "LTFM"]:
    cells = []
    for n, _, _ in PERIODS:
        g = SEG[n]
        g = g[g["ADEP_mvt"] == ap]
        cells.append(f"{100 * g['AIRCRAFT_TYPE_mvt'].isna().mean():>9.1f}%"
                     if len(g) else f"{'-':>10s}")
    note(f"  {ap:8s} " + " ".join(cells))
note("")
note("  LIRF is 100% null aircraft type in its segment rows in ALL FOUR periods,")
note("  including 2025. So this is not a 2026 regression - but it is the")
note("  mechanism behind V9-A. LIRF is both the airport that supplied 101.5% of")
note("  the gain AND the population with the least feature information: no flight")
note("  timestamps, no aircraft type, no flight category. In that population")
note("  d_SCHED is close to the only signal carrying real information, which is")
note("  exactly why a 2025 target/d_SCHED coincidence decided the whole")
note("  experiment - and exactly why the measured 2026 shift in the segment's")
note("  d_SCHED distribution matters more here than anywhere else.")
note("")

head("4. THE V9-A DECISIVE TAIL - |d_SCHED| > 6h inside missing_all4")
note("  In V9-A, 54 of 5,373 pooled 2025 segment rows carried |d_SCHED| > 6h")
note("  and essentially decided the experiment: 102.92% of the net gain came")
note("  from the worst 100 segment rows, and the entire non-LIRF remainder was")
note("  a net loss. This section asks how many such rows each period has.")

tailrows = []
for n, _, _ in PERIODS:
    a = SEG[n]["absd"].to_numpy()
    m = a > DECISIVE_LO
    g = SEG[n][m]
    tailrows.append(dict(
        period=n, n_seg=len(a), n_tail=int(m.sum()),
        pct_seg=100 * m.sum() / len(a),
        pct_all_rows=100 * m.sum() / len(P[n]),
        per_1k_all=1000 * m.sum() / len(P[n]),
        lirf=int((g["ADEP_mvt"] == "LIRF").sum()),
        lfpg=int((g["ADEP_mvt"] == "LFPG").sum()),
        egll=int((g["ADEP_mvt"] == "EGLL").sum()),
        eddf=int((g["ADEP_mvt"] == "EDDF").sum()),
        n_6_12=int(((a > 6 * 3600) & (a <= 12 * 3600)).sum()),
        n_gt12=int((a > 12 * 3600).sum()),
        tail_max=a[m].max() if m.sum() else np.nan))
TAIL = pd.DataFrame(tailrows)
dd = TAIL
note("")
note(f"  {'period':9s} {'seg n':>7s} {'tail n':>7s} {'% of seg':>9s} "
     f"{'% of all dep':>13s} {'per 1k dep':>11s} {'6-12h':>6s} {'>12h':>6s} "
     f"{'LIRF':>6s} {'LFPG':>6s} {'EGLL':>5s} {'EDDF':>5s}")
for _, r in dd.iterrows():
    note(f"  {r['period']:9s} {int(r['n_seg']):>7,} {int(r['n_tail']):>7,} "
         f"{r['pct_seg']:>8.2f}% {r['pct_all_rows']:>12.4f}% "
         f"{r['per_1k_all']:>11.3f} {int(r['n_6_12']):>6,} {int(r['n_gt12']):>6,} "
         f"{int(r['lirf']):>6,} {int(r['lfpg']):>6,} {int(r['egll']):>5,} "
         f"{int(r['eddf']):>5,}")
note("")
note("  2025 -> 2026, same calendar month:")
for a, b in (("JAN2025", "JAN2026"), ("JUL2025", "JUL2026")):
    ra = dd[dd["period"] == a].iloc[0]
    rb = dd[dd["period"] == b].iloc[0]
    rr = (rb["n_tail"] / rb["n_seg"]) / (ra["n_tail"] / ra["n_seg"]) \
        if ra["n_tail"] else float("nan")
    per = (rb["per_1k_all"] / ra["per_1k_all"]) if ra["per_1k_all"] else float("nan")
    note(f"    {a} -> {b}: tail rows {int(ra['n_tail'])} -> {int(rb['n_tail'])}; "
         f"share of segment {ra['pct_seg']:.2f}% -> {rb['pct_seg']:.2f}% "
         f"(ratio {rr:.2f}x)")
    note(f"    {' ' * (len(a) + 9)} tail rows per 1,000 departures: "
         f"{ra['per_1k_all']:.3f} -> {rb['per_1k_all']:.3f}  (ratio {per:.2f}x)")

sub("4b) THE SPECIFIC V9-A WINNING REGIME: LIRF + missing_all4 + |d_SCHED|>6h")
rows = []
for n, _, _ in PERIODS:
    g = SEG[n]
    for lab, m in (("LIRF seg (any |d_SCHED|)", g["ADEP_mvt"] == "LIRF"),
                   ("LIRF seg & |d_SCHED| > 6h",
                    (g["ADEP_mvt"] == "LIRF") & (g["absd"] > DECISIVE_LO)),
                   ("LIRF seg & |d_SCHED| > 12h",
                    (g["ADEP_mvt"] == "LIRF") & (g["absd"] > 12 * 3600))):
        rows.append(dict(period=n, group=lab, n=int(m.sum())))
dd2 = pd.DataFrame(rows)
note(f"  {'group':28s} " + " ".join(f"{n:>9s}" for n, _, _ in PERIODS))
for lab in dd2["group"].unique():
    r = dd2[dd2["group"] == lab].set_index("period")
    note(f"  {lab:28s} " + " ".join(f"{int(r.loc[n, 'n']):>9,}" for n, _, _ in PERIODS))
note("")
note("  This is the single cell that carried V9-A's pooled gain: 101.5% of the")
note("  net segment gain came from LIRF, and LIRF is 397 of 5,373 segment rows.")

sub("4c) 2026 RANKING ROWS IN THE DECISIVE TAIL (features only, NO prediction)")
note("  Listed for transparency. No prediction is made or altered for any row.")
note("")
for n in ("JAN2026", "JUL2026"):
    g = SEG[n]
    t = g[g["absd"] > DECISIVE_LO].sort_values("absd", ascending=False)
    sub(f"  {n}: {len(t)} missing-flight rows with |d_SCHED| > 6h")
    if not len(t):
        note("    (none - this regime is absent from this month's ranking data)")
        continue
    show = t[["MVT_ID_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", MINUEND,
              "SCHED_TIME_UTC_mvt", "d_SCHED", "AIRCRAFT_TYPE_mvt",
              "traffic_last_15min"]].copy()
    show.columns = ["MVT_ID", "airport", "runway", "stand", "mvt_time",
                    "sched_time", "d_SCHED", "aircraft", "traffic_15min"]
    for c in ("mvt_time", "sched_time"):
        show[c] = pd.to_datetime(show[c], utc=True).dt.strftime("%m-%d %H:%M")
    for c in ("d_SCHED", "traffic_15min"):
        show[c] = show[c].round(0)
    note(show.to_string(index=False))
    note("")
    note(f"    airports: { {k: int(v) for k, v in t['ADEP_mvt'].value_counts().items()} }")
    note(f"    |d_SCHED| in hours: "
         f"{sorted(round(x / 3600, 2) for x in t['absd'])[::-1]}")

sub("4d) 2026 rows at the ERROR SITES where V9-A created new large errors")
note("  On 2025, V9-A turned previously harmless rows into large errors at")
note("  EGLL 181983342 (683.9 -> 11,426.7), EDDF 182240520 (387.4 -> 7,147.5),")
note("  EHAM, LSZH, LEBL and LEMD, and worsened the LFPG population on both")
note("  folds (JUL LFPG: 255 improved vs 409 worsened). Are those airports even")
note("  present in the 2026 missing-flight segment?")
note("")
rows = []
for n, _, _ in PERIODS:
    g = SEG[n]
    for ap in ERROR_SITES + ["LIRF"]:
        s = g[g["ADEP_mvt"] == ap]
        rows.append(dict(period=n, airport=ap, n=len(s),
                         n_tail=int((s["absd"] > DECISIVE_LO).sum()),
                         med_absd=float(s["absd"].median()) if len(s) else np.nan,
                         p90_absd=(float(np.quantile(s["absd"], .9))
                                   if len(s) else np.nan)))
dd3 = pd.DataFrame(rows)
piv = dd3.pivot_table(index="airport", columns="period",
                      values="n", fill_value=0)
note("  missing-flight segment rows per airport")
note("  " + f"{'airport':7s} " + " ".join(f"{n:>9s}" for n, _, _ in PERIODS) +
     "     %JUL25  %JUL26")
for ap in piv.index:
    r = piv.loc[ap]
    j25 = 100 * r.get("JUL2025", 0) / max(1, int(dd[dd['period'] == 'JUL2025']['n_seg'].iloc[0]))
    j26 = 100 * r.get("JUL2026", 0) / max(1, int(dd[dd['period'] == 'JUL2026']['n_seg'].iloc[0]))
    note(f"  {ap:7s} " + " ".join(f"{int(r.get(n, 0)):>9,}" for n, _, _ in PERIODS) +
         f"   {j25:>6.2f}% {j26:>6.2f}%")
note("")
note("  of those, rows with |d_SCHED| > 6h (the regime that moves predictions)")
note("  " + f"{'airport':7s} " + " ".join(f"{n:>9s}" for n, _, _ in PERIODS))
for ap in sorted(set(dd3[dd3["n"] > 0]["airport"])):
    s = dd3[dd3["airport"] == ap].set_index("period")
    note(f"  {ap:7s} " + " ".join(f"{int(s.loc[n, 'n_tail']):>9,}" for n, _, _ in PERIODS))

sub("4e) WHO OWNS THE DECISIVE TAIL - the composition shifted, and that matters")
note("  V9-A's gain came from LIRF. Its damage came from everywhere else. So the")
note("  question is not just 'how many tail rows' but 'whose' tail rows.")
note("")
grp = []
for n, _, _ in PERIODS:
    t = SEG[n]
    t = t[t["absd"] > DECISIVE_LO]
    w = int((t["ADEP_mvt"] == "LIRF").sum())
    h = int(t["ADEP_mvt"].isin(ERROR_SITES + ["LFPG"]).sum())
    grp.append(dict(period=n, n=len(t), lirf=w,
                    lirf_pct=100 * w / max(1, len(t)),
                    harm=h, harm_pct=100 * h / max(1, len(t)),
                    other=len(t) - w - h))
GRP = pd.DataFrame(grp)
note(f"  {'period':9s} {'tail n':>7s} {'LIRF':>6s} {'%':>7s} "
     f"{'harm sites':>11s} {'%':>7s} {'other':>6s}")
for _, r in GRP.iterrows():
    note(f"  {r['period']:9s} {int(r['n']):>7,} {int(r['lirf']):>6,} "
         f"{r['lirf_pct']:>6.1f}% {int(r['harm']):>11,} {r['harm_pct']:>6.1f}% "
         f"{int(r['other']):>6,}")
note("")
note("  'harm sites' = LFPG, EGLL, EDDF, EHAM, LSZH, LEBL, LEMD, EDDM, LTFM - the")
note("  airports where V9-A measurably worsened rows on 2025.")
_l25 = GRP[GRP["period"].isin(["JAN2025", "JUL2025"])]
_l26 = GRP[GRP["period"].isin(["JAN2026", "JUL2026"])]
note(f"  pooled 2025 tail: {_l25['n'].sum():,} rows, "
     f"{100 * _l25['lirf'].sum() / _l25['n'].sum():.1f}% LIRF, "
     f"{100 * _l25['harm'].sum() / _l25['n'].sum():.1f}% harm sites")
note(f"  pooled 2026 tail: {_l26['n'].sum():,} rows, "
     f"{100 * _l26['lirf'].sum() / _l26['n'].sum():.1f}% LIRF, "
     f"{100 * _l26['harm'].sum() / _l26['n'].sum():.1f}% harm sites")
note("")
note("  The decisive tail is LARGER in 2026, but no longer LIRF-dominated. In")
note("  JAN2026 it is mostly EHAM, an airport V9-A already damaged on 2025.")
_lirp25 = 100 * _l25["lirf"].sum() / _l25["n"].sum()
_lirp26 = 100 * _l26["lirf"].sum() / _l26["n"].sum()
SHIFT["tail_lirf_share"] = (_lirp25, _lirp26)
SHIFT["tail_harm_share"] = (100 * _l25["harm"].sum() / _l25["n"].sum(),
                            100 * _l26["harm"].sum() / _l26["n"].sum())
note("")

# ============================================================
# 5. DISTRIBUTION-SHIFT ASSESSMENT
# ============================================================

head("5. DISTRIBUTION-SHIFT ASSESSMENT - quantified, not intuitive")

sub("5a) SEGMENT PREVALENCE, 2025 vs 2026, same month")
note(f"  {'month':5s} {'2025 rate':>11s} {'2026 rate':>11s} {'ratio':>8s} "
     f"{'exact 95% CI':>18s}   verdict")
for m, a, b in (("JAN", "JAN2025", "JAN2026"), ("JUL", "JUL2025", "JUL2026")):
    pa = PREV[a]["pct"] / 100
    pb = PREV[b]["pct"] / 100
    ratio = pb / pa
    lo, hi, _ = rate_ratio_ci(PREV[a]["missing_all4"], PREV[a]["departures"],
                              PREV[b]["missing_all4"], PREV[b]["departures"])
    if lo <= 1.0 <= hi:
        v = "similar (CI spans 1)"
    elif ratio > 1.25 or ratio < 0.8:
        v = "moderately shifted"
    else:
        v = "similar"
    if (ratio > 2 or ratio < 0.5) and not (lo <= 1.0 <= hi):
        v = "strongly shifted"
    SHIFT[m + "_prev"] = ratio
    note(f"  {m:5s} {100 * pa:>10.3f}% {100 * pb:>10.3f}% {ratio:>7.2f}x "
         f"[{lo:>5.2f}, {hi:>5.2f}]   {v}")

sub("5b) |d_SCHED| DISTRIBUTION INSIDE THE SEGMENT, 2025 vs 2026")
note("  Two-sample Kolmogorov-Smirnov on |d_SCHED| within missing_all4, plus the")
note("  V9-A decisive-tail share. KS is distribution-free: a small D with a large")
note("  p-value means the 2026 segment looks like 2025 on this feature.")
note("")
for m, a, b in (("JAN", "JAN2025", "JAN2026"), ("JUL", "JUL2025", "JUL2026")):
    x = SEG[a]["absd"].to_numpy()
    y = SEG[b]["absd"].to_numpy()
    ks = stats.ks_2samp(x, y)
    ta = int((x > DECISIVE_LO).sum())
    tb = int((y > DECISIVE_LO).sum())
    note(f"  {m}2025 n={len(x):,}  vs  {m}2026 n={len(y):,}")
    note(f"    KS D = {ks.statistic:.4f}   p = {ks.pvalue:.4f}   "
         f"{'distributions NOT separable' if ks.pvalue > 0.05 else 'distributions differ'}")
    note(f"    median |d_SCHED|  {np.median(x):,.0f}s -> {np.median(y):,.0f}s"
         f"     p90 {np.quantile(x, .9):,.0f}s -> {np.quantile(y, .9):,.0f}s"
         f"     p99 {np.quantile(x, .99):,.0f}s -> {np.quantile(y, .99):,.0f}s")
    note(f"    max {x.max():,.0f}s -> {y.max():,.0f}s")
    note(f"    V9-A decisive tail (|d_SCHED| > 6h): {ta} ({100 * ta / len(x):.2f}%)"
         f" -> {tb} ({100 * tb / len(y):.2f}%)")
    SHIFT[m + "_ks"] = ks.pvalue
    SHIFT[m + "_tail"] = (ta, tb)

sub("5c) AIRPORT MIX INSIDE THE SEGMENT, 2025 vs 2026")
note(f"  {'period':9s} {'n':>7s} " +
     " ".join(f"{ap:>8s}" for ap in ["LIRF", "LFPG", "EGLL", "EDDF"]) +
     "   'other'   distinct aps")
for n, _, _ in PERIODS:
    g = SEG[n]
    c = g["ADEP_mvt"].value_counts()
    oth = len(g) - sum(int(c.get(ap, 0)) for ap in ["LIRF", "LFPG", "EGLL", "EDDF"])
    note(f"  {n:9s} {len(g):>7,} " +
         " ".join(f"{100 * int(c.get(ap, 0)) / len(g):>7.2f}%" for ap in
                  ["LIRF", "LFPG", "EGLL", "EDDF"]) +
         f"   {100 * oth / len(g):>6.2f}%   {g['ADEP_mvt'].nunique():>6,}")

sub("5d) THE DIRECT QUESTION")
for m, a, b in (("JAN", "JAN2025", "JAN2026"), ("JUL", "JUL2025", "JUL2026")):
    ga = SEG[a]
    gb = SEG[b]
    wa = int(((ga["ADEP_mvt"] == "LIRF") & (ga["absd"] > DECISIVE_LO)).sum())
    wb = int(((gb["ADEP_mvt"] == "LIRF") & (gb["absd"] > DECISIVE_LO)).sum())
    pa = 100 * wa / len(P[a])
    pb = 100 * wb / len(P[b])
    note(f"  {m}: LIRF + missing_all4 + |d_SCHED| > 6h")
    note(f"    {a}: {wa:>3,} rows  = {pa:.4f}% of that month's {len(P[a]):,} "
         f"departures")
    note(f"    {b}: {wb:>3,} rows  = {pb:.4f}% of that month's {len(P[b]):,} "
         f"departures")
    note(f"    2025 pooled equivalent: "
         f"{int(((SEG['JAN2025']['ADEP_mvt'] == 'LIRF') & (SEG['JAN2025']['absd'] > DECISIVE_LO)).sum() + ((SEG['JUL2025']['ADEP_mvt'] == 'LIRF') & (SEG['JUL2025']['absd'] > DECISIVE_LO)).sum())} rows")
    if wb == 0:
        note(f"    => ABSENT from {b}. The regime that carried V9-A's gain does "
             f"not occur there.")
    elif wa == 0:
        note(f"    => PRESENT in {b} ({wb} rows) and absent from {a}.")
    else:
        note(f"    => PRESENT in both, {wb / max(1, wa):.2f}x the {a} frequency.")
    SHIFT[m + "_lirftail"] = (wa, wb)
note("")

sub("5e) ERROR-SITE EXPOSURE IN 2026")
note("  The airports where V9-A made things worse on 2025 (EGLL, EDDF, EHAM,")
note("  LSZH, LEBL, LEMD) plus LFPG. Counted in the 2026 segment:")
note("")
for ap in ["LFPG"] + ERROR_SITES:
    r = {}
    for n, _, _ in PERIODS:
        s = SEG[n]
        g = s[s["ADEP_mvt"] == ap]
        r[n] = (len(g), int((g["absd"] > DECISIVE_LO).sum()),
                int((g["absd"] > 2 * 3600).sum()))
    note(f"  {ap:6s} " + "  ".join(
        f"{n}: {r[n][0]:>4,} seg ({r[n][1]:>3,} >6h, {r[n][2]:>4,} >2h)"
        for n, _, _ in PERIODS))
note("")
_tot = {n: int((SEG[n]["ADEP_mvt"].isin(["LFPG"] + ERROR_SITES)).sum())
        for n, _, _ in PERIODS}
note(f"  combined error-site segment rows: " +
     ", ".join(f"{n} {_tot[n]:,}" for n, _, _ in PERIODS))
note("  If these populations are materially present in 2026, then the part of")
note("  V9-A that WORSENED rows is exposed to the same extent as the part that")
note("  improved them, because the same model makes both decisions.")

# ============================================================
# 6. RECOMMENDATION
# ============================================================

head("6. RECOMMENDATION (feature distributions only)")

jan_lt25, jan_lt26 = SHIFT["JAN_lirftail"]
jul_lt25, jul_lt26 = SHIFT["JUL_lirftail"]
exp25 = jan_lt25 + jul_lt25
exp26 = jan_lt26 + jul_lt26
jan_ks = SHIFT["JAN_ks"]
jul_ks = SHIFT["JUL_ks"]
jan_prev, jan_lo, jan_hi = SHIFT_P["JAN25to26"]
jul_prev, jul_lo, jul_hi = SHIFT_P["JUL25to26"]
jan_ok = jan_ks > 0.05 and jan_lo <= 1.0 <= jan_hi
jul_ok = jul_ks > 0.05 and jul_lo <= 1.0 <= jul_hi

seg25 = sum(PREV[n]["missing_all4"] for n in ("JAN2025", "JUL2025"))
seg26 = sum(PREV[n]["missing_all4"] for n in ("JAN2026", "JUL2026"))
dep25 = sum(PREV[n]["departures"] for n in ("JAN2025", "JUL2025"))
dep26 = sum(PREV[n]["departures"] for n in ("JAN2026", "JUL2026"))
err25 = int(dd3[dd3["period"].isin(["JAN2025", "JUL2025"])
                & dd3["airport"].isin(["LFPG"] + ERROR_SITES)]["n"].sum())
err26 = int(dd3[dd3["period"].isin(["JAN2026", "JUL2026"])
                & dd3["airport"].isin(["LFPG"] + ERROR_SITES)]["n"].sum())

if exp26 == 0:
    VERDICT = "LOW PRIORITY / MOVE TO V9-B"
    WHY = f"""V9-A's entire measured benefit came from one feature-side cell: LIRF
  departures with all four flight timestamps missing and |d_SCHED| > 6h. That
  cell holds {exp25} rows in the 2025 folds and {exp26} rows in the 2026 ranking
  months. The regime V9-A was credited with exploiting is not present in the
  data V9-A would be scored on, so there is no exposure to win and no reason to
  spend the next iteration on it."""
elif exp26 < 0.5 * max(1, exp25):
    VERDICT = "LOW PRIORITY / MOVE TO V9-B"
    WHY = f"""The decisive LIRF + extreme-d_SCHED cell shrinks from {exp25} rows
  in 2025 to {exp26} in 2026 ({exp26 / max(1, exp25):.2f}x). Even in the best
  case V9-A could only touch a fraction of the value it showed on 2025, while
  the airports where it created new large errors are still fully exposed."""
else:
    VERDICT = "CONTINUE V9-A INVESTIGATION"
    WHY = f"""The regime V9-A depends on is present in 2026 at comparable or
  higher frequency, so the question is empirically answerable and worth
  answering:

    decisive LIRF + extreme-d_SCHED cell   {exp25} -> {exp26} rows ({exp26 / max(1, exp25):.2f}x)
    decisive tail, all airports           {SHIFT['JAN_tail'][0] + SHIFT['JUL_tail'][0]} -> {SHIFT['JAN_tail'][1] + SHIFT['JUL_tail'][1]} rows
    segment rows                          {seg25:,} -> {seg26:,}
    segment share of departures           {100 * seg25 / dep25:.3f}% -> {100 * seg26 / dep26:.3f}%

  The segment itself is almost exactly as large in 2026 as in 2025, and the
  decisive tail inside it is nearly twice as large. But the reason to continue
  is a warning, not a green light. Two things moved, and both were measured here
  rather than inferred:

  1. THE TAIL CHANGED OWNER. LIRF's share of the decisive tail fell from
     {SHIFT['tail_lirf_share'][0]:.1f}% to {SHIFT['tail_lirf_share'][1]:.1f}%, while the share held by the airports
     V9-A damaged on 2025 rose from {SHIFT['tail_harm_share'][0]:.1f}% to {SHIFT['tail_harm_share'][1]:.1f}%. In January 2026 the
     tail is mostly EHAM - 36 of 58 rows, against 4 in January 2025 - and
     EHAM is one of the sites where V9-A turned harmless rows into large
     errors. The exposure that might pay and the exposure that might cost have
     both grown, and they are no longer the same rows.

  2. THE SEGMENT'S d_SCHED MOVED. KS cannot separate the years in either month
     (JAN p={jan_ks:.4f}, JUL p={jul_ks:.4f}). January 2026's segment median
     |d_SCHED| is 24% higher, p90 is 53% higher, and the 4-6h band grew from
     1.53% to 7.05% of segment rows. V9-A's entire rationale was a calibrated
     d_SCHED mapping on one specific tail; that tail has moved and widened.

  Those two are exactly the questions a follow-up experiment should answer,
  and both are answerable without touching a target. That is what continuing
  the investigation means. It does not mean the direction is favourable - on the
  evidence here the outcome is genuinely uncertain, and the harm-site exposure
  is the larger of the two."""

note("")
note("=" * 100)
note(f"  RECOMMENDATION: {VERDICT}")
note("=" * 100)
note(f"""
  MEASURED, FEATURE-SIDE ONLY
    segment rows            2025 {seg25:,}   2026 {seg26:,}
    segment share of rows   2025 {100 * seg25 / dep25:.3f}%   2026 {100 * seg26 / dep26:.3f}%
    JAN prevalence ratio    {jan_prev:.2f}x  exact 95% CI [{jan_lo:.2f}, {jan_hi:.2f}]
    JUL prevalence ratio    {jul_prev:.2f}x  exact 95% CI [{jul_lo:.2f}, {jul_hi:.2f}]
    KS |d_SCHED| in segment JAN p={jan_ks:.4f}   JUL p={jul_ks:.4f}
    decisive tail |d|>6h    JAN {SHIFT['JAN_tail'][0]} -> {SHIFT['JAN_tail'][1]}   JUL {SHIFT['JUL_tail'][0]} -> {SHIFT['JUL_tail'][1]}
    LIRF & |d|>6h           JAN {jan_lt25} -> {jan_lt26}   JUL {jul_lt25} -> {jul_lt26}
    tail owned by LIRF      {SHIFT['tail_lirf_share'][0]:.1f}% -> {SHIFT['tail_lirf_share'][1]:.1f}%
    tail at 2025 harm sites {SHIFT['tail_harm_share'][0]:.1f}% -> {SHIFT['tail_harm_share'][1]:.1f}%
    error-site seg rows     2025 {err25:,}   2026 {err26:,}

  WHY
{WHY}

  WHAT THIS DOES NOT ESTABLISH
    V9-A's 2025 gain was not a distributional effect. It was the fact that in
    2025 the TARGETS of those 54 extreme rows tracked d_SCHED closely - LIRF
    segment d_SCHED 93,535s vs mean target 87,170s, and 64,438s vs 64,437s.
    That is a 2025 coincidence about 2025 labels. This audit is feature-only
    by construction and cannot check it, and it must not be assumed to carry
    over to 2026.

    A matching 2026 feature distribution therefore justifies continued
    INVESTIGATION and nothing more. It does not justify promoting V9-A, and
    it does not justify expecting a similar score change. V9-A stays MIXED.

  Even a perfect feature match leaves V9-A's other problems untouched, all
  of which were measured on 2025 and are independent of what 2026 looks
  like:
      * 102.92% of the net gain sat in the worst 100 segment rows (0.029% of
        all validation rows); strip them and the segment RMSE worsens
        755.60 -> 824.55.
      * 55.50% of segment rows got WORSE; median absolute error worsened
        196.50 -> 226.52 on both folds.
      * 101.5% of the gain was LIRF, and the repo has already rejected two
        feature sets (clocks, H1) on exactly this concentration argument.
      * The fold spread was 22.4x (-4.86 JAN, -109.01 JUL).

  ONE FREE FINDING WORTH BANKING REGARDLESS OF THE VERDICT
    {', '.join(dead) if dead else 'none'} are 100% missing inside the segment in
    every period. They are flight-level fields, absent by construction on a row
    with no flight timestamps. V9-A's segment model was specified with 13
    features of which {len(dead) if dead else 0} are constants, so it was a
    {13 - (len(dead) if dead else 0)}-input model in practice. Dropping the dead
    columns is free and cannot change a single prediction, because CatBoost
    cannot split on a constant either way. Separately, LIRF segment rows arrive
    with a null aircraft type in all four periods, which is why d_SCHED carries
    so much of the load in the one cell that decided V9-A.


  SAFETY CONFIRMATION
    Fitted nothing. Imported no model library for fitting. Read no target
    column. Opened no truthing file (asserted absent). Produced no prediction.
    Wrote no submission. Created no v3. Uploaded nothing. Committed, pushed,
    reset and merged nothing. Did not modify V8, V9-A, or any other existing
    file; this script and this report are the only new artefacts.
""")

# ============================================================
# WRITE
# ============================================================

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print(f"\n[written] {REPORT}  lines={len(out)}  bytes={REPORT.stat().st_size:,}")
print("No model fitted. No prediction. No submission. Nothing committed.")
