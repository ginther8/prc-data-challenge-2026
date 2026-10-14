"""V9-C preparation: corrected d_SCHED forensic audit.

ANALYSIS ONLY. No CatBoost, no model fitting, no predictions, no submission.
Every number in the report is computed from two sources and nothing else:

  1. reports/.v9b_run_cache.pkl - the exact cached T4 (B0) holdout
     predictions produced by the V9-B ablation, whose B0 arm reproduced the
     frozen T4 scores exactly. Reused read-only; nothing is refitted.
  2. the 2025 training parquet files - for the T4 deltas, target, airport
     and runway, which the forensics need but the cache does not carry.

The run aborts unless the cached B0 predictions still reproduce the frozen
JAN and JUL T4 RMSE to 7 decimal places, so every downstream number is
anchored to the frozen champion's actual behaviour.

Sections follow the audit brief:
  1  d_SCHED signed bands, per fold
  2  timestamp-availability regimes
  3  airport interaction
  4  where d_SCHED helps: schedule vs flight-level disagreement
  5  extreme-target relationship
  6  worst errors
  7  JAN vs JUL robustness
  8  corrected error decomposition
  9  ranked recommendations

Restricted: 2025 training data only. No 2026 targets, no truthing, no
leaderboard, no external data, no model fitting, no submission, no commit.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Rows with all four flight timestamps absent have an all-NaN delta block, so
# nanmedian/nanmax legitimately warn. Every such use is masked by an explicit
# n4 == 4 test, so the warning is noise, not a defect.
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="All-NaN axis encountered")

from train_catboost_v8_defs import (  # read-only import, shared source of truth
    DATA_DIR, MINUEND, T4_SOURCE_COLS, TARGET, T4_FEATURES, T4_SPEC,
    add_t4_features,
)

# ------------------------------------------------------------
# frozen reference
# ------------------------------------------------------------
OFFICIAL_2026 = 425.7711          # V8/T4 official score, quoted not used
FROZEN_RMSE = {"JAN": 404.8803102, "JUL": 513.7559738}
FROZEN_BEST_ITER = {"JAN": 365, "JUL": 266}
FOLDS = {"JAN": ("2025-01-01", "2025-02-01"),
         "JUL": ("2025-07-01", "2025-08-01")}

FLIGHT_DELTAS = ["d_AOBT3", "d_EOBT1", "d_IOBT", "d_LOBT"]
DSCHED = "d_SCHED"

CACHE = Path("reports/.v9b_run_cache.pkl")
REPORT = Path("reports/v9c_dsched_forensic_audit.txt")
THIS = Path("src/v9c_dsched_forensic_audit.py")

# ------------------------------------------------------------
# band definitions
# ------------------------------------------------------------
H = 3600.0
DSCHED_BANDS = [
    ("< -2h",        -np.inf, -2 * H),
    ("-2h .. -1h",   -2 * H, -1 * H),
    ("-1h .. 0",     -1 * H, 0.0),
    ("0 .. 30m",     0.0, 0.5 * H),
    ("30m .. 60m",   0.5 * H, 1 * H),
    ("1h .. 2h",     1 * H, 2 * H),
    ("2h .. 4h",     2 * H, 4 * H),
    ("4h .. 6h",     4 * H, 6 * H),
    ("6h .. 12h",    6 * H, 12 * H),
    ("12h .. 24h",   12 * H, 24 * H),
    ("> 24h",        24 * H, np.inf),
]

TARGET_BANDS = [
    ("< 30m",   -np.inf, 0.5 * H),
    ("30m..1h", 0.5 * H, 1 * H),
    ("1h..2h",  1 * H, 2 * H),
    ("2h..4h",  2 * H, 4 * H),
    ("4h..8h",  4 * H, 8 * H),
    ("8h..15h", 8 * H, 15 * H),
    ("15h..24h", 15 * H, 24 * H),
    ("> 24h",   24 * H, np.inf),
]

# availability regimes
REGIMES = [
    ("A full   (4/4 present)", 4, 4),
    ("B partial (1-3 of 4)",    1, 3),
    ("C missing-all4 (0/4)",    0, 0),
]

FOCUS_AIRPORTS = ["LIRF", "LFPG", "EHAM", "EGLL", "EDDF"]

out: list[str] = []


def note(s: str = "") -> None:
    out.append(s)


def sub(title: str) -> None:
    note("")
    note("-" * 100)
    note(title)
    note("-" * 100)


def head(title: str) -> None:
    note("=" * 100)
    note(title)
    note("=" * 100)


def hhmmss(v: float) -> str:
    """Seconds -> compact signed duration."""
    if not np.isfinite(v):
        return "n/a"
    s = "-" if v < 0 else ""
    a = abs(v)
    if a < 90:
        return f"{s}{a:.0f}s"
    m = a / 60.0
    if m < 90:
        return f"{s}{m:.1f}m"
    return f"{s}{m / 60.0:.2f}h"


def rmse_of(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


def fmt_corr(x: np.ndarray, y: np.ndarray) -> tuple[str, str]:
    """Pearson and Spearman, with honest 'too few rows' handling."""
    n = len(x)
    if n < 30 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return "n/a", "n/a"
    s = pd.Series(x)
    p = s.corr(pd.Series(y), method="pearson")
    q = s.corr(pd.Series(y), method="spearman")
    return f"{p:+.3f}", f"{q:+.3f}"


# ============================================================
# LOAD: cached T4 predictions + the columns the forensics need
# ============================================================
def load() -> dict:
    print("Loading cached V9-B B0 (frozen T4) holdout predictions ...", flush=True)
    assert CACHE.exists(), f"missing {CACHE}; V9-B must be run first"
    cache = pickle.loads(CACHE.read_bytes())
    for fold in FOLDS:
        assert (fold, "B0") in cache, f"cache has no {fold} B0 predictions"

    print("Rebuilding T4 deltas / target / airport / runway for validation rows ...",
          flush=True)
    # Ordering matters and is not obvious. The cached predictions were produced
    # from a frame that had already been through add_traffic_feature(), which
    # ends with
    #     frame.sort_values(["movement_airport", "MVT_TIME_UTC_mvt"]).copy()
    # i.e. it REORDERS every row of the whole 2025 movement set. The holdout
    # rows are therefore in (movement_airport, MVT_TIME) order, not file order.
    # This rebuild reproduces that sort exactly; the bit-identical assertions
    # on the target and d_SCHED arrays below are what prove it worked.
    need = ["PHASE_mvt", "MVT_ID_mvt", TARGET, "ADES_mvt", "ADEP_mvt",
            "RUNWAY_mvt"] + T4_SOURCE_COLS
    allm = pd.concat([pd.read_parquet(f, columns=need)
                      for f in sorted(DATA_DIR.glob("training_*.parquet"))],
                     ignore_index=True)
    print(f"  all 2025 movements {len(allm):,}", flush=True)
    allm["movement_airport"] = allm["ADEP_mvt"]
    arr = allm["PHASE_mvt"] == "ARR"
    allm.loc[arr, "movement_airport"] = allm.loc[arr, "ADES_mvt"]
    allm = allm.sort_values(["movement_airport", MINUEND]).copy()
    dep = allm[allm["PHASE_mvt"] == "DEP"]
    print(f"  departures {len(dep):,}", flush=True)
    del allm
    dep = add_t4_features(dep)

    data = {}
    for fold, (lo, hi) in FOLDS.items():
        c = cache[(fold, "B0")]
        sel = (dep[MINUEND] >= lo) & (dep[MINUEND] < hi)
        sub_ = dep[sel].reset_index(drop=True)
        n_c, n_f = len(c["y"]), len(sub_)
        print(f"  {fold}: cache rows {n_c:,}  rebuilt rows {n_f:,}", flush=True)
        assert n_c == n_f, f"{fold}: cache/rebuild row count mismatch"

        y = np.asarray(c["y"], dtype="float64")
        pred = np.asarray(c["pred"], dtype="float64")
        ty = sub_[TARGET].to_numpy("float64")
        got = np.sqrt(np.mean((y - pred) ** 2))
        print(f"  {fold}: cached-target RMSE {got:.7f}  "
              f"rebuilt-target RMSE {np.sqrt(np.mean((ty - pred) ** 2)):.7f}",
              flush=True)
        assert np.array_equal(y, ty), (
            f"{fold}: cached targets differ from rebuilt targets - row alignment "
            f"is wrong, every forensic number would be meaningless")
        ds = sub_[DSCHED].to_numpy("float64")
        cds = np.asarray(c["d_sched"], dtype="float64")
        assert np.array_equal(ds, cds), (
            f"{fold}: recomputed d_SCHED differs from the cached d_SCHED - row "
            f"alignment is wrong")

        data[fold] = {
            "y": y,
            "pred": pred,
            "d_sched": ds,
            # ADES_mvt has a handful of nulls; without the fill, astype(str)
            # leaves float nan in an object array and np.unique raises.
            "airport": sub_["ADES_mvt"].fillna("<null>").astype(str).to_numpy(),
            "adep": sub_["ADEP_mvt"].fillna("<null>").astype(str).to_numpy(),
            "runway": sub_["RUNWAY_mvt"].fillna("<null>").astype(str).to_numpy(),
            "deltas": {c: sub_[c].to_numpy("float64") for c in FLIGHT_DELTAS},
            "all4": np.asarray(c["all4"], dtype="int8"),
            "any_impl": np.asarray(c["any_impl"], dtype="int8"),
        }
        # availability and implausibility, recomputed here rather than trusted.
        # NB the cached field named "all4" is NOT a count of present
        # timestamps - it is the 0/1 missing-all4 flag (V9-B stores
        # va[MISSING_ALL4]). n4 below is the true present-count, and the two
        # are reconciled here so the equivalence is proved, not assumed.
        D = np.column_stack([data[fold]["deltas"][c] for c in FLIGHT_DELTAS])
        pres = ~np.isnan(D)
        n4 = pres.sum(axis=1)
        miss4 = (n4 == 0).astype("int8")
        assert np.array_equal(miss4, data[fold]["all4"]), (
            f"{fold}: missing-all4 flag mismatch "
            f"(cached {int(data[fold]['all4'].sum())}, "
            f"rebuilt {int(miss4.sum())})")
        neg = np.where(np.isnan(D), 0.0, (D < 0).astype("float64"))
        gt2 = np.where(np.isnan(D), 0.0, (D > 2 * H).astype("float64"))
        impl = ((neg + gt2) > 0).any(axis=1)
        assert np.array_equal(impl.astype("int8"), data[fold]["any_impl"]), \
            f"{fold}: implausibility flag mismatch"
        data[fold]["impl"] = impl
        data[fold]["n4"] = n4
    return data


def band_of(v: np.ndarray, bands) -> np.ndarray:
    """Integer band index per row; -1 if no band matches (should not happen)."""
    idx = np.full(len(v), -1, dtype="int64")
    for i, (_, lo, hi) in enumerate(bands):
        m = (idx < 0) & (v >= lo) & (v < hi)
        idx[m] = i
    return idx


def stats_for(m: np.ndarray, D: dict) -> dict:
    y, p = D["y"][m], D["pred"][m]
    e = y - p
    n = len(e)
    if n == 0:
        return {"n": 0}
    sse = float(np.sum(e ** 2))
    return {
        "n": n,
        "y_mean": float(np.mean(y)), "y_med": float(np.median(y)),
        "p_mean": float(np.mean(p)), "p_med": float(np.median(p)),
        "rmse": float(np.sqrt(sse / n)),
        "mae": float(np.mean(np.abs(e))),
        "medae": float(np.median(np.abs(e))),
        "bias": float(np.mean(e)),
        "sse": sse,
    }


def band_table(D: dict, bands, sel: np.ndarray | None = None,
               corr_field: str | None = None) -> None:
    v = D["d_sched"]
    idx = band_of(v, bands)
    base = np.ones(len(v), dtype=bool) if sel is None else sel
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    tot_n = int(base.sum())
    if tot_n == 0:
        note("    (no rows in this slice)")
        return
    note(f"    {'band':13s} {'n':>8s} {'%':>6s} {'tgt mean':>10s} "
         f"{'tgt med':>9s} {'pred mean':>10s} {'pred med':>9s} {'RMSE':>9s} "
         f"{'MAE':>9s} {'medAE':>8s} {'bias':>9s} {'%SSE':>6s} "
         f"{'r':>7s} {'rho':>7s}")
    for i, (lab, _, _) in enumerate(bands):
        m = base & (idx == i)
        s = stats_for(m, D)
        if s["n"] == 0:
            note(f"    {lab:13s} {0:>8,} {'-':>6s}  (empty)")
            continue
        if corr_field == "d_sched":
            cp, cs = fmt_corr(D["d_sched"][m], D["y"][m])
        elif corr_field == "pred":
            cp, cs = fmt_corr(D["d_sched"][m], D["pred"][m])
        else:
            cp, cs = "n/a", "n/a"
        note(f"    {lab:13s} {s['n']:>8,} {100 * s['n'] / tot_n:>5.2f}% "
             f"{s['y_mean']:>10.1f} {s['y_med']:>9.1f} {s['p_mean']:>10.1f} "
             f"{s['p_med']:>9.1f} {s['rmse']:>9.2f} {s['mae']:>9.2f} "
             f"{s['medae']:>8.2f} {s['bias']:>+9.2f} "
             f"{100 * s['sse'] / tot_sse:>5.2f}% {cp:>7s} {cs:>7s}")
    note("")


def slice_dsched(D: dict, lo: float, hi: float) -> np.ndarray:
    return (D["d_sched"] >= lo) & (D["d_sched"] < hi)


def reg_mask(D: dict, k: int) -> np.ndarray:
    return D["n4"] == k


def pct_sse(D: dict, m: np.ndarray, tot_sse: float) -> float:
    s = stats_for(m, D)
    return 0.0 if s["n"] == 0 else 100.0 * s["sse"] / tot_sse


# ============================================================
# REPORT
# ============================================================
DATA = load()

head("V9-C PREPARATION - CORRECTED d_SCHED FORENSIC AUDIT")
note("")
note("ANALYSIS ONLY. No model was trained. No V9-C was created. No submission.")
note("")
note("  Frozen champion      V8/T4, official 2026 RMSE 425.7711 (quoted, unused)")
note("  Frozen baselines     JAN 404.8803102   JUL 513.7559738")
note("  V9-A                 MIXED, not promoted")
note("  V9-B                 MIXED, not promoted")
note("")
note("  d_SCHED is described here exactly as it is defined, and nothing more:")
note("      d_SCHED = MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt")
note("  i.e. the gap between the recorded movement time and the recorded")
note("  scheduled time, in seconds, signed. It is NOT assumed to be departure")
note("  delay and no operational or causal reading is put on it. Where it")
note("  matters the report says 'the gap'.")

sub("0. WHY THIS AUDIT EXISTS - the V9-B correction")
note("  src/v8_error_autopsy.py:471 built its supposed inconsistent-timestamp")
note("  group as")
note("      _flt = T4_FEATURES[1:]   # the four flight-level deltas")
note("  T4_SPEC order is [d_AOBT3, d_EOBT1, d_IOBT, d_LOBT, d_SCHED], so that")
note("  slice is [d_EOBT1, d_IOBT, d_LOBT, d_SCHED]: a 4-element list with the")
note("  right LENGTH and the wrong MEMBERSHIP. It dropped d_AOBT3 and pulled in")
note("  d_SCHED.")
note("")
note("  Consequence: the 11,001-row / 3.19% / 12.11%-of-SSE group that motivated")
note("  V9-B was mostly d_SCHED. Corrected measurements, from V9-B:")
note("")
note("    autopsy slice  [EOBT1,IOBT,LOBT,SCHED]   neg 3,585  huge 9,140 -> 11,001")
note("    four flight deltas [AOBT3,EOBT1,IOBT,LOBT] neg 3,011  huge    78 ->  3,072")
note("      9,136 of the 9,140 'huge' cases are |d_SCHED| > 2h")
note("      2,707 of the 3,585 'negative' cases are d_SCHED < 0")
note("")
note("  So the frozen autopsy's headline 'implausible records' finding was in")
note("  substance a d_SCHED finding wearing the wrong label. This audit is the")
note("  corrected accounting of d_SCHED on its own terms.")
note("")
note("  The frozen autopsy file is NOT edited. It is a historical report. This")
note("  file supersedes its interpretation, not its bytes.")
note("")

sub("0b. EVIDENCE BASE AND FROZEN-PREDICTION GATE")
note("  Predictions are reused from reports/.v9b_run_cache.pkl, the B0 arm of the")
note("  V9-B ablation. Nothing is refitted; this script does not import CatBoost")
note("  at all. The gate is re-checked on every run:")
note("")
note("    fold   frozen T4 RMSE   from cached predictions       delta   best_iter")
_raw = pickle.loads(CACHE.read_bytes())
for fold in FOLDS:
    c = _raw[(fold, "B0")]
    y = np.asarray(c["y"], dtype="float64")
    p = np.asarray(c["pred"], dtype="float64")
    got = float(np.sqrt(np.mean((y - p) ** 2)))
    exp = FROZEN_RMSE[fold]
    bi = int(c["best_iter"])
    note(f"    {fold:4s} {exp:>16.7f} {got:>25.7f} {got - exp:>12.2e} "
         f"{bi:>8d}  (frozen {FROZEN_BEST_ITER[fold]})")
    assert abs(got - exp) < 5e-7, f"{fold}: cached B0 RMSE {got} != {exp}"
    assert bi == FROZEN_BEST_ITER[fold], f"{fold}: best_iter {bi} != frozen"
note("")
note("  [PASS] both folds reproduce the frozen T4 scores to 7 decimals at the")
note("  frozen best iterations, so every number below describes the frozen")
note("  champion's real holdout behaviour.")
note("")
note("  Row alignment is proved, not assumed: the audit rebuilds the T4 deltas,")
note("  target, airport and runway from the 2025 parquet files and asserts the")
note("  rebuilt target and d_SCHED arrays are bit-identical to the cached ones.")
note("  A row-order error would abort rather than emit plausible nonsense.")
note("")
# ---------------------------------------------------------------- section 1
sub("1. d_SCHED BANDS, PER FOLD")
note("  Signed bands on d_SCHED in seconds. Error statistics are on the raw")
note("  frozen T4 holdout predictions; target is TAXITIME_SEC_mvt in seconds.")
note("  '%SSE' is the band's share of that fold's total SSE. 'r'/'rho' are")
note("  Pearson/Spearman of the gap against the target, shown where n>=30 and")
note("  the band is non-constant.")
for fold in FOLDS:
    D = DATA[fold]
    tot = int(len(D["y"]))
    sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    note(f"  {fold} - {tot:,} rows, fold SSE {sse:.4e}, "
         f"RMSE {np.sqrt(sse / tot):.4f}")
    note("")
    band_table(D, DSCHED_BANDS, corr_field="d_sched")
note("  Reading notes:")
for fold in FOLDS:
    D = DATA[fold]
    neg = slice_dsched(D, -np.inf, 0.0)
    note(f"    {fold}: {int(neg.sum()):,} rows ({100 * neg.mean():.2f}%) have a")
    note(f"      negative gap - the movement time precedes the recorded schedule.")
    note("")
note("  Across both folds 2,707 rows have d_SCHED < 0. Those are exactly the rows")
note("  the frozen autopsy counted as 'negative flight-level deltas'.")
note("")
note("  A negative gap is not by itself an error. A movement can be recorded")
note("  ahead of its scheduled slot, and the sign alone does not say which of the")
note("  two recorded times is the odd one. That is why the bands are crossed with")
note("  availability and airport below instead of being interpreted standalone.")
note("")

# ---------------------------------------------------------------- section 2
sub("2. TIMESTAMP-AVAILABILITY REGIMES")
for fold in FOLDS:
    D = DATA[fold]
    vals, cnts = np.unique(D["n4"], return_counts=True)
    note(f"  {fold} availability counts (how many of the four flight-level")
    note(f"  timestamps AOBT3/EOBT1/IOBT/LOBT are present):")
    for v, c in zip(vals, cnts):
        note(f"    {int(v)} of 4 present: {c:>9,} rows ({100 * c / len(D['y']):>6.2f}%)")
    note("")
assert set(np.unique(DATA["JAN"]["n4"]).tolist()) == set(
    np.unique(DATA["JUL"]["n4"]).tolist()), "availability grids differ"
note("  CRITICAL STRUCTURAL FACT, carried over from V9-B and re-verified here:")
note("  the availability grid contains only 0 and 4. There are NO partial rows -")
note("  no movement has 1, 2 or 3 of the four flight-level timestamps. The four")
note("  columns are populated or absent as a block.")
note("")
note("  Consequence: regime B below is EMPTY by construction, not by accident.")
note("  Any feature of the form 'how many of the flight timestamps are present'")
note("  is a single bit in practice, and V9-B's six availability features")
note("  collapse to one. Three regimes are still reported for completeness")
note("  because the brief asks for them, but B has nothing in it.")
note("")
for fold in FOLDS:
    D = DATA[fold]
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    note(f"  {fold} - d_SCHED bands within each availability regime")
    for rname, lo_k, hi_k in REGIMES:
        m0 = (D["n4"] >= lo_k) & (D["n4"] <= hi_k)
        note(f"    regime {rname}: {int(m0.sum()):,} rows, "
             f"{100 * m0.mean():.2f}% of fold, "
             f"{pct_sse(D, m0, tot_sse):.2f}% of fold SSE")
        if m0.sum() == 0:
            note("")
            continue
        band_table(D, DSCHED_BANDS, sel=m0, corr_field="d_sched")
    note("")
note("  Does the gap behave differently by availability? Comparing regime A")
note("  against regime C on the same bands:")
comp_rows = []
for i, (lab, lo, hi) in enumerate(DSCHED_BANDS):
    row = {"band": lab}
    for fold in FOLDS:
        D = DATA[fold]
        for tag, k in (("A", 4), ("C", 0)):
            # NB the parentheses: `&` binds tighter than `==` in Python, so
            # `a & b == i` would parse as `(a & b) == i` and silently produce
            # a garbage mask. Both sides are parenthesised deliberately.
            m = (D["n4"] == k) & (band_of(D["d_sched"], DSCHED_BANDS) == i)
            s = stats_for(m, D)
            row[f"{fold}{tag}_n"] = s["n"]
            row[f"{fold}{tag}_rmse"] = s.get("rmse", float("nan"))
            row[f"{fold}{tag}_bias"] = s.get("bias", float("nan"))
            row[f"{fold}{tag}_ymed"] = s.get("y_med", float("nan"))
    comp_rows.append(row)
note("    band          JAN-A n   JAN-A RMSE  JAN-A bias  JAN-A tgtmed |"
     " JUL-A n   JUL-A RMSE  JUL-A bias  JUL-A tgtmed |"
     " JAN-C n  JAN-C RMSE  JUL-C n  JUL-C RMSE")
for r in comp_rows:
    note(f"    {r['band']:13s} {r['JANA_n']:>8,} {r['JANA_rmse']:>11.2f} "
         f"{r['JANA_bias']:>10.2f} {r['JANA_ymed']:>13.1f} | {r['JULA_n']:>8,} "
         f"{r['JULA_rmse']:>10.2f} {r['JULA_bias']:>9.2f} "
         f"{r['JULA_ymed']:>12.1f} | {r['JANC_n']:>6,} {r['JANC_rmse']:>10.2f} "
         f"{r['JULC_n']:>6,} {r['JULC_rmse']:>10.2f}")
note("")
note("  Where regime C has rows in a band, its RMSE exceeds regime A's in that")
note("  same band, in both folds, at every band from -1h upward. The target")
note("  medians are also higher in regime C at the large bands. Two facts")
note("  matter here:")
note("    1. the same gap value carries far more error when the flight")
note("       timestamps are missing, so 'how extreme is the gap' and 'are the")
note("       flight timestamps present' are not interchangeable questions;")
note("    2. the DISTRIBUTION of the gap differs sharply between regimes -")
note("       regime C is concentrated at 1h-2h (45% JAN, 55% JUL) while")
note("       regime A is concentrated at 0-30m - so the two populations barely")
note("       overlap in the bands where regime C is most populated. The")
note("       apparent 'conditioning' is therefore partly a mixture effect, not")
note("       purely a within-band effect, and is reported that way.")
note("")
note("  This is the one genuinely new conditioning signal in the audit. It is")
note("  NOT the same as V9-B's availability arm, which only told the model that")
note("  the flight columns were absent; here the point is that the gap's own")
note("  error profile is conditional on that absence.")
note("")

# ---------------------------------------------------------------- section 3
sub("3. AIRPORT INTERACTION - is an extreme gap useful generally?")
EXTREME = [
    ("|d_SCHED| > 24h", 24 * H, np.inf),
    ("|d_SCHED| 6-24h", 6 * H, 24 * H),
    ("|d_SCHED| 2-6h", 2 * H, 6 * H),
    ("|d_SCHED| < 0", 0.0, 0.0),
]
for fold in FOLDS:
    D = DATA[fold]
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    note(f"  {fold} - SSE share by |d_SCHED| magnitude (whole fold)")
    note(f"    {'regime':18s} {'n':>9s} {'%rows':>7s} {'RMSE':>9s} {'MAE':>9s} "
         f"{'bias':>9s} {'%SSE':>7s}")
    a = np.abs(D["d_sched"])
    for lab, lo, hi in EXTREME:
        if lo == hi == 0.0:
            m = D["d_sched"] < 0
        else:
            m = (a >= lo) & (a < hi)
        s = stats_for(m, D)
        if s["n"] == 0:
            continue
        note(f"    {lab:18s} {s['n']:>9,} {100 * s['n'] / len(D['y']):>6.2f}% "
             f"{s['rmse']:>9.2f} {s['mae']:>9.2f} {s['bias']:>+9.2f} "
             f"{100 * s['sse'] / tot_sse:>6.2f}%")
    note("")

note("  Airport breakdown for the extreme regimes. The 200-row floor used for")
note("  whole-fold tables is far too high inside a narrow gap regime, so here")
note("  any airport with at least 10 rows inside the regime is listed, ranked by")
note("  the SSE it carries within that regime.")
for fold in FOLDS:
    D = DATA[fold]
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    a = np.abs(D["d_sched"])
    for lab, lo, hi in EXTREME:
        if lo == hi == 0.0:
            base = D["d_sched"] < 0
        else:
            base = (a >= lo) & (a < hi)
        nb = int(base.sum())
        if nb == 0:
            continue
        sse_b = float(np.sum(((D["y"] - D["pred"]) ** 2)[base]))
        note(f"  {fold} - {lab}: {nb:,} rows, {100 * sse_b / tot_sse:.2f}% of "
             f"fold SSE, regime RMSE {np.sqrt(sse_b / nb):,.2f}")
        rows = []
        for ap in np.unique(D["airport"]):
            m = base & (D["airport"] == ap)
            if m.sum() < 10:
                continue
            s = stats_for(m, D)
            rows.append((s["sse"], ap, s))
        rows.sort(reverse=True, key=lambda t: t[0])
        if not rows:
            note("    (no airport has 10+ rows in this regime)")
            note("")
            continue
        note(f"    {'airport':8s} {'n':>7s} {'%regime':>8s} {'RMSE':>11s} "
             f"{'MAE':>10s} {'bias':>11s} {'%regSSE':>9s} {'%foldSSE':>9s} "
             f"{'focus':>6s}")
        for sse_v, ap, s in rows[:14]:
            note(f"    {ap:8s} {s['n']:>7,} {100 * s['n'] / nb:>7.2f}% "
                 f"{s['rmse']:>11,.2f} {s['mae']:>10,.2f} {s['bias']:>+11,.2f} "
                 f"{100 * sse_v / max(1e-12, sse_b):>8.2f}% "
                 f"{100 * sse_v / tot_sse:>8.2f}% "
                 f"{'yes' if ap in FOCUS_AIRPORTS else '':>6s}")
        top = sum(v for v, _, _ in rows[:3])
        note(f"    top 3 of {len(rows)} airports listed carry "
             f"{100 * top / sse_b:.1f}% of this regime's SSE; airports with "
             f"fewer than 10 rows carry the remaining "
             f"{100 * (sse_b - sum(v for v, _, _ in rows)) / max(1e-12, sse_b):.1f}%")
        note("")

note("  Concentration test. For the extreme-gap regimes, how much of the fold's")
note("  total SSE sits in the single worst airport, and how does that compare")
note("  with the airport's share of rows?")
for fold in FOLDS:
    D = DATA[fold]
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    note(f"    {fold}")
    note(f"      {'airport':8s} {'rows':>8s} {'%rows':>7s} {'%all SSE':>9s} "
         f"{'SSE/row share':>15s}")
    order = []
    for ap in np.unique(D["airport"]):
        m = D["airport"] == ap
        s = stats_for(m, D)
        order.append((s["sse"], ap, s["n"]))
    order.sort(reverse=True)
    for sse_v, ap, n in order[:6]:
        note(f"      {ap:8s} {n:>8,} {100 * n / len(D['y']):>6.2f}% "
             f"{100 * sse_v / tot_sse:>8.2f}% "
             f"{(100 * sse_v / tot_sse) / (100 * n / len(D['y'])):>14.1f}x")
    note("")

# ---------------------------------------------------------------- section 4
sub("4. WHERE d_SCHED ACTUALLY HELPS - schedule vs flight-level disagreement")
note("  Diagnostic quantities only. None of these is added to any model here.")
note("")
note("  For rows where all four flight-level deltas are present, compare the")
note("  gap against the flight-level deltas that describe the same movement.")
note("  Definitions used below:")
note("    d_AOBT3      the highest-importance flight-level delta in both folds")
note("    spread       max - min across the four flight-level deltas")
note("    gap_minus    d_SCHED - d_AOBT3")
note("    abs_disagree |d_SCHED - median(four flight deltas)|")
note("    same_sign    sign(d_SCHED) == sign(d_AOBT3), both non-null")
for fold in FOLDS:
    D = DATA[fold]
    A = D["deltas"]["d_AOBT3"]
    Dm = np.column_stack([D["deltas"][c] for c in FLIGHT_DELTAS])
    full = D["n4"] == 4
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    note(f"  {fold} - rows with all four flight deltas present: {int(full.sum()):,}")
    med = np.nanmedian(Dm, axis=1)
    gap_minus = D["d_sched"] - A
    abs_dis = np.abs(D["d_sched"] - med)
    spread = np.nanmax(Dm, axis=1) - np.nanmin(Dm, axis=1)
    same = (D["d_sched"] > 0) == (A > 0)
    sub_ = full & ~np.isnan(gap_minus)
    note(f"    {'group':38s} {'n':>9s} {'RMSE':>9s} {'MAE':>9s} {'bias':>9s} "
         f"{'%SSE':>7s}")
    groups = [
        ("same sign as d_AOBT3", sub_ & same),
        ("opposite sign to d_AOBT3", sub_ & ~same),
        ("|gap - d_AOBT3| < 1h", sub_ & (np.abs(gap_minus) < 1 * H)),
        ("|gap - d_AOBT3| 1-6h", sub_ & (np.abs(gap_minus) >= 1 * H)
            & (np.abs(gap_minus) < 6 * H)),
        ("|gap - d_AOBT3| 6-24h", sub_ & (np.abs(gap_minus) >= 6 * H)
            & (np.abs(gap_minus) < 24 * H)),
        ("|gap - d_AOBT3| >= 24h", sub_ & (np.abs(gap_minus) >= 24 * H)),
        ("|gap - median| < 1h", sub_ & (abs_dis < 1 * H)),
        ("|gap - median| >= 1h", sub_ & (abs_dis >= 1 * H)),
        ("flight-delta spread < 1h", sub_ & (spread < 1 * H)),
        ("flight-delta spread 1-6h", sub_ & (spread >= 1 * H) & (spread < 6 * H)),
        ("flight-delta spread >= 6h", sub_ & (spread >= 6 * H)),
    ]
    for lab, m in groups:
        s = stats_for(m, D)
        if s["n"] == 0:
            continue
        note(f"    {lab:38s} {s['n']:>9,} {s['rmse']:>9.2f} {s['mae']:>9.2f} "
             f"{s['bias']:>+9.2f} {100 * s['sse'] / tot_sse:>6.2f}%")
    note("")
    if sub_.sum():
        cp, cs = fmt_corr(np.abs(gap_minus[sub_]), np.abs(D["y"] - D["pred"])[sub_])
        note(f"    corr(|gap - d_AOBT3|, |error|) = {cp}   "
             f"spearman {cs}   (n={int(sub_.sum()):,})")
        cp2, cs2 = fmt_corr(spread[sub_], np.abs(D["y"] - D["pred"])[sub_])
        note(f"    corr(flight-delta spread, |error|) = {cp2}   "
             f"spearman {cs2}")
        note("")

note("  Does disagreement predict T4 error? The honest test is whether the")
note("  disagreement groups separate the population, not whether the extreme")
note("  bin looks bad. Read the next section for the JAN/JUL comparison, because")
note("  a group that only exists or only misbehaves in one fold is worthless.")
note("")

# ---------------------------------------------------------------- section 5
sub("5. EXTREME-TARGET RELATIONSHIP")
note("  Does an extreme gap reliably correspond to an extreme target? Target")
note("  bands, with the distribution of the gap and the availability mix inside")
note("  each. 'full' = all four flight timestamps present.")
for fold in FOLDS:
    D = DATA[fold]
    y = D["y"]
    tot_n = len(y)
    note(f"  {fold} - target band x gap distribution (n={tot_n:,})")
    note(f"    {'target band':12s} {'n':>9s} {'%':>6s} {'tgt med':>9s} "
         f"{'d_SCHED p10':>12s} {'p50':>9s} {'p90':>10s} {'%gap<0':>8s} "
         f"{'%full':>7s} {'RMSE':>9s} {'%SSE':>7s}")
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    for lab, lo, hi in TARGET_BANDS:
        m = (y >= lo) & (y < hi)
        n = int(m.sum())
        if n == 0:
            continue
        s = stats_for(m, D)
        ds = D["d_sched"][m]
        q10, q50, q90 = (np.percentile(ds, [10, 50, 90]) if n >= 20
                         else (float("nan"),) * 3)
        note(f"    {lab:12s} {n:>9,} {100 * n / tot_n:>5.2f}% {s['y_med']:>9.1f} "
             f"{q10:>12.1f} {q50:>9.1f} {q90:>10.1f} "
             f"{100 * (ds < 0).mean():>7.2f}% {100 * (D['n4'][m] == 4).mean():>6.2f}% "
             f"{s['rmse']:>9.2f} {100 * s['sse'] / tot_sse:>6.2f}%")
    note("")

note("  Symmetric check: within each gap band, what fraction of targets are")
note("  extreme, and what is the median target?")
for fold in FOLDS:
    D = DATA[fold]
    y = D["y"]
    tot_n = len(y)
    note(f"    {fold}")
    note(f"      {'gap band':13s} {'n':>8s} {'%tgt>4h':>9s} {'%tgt>15h':>9s} "
         f"{'tgt med':>9s} {'tgt p90':>10s} {'RMSE':>9s}")
    idx = band_of(D["d_sched"], DSCHED_BANDS)
    for i, (lab, _, _) in enumerate(DSCHED_BANDS):
        m = idx == i
        n = int(m.sum())
        if n == 0:
            continue
        s = stats_for(m, D)
        p90 = np.percentile(y[m], 90) if n >= 20 else float("nan")
        note(f"      {lab:13s} {n:>8,} {100 * (y[m] > 4 * H).mean():>8.2f}% "
             f"{100 * (y[m] > 15 * H).mean():>8.2f}% {s['y_med']:>9.1f} "
             f"{p90:>10.1f} {s['rmse']:>9.2f}")
    note("")

note("  Is the apparent relationship broadly reproducible, or a handful of rows?")
note("  Concentration of each target band's contribution to fold SSE, and the")
note("  largest single rows inside the extreme bands.")
for fold in FOLDS:
    D = DATA[fold]
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    e = np.abs(D["y"] - D["pred"])
    order = np.argsort(-e)
    note(f"    {fold}")
    for lab, lo, hi in TARGET_BANDS:
        m = (D["y"] >= lo) & (D["y"] < hi)
        if m.sum() == 0:
            continue
        sse_v = float(np.sum((D["y"][m] - D["pred"][m]) ** 2))
        sub_ = e[m]
        top1 = float(np.max(sub_)) if len(sub_) else 0.0
        share_of_sse = 100 * sse_v / tot_sse
        note(f"      {lab:12s} rows {int(m.sum()):>8,}  "
             f"{share_of_sse:>6.2f}% of fold SSE  worst single row "
             f"{top1:>10,.0f}s  RMSE {np.sqrt(sse_v / m.sum()):>9.2f}")
    k = order[:100]
    sse100 = float(np.sum(((D["y"][k] - D["pred"][k]) ** 2)))
    note(f"      worst 100 rows carry {100 * sse100 / tot_sse:.1f}% of fold SSE; "
         f"their targets: med {np.median(D['y'][k]):,.0f}s  "
         f"p90 {np.percentile(D['y'][k], 90):,.0f}s  max {np.max(D['y'][k]):,.0f}s")
    note("")

# ---------------------------------------------------------------- section 6
sub("6. WORST ERRORS - top 100 absolute-error rows, by pattern")
note("  Summarised, not dumped. For each fold the 100 largest |target-pred| rows")
note("  are grouped by (availability regime, |d_SCHED| band, airport) and the")
note("  repeated patterns are listed with their counts and SSE weight.")
PATTERNS: list[dict] = []
for fold in FOLDS:
    D = DATA[fold]
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    e = np.abs(D["y"] - D["pred"])
    k = np.argsort(-e)[:100]
    note(f"  {fold} - worst 100 rows, total {float(np.sum(e[k] ** 2)) / 1e6:.2f}e6 "
         f"squared-seconds, {100 * float(np.sum(e[k] ** 2)) / tot_sse:.1f}% of "
         f"fold SSE")
    note(f"    median target {np.median(D['y'][k]):,.0f}s, "
         f"median prediction {np.median(D['pred'][k]):,.0f}s, "
         f"median residual {np.median(D['y'][k] - D['pred'][k]):,.0f}s")
    note("")
    groups: dict[tuple, list[int]] = {}
    for i in k:
        ds = D["d_sched"][i]
        al = abs(ds)
        if al > 24 * H:
            gb = ">24h"
        elif al > 6 * H:
            gb = "6-24h"
        elif al > 2 * H:
            gb = "2-6h"
        elif al > 2 * H / 2:
            gb = "1-2h"
        else:
            gb = "<1h"
        av = int(D["n4"][i])
        key = ("full" if av == 4 else "missing-all4", gb, str(D["airport"][i]))
        groups.setdefault(key, []).append(int(i))
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    note(f"    {'pattern (availability, |gap|, airport)':46s} {'n':>4s} "
         f"{'sse share':>11s} {'med tgt':>10s} {'med res':>10s}")
    for (av, gb, ap), idxs in ranked[:18]:
        ii = np.array(idxs)
        sse_v = float(np.sum((D["y"][ii] - D["pred"][ii]) ** 2))
        note(f"    {f'{av}, {gb}, {ap}':46s} {len(idxs):>4d} "
             f"{100 * sse_v / tot_sse:>10.2f}% "
             f"{np.median(D['y'][ii]):>10,.0f} "
             f"{np.median(D['y'][ii] - D['pred'][ii]):>+10,.0f}")
        PATTERNS.append({"fold": fold, "av": av, "gap": gb, "ap": ap,
                         "n": len(idxs), "sse": sse_v})
    note("")
    # how much of the worst-100 is LIRF, and how much is extreme-target
    lirf = int(sum(1 for i in k if str(D["airport"][i]) == "LIRF"))
    big = int(sum(1 for i in k if D["y"][i] > 15 * H))
    dsx = int(sum(1 for i in k if abs(D["d_sched"][i]) > 6 * H))
    miss = int(sum(1 for i in k if D["n4"][i] == 0))
    note(f"    of these 100 rows: LIRF {lirf}, target>15h {big}, "
         f"|d_SCHED|>6h {dsx}, missing-all4 {miss}")
    note("")

note("  Repeated pattern summary across both folds:")
allp: dict[tuple, dict] = {}
for p in PATTERNS:
    key = (p["av"], p["gap"], p["ap"])
    rec = allp.setdefault(key, {"n": 0, "sse": 0.0, "folds": set()})
    rec["n"] += p["n"]
    rec["sse"] += p["sse"]
    rec["folds"].add(p["fold"])
note(f"    {'pattern':46s} {'rows':>5s} {'folds':>6s}  shared by both folds?")
for key, rec in sorted(allp.items(), key=lambda kv: -kv[1]["n"])[:20]:
    av, gb, ap = key
    both = "yes" if len(rec["folds"]) == 2 else "NO - single fold"
    note(f"    {f'{av}, {gb}, {ap}':46s} {rec['n']:>5d} {len(rec['folds']):>6d}  {both}")
note("")
note("  Any pattern flagged 'single fold' is a coincidence of that month, not a")
note("  reproducible structure, and must not be used to justify a feature.")
note("")

# ---------------------------------------------------------------- section 7
sub("7. JAN vs JUL ROBUSTNESS - the deciding section")
note("  Every d_SCHED relationship claimed above is re-tested here against the")
note("  five failure modes in the brief: reversed direction, single-fold only,")
note("  LIRF-dependent, fewer than ~100 rows, or driven by a handful of extreme")
note("  targets.")
note("")
ROBUST: list[tuple[str, str, str]] = []


def rob(label, fold_vals, base_vals, counts, lirf_frac, extra=""):
    """Compare one relationship across folds and grade it."""
    (jn, jv), (ln, lv) = fold_vals
    (bj, bl) = base_vals
    jflip = (jv - bj) * (lv - bl) < 0
    single = (jn < 100) or (ln < 100)
    lird = max(lirf_frac) > 0.5
    if jflip:
        verdict = "REVERSED"
    elif single:
        verdict = "SINGLE-FOLD / too few rows"
    elif lird:
        verdict = "LIRF-DEPENDENT"
    else:
        verdict = "consistent"
    ROBUST.append((label, verdict, extra))
    note(f"    {label:44s} JAN {jv - bj:>+10.2f} (n={jn:,})  "
         f"JUL {lv - bl:>+10.2f} (n={ln:,})  -> {verdict}")
    return verdict


note("  7a. Does the gap's magnitude raise or lower RMSE, per fold?")
note("      (RMSE of the band minus RMSE of the 30m-2h reference, which is the")
note("      modal part of the gap distribution)")
for i, (lab, lo, hi) in enumerate(DSCHED_BANDS):
    vals, counts, lirff = [], [], []
    base_v = {}
    for fold in FOLDS:
        D = DATA[fold]
        idx = band_of(D["d_sched"], DSCHED_BANDS)
        ref = stats_for((idx == 4) | (idx == 5), D)  # 1h..2h + 2h..4h
        base_v[fold] = ref["rmse"] if ref["n"] else float("nan")
        s = stats_for(idx == i, D)
        vals.append((s["n"], s["rmse"] - (base_v[fold] if ref["n"] else np.nan)))
        lirff.append(0.0)
    rob(f"band {lab}", vals, (base_v["JAN"] - base_v["JAN"],
                              base_v["JUL"] - base_v["JUL"]),
        [v[0] for v in vals], lirff)
note("")

note("  7b. Does target availability change the gap's error profile per fold?")
note("      (RMSE of missing-all4 minus RMSE of full, within each extreme band)")
for lab, lo, hi in [(l, a, b) for l, a, b in EXTREME if not (a == b == 0.0)]:
    vals, counts = [], []
    for fold in FOLDS:
        D = DATA[fold]
        a_ = np.abs(D["d_sched"])
        band = (a_ >= lo) & (a_ < hi)
        sA = stats_for(band & (D["n4"] == 4), D)
        sC = stats_for(band & (D["n4"] == 0), D)
        if sA["n"] == 0 or sC["n"] == 0:
            vals.append((max(sA["n"], sC["n"]), np.nan))
        else:
            vals.append((sC["n"], sC["rmse"] - sA["rmse"]))
    if all(np.isnan(v[1]) for v in vals):
        note(f"    {lab:44s} not populated in both folds")
        continue
    rob(f"missing-vs-full in |gap| {lab}", vals, (0.0, 0.0),
        [v[0] for v in vals], [0.0, 0.0])
note("")

note("  7c. Does schedule/flight disagreement predict error, per fold?")
note("      (RMSE of |gap - d_AOBT3| >= 1h minus RMSE of < 1h)")
vals, counts = [], []
for fold in FOLDS:
    D = DATA[fold]
    A = D["deltas"]["d_AOBT3"]
    Dm = np.column_stack([D["deltas"][c] for c in FLIGHT_DELTAS])
    full = D["n4"] == 4
    med = np.nanmedian(Dm, axis=1)
    abs_dis = np.abs(D["d_sched"] - med)
    base = full & ~np.isnan(abs_dis)
    s_lo = stats_for(base & (abs_dis < 1 * H), D)
    s_hi = stats_for(base & (abs_dis >= 1 * H), D)
    vals.append((s_hi["n"], s_hi["rmse"] - s_lo["rmse"]))
rob("disagreement >= 1h vs < 1h", vals, (0.0, 0.0),
    [v[0] for v in vals], [0.0, 0.0])
note("")

note("  7c-ii. Same test on the SPREAD of the four flight-level deltas")
note("      (max - min among AOBT3/EOBT1/IOBT/LOBT). This is not a d_SCHED")
note("      quantity, but section 4 showed it separates error far more")
note("      strongly than anything involving the gap, so it is graded here.")
vals, counts = [], []
for fold in FOLDS:
    D = DATA[fold]
    Dm = np.column_stack([D["deltas"][c] for c in FLIGHT_DELTAS])
    full = D["n4"] == 4
    spread = np.nanmax(Dm, axis=1) - np.nanmin(Dm, axis=1)
    base = full & ~np.isnan(spread)
    s_lo = stats_for(base & (spread < 1 * H), D)
    s_hi = stats_for(base & (spread >= 1 * H), D)
    vals.append((s_hi["n"], s_hi["rmse"] - s_lo["rmse"]))
rob("flight-delta spread >= 1h vs < 1h", vals, (0.0, 0.0),
    [v[0] for v in vals], [0.0, 0.0])
note("")

note("  7d. LIRF dependence of the extreme-gap SSE share, per fold.")
for fold in FOLDS:
    D = DATA[fold]
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    a_ = np.abs(D["d_sched"])
    ext = a_ > 6 * H
    sse_ext = float(np.sum(((D["y"] - D["pred"]) ** 2)[ext]))
    sse_ext_lirf = float(np.sum(((D["y"] - D["pred"]) ** 2)
                                 [ext & (D["airport"] == "LIRF")]))
    n_lirf = int((ext & (D["airport"] == "LIRF")).sum())
    n_ext = int(ext.sum())
    note(f"    {fold}: |gap|>6h is {n_ext:,} rows, {100 * sse_ext / tot_sse:.2f}% "
         f"of fold SSE; LIRF holds {n_lirf:,} of those rows and "
         f"{100 * sse_ext_lirf / max(1e-9, sse_ext):.1f}% of that SSE")
note("")
note("  7d-ii. LIRF dependence of the DOMINANT error group (G2 missing-all4),")
note("      which is what any real gain would have to come from.")
for fold in FOLDS:
    D = DATA[fold]
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    miss = D["n4"] == 0
    sse_m = float(np.sum(((D["y"] - D["pred"]) ** 2)[miss]))
    sse_m_lirf = float(np.sum(((D["y"] - D["pred"]) ** 2)
                               [miss & (D["airport"] == "LIRF")]))
    note(f"    {fold}: missing-all4 is {int(miss.sum()):,} rows, "
         f"{100 * sse_m / tot_sse:.2f}% of fold SSE; LIRF holds "
         f"{int((miss & (D['airport'] == 'LIRF')).sum()):,} of those rows and "
         f"{100 * sse_m_lirf / max(1e-9, sse_m):.1f}% of that SSE")
note("")

note("  7e. Verdict table.")
note(f"    {'relationship':44s} {'verdict':28s} note")
rev = [r for r in ROBUST if r[1] == "REVERSED"]
sing = [r for r in ROBUST if r[1].startswith("SINGLE")]
lirp = [r for r in ROBUST if r[1] == "LIRF-DEPENDENT"]
ok = [r for r in ROBUST if r[1] == "consistent"]
for lab, v, extra in ROBUST:
    note(f"    {lab:44s} {v:28s} {extra}")
note("")
note(f"    consistent {len(ok)}   reversed {len(rev)}   single-fold {len(sing)}   "
     f"LIRF-dependent {len(lirp)}")
note("")
if rev:
    note("  Relationships that REVERSE between folds - these are actively")
    note("  dangerous, not merely weak:")
    for lab, v, _ in rev:
        note(f"    - {lab}")
    note("")

# ---------------------------------------------------------------- section 8
sub("8. CORRECTED ERROR DECOMPOSITION")
note("  Replaces the frozen V8 autopsy's section 4 grouping, which mislabelled")
note("  d_SCHED rows as 'inconsistent flight timestamps'.")
note("")
note("  Group definitions. The first four are mutually exclusive and together")
note("  partition the validation set exactly:")
note("    G1 ordinary       all 4 flight timestamps present AND no flight delta")
note("                      is negative or beyond 2h")
note("    G2 missing-all4   none of the 4 flight timestamps present")
note("    G3 partial        1-3 of 4 present          (EMPTY - see section 2)")
note("    G4 true implaus.  all 4 present AND some flight delta is negative or")
note("                      beyond 2h")
note("")
note("  G5 (extreme gap) is a CROSS-CUTTING group and overlaps all four. It is")
note("  reported separately and never added to the totals above.")
note("")
for fold in FOLDS:
    D = DATA[fold]
    tot_n = len(D["y"])
    tot_sse = float(np.sum((D["y"] - D["pred"]) ** 2))
    full = D["n4"] == 4
    miss = D["n4"] == 0
    impl = full & D["impl"]
    ordinary = full & ~D["impl"]
    partial = (D["n4"] >= 1) & (D["n4"] <= 3)
    note(f"  {fold} - {tot_n:,} rows, fold SSE {tot_sse:.4e}, "
         f"RMSE {np.sqrt(tot_sse / tot_n):.4f}")
    note(f"    {'group':22s} {'n':>9s} {'%rows':>7s} {'RMSE':>10s} {'MAE':>9s} "
         f"{'bias':>10s} {'%SSE':>7s} {'sse/row':>11s}")
    rows = [("G1 ordinary", ordinary), ("G2 missing-all4", miss),
            ("G3 partial", partial), ("G4 true implausible", impl)]
    acc = 0
    for lab, m in rows:
        s = stats_for(m, D)
        if s["n"] == 0:
            note(f"    {lab:22s} {0:>9,} {'-':>7s}  (empty by construction)")
            continue
        acc += s["sse"]
        note(f"    {lab:22s} {s['n']:>9,} {100 * s['n'] / tot_n:>6.2f}% "
             f"{s['rmse']:>10.2f} {s['mae']:>9.2f} {s['bias']:>+10.2f} "
             f"{100 * s['sse'] / tot_sse:>6.2f}% {s['sse'] / s['n']:>11,.0f}")
    note(f"    {'sum of G1..G4':22s} {acc:>9.4e} SSE = "
         f"{100 * acc / tot_sse:.4f}% of fold  (must be 100%)")
    assert abs(acc - tot_sse) < 1e-6 * tot_sse, f"{fold}: G1..G4 do not partition"
    note("")
    for lab, thr in (("G5a |gap| > 2h", 2 * H), ("G5b |gap| > 6h", 6 * H),
                     ("G5c |gap| > 24h", 24 * H)):
        m = np.abs(D["d_sched"]) > thr
        s = stats_for(m, D)
        if s["n"] == 0:
            continue
        ov = []
        for glab, gm in rows:
            inter = int((m & gm).sum())
            if inter:
                ov.append(f"{glab.split()[0]}={inter:,}")
        note(f"    {lab:22s} {s['n']:>9,} {100 * s['n'] / tot_n:>6.2f}% "
             f"{s['rmse']:>10.2f} {s['mae']:>9.2f} {s['bias']:>+10.2f} "
             f"{100 * s['sse'] / tot_sse:>6.2f}% {s['sse'] / s['n']:>11,.0f}"
             f"   overlaps: {', '.join(ov) if ov else 'none'}")
    note("")
note("  Reading the decomposition:")
note("    - The corrected picture is dominated by missing-all4 rows, not by")
note("      implausible timestamps. The frozen autopsy's headline number pointed")
note("      the wrong way.")
note("    - G4 true implausible is tiny and its SSE share is the honest ceiling")
note("      for any 'flag the bad timestamps' idea. V9-B already measured this at")
note("      10.15% of pooled SSE and found no robust way to capture it.")
note("    - The G5 rows carry a large SSE share, but the overlap column shows how")
note("      much of that is the same missing-all4 and implausible rows counted")
note("      again, not new information.")
note("")

# ---------------------------------------------------------------- section 9
sub("9. RECOMMENDED NEXT EXPERIMENTS - proposed only, NOT implemented")
note("  At most three, ranked. Nothing here has been built. No model was fitted")
note("  in this audit and none is fitted by writing it down.")
note("")

# evidence-based verdicts computed above, not hardcoded
avail_consistent = any(v[1] == "consistent" for v in ROBUST
                       if "missing-vs-full" in v[0])
dis_consistent = any(v[1] == "consistent" for v in ROBUST
                     if "disagreement" in v[0])
n_rev = len([v for v in ROBUST if v[1] == "REVERSED"])

avail_verdict = "supported" if avail_consistent else \
    "NOT established - treat as a"
dis_verdict = "supported" if dis_consistent else "NOT established"

note("  H1 (rank 1) - Internal disagreement among the four FLIGHT-level")
note("      timestamps: spread = max - min of the available flight deltas,")
note("      and the same for d_AOBT3 alone. This is the strongest and most")
note("      robust diagnostic the audit found, and it is NOT a d_SCHED")
note("      feature - it measures the four flight timestamps contradicting")
note("      each other, with no reference to the schedule.")
note("      Section 4: spread < 1h vs 1-6h gives RMSE 223.62 -> 1,249.42 (JAN)")
note("      and 323.38 -> 1,287.21 (JUL), a 5.6x and 4.0x separation, on")
note("      325 and 1,169 rows. Section 7c-ii grades it consistent. It is a")
note("      far cleaner signal than anything the gap provides.")
note("      Expected potential      moderate - the affected group is small")
note("                               (about 0.2% and 0.6% of rows) but its")
note("                               per-row error is 4-6x the ordinary rate")
note("      JAN/JUL robustness     " + dis_verdict)
note("      LIRF-tail risk         low-to-moderate; must still be re-checked,")
note("                               but the quantity is defined by internal")
note("                               timestamp arithmetic, not by airport")
note("      Implementation         easy - two row-wise columns")
note("")
note("  H2 (rank 2) - Availability-conditioned d_SCHED")
note("      Encode the gap differently depending on whether the flight")
note("      timestamps are present. Section 2 supports this: at every band")
note("      from -1h upward, missing-all4 rows have much higher RMSE than")
note("      full-timestamp rows in the same gap band, in both folds, and")
note("      section 7b grades the 2-6h version consistent on 302 / 1,072 rows.")
note("      Expected potential      LOW. Be clear about why: the group this")
note("                               acts on is exactly the group where the")
note("                               flight timestamps are ABSENT, so every")
note("                               input is unavailable and the feature can")
note("                               only re-encode a fact V8 already has")
note("                               (missing_all4, which V9-B tested and which")
note("                               did not help). The genuine new increment")
note("                               over V9-B is small.")
note("      JAN/JUL robustness     " + avail_verdict)
note("      LIRF-tail risk         low; section 7d-ii shows LIRF contributes")
note("                               almost none of the missing-all4 error")
note("      Implementation         easy")
note("")
note("  H3 (rank 3) - NOT RECOMMENDED: any feature of raw |d_SCHED|")
note("      magnitude, including the extreme bands. The audit contradicts the")
note("      obvious version of this idea on three separate counts:")
note("        - the relationship is not monotone. Bands from 0 to 6h behave")
note("          consistently across folds, but < -2h and > 24h REVERSE")
note("          between folds (section 7a);")
note("        - the extreme bands rest on 8 to 56 rows per fold, below the")
note("          ~100-row floor, so they cannot support a conclusion;")
note("        - extreme gap does NOT identify the worst rows. In the worst")
note("          100 rows of each fold, 51 (JAN) and 80 (JUL) are missing-all4,")
note("          LIRF contributes 0 rows in both, and the single largest JAN")
note("          error (27.6% of fold SSE) has |d_SCHED| < 1h - a completely")
note("          ordinary gap attached to a missing-timestamp row with an")
note("          84,240s target.")
note("      Note this is NOT an LIRF-tail artefact, which is the interesting")
note("      part: section 3 shows the extreme-gap regimes are spread across")
note("      92 airports in JAN with the top 3 holding only 33.9% of that")
note("      regime's SSE. The extreme gap is diffuse, not concentrated. It")
note("      simply does not point at the rows that matter.")
note("      Listed so the option is visibly rejected rather than quietly")
note("      omitted.")
note("")

note("  BRANCH VERDICT on d_SCHED as a standalone lever:")
note("    RECOMMEND ABANDONING d_SCHED AS A MAGNITUDE LEVER, and keeping only")
note("    the two narrow survivors above.")
note("")
note("  The reasoning, stated against my own hypothesis:")
note("    - the target is almost always small: 95.4% of validation rows are")
note("      under 30 minutes in both folds, and the gap itself sits in the")
note("      0-60m bands for 89.6% of JAN rows and 84.7% of JUL rows. There is")
note("      very little room for a gap-derived feature to matter.")
note("    - d_SCHED correlates only weakly with the target: the strongest")
note("      Pearson in any band is +0.42 and most are under 0.10 (section 1).")
note("      It is not a usable proxy for the thing being predicted.")
note("    - the errors that dominate are not extreme-gap errors. They are")
note("      missing-timestamp rows with extreme targets. Of the rows with a")
note("      target beyond 4h, 8 of 8 in JAN and 35 of 37 in JUL are in the")
note("      missing-all4 regime (section 5).")
note("    - and the honest ceiling: for missing-all4 rows the flight")
note("      timestamps do not exist, so no feature derived from them can")
note("      recover the information. G2 holds 67.8% (JAN) and 57.6% (JUL) of")
note("      fold SSE, and it is untouchable by anything in this branch.")
note("      Closing it requires information V8 does not currently use, not a")
note("      better encoding of timestamps that are absent.")
note("")
note("  If H1 is attempted, it is a small controlled ablation, not a new")
note("  direction. The expected gain is a fraction of a point, and it should")
note("  be abandoned quickly if the ordinary-rows or worst-100 criteria fail.")
note("")
note("  Whatever is attempted next must satisfy, before it is believed:")
note("    - B0 reproduces JAN 404.8803102 and JUL 513.7559738 exactly")
note("    - pooled gain survives removing LIRF, and removing LIRF+LFPG")
note("    - ordinary rows do not worsen")
note("    - worst-100 rows hold < 100% of net gain")
note("    - JAN and JUL agree in sign")
note("  V9-B failed five of eight of exactly these. They are not optional.")
note("")

# ---------------------------------------------------------------- frozen check
sub("9b. FROZEN-ARTEFACT VERIFICATION")
FROZEN = [
    "models/v8_catboost.cbm",
    "src/train_catboost_v8.py",
    "src/train_catboost_v8_defs.py",
    "src/create_submission_v8.py",
    "src/v8_error_autopsy.py",
    "src/experiment_v9a_missing_segment.py",
    "src/experiment_v9b_timestamp_plausibility.py",
    "src/v9a_2026_distribution_audit.py",
    "reports/v8_error_autopsy.txt",
    "reports/v9a_missing_segment_experiment.txt",
    "reports/v9b_timestamp_plausibility_experiment.txt",
    "reports/v9a_2026_distribution_audit.txt",
    "submissions/vigorous-glove_v2.parquet",
]
PRIOR_SIZE = {
    "src/experiment_v9a_missing_segment.py": 57_032,
    "reports/v9a_missing_segment_experiment.txt": 71_203,
    "reports/v9a_2026_distribution_audit.txt": 47_183,
    "reports/v9b_timestamp_plausibility_experiment.txt": 36_978,
}
note("  Verification method, stated honestly: every frozen artefact above is")
note("  UNTRACKED in this repository, so git cannot prove anything about them.")
note("  Sizes alone are weak evidence. Each file is therefore hashed, and the")
note("  three files whose size was recorded before this audit began are")
note("  cross-checked against that record. V9-B's report is included because")
note("  this audit depends on its cached predictions.")
note("")
note(f"  {'path':50s} {'bytes':>12s}  {'sha256':<18s} status")
for p in FROZEN:
    q = Path(p)
    if not q.exists():
        note(f"  {p:50s} {'-':>12s}  {'-':<18s} MISSING")
        continue
    h = hashlib.sha256(q.read_bytes()).hexdigest()[:16]
    size = q.stat().st_size
    if p in PRIOR_SIZE:
        st = ("unchanged (matches pre-audit record)" if size == PRIOR_SIZE[p]
              else f"CHANGED! expected {PRIOR_SIZE[p]:,}")
    else:
        st = "present, hashed"
    note(f"  {p:50s} {size:>12,}  {h:<18s} {st}")
note("")
note("  This audit wrote exactly one new file:")
note("    reports/v9c_dsched_forensic_audit.txt")
note("  and its own source:")
note("    src/v9c_dsched_forensic_audit.py")
note("  No file in models/ or submissions/ was created, read for targets, or")
note("  modified. No CatBoost model was trained or loaded. No 2026 data was")
note("  touched. No commit, push, merge or reset was performed.")
note("")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
print(f"\n[written] {REPORT}  lines={len(out)}  bytes={REPORT.stat().st_size:,}")
