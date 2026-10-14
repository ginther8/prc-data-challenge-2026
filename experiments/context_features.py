"""Airport-context features for every departure (2025 training + 2026 ranking).

All features use only OTHER movements that happened BEFORE the departure's
takeoff time T (strictly earlier), so they are available at prediction time.
Arrival taxi-in times are supplied for ranking arrivals too (only DEPARTURE
targets are hidden). No departure target is ever used.

  ctx_arr_taxiin_30 / _60  mean taxi-in (clipped to 0..3600 s) of arrivals at
                           the same airport that LANDED in [T-30/60 min, T)
  ctx_arr_n_60             number of those arrivals
  ctx_arr_taxiin_rwy_60    same, arrivals on the departure's runway only
  ctx_stand_since_arr      seconds since the last arrival reached the same
                           stand (in-block time of that arrival, before T)
  ctx_dep_adsb_60          mean ADS-B-observed taxi duration (adsb_est_move,
                           i.e. T_other - first movement) of OTHER departures
                           that took off in [T-60 min, T) at the same airport
  ctx_dep_adsb_n_60        number of those departures with a usable track
  ctx_dep_aobt_60          mean (MVT - AOBT_3) of departures that took off
                           in [T-60 min, T)

(Arrival-context idea: team outgoing-boat, github.com/EnioAguiar/prc-taxiout-2026;
independent re-implementation.)

Run from the project root (a few minutes):
    python -u experiments/context_features.py
Output: data/processed/context_features.parquet
"""

import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = PROC / "context_features.parquet"
ADSB = PROC / "adsb_features.parquet"
MID = "MVT_ID_mvt"
COLS = [MID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
        "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def secs(s):
    return ((s - EPOCH).dt.total_seconds()).to_numpy(np.float64)


def window_mean(t_src, v_src, t_q, w):
    """mean/count of v_src with t_src in [t_q - w, t_q) (t_src sorted)."""
    cs = np.concatenate([[0.0], np.cumsum(v_src)])
    hi = np.searchsorted(t_src, t_q, "left")
    lo = np.searchsorted(t_src, t_q - w, "left")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return m, n


def main():
    t0 = time.time()
    files = sorted(glob.glob(str(RAW / "training_*.parquet"))) + [str(RAW / "ranking.parquet")]
    if len(files) != 13:
        print(f"FAIL: expected 12 training files + ranking, found {len(files)}")
        sys.exit(1)
    d = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in files], ignore_index=True)
    if d[MID].duplicated().any():
        print("FAIL: duplicate MVT_ID_mvt")
        sys.exit(1)
    d["T"] = secs(d["MVT_TIME_UTC_mvt"])
    d["apt"] = np.where(d["PHASE_mvt"] == "DEP", d["ADEP_mvt"].astype(str), d["ADES_mvt"].astype(str))
    arr = d[d["PHASE_mvt"] == "ARR"].copy()
    dep = d[d["PHASE_mvt"] == "DEP"].copy()
    arr["taxiin"] = arr["TAXITIME_SEC_mvt"].astype(float).clip(0, 3600)
    arr["inblock"] = arr["T"] + arr["TAXITIME_SEC_mvt"].astype(float)
    dep["aobt_d"] = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    if ADSB.exists():
        a = pd.read_parquet(ADSB, columns=[MID, "adsb_ok", "adsb_est_move"])
        dep = dep.merge(a, on=MID, how="left")
    else:
        print(f"WARNING: {ADSB} not found - ADS-B neighbour features left empty")
        dep["adsb_ok"] = np.nan
        dep["adsb_est_move"] = np.nan
    print(f"movements: {len(d):,}  arrivals {len(arr):,}  departures {len(dep):,}", flush=True)

    res = []
    for apt, g in dep.groupby("apt"):
        g = g.sort_values("T")
        T = g["T"].to_numpy()
        out = pd.DataFrame({MID: g[MID].to_numpy()})
        a = arr[arr["apt"] == apt].sort_values("T")
        at, av = a["T"].to_numpy(), a["taxiin"].fillna(0).to_numpy()
        ok = a["taxiin"].notna().to_numpy()
        at, av = at[ok], av[ok]
        out["ctx_arr_taxiin_30"], _ = window_mean(at, av, T, 1800)
        out["ctx_arr_taxiin_60"], out["ctx_arr_n_60"] = window_mean(at, av, T, 3600)
        # same runway
        rw = np.full(len(g), np.nan)
        a_ok = a[ok]
        for r, gi in g.groupby(g["RUNWAY_mvt"].astype(str)).indices.items():
            ar = a_ok[a_ok["RUNWAY_mvt"].astype(str) == r]
            if len(ar):
                rw[gi], _ = window_mean(ar["T"].to_numpy(), ar["taxiin"].to_numpy(), T[gi], 3600)
        out["ctx_arr_taxiin_rwy_60"] = rw
        # last arrival at the same stand (in-block before T)
        since = np.full(len(g), np.nan)
        ab = a[a["STAND_mvt"].notna() & a["inblock"].notna()]
        stands = g["STAND_mvt"].astype(str).to_numpy()
        for st, gi in pd.Series(np.arange(len(g))).groupby(stands).indices.items():
            s = np.sort(ab.loc[ab["STAND_mvt"].astype(str) == st, "inblock"].to_numpy())
            if len(s):
                k = np.searchsorted(s, T[gi], "left") - 1
                since[gi] = np.where(k >= 0, T[gi] - s[np.maximum(k, 0)], np.nan)
        out["ctx_stand_since_arr"] = since
        # other departures (strictly earlier takeoff, so the row itself is excluded)
        m_ad = (g["adsb_ok"] == 1).to_numpy() & np.isfinite(g["adsb_est_move"].to_numpy(dtype=float))
        out["ctx_dep_adsb_60"], out["ctx_dep_adsb_n_60"] = window_mean(
            T[m_ad], np.clip(g["adsb_est_move"].to_numpy(dtype=float)[m_ad], 0, 3600), T, 3600)
        m_ao = np.isfinite(g["aobt_d"].to_numpy(dtype=float))
        out["ctx_dep_aobt_60"], _ = window_mean(
            T[m_ao], np.clip(g["aobt_d"].to_numpy(dtype=float)[m_ao], 0, 3600), T, 3600)
        res.append(out)
        print(f"  {apt}: {len(g):,} departures", flush=True)

    f = pd.concat(res, ignore_index=True)
    if len(f) != len(dep) or f[MID].duplicated().any():
        print("FAIL: row count / duplicate check")
        sys.exit(1)
    for c in f.columns[1:]:
        f[c] = f[c].astype("float32")
    f.to_parquet(OUT, index=False)
    print("coverage (share non-missing):")
    for c in f.columns[1:]:
        print(f"  {c:<24}{f[c].notna().mean():7.1%}")
    print(f"wrote {OUT}  rows={len(f):,}  ({time.time() - t0:.0f}s)")
    print("DONE")


if __name__ == "__main__":
    main()
