"""Gate-retention features at each departure's AOBT_3 time t (other flights only).

  gr_overdue        departures at the same airport with EOBT_1 <= t < AOBT_3 (past EOBT, still at gate)
  gr_overdue_rwy    same, same runway
  gr_active_rwy     same-runway departures with AOBT_3 <= t < MVT (off-block, not yet airborne), self excluded
  gr_aobt_sched_60  mean (AOBT_3 - SCHED) of departures with AOBT_3 in [t-60 min, t)
  gr_cadence_60     departures with MVT in [t-60 min, t)
Rows without AOBT_3 get NaN. Idea: team outgoing-boat (github.com/EnioAguiar/prc-taxiout-2026),
independent re-implementation. No departure target is used.
    python -u experiments\\gate_features.py      ->  data/processed/gate_features.parquet
"""

import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "gate_features.parquet"
MID = "MVT_ID_mvt"
COLS = [MID, "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "EOBT_1_flt", "AOBT_3_flt"]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def secs(s):
    return (s - EPOCH).dt.total_seconds().to_numpy(np.float64)


def open_count(start, end, t):
    """number of intervals [start, end) containing t."""
    ok = np.isfinite(start) & np.isfinite(end) & (end > start)
    s, e = np.sort(start[ok]), np.sort(end[ok])
    return (np.searchsorted(s, t, "right") - np.searchsorted(e, t, "right")).astype(float)


def win_mean(ts, v, t, w):
    o = np.argsort(ts)
    ts, v = ts[o], v[o]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    hi, lo = np.searchsorted(ts, t, "left"), np.searchsorted(ts, t - w, "left")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan), n.astype(float)


def main():
    t0 = time.time()
    files = sorted(glob.glob(str(RAW / "training_*.parquet"))) + [str(RAW / "ranking.parquet")]
    d = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in files], ignore_index=True)
    d = d[d["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    d["mvt"], d["sch"] = secs(d["MVT_TIME_UTC_mvt"]), secs(d["SCHED_TIME_UTC_mvt"])
    d["eobt"], d["aobt"] = secs(d["EOBT_1_flt"]), secs(d["AOBT_3_flt"])
    d["rwy"] = d["RUNWAY_mvt"].astype(str)
    res = []
    for apt, g in d.groupby(d["ADEP_mvt"].astype(str)):
        t = g["aobt"].to_numpy()
        has = np.isfinite(t)
        tq = np.where(has, t, 0.0)
        out = pd.DataFrame({MID: g[MID].to_numpy()})
        out["gr_overdue"] = open_count(g["eobt"].to_numpy(), g["aobt"].to_numpy(), tq)
        ov, ac = np.full(len(g), np.nan), np.full(len(g), np.nan)
        for r, idx in g.groupby("rwy").indices.items():
            gg = g.iloc[idx]
            ov[idx] = open_count(gg["eobt"].to_numpy(), gg["aobt"].to_numpy(), tq[idx])
            a = open_count(gg["aobt"].to_numpy(), gg["mvt"].to_numpy(), tq[idx])
            self_in = (gg["aobt"].to_numpy() < gg["mvt"].to_numpy())  # own interval contains own t
            ac[idx] = a - self_in
        out["gr_overdue_rwy"], out["gr_active_rwy"] = ov, ac
        ok = np.isfinite(g["aobt"].to_numpy()) & np.isfinite(g["sch"].to_numpy())
        out["gr_aobt_sched_60"], _ = win_mean(g["aobt"].to_numpy()[ok],
                                              np.clip((g["aobt"] - g["sch"]).to_numpy()[ok], -7200, 7200),
                                              tq, 3600)
        _, out["gr_cadence_60"] = win_mean(g["mvt"].to_numpy(), np.zeros(len(g)), tq, 3600)
        for c in out.columns[1:]:
            out.loc[~has, c] = np.nan
        res.append(out)
        print(f"  {apt}: {len(g):,}", flush=True)
    f = pd.concat(res, ignore_index=True)
    for c in f.columns[1:]:
        f[c] = f[c].astype("float32")
    f.to_parquet(OUT, index=False)
    print(f.describe().T[["count", "mean", "50%", "max"]].round(2))
    print(f"wrote {OUT} rows={len(f):,} ({time.time() - t0:.0f}s)\nDONE")


if __name__ == "__main__":
    main()
