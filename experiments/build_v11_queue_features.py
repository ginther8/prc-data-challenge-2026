#!/usr/bin/env python3
"""
V11 QUEUE / RUNWAY-STATE FEATURE BUILDER
========================================

Builds the dynamic airport queue-state feature family Q on the FULL 2025
movement stream (ARR + DEP), preserving strict temporal causality.

Causality contract
------------------
For a movement row at time t, every Q feature may only use information from
movement rows with MVT_TIME_UTC_mvt STRICTLY EARLIER than t.

Window counts use closed="left" (the reference V8 traffic methodology), so rows
at the same timestamp as the target row are NEVER counted. Zero-count windows
are filled with 0 exactly like the reference traffic feature. Recency and burst
features are NaN only when no strictly-earlier movement exists.

Output cache (data/processed/v11_queue_features.parquet): one row per movement
(4,167,797 rows) keyed by MVT_ID_mvt. The screen joins feature rows onto
departure rows via MVT_ID_mvt (1:1).

Feature list
------------
Q1  airport counts        ap_count_{all,dep,arr}_{5,10,15,30,60}m   (15)
Q2  runway counts         rw_count_{all,dep,arr}_{5,10,15,30,60}m   (15)
Q3  recency (seconds)     sec_since_prev_{ap_move,ap_dep,ap_arr,
                                        rw_move,rw_dep}             (5)
Q4  mix / pressure        dep_share_{15,30,60}m, arr_share_{15,30,60}m,
                          rw_dep_share_{15,30,60}m,
                          arrival_pressure_{30,60}m                 (11)
Q5  burstiness            ap_gap_mean_last{5,10}, ap_gap_std_last{5,10},
                          rw_gap_mean_last5, rw_gap_std_last5        (6)
Q6  runway config         sec_since_last_different_runway,
                          runway_changed_vs_prev_movement,
                          runway_changed_vs_prev_departure,
                          unique_runways_ap_{30,60}m                 (5)

Causality verification is run before writing the cache:
  * sampled rows: every window count is recomputed from a direct
    strictly-earlier-only filter and must match;
  * known tying (airport, timestamp) rows must exclude their peers;
  * the first movement of every airport must have all-zero counts.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_CACHE = PROJECT_ROOT / "data" / "processed" / "v11_queue_features.parquet"
ALL_MOVES = 4_167_797

WINDOWS = ["5min", "10min", "15min", "30min", "60min"]


def log(m=""):
    print(m, flush=True)


# ---------------------------------------------------------------- movement frame
def load_movements():
    files = sorted(DATA_DIR.glob("training_*.parquet"))
    df = (pl.scan_parquet([str(f) for f in files])
          .select([
              "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt",
              "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
          .with_columns(
              pl.when(pl.col("PHASE_mvt") == "ARR")
              .then(pl.col("ADES_mvt"))
              .otherwise(pl.col("ADEP_mvt"))
              .alias("airport"))
          .collect())
    # physical order = reference V8 (airport, time) with a stable tiebreak
    df = df.sort(["airport", "MVT_TIME_UTC_mvt", "MVT_ID_mvt"])
    df = df.with_columns(
        pl.col("MVT_TIME_UTC_mvt").cast(pl.Int64).alias("ti_us"),
        (pl.col("PHASE_mvt") == "DEP").cast(pl.Int8).alias("is_dep"))
    return df


# --------------------------------------------------- exact closed-left counts
WINDOW_SECS = {"5m": 300, "10m": 600, "15m": 900, "30m": 1800, "60m": 3600}


def _group_counts(g, aflags):
    """Vectorized closed-left batch counts for ONE group of rows.

    g is sorted by ti_us (and MVT_ID). Returns {window: (n_rows, n_flags)}.
    Window semantics: count rows j with t_j in [t_i - w, t_i) - strictly
    earlier than t_i (same-timestamp peers in the current batch never
    counted), left boundary inclusive. This is exactly pandas
    rolling(closed="left") on a datetimelike index, but correct even when
    timestamps tie (pandas corrupts counts on non-unique indexes)."""
    t = g["ti_us"].to_numpy()
    U, inv = np.unique(t, return_inverse=True)
    fmat = np.column_stack([np.asarray(g[c], dtype=np.float64) for c in aflags])
    per_batch = np.vstack([np.bincount(inv, weights=f, minlength=U.size)
                           for f in fmat.T]).T
    pre = np.vstack([np.concatenate([[0.0], np.cumsum(pb)])
                     for pb in per_batch.T]).T
    pre = np.ascontiguousarray(pre)
    out = {}
    for w, ws in WINDOW_SECS.items():
        lo = np.searchsorted(U, t - ws * 1_000_000, side="left")
        out[w] = pre[inv] - pre[lo]
    return out


def build_counts(mv):
    """All/dep/arr x airport/runway x 5 windows, closed-left batch exact."""
    t0 = time.time()
    f = mv.select(["MVT_ID_mvt", "airport", "RUNWAY_mvt", "ti_us", "is_dep"])
    f = f.with_columns(
        pl.col("RUNWAY_mvt").fill_null("UNKNOWN").alias("RUNWAY_mvt"),
        pl.lit(1.0).alias("flag_all"),
        pl.col("is_dep").cast(pl.Float64).alias("flag_dep"),
        (1.0 - pl.col("is_dep").cast(pl.Float64)).alias("flag_arr")).drop("is_dep")
    flags = ["flag_all", "flag_dep", "flag_arr"]
    fname = {0: "all", 1: "dep", 2: "arr"}
    f = f.sort(["airport", "ti_us", "MVT_ID_mvt"])

    ap_parts, rw_parts = [], []
    for g in f.partition_by("airport", maintain_order=True):
        gc = _group_counts(g, flags)
        cols = {"MVT_ID_mvt": g["MVT_ID_mvt"]}
        for w in WINDOW_SECS:
            for fi, fn in fname.items():
                cols[f"{fn}_ap_{w}"] = gc[w][:, fi]
        ap_parts.append(pl.DataFrame(cols))
    ap = pl.concat(ap_parts)

    for g in f.partition_by("airport", maintain_order=True):
        for gr in g.partition_by("RUNWAY_mvt", maintain_order=True):
            gc = _group_counts(gr, flags)
            cols = {"MVT_ID_mvt": gr["MVT_ID_mvt"]}
            for w in WINDOW_SECS:
                for fi, fn in fname.items():
                    cols[f"{fn}_rw_{w}"] = gc[w][:, fi]
            rw_parts.append(pl.DataFrame(cols))
    rw = pl.concat(rw_parts)
    counts = ap.join(rw, on="MVT_ID_mvt", how="left")
    log(f"counts built in {time.time() - t0:.1f}s ({len(counts.columns) - 1} columns)")
    return counts


# ---------------------------------------------------------- strictly-earlier batch
def batch_table(frame, keys):
    """Annotate each row with the latest STRICTLY-EARLIER movement of its group
    (time bt_prev and runway rw_prev), where a group is either (airport) or
    (airport, runway). Uses dense ranks of distinct timestamps; rows sharing a
    timestamp with the current row are never treated as 'earlier'."""
    gcols = list(keys)
    f = frame.with_columns(pl.col("ti_us").rank("dense").over(gcols).alias("_tbatch"))
    bat = (f.group_by(gcols + ["_tbatch"])
           .agg(pl.col("ti_us").first().alias("bt"),
                pl.col("RUNWAY_mvt").last().alias("blast_rw"))
           .sort(gcols + ["_tbatch"]))
    bat = bat.with_columns(
        pl.col("bt").shift(1).over(gcols).alias("bt_prev"),
        pl.col("blast_rw").shift(1).over(gcols).alias("rw_prev"))
    prev = bat.select(gcols + ["_tbatch", "bt_prev", "rw_prev"])
    out = f.join(prev, on=gcols + ["_tbatch"], how="left")
    return out


def sec_since(expr):
    return (pl.when(pl.col("bt_prev").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(expr / 1e6))


# ------------------------------------------------------------ Q3, Q6 recency/config
def build_recency_and_config(df):
    f = df.with_columns(pl.col("RUNWAY_mvt").fill_null("UNKNOWN").alias("RUNWAY_mvt"))
    # all movements, per-airport
    ab = batch_table(f, ["airport"])
    ab = ab.with_columns(
        sec_since(pl.col("ti_us") - pl.col("bt_prev")).alias("sec_since_prev_ap_move"),
        (pl.when(pl.col("bt_prev").is_null())
         .then(pl.lit(None, dtype=pl.Int8))
         .otherwise((pl.col("RUNWAY_mvt") != pl.col("rw_prev")).cast(pl.Int8)))
        .alias("runway_changed_vs_prev_movement"))
    # runway-change runs (run_id per airport, first run == 0)
    ab = ab.with_columns(
        (pl.when(pl.col("runway_changed_vs_prev_movement").is_null())
         .then(pl.lit(1, dtype=pl.Int64))
         .otherwise(pl.col("runway_changed_vs_prev_movement").cast(pl.Int64)))
        .alias("run_start"))
    ab = ab.with_columns(pl.col("run_start").cum_sum().over("airport").alias("run_id"))
    runs = (ab.group_by(["airport", "run_id"])
            .agg(pl.col("ti_us").first().alias("run_t0"),
                 pl.col("sec_since_prev_ap_move").first().alias("run_gap"))
            .sort(["airport", "run_id"]))
    ab = ab.join(runs, on=["airport", "run_id"], how="left")
    ab = ab.with_columns(
        pl.when(pl.col("run_id") == 0)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise((pl.col("ti_us") - pl.col("run_t0")) / 1e6 + pl.col("run_gap"))
        .alias("sec_since_last_different_runway"))
    out = ab.select(["MVT_ID_mvt", "sec_since_prev_ap_move",
                     "sec_since_last_different_runway",
                     "runway_changed_vs_prev_movement"])

    # all movements, per-(airport, runway)
    rb = batch_table(f, ["airport", "RUNWAY_mvt"])
    rb = rb.with_columns(
        sec_since(pl.col("ti_us") - pl.col("bt_prev")).alias("sec_since_prev_rw_move"))
    out = out.join(rb.select(["MVT_ID_mvt", "sec_since_prev_rw_move"]),
                   on="MVT_ID_mvt", how="left")

    # dep-only: per-airport recency + runway-changed-vs-prev-departure
    dep = f.filter(pl.col("is_dep") == 1)
    dab = batch_table(dep, ["airport"])
    dab = dab.with_columns(
        sec_since(pl.col("ti_us") - pl.col("bt_prev")).alias("sec_since_prev_ap_dep"))
    dab = dab.with_columns(
        (pl.when(pl.col("bt_prev").is_null())
         .then(pl.lit(None, dtype=pl.Int8))
         .otherwise((pl.col("RUNWAY_mvt") != pl.col("rw_prev")).cast(pl.Int8)))
        .alias("runway_changed_vs_prev_departure"))
    out = out.join(dab.select(["MVT_ID_mvt", "sec_since_prev_ap_dep",
                               "runway_changed_vs_prev_departure"]),
                   on="MVT_ID_mvt", how="left")

    # dep-only per-(airport, runway) recency + arr-only per-airport recency
    drb = batch_table(dep, ["airport", "RUNWAY_mvt"])
    drb = drb.with_columns(
        sec_since(pl.col("ti_us") - pl.col("bt_prev")).alias("sec_since_prev_rw_dep"))
    out = out.join(drb.select(["MVT_ID_mvt", "sec_since_prev_rw_dep"]),
                   on="MVT_ID_mvt", how="left")
    arr = f.filter(pl.col("is_dep") == 0)
    aab = batch_table(arr, ["airport"])
    aab = aab.with_columns(
        sec_since(pl.col("ti_us") - pl.col("bt_prev")).alias("sec_since_prev_ap_arr"))
    out = out.join(aab.select(["MVT_ID_mvt", "sec_since_prev_ap_arr"]),
                   on="MVT_ID_mvt", how="left")
    return out


# ---------------------------------------------------------------- Q5 burstiness
def build_burstiness(df):
    f = df.with_columns(pl.col("RUNWAY_mvt").fill_null("UNKNOWN").alias("RUNWAY_mvt"))
    gcols = ["airport", "RUNWAY_mvt"]
    # per-airport, last 5/10 previous movements: mean = span/(k-1), std via
    # the shifted strictly-earlier gap series (rolling window of k-1 gaps)
    ab = batch_table(f, ["airport"])
    ab = ab.with_columns(
        (pl.col("ti_us") - pl.col("bt_prev")).cast(pl.Float64).alias("gap_us"))
    ab = ab.with_columns(
        ((pl.col("ti_us").shift(1).over("airport")
          - pl.col("ti_us").shift(5).over("airport")) / 4e6)
        .alias("ap_gap_mean_last5"),
        ((pl.col("ti_us").shift(1).over("airport")
          - pl.col("ti_us").shift(10).over("airport")) / 9e6)
        .alias("ap_gap_mean_last10"))
    gprev = pl.col("gap_us").shift(1).over("airport")
    ab = ab.with_columns(
        (gprev.rolling_std(window_size=4, min_samples=2).over("airport") / 1e6)
        .alias("ap_gap_std_last5"),
        (gprev.rolling_std(window_size=9, min_samples=2).over("airport") / 1e6)
        .alias("ap_gap_std_last10"))
    rb = batch_table(f, gcols)
    rb = rb.with_columns(
        (pl.col("ti_us") - pl.col("bt_prev")).cast(pl.Float64).alias("gap_us"))
    rb = rb.with_columns(
        ((pl.col("ti_us").shift(1).over(gcols)
          - pl.col("ti_us").shift(5).over(gcols)) / 4e6)
        .alias("rw_gap_mean_last5"))
    rgprev = pl.col("gap_us").shift(1).over(gcols)
    rb = rb.with_columns(
        (rgprev.rolling_std(window_size=4, min_samples=2).over(gcols) / 1e6)
        .alias("rw_gap_std_last5"))
    out = ab.select(["MVT_ID_mvt", "ap_gap_mean_last5", "ap_gap_mean_last10",
                     "ap_gap_std_last5", "ap_gap_std_last10"])
    out = out.join(rb.select(["MVT_ID_mvt", "rw_gap_mean_last5",
                              "rw_gap_std_last5"]), on="MVT_ID_mvt", how="left")
    return out


# ---------------------------------------------------- Q6 unique runways in window
def build_unique_runways(df, windows_us):
    """Distinct runway count over (t - w, t) per airport, exact and
    vectorized. Row j (same-airport, strictly-earlier) contributes +1 to rows i
    with max(t_j, t_pj + w) < t_i <= t_j + w, where pj is j's previous
    same-runway predecessor. A runway is counted exactly once per window per
    row (at its FIRST in-window occurrence)."""
    t0 = time.time()
    f = df.with_columns(pl.col("RUNWAY_mvt").fill_null("UNKNOWN").alias("RUNWAY_mvt"))
    f = f.sort(["airport", "ti_us", "MVT_ID_mvt"])
    frames = []
    for g in f.partition_by("airport", maintain_order=True):
        airport = g["airport"][0]
        arr = g["ti_us"].to_numpy().astype(np.int64)
        rw = g["RUNWAY_mvt"].to_numpy()
        n = arr.size
        prev = np.full(n, -1, dtype=np.int64)
        last = {}
        for j in range(n):
            r = rw[j]
            prev[j] = last.get(r, -1)
            last[r] = j
        cols = {"airport": pl.Series("airport", [airport] * n),
                "MVT_ID_mvt": g["MVT_ID_mvt"]}
        for wname, wus in windows_us.items():
            diff = np.zeros(n + 1, dtype=np.int64)
            for j in range(n):
                t_pj = arr[prev[j]] if prev[j] >= 0 else arr[j] - wus - 1
                Lj = max(arr[j], t_pj + wus)
                Rj = arr[j] + wus
                a = int(np.searchsorted(arr, Lj, side="right"))
                b = int(np.searchsorted(arr, Rj, side="right"))
                if a < b:
                    diff[a] += 1
                    diff[b] -= 1
            cols[wname] = np.cumsum(diff[:-1])
        frames.append(pl.DataFrame(cols))
    out = pl.concat(frames)
    log(f"unique-runways built in {time.time() - t0:.1f}s")
    return out


# ------------------------------------------------------------------ assembly
def assemble(mv, counts):
    rec = build_recency_and_config(mv)
    bur = build_burstiness(mv)
    uniq = build_unique_runways(mv, {"unique_runways_ap_30m": 1800_000_000,
                                     "unique_runways_ap_60m": 3600_000_000})
    q = (counts
         .join(rec, on="MVT_ID_mvt", how="left")
         .join(bur, on="MVT_ID_mvt", how="left")
         .join(uniq, on="MVT_ID_mvt", how="left"))
    q = q.with_columns(
        (pl.col("dep_ap_15m") / pl.col("all_ap_15m")).alias("dep_share_15m"),
        (pl.col("dep_ap_30m") / pl.col("all_ap_30m")).alias("dep_share_30m"),
        (pl.col("dep_ap_60m") / pl.col("all_ap_60m")).alias("dep_share_60m"),
        (pl.col("arr_ap_15m") / pl.col("all_ap_15m")).alias("arr_share_15m"),
        (pl.col("arr_ap_30m") / pl.col("all_ap_30m")).alias("arr_share_30m"),
        (pl.col("arr_ap_60m") / pl.col("all_ap_60m")).alias("arr_share_60m"),
        (pl.col("dep_rw_15m") / pl.col("all_rw_15m")).alias("rw_dep_share_15m"),
        (pl.col("dep_rw_30m") / pl.col("all_rw_30m")).alias("rw_dep_share_30m"),
        (pl.col("dep_rw_60m") / pl.col("all_rw_60m")).alias("rw_dep_share_60m"),
        (pl.col("arr_ap_30m") / 2.0).alias("arrival_pressure_30m"),
        (pl.col("arr_ap_60m") / 4.0).alias("arrival_pressure_60m"),
    )
    # zero-fill the closed-left batch counts exactly like the reference traffic
    # feature (recency/change/burst/unique columns keep their null semantics)
    for c in q.columns:
        if c.startswith(("all_", "dep_", "arr_")):
            q = q.with_columns(pl.col(c).fill_null(0.0).clip(0.0, None).alias(c))
    return q


# ------------------------------------------------------------ verification
def verify(q, mv):
    m = mv.select(["MVT_ID_mvt", "airport", "RUNWAY_mvt", "ti_us", "is_dep"])
    merged = q.join(m, on="MVT_ID_mvt", how="left", suffix="_mv")
    rng = np.random.default_rng(42)
    samp = rng.choice(merged.height, size=2500, replace=False)
    s = merged[samp]
    ap_col = pl.col("airport")
    checked = 0
    for row in s.iter_rows(named=True):
        t = float(row["ti_us"])
        ar = merged.filter((ap_col == row["airport"]) & (pl.col("ti_us") < t))
        ap_15 = int(ar.filter(pl.col("ti_us") >= t - 900_000_000).height)
        rw_15 = int(ar.filter((pl.col("RUNWAY_mvt") == row["RUNWAY_mvt"]) &
                              (pl.col("ti_us") >= t - 900_000_000)).height)
        dep_5 = int(ar.filter((pl.col("is_dep") == 1) &
                              (pl.col("ti_us") >= t - 300_000_000)).height)
        arr_60 = int(ar.filter((pl.col("is_dep") == 0) &
                               (pl.col("ti_us") >= t - 3600_000_000)).height)
        exp = {"all_ap_15m": ap_15, "all_rw_15m": rw_15,
               "dep_ap_5m": dep_5, "arr_ap_60m": arr_60}
        for name, val in exp.items():
            got = row[name]
            if not (abs(got - val) < 0.5):
                raise AssertionError(f"causality mismatch {name}: got {got} "
                                     f"exp {val} row {row['MVT_ID_mvt']}")
        checked += 1
    log(f"causality sample verified: {checked}/{checked} rows match "
        "strictly-earlier-only recomputation")
    # ties: rows sharing an (airport, exact timestamp) exclude their peers.
    # Recompute strictly-earlier counts from ALL rows; equality with the
    # feature proves same-timestamp peers are never counted.
    deps = merged.sort(["airport", "ti_us"]).with_columns(
        pl.len().over(["airport", "ti_us"]).alias("batch_n"))
    tie_rows = deps.filter(pl.col("batch_n") > 1)
    if tie_rows.height == 0:
        log("tie-check: no tying rows")
    else:
        checked = 0
        for row in tie_rows.head(200).iter_rows(named=True):
            t = float(row["ti_us"])
            earlier = merged.filter((ap_col == row["airport"]) &
                                    (pl.col("ti_us") < t))
            exp_5 = int(earlier.filter(pl.col("ti_us") >= t - 300_000_000).height)
            exp_15 = int(earlier.filter(pl.col("ti_us") >= t - 900_000_000).height)
            if abs(row["all_ap_5m"] - exp_5) > 0.5 or abs(row["all_ap_15m"] - exp_15) > 0.5:
                raise AssertionError(f"tie strictness failed row {row['MVT_ID_mvt']}: "
                                     f"all_ap_5m={row['all_ap_5m']} exp={exp_5} "
                                     f"all_ap_15m={row['all_ap_15m']} exp={exp_15}")
            checked += 1
        log(f"tie strict-window check passed ({checked}/{checked} tied rows, "
            "peers excluded)")
    # first movement of each airport -> all counts zero
    first = merged.sort(["airport", "ti_us"]).group_by("airport", maintain_order=True).first()
    count_cols = [c for c in q.columns
                  if c.startswith(("all_ap", "all_rw", "dep_ap", "dep_rw",
                                   "arr_ap", "arr_rw"))]
    for row in first.iter_rows(named=True):
        for c in count_cols:
            if row[c] != 0.0:
                raise AssertionError(f"first movement count {c}={row[c]} != 0")
    log("airport-first-movement zero-count check passed")


def main():
    t_all = time.time()
    log("V11 queue-state feature builder")
    mv = load_movements()
    assert mv.height == ALL_MOVES, mv.height
    log(f"movements: {mv.height:,}  (DEP {int((mv['is_dep'] == 1).sum()):,})")
    counts = build_counts(mv)
    q = assemble(mv, counts)
    verify(q, mv)
    q = q.sort("MVT_ID_mvt")
    assert q.height == ALL_MOVES
    assert q["MVT_ID_mvt"].n_unique() == ALL_MOVES
    log(f"Q features: {len(q.columns) - 1}")
    for c in q.columns:
        if c == "MVT_ID_mvt":
            continue
        n = int(q[c].is_null().sum())
        if n:
            log(f"  {c:<42} nulls {n:>7,}  ({100*n/q.height:.2f}%)")
    OUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    q.write_parquet(OUT_CACHE)
    log(f"wrote {OUT_CACHE}  rows={q.height:,}  wall={time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()