"""V10-B1 - build the ERA5 weather join for the PRC taxi-time competition.

This module is DATA ENGINEERING ONLY. It never trains a model, never reads a
target, and never produces a prediction. It turns raw ERA5 hourly extracts into
a compact per-airport hourly table and joins that table onto movement rows.

WHAT IS READ FROM THE COMPETITION DATA
--------------------------------------
Exactly three columns, ever:

    MVT_ID_mvt        movement identifier (join key / row identity)
    ADEP_mvt          the physical DEPARTURE airport of the movement
    MVT_TIME_UTC_mvt  the movement (off-block) instant, in UTC

`MOVEMENT_COLS` is the single source of truth for that list and every read in
this file goes through `read_movements()`. `TAXITIME_SEC_mvt` (the target) and
`BLOCK_TIME_UTC_mvt` (the far end of the taxi interval) are structurally
excluded, so no weather value can be a function of the target. Weather depends
only on: airport location + movement timestamp + ERA5. See the leakage audit in
reports/v10b1_weather_join_audit.txt.

WHY ADEP_mvt AND NOT ADES_mvt
-----------------------------
For a departure, ADEP_mvt is the airport the aircraft physically taxis out
from, and that is where the weather must come from. ADES_mvt is the destination
and is a completely different place. Using the destination would attach the
weather of the arrival airport to a departure. Verified: on 344,516 of the
344,841 ranking departures ADEP_mvt != ADES_mvt, so this is not a rare edge
case - getting it wrong would mis-place almost every row.

TIME SEMANTICS - THE PART THAT IS EASY TO GET WRONG
---------------------------------------------------
ERA5 hourly single levels mixes two kinds of variable on one time axis, and the
mixing is documented, not a defect (see V10-A section 2):

  * "instantaneous" variables are the state VALID AT the timestamp T.
  * "accumulated" variables are totals over the hour ENDING at T, i.e. the
    window (T-1h, T]. total_precipitation and 10m_wind_gust_since_previous_
    post_processing are of this kind and are forecast-derived rather than
    analyses.

A single uniform nearest-hour join silently mis-times the second group relative
to the first by up to a full hour. This module therefore uses two different
offsets, applied per variable kind:

  instantaneous -> era5_instant_ts = FLOOR(MVT to the hour)
                   the state valid at the top of the hour containing the
                   movement. Never later than the movement: strictly causal.

  accumulated    -> era5_accum_ts = CEIL(MVT to the hour)
                   the record stamped CEIL(MVT) covers (FLOOR(MVT), CEIL(MVT)],
                   which is exactly the hour containing the movement, so it
                   always contains the movement instant.

Worked proof for the four timestamps named in the V10-B1 brief (all UTC):

  MVT        instantaneous -> floor   accum -> ceil    accum window covered
  ---------  ---------------------  --------------    ------------------------
  10:05:00   10:00:00              11:00:00            (10:00, 11:00]
  10:30:00   10:00:00              11:00:00            (10:00, 11:00]
  10:59:00   10:00:00              11:00:00            (10:00, 11:00]
  11:00:00   11:00:00              11:00:00            (10:00, 11:00]

Two properties are worth stating explicitly.

1. An exact-hour movement is handled correctly and identically by both rules.
   ceil(11:00) = 11:00, not 12:00, so the record stamped 11:00 - whose window
   (10:00, 11:00] includes 11:00 - is the one used. There is no off-by-one at
   the hour boundary.

2. The ceil rule references a timestamp later than the movement, and that is
   DELIBERATE, not accidental look-ahead. It is not a different hour: the value
   stamped 11:00 is an aggregate over 10:00-11:00, so for a 10:30 movement it
   is the aggregate of the hour the aircraft is taxiing in. Using it is the
   physically correct reading of an hour-accumulated field. It never reaches
   into the following hour. If a reviewer prefers a strictly non-referencing
   variant, `--accum-policy floor` implements the uniform "previous completed
   hour" alternative for every variable, and V10-B2 should ablate the two.

Both offsets are pure functions of an input column, so train-time and
serve-time behaviour are identical by construction, and neither can differ
between the 2025 training files and the 2026 ranking file.

DERIVED QUANTITIES
------------------
Only deterministic, target-free transformations of the ERA5 fields:
  wind_speed        = sqrt(u10^2 + v10^2)                       [m/s]
  wind_direction    = meteorological direction wind blows FROM  [deg]
  dewpoint_spread   = t2m - d2m                                 [K]
  msl_anomaly_30d   = msl - trailing 30-day mean of msl at that airport
The last one is a weather-only climatology. It needs 30 days of leading history,
so for 2025-01 it requires December 2024, which the current download plan does
not cover; `build_hourly` leaves it NaN and the audit reports the gap rather
than silently filling it.

USAGE
-----
  python src/prepare_era5_weather.py plan
      Required ERA5 record coverage, computed from movement timestamps alone.
      Runs with NO weather data present. This is the pre-download check.

  python src/prepare_era5_weather.py build
      Ingest raw CDS CSV extracts -> data/external/era5/era5_hourly.parquet

  python src/prepare_era5_weather.py join
      Join hourly weather onto 2025 training + 2026 ranking movements
      -> data/external/era5/weather_joined.parquet

  python src/prepare_era5_weather.py all
      build, then join
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_weather_era5 import AIRPORTS, VARIABLES  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ERA5_DIR = Path("data/external/era5")
# The V10-A downloader writes its extracts straight into ERA5_DIR (its own OUT_DIR
# default), so the ingester must read from there rather than a raw/ subdirectory.
RAW_DIR = ERA5_DIR
HOURLY_PARQUET = ERA5_DIR / "era5_hourly.parquet"
JOINED_PARQUET = ERA5_DIR / "weather_joined.parquet"
TRAINING_GLOB = "data/raw/training_*.parquet"
RANKING_PARQUET = "data/raw/ranking.parquet"

# --------------------------------------------------------------------------
# Leakage barrier
# --------------------------------------------------------------------------
# The ONLY columns this module is permitted to read from competition data.
# TAXITIME_SEC_mvt (target) and BLOCK_TIME_UTC_mvt are absent by construction.
MOVEMENT_COLS = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]
PHASE_COL = "PHASE_mvt"

FORBIDDEN_COLS = ("TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt")

# --------------------------------------------------------------------------
# Variable registry, derived from the V10-A scaffold so there is one source
# --------------------------------------------------------------------------
# short name -> (era5 name, kind, units, kind_code)
#   kind_code 0 = instantaneous (floor join)
#   kind_code 1 = accumulated    (ceil  join)
VAR_META = {short: (era5, kind, units) for era5, short, kind, units in VARIABLES}
INSTANT_VARS = [s for e, s, k, u in VARIABLES if k == "instantaneous"]
ACCUM_VARS = [s for e, s, k, u in VARIABLES if k == "accumulated"]
RAW_VARS = INSTANT_VARS + ACCUM_VARS

UNITS = {short: units for era5, short, kind, units in VARIABLES}

AIRPORT_ICAOS = [a[0] for a in AIRPORTS]
AIRPORT_LAT = {a[0]: a[1] for a in AIRPORTS}
AIRPORT_LON = {a[0]: a[2] for a in AIRPORTS}

MSL_ANOMALY_DAYS = 30


# ==========================================================================
# Reading competition data
# ==========================================================================
def read_movements(path, label):
    """Read movement rows using only MOVEMENT_COLS (+PHASE for the DEP filter).

    Reads columns by name via pyarrow, so the target is never materialised in
    memory even transiently. Returns departures only.
    """
    use = [c for c in ([PHASE_COL] + MOVEMENT_COLS) if c in _cols(path)]
    frame = pd.read_parquet(path, columns=use)
    for bad in FORBIDDEN_COLS:
        assert bad not in frame.columns, f"leakage barrier broken in {label}"
    if PHASE_COL in frame.columns:
        frame = frame[frame[PHASE_COL] == "DEP"]
    frame = frame[MOVEMENT_COLS].copy()
    frame["MVT_TIME_UTC_mvt"] = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True)
    assert frame["MVT_TIME_UTC_mvt"].dt.tz is not None, "movement times must be tz-aware"
    assert str(frame["MVT_TIME_UTC_mvt"].dt.tz) == "UTC", "movement times must be UTC"
    return frame


def _cols(path):
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


def load_train_movements():
    frames = []
    for p in sorted(glob.glob(TRAINING_GLOB)):
        frames.append(read_movements(p, p))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["source"] = "training_2025"
    return out


def load_ranking_movements():
    out = read_movements(RANKING_PARQUET, "ranking")
    out.attrs["source"] = "ranking_2026"
    return out


# ==========================================================================
# Time semantics
# ==========================================================================
def era5_inst_ts(mvt):
    """Hourly ERA5 record for instantaneous/state variables: FLOOR."""
    return mvt.dt.floor("h")


def era5_acc_ts(mvt, policy="ceil"):
    """Hourly ERA5 record for accumulated variables.

    policy="ceil"  -> V10-A recommendation. The record stamped CEIL(MVT) covers
                      (FLOOR(MVT), CEIL(MVT)], i.e. the hour containing MVT.
    policy="floor" -> the strict alternative: the previous completed hour for
                      every variable, which never references a later stamp at
                      all but lags accumulated fields by up to an hour.
    """
    if policy == "ceil":
        return mvt.dt.ceil("h")
    if policy == "floor":
        return mvt.dt.floor("h")
    raise ValueError(f"unknown accum policy {policy!r}")


def time_semantics_proof():
    """Return the worked example table for 10:05, 10:30, 10:59, 11:00 UTC."""
    demo = pd.DataFrame(
        {"mvt": pd.to_datetime(
            ["2026-01-15 10:05:00", "2026-01-15 10:30:00",
             "2026-01-15 10:59:00", "2026-01-15 11:00:00"], utc=True)}
    )
    inst = era5_inst_ts(demo["mvt"])
    acc_ceil = era5_acc_ts(demo["mvt"], "ceil")
    acc_floor = era5_acc_ts(demo["mvt"], "floor")
    return pd.DataFrame({
        "mvt_utc": demo["mvt"].dt.strftime("%H:%M:%S"),
        "instantaneous_floor": inst.dt.strftime("%H:%M:%S"),
        "accumulated_ceil": acc_ceil.dt.strftime("%H:%M:%S"),
        "accumulated_window": [
            f"({(b - pd.Timedelta('1h')).strftime('%H:%M')}, {b.strftime('%H:%M')}]"
            for b in acc_ceil
        ],
        "accumulated_floor_alt": acc_floor.dt.strftime("%H:%M:%S"),
    })


# ==========================================================================
# Coverage planning (works with no weather data)
# ==========================================================================
def required_era5_times(mvt, policy="ceil"):
    """The exact set of ERA5 timestamps the join will demand, per variable kind."""
    inst = era5_inst_ts(mvt)
    acc = era5_acc_ts(mvt, policy)
    return inst, acc


def plan_coverage(movements, policy="ceil"):
    inst, acc = required_era5_times(movements["MVT_TIME_UTC_mvt"], policy)
    need = pd.concat([inst, acc]).drop_duplicates().sort_values()
    return {
        "n_movements": int(len(movements)),
        "first_mvt": movements["MVT_TIME_UTC_mvt"].min(),
        "last_mvt": movements["MVT_TIME_UTC_mvt"].max(),
        "n_required_records": int(len(need)),
        "required_first": need.min(),
        "required_last": need.max(),
        "required_times": need,
        "last_instant": inst.max(),
        "last_accum": acc.max(),
    }


def download_window_needed(movements, policy="ceil"):
    """Inclusive [start, end] day range that the downloader must cover.

    Deliberately reports the ACCUMULATED requirement separately, because that is
    the binding constraint: a movement at 23:57 needs the next day's 00:00
    record, which a calendar-month download that stops at month end does not
    contain. This is the specific coverage bug V10-B1 was asked to catch.
    """
    mvt = movements["MVT_TIME_UTC_mvt"]
    inst, acc = required_era5_times(mvt, policy)
    return {
        "instant_start": inst.min().floor("D"),
        "instant_end": inst.max().floor("D"),
        "accum_start": acc.min().floor("D"),
        "accum_end": acc.max().floor("D"),
    }


# ==========================================================================
# Raw ERA5 ingestion
# ==========================================================================
_TIME_COL_CANDIDATES = (
    "validity_time", "valid_time", "validity date", "validity time",
    "date", "time", "datetime", "date_time", "timestamp",
)


def _normalise(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _map_variable_columns(columns, source_name):
    """Match CDS output column names onto our short names, case-insensitively."""
    lut = {_normalise(short): short for short in RAW_VARS}
    lut.update({_normalise(era5): short for era5, short, _k, _u in VARIABLES})
    mapping, unmapped = {}, []
    for col in columns:
        key = _normalise(col)
        if key in lut:
            mapping[col] = lut[key]
        else:
            unmapped.append(col)
    missing = [s for s in RAW_VARS if s not in mapping.values()]
    return mapping, unmapped, missing


def _read_cds_csv(path):
    """Read one CDS extract. The time-series dataset returns a ZIP even when
    data_format='csv', so transparently unwrap the single CSV member if needed.

    Verified against a real response on 2026-09-27:
        member  reanalysis-era5-single-levels-timeseries-sfc<hash>.csv
        header  valid_time,t2m,latitude,longitude
        valid_time is a NAIVE timestamp and must be localised to UTC;
        variable columns use the SHORT ERA5 names (t2m), not the long form.
    """
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if len(members) != 1:
                raise SystemExit(
                    f"{path.name}: expected exactly 1 CSV inside the archive, "
                    f"found {members}"
                )
            with zf.open(members[0]) as fh:
                return pd.read_csv(fh)
    return pd.read_csv(path)


def ingest_raw(raw_dir=RAW_DIR, verbose=True):
    """Load every raw CDS CSV extract into one long, validated frame.

    The exact CSV header of the CDS time-series product was confirmed by a real
    authenticated request (see _read_cds_csv), but the time column and the variable
    columns are still DISCOVERED rather than hard-coded, and any variable that cannot be
    matched is a hard error rather than a silent gap.
    """
    files = sorted(Path(raw_dir).glob("*.csv")) + sorted(Path(raw_dir).glob("*.zip"))
    if not files:
        raise SystemExit(
            f"no raw ERA5 files in {raw_dir}\n"
            "run: python src/download_weather_era5.py --execute"
        )
    frames = []
    for path in files:
        icao = _icao_from_filename(path.name)
        head = _read_cds_csv(path).head(5)
        time_col = None
        for cand in _TIME_COL_CANDIDATES:
            hit = [c for c in head.columns if _normalise(c) == _normalise(cand)]
            if hit:
                time_col = hit[0]
                break
        if time_col is None:
            raise SystemExit(
                f"{path.name}: could not identify a time column. Columns were "
                f"{list(head.columns)}. Update _TIME_COL_CANDIDATES."
            )
        mapping, unmapped, missing = _map_variable_columns(head.columns, path.name)
        if missing:
            raise SystemExit(
                f"{path.name}: ERA5 variable(s) {missing} not found in the file. "
                f"Present columns: {list(head.columns)}"
            )
        if verbose and unmapped:
            print(f"  {path.name}: ignoring extra columns {unmapped}")

        df = _read_cds_csv(path)
        df = df.rename(columns=mapping)
        keep = [time_col] + [s for s in RAW_VARS if s in df.columns]
        df = df[keep].copy()
        ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        if ts.isna().any():
            raise SystemExit(f"{path.name}: {int(ts.isna().sum())} unparseable timestamps")
        df["era5_time"] = ts
        df["airport"] = icao
        frames.append(df[["airport", "era5_time"] + RAW_VARS])

    raw = pd.concat(frames, ignore_index=True)

    # A single CSV may split a month across files; collapse to hourly rows.
    agg = {v: "max" for v in RAW_VARS}
    raw = raw.groupby(["airport", "era5_time"], as_index=False).agg(agg)

    dup = int(raw.duplicated(["airport", "era5_time"]).sum())
    if dup:
        raise SystemExit(f"{dup} duplicate (airport, era5_time) rows remain after collapse")

    off_grid = int((raw["era5_time"].dt.minute != 0).sum() + (raw["era5_time"].dt.second != 0).sum())
    if off_grid:
        print(f"WARNING: {off_grid} timestamps are not on the hour")

    bad = sorted(set(raw["airport"]) - set(AIRPORT_ICAOS))
    if bad:
        raise SystemExit(f"unexpected airport code(s) in raw files: {bad}")

    if verbose:
        print(f"  ingested {len(files)} file(s) -> {len(raw):,} hourly rows")
        print(f"  time range {raw['era5_time'].min()} .. {raw['era5_time'].max()}")
    return raw


def _icao_from_filename(name):
    stem = name.split(".")[0]
    icao = stem.split("_")[0].upper()
    if icao not in AIRPORT_ICAOS:
        raise SystemExit(f"cannot determine airport from filename {name!r}")
    return icao


# ==========================================================================
# Derived hourly table
# ==========================================================================
def wind_speed(u, v):
    return np.hypot(u, v)


def wind_direction_from(u, v):
    """Meteorological convention: the direction the wind blows FROM, degrees.

    Checked: u=0, v>0 (blowing toward the north, i.e. from the south) -> 180.
             u<0, v=0 (blowing toward the west, i.e. from the east)  ->  90.
    """
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def build_hourly(raw, policy="ceil"):
    """Attach derived columns and the ERA5 time-semantics classification."""
    df = raw.sort_values(["airport", "era5_time"]).reset_index(drop=True)

    df["wind_speed"] = wind_speed(df["u10"], df["v10"])
    df["wind_direction"] = wind_direction_from(df["u10"], df["v10"])
    df["dewpoint_spread"] = df["t2m"] - df["d2m"]

    # Weather-only trailing climatology. Needs 30 days of leading history.
    df["msl_anomaly_30d"] = np.nan
    for icao, grp in df.groupby("airport", sort=False):
        s = grp.set_index("era5_time")["msl"]
        roll = s.rolling(f"{MSL_ANOMALY_DAYS}D", min_periods=MSL_ANOMALY_DAYS).mean()
        df.loc[grp.index, "msl_anomaly_30d"] = (s - roll).to_numpy()

    df["instant_vars"] = ",".join(INSTANT_VARS)
    df["accum_vars"] = ",".join(ACCUM_VARS)
    df["accum_policy"] = policy
    return df


# ==========================================================================
# Join
# ==========================================================================
def join_movements(movements, hourly, policy="ceil", source=""):
    """Attach weather to movement rows. Returns (joined, stats)."""
    mvt = movements["MVT_TIME_UTC_mvt"]
    inst_ts = era5_inst_ts(mvt)
    acc_ts = era5_acc_ts(mvt, policy)

    left = pd.DataFrame({
        "MVT_ID_mvt": movements["MVT_ID_mvt"].to_numpy(),
        "ADEP_mvt": movements["ADEP_mvt"].to_numpy(),
        "mvt_time_utc": mvt.to_numpy(),
        "era5_instant_ts": inst_ts.to_numpy(),
        "era5_accum_ts": acc_ts.to_numpy(),
    })

    inst_lookup = hourly[["airport", "era5_time"] + INSTANT_VARS].rename(
        columns={"era5_time": "era5_instant_ts", **{v: f"{v}_inst" for v in INSTANT_VARS}}
    )
    acc_lookup = hourly[["airport", "era5_time"] + ACCUM_VARS].rename(
        columns={"era5_time": "era5_accum_ts", **{v: f"{v}_acc" for v in ACCUM_VARS}}
    )

    out = left.merge(
        inst_lookup, how="left",
        left_on=["ADEP_mvt", "era5_instant_ts"],
        right_on=["airport", "era5_instant_ts"],
    ).drop(columns=["airport"])
    out = out.merge(
        acc_lookup, how="left",
        left_on=["ADEP_mvt", "era5_accum_ts"],
        right_on=["airport", "era5_accum_ts"],
    ).drop(columns=["airport"])

    inst_cols = [f"{v}_inst" for v in INSTANT_VARS]
    acc_cols = [f"{v}_acc" for v in ACCUM_VARS]
    weather_cols = inst_cols + acc_cols

    matched = out[weather_cols].notna().any(axis=1)
    both = out[inst_cols].notna().all(axis=1) & out[acc_cols].notna().all(axis=1)

    out["weather_matched"] = matched
    out["weather_complete"] = both
    out["source"] = source

    stats = {
        "source": source,
        "rows": int(len(out)),
        "matched": int(matched.sum()),
        "complete": int(both.sum()),
        "unmatched": int((~matched).sum()),
        "match_rate": float(matched.mean()) if len(out) else float("nan"),
        "complete_rate": float(both.mean()) if len(out) else float("nan"),
        "partial_rows": int((matched & ~both).sum()),
        "duplicate_mvt_ids": int(out["MVT_ID_mvt"].duplicated().sum()),
    }
    return out, stats


def combine_derived(joined):
    """Flatten the _inst/_acc suffixed columns into single modelling columns.

    Instantaneous variables come from the floor record, accumulated variables
    from the ceil record; the raw values themselves are preserved untouched in
    the suffixed columns.
    """
    out = joined.copy()
    for v in INSTANT_VARS:
        out[v] = out[f"{v}_inst"]
    for v in ACCUM_VARS:
        out[v] = out[f"{v}_acc"]
    return out


# ==========================================================================
# Sanity checks on physical plausibility
# ==========================================================================
PLAUSIBILITY = {
    "t2m": (200.0, 340.0, "K"),
    "d2m": (180.0, 340.0, "K"),
    "u10": (-70.0, 70.0, "m s-1"),
    "v10": (-70.0, 70.0, "m s-1"),
    "fg10": (0.0, 120.0, "m s-1"),
    "tp": (0.0, 0.30, "m"),
    "msl": (95000.0, 105000.0, "Pa"),
    "tcc": (0.0, 1.0, "fraction"),
}


def plausibility_flags(hourly):
    rows = []
    for var, (lo, hi, unit) in PLAUSIBILITY.items():
        s = hourly[var]
        n_lo = int((s < lo).sum())
        n_hi = int((s > hi).sum())
        rows.append({
            "variable": var, "unit": unit, "lo": lo, "hi": hi,
            "min": float(s.min()), "max": float(s.max()),
            "mean": float(s.mean()), "p50": float(s.median()),
            "n_below": n_lo, "n_above": n_hi,
            "n_null": int(s.isna().sum()),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# CLI
# ==========================================================================
def cmd_plan(args):
    print("=" * 78)
    print("V10-B1 STEP 1 - required ERA5 coverage, derived from movement times only")
    print("=" * 78)
    print()
    print("Time semantics proof (all UTC):")
    print(time_semantics_proof().to_string(index=False))
    print()

    for label, loader in (("TRAINING 2025", load_train_movements),
                          ("RANKING 2026", load_ranking_movements)):
        mv = loader()
        mv["ym"] = mv["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
        print("-" * 78)
        print(label)
        print(f"  departure movements      : {len(mv):,}")
        print("  by MVT month             :")
        for ym, n in mv["ym"].value_counts().sort_index().items():
            print(f"      {ym}  {n:>9,}")
        print(f"  MVT range                : {mv['MVT_TIME_UTC_mvt'].min()}"
              f" .. {mv['MVT_TIME_UTC_mvt'].max()}")
        print(f"  distinct ADEP            : {mv['ADEP_mvt'].nunique()}"
              f"  ({', '.join(sorted(mv['ADEP_mvt'].unique()))})")
        print(f"  null ADEP                : {int(mv['ADEP_mvt'].isna().sum())}")
        print(f"  null MVT time            : {int(mv['MVT_TIME_UTC_mvt'].isna().sum())}")

        pl = plan_coverage(mv, args.policy)
        win = download_window_needed(mv, args.policy)
        print(f"  required ERA5 records    : {pl['n_required_records']:,} distinct hours")
        print(f"  required span            : {pl['required_first']} .. {pl['required_last']}")
        print(f"  latest instantaneous     : {pl['last_instant']}")
        print(f"  latest accumulated  (ceil): {pl['last_accum']}")
        print(f"  download days needed     : {win['instant_start'].date()} .."
              f" {win['accum_end'].date()}  (accum end is the binding constraint)")
        mv_ym = mv.drop(columns=["ym"])
        print(f"  airports needing weather : {len(set(mv_ym['ADEP_mvt']) & set(AIRPORT_ICAOS))}"
              f" / {mv_ym['ADEP_mvt'].nunique()}")
        missing_icao = sorted(set(mv_ym["ADEP_mvt"]) - set(AIRPORT_ICAOS))
        print(f"  ADEP codes without coords: {missing_icao if missing_icao else 'none'}")
        print()
    return 0


def cmd_build(args):
    print("=" * 78)
    print("V10-B1 STEP 2 - ingest raw ERA5 extracts into a compact hourly table")
    print("=" * 78)
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    raw = ingest_raw()
    hourly = build_hourly(raw, args.policy)
    hourly.to_parquet(HOURLY_PARQUET, index=False)
    print(f"  wrote {HOURLY_PARQUET}  ({len(hourly):,} rows)")
    print()
    print("  plausibility:")
    print(plausibility_flags(hourly).to_string(index=False))
    return 0


def cmd_join(args):
    print("=" * 78)
    print("V10-B1 STEP 3 - join weather onto movements (QA only, no model)")
    print("=" * 78)
    if not HOURLY_PARQUET.exists():
        raise SystemExit(f"{HOURLY_PARQUET} not found; run 'build' first")
    hourly = pd.read_parquet(HOURLY_PARQUET)
    print(f"  hourly table: {len(hourly):,} rows, "
          f"{hourly['era5_time'].min()} .. {hourly['era5_time'].max()}")
    print()

    parts, all_stats = [], []
    for label, loader in (("training_2025", load_train_movements),
                          ("ranking_2026", load_ranking_movements)):
        mv = loader()
        j, st = join_movements(mv, hourly, args.policy, source=label)
        parts.append(combine_derived(j))
        all_stats.append(st)
        print(f"  {label}: {st['rows']:,} rows, matched {st['match_rate'] * 100:.4f}%, "
              f"unmatched {st['unmatched']:,}, partial {st['partial_rows']:,}")

    joined = pd.concat(parts, ignore_index=True)
    joined.to_parquet(JOINED_PARQUET, index=False)
    print()
    print(f"  wrote {JOINED_PARQUET}  ({len(joined):,} rows)")
    return 0


def cmd_all(args):
    cmd_build(args)
    print()
    return cmd_join(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["plan", "build", "join", "all"])
    ap.add_argument("--policy", choices=["ceil", "floor"], default="ceil",
                    help="join offset for accumulated variables; V10-A recommends ceil")
    args = ap.parse_args(argv)
    return {
        "plan": cmd_plan, "build": cmd_build,
        "join": cmd_join, "all": cmd_all,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
