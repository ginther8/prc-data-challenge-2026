"""Rebuild data/processed/v8_prep_cache.parquet from the raw 2025 training files.

Uses ONLY the feature functions in reference/v8/train_catboost_v8_defs.py, in
the same order the V4 submission script applies them to the ranking data:

    load all movements (BLOCK_TIME_UTC_mvt never loaded)
    add_traffic_feature      - on ALL movements (DEP + ARR), whole 2025 stream
    keep PHASE_mvt == "DEP"
    add_time_features
    add_t4_features          (+ verify_t4_deltas epoch re-check)
    apply_categorical_handling

Output columns: MVT_ID_mvt, MVT_TIME_UTC_mvt, TAXITIME_SEC_mvt,
                the 12 V7 features, the 5 T4 deltas.
FLIGHT_mvt is deliberately NOT included (the benchmark joins it itself).
Missing T4 deltas are stored as nulls (not NaN) so MISSING_4 detection works.

Run from the project root:
    python experiments/rebuild_v8_prep_cache.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "reference" / "v8"))

from train_catboost_v8_defs import (  # noqa: E402
    DATA_DIR, MINUEND, T4_FEATURES, T4_SPEC, TARGET, V7_CATEGORICALS,
    V7_FEATURES, add_t4_features, add_time_features, add_traffic_feature,
    apply_categorical_handling, verify_t4_deltas,
)

MID = "MVT_ID_mvt"
FORBIDDEN = "BLOCK_TIME_UTC_mvt"
EXPECTED_MOVEMENTS = 4_167_797
EXPECTED_DEPARTURES = 2_085_047
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "v8_prep_cache.parquet"

RAW_COLS = sorted({
    MID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", MINUEND, TARGET,
    *[c for c in V7_FEATURES if c not in ("hour", "day_of_week", "month",
                                          "traffic_last_15min")],
    *[src for _, src in T4_SPEC],
})
OUT_COLS = [MID, MINUEND, TARGET] + V7_FEATURES + T4_FEATURES


def fail(msg):
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


def main():
    t0 = time.time()
    if FORBIDDEN in RAW_COLS:
        fail("forbidden column requested")
    files = sorted(Path(DATA_DIR).glob("training_*.parquet"))
    if len(files) != 12:
        fail(f"expected 12 training files, found {len(files)}")
    print(f"loading {len(files)} files, {len(RAW_COLS)} columns "
          f"({FORBIDDEN} not loaded)", flush=True)
    frame = pl.concat([pl.from_arrow(pq.read_table(f, columns=RAW_COLS))
                       for f in files], how="vertical_relaxed").to_pandas()
    if len(frame) != EXPECTED_MOVEMENTS:
        fail(f"movements {len(frame):,} != {EXPECTED_MOVEMENTS:,}")
    if frame[MID].duplicated().any():
        fail("duplicate MVT_ID_mvt")
    print(f"  movements: {len(frame):,}", flush=True)

    frame = add_traffic_feature(frame)
    dep = frame[frame["PHASE_mvt"] == "DEP"].copy()
    del frame
    if len(dep) != EXPECTED_DEPARTURES:
        fail(f"departures {len(dep):,} != {EXPECTED_DEPARTURES:,}")
    print(f"  departures: {len(dep):,}", flush=True)

    dep = add_time_features(dep)
    dep = add_t4_features(dep)
    print("  T4 epoch re-check:", flush=True)
    verify_t4_deltas(dep)
    dep = apply_categorical_handling(dep)

    out = pl.from_pandas(dep[OUT_COLS], nan_to_null=True)
    if FORBIDDEN in out.columns or "FLIGHT_mvt" in out.columns:
        fail("unexpected column in output")
    if out[MID].n_unique() != out.height:
        fail("duplicate ids in output")
    for c in T4_FEATURES:
        nn = out[c].is_nan().sum() if out[c].dtype.is_float() else 0
        if nn:
            fail(f"{c} still contains {nn} NaN (must be null)")
    nm = sum(out[c].is_null().cast(pl.Int32) for c in
             ["d_AOBT3", "d_EOBT1", "d_IOBT", "d_LOBT"])
    pattern = out.select(nm.alias("k"))["k"].value_counts().sort("k")
    print("  null-count pattern over d_AOBT3/d_EOBT1/d_IOBT/d_LOBT:")
    for k, n in pattern.iter_rows():
        print(f"    {k} missing: {n:,}")
    if out[TARGET].null_count():
        fail("target has nulls")
    for c in V7_CATEGORICALS:
        if out[c].null_count():
            fail(f"categorical {c} has nulls after handling")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT_PATH)
    print(f"  traffic_last_15min mean={out['traffic_last_15min'].mean():.3f} "
          f"max={out['traffic_last_15min'].max()}")
    print(f"wrote {OUT_PATH}  rows={out.height:,}  cols={out.width}  "
          f"({time.time()-t0:.0f}s)")
    print("DONE")


if __name__ == "__main__":
    main()
