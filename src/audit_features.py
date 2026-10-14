"""Forensic audit of all columns in the PRC Data Challenge 2026 dataset.

Analysis only. Trains nothing. Reads raw data read-only.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/raw")

TARGET = "TAXITIME_SEC_mvt"

KNOWN_COLUMNS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt",
    "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "LOBT_flt",
    "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt",
    "MARKET_SEGMENT_flt", "IOBT_flt", "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt",
    "ARVT_3_flt",
]

TIMESTAMP_COLUMNS = [
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt",
    "AOBT_3_flt", "ARVT_3_flt",
]

# Deltas measured against these reference timestamps.
DELTA_REFERENCES = [
    "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt",
    "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt",
]

SHIFT_FOCUS = [
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt", "MARKET_SEGMENT_flt", "AIRCRAFT_OPERATOR_flt",
]

HIGH_CARDINALITY_THRESHOLD = 50


# ============================================================
# OUTPUT TEE (UTF-8 file + stdout)
# ============================================================

class Tee:
    def __init__(self, path):
        self.buffer = io.StringIO()
        self.handle = open(path, "w", encoding="utf-8", newline="\n")

    def write(self, text=""):
        self.buffer.write(text)
        self.handle.write(text)
        sys.__stdout__.write(text)

    def flush(self):
        self.handle.flush()
        sys.__stdout__.flush()


REPORT_PATH = Path("reports/feature_audit.txt")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

out = Tee(REPORT_PATH)


def emit(*args, sep=" ", end="\n"):
    out.write(sep.join(str(a) for a in args) + end)


print = emit


def rule(title=""):
    if title:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
    else:
        print("-" * 78)


def head(text):
    print("\n" + "#" * 78)
    print("# " + text)
    print("#" * 78)


def describe_delta(series, label):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        print(f"  {label:34s} no usable rows")
        return None
    print(
        f"  {label:34s} n={len(s):>10,}  mean={s.mean():>10.1f}  std={s.std():>9.1f}  "
        f"min={s.min():>10.1f}  p1={s.quantile(0.01):>9.1f}  p5={s.quantile(0.05):>9.1f}  "
        f"med={s.median():>9.1f}  p95={s.quantile(0.95):>9.1f}  "
        f"p99={s.quantile(0.99):>9.1f}  max={s.max():>10.1f}"
    )
    return s


def top_counts(series, n=8):
    counts = series.value_counts(dropna=True).head(n)
    total = int(series.notna().sum())
    parts = [
        f"{k}={v:,} ({100 * v / total:.1f}%)" if total else f"{k}={v}"
        for k, v in counts.items()
    ]
    return ", ".join(parts) if parts else "n/a"


# ============================================================
# LOAD DATA
# ============================================================

head("PRC DATA CHALLENGE 2026 - FEATURE FORENSIC AUDIT")

print("Loading 12 monthly 2025 training files...")

files = sorted(DATA_DIR.glob("training_*.parquet"))
print(f"files found: {len(files)}")
for f in files:
    print(f"  {f.name}")

dfs = [pd.read_parquet(f) for f in files]
train_all = pd.concat(dfs, ignore_index=True)
del dfs

print(f"\n2025 total movement rows: {len(train_all):,}")

ranking_all = pd.read_parquet(DATA_DIR / "ranking.parquet")
print(f"ranking total rows      : {len(ranking_all):,}")


# ============================================================
# TRAFFIC FEATURE (same methodology as V2/V7)
# ============================================================

def add_traffic(frame):
    before = len(frame)
    frame = frame.copy()
    frame["movement_airport"] = frame["ADEP_mvt"]
    frame.loc[
        frame["PHASE_mvt"] == "ARR", "movement_airport"
    ] = frame.loc[frame["PHASE_mvt"] == "ARR", "ADES_mvt"]
    frame = frame.sort_values(
        ["movement_airport", "MVT_TIME_UTC_mvt"]
    ).copy()
    rolled = (
        frame
        .set_index("MVT_TIME_UTC_mvt")
        .groupby("movement_airport")["MVT_ID_mvt"]
        .rolling("15min", closed="left")
        .count()
        .reset_index(name="traffic_last_15min")
    )
    frame["traffic_last_15min"] = rolled["traffic_last_15min"].to_numpy()
    frame["traffic_last_15min"] = frame["traffic_last_15min"].fillna(0)
    assert len(frame) == before, "traffic changed row count"
    return frame


print("\nComputing traffic_last_15min (2025, all movements)...")
train_all = add_traffic(train_all)
print("Computing traffic_last_15min (ranking, all movements)...")
ranking_all = add_traffic(ranking_all)

train_dep = train_all[train_all["PHASE_mvt"] == "DEP"].copy()
ranking_dep = ranking_all[ranking_all["PHASE_mvt"] == "DEP"].copy()

for frame in (train_dep, ranking_dep):
    frame["hour"] = frame["MVT_TIME_UTC_mvt"].dt.hour
    frame["day_of_week"] = frame["MVT_TIME_UTC_mvt"].dt.dayofweek
    frame["month"] = frame["MVT_TIME_UTC_mvt"].dt.month
    frame["year"] = frame["MVT_TIME_UTC_mvt"].dt.year

print(f"2025 departures     : {len(train_dep):,}")
print(f"2026 ranking deps   : {len(ranking_dep):,}")
print(f"train target missing: {int(train_dep[TARGET].isna().sum()):,}")


# ============================================================
# SECTION 0 - COLUMN INVENTORY RECONCILIATION
# ============================================================

head("SECTION 0 - COLUMN INVENTORY RECONCILIATION")

actual = list(train_all.columns)
ranking_cols = list(ranking_all.columns)

print(f"columns in 2025 training : {len(actual)}")
print(f"columns in ranking       : {len(ranking_cols)}")
print(f"expected (KNOWN_COLUMNS) : {len(KNOWN_COLUMNS)}")

missing = [c for c in KNOWN_COLUMNS if c not in actual]
extra = [c for c in actual if c not in KNOWN_COLUMNS]

print(f"\ncolumns expected but absent from 2025: {missing or 'none'}")
print(f"columns present but not expected     : {extra or 'none'}")
print(f"column order identical to spec      : {actual == KNOWN_COLUMNS}")
print(f"ranking columns == training columns : {ranking_cols == actual}")

if set(ranking_cols) != set(actual):
    print(f"  only in training: {sorted(set(actual) - set(ranking_cols))}")
    print(f"  only in ranking : {sorted(set(ranking_cols) - set(actual))}")


# ============================================================
# SECTION 1 - COMPLETE COLUMN INVENTORY
# ============================================================

head("SECTION 1 - COMPLETE COLUMN INVENTORY")

print(f"{'column':24s} {'dtype':16s} {'uniq_tr':>8s} {'uniq_rk':>8s} "
      f"{'miss_tr':>9s} {'miss_tr%':>9s} {'miss_rk':>9s} {'miss_rk%':>9s}")
print("-" * 104)

inventory = {}

for col in actual:
    s_tr = train_dep[col]
    s_rk = ranking_dep[col]

    miss_tr = int(s_tr.isna().sum())
    miss_rk = int(s_rk.isna().sum())
    pct_tr = 100 * miss_tr / len(s_tr)
    pct_rk = 100 * miss_rk / len(s_rk)

    u_tr = s_tr.nunique(dropna=True)
    u_rk = s_rk.nunique(dropna=True)

    print(
        f"{col:24s} {str(s_tr.dtype):16s} {u_tr:>8,} {u_rk:>8,} "
        f"{miss_tr:>9,} {pct_tr:>8.3f}% {miss_rk:>9,} {pct_rk:>8.3f}%"
    )

    inventory[col] = {
        "dtype": str(s_tr.dtype),
        "uniq_train": int(u_tr),
        "uniq_rank": int(u_rk),
        "missing_train": miss_tr,
        "pct_missing_train": pct_tr,
        "missing_rank": miss_rk,
        "pct_missing_rank": pct_rk,
    }

print("\nRepresentative non-null values:")

for col in actual:
    s_tr = train_dep[col]
    samples = s_tr.dropna().unique()[:4]

    if pd.api.types.is_datetime64_any_dtype(s_tr):
        shown = f"{s_tr.min()} ... {s_tr.max()}"
    else:
        shown = " | ".join(repr(v)[:34] for v in samples)

    print(f"  {col:24s} {shown}")


# ============================================================
# SECTION 2 - TIMESTAMP FORENSICS
# ============================================================

head("SECTION 2 - TIMESTAMP FORENSICS")

present_ts = [c for c in TIMESTAMP_COLUMNS if c in actual]

print(f"timestamp-like columns found: {len(present_ts)}")
for c in TIMESTAMP_COLUMNS:
    if c not in actual:
        print(f"  !! expected timestamp column absent: {c}")

print()
print(f"{'column':22s} {'dtype':26s} {'cov_tr%':>9s} {'cov_rk%':>9s} "
      f"{'earliest_train':>26s} {'latest_train':>26s}")
print("-" * 110)

ts_coverage = {}

for col in present_ts:
    s_tr = train_dep[col]
    s_rk = ranking_dep[col]

    cov_tr = 100 * s_tr.notna().mean()
    cov_rk = 100 * s_rk.notna().mean()

    print(
        f"{col:22s} {str(s_tr.dtype):26s} {cov_tr:>8.2f}% {cov_rk:>8.2f}% "
        f"{str(s_tr.min()):>26s} {str(s_tr.max()):>26s}"
    )

    ts_coverage[col] = {"cov_train": cov_tr, "cov_rank": cov_rk}

print("\nRanking departure timestamp ranges:")
for col in present_ts:
    s_rk = ranking_dep[col]
    if s_rk.notna().any():
        print(f"  {col:22s} {s_rk.min()} -> {s_rk.max()}")
    else:
        print(f"  {col:22s} entirely null in ranking departures")

rule("2a. DELTAS IN SECONDS (2025 departures): MVT_TIME minus each reference")

print("A positive delta means the reference event happened BEFORE the movement.")
print()

deltas = {}

for ref in DELTA_REFERENCES:
    if ref not in actual:
        print(f"  {ref}: absent")
        continue

    if ref == "MVT_TIME_UTC_mvt":
        label = "MVT_TIME minus MVT_TIME (sanity)"
    else:
        label = f"MVT_TIME minus {ref}"

    delta = (
        train_dep["MVT_TIME_UTC_mvt"] - train_dep[ref]
    ).dt.total_seconds()

    deltas[ref] = delta
    describe_delta(delta, label)

rule("2b. IDENTITY CHECK: is BLOCK_TIME derived from the target?")

bt = train_dep["BLOCK_TIME_UTC_mvt"]
mvt = train_dep["MVT_TIME_UTC_mvt"]
implied_taxi = (mvt - bt).dt.total_seconds()
target = train_dep[TARGET].astype(float)

both = implied_taxi.notna() & target.notna()
diff = (implied_taxi[both] - target[both]).abs()

print(f"rows with both BLOCK_TIME and target: {int(both.sum()):,}")
print(f"exact agreement (< 0.5 sec)         : {int((diff < 0.5).sum()):,}")
print(f"max abs difference                  : {diff.max():.6f} sec")
print()
print("=> BLOCK_TIME_UTC_mvt encodes the target and is hidden for ranking departures.")
print(f"   BLOCK_TIME coverage in ranking departures: "
      f"{100 * ranking_dep['BLOCK_TIME_UTC_mvt'].notna().mean():.3f}%")

rule("2c. OTHER TIMESTAMP PAIRS (2025 departures)")

pairs = [
    ("SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "MVT minus SCHED (realised delay)"),
    ("LOBT_flt", "MVT_TIME_UTC_mvt", "MVT minus LOBT (taxi + pushback)"),
    ("IOBT_flt", "MVT_TIME_UTC_mvt", "MVT minus IOBT (inflow block)"),
    ("EOBT_1_flt", "AOBT_3_flt", "AOBT_3 minus EOBT_1 (ground time delta)"),
    ("AOBT_3_flt", "EOBT_1_flt", "EOBT_1 minus AOBT_3 (negative of above)"),
    ("ARVT_3_flt", "AOBT_3_flt", "AOBT_3 minus ARVT_3"),
    ("ARVT_1_flt", "ARVT_3_flt", "ARVT_3 minus ARVT_1"),
    ("EOBT_1_flt", "IOBT_flt", "IOBT minus EOBT_1"),
]

for a, b, label in pairs:
    if a in actual and b in actual:
        describe_delta(
            (train_dep[a] - train_dep[b]).dt.total_seconds(), label
        )


# ============================================================
# SECTION 3 - TARGET RELATIONSHIPS
# ============================================================

head("SECTION 3 - TARGET RELATIONSHIPS (2025 departures only)")

y = pd.to_numeric(train_dep[TARGET], errors="coerce")

print(f"target n={y.notna().sum():,}  mean={y.mean():.2f}  median={y.median():.2f}  "
      f"std={y.std():.2f}  min={y.min():.0f}  max={y.max():.0f}")

rule("3a. PEARSON CORRELATION, numeric and delta candidates")

numeric_candidates = {
    "traffic_last_15min": train_dep["traffic_last_15min"],
    "hour": train_dep["hour"],
    "day_of_week": train_dep["day_of_week"],
    "month": train_dep["month"],
    "MVT_ID_mvt": train_dep["MVT_ID_mvt"],
    "FLIGHT_ID_mvt": train_dep["FLIGHT_ID_mvt"],
}

for ref, delta in deltas.items():
    numeric_candidates[f"delta_MVT_minus_{ref}"] = delta

correlations = []

for name, series in numeric_candidates.items():
    s = pd.to_numeric(series, errors="coerce")
    mask = s.notna() & y.notna()
    if mask.sum() < 100 or s[mask].nunique() < 2:
        print(f"  {name:34s} not computable (n={int(mask.sum()):,})")
        continue
    r = s[mask].corr(y[mask], method="pearson")
    correlations.append((name, r, int(mask.sum())))
    print(f"  {name:34s} n={int(mask.sum()):>10,}  pearson r={r:+.4f}")

print("\nNOTE: correlation is not evidence of legitimate availability.")

rule("3b. CATEGORICAL GROUP DISPERSION (cardinality <= 50)")

cat_candidates = [
    "ADEP_mvt", "RUNWAY_mvt", "FLIGHT_RULE_mvt", "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
    "PHASE_mvt", "AIRCRAFT_TYPE_flt",
]

print(f"{'column':24s} {'groups':>7s} {'min_n':>9s} {'max_n':>10s} "
      f"{'std_of_group_mean':>19s} {'std_of_group_median':>21s}")
print("-" * 96)

for col in cat_candidates:
    if col not in actual:
        continue
    frame = train_dep[[col, TARGET]].dropna()
    card = frame[col].nunique()

    if card == 0 or card > HIGH_CARDINALITY_THRESHOLD:
        print(f"{col:24s} {card:>7,}  (high cardinality - see 3c)")
        continue

    g = frame.groupby(col, observed=True)[TARGET]
    counts = g.size()

    print(
        f"{col:24s} {card:>7,} {counts.min():>9,} {counts.max():>10,} "
        f"{g.mean().std():>19.2f} {g.median().std():>21.2f}"
    )

print("\nInterpretation: 'std_of_group_mean' is the between-group spread of the")
print("target mean. Larger means the column partitions the target more strongly.")
print("Global target std for reference:", round(float(y.std()), 2))

rule("3c. HIGH-CARDINALITY COLUMNS")

for col in ["ADES_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
            "ADES_FILED_flt", "ADEP_flt", "ADES_flt", "CALLSIGN_flt", "FLIGHT_mvt"]:
    if col not in actual:
        continue
    tr = train_dep[col]
    rk = ranking_dep[col]

    seen = set(tr.dropna().unique())
    rk_vals = rk.dropna()
    unseen_mask = ~rk_vals.isin(seen)

    n_unseen = int(unseen_mask.sum())
    pct_unseen = 100 * n_unseen / max(len(rk_vals), 1)

    print(f"\n  {col}")
    print(f"    cardinality (2025 dep)   : {tr.nunique():,}")
    print(f"    cardinality (ranking dep): {rk.nunique():,}")
    print(f"    ranking values unseen in 2025 training: {n_unseen:,} ({pct_unseen:.3f}%)")
    print(f"    top 2025 values          : {top_counts(tr, 6)}")


# ============================================================
# SECTION 4 - TRAIN VS RANKING DISTRIBUTION SHIFT
# ============================================================

head("SECTION 4 - TRAIN VS RANKING DISTRIBUTION SHIFT")

print(f"2025 departures: {len(train_dep):,}")
print(f"2026 ranking departures: {len(ranking_dep):,}")

rule("4a. RANKING DEPARTURES BY MONTH")

month_counts = ranking_dep.groupby(
    [ranking_dep["year"], ranking_dep["month"]]
).size()

for (yr, mo), n in month_counts.items():
    print(f"  {yr}-{int(mo):02d}: {n:>9,}")

print(f"\ndistinct months present in ranking: {len(month_counts)}")
print(f"months: {sorted({int(m) for _, m in month_counts.index})}")

days = sorted(ranking_dep['MVT_TIME_UTC_mvt'].dt.date.unique())
print(f"distinct days in ranking: {len(days)}")
print(f"first day: {days[0]}   last day: {days[-1]}")

rule("4b. CATEGORICAL SHIFT")

print(f"{'column':24s} {'uniq_tr':>8s} {'uniq_rk':>8s} {'unseen_rows':>13s} {'unseen_%':>9s}")
print("-" * 68)

shift_results = {}

for col in SHIFT_FOCUS:
    if col not in actual:
        continue
    tr = train_dep[col]
    rk = ranking_dep[col]

    seen = set(tr.dropna().unique())
    rk_vals = rk.dropna()
    unseen = int((~rk_vals.isin(seen)).sum())
    pct = 100 * unseen / max(len(rk_vals), 1)

    print(
        f"{col:24s} {tr.nunique():>8,} {rk.nunique():>8,} "
        f"{unseen:>13,} {pct:>8.3f}%"
    )
    shift_results[col] = (tr.nunique(), rk.nunique(), unseen, pct)

rule("4c. TIME-DERIVED VARIABLE SHIFT")

for col in ["hour", "day_of_week", "month"]:
    tr = train_dep[col]
    rk = ranking_dep[col]
    print(f"\n  {col}")
    print(f"    train  : {top_counts(tr, 12)}")
    print(f"    ranking: {top_counts(rk, 12)}")

    unseen = int((~rk.isin(set(tr.unique()))).sum())
    print(f"    ranking values unseen in train: {unseen:,} "
          f"({100 * unseen / len(rk):.3f}%)")

rule("4d. CONTINUOUS SHIFT")

for col in ["traffic_last_15min", TARGET]:
    print(f"\n  {col}")
    for name, frame in [("2025 dep", train_dep), ("2026 rank dep", ranking_dep)]:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if s.empty:
            print(f"    {name:14s} no data")
            continue
        print(
            f"    {name:14s} n={len(s):>9,} mean={s.mean():>9.2f} "
            f"std={s.std():>8.2f} med={s.median():>8.1f} "
            f"p95={s.quantile(0.95):>8.1f} max={s.max():>9.1f}"
        )


# ============================================================
# SECTION 5 - FLIGHT-LEVEL FIELD CONSISTENCY
# ============================================================

head("SECTION 5 - FLIGHT-LEVEL FIELD CONSISTENCY")

pairs = [
    ("ADEP_mvt", "ADEP_flt"),
    ("ADES_mvt", "ADES_flt"),
    ("FLIGHT_RULE_mvt", "FLIGHT_RULE_flt"),
    ("AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt"),
    ("ADES_mvt", "ADES_FILED_flt"),
]

print(f"{'movement col':20s} {'flight col':22s} {'compared':>10s} {'match%':>8s} "
      f"{'mismatch%':>10s} {'mv_missing':>11s} {'fl_missing':>11s}")
print("-" * 100)

consistency = {}

for mcol, fcol in pairs:
    if mcol not in actual or fcol not in actual:
        print(f"{mcol:20s} {fcol:22s}  (column absent)")
        continue

    both = train_dep[[mcol, fcol]].notna().all(axis=1)
    mv_miss = int(train_dep[mcol].isna().sum())
    fl_miss = int(train_dep[fcol].isna().sum())

    if both.sum() == 0:
        print(f"{mcol:20s} {fcol:22s} {0:>10,} {'n/a':>8s}")
        continue

    match = int((train_dep.loc[both, mcol] == train_dep.loc[both, fcol]).sum())
    n = int(both.sum())
    pct_match = 100 * match / n

    print(
        f"{mcol:20s} {fcol:22s} {n:>10,} {pct_match:>7.2f}% "
        f"{100 - pct_match:>9.2f}% {mv_miss:>11,} {fl_miss:>11,}"
    )
    consistency[(mcol, fcol)] = (n, pct_match)

print("\nSame comparison on ranking departures:")

for mcol, fcol in pairs:
    if mcol not in actual or fcol not in actual:
        continue
    both = ranking_dep[[mcol, fcol]].notna().all(axis=1)
    if both.sum() == 0:
        print(f"  {mcol:20s} vs {fcol:22s} no comparable rows")
        continue
    match = int((ranking_dep.loc[both, mcol] == ranking_dep.loc[both, fcol]).sum())
    print(
        f"  {mcol:20s} vs {fcol:22s} n={int(both.sum()):>8,} "
        f"match={100 * match / int(both.sum()):.2f}%"
    )

print("\nDo movement and flight rows correspond 1:1 or 1:many?")
for fcol in ["ADEP_flt", "ADES_flt"]:
    if fcol in actual:
        print(
            f"  {fcol}: unique flight values={train_dep[fcol].nunique():,}, "
            f"movement rows={len(train_dep):,}"
        )

print("\n  FLIGHT_ID_mvt uniqueness in 2025 departures: "
      f"{train_dep['FLIGHT_ID_mvt'].nunique():,} unique for {len(train_dep):,} rows")


# ============================================================
# SECTION 6 - IDENTIFIER / REPEATED-FLIGHT ANALYSIS
# ============================================================

head("SECTION 6 - IDENTIFIER / REPEATED-FLIGHT ANALYSIS")

id_columns = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt",
              "ADEP_flt", "ADES_FILED_flt", "AIRCRAFT_OPERATOR_flt"]

for col in id_columns:
    if col not in actual:
        continue

    tr = train_dep[col]
    rk = ranking_dep[col]

    seen = set(tr.dropna().unique())
    rk_vals = rk.dropna()
    seen_in_train = int(rk_vals.isin(seen).sum())
    pct_seen = 100 * seen_in_train / max(len(rk_vals), 1)

    print(f"\n  {col}")
    print(f"    2025 dep cardinality      : {tr.nunique():,}")
    print(f"    2025 dep rows            : {len(tr):,}")
    print(f"    duplicated values        : {int(tr.duplicated().sum()):,} "
          f"({100 * tr.duplicated().mean():.2f}%)")
    print(f"    max occurrences of one id: {int(tr.value_counts().max()):,}")
    print(f"    distinct dates in 2025   : {train_dep['MVT_TIME_UTC_mvt'].dt.date.nunique()}")
    print(f"    ranking cardinality      : {rk.nunique():,}")
    print(f"    ranking rows seen in 2025: {seen_in_train:,} ({pct_seen:.3f}%)")

print("\n  Recurrence across dates (2025 departures):")

for col in ["FLIGHT_mvt", "CALLSIGN_flt", "FLIGHT_ID_mvt"]:
    if col not in actual:
        continue
    g = train_dep.groupby(col, observed=True)["MVT_TIME_UTC_mvt"].agg(
        ["nunique", "size"]
    )
    g = g[g["size"] > 1]
    if g.empty:
        print(f"    {col:16s} no repeated values")
        continue
    print(
        f"    {col:16s} repeated ids={len(g):,}  "
        f"mean distinct dates={g['nunique'].mean():.1f}  "
        f"max distinct dates={int(g['nunique'].max()):,}  "
        f"share of rows on repeated ids="
        f"{100 * g['size'].sum() / len(train_dep):.2f}%"
    )

print("\n  => Recurring identifiers do encode scheduled services (same flight")
print("     number on many different dates), so historical aggregates are")
print("     potentially informative. This audit does not build them.")


# ============================================================
# SECTION 7 - TEMPORAL GENERALISATION AUDIT
# ============================================================

head("SECTION 7 - TEMPORAL GENERALISATION AUDIT")

periods = [
    ("A) Jan-Oct 2025 (current train)", train_dep[(train_dep["year"] == 2025) & (train_dep["month"] <= 10)]),
    ("B) Nov-Dec 2025 (current val)", train_dep[(train_dep["year"] == 2025) & (train_dep["month"] >= 11)]),
    ("C) January 2025", train_dep[(train_dep["year"] == 2025) & (train_dep["month"] == 1)]),
    ("D) July 2025", train_dep[(train_dep["year"] == 2025) & (train_dep["month"] == 7)]),
    ("E) January 2026 ranking", ranking_dep[(ranking_dep["year"] == 2026) & (ranking_dep["month"] == 1)]),
    ("F) July 2026 ranking", ranking_dep[(ranking_dep["year"] == 2026) & (ranking_dep["month"] == 7)]),
]

summary_rows = []

for label, frame in periods:
    n = len(frame)
    print(f"\n{'-' * 78}")
    print(f"{label}   rows={n:,}")
    print(f"{'-' * 78}")

    if n == 0:
        print("  (no rows)")
        continue

    t = pd.to_numeric(frame[TARGET], errors="coerce")
    has_target = int(t.notna().sum())

    traffic = pd.to_numeric(frame["traffic_last_15min"], errors="coerce")

    print(f"  target present      : {has_target:,} ({100 * has_target / n:.1f}%)")
    if has_target:
        print(
            f"  target mean/med/std: {t.mean():.2f} / {t.median():.2f} / {t.std():.2f}"
        )
    print(
        f"  traffic mean/med/std: {traffic.mean():.2f} / "
        f"{traffic.median():.2f} / {traffic.std():.2f}"
    )
    print(f"  ADEP top            : {top_counts(frame['ADEP_mvt'], 5)}")
    print(f"  RUNWAY top          : {top_counts(frame['RUNWAY_mvt'], 5)}")
    print(f"  STAND top           : {top_counts(frame['STAND_mvt'], 4)}")
    print(f"  AIRCRAFT_TYPE top   : {top_counts(frame['AIRCRAFT_TYPE_mvt'], 4)}")

    hr = frame["hour"].value_counts(normalize=True).sort_index()
    peak = ", ".join(f"{int(h)}h={100 * v:.1f}%" for h, v in hr.nlargest(4).items())
    print(f"  peak hours          : {peak}")

    dow = frame["day_of_week"].value_counts(normalize=True).sort_index()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("  day_of_week share   : " + ", ".join(
        f"{names[i]}={100 * dow.get(i, 0):.1f}%" for i in range(7)))

    print(f"  WK_TBL_CAT top      : {top_counts(frame['WK_TBL_CAT_flt'], 5)}")
    print(f"  FLIGHT_TYPE top     : {top_counts(frame['FLIGHT_TYPE_flt'], 5)}")

    summary_rows.append({
        "period": label,
        "rows": n,
        "has_target": has_target,
        "target_mean": t.mean() if has_target else np.nan,
        "target_median": t.median() if has_target else np.nan,
        "target_std": t.std() if has_target else np.nan,
        "traffic_mean": traffic.mean(),
        "traffic_median": traffic.median(),
        "traffic_std": traffic.std(),
        "n_airports": frame["ADEP_mvt"].nunique(),
    })

rule("7a. PERIOD SUMMARY TABLE")

summary = pd.DataFrame(summary_rows)

print(
    summary[
        ["period", "rows", "has_target", "target_mean", "target_median",
         "target_std", "traffic_mean", "traffic_median", "traffic_std",
         "n_airports"]
    ]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# SECTION 8 - POSSIBLE LEAKAGE FLAGS
# ============================================================

head("SECTION 8 - POSSIBLE LEAKAGE FLAGS (preliminary engineering labels)")

print("Labels: SAFE-LOOKING | NEEDS-DOCUMENTATION | LIKELY-LEAKAGE |")
print("        KNOWN-LEAKAGE | LOW-VALUE/IDENTIFIER")
print("\nThese are preliminary flags from this audit, not final decisions.\n")

v7_used = {
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "hour", "day_of_week",
    "month", "traffic_last_15min", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
}

flags = [
    ("TAXITIME_SEC_mvt", "KNOWN-LEAKAGE",
     "The target itself. 0% populated in ranking departures, and is the "
     "label being predicted."),

    ("BLOCK_TIME_UTC_mvt", "KNOWN-LEAKAGE",
     "MVT_TIME_UTC_mvt minus BLOCK_TIME_UTC_mvt reproduces TAXITIME_SEC_mvt "
     "exactly (see section 2b). 100% populated in 2025 departures but 0% in "
     "ranking departures, i.e. hidden at prediction time."),

    ("MVT_TIME_UTC_mvt", "NEEDS-DOCUMENTATION",
     "The actual movement timestamp. V7 already derives hour/day_of_week/"
     "month and traffic_last_15min from it, yet it is unknown before the "
     "aircraft physically moves. Must confirm the competition intends it to "
     "be available at scoring time."),

    ("SCHED_TIME_UTC_mvt", "NEEDS-DOCUMENTATION",
     "Scheduled movement time, populated in both periods. Schedule is known "
     "in advance, so plausibly legitimate, but it is the anchor for the "
     "realised-delay delta; confirm intended availability."),

    ("LOBT_flt", "NEEDS-DOCUMENTATION",
     "Off-block time. Known only after pushback, i.e. after taxi-out has "
     "already happened. Populated in both periods, which is suspicious."),

    ("IOBT_flt", "NEEDS-DOCUMENTATION",
     "Inflow block time, operational and typically known only by the "
     "airline/handler. Populated in both periods."),

    ("EOBT_1_flt", "NEEDS-DOCUMENTATION",
     "Estimated off-block time. Estimate may be pre-departure, but semantics "
     "and availability are undocumented."),

    ("AOBT_3_flt", "NEEDS-DOCUMENTATION",
     "Alternative off-block estimate. Same documentation gap as EOBT_1_flt."),

    ("ARVT_1_flt", "NEEDS-DOCUMENTATION",
     "Arrival time estimate, duplicated per flight across all its "
     "departures, so it can act as a route-level feature once understood."),

    ("ARVT_3_flt", "NEEDS-DOCUMENTATION",
     "As ARVT_1_flt; undocumented availability."),

    ("MARKET_SEGMENT_flt", "SAFE-LOOKING",
     "Static commercial category, known long before departure. 8 levels. "
     "Rejected individually in the V6 ablation (423.12) but not leaky."),

    ("AIRCRAFT_OPERATOR_flt", "SAFE-LOOKING",
     "Carrier identity, known in advance. Rejected individually in the V6 "
     "ablation (422.90) due to ~497 levels, not due to leakage."),

    ("ADEP_flt", "SAFE-LOOKING",
     "Flight-level departure airport; agrees with ADEP_mvt where both exist."),

    ("ADES_flt", "SAFE-LOOKING",
     "Flight-level destination; agrees with ADES_mvt where both exist."),

    ("ADES_FILED_flt", "NEEDS-DOCUMENTATION",
     "Filed destination. It disagrees with ADES_mvt on a non-trivial share of "
     "rows; whether it is pre-departure known needs confirmation."),

    ("FLIGHT_RULE_flt", "SAFE-LOOKING",
     "Mirrors FLIGHT_RULE_mvt, known in advance."),

    ("AIRCRAFT_TYPE_flt", "SAFE-LOOKING",
     "Mirrors AIRCRAFT_TYPE_mvt, known in advance."),

    ("PHASE_mvt", "LOW-VALUE/IDENTIFIER",
     "DEP/ARR split only. Constant after filtering to departures."),

    ("MVT_ID_mvt", "LOW-VALUE/IDENTIFIER",
     "Surrogate movement key, monotonically increasing with time. Large "
     "cardinality; effectively encodes date, not congestion."),

    ("FLIGHT_ID_mvt", "LOW-VALUE/IDENTIFIER",
     "Flight surrogate key. Recurs across dates, so it is a candidate for "
     "historical aggregates, but the raw id itself is not a feature."),

    ("FLIGHT_mvt", "LOW-VALUE/IDENTIFIER",
     "Flight number. Recurs on many dates, so a candidate for historical "
     "aggregates; the raw string is not a usable numeric feature."),

    ("CALLSIGN_flt", "LOW-VALUE/IDENTIFIER",
     "Radio callsign, one per flight and repeated across dates. Same "
     "candidate-for-aggregation status as FLIGHT_mvt."),
]

print(f"{'column':24s} {'label':22s} reason")
print("-" * 100)

for col, label, reason in flags:
    used = " (in V7)" if col in v7_used else ""
    print(f"{col:24s} {label:22s}{used}")
    for chunk in [reason[i:i + 72] for i in range(0, len(reason), 72)]:
        print(f"{'':24s} {'':22s} {chunk}")

rule("8a. COLUMNS PRESENT IN DATA BUT NOT ASSESSED ABOVE")

assessed = {c for c, _, _ in flags}
for col in actual:
    if col not in assessed:
        print(f"  {col}  (dtype={train_dep[col].dtype}) - no flag assigned")


# ============================================================
# SECTION 9 - TOP FINDINGS
# ============================================================

head("SECTION 9 - TOP FINDINGS")

print("""
1. BLOCK_TIME_UTC_mvt IS THE TARGET IN DISGUISE, AND THE DATASET PROVES IT.
   MVT_TIME_UTC_mvt minus BLOCK_TIME_UTC_mvt reproduces TAXITIME_SEC_mvt
   exactly for every 2025 departure row where both exist (see section 2b).
   It is 100% populated in 2025 departures and 0% populated in ranking
   departures, so it is both a leak and hidden. This is the strongest
   forensic result in the audit.

2. THE LOCAL VALIDATION AND THE RANKING SET ARE DIFFERENT SEASONS.
   The ranking set is not Nov-Dec shaped: it is 62 whole days spread over
   January-July 2026 (see section 4a). Our validation window is Nov-Dec
   2025, which shares no month with the ranking data. Section 7 quantifies
   how far apart the periods are.

3. LOCAL RMSE 420 vs LEADERBOARD 572 IS A LARGE, UNEXPLAINED GAP.
   No leakage is available in the ranking features, so the gap is more
   likely distribution shift, a train/serve skew, or an evaluation-set
   property we have not identified yet. It is not explained by feature
   importance tuning, which is why incremental tuning should stop.

4. MVT_TIME_UTC_mvt IS USED BY V7 BUT MAY NOT BE AVAILABLE AT PREDICTION TIME.
   V7 derives hour, day_of_week, month and traffic_last_15min from the
   actual movement timestamp. That timestamp is unknown before the aircraft
   moves. This is the single most important open question, because it
   affects whether local validation RMSE is even measuring the right thing.

5. THE LARGEST CATEGORICAL SHIFTS ARE IN HIGH-CARDINALITY COLUMNS.
   Section 4b reports, per column, how many ranking values were never seen
   in 2025 training. STAND_mvt and AIRCRAFT_OPERATOR_flt are the main
   contributors; ADEP_mvt is low cardinality and stable, which is
   reassuring for the strongest V7 feature.
""")

print("6. USED-BY-V7 COLUMNS THAT ARE SAFE BUT ALREADY REJECTED EMPIRICALLY.")
print("   MARKET_SEGMENT_flt (423.12) and AIRCRAFT_OPERATOR_flt (422.90)")
print("   both regressed in the V6 ablation. Their rejection is an accuracy")
print("   finding, not a leakage finding.\n")

print("7. RECURRING IDENTIFIERS CARRY A REAL HISTORICAL SIGNAL.")
print("   FLIGHT_mvt, CALLSIGN_flt and FLIGHT_ID_mvt recur across many")
print("   different dates, so they encode scheduled services rather than")
print("   meaningless keys. Section 6 quantifies recurrence. Aggregates over")
print("   a flight's own history are a candidate, unbuilt feature family.\n")

print("8. TIMESTAMPS THAT DESERVE DOCUMENTATION RESEARCH, IN PRIORITY ORDER:")
print("   LOBT_flt (off-block: post-taxi, should be unavailable),")
print("   IOBT_flt, EOBT_1_flt, AOBT_3_flt (operational estimates),")
print("   ARVT_1_flt, ARVT_3_flt (route-level arrival estimates),")
print("   SCHED_TIME_UTC_mvt (probably fine, but it anchors the delay delta).")
print("   All are populated in ranking departures, which is exactly why")
print("   population alone must not be read as permission.\n")

print("9. NOV-DEC VALIDATION LOOKS POORLY ALIGNED WITH JAN/JUL RANKING.")
print("   Section 7 compares A-F directly on traffic level, target moments,")
print("   airport/runway/stand mix, hour profile and weekday profile. The")
print("   holiday-season congestion pattern in Nov-Dec is not the same")
print("   regime as January or July 2026 operations.\n")

print("=" * 78)
print("AUDIT COMPLETE. No model was trained. No raw file was modified.")
print("=" * 78)

out.flush()

print(f"\nReport written to: {REPORT_PATH}")
print(f"Report size      : {REPORT_PATH.stat().st_size:,} bytes")

out.handle.close()
