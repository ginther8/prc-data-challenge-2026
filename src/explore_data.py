from pathlib import Path

import pandas as pd


# Path to the folder containing the competition data
DATA_DIR = Path("data/raw")

# Find all 12 monthly training files
training_files = sorted(DATA_DIR.glob("training_*.parquet"))

print(f"Found {len(training_files)} training files:")

for file in training_files:
    print(file.name)


print("\nLoading training data...")

training_dfs = []

for file in training_files:
    df = pd.read_parquet(file)
    training_dfs.append(df)

train = pd.concat(training_dfs, ignore_index=True)

print("\nCombined training dataset:")
print(f"Rows: {len(train):,}")
print(f"Columns: {len(train.columns)}")

print("\nMovement phases:")
print(train["PHASE_mvt"].value_counts())

print("\nMovement phases (%):")
print(train["PHASE_mvt"].value_counts(normalize=True) * 100)

# Focus on departures
departures = train[train["PHASE_mvt"] == "DEP"].copy()
departures["TAXITIME_SEC_mvt"] = departures["TAXITIME_SEC_mvt"].astype(int)

print("\nDeparture rows:")
print(f"{len(departures):,}")

print("\nTaxi-out time statistics:")
print(departures["TAXITIME_SEC_mvt"].describe().round().astype(int))

print("\nSuspicious taxi-out times:")

print("Negative or zero:")
print((departures["TAXITIME_SEC_mvt"] <= 0).sum())

print("\nOver 1 hour:")
print((departures["TAXITIME_SEC_mvt"] > 3600).sum())

print("\nOver 2 hours:")
print((departures["TAXITIME_SEC_mvt"] > 7200).sum())


print("\n10 longest taxi-out times:")

columns_to_show = [
    "FLIGHT_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",
]

print(
    departures
    .nlargest(10, "TAXITIME_SEC_mvt")[columns_to_show]
    .to_string(index=False)
)

print("\nTaxi-out time by airport:")

airport_stats = (
    departures
    .groupby("ADEP_mvt")["TAXITIME_SEC_mvt"]
    .agg(["count", "mean", "median"])
    .round()
    .astype(int)
    .sort_values("median", ascending=False)
)

print(airport_stats)

# ============================================================
# 1. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = train.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print(missing)


# ============================================================
# 2. DEPARTURE TAXI-TIME DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TAXI-OUT DISTRIBUTION")
print("=" * 70)

print(
    departures["TAXITIME_SEC_mvt"]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    .round()
    .astype(int)
)

print("\nPotential anomalies:")
print("<= 0 sec:   ", (departures["TAXITIME_SEC_mvt"] <= 0).sum())
print("> 1 hour:   ", (departures["TAXITIME_SEC_mvt"] > 3600).sum())
print("> 2 hours:  ", (departures["TAXITIME_SEC_mvt"] > 7200).sum())


# ============================================================
# 3. TAXI TIME BY AIRPORT
# ============================================================

print("\n" + "=" * 70)
print("TAXI-OUT BY AIRPORT")
print("=" * 70)

airport_stats = (
    departures
    .groupby("ADEP_mvt")["TAXITIME_SEC_mvt"]
    .agg(["count", "mean", "median", "std"])
    .round()
    .astype(int)
    .sort_values("median", ascending=False)
)

print(airport_stats)


# ============================================================
# 4. RUNWAYS PER AIRPORT
# ============================================================

print("\n" + "=" * 70)
print("RUNWAYS BY AIRPORT")
print("=" * 70)

for airport in sorted(departures["ADEP_mvt"].dropna().unique()):

    airport_data = departures[
        departures["ADEP_mvt"] == airport
    ]

    runway_stats = (
        airport_data
        .groupby("RUNWAY_mvt")["TAXITIME_SEC_mvt"]
        .agg(["count", "median"])
        .round()
        .astype(int)
        .sort_values("count", ascending=False)
    )

    print(f"\n{airport}")
    print(runway_stats)


# ============================================================
# 5. MOST COMMON STAND + RUNWAY COMBINATIONS
# ============================================================

print("\n" + "=" * 70)
print("TOP STAND + RUNWAY COMBINATIONS")
print("=" * 70)

stand_runway = (
    departures
    .groupby(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])
    ["TAXITIME_SEC_mvt"]
    .agg(["count", "median"])
    .reset_index()
)

# Ignore combinations with very few examples
stand_runway = stand_runway[stand_runway["count"] >= 100]

for airport in sorted(departures["ADEP_mvt"].dropna().unique()):

    result = (
        stand_runway[stand_runway["ADEP_mvt"] == airport]
        .sort_values("count", ascending=False)
        .head(10)
    )

    print(f"\n{airport}")
    print(result.to_string(index=False))


# ============================================================
# 6. TAXI TIME BY HOUR OF DAY
# ============================================================

print("\n" + "=" * 70)
print("TAXI-OUT BY HOUR")
print("=" * 70)

departures["hour"] = departures["MVT_TIME_UTC_mvt"].dt.hour

hour_stats = (
    departures
    .groupby("hour")["TAXITIME_SEC_mvt"]
    .agg(["count", "median"])
    .round()
    .astype(int)
)

print(hour_stats)


# ============================================================
# 7. TAXI TIME BY MONTH
# ============================================================

print("\n" + "=" * 70)
print("TAXI-OUT BY MONTH")
print("=" * 70)

departures["month"] = departures["MVT_TIME_UTC_mvt"].dt.month

month_stats = (
    departures
    .groupby("month")["TAXITIME_SEC_mvt"]
    .agg(["count", "mean", "median"])
    .round()
    .astype(int)
)

print(month_stats)


# ============================================================
# 8. AIRCRAFT TYPES
# ============================================================

print("\n" + "=" * 70)
print("TOP AIRCRAFT TYPES")
print("=" * 70)

aircraft_stats = (
    departures
    .groupby("AIRCRAFT_TYPE_mvt")["TAXITIME_SEC_mvt"]
    .agg(["count", "median"])
    .round()
    .astype(int)
    .sort_values("count", ascending=False)
    .head(20)
)

print(aircraft_stats)


# ============================================================
# 9. TRAINING DATE RANGE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING DATE RANGE")
print("=" * 70)

print("First movement:", train["MVT_TIME_UTC_mvt"].min())
print("Last movement: ", train["MVT_TIME_UTC_mvt"].max())


# ============================================================
# 10. RANKING DATA
# ============================================================

print("\n" + "=" * 70)
print("RANKING DATA")
print("=" * 70)

ranking = pd.read_parquet(DATA_DIR / "ranking.parquet")

print(f"Rows: {len(ranking):,}")
print(f"Columns: {len(ranking.columns)}")

ranking_departures = ranking[
    ranking["PHASE_mvt"] == "DEP"
].copy()

print(f"Departure predictions required: {len(ranking_departures):,}")

print("\nRanking date range:")
print("First:", ranking["MVT_TIME_UTC_mvt"].min())
print("Last: ", ranking["MVT_TIME_UTC_mvt"].max())

print("\nRanking departures by airport:")
print(ranking_departures["ADEP_mvt"].value_counts())


# ============================================================
# 11. TRAIN VS RANKING COLUMN COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TRAIN vs RANKING COLUMNS")
print("=" * 70)

train_columns = set(train.columns)
ranking_columns = set(ranking.columns)

print("Only in training:")
print(sorted(train_columns - ranking_columns))

print("\nOnly in ranking:")
print(sorted(ranking_columns - train_columns))


# ============================================================
# 12. RANKING DEPARTURE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("RANKING DEPARTURE MISSING VALUES")
print("=" * 70)

ranking_missing = ranking_departures.isna().sum()
ranking_missing = ranking_missing[ranking_missing > 0].sort_values(
    ascending=False
)

print(ranking_missing)


# ============================================================
# 13. UNIQUE VALUES / CARDINALITY
# ============================================================

print("\n" + "=" * 70)
print("UNIQUE VALUES IN IMPORTANT FEATURES")
print("=" * 70)

important_features = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt",
]

for column in important_features:
    print(f"{column:25s}: {departures[column].nunique():,}")


# ============================================================
# 14. EXTREME TAXI TIMES BY AIRPORT
# ============================================================

print("\n" + "=" * 70)
print("TAXI TIMES > 1 HOUR BY AIRPORT")
print("=" * 70)

extreme = departures[
    departures["TAXITIME_SEC_mvt"] > 3600
]

print(extreme["ADEP_mvt"].value_counts())


print("\nEDA COMPLETE")