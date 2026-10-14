"""Error decomposition for the V7 January and July seasonal holdouts.

Reproduces the V7 TEST B (January holdout) and TEST C (July holdout) models
exactly as in diagnose_v7_seasonal_validation.py, then decomposes their
validation errors. DIAGNOSTIC ONLY.

Constraints honoured:
- V7 feature list, categorical list, traffic definition, time features and
  CatBoost configuration are copied unchanged from V7.
- Validation targets are NEVER used to build prediction features. Bins and
  groupings below are applied after prediction, for analysis only.
- No new model, no tuning, no V8, no submission, no writes outside reports/.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error


# ============================================================
# REPORT TEE (print-compatible: handles every write() call)
# ============================================================

REPORT_PATH = Path("reports") / "v7_error_analysis.txt"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

_report_file = REPORT_PATH.open("w", encoding="utf-8", newline="\n")
_real_stdout = sys.stdout


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = Tee(_real_stdout, _report_file)


def rule(char="-", width=100):
    print(char * width)


def header(title, char="#", width=100):
    print()
    print(char * width)
    print(title)
    print(char * width)


# ============================================================
# V7 FEATURE DEFINITIONS (UNCHANGED)
# ============================================================

FEATURES = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt",
    "hour",
    "day_of_week",
    "month",
    "traffic_last_15min",
    "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt",
]

CATEGORICAL_FEATURES = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt",
    "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt",
]

TARGET = "TAXITIME_SEC_mvt"

# Diagnostic context columns, carried through for analysis only.
CARRY = [
    "MVT_ID_mvt",
    "FLIGHT_mvt",
    "CALLSIGN_flt",
    "SCHED_TIME_UTC_mvt",
    "MVT_TIME_UTC_mvt",
]

EXPECTED = {"JANUARY": 519.90, "JULY": 630.63}

BIN_EDGES = [-np.inf, 600, 900, 1200, 1800, 2700, 3600, 5400, np.inf]
BIN_LABELS = [
    "< 10 min",
    "10-15 min",
    "15-20 min",
    "20-30 min",
    "30-45 min",
    "45-60 min",
    "60-90 min",
    "> 90 min",
]

THRESHOLDS = [1800, 2700, 3600, 5400, 7200]
CONCENTRATION_PCTS = [0.1, 0.5, 1, 2, 5, 10, 20]
EXCLUDE_PCTS = [0.1, 0.5, 1, 5]


# ============================================================
# LOAD AND PREPARE (IDENTICAL TO V7)
# ============================================================

print("Loading 2025 training data...")

DATA_DIR = Path("data/raw")
training_files = sorted(DATA_DIR.glob("training_*.parquet"))
print(f"files found: {len(training_files)}")

dfs = [pd.read_parquet(f) for f in training_files]
data = pd.concat(dfs, ignore_index=True)
del dfs
print(f"Total movements: {len(data):,}")


# --- congestion feature, identical to V7 ---
data["movement_airport"] = data["ADEP_mvt"]

data.loc[
    data["PHASE_mvt"] == "ARR",
    "movement_airport"
] = data.loc[
    data["PHASE_mvt"] == "ARR",
    "ADES_mvt"
]

data = data.sort_values(
    ["movement_airport", "MVT_TIME_UTC_mvt"]
).copy()

traffic_15min = (
    data
    .set_index("MVT_TIME_UTC_mvt")
    .groupby("movement_airport")["MVT_ID_mvt"]
    .rolling("15min", closed="left")
    .count()
    .reset_index(name="traffic_last_15min")
)

data["traffic_last_15min"] = (
    traffic_15min["traffic_last_15min"].to_numpy()
)

data["traffic_last_15min"] = (
    data["traffic_last_15min"]
    .fillna(0)
)

del traffic_15min, data["movement_airport"]


# --- keep departures ---
departures = data[data["PHASE_mvt"] == "DEP"].copy()
del data
print(f"Departure rows: {len(departures):,}")

assert len(departures) == 2_085_047, (
    f"Unexpected departure count: {len(departures):,}"
)


# --- feature engineering, identical to V7 ---
departures["hour"] = departures["MVT_TIME_UTC_mvt"].dt.hour
departures["day_of_week"] = departures["MVT_TIME_UTC_mvt"].dt.dayofweek
departures["month"] = departures["MVT_TIME_UTC_mvt"].dt.month

for column in CATEGORICAL_FEATURES:
    departures[column] = (
        departures[column]
        .fillna("UNKNOWN")
        .astype(str)
    )

departures["year"] = departures["MVT_TIME_UTC_mvt"].dt.year
assert set(FEATURES).isdisjoint({"year"}), "year must not be a feature"


# ============================================================
# REPRODUCE THE TWO DIAGNOSTIC MODELS
# ============================================================

is_2025 = departures["year"] == 2025

SPLITS = [
    ("JANUARY", "January 2025 holdout",
     departures[is_2025 & (departures["month"] != 1)],
     departures[is_2025 & (departures["month"] == 1)]),
    ("JULY", "July 2025 holdout",
     departures[is_2025 & (departures["month"] != 7)],
     departures[is_2025 & (departures["month"] == 7)]),
]

frames = {}

for name, subtitle, train, validation in SPLITS:

    header(f"REPRODUCING V7 MODEL - {subtitle}")

    train_ids = set(train["MVT_ID_mvt"].to_numpy())
    val_ids = set(validation["MVT_ID_mvt"].to_numpy())
    overlap = train_ids & val_ids
    assert len(overlap) == 0, (
        f"{name}: {len(overlap):,} validation IDs leaked into training"
    )

    print(f"Training rows           : {len(train):,}")
    print(f"Validation rows         : {len(validation):,}")
    print(f"ID overlap train/valid  : {len(overlap):,}")

    model = CatBoostRegressor(
        iterations=1000,
        depth=8,
        learning_rate=0.1,
        loss_function="RMSE",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        use_best_model=True,
    )

    model.fit(
        train[FEATURES],
        train[TARGET],
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(validation[FEATURES], validation[TARGET]),
    )

    pred = model.predict(validation[FEATURES])

    rmse = root_mean_squared_error(validation[TARGET], pred)
    print(f"RMSE                    : {rmse:.2f} sec")
    print(f"Best iteration          : {model.get_best_iteration()}")

    # fidelity check against the recorded diagnostic
    delta = abs(rmse - EXPECTED[name])
    print(f"Expected (diagnostic)   : {EXPECTED[name]:.2f}  "
          f"(delta {delta:.4f})")
    assert delta < 0.05, (
        f"{name}: reproduction failed, delta {delta:.4f}"
    )
    print("Reproduction verified.")

    frame = validation[[*FEATURES, *CARRY, TARGET]].copy()
    frame["pred"] = pred
    frame["resid"] = frame["pred"] - frame[TARGET]
    frame["abs_err"] = frame["resid"].abs()
    frame["sq_err"] = frame["resid"] ** 2

    frames[name] = frame

    del model, train, validation

gc.collect()


# ============================================================
# HELPERS
# ============================================================

def rmse_of(sq):
    return float(np.sqrt(np.mean(np.asarray(sq, dtype="float64"))))


def group_metrics(frame, key, min_rows=1):
    """Per-group metrics via aggregation only (no groupby.apply)."""
    g = frame.groupby(key, observed=True)
    t = g.agg(
        rows=("sq_err", "size"),
        actual_mean=(TARGET, "mean"),
        actual_median=(TARGET, "median"),
        actual_std=(TARGET, "std"),
        pred_mean=("pred", "mean"),
        pred_median=("pred", "median"),
        pred_std=("pred", "std"),
        mae=("abs_err", "mean"),
        bias=("resid", "mean"),
        mean_sq=("sq_err", "mean"),
        total_sq=("sq_err", "sum"),
    )
    t = t[t["rows"] >= min_rows]
    t["rmse"] = np.sqrt(t["mean_sq"])
    return t


def print_table(title, t, cols, fmts, sort_by=None, ascending=False):
    print()
    print(title)
    if sort_by is not None:
        t = t.sort_values(sort_by, ascending=ascending)
    # The first requested field is usually the group key, which lives in the
    # index. Materialise it as a real column so lookups by name succeed.
    first = next(iter(fmts))
    if first not in t.columns and t.index.name == first:
        t = t.reset_index()
    head = " ".join(f"{c:>{w}s}" for c, w in fmts.items())
    print(head)
    rule("-", len(head))
    # to_dict("records") preserves per-column dtypes; iterrows() would upcast
    # integer columns to float in all-numeric tables.
    for row in t.to_dict("records"):
        cells = []
        for c, w in fmts.items():
            v = row[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append(f"{str(v):>{w}s}")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):>{w}d}")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:>{w}.2f}")
            else:
                cells.append(f"{str(v):>{w}s}")
        print(" ".join(cells))
    return t


# ============================================================
# 1. SQUARED-ERROR CONTRIBUTION BY AIRPORT
# ============================================================

for name, _ in [(s[0], s[1]) for s in SPLITS]:
    frame = frames[name]
    total_sq = frame["sq_err"].sum()
    n = len(frame)

    header(f"1. SQUARED-ERROR CONTRIBUTION BY AIRPORT - {name}")

    t = group_metrics(frame, "ADEP_mvt")
    t["pct_sq"] = 100 * t["total_sq"] / total_sq
    t["pct_rows"] = 100 * t["rows"] / n

    print_table(
        "Per-airport error decomposition (sorted by total squared error)",
        t,
        None,
        {
            "ADEP_mvt": 8, "rows": 9, "pct_rows": 8, "pct_sq": 8,
            "rmse": 10, "mae": 10, "bias": 10,
            "actual_mean": 11, "actual_median": 13, "actual_std": 11,
            "pred_mean": 10, "pred_median": 12, "pred_std": 10,
            "total_sq": 15,
        },
        sort_by="total_sq",
    )

    print(f"\nTotal squared error      : {total_sq:,.0f}")
    print(f"Overall RMSE             : {rmse_of(frame['sq_err']):.2f}")
    print(f"Overall MAE              : {frame['abs_err'].mean():.2f}")
    print(f"Overall bias             : {frame['resid'].mean():.2f}")

    # cumulative contribution
    cum = t.sort_values("total_sq", ascending=False)
    cum["cum_pct_sq"] = cum["pct_sq"].cumsum()
    print("\nCumulative share of total squared error (worst first):")
    for airport, row in cum.iterrows():
        print(f"  {airport:8s} {row['total_sq']:>16,.0f}  "
              f"{row['pct_sq']:>6.2f}%  cum {row['cum_pct_sq']:>6.2f}%")

    # leave-one-airport-out
    print("\nRMSE excluding each airport (one at a time):")
    for airport in t.sort_values("total_sq", ascending=False).index:
        kept = frame[frame["ADEP_mvt"] != airport]
        print(f"  without {airport:8s} n={len(kept):>9,}  "
              f"RMSE={rmse_of(kept['sq_err']):>8.2f}  "
              f"(delta {rmse_of(kept['sq_err']) - rmse_of(frame['sq_err']):+8.2f})")

    print("\nTargeted exclusions:")
    for label, drop in [
        ("excluding LIRF", ["LIRF"]),
        ("excluding LFPG", ["LFPG"]),
        ("excluding LIRF + LFPG", ["LIRF", "LFPG"]),
    ]:
        kept = frame[~frame["ADEP_mvt"].isin(drop)]
        if len(kept) == 0:
            print(f"  {label:24s} -> no rows")
            continue
        print(f"  {label:24s} n={len(kept):>9,}  "
              f"RMSE={rmse_of(kept['sq_err']):>8.2f}  "
              f"(delta {rmse_of(kept['sq_err']) - rmse_of(frame['sq_err']):+8.2f})  "
              f"sq_err={kept['sq_err'].sum():,.0f}")


# ============================================================
# 2. ERROR BY ACTUAL TAXI-TIME RANGE
# ============================================================

for name, _ in [(s[0], s[1]) for s in SPLITS]:
    frame = frames[name]
    total_sq = frame["sq_err"].sum()
    n = len(frame)

    header(f"2. ERROR BY ACTUAL TAXI-TIME RANGE - {name}")

    d = frame.copy()
    d["tbin"] = pd.cut(
        d[TARGET], bins=BIN_EDGES, labels=BIN_LABELS, right=False,
    )

    g = d.groupby("tbin", observed=False)
    t = g.agg(
        rows=("sq_err", "size"),
        actual_mean=(TARGET, "mean"),
        pred_mean=("pred", "mean"),
        mae=("abs_err", "mean"),
        mean_sq=("sq_err", "mean"),
        total_sq=("sq_err", "sum"),
    )
    t["rmse"] = np.sqrt(t["mean_sq"])
    t["pct_rows"] = 100 * t["rows"] / n
    t["pct_sq"] = 100 * t["total_sq"] / total_sq

    print_table(
        "Error by actual taxi-time bin",
        t,
        None,
        {
            "tbin": 11, "rows": 9, "pct_rows": 8,
            "rmse": 10, "mae": 10,
            "actual_mean": 11, "pred_mean": 10,
            "total_sq": 15, "pct_sq": 8,
        },
    )

    print(f"\nTotal squared error      : {total_sq:,.0f}")

    over = d[d[TARGET] > 1800]
    print(f"Rows with actual > 30 min (>1800 s): {len(over):,} "
          f"({100 * len(over) / n:.2f}% of rows), "
          f"{100 * over['sq_err'].sum() / total_sq:.2f}% of squared error")
    over45 = d[d[TARGET] > 2700]
    print(f"Rows with actual > 45 min (>2700 s): {len(over45):,} "
          f"({100 * len(over45) / n:.2f}% of rows), "
          f"{100 * over45['sq_err'].sum() / total_sq:.2f}% of squared error")
    over60 = d[d[TARGET] > 3600]
    print(f"Rows with actual > 60 min (>3600 s): {len(over60):,} "
          f"({100 * len(over60) / n:.2f}% of rows), "
          f"{100 * over60['sq_err'].sum() / total_sq:.2f}% of squared error")


# ============================================================
# 3. ERROR CONCENTRATION
# ============================================================

for name, _ in [(s[0], s[1]) for s in SPLITS]:
    frame = frames[name]

    header(f"3. ERROR CONCENTRATION - {name}")

    s = frame["sq_err"].sort_values(ascending=False)
    total = s.sum()
    n = len(s)
    base = rmse_of(s)

    print(f"Rows: {n:,}   Total squared error: {total:,.0f}   "
          f"Overall RMSE: {base:.2f}")
    print()
    print(f"{'worst %':>9s} {'rows':>9s} {'pct rows':>10s} "
          f"{'pct of sq err':>15s} {'cum pct':>9s} "
          f"{'RMSE excl':>11s} {'delta':>9s}")
    rule("-", 80)

    for pct in CONCENTRATION_PCTS:
        k = max(1, int(round(n * pct / 100)))
        share = 100 * s.iloc[:k].sum() / total
        rest = total - s.iloc[:k].sum()
        new_rmse = float(np.sqrt(rest / (n - k))) if n - k > 0 else float("nan")
        print(f"{pct:>8.1f}% {k:>9,} {100 * k / n:>9.2f}% "
              f"{share:>14.2f}% {share:>8.2f}% "
              f"{new_rmse:>11.2f} {new_rmse - base:>+9.2f}")

    print("\nRMSE after excluding the worst rows (diagnostic only):")
    for pct in EXCLUDE_PCTS:
        k = max(1, int(round(n * pct / 100)))
        rest = total - s.iloc[:k].sum()
        new_rmse = float(np.sqrt(rest / (n - k)))
        print(f"  exclude worst {pct:>4.1f}% ({k:>6,} rows): "
              f"RMSE {base:.2f} -> {new_rmse:.2f}  "
              f"({new_rmse - base:+.2f})")


# ============================================================
# 4. LARGE-ERROR CASES
# ============================================================

LARGE_FIELDS = [
    "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
    "FLIGHT_mvt", "CALLSIGN_flt", "ADES_mvt", "hour",
    "traffic_last_15min", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
]

csv_rows = []

for name, _ in [(s[0], s[1]) for s in SPLITS]:
    frame = frames[name]

    header(f"4. LARGE-ERROR CASES (top 100 by absolute error) - {name}")

    top = frame.reindex(
        frame["abs_err"].sort_values(ascending=False).index
    ).head(100).copy()
    top.insert(0, "period", name)

    csv_rows.append(top[["period", *LARGE_FIELDS, TARGET, "pred", "resid",
                         "abs_err", "sq_err"]])

    print(f"Inspected rows           : {len(top)}")
    print(f"Their squared error      : {top['sq_err'].sum():,.0f} "
          f"({100 * top['sq_err'].sum() / frame['sq_err'].sum():.2f}% of period total)")
    print(f"Actual taxi time  min/med/max : "
          f"{top[TARGET].min():.0f} / {top[TARGET].median():.0f} / {top[TARGET].max():.0f} sec")
    print(f"Residual          min/med/max : "
          f"{top['resid'].min():.0f} / {top['resid'].median():.0f} / {top['resid'].max():.0f} sec")
    print(f"Overpredicted rows          : {(top['resid'] > 0).sum()} / 100")
    print(f"Underpredicted rows         : {(top['resid'] < 0).sum()} / 100")

    print("\nFrequency patterns in the 100 largest-error rows:")
    for field in LARGE_FIELDS:
        vc = top[field].value_counts()
        top_share = 100 * vc.iloc[0] / len(top)
        parts = ", ".join(f"{k}={v}" for k, v in vc.head(6).items())
        print(f"  {field:20s} distinct={len(vc):>3d}  "
              f"top_value_share={top_share:>5.1f}%  {parts}")

    print("\nActual taxi-time bin distribution of the 100 largest-error rows:")
    tb = pd.cut(top[TARGET], bins=BIN_EDGES, labels=BIN_LABELS, right=False)
    for label, count in tb.value_counts().reindex(BIN_LABELS).items():
        print(f"  {label:11s} {int(count):>4d}  ({100 * count / len(top):>5.1f}%)")

    print("\nHour distribution of the 100 largest-error rows:")
    hv = top["hour"].value_counts().sort_index()
    print("  " + "  ".join(f"{int(h):02d}h={int(c)}" for h, c in hv.items()))

    print("\ntraffic_last_15min of the 100 largest-error rows:")
    q = top["traffic_last_15min"]
    print(f"  min={q.min():.0f} p25={q.quantile(0.25):.0f} "
          f"median={q.median():.0f} p75={q.quantile(0.75):.0f} max={q.max():.0f}")

    print("\nMost frequent repeated flight numbers / callsigns:")
    for field in ["FLIGHT_mvt", "CALLSIGN_flt", "STAND_mvt"]:
        vc = top[field].value_counts()
        rep = vc[vc > 1]
        if len(rep):
            print(f"  {field}: " + ", ".join(f"{k}={v}" for k, v in rep.items()))

pd.concat(csv_rows, ignore_index=True).to_csv(
    Path("reports") / "v7_largest_errors.csv", index=False
)
print(f"\nDetailed large-error rows saved to reports/v7_largest_errors.csv")


# ============================================================
# 5 + 6. DEEP DIVE: LIRF AND LFPG
# ============================================================

def deep_dive(frame, airport, name, emphasis=""):
    total_sq = frame["sq_err"].sum()
    sub = frame[frame["ADEP_mvt"] == airport]

    header(f"{'5' if airport == 'LIRF' else '6'}. "
           f"{airport} DEEP DIVE - {name}{emphasis}")

    if len(sub) == 0:
        print("No rows for this airport in this period.")
        return

    print(f"Rows                     : {len(sub):,} "
          f"({100 * len(sub) / len(frame):.2f}% of period)")
    print(f"RMSE                     : {rmse_of(sub['sq_err']):.2f}")
    print(f"MAE                      : {sub['abs_err'].mean():.2f}")
    print(f"Bias (pred - actual)     : {sub['resid'].mean():.2f}")
    print(f"Total squared error      : {sub['sq_err'].sum():,.0f}")
    print(f"Share of period sq error : "
          f"{100 * sub['sq_err'].sum() / total_sq:.2f}%")

    print("\nTarget distribution (sec):")
    print(f"  min   {sub[TARGET].min():>12,.0f}")
    for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"  p{int(q * 100):<4} {sub[TARGET].quantile(q):>12,.0f}")
    print(f"  max   {sub[TARGET].max():>12,.0f}")

    print("\nPrediction distribution (sec):")
    print(f"  min   {sub['pred'].min():>12,.2f}")
    for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"  p{int(q * 100):<4} {sub['pred'].quantile(q):>12,.2f}")
    print(f"  max   {sub['pred'].max():>12,.2f}")

    print(f"\nUnder-dispersion (pred_std / actual_std): "
          f"{sub['pred'].std() / sub[TARGET].std():.3f}")

    print("\nShare of rows above thresholds:")
    for sec, label in [(1800, "> 30 min"), (2700, "> 45 min"),
                       (3600, "> 60 min"), (5400, "> 90 min")]:
        c = int((sub[TARGET] > sec).sum())
        print(f"  {label:9s} (> {sec:>5d} s): {c:>7,} rows  "
              f"{100 * c / len(sub):>6.2f}% of {airport}   "
              f"{100 * sub.loc[sub[TARGET] > sec, 'sq_err'].sum() / sub['sq_err'].sum():>6.2f}% "
              f"of {airport} sq err")

    for key, min_rows in [("RUNWAY_mvt", 20), ("STAND_mvt", 20),
                          ("hour", 20), ("FLIGHT_mvt", 30),
                          ("CALLSIGN_flt", 30)]:
        t = group_metrics(sub, key, min_rows=min_rows)
        if len(t) == 0:
            print(f"\nNo {key} groups with >= {min_rows} rows")
            continue
        t["pct_of_airport_sq"] = 100 * t["total_sq"] / sub["sq_err"].sum()
        print_table(
            f"\n{airport} by {key} (groups with >= {min_rows} rows, "
            f"sorted by squared error)",
            t,
            None,
            {
                key: 10, "rows": 8, "rmse": 10, "mae": 10, "bias": 10,
                "actual_mean": 11, "actual_std": 11,
                "pred_mean": 10, "pred_std": 10,
                "total_sq": 15, "pct_of_airport_sq": 8,
            },
            sort_by="total_sq",
        )

    # within-airport concentration
    s = sub["sq_err"].sort_values(ascending=False)
    n = len(s)
    print(f"\n{airport} error concentration (n={n:,}):")
    for pct in [0.5, 1, 2, 5, 10]:
        k = max(1, int(round(n * pct / 100)))
        print(f"  worst {pct:>4.1f}% ({k:>5,} rows) -> "
              f"{100 * s.iloc[:k].sum() / s.sum():>6.2f}% of {airport} sq err")


for name, _ in [(s[0], s[1]) for s in SPLITS]:
    deep_dive(frames[name], "LIRF", name)
    deep_dive(frames[name], "LFPG", name, emphasis=" (emphasis: July)")


# ============================================================
# 7. EXTREME TARGET ANALYSIS
# ============================================================

for name, _ in [(s[0], s[1]) for s in SPLITS]:
    frame = frames[name]
    total_sq = frame["sq_err"].sum()
    n = len(frame)

    header(f"7. EXTREME TARGET ANALYSIS - {name}")

    print(f"{'threshold':>12s} {'rows':>8s} {'pct rows':>10s} "
          f"{'pct sq err':>12s} {'mean pred':>11s} {'mean actual':>12s} "
          f"{'mean resid':>11s}")
    rule("-", 82)

    for sec in THRESHOLDS:
        sub = frame[frame[TARGET] > sec]
        c = len(sub)
        if c == 0:
            print(f"{'> ' + str(sec):>12s} {0:>8,} {0.0:>9.2f}% "
                  f"{0.0:>11.2f}% {'-':>11s} {'-':>12s} {'-':>11s}")
            continue
        print(f"{'> ' + str(sec):>12s} {c:>8,} "
              f"{100 * c / n:>9.2f}% "
              f"{100 * sub['sq_err'].sum() / total_sq:>11.2f}% "
              f"{sub['pred'].mean():>11.2f} {sub[TARGET].mean():>12,.2f} "
              f"{sub['resid'].mean():>11.2f}")

    print(f"\nMax target in period: {frame[TARGET].max():,.0f} sec "
          f"({frame[TARGET].max() / 3600:.1f} h)")
    print(f"Min target in period: {frame[TARGET].min():,.0f} sec")


# ============================================================
# 8. MODEL UNDER-DISPERSION
# ============================================================

PCTS = [0.50, 0.75, 0.90, 0.95, 0.99]

for name, _ in [(s[0], s[1]) for s in SPLITS]:
    frame = frames[name]

    header(f"8. MODEL UNDER-DISPERSION - {name}")

    print("Period level:")
    print(f"  actual mean {frame[TARGET].mean():>9.2f}   "
          f"pred mean {frame['pred'].mean():>9.2f}   "
          f"bias {frame['resid'].mean():>+9.2f}")
    print(f"  actual std  {frame[TARGET].std():>9.2f}   "
          f"pred std  {frame['pred'].std():>9.2f}   "
          f"ratio {frame['pred'].std() / frame[TARGET].std():.3f}")
    print(f"  actual max  {frame[TARGET].max():>9,.0f}   "
          f"pred max  {frame['pred'].max():>9,.2f}")
    print(f"  pred range collapse: predicted span covers "
          f"{100 * (frame['pred'].max() - frame['pred'].min()) / (frame[TARGET].max() - frame[TARGET].min()):.1f}% "
          f"of the actual span")

    print("\nPercentile comparison (actual vs prediction):")
    print(f"  {'pct':>6s} {'actual':>12s} {'pred':>12s} {'diff':>12s} {'ratio':>8s}")
    rule("-", 56)
    for q in PCTS:
        a = frame[TARGET].quantile(q)
        p = frame["pred"].quantile(q)
        print(f"  p{int(q * 100):<4d} {a:>12,.2f} {p:>12,.2f} "
              f"{p - a:>+12,.2f} {p / a:>8.3f}")

    t = group_metrics(frame, "ADEP_mvt")
    t["ratio"] = t["pred_std"] / t["actual_std"]
    t = t.sort_values("ratio")
    print_table(
        "\nPer-airport under-dispersion (sorted by pred_std/actual_std)",
        t,
        None,
        {
            "ADEP_mvt": 8, "rows": 9, "ratio": 8,
            "actual_std": 11, "pred_std": 11,
            "actual_mean": 11, "pred_mean": 10, "bias": 10, "rmse": 10,
        },
    )


# ============================================================
# FINAL FACTUAL SUMMARY
# ============================================================

header("FINAL FACTUAL SUMMARY", "=")

for name, _ in [(s[0], s[1]) for s in SPLITS]:
    frame = frames[name]
    total_sq = frame["sq_err"].sum()
    n = len(frame)
    base = rmse_of(frame["sq_err"])

    print(f"\n--- {name} ---")
    print(f"rows={n:,}  RMSE={base:.2f}  MAE={frame['abs_err'].mean():.2f}  "
          f"bias={frame['resid'].mean():+.2f}  "
          f"total_sq={total_sq:,.0f}")

    t = group_metrics(frame, "ADEP_mvt")
    t["pct_sq"] = 100 * t["total_sq"] / total_sq
    t = t.sort_values("total_sq", ascending=False)

    print("Q1/Q2 airports by squared-error share:")
    for airport, row in t.head(3).iterrows():
        print(f"  {airport:8s} {row['pct_sq']:>6.2f}%  "
              f"(RMSE {row['rmse']:.2f}, n={int(row['rows']):,})")

    l = t.loc["LIRF", "pct_sq"] if "LIRF" in t.index else float("nan")
    p = t.loc["LFPG", "pct_sq"] if "LFPG" in t.index else float("nan")
    print(f"LIRF share={l:.2f}%   LFPG share={p:.2f}%   "
          f"both={l + p:.2f}%")

    s = frame["sq_err"].sort_values(ascending=False)
    for pct in [1, 5]:
        k = max(1, int(round(n * pct / 100)))
        print(f"worst {pct}% of rows -> {100 * s.iloc[:k].sum() / total_sq:.2f}% "
              f"of sq err, RMSE without them "
              f"{np.sqrt((total_sq - s.iloc[:k].sum()) / (n - k)):.2f}")

    for sec, label in [(1800, "> 30 min"), (2700, "> 45 min"), (3600, "> 60 min")]:
        sub = frame[frame[TARGET] > sec]
        print(f"actual {label:9s} -> {len(sub):>7,} rows "
              f"({100 * len(sub) / n:>5.2f}%), "
              f"{100 * sub['sq_err'].sum() / total_sq:>6.2f}% of sq err, "
              f"mean pred {sub['pred'].mean():.2f} vs actual "
              f"{sub[TARGET].mean():,.0f}")

    kept = frame[~frame["ADEP_mvt"].isin(["LIRF", "LFPG"])]
    print(f"RMSE excluding LIRF+LFPG: {rmse_of(kept['sq_err']):.2f} "
          f"(vs {base:.2f})")

    print(f"under-dispersion ratio: "
          f"{frame['pred'].std() / frame[TARGET].std():.3f}")
    worst = frame.reindex(frame["abs_err"].sort_values(ascending=False).index).head(100)
    print(f"worst 100 rows: airports={dict(worst['ADEP_mvt'].value_counts().head(4))}")
    print(f"worst 100 rows: over/under predicted = "
          f"{(worst['resid'] > 0).sum()}/{(worst['resid'] < 0).sum()}")
    print(f"worst 100 rows: mean actual={worst[TARGET].mean():,.0f} s, "
          f"mean pred={worst['pred'].mean():.2f} s")

header("END OF ANALYSIS. No model trained beyond reproducing the two "
       "diagnostic models. No V8. No submission.", "=")

# ============================================================
# CLOSE REPORT
# ============================================================

sys.stdout = _real_stdout
_report_file.close()

_size = REPORT_PATH.stat().st_size
_lines = len(REPORT_PATH.read_text(encoding="utf-8").splitlines())
print(f"\nComplete analysis written to: {REPORT_PATH}")
print(f"Report size : {_size:,} bytes")
print(f"Report lines: {_lines:,}")
