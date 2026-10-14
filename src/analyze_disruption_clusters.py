"""Temporal disruption-cluster analysis for the V7 January / July holdouts.

Focus airports: LIRF (July 2025 primary, January 2025 secondary), LFPG
(January 2025 primary).

Reproduces the V7 TEST B and TEST C models exactly, then asks whether extreme
taxi-time events form temporal clusters that preceding airport activity could
detect.

DIAGNOSTIC ONLY. Nothing here is a proposed feature:
- preceding-arrival measurements are NOT asserted to be competition-legal
- preceding-departure TARGETS are hidden in ranking, so section E is a
  propagation study only
- no model is trained on any of these derived measurements
- V7 features and CatBoost config are unchanged
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error


# ============================================================
# REPORT TEE
# ============================================================

REPORT_PATH = Path("reports") / "v7_disruption_clusters.txt"
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


def rule(char="-", width=112):
    print(char * width)


def header(title, char="#", width=112):
    print()
    print(char * width)
    print(title)
    print(char * width)


def table(title, rows, fmts, sort_by=None, descending=False):
    print()
    print(title)
    head = " ".join(f"{c:>{w}s}" for c, w in fmts.items())
    print(head)
    rule("-", len(head))
    order = rows if sort_by is None else sorted(
        rows, key=lambda r: r[sort_by], reverse=descending)
    for r in order:
        cells = []
        for c, w in fmts.items():
            v = r.get(c, "")
            if isinstance(v, (bool, np.bool_)):
                cells.append(f"{str(v):>{w}s}")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):>{w}d}")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:>{w}.2f}")
            else:
                cells.append(f"{str(v):>{w}s}")
        print(" ".join(cells))
    return order


def rmse_of(sq):
    return float(np.sqrt(np.mean(np.asarray(sq, dtype="float64"))))


def to_ns(values):
    """Nanosecond int64 view of timestamps.

    DatetimeIndex.asi8 returns whatever unit the index happens to hold (a
    string-derived index can be second-resolution), which would silently break
    every window arithmetic below. Force nanoseconds explicitly.
    """
    return np.asarray(
        pd.DatetimeIndex(values).to_numpy(dtype="datetime64[ns]"),
        dtype="int64",
    )


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = [
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "hour", "day_of_week",
    "month", "traffic_last_15min", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
]

CATEGORICAL_FEATURES = [
    "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt",
]

TARGET = "TAXITIME_SEC_mvt"
WINDOWS = [15, 30, 60, 120]
EXTREME_THRESHOLDS = [
    ("> 30 min", 1800), ("> 45 min", 2700), ("> 60 min", 3600),
    ("> 90 min", 5400),
]
MISSING_TRACKED = [
    "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
]
EXPECTED = {"JANUARY": 519.90, "JULY": 630.63}
FOCUS_AIRPORTS = ["LIRF", "LFPG"]
FOCUS = [("LIRF", "JULY"), ("LFPG", "JANUARY"), ("LIRF", "JANUARY")]
AEZ = "AEZ146R"
NS = 10 ** 9


# ============================================================
# LOAD
# ============================================================

print("Loading 2025 training data...")
DATA_DIR = Path("data/raw")
files = sorted(DATA_DIR.glob("training_*.parquet"))
print(f"files found: {len(files)}")

data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
gc.collect()
print(f"Total movements: {len(data):,}")

# missingness indicators, recorded BEFORE the V7 fillna
for c in MISSING_TRACKED:
    data[c + "__missing"] = data[c].isna()

# congestion feature, identical to V7
data["movement_airport"] = data["ADEP_mvt"]
data.loc[data["PHASE_mvt"] == "ARR", "movement_airport"] = data.loc[
    data["PHASE_mvt"] == "ARR", "ADES_mvt"
]
data = data.sort_values(["movement_airport", "MVT_TIME_UTC_mvt"]).copy()

traffic_15min = (
    data
    .set_index("MVT_TIME_UTC_mvt")
    .groupby("movement_airport")["MVT_ID_mvt"]
    .rolling("15min", closed="left")
    .count()
    .reset_index(name="traffic_last_15min")
)
data["traffic_last_15min"] = traffic_15min["traffic_last_15min"].to_numpy()
data["traffic_last_15min"] = data["traffic_last_15min"].fillna(0)
del traffic_15min, data["movement_airport"]
gc.collect()

# an arrival happens at its destination airport
arrivals_all = data[data["PHASE_mvt"] == "ARR"][
    ["ADES_mvt", "MVT_TIME_UTC_mvt", TARGET]
].copy()
arrivals_all = arrivals_all[arrivals_all["ADES_mvt"].isin(FOCUS_AIRPORTS)]

departures = data[data["PHASE_mvt"] == "DEP"].copy()
del data
gc.collect()
print(f"Departure rows: {len(departures):,}")
print(f"Arrival rows at {FOCUS_AIRPORTS}: {len(arrivals_all):,}")

assert len(departures) == 2_085_047, f"unexpected departures {len(departures):,}"

# --- V7 feature engineering ---
departures["hour"] = departures["MVT_TIME_UTC_mvt"].dt.hour
departures["day_of_week"] = departures["MVT_TIME_UTC_mvt"].dt.dayofweek
departures["month"] = departures["MVT_TIME_UTC_mvt"].dt.month
departures["date"] = departures["MVT_TIME_UTC_mvt"].dt.normalize()

for column in CATEGORICAL_FEATURES:
    departures[column] = departures[column].fillna("UNKNOWN").astype(str)

departures["year"] = departures["MVT_TIME_UTC_mvt"].dt.year
assert set(FEATURES).isdisjoint({"year", "date"}), "helpers must not be features"

CARRY = ["MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt",
         "date", "year"] + [f"{c}__missing" for c in MISSING_TRACKED]


# ============================================================
# REPRODUCE THE TWO V7 DIAGNOSTIC MODELS
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
    overlap = set(train["MVT_ID_mvt"].to_numpy()) & set(
        validation["MVT_ID_mvt"].to_numpy())
    assert len(overlap) == 0, f"{name}: {len(overlap):,} val IDs in training"
    print(f"Training rows          : {len(train):,}")
    print(f"Validation rows        : {len(validation):,}")
    print(f"ID overlap             : {len(overlap):,}")

    model = CatBoostRegressor(
        iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
        random_seed=42, verbose=100, early_stopping_rounds=50,
        use_best_model=True,
    )
    model.fit(
        train[FEATURES], train[TARGET], cat_features=CATEGORICAL_FEATURES,
        eval_set=(validation[FEATURES], validation[TARGET]),
    )
    pred = model.predict(validation[FEATURES])
    rmse = root_mean_squared_error(validation[TARGET], pred)
    delta = abs(rmse - EXPECTED[name])
    print(f"RMSE                   : {rmse:.2f}")
    print(f"Expected               : {EXPECTED[name]:.2f} (delta {delta:.4f})")
    assert delta < 0.05, f"{name}: reproduction failed ({delta:.4f})"
    print("Reproduction verified.")

    f = validation[[*FEATURES, *CARRY, TARGET]].copy()
    f["pred"] = pred
    f["resid"] = f["pred"] - f[TARGET]
    f["abs_err"] = f["resid"].abs()
    f["sq_err"] = f["resid"] ** 2
    frames[name] = f
    del model, train, validation, f
    gc.collect()

# full-2025 per-airport series (needed for causal ordering across months)
dep_all = {}
for ap in FOCUS_AIRPORTS:
    dep_all[ap] = departures[departures["ADEP_mvt"] == ap][
        ["MVT_TIME_UTC_mvt", TARGET, "month", "date"]
    ].sort_values("MVT_TIME_UTC_mvt").reset_index(drop=True)

arr_all = {}
for ap in FOCUS_AIRPORTS:
    arr_all[ap] = arrivals_all[arrivals_all["ADES_mvt"] == ap][
        ["MVT_TIME_UTC_mvt", TARGET]
    ].sort_values("MVT_TIME_UTC_mvt").reset_index(drop=True)

focus_frames = {}
for ap, mon in FOCUS:
    sub = frames[mon]
    sub = sub[sub["ADEP_mvt"] == ap].reset_index(drop=True)
    focus_frames[(ap, mon)] = sub
    print(f"focus {ap} {mon:8s}: {len(sub):>6,} rows, "
          f"{sub['date'].nunique()} days, RMSE {rmse_of(sub['sq_err']):.2f}")


# ============================================================
# CONTEXT: LABEL AVAILABILITY IN RANKING
# ============================================================

header("CONTEXT - TAXITIME_SEC_mvt AVAILABILITY IN RANKING (legality unknown)")

rk = pd.read_parquet(
    DATA_DIR / "ranking.parquet",
    columns=["PHASE_mvt", "TAXITIME_SEC_mvt"],
)
for ph, g in rk.groupby("PHASE_mvt"):
    frac = 100 * g[TARGET].notna().mean()
    mean = g[TARGET].mean() if g[TARGET].notna().any() else float("nan")
    print(f"ranking {ph}: n={len(g):>7,}  taxi populated={frac:6.2f}%  "
          f"mean={mean:.2f}")
print("\nArrival taxi times are present in ranking; departure taxi times are not.")
print("Whether preceding ranking arrivals may be used is NOT established here.")
del rk
gc.collect()

# ============================================================
# A. DAY-LEVEL ANALYSIS
# ============================================================

day_tables = {}
worst_days = {}

for ap, mon in FOCUS:
    f = focus_frames[(ap, mon)]
    total_sq = f["sq_err"].sum()
    n = len(f)

    header(f"A. DAY-LEVEL ANALYSIS - {ap} {mon}  "
           f"(n={n:,}, RMSE {rmse_of(f['sq_err']):.2f})")

    rows = []
    for d, g in f.groupby("date", observed=True):
        rows.append({
            "date": str(pd.Timestamp(d).date()),
            "dep": len(g),
            "act_mean": g[TARGET].mean(),
            "act_med": g[TARGET].median(),
            "act_p90": g[TARGET].quantile(0.90),
            "act_p95": g[TARGET].quantile(0.95),
            "act_p99": g[TARGET].quantile(0.99),
            "act_max": g[TARGET].max(),
            "pct30": 100 * (g[TARGET] > 1800).sum() / len(g),
            "pct45": 100 * (g[TARGET] > 2700).sum() / len(g),
            "pct60": 100 * (g[TARGET] > 3600).sum() / len(g),
            "sq": g["sq_err"].sum(),
            "pct_sq": 100 * g["sq_err"].sum() / total_sq,
            "traffic": g["traffic_last_15min"].mean(),
            "m_air": 100 * g["AIRCRAFT_TYPE_mvt__missing"].mean(),
            "m_wk": 100 * g["WK_TBL_CAT_flt__missing"].mean(),
            "m_ft": 100 * g["FLIGHT_TYPE_flt__missing"].mean(),
        })
    day_tables[(ap, mon)] = rows

    ranked = sorted(rows, key=lambda r: r["sq"], reverse=True)
    table(
        "Every day, ranked by V7 squared-error contribution",
        ranked,
        {"date": 10, "dep": 7, "act_mean": 9, "act_med": 8, "act_p90": 8,
         "act_p95": 8, "act_p99": 9, "act_max": 10, "pct30": 7, "pct45": 7,
         "pct60": 7, "sq": 16, "pct_sq": 8, "traffic": 8,
         "m_air": 7, "m_wk": 7, "m_ft": 7},
    )

    worst_days[(ap, mon)] = ranked[:10]

    s = sorted((r["sq"] for r in rows), reverse=True)
    tot = float(sum(s))
    print()
    for k in (1, 3, 5, 10):
        print(f"top {k:>2d} day(s) carry {100 * sum(s[:k]) / tot:>6.2f}% "
              f"of {ap} {mon} squared error")
    cum = np.cumsum(s)
    nd50 = int(np.searchsorted(cum, 0.50 * tot) + 1)
    nd80 = int(np.searchsorted(cum, 0.80 * tot) + 1)
    print(f"days needed for 50% of squared error : {nd50} of {len(rows)}")
    print(f"days needed for 80% of squared error : {nd80} of {len(rows)}")
    print(f"max daily departures {max(r['dep'] for r in rows):,}   "
          f"min {min(r['dep'] for r in rows):,}")


# ============================================================
# B. HOUR-LEVEL ANALYSIS FOR THE WORST 10 AIRPORT-DAYS
# ============================================================

for ap, mon in FOCUS:
    f = focus_frames[(ap, mon)]

    header(f"B. HOUR-LEVEL ANALYSIS - worst 10 {ap} {mon} days")

    for rank, d in enumerate(worst_days[(ap, mon)], 1):
        day = pd.Timestamp(d["date"])
        g = f[f["date"] == day]
        if len(g) == 0:
            continue
        print()
        print(f"--- day rank {rank}: {day.date()}  "
              f"(dep={len(g):,}, act_max={g[TARGET].max():,.0f}s, "
              f"day sq={d['sq']:,.0f}, {d['pct_sq']:.2f}% of {ap} {mon}) ---")

        rows = []
        for h, gh in g.groupby("hour", observed=True):
            rw = gh["RUNWAY_mvt"].value_counts()
            top_rw = ", ".join(f"{k}:{v}" for k, v in rw.head(2).items())
            rows.append({
                "hour": int(h),
                "dep": len(gh),
                "act_mean": gh[TARGET].mean(),
                "act_med": gh[TARGET].median(),
                "act_p90": gh[TARGET].quantile(0.90),
                "act_max": gh[TARGET].max(),
                "pred_mean": gh["pred"].mean(),
                "rmse": rmse_of(gh["sq_err"]),
                "n30": int((gh[TARGET] > 1800).sum()),
                "n45": int((gh[TARGET] > 2700).sum()),
                "n60": int((gh[TARGET] > 3600).sum()),
                "n90": int((gh[TARGET] > 5400).sum()),
                "runways": top_rw,
                "traffic": gh["traffic_last_15min"].mean(),
                "m_air": 100 * gh["AIRCRAFT_TYPE_mvt__missing"].mean(),
                "m_wk": 100 * gh["WK_TBL_CAT_flt__missing"].mean(),
                "m_ft": 100 * gh["FLIGHT_TYPE_flt__missing"].mean(),
            })
        table(
            "  by hour",
            rows,
            {"hour": 5, "dep": 6, "act_mean": 9, "act_med": 8, "act_p90": 9,
             "act_max": 10, "pred_mean": 10, "rmse": 9, "n30": 5, "n45": 5,
             "n60": 5, "n90": 5, "runways": 22, "traffic": 8,
             "m_air": 7, "m_wk": 7, "m_ft": 7},
        )


# ============================================================
# C. TEMPORAL CLUSTERING OF EXTREME DEPARTURES
# ============================================================

def cluster_stats(ap, month_label, focus):
    """Extreme-vs-baseline temporal clustering at one airport.

    Extreme flags and 'previous extreme' ordering are computed on the airport's
    full 2025 departure series, then evaluated on the focus month.
    """
    d = dep_all[ap]
    times = to_ns(d["MVT_TIME_UTC_mvt"])
    vals = d[TARGET].to_numpy(dtype="float64")

    out = {}
    for label, thr in EXTREME_THRESHOLDS:
        ext = vals > thr
        ext_times = np.sort(times[ext])
        q = to_ns(focus["MVT_TIME_UTC_mvt"])
        q_vals = focus[TARGET].to_numpy(dtype="float64")
        q_ext = q_vals > thr

        # index of last extreme strictly before each query time
        prev_idx = np.searchsorted(ext_times, q, side="left") - 1
        has_prev = prev_idx >= 0
        gap_min = np.full(len(q), np.nan)
        gap_min[has_prev] = (q[has_prev] - ext_times[prev_idx[has_prev]]) / (60 * NS)

        rec = {
            "n_extreme": int(q_ext.sum()),
            "n_total": len(q),
            "pct_extreme": 100 * q_ext.mean(),
            "gap_median": float(np.nanmedian(gap_min[q_ext])) if q_ext.any() else np.nan,
            "gap_min": float(np.nanmin(gap_min[q_ext])) if q_ext.any() else np.nan,
        }
        for W in WINDOWS:
            lo = np.searchsorted(ext_times, q - W * 60 * NS, side="right")
            hi = np.searchsorted(ext_times, q, side="left")
            cnt = hi - lo
            rec[f"ext_pct_{W}"] = 100 * float((cnt[q_ext] > 0).mean()) if q_ext.any() else np.nan
            rec[f"base_pct_{W}"] = 100 * float((cnt[~q_ext] > 0).mean()) if (~q_ext).any() else np.nan
            rec[f"ext_cnt_{W}"] = float(cnt[q_ext].mean()) if q_ext.any() else np.nan
            rec[f"base_cnt_{W}"] = float(cnt[~q_ext].mean()) if (~q_ext).any() else np.nan
            # all-departure density in the same window, for reference
            lo2 = np.searchsorted(times, q - W * 60 * NS, side="right")
            hi2 = np.searchsorted(times, q, side="left")
            rec[f"den_{W}"] = float((hi2 - lo2)[q_ext].mean()) if q_ext.any() else np.nan
        out[label] = rec
    return out


cluster_summary = {}

for ap, mon in FOCUS:
    f = focus_frames[(ap, mon)]
    header(f"C. TEMPORAL CLUSTERING OF EXTREME DEPARTURES - {ap} {mon}")
    print(f"n={len(f):,}  clustering computed on the airport's full 2025 series")

    res = cluster_stats(ap, mon, f)
    cluster_summary[(ap, mon)] = res

    for label, r in res.items():
        print()
        print(f"--- extreme definition: actual {label} ---")
        print(f"extreme rows {r['n_extreme']:,} of {r['n_total']:,} "
              f"({r['pct_extreme']:.2f}%)")
        print(f"median gap since previous extreme: {r['gap_median']:.1f} min "
              f"(min {r['gap_min']:.1f} min)")
        print()
        print(f"  {'window':>7s} {'extreme w/ prior':>17s} {'baseline':>10s} "
              f"{'ratio':>7s} {'mean cnt ext':>13s} {'mean cnt base':>14s} "
              f"{'mean all deps':>14s}")
        rule("-", 92)
        for W in WINDOWS:
            e, b = r[f"ext_pct_{W}"], r[f"base_pct_{W}"]
            ratio = e / b if b else float("nan")
            print(f"  {W:>5d}m {e:>16.2f}% {b:>9.2f}% {ratio:>7.2f}x "
                  f"{r[f'ext_cnt_{W}']:>13.3f} {r[f'base_cnt_{W}']:>14.3f} "
                  f"{r[f'den_{W}']:>14.2f}")
        print("  (baseline = non-extreme departures at the same airport, "
              "same month)")

# ============================================================
# TRAILING-WINDOW CAUSAL MEASUREMENT
# ============================================================

def trailing_stats(src_time, src_val, q_time, windows):
    """Statistics over source events strictly BEFORE each query time.

    Window is the open interval (q - W, q). No future event is ever used:
    the upper bound is a 'left' search, i.e. first index with time >= q.
    """
    st = to_ns(src_time)
    sv = np.asarray(src_val, dtype="float64")
    order = np.argsort(st, kind="stable")
    st = st[order]
    sv = sv[order]
    q = to_ns(q_time)

    valid = ~np.isnan(sv)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, sv, 0.0))])
    cval = np.concatenate([[0], np.cumsum(valid.astype("int64"))])

    hi = np.searchsorted(st, q, side="left")
    out = {}
    prev = np.full(len(q), np.nan)
    has = hi > 0
    prev[has] = sv[hi[has] - 1]
    out["prev"] = prev
    out["has_prev"] = has

    for W in windows:
        lo = np.searchsorted(st, q - W * 60 * NS, side="right")
        n = cval[hi] - cval[lo]
        tot = csum[hi] - csum[lo]
        mean = np.divide(tot, n, out=np.full(len(q), np.nan), where=n > 0)
        mx = np.full(len(q), np.nan)
        md = np.full(len(q), np.nan)
        for i in range(len(q)):
            a, b = lo[i], hi[i]
            if b > a:
                seg = sv[a:b]
                seg = seg[~np.isnan(seg)]
                if seg.size:
                    mx[i] = seg.max()
                    md[i] = np.median(seg)
        out[f"n_{W}"] = n
        out[f"mean_{W}"] = mean
        out[f"max_{W}"] = mx
        out[f"median_{W}"] = md
    return out


def group_by_duration(f, label):
    bins = [-np.inf, 1800, 2700, 3600, 5400, np.inf]
    names = ["<= 30 min", "30-45 min", "45-60 min", "60-90 min", "> 90 min"]
    g = f.copy()
    g["grp"] = pd.cut(g[TARGET], bins=bins, labels=names, right=True)
    return g, names


# ============================================================
# D. PRECEDING ARRIVAL SIGNAL - DIAGNOSTIC ONLY
# ============================================================

arr_signal = {}

for ap, mon in FOCUS:
    f = focus_frames[(ap, mon)]

    header(f"D. PRECEDING ARRIVAL SIGNAL (diagnostic only, legality unknown)"
           f" - {ap} {mon}")

    st = trailing_stats(
        arr_all[ap]["MVT_TIME_UTC_mvt"], arr_all[ap][TARGET],
        f["MVT_TIME_UTC_mvt"], WINDOWS,
    )
    for k, v in st.items():
        f["arr_" + k] = v

    print(f"arrival source rows at {ap} (all 2025): {len(arr_all[ap]):,}")
    print(f"departures analysed: {len(f):,}")
    print(f"departures with at least one preceding arrival: "
          f"{int(f['arr_has_prev'].sum()):,} "
          f"({100 * f['arr_has_prev'].mean():.2f}%)")
    print("windows are strictly (t - W, t); no future arrival is used")

    g, names = group_by_duration(f, ap)
    rows = []
    for name in names:
        s = g[g["grp"] == name]
        if len(s) == 0:
            continue
        rec = {
            "grp": name, "n": len(s),
            "pct": 100 * len(s) / len(g),
            "dep_mean": s[TARGET].mean(),
            "prev_arr": s["arr_prev"].mean(),
            "prev_arr_med": s["arr_prev"].median(),
        }
        for W in WINDOWS:
            rec[f"n{W}"] = s[f"arr_n_{W}"].mean()
            rec[f"mean{W}"] = s[f"arr_mean_{W}"].mean()
            rec[f"med{W}"] = s[f"arr_median_{W}"].mean()
            rec[f"max{W}"] = s[f"arr_max_{W}"].mean()
        rows.append(rec)

    table(
        "Preceding arrival taxi time by departure duration group",
        rows,
        {"grp": 11, "n": 8, "pct": 7, "dep_mean": 10, "prev_arr": 10,
         "prev_arr_med": 12,
         "n15": 7, "mean15": 9, "med15": 9, "max15": 10,
         "n30": 7, "mean30": 9, "med30": 9, "max30": 10,
         "n60": 7, "mean60": 9, "med60": 9, "max60": 10,
         "n120": 7, "mean120": 9, "max120": 10},
    )

    print("\nSeparation of preceding-30min mean arrival taxi time:")
    base = g[g["grp"] == "<= 30 min"]["arr_mean_30"].mean()
    for name in names:
        s = g[g["grp"] == name]["arr_mean_30"].mean()
        print(f"  {name:11s} preceding-30min arrival mean = {s:>9.2f} s   "
              f"vs normal {base:>9.2f} s   ratio {s / base:.3f}")

    print("\nCorrelations with departure taxi time:")
    for col, cname in [("arr_prev", "previous arrival"),
                       ("arr_mean_15", "mean arrival 15m"),
                       ("arr_mean_30", "mean arrival 30m"),
                       ("arr_mean_60", "mean arrival 60m"),
                       ("arr_mean_120", "mean arrival 120m"),
                       ("arr_max_30", "max arrival 30m"),
                       ("arr_n_30", "count arrival 30m")]:
        s = f[[col, TARGET]].dropna()
        if len(s) > 10:
            pe = s[col].corr(s[TARGET])
            sp = s[col].corr(s[TARGET], method="spearman")
            print(f"  {cname:22s} n={len(s):>7,}  "
                  f"pearson={pe:+.4f}  spearman={sp:+.4f}")

    # does the preceding-arrival signal separate the extreme tail at all?
    for W in [30, 60]:
        col = f"arr_mean_{W}"
        hi = f[f[TARGET] > 3600][col].mean()
        lo = f[f[TARGET] <= 1800][col].mean()
        print(f"\n  window {W}m: preceding-arrival mean for >60min dep = "
              f"{hi:.2f} s, for <=30min dep = {lo:.2f} s, delta = {hi - lo:+.2f} s")
        # how much of the >60min group's squared error sits in the top decile
        sub = f[f[TARGET] > 3600]
        if len(sub) > 0 and sub[col].notna().any():
            cut = f[col].quantile(0.90)
            share = 100 * sub.loc[sub[col] > cut, "sq_err"].sum() / sub["sq_err"].sum()
            print(f"    share of >60min squared error where preceding-arrival "
                  f"mean is in the top decile of all departures: {share:.2f}%")

    arr_signal[(ap, mon)] = True


# ============================================================
# E. PRECEDING DEPARTURE SIGNAL (propagation study only)
# ============================================================

dep_signal = {}

for ap, mon in FOCUS:
    f = focus_frames[(ap, mon)]

    header(f"E. PRECEDING DEPARTURE SIGNAL (propagation study; NOT a ranking "
           f"feature - prior targets are hidden) - {ap} {mon}")

    st = trailing_stats(
        dep_all[ap]["MVT_TIME_UTC_mvt"], dep_all[ap][TARGET],
        f["MVT_TIME_UTC_mvt"], WINDOWS,
    )
    for k, v in st.items():
        f["dep_" + k] = v

    print(f"departure source rows at {ap} (all 2025): {len(dep_all[ap]):,}")
    print("windows are strictly (t - W, t); no future departure is used")

    g, names = group_by_duration(f, ap)
    rows = []
    for name in names:
        s = g[g["grp"] == name]
        if len(s) == 0:
            continue
        rec = {
            "grp": name, "n": len(s), "pct": 100 * len(s) / len(g),
            "dep_mean": s[TARGET].mean(),
            "prev_dep": s["dep_prev"].mean(),
        }
        for W in WINDOWS:
            rec[f"n{W}"] = s[f"dep_n_{W}"].mean()
            rec[f"mean{W}"] = s[f"dep_mean_{W}"].mean()
            rec[f"max{W}"] = s[f"dep_max_{W}"].mean()
        rows.append(rec)

    table(
        "Preceding departure taxi time by departure duration group",
        rows,
        {"grp": 11, "n": 8, "pct": 7, "dep_mean": 10, "prev_dep": 10,
         "n15": 7, "mean15": 9, "max15": 10,
         "n30": 7, "mean30": 9, "max30": 10,
         "n60": 7, "mean60": 9, "max60": 10,
         "n120": 7, "mean120": 9, "max120": 10},
    )

    print("\nCorrelations with departure taxi time:")
    for col, cname in [("dep_prev", "previous departure"),
                       ("dep_mean_15", "mean prior dep 15m"),
                       ("dep_mean_30", "mean prior dep 30m"),
                       ("dep_mean_60", "mean prior dep 60m"),
                       ("dep_mean_120", "mean prior dep 120m"),
                       ("dep_max_30", "max prior dep 30m"),
                       ("dep_n_30", "count prior dep 30m")]:
        s = f[[col, TARGET]].dropna()
        if len(s) > 10:
            print(f"  {cname:22s} n={len(s):>7,}  "
                  f"pearson={s[col].corr(s[TARGET]):+.4f}  "
                  f"spearman={s[col].corr(s[TARGET], method='spearman'):+.4f}")

    dep_signal[(ap, mon)] = True

# ============================================================
# F. EVENT TIMELINES FOR THE 5 HIGHEST-ERROR DAYS
# ============================================================

TIMELINE_SETS = [("LIRF", "JULY"), ("LFPG", "JANUARY")]

for ap, mon in TIMELINE_SETS:
    f = focus_frames[(ap, mon)]
    top5 = [pd.Timestamp(d["date"]) for d in
            sorted(day_tables[(ap, mon)], key=lambda r: r["sq"],
                   reverse=True)[:5]]

    header(f"F. EVENT TIMELINES - 5 highest-error days at {ap} {mon} "
           f"(30-minute buckets)")

    for day in top5:
        gd = f[f["date"] == day]
        ga = arr_all[ap]
        ga = ga[ga["MVT_TIME_UTC_mvt"].dt.normalize() == day]
        if len(gd) == 0:
            continue

        day_sq = gd["sq_err"].sum()
        print()
        print(f"=== {ap} {day.date()}  dep={len(gd):,}  arr={len(ga):,}  "
              f"act_max={gd[TARGET].max():,.0f}s  "
              f"n>30min={(gd[TARGET] > 1800).sum()}  "
              f"n>60min={(gd[TARGET] > 3600).sum()}  "
              f"day_sq={day_sq:,.0f}  "
              f"day_rmse={rmse_of(gd['sq_err']):.2f} ===")

        gd = gd.copy()
        gd["bucket"] = gd["MVT_TIME_UTC_mvt"].dt.floor("30min")
        ga = ga.copy()
        ga["bucket"] = ga["MVT_TIME_UTC_mvt"].dt.floor("30min")

        agg_d = gd.groupby("bucket", observed=True).agg(
            dep_n=("sq_err", "size"),
            d_mean=(TARGET, "mean"),
            d_max=(TARGET, "max"),
        )
        agg_d["d_p90"] = gd.groupby("bucket", observed=True)[TARGET].quantile(0.90)
        agg_d["d_n30"] = gd.groupby("bucket", observed=True)[TARGET].apply(
            lambda s: int((s > 1800).sum()))
        agg_d["d_n60"] = gd.groupby("bucket", observed=True)[TARGET].apply(
            lambda s: int((s > 3600).sum()))
        agg_d["d_traffic"] = gd.groupby("bucket", observed=True)[
            "traffic_last_15min"].mean()
        agg_d["d_rw"] = gd.groupby("bucket", observed=True)["RUNWAY_mvt"].agg(
            lambda s: s.value_counts().index[0])
        agg_d["d_miss"] = gd.groupby("bucket", observed=True).apply(
            lambda s: 100 * (
                s["AIRCRAFT_TYPE_mvt__missing"].mean()
                + s["WK_TBL_CAT_flt__missing"].mean()
                + s["FLIGHT_TYPE_flt__missing"].mean()) / 3,
            include_groups=False)

        if len(ga):
            agg_a = ga.groupby("bucket", observed=True).agg(
                a_n=(TARGET, "size"), a_mean=(TARGET, "mean"),
                a_max=(TARGET, "max"),
            )
        else:
            agg_a = pd.DataFrame(columns=["a_n", "a_mean", "a_max"])

        merged = agg_d.join(agg_a, how="outer").sort_index()
        rows = []
        for b, r in merged.iterrows():
            if pd.isna(r["dep_n"]) and pd.isna(r["a_n"]):
                continue
            rows.append({
                "bucket": pd.Timestamp(b).strftime("%H:%M"),
                "dn": 0 if pd.isna(r["dep_n"]) else int(r["dep_n"]),
                "dmean": r["d_mean"], "dp90": r["d_p90"], "dmax": r["d_max"],
                "n30": 0 if pd.isna(r["d_n30"]) else int(r["d_n30"]),
                "n60": 0 if pd.isna(r["d_n60"]) else int(r["d_n60"]),
                "an": 0 if pd.isna(r["a_n"]) else int(r["a_n"]),
                "amean": r["a_mean"], "amax": r["a_max"],
                "traf": r["d_traffic"], "rw": r["d_rw"],
                "miss": r["d_miss"],
            })
        table(
            "  30-min buckets (d*=departure, a*=arrival)",
            rows,
            {"bucket": 7, "dn": 5, "dmean": 9, "dp90": 9, "dmax": 10,
             "n30": 5, "n60": 5, "an": 5, "amean": 9, "amax": 10,
             "traf": 7, "rw": 6, "miss": 7},
        )


# ============================================================
# G. AEZ146R
# ============================================================

header(f"G. AEZ146R INVESTIGATION (LIRF July 2025)")

lirf_jul = focus_frames[("LIRF", "JULY")]

aez_flt = lirf_jul[lirf_jul["FLIGHT_mvt"] == AEZ]
aez_csl = lirf_jul[lirf_jul["CALLSIGN_flt"] == AEZ]
aez = aez_flt if len(aez_flt) >= len(aez_csl) else aez_csl
match_col = "FLIGHT_mvt" if len(aez_flt) >= len(aez_csl) else "CALLSIGN_flt"

print(f"matched on {match_col}: {len(aez)} rows at LIRF in July 2025")
print(f"  of which FLIGHT_mvt=={AEZ}: {len(aez_flt)}   "
      f"CALLSIGN_flt=={AEZ}: {len(aez_csl)}")

rows = []
for _, r in aez.iterrows():
    rows.append({
        "ts": r["MVT_TIME_UTC_mvt"].strftime("%m-%d %H:%M"),
        "runway": r["RUNWAY_mvt"], "stand": r["STAND_mvt"],
        "dest": r["ADES_mvt"], "act": r[TARGET],
        "pred": r["pred"], "resid": r["resid"],
        "actype": r["AIRCRAFT_TYPE_mvt"],
        "wk": r["WK_TBL_CAT_flt"], "ftype": r["FLIGHT_TYPE_flt"],
        "flight": r["FLIGHT_mvt"], "cs": r["CALLSIGN_flt"],
        "traf": r["traffic_last_15min"],
    })
table(
    "All matching rows",
    sorted(rows, key=lambda x: x["ts"]),
    {"ts": 12, "flight": 9, "cs": 9, "dest": 5, "runway": 7, "stand": 6,
     "actype": 8, "wk": 7, "ftype": 7, "traf": 6,
     "act": 10, "pred": 9, "resid": 10},
)

# --- where do these rows sit relative to airport-wide episodes? ---
lj = dep_all["LIRF"]
lj_t = to_ns(lj["MVT_TIME_UTC_mvt"])
lj_v = lj[TARGET].to_numpy(dtype="float64")
lj_ext = lj_v > 1800
lj_ext_times = np.sort(lj_t[lj_ext])

aez_t = to_ns(aez["MVT_TIME_UTC_mvt"])
lo60 = np.searchsorted(lj_ext_times, aez_t - 60 * 60 * NS, side="right")
hi60 = np.searchsorted(lj_ext_times, aez_t, side="left")
ctx60 = hi60 - lo60

lj_month_mean60 = None
lo_all = np.searchsorted(lj_ext_times, lj_t - 60 * 60 * NS, side="right")
hi_all = np.searchsorted(lj_ext_times, lj_t, side="left")
lj_month_mean60 = float((hi_all - lo_all).mean())

# per-row day context
day_rank = {}
all_days = sorted(day_tables[("LIRF", "JULY")], key=lambda r: r["sq"],
                  reverse=True)
for i, d in enumerate(all_days, 1):
    day_rank[d["date"]] = (i, d["sq"], d["pct_sq"], d["pct30"], d["dep"])

rows = []
for (_, r), c60 in zip(aez.iterrows(), ctx60):
    ds = r["date"].strftime("%Y-%m-%d")
    rk_, dsq, dpct, dp30, ddep = day_rank.get(ds, (0, 0, 0, 0, 0))
    rows.append({
        "ts": r["MVT_TIME_UTC_mvt"].strftime("%m-%d %H:%M"),
        "act": r[TARGET], "sq": r["sq_err"],
        "dayrank": rk_, "daysq": dsq, "daypct": dpct,
        "day_p30": dp30, "ext_pm60": int(c60),
    })
table(
    "Context of each row: is it inside an airport-wide extreme episode?",
    sorted(rows, key=lambda x: x["ts"]),
    {"ts": 12, "act": 10, "sq": 16, "dayrank": 8, "daysq": 16,
     "daypct": 8, "day_p30": 9, "ext_pm60": 10},
)

print(f"\nLIRF all-2025 mean extreme(>30min) departures in preceding 60 min: "
      f"{lj_month_mean60:.2f}")
print(f"{AEZ} rows mean extreme(>30min) in preceding 60 min: {ctx60.mean():.2f}")
print(f"distinct dates: {aez['date'].nunique()}   rows: {len(aez)}")
print(f"dates that are in the top-10 highest-error days: "
      f"{sum(1 for d in aez['date'].dt.strftime('%Y-%m-%d') if day_rank[d][0] <= 10)}")

# --- metadata missingness vs LIRF July baseline ---
print("\nMetadata missingness:")
for c in MISSING_TRACKED:
    am = 100 * aez[c + "__missing"].mean()
    bm = 100 * lirf_jul[c + "__missing"].mean()
    print(f"  {c:20s} {AEZ}={am:6.2f}%   LIRF July baseline={bm:6.2f}%")

print("\nCategorical value distribution (is anything degenerate?):")
for c in ["RUNWAY_mvt", "STAND_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt",
          "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt"]:
    vc = aez[c].value_counts()
    print(f"  {c:20s} distinct={len(vc):>3d}  "
          + ", ".join(f"{k}={v}" for k, v in vc.head(6).items()))

# --- does it occur elsewhere in 2025? ---
print(f"\n{AEZ} anywhere in 2025 (any airport, any month, departures):")
any_dep = departures[
    (departures["FLIGHT_mvt"] == AEZ) | (departures["CALLSIGN_flt"] == AEZ)]
print(f"  total departure rows: {len(any_dep):,}")
if len(any_dep) > 0:
    print(f"  airports: {dict(any_dep['ADEP_mvt'].value_counts())}")
    bym = any_dep.groupby(any_dep["MVT_TIME_UTC_mvt"].dt.month)[TARGET].agg(
        ["size", "mean", "median", "min", "max"])
    table("  taxi time by month (all airports)", bym.reset_index().rename(
        columns={"month": "mo"}).to_dict("records"),
        {"mo": 4, "size": 8, "mean": 10, "median": 10, "min": 10, "max": 12})
    print(f"  overall taxi: mean={any_dep[TARGET].mean():.2f} "
          f"median={any_dep[TARGET].median():.2f} "
          f"p90={any_dep[TARGET].quantile(0.9):.2f} "
          f"max={any_dep[TARGET].max():,.2f}")
    print(f"  LIRF July baseline taxi: mean={lirf_jul[TARGET].mean():.2f} "
          f"median={lirf_jul[TARGET].median():.2f} "
          f"p90={lirf_jul[TARGET].quantile(0.9):.2f}")
    outside = any_dep[any_dep["month"] != 7]
    print(f"  rows outside July: {len(outside):,}")
    if len(outside) > 0:
        print(f"    outside-July taxi: mean={outside[TARGET].mean():.2f} "
              f"median={outside[TARGET].median():.2f} "
              f"p90={outside[TARGET].quantile(0.9):.2f} "
              f"max={outside[TARGET].max():,.2f}")
        print(f"    outside-July share >30min: "
              f"{100 * (outside[TARGET] > 1800).mean():.2f}%")
    july_rows = any_dep[any_dep["month"] == 7]
    if len(july_rows) > 0:
        print(f"    July share >30min: "
              f"{100 * (july_rows[TARGET] > 1800).mean():.2f}%")

# ============================================================
# FINAL FACTUAL SUMMARY
# ============================================================

header("FINAL FACTUAL SUMMARY (measured values only)", "=")

for ap, mon in FOCUS:
    f = focus_frames[(ap, mon)]
    res = cluster_summary[(ap, mon)]
    print(f"\n### {ap} {mon}  n={len(f):,}  "
          f"RMSE={rmse_of(f['sq_err']):.2f}  "
          f"MAE={f['abs_err'].mean():.2f}  bias={f['resid'].mean():+.2f}")

    # 1 / 3 clustering
    print("Q1+Q3 clustering (share of rows with a prior extreme in window):")
    for label, r in res.items():
        parts = "  ".join(
            f"{W}m ext {r[f'ext_pct_{W}']:.1f}% vs base "
            f"{r[f'base_pct_{W}']:.1f}% ({r[f'ext_pct_{W}'] / r[f'base_pct_{W}']:.1f}x)"
            for W in WINDOWS)
        print(f"  {label:9s} n_ext={r['n_extreme']:>5,} "
              f"({r['pct_extreme']:5.2f}%)  gap_med={r['gap_median']:6.1f}m")
        print(f"            {parts}")

    # 2 day concentration
    s = sorted((r["sq"] for r in day_tables[(ap, mon)]), reverse=True)
    tot = float(sum(s))
    cum = np.cumsum(s)
    nd50 = int(np.searchsorted(cum, 0.50 * tot) + 1)
    nd80 = int(np.searchsorted(cum, 0.80 * tot) + 1)
    print(f"Q2 day concentration: top1={100 * s[0] / tot:.2f}%  "
          f"top3={100 * sum(s[:3]) / tot:.2f}%  top5={100 * sum(s[:5]) / tot:.2f}%  "
          f"days_for_50%={nd50}/{len(s)}  days_for_80%={nd80}/{len(s)}")

    # 4 preceding arrival separation
    g, names = group_by_duration(f, ap)
    base = g[g["grp"] == "<= 30 min"]
    print("Q4 preceding arrival signal (30-min window, mean arrival taxi):")
    for name in names:
        s_ = g[g["grp"] == name]
        if len(s_) == 0:
            continue
        v = s_["arr_mean_30"].mean()
        b = base["arr_mean_30"].mean()
        print(f"  {name:11s} n={len(s_):>6,}  preceding30={v:>8.2f}  "
              f"ratio_vs_normal={v / b:.3f}  "
              f"corr_with_dep={s_['arr_mean_30'].corr(s_[TARGET]):+.4f}")
    c30 = f[["arr_mean_30", TARGET]].dropna()
    print(f"  overall corr(preceding-30m arrival mean, departure taxi): "
          f"pearson={c30['arr_mean_30'].corr(c30[TARGET]):+.4f}  "
          f"spearman={c30['arr_mean_30'].corr(c30[TARGET], method='spearman'):+.4f}")

    # 5 traffic
    tr = g.assign(grp30=g[TARGET] > 1800).groupby("grp30")[
        "traffic_last_15min"].mean()
    print(f"Q5 traffic_last_15min mean: normal={tr.get(False, float('nan')):.2f}  "
          f"extreme>30min={tr.get(True, float('nan')):.2f}  "
          f"delta={tr.get(True, float('nan')) - tr.get(False, float('nan')):+.2f}")

    # 6 metadata missingness
    print("Q6 metadata missingness (extreme>30min minus normal, percentage points):")
    for c in MISSING_TRACKED:
        mm = g.assign(e=g[TARGET] > 1800).groupby("e")[c + "__missing"].mean()
        a = 100 * mm.get(True, float("nan"))
        b = 100 * mm.get(False, float("nan"))
        print(f"  {c:20s} extreme={a:6.2f}%  normal={b:6.2f}%  delta={a - b:+.2f}")

    # 8 causal signal potential
    print("Q8 separation strength (extreme>30min vs normal):")
    for col, cname in [("arr_mean_30", "preceding 30m arrival mean"),
                       ("arr_max_30", "preceding 30m arrival max"),
                       ("dep_mean_30", "preceding 30m departure mean")]:
        a = g.loc[g[TARGET] > 1800, col].mean()
        b = g.loc[g[TARGET] <= 1800, col].mean()
        pooled = g[col].std()
        print(f"  {cname:28s} extreme={a:8.2f}  normal={b:8.2f}  "
              f"delta={a - b:+8.2f}  Cohen_d={(a - b) / pooled:.3f}")

header("END. Diagnostic only. No V8, no new model, no submission.", "=")

sys.stdout = _real_stdout
_report_file.close()

_size = REPORT_PATH.stat().st_size
_lines = len(REPORT_PATH.read_text(encoding="utf-8").splitlines())
print(f"\nComplete analysis written to: {REPORT_PATH}")
print(f"Report size : {_size:,} bytes")
print(f"Report lines: {_lines:,}")
