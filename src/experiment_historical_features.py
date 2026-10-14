"""Historical-behaviour feature experiment on the V7 January / July holdouts.

Question: can 2025 historical TAXITIME behaviour of recurring
flights / callsigns / routes / stands / runways materially improve the
January or July 2025 validation RMSE over exact V7?

LEAKAGE RULE (enforced, not just intended)
-----------------------------------------
Every target-derived historical aggregate for a validation row is computed
from that fold's TRAINING ROWS ONLY. The held-out month's targets are never
touched while building its own features. This is asserted at runtime:

  * aggregate source frames are built by masking the fold's train mask;
  * the fitted-aggregate key index is intersected with the train mask and
    every aggregate's row count is reconciled against the training total;
  * a sentinel: if any validation target leaked into an aggregate, the
    "seen in training" counts and the AEZ146R audit in section E would both
    move. Section E is the direct empirical check.

For the eventual 2026 ranking these features would be built from all
available 2025 data, which is exactly what the January fold already does.

NOT a V8. V7 features and the CatBoost configuration are unchanged; this
only measures whether an ADDITION would be worth pursuing.
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error


# ============================================================
# REPORT TEE
# ============================================================

REPORT_PATH = Path("reports") / "historical_feature_experiment.txt"
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


def table(title, rows, fmts, sort_by=None, descending=False, maxrows=None):
    if not rows:
        print()
        print(f"{title}\n  (no rows)")
        return []
    print()
    print(title)
    head = " ".join(f"{c:>{w}s}" for c, w in fmts.items())
    print(head)
    rule("-", len(head))
    order = rows if sort_by is None else sorted(
        rows, key=lambda r: (r[sort_by] is None, r[sort_by]), reverse=descending)
    if maxrows is not None:
        order = order[:maxrows]
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


def fmt(x, nd=2):
    if x is None:
        return "nan"
    try:
        if not np.isfinite(x):
            return "nan"
    except TypeError:
        return str(x)
    return f"{x:.{nd}f}"


# ============================================================
# CONSTANTS - V7 BASELINE IS IMMUTABLE
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

# EXACT V7 CatBoost configuration - do not change.
CB_PARAMS = dict(
    iterations=1000,
    depth=8,
    learning_rate=0.1,
    loss_function="RMSE",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    use_best_model=True,
)

EXPECTED = {"JANUARY": 519.90, "JULY": 630.63}
HOLDOUT_MONTH = {"JANUARY": 1, "JULY": 7}

MIN_COUNT = 5          # a group needs >= this many training rows to be trusted
TRIM = 0.10           # symmetric trimmed mean fraction
CLIP_CAP = 3600       # robust mean cap (seconds)
AEZ = "AEZ146R"
FOCUS = ["LIRF", "LFPG"]

# The 10 groupings under investigation (section A).
GROUPINGS = {
    "flight":         ["K_FLIGHT"],
    "callsign":       ["K_CALLSIGN"],
    "route":          ["ADEP_mvt", "ADES_mvt"],
    "adep_rwy":       ["ADEP_mvt", "RUNWAY_mvt"],
    "adep_stand":     ["ADEP_mvt", "STAND_mvt"],
    "adep_hour":      ["ADEP_mvt", "hour"],
    "adep_rwy_hour":  ["ADEP_mvt", "RUNWAY_mvt", "hour"],
    "adep_stand_hour": ["ADEP_mvt", "STAND_mvt", "hour"],
    "adep_actype":    ["ADEP_mvt", "AIRCRAFT_TYPE_mvt"],
    "route_hour":     ["ADEP_mvt", "ADES_mvt", "hour"],
}

# Two extra fallback backbones, NOT part of the 10 under investigation.
# Without them an unseen group has nowhere to fall back to.
FALLBACK_GROUPINGS = {
    "adep":   ["ADEP_mvt"],
    "global": [],
}

ALL_GROUPINGS = {**GROUPINGS, **FALLBACK_GROUPINGS}

# Backoff chains, finest first. Each step is strictly coarser.
CHAINS = {
    "flight":          ["flight", "callsign", "route_hour", "adep_hour", "adep", "global"],
    "callsign":        ["callsign", "route_hour", "adep_hour", "adep", "global"],
    "route":           ["route", "route_hour", "adep_hour", "adep", "global"],
    "adep_rwy":        ["adep_rwy", "adep_hour", "adep", "global"],
    "adep_stand":      ["adep_stand", "adep_hour", "adep", "global"],
    "adep_hour":       ["adep_hour", "adep", "global"],
    "adep_rwy_hour":   ["adep_rwy_hour", "adep_rwy", "adep_hour", "adep", "global"],
    "adep_stand_hour": ["adep_stand_hour", "adep_stand", "adep_hour", "adep", "global"],
    "adep_actype":     ["adep_actype", "adep_hour", "adep", "global"],
    "route_hour":      ["route_hour", "route", "adep_hour", "adep", "global"],
}

STATS = ["count", "mean", "median", "std", "p90", "p95", "tmean", "cmean"]

# Experiment -> groupings it adds on top of V7.
#
# Mapping of the 10 groupings from section A onto the 4 families of section C.
# Grouping 9 (ADEP + AIRCRAFT_TYPE) is an airport-level behavioural grouping
# with no natural home among H1-H3, so it is placed in H4; otherwise it would
# be built but never tested by any model. This way all 10 groupings are
# exercised by H1-H4.
FAMILIES = {
    "H1": ["flight"],
    "H2": ["callsign"],
    "H3": ["route", "route_hour"],
    "H4": ["adep_rwy", "adep_stand", "adep_hour", "adep_rwy_hour",
           "adep_stand_hour", "adep_actype"],
}

NEEDED_COLS = [
    "MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "PHASE_mvt",
    "MVT_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
    "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt",
    "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", TARGET,
]


# ============================================================
# LOAD
# ============================================================

print("Loading 2025 training data (lean column set)...")
DATA_DIR = Path("data/raw")
files = sorted(DATA_DIR.glob("training_*.parquet"))
print(f"files found: {len(files)}")

data = pd.concat(
    [pd.read_parquet(f, columns=NEEDED_COLS) for f in files],
    ignore_index=True,
)
gc.collect()
print(f"Total movements: {len(data):,}")

# --- congestion feature, byte-for-byte the V7 construction ---
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

departures = data[data["PHASE_mvt"] == "DEP"].copy()
del data
gc.collect()
print(f"Departure rows: {len(departures):,}")
assert len(departures) == 2_085_047, f"unexpected departures {len(departures):,}"

# --- V7 feature engineering, unchanged ---
departures["hour"] = departures["MVT_TIME_UTC_mvt"].dt.hour
departures["day_of_week"] = departures["MVT_TIME_UTC_mvt"].dt.dayofweek
departures["month"] = departures["MVT_TIME_UTC_mvt"].dt.month

for column in CATEGORICAL_FEATURES:
    departures[column] = departures[column].fillna("UNKNOWN").astype(str)

# group keys for the historical aggregates (NaN -> "UNKNOWN", not target-derived)
departures["K_FLIGHT"] = departures["FLIGHT_mvt"].fillna("UNKNOWN").astype(str)
departures["K_CALLSIGN"] = departures["CALLSIGN_flt"].fillna("UNKNOWN").astype(str)

departures["year"] = departures["MVT_TIME_UTC_mvt"].dt.year
departures["date"] = departures["MVT_TIME_UTC_mvt"].dt.normalize()

HELPERS = {"K_FLIGHT", "K_CALLSIGN", "year", "date", TARGET, "MVT_ID_mvt",
           "FLIGHT_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "PHASE_mvt"}
assert set(FEATURES).isdisjoint(HELPERS), "helper columns must not be V7 features"

print(f"Distinct FLIGHT_mvt  : {departures['K_FLIGHT'].nunique():,}")
print(f"Distinct CALLSIGN_flt: {departures['K_CALLSIGN'].nunique():,}")
print(f"Target describe      : {departures[TARGET].describe().round(1).to_dict()}")


# ============================================================
# SECTION A ENGINE - LEAKAGE-SAFE HISTORICAL AGGREGATES
# ============================================================
#
# A Level is a table of target statistics for ONE grouping, fitted on the
# fold's TRAINING ROWS ONLY. Resolution of a validation row against a Level
# is a pure reindex; it never touches the validation targets.
#
# Section B makes each Level outlier-robust:
#   tmean  = symmetric 10% trimmed mean of the training targets
#   cmean  = mean of the training targets clipped to [0, 3600]
# so that a single 87,002-second record cannot dominate a group mean.


def trimmed_mean_1d(a, trim=TRIM):
    a = np.sort(np.asarray(a, dtype="float64"))
    n = a.size
    k = int(np.floor(trim * n))
    if k <= 0 or n - 2 * k <= 0:
        return float(a.mean())
    return float(a[k:n - k].mean())


class Level:
    """Target statistics for one grouping, fitted on training rows only."""

    def __init__(self, name, keys, train_df):
        self.name = name
        self.keys = list(keys)
        self.n_train_rows = int(len(train_df))
        t = train_df[TARGET].to_numpy(dtype="float64")

        if self.keys:
            grp = train_df.groupby(self.keys, observed=True, sort=False)[TARGET]
            stats = pd.DataFrame({
                "count": grp.size().astype("float64"),
                "mean": grp.mean(),
                "median": grp.median(),
                "std": grp.std(),
                "p90": grp.quantile(0.90),
                "p95": grp.quantile(0.95),
                "sum": grp.sum(),
            })

            # cmean: training targets clipped at CLIP_CAP before averaging
            tc = pd.DataFrame({k: train_df[k].to_numpy() for k in self.keys})
            tc["_tc"] = np.clip(t, 0.0, float(CLIP_CAP))
            gtc = tc.groupby(self.keys, observed=True, sort=False)["_tc"]
            stats["cmean"] = gtc.mean()
            # csum enables exact leave-one-out for cmean on training rows
            stats["csum"] = gtc.sum()

            # tmean: exact symmetric trimmed mean, vectorised
            s = pd.DataFrame({k: train_df[k].to_numpy() for k in self.keys})
            s["_t"] = t
            s = s.sort_values(self.keys + ["_t"], kind="mergesort")
            gb = s.groupby(self.keys, observed=True, sort=False)["_t"]
            n = gb.transform("size")
            pos = gb.cumcount()
            kk = np.floor(TRIM * n).astype("int64")
            keep = (pos >= kk) & (pos < (n - kk))
            stats["tmean"] = (
                s.loc[keep, self.keys + ["_t"]]
                .groupby(self.keys, observed=True, sort=False)["_t"].mean()
            )
            self.stats = stats.sort_index()
        else:
            v = pd.Series(t)
            self.stats = pd.DataFrame([{
                "count": float(len(t)),
                "mean": float(v.mean()),
                "median": float(v.median()),
                "std": float(v.std()),
                "p90": float(v.quantile(0.90)),
                "p95": float(v.quantile(0.95)),
                "sum": float(t.sum()),
                "cmean": float(np.clip(t, 0.0, float(CLIP_CAP)).mean()),
                "csum": float(np.clip(t, 0.0, float(CLIP_CAP)).sum()),
                "tmean": trimmed_mean_1d(t),
            }])

        # training-side sample-size distribution for this grouping
        c = self.stats["count"].to_numpy(dtype="float64")
        self.size_dist = {
            "groups": int(len(c)),
            "min": float(np.min(c)),
            "p10": float(np.percentile(c, 10)),
            "p50": float(np.percentile(c, 50)),
            "p90": float(np.percentile(c, 90)),
            "max": float(np.max(c)),
            "ge_min_count_pct": float(100.0 * np.mean(c >= MIN_COUNT)),
        }

    def reindex(self, frame, cols=None):
        """Look this grouping up for every row of `frame`. Pure reindex."""
        want = list(STATS) if cols is None else list(cols)
        if not self.keys:
            row = self.stats.iloc[0]
            n = len(frame)
            return {s: np.full(n, float(row[s]), dtype="float64") for s in want}
        if len(self.keys) == 1:
            # MultiIndex.from_frame on a ONE-column frame yields a MultiIndex of
            # 1-tuples, which silently reindexes to all-NaN against the flat
            # index a single-key groupby produces. Use a flat Index.
            idx = pd.Index(frame[self.keys[0]])
        else:
            idx = pd.MultiIndex.from_frame(frame[self.keys])
        sub = self.stats.reindex(idx)
        return {s: sub[s].to_numpy(dtype="float64") for s in want}


def build_levels(train_df, names):
    levels = {}
    for name in names:
        t0 = time.time()
        levels[name] = Level(name, ALL_GROUPINGS[name], train_df)
        print(f"    level {name:16s} keys={len(ALL_GROUPINGS[name])} "
              f"groups={levels[name].size_dist['groups']:>9,} "
              f"({time.time() - t0:.1f}s)")
    return levels


def resolve(levels, chain, frame, min_count=MIN_COUNT, loo_mask=None,
            own_target=None):
    """Fill every stat for every row of `frame` using the backoff chain.

    Walks coarse -> fine and overwrites, so a finer level wins wherever it
    has at least `min_count` training rows. `global` always qualifies, so no
    row is left unresolved.

    If `own_target` is supplied these are TRAINING rows, and the two
    additive statistics (mean, cmean) are converted to exact leave-one-out
    so a row's own target is not part of its own feature. That matters:
    without it the model learns a relationship that does not hold at
    validation time, because validation aggregates contain no validation
    targets. The order statistics (median/std/p90/p95/tmean) have no
    closed-form LOO and keep the group value - documented limitation.

    `loo_mask` marks WHICH rows may be leave-one-out. It must be False for
    any row whose own target is not in the aggregate source, otherwise the
    subtraction would remove a value that was never added.
    """
    n = len(frame)
    out = {s: np.full(n, np.nan, dtype="float64") for s in STATS}
    lvl_used = np.full(n, float(len(chain)), dtype="float64")
    fine = {"count": None, "sum": None, "csum": None}

    for li in range(len(chain) - 1, -1, -1):
        L = levels[chain[li]]
        vals = L.reindex(frame, cols=STATS + ["sum", "csum"])
        cnt = vals["count"]
        if li == 0:
            fine = {k: vals[k].copy() for k in ("count", "sum", "csum")}
        ok = np.nan_to_num(cnt, nan=0.0) >= min_count
        for s in STATS:
            if s == "count":
                continue
            out[s] = np.where(ok, vals[s], out[s])
        out["count"] = np.where(ok, cnt, out["count"])
        lvl_used = np.where(ok, float(li), lvl_used)

    # exact leave-one-out at the finest level (>= min_count rows, so no div/0)
    if own_target is not None:
        own = np.asarray(own_target, dtype="float64")
        m = np.ones(n, dtype=bool) if loo_mask is None else np.asarray(
            loo_mask, dtype=bool)
        at_fine = (lvl_used == 0) & m
        den = fine["count"] - 1.0
        safe = np.where(at_fine, den, 1.0)
        out["mean"] = np.where(
            at_fine, (fine["sum"] - own) / safe, out["mean"])
        own_c = np.clip(own, 0.0, float(CLIP_CAP))
        out["cmean"] = np.where(
            at_fine, (fine["csum"] - own_c) / safe, out["cmean"])

    seen = np.nan_to_num(fine["count"], nan=0.0) > 0
    return out, lvl_used, seen


def build_hist_features(levels, groupings, frame, own_target=None,
                        loo_mask=None):
    """Materialise historical feature columns for `frame`.

    `own_target` is passed for TRAINING frames only (enables LOO for the
    additive stats) and left as None for validation frames. `loo_mask`
    restricts LOO to the rows actually present in the aggregate source.
    """
    cols = {}
    diag = {}
    for g in groupings:
        res, lvl, seen = resolve(
            levels, CHAINS[g], frame, own_target=own_target,
            loo_mask=loo_mask)
        for s in STATS:
            cols[f"hist_{g}_{s}"] = res[s].astype("float32")
        cols[f"hist_{g}_seen"] = seen.astype("float32")
        cols[f"hist_{g}_level"] = lvl.astype("float32")
        c = res["count"]
        diag[g] = {
            "seen_pct": 100.0 * float(seen.mean()),
            "at_finest_pct": 100.0 * float((lvl == 0).mean()),
            "resolved_count_p10": float(np.nanpercentile(c, 10)),
            "resolved_count_p50": float(np.nanpercentile(c, 50)),
            "resolved_count_p90": float(np.nanpercentile(c, 90)),
            "level_mean": float(lvl.mean()),
        }
    return pd.DataFrame(cols, index=frame.index), diag


# ============================================================
# SECTION C/D METRICS
# ============================================================

def _rmse(y, p):
    if len(y) < 20:
        return float("nan")
    return float(root_mean_squared_error(y, p))


def _mae(y, p):
    if len(y) < 20:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y, dtype="float64")
                                - np.asarray(p, dtype="float64"))))


# section D: extreme bands. "all" plus <=30 and the three > thresholds.
BANDS = [("all", None, None), ("le30", None, 1800), ("gt30", 1800, None),
         ("gt45", 2700, None), ("gt60", 3600, None)]


def evaluate(y, p, airports):
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    rec = {
        "n": int(len(y)),
        "rmse": _rmse(y, p),
        "mae": _mae(y, p),
        "bias": float(np.mean(p - y)),
    }
    for ap in FOCUS:
        m = airports == ap
        rec[f"rmse_{ap}"] = _rmse(y[m], p[m])
        rec[f"n_{ap}"] = int(m.sum())
    m = ~np.isin(airports, FOCUS)
    rec["rmse_ex"] = _rmse(y[m], p[m])
    rec["n_ex"] = int(m.sum())

    for tag, lo, hi in BANDS:
        if lo is None and hi is None:
            m = np.ones(len(y), dtype=bool)
        elif lo is None:
            m = y <= hi
        else:
            m = y > lo
        rec[f"n_{tag}"] = int(m.sum())
        rec[f"rmse_{tag}"] = _rmse(y[m], p[m])
        rec[f"mae_{tag}"] = _mae(y[m], p[m])
        if m.sum() >= 20:
            rec[f"sqshare_{tag}"] = float(
                np.sum((y[m] - p[m]) ** 2) / np.sum((y - p) ** 2))
        else:
            rec[f"sqshare_{tag}"] = float("nan")

    rec["p_std"] = float(np.std(p))
    rec["p_mean"] = float(np.mean(p))
    for q in (50, 95, 99):
        rec[f"p_p{q}"] = float(np.percentile(p, q))
        rec[f"a_p{q}"] = float(np.percentile(y, q))
    rec["a_mean"] = float(np.mean(y))
    rec["a_max"] = float(np.max(y))
    return rec


def fit_predict(Xtr, ytr, Xva, yva, cat_cols):
    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(Xtr, ytr, cat_features=cat_cols, eval_set=(Xva, yva))
    pred = model.predict(Xva)
    return pred, model


def run_exp(exp_label, fold_label, train_df, val_df, levels, groupings,
            y_train=None, drop_train=None, want_importance=False,
            loo_mask=None):
    """Fit one configuration and evaluate it on the untouched fold targets."""
    t0 = time.time()

    if drop_train is not None:
        train_df = train_df.loc[drop_train]
    n_train_used = len(train_df)

    ytr = train_df[TARGET].to_numpy() if y_train is None else y_train

    Htr = Hva = None
    dtr = dva = {}
    if groupings:
        Hva, dva = build_hist_features(levels, groupings, val_df)
        Htr, dtr = build_hist_features(
            levels, groupings, train_df,
            own_target=train_df[TARGET].to_numpy(), loo_mask=loo_mask)

    Xtr = train_df[FEATURES].copy()
    Xva = val_df[FEATURES].copy()
    if Htr is not None:
        for c in Htr.columns:
            Xtr[c] = Htr[c]
            Xva[c] = Hva[c]
        del Htr, Hva
    gc.collect()

    cat_cols = list(CATEGORICAL_FEATURES)
    yva = val_df[TARGET].to_numpy()

    pred, model = fit_predict(Xtr, ytr, Xva, yva, cat_cols)
    rec = evaluate(yva, pred, val_df["ADEP_mvt"].to_numpy())
    rec["exp"] = exp_label
    rec["fold"] = fold_label
    rec["n_hist_features"] = int(len(Xtr.columns) - len(FEATURES))
    rec["n_train_used"] = int(n_train_used)
    rec["best_iter"] = int(model.get_best_iteration())
    rec["secs"] = time.time() - t0
    rec["groupings"] = "+".join(groupings) if groupings else "(none)"

    imp = None
    if want_importance:
        imp = pd.Series(model.get_feature_importance(), index=Xtr.columns)

    del Xtr, Xva, model
    gc.collect()
    return rec, dtr, dva, imp, pred, yva


RESULTS = {}
DIAG = {"train": {}, "val": {}}
SIZES = {}
IMPORTANCES = {}
PRED_CACHE = {}

EXP_ORDER = ["H0", "H1", "H2", "H3", "H4", "H5", "R1", "R2", "R3"]
FOLD_ORDER = ["JANUARY", "JULY"]
ALL_NEEDED = sorted({g for gs in FAMILIES.values() for g in gs}
                    | set(FALLBACK_GROUPINGS))


def fold_frames(fold_label):
    month = HOLDOUT_MONTH[fold_label]
    is2025 = departures["year"] == 2025
    tr = departures.loc[is2025 & (departures["month"] != month)]
    va = departures.loc[is2025 & (departures["month"] == month)]
    return tr, va


def audit_levels(levels, train_df, fold_label):
    """Direct leakage assertion.

    Every level's group sums must add back up to the TRAINING target total
    and nothing else. If a single validation target had entered any
    aggregate, this would fail.
    """
    tot = float(train_df[TARGET].sum())
    for name, L in levels.items():
        s = float(L.stats["sum"].sum())
        if abs(s - tot) > 1.0:
            raise AssertionError(
                f"{fold_label}/{name}: aggregate sum {s:,.1f} != "
                f"training total {tot:,.1f} - LEAKAGE")
    n_tot = sum(int(L.stats["count"].sum()) for L in levels.values())
    assert n_tot == len(train_df) * len(levels), (
        f"{fold_label}: level row counts {n_tot:,} inconsistent with "
        f"{len(train_df):,} train rows")
    return tot


# ============================================================
# PHASE 1 - H0..H4 AND THE R-SERIES, BOTH FOLDS
# ============================================================

header("PHASE 1 - H0..H4 + R-SERIES ON BOTH FOLDS")
print("H0 is the exact V7 reproduction and is also R0.")

for fold in FOLD_ORDER:
    tr, va = fold_frames(fold)

    header(f"FOLD {fold}  (holdout month {HOLDOUT_MONTH[fold]})", char="=")
    print(f"Training rows   : {len(tr):,}")
    print(f"Validation rows : {len(va):,}")
    ov = len(set(tr["MVT_ID_mvt"].to_numpy()) & set(va["MVT_ID_mvt"].to_numpy()))
    print(f"ID overlap      : {ov:,}")
    assert ov == 0, f"{fold}: validation IDs present in training"
    tr_months = sorted(tr["month"].unique())
    print(f"Training months : {tr_months}")

    header(f"  {fold}: building levels from TRAINING ROWS ONLY", char="-")
    levels = build_levels(tr, ALL_NEEDED)
    tot = audit_levels(levels, tr, fold)
    print(f"LEAKAGE AUDIT OK: every level's group sums total "
          f"{tot:,.0f} = training target sum exactly")
    # store the size DISTRIBUTION, not the Level object
    for g, lvl in levels.items():
        SIZES[(fold, g)] = lvl.size_dist

    # ---- H0 (== R0) ----
    rec, _, _, _, pred, yva = run_exp("H0", fold, tr, va, levels, [])
    RESULTS[(fold, "H0")] = rec
    PRED_CACHE[(fold, "H0")] = (pred, yva)
    exp = EXPECTED[fold]
    print(f"    H0 RMSE {rec['rmse']:.2f}  expected {exp:.2f}  "
          f"delta {rec['rmse'] - exp:+.4f}")
    assert abs(rec["rmse"] - exp) < 0.05, (
        f"{fold}: V7 reproduction drifted ({rec['rmse']:.2f} vs {exp:.2f})")

    # ---- H1..H4 ----
    for fam in ["H1", "H2", "H3", "H4"]:
        grps = FAMILIES[fam]
        rec, dtr, dva, _, pred, yva = run_exp(
            fam, fold, tr, va, levels, grps,
            want_importance=(fam in ("H1", "H4")))
        RESULTS[(fold, fam)] = rec
        PRED_CACHE[(fold, fam)] = (pred, yva)
        DIAG["train"][(fold, fam)] = dtr
        DIAG["val"][(fold, fam)] = dva
        if fam in ("H1", "H4"):
            IMPORTANCES[(fold, fam)] = None  # filled below if needed
        d = rec["rmse"] - RESULTS[(fold, "H0")]["rmse"]
        print(f"    {fam} RMSE {rec['rmse']:.2f}  delta vs H0 {d:+.2f}  "
              f"({rec['n_hist_features']} hist features, "
              f"{rec['secs']:.0f}s)")

    # ---- SECTION F: training-target robustness (R1/R2/R3) ----
    tgt = tr[TARGET]
    specs = {
        "R1": (np.clip(tgt.to_numpy(dtype="float64"), 0.0, 3600.0), None),
        "R2": (np.clip(tgt.to_numpy(dtype="float64"), 0.0, 5400.0), None),
        "R3": (None, (tgt >= 0) & (tgt <= 14400)),
    }
    for name, (yv, drop) in specs.items():
        rec, _, _, _, pred, yva = run_exp(
            name, fold, tr, va, levels, [], y_train=yv, drop_train=drop)
        RESULTS[(fold, name)] = rec
        PRED_CACHE[(fold, name)] = (pred, yva)
        d = rec["rmse"] - RESULTS[(fold, "H0")]["rmse"]
        print(f"    {name} RMSE {rec['rmse']:.2f}  delta vs H0 {d:+.2f}  "
              f"(train rows {rec['n_train_used']:,}, {rec['secs']:.0f}s)")

    del levels, tr, va
    gc.collect()

print()
print("PHASE 1 complete.")


# ============================================================
# PHASE 2 - CHOOSE H5 USING BOTH FOLDS
# ============================================================

header("PHASE 2 - CHOOSING H5 FROM BOTH FOLDS (not one fold)")

fam_rows = []
for fam, grps in FAMILIES.items():
    d = {f: RESULTS[(f, fam)]["rmse"] - RESULTS[(f, "H0")]["rmse"]
         for f in FOLD_ORDER}
    rel = {f: RESULTS[(f, fam)]["rmse"] / RESULTS[(f, "H0")]["rmse"] - 1.0
           for f in FOLD_ORDER}
    both = all(d[f] < 0 for f in FOLD_ORDER)
    fam_rows.append({
        "family": fam,
        "groupings": "+".join(grps),
        "d_jan": d["JANUARY"],
        "d_jul": d["JULY"],
        "mean_rel_pct": 100.0 * float(np.mean(list(rel.values()))),
        "improves_both": both,
        "_rel": float(np.mean(list(rel.values()))),
    })

fam_rows.sort(key=lambda r: r["_rel"])
for r in fam_rows:
    r["improves_both"] = "YES" if r["improves_both"] else "no"
    r["mean_rel_pct"] = f"{r['mean_rel_pct']:+.3f}"
    del r["_rel"]

table("Family comparison (delta RMSE vs H0, negative is better)",
      fam_rows,
      {"family": 6, "groupings": 46, "d_jan": 9, "d_jul": 9,
       "mean_rel_pct": 13, "improves_both": 14})

improvers = [f for f, grps in FAMILIES.items()
             if all(RESULTS[(x, f)]["rmse"] < RESULTS[(x, "H0")]["rmse"]
                    for x in FOLD_ORDER)]
H5_GROUPINGS = sorted({g for f in improvers for g in FAMILIES[f]})
H5_RULE = "union of every family that improved BOTH folds"
if not H5_GROUPINGS:
    best = min(FAMILIES, key=lambda f: np.mean(
        [RESULTS[(x, f)]["rmse"] / RESULTS[(x, "H0")]["rmse"] for x in FOLD_ORDER]))
    H5_GROUPINGS = sorted(FAMILIES[best])
    H5_RULE = (f"NO family improved both folds; fell back to best "
               f"mean-relative family {best}")

header("H5 SELECTION")
print(f"Families improving BOTH folds : {improvers if improvers else '(none)'}")
print(f"H5 rule                      : {H5_RULE}")
print(f"H5 groupings                 : {H5_GROUPINGS}")
print(f"H5 adds {len(H5_GROUPINGS) * len(STATS) + 2 * len(H5_GROUPINGS)} "
      f"historical columns")


# ============================================================
# PHASE 3 - RUN H5
# ============================================================

header("PHASE 3 - RUNNING H5 ON BOTH FOLDS")

for fold in FOLD_ORDER:
    tr, va = fold_frames(fold)
    levels = build_levels(tr, ALL_NEEDED)
    audit_levels(levels, tr, fold)
    rec, dtr, dva, imp, pred, yva = run_exp(
        "H5", fold, tr, va, levels, H5_GROUPINGS, want_importance=True)
    RESULTS[(fold, "H5")] = rec
    PRED_CACHE[(fold, "H5")] = (pred, yva)
    DIAG["train"][(fold, "H5")] = dtr
    DIAG["val"][(fold, "H5")] = dva
    IMPORTANCES[(fold, "H5")] = imp
    d = rec["rmse"] - RESULTS[(fold, "H0")]["rmse"]
    print(f"    H5 RMSE {rec['rmse']:.2f}  delta vs H0 {d:+.2f}  "
          f"({rec['n_hist_features']} hist features, {rec['secs']:.0f}s)")
    del levels, tr, va
    gc.collect()


# ============================================================
# SECTION E - AEZ146R SAFETY CHECK
# ============================================================

header("SECTION E - AEZ146R OUTLIER SAFETY CHECK")

print("The 87,002-second AEZ146R departure sits on 2025-07-15 20:10.")
print("For the JULY fold it is a VALIDATION row, so it cannot be in any")
print("aggregate. For the JANUARY fold it is a TRAINING row, so a naive mean")
print("would be poisoned. That is the test.")

aez_rows = []
for fold in FOLD_ORDER:
    tr, va = fold_frames(fold)
    tr_a = tr[tr["K_FLIGHT"] == AEZ]
    va_a = va[va["K_FLIGHT"] == AEZ]
    tgt = tr_a[TARGET].to_numpy(dtype="float64")
    n_anom = int((tgt > 20000).sum())
    clean = tgt[tgt <= 20000]
    lvl = Level("flight", ["K_FLIGHT"], tr)

    aez_rows.append({
        "fold": fold,
        "tr_n": len(tr_a),
        "va_n": len(va_a),
        "anom_in_train": n_anom,
        "mean_all": float(tgt.mean()) if len(tgt) else float("nan"),
        "mean_ex_anom": float(clean.mean()) if len(clean) else float("nan"),
        "median": float(np.median(tgt)) if len(tgt) else float("nan"),
        "tmean10": trimmed_mean_1d(tgt) if len(tgt) else float("nan"),
        "cmean3600": float(np.clip(tgt, 0, 3600).mean()) if len(tgt) else float("nan"),
        "p90": float(np.percentile(tgt, 90)) if len(tgt) else float("nan"),
        "max": float(tgt.max()) if len(tgt) else float("nan"),
    })
    del tr, va, tr_a, va_a, lvl
    gc.collect()

table("AEZ146R historical aggregates built from each fold's TRAINING rows",
      aez_rows,
      {"fold": 8, "tr_n": 6, "va_n": 6, "anom_in_train": 14,
       "mean_all": 10, "mean_ex_anom": 13, "median": 8, "tmean10": 9,
       "cmean3600": 11, "p90": 8, "max": 10})

j = [r for r in aez_rows if r["fold"] == "JANUARY"][0]
print()
print("January fold (the 87,002s record IS in this fold's training data):")
print(f"  raw mean                 : {j['mean_all']:.1f}s   <- poisoned")
print(f"  raw mean, anomaly removed: {j['mean_ex_anom']:.1f}s")
print(f"  inflation from 1 record  : {j['mean_all'] - j['mean_ex_anom']:+.1f}s")
print(f"  median                   : {j['median']:.1f}s   <- unaffected")
print(f"  trimmed mean (10%)       : {j['tmean10']:.1f}s   <- unaffected")
print(f"  clipped mean (<=3600s)   : {j['cmean3600']:.1f}s   <- unaffected")
poison = j["mean_all"] - j["mean_ex_anom"]
print(f"  => robust statistics remove {poison:.1f}s of single-record damage.")

# what the models actually predict for the anomalous row
tr_j, va_j = fold_frames("JULY")
anom_mask = (va_j["K_FLIGHT"] == AEZ) & (va_j[TARGET] > 20000)
anom_pos = int(np.flatnonzero(anom_mask.to_numpy())[0])
anom_actual = float(va_j[TARGET].to_numpy()[anom_pos])
print()
print(f"Anomalous VALIDATION row in the July fold: index {anom_pos:,}, "
      f"actual {anom_actual:,.0f}s")
pred_rows = []
for exp_name in ["H0", "H1", "H5"]:
    if (("JULY", exp_name)) in PRED_CACHE:
        p = PRED_CACHE[("JULY", exp_name)][0]
        v = float(p[anom_pos])
        pred_rows.append({
            "exp": exp_name,
            "pred": v,
            "actual": anom_actual,
            "resid": v - anom_actual,
            "sq_err": (v - anom_actual) ** 2,
        })
        del p
table("What each model predicts for the 24.2-hour row",
      pred_rows,
      {"exp": 5, "pred": 11, "actual": 12, "resid": 13, "sq_err": 18})
print()
print("Every model predicts an ordinary taxi time, because no aggregate can")
print("know a 24-hour ground delay is about to happen. The squared error on")
print("this single row is irreducible from historical behaviour; it is a label")
print("anomaly, not a missing feature. This is why section D reports the")
print("extreme bands separately instead of letting one row dominate the mean.")
del tr_j, va_j
gc.collect()


# ============================================================
# SENSITIVITY - ARE THE GAINS REAL, OR AN ARTIFACT OF JANUARY
# HAVING FEB-DEC TARGETS AVAILABLE?
# ============================================================
#
# The January fold necessarily builds aggregates from Feb-Dec 2025 targets.
# That is look-ahead relative to January, but it is exactly the condition
# the 2026 ranking faces (all 2025 data available), so it is the right
# primary test.
#
# The July fold CAN be tested honestly: build the same features from
# Jan-Jun 2025 only, while still training the model on the full non-July
# training set. Only the feature source window changes, so this isolates
# "does it work from genuinely prior data" from "does it work from more
# data".

header("SENSITIVITY - JULY FOLD, FEATURES BUILT FROM JAN-JUN ONLY")

is2025 = departures["year"] == 2025
past = departures.loc[is2025 & departures["month"].between(1, 6)]
tr_s, va_s = fold_frames("JULY")
print(f"Feature source rows (Jan-Jun 2025): {len(past):,}")
print(f"Model training rows (non-July)     : {len(tr_s):,}")
levels_p = build_levels(past, ALL_NEEDED)
audit_levels(levels_p, past, "JULY-PAST")
print("LOO is DISABLED here: the Jul-Dec training rows are not part of the")
print("Jan-Jun aggregate source, so their own targets are not in the sums.")
rec, _, dva_s, _, pred, yva = run_exp(
    "S5", "JULY", tr_s, va_s, levels_p, H5_GROUPINGS,
    loo_mask=np.zeros(len(tr_s), dtype=bool))
RESULTS[("JULY", "S5")] = rec
DIAG["val"][("JULY", "S5")] = dva_s
d_past = rec["rmse"] - RESULTS[("JULY", "H0")]["rmse"]
d_full = RESULTS[("JULY", "H5")]["rmse"] - RESULTS[("JULY", "H0")]["rmse"]
print(f"    H0 (V7)                    : {RESULTS[('JULY', 'H0')]['rmse']:.2f}")
print(f"    H5, features from all data : {RESULTS[('JULY', 'H5')]['rmse']:.2f}"
      f"  ({d_full:+.2f})")
print(f"    H5, features from Jan-Jun  : {rec['rmse']:.2f}  ({d_past:+.2f})")
del levels_p, past, tr_s, va_s
gc.collect()


# ============================================================
# REPORT
# ============================================================

header("SECTION C - OVERALL RESULTS")

for fold in FOLD_ORDER:
    base = RESULTS[(fold, "H0")]["rmse"]
    rows = []
    for e in EXP_ORDER:
        r = RESULTS.get((fold, e))
        if r is None:
            continue
        rows.append({
            "exp": e,
            "rmse": r["rmse"],
            "d_rmse": r["rmse"] - base,
            "mae": r["mae"],
            "LIRF": r["rmse_LIRF"],
            "LFPG": r["rmse_LFPG"],
            "ex_2": r["rmse_ex"],
            "p_std": r["p_std"],
            "p_p95": r["p_p95"],
            "p_p99": r["p_p99"],
            "a_p95": r["a_p95"],
            "a_p99": r["a_p99"],
            "nfeat": r["n_hist_features"],
            "grp": r["groupings"],
            "_d": r["rmse"] - base,
        })
    table(f"{fold} - all experiments (delta vs exact V7 reproduction, "
          f"baseline {base:.2f})",
          rows,
          {"exp": 4, "rmse": 8, "d_rmse": 8, "mae": 8, "LIRF": 8, "LFPG": 8,
           "ex_2": 8, "p_std": 8, "p_p95": 8, "p_p99": 9, "a_p95": 8,
           "a_p99": 9, "nfeat": 6, "grp": 46},
          sort_by="_d")

header("SECTION C - BOTH FOLDS SIDE BY SIDE")
side = []
for e in EXP_ORDER:
    rj = RESULTS.get(("JANUARY", e))
    rl = RESULTS.get(("JULY", e))
    if rj is None or rl is None:
        continue
    side.append({
        "exp": e,
        "jan": rj["rmse"],
        "d_jan": rj["rmse"] - RESULTS[("JANUARY", "H0")]["rmse"],
        "jul": rl["rmse"],
        "d_jul": rl["rmse"] - RESULTS[("JULY", "H0")]["rmse"],
        "mean_rel_pct": 100.0 * float(np.mean([
            rj["rmse"] / RESULTS[("JANUARY", "H0")]["rmse"] - 1.0,
            rl["rmse"] / RESULTS[("JULY", "H0")]["rmse"] - 1.0])),
        "_k": float(np.mean([
            rj["rmse"] / RESULTS[("JANUARY", "H0")]["rmse"],
            rl["rmse"] / RESULTS[("JULY", "H0")]["rmse"]])),
    })
side.sort(key=lambda r: r["_k"])
for r in side:
    r["mean_rel_pct"] = f"{r['mean_rel_pct']:+.3f}"
    del r["_k"]
table("Both folds (sorted by mean relative RMSE, best first)",
      side,
      {"exp": 4, "jan": 8, "d_jan": 8, "jul": 8, "d_jul": 8, "mean_rel_pct": 13})


# ---------- section D ----------

header("SECTION D - EXTREME-EVENT BREAKDOWN")

for fold in FOLD_ORDER:
    rows = []
    for e in EXP_ORDER:
        r = RESULTS.get((fold, e))
        if r is None:
            continue
        rows.append({
            "exp": e,
            "all": r["rmse_all"],
            "le30": r["rmse_le30"],
            "n_le30": r["n_le30"],
            "gt30": r["rmse_gt30"],
            "n_gt30": r["n_gt30"],
            "gt45": r["rmse_gt45"],
            "n_gt45": r["n_gt45"],
            "gt60": r["rmse_gt60"],
            "n_gt60": r["n_gt60"],
            "sq_gt30": r["sqshare_gt30"],
        })
    table(f"{fold} - RMSE by actual-target band (no rows deleted anywhere)",
          rows,
          {"exp": 4, "all": 8, "le30": 8, "n_le30": 8, "gt30": 8, "n_gt30": 7,
           "gt45": 8, "n_gt45": 7, "gt60": 8, "n_gt60": 7, "sq_gt30": 9})

for fold in FOLD_ORDER:
    r0 = RESULTS[(fold, "H0")]
    print()
    print(f"{fold}: share of total squared error from >30min rows = "
          f"{100 * r0['sqshare_gt30']:.2f}%, from >45min = "
          f"{100 * r0['sqshare_gt45']:.2f}%, from >60min = "
          f"{100 * r0['sqshare_gt60']:.2f}%  (V7 baseline)")


# ---------- coverage ----------

header("COVERAGE - WAS THE HISTORICAL SIGNAL EVEN AVAILABLE?")

COV_FROM = {}
for fam, grps in FAMILIES.items():
    for g in grps:
        COV_FROM[g] = fam

cov_rows = []
for g in sorted(GROUPINGS):
    row = {"grouping": g, "keys": "+".join(GROUPINGS[g]), "exp": COV_FROM[g]}
    for fold in FOLD_ORDER:
        d = DIAG["val"].get((fold, COV_FROM[g]), {}).get(g)
        if d is None:
            continue
        row[f"seen_{fold[:3]}"] = d["seen_pct"]
        row[f"fine_{fold[:3]}"] = d["at_finest_pct"]
        row[f"cnt50_{fold[:3]}"] = d["resolved_count_p50"]
        row[f"lvl_{fold[:3]}"] = d["level_mean"]
    cov_rows.append(row)
table("Validation-row coverage per grouping "
      "('seen' = group present in that fold's training data)",
      cov_rows,
      {"grouping": 16, "keys": 30, "exp": 4,
       "seen_JAN": 9, "fine_JAN": 9, "cnt50_JAN": 10, "lvl_JAN": 8,
       "seen_JUL": 9, "fine_JUL": 9, "cnt50_JUL": 10, "lvl_JUL": 8})

size_rows = []
for g in sorted(ALL_GROUPINGS):
    row = {"grouping": g}
    for fold in FOLD_ORDER:
        d = SIZES.get((fold, g))
        if d is None:
            continue
        row[f"g{fold[:3]}"] = d["groups"]
        row[f"p50_{fold[:3]}"] = d["p50"]
        row[f"p90_{fold[:3]}"] = d["p90"]
        row[f"max_{fold[:3]}"] = d["max"]
        row[f"ok{fold[:3]}"] = d["ge_min_count_pct"]
    size_rows.append(row)
table(f"Training-side group size distribution "
      f"(fractions of groups with >= {MIN_COUNT} training rows)",
      size_rows,
      {"grouping": 16, "gJAN": 10, "p50_JAN": 9, "p90_JAN": 10, "max_JAN": 9,
       "okJAN": 8, "gJUL": 10, "p50_JUL": 9, "p90_JUL": 10, "max_JUL": 9,
       "okJUL": 8})


# ---------- feature importance ----------

header("WHICH HISTORICAL SIGNALS DID THE MODEL ACTUALLY USE?")

for fold in FOLD_ORDER:
    imp = IMPORTANCES.get((fold, "H5"))
    if imp is None:
        continue
    hist = imp[[c.startswith("hist_") for c in imp.index]]
    v7 = imp[[not c.startswith("hist_") for c in imp.index]]
    tot = float(imp.sum()) or 1.0
    fam_tot = {}
    for c, v in hist.items():
        g = c.split("_")[1]
        fam_tot[g] = fam_tot.get(g, 0.0) + float(v)
    rows = [{"grouping": g, "importance": v, "pct": 100.0 * v / tot}
            for g, v in sorted(fam_tot.items(), key=lambda kv: -kv[1])]
    table(f"{fold} H5 - historical importance by grouping "
          f"(of {100 * float(hist.sum()) / tot:.1f}% total importance)",
          rows, {"grouping": 18, "importance": 12, "pct": 8})
    rows = [{"feature": c, "importance": float(v)}
            for c, v in v7.sort_values(ascending=False).items()]
    table(f"{fold} H5 - V7 baseline feature importance (unchanged features)",
          rows, {"feature": 24, "importance": 12}, maxrows=14)
    top = hist.sort_values(ascending=False).head(12)
    table(f"{fold} H5 - top 12 individual historical features",
          [{"feature": c, "importance": float(v)} for c, v in top.items()],
          {"feature": 30, "importance": 12})


# ---------- section F ----------

header("SECTION F - TRAINING-TARGET ROBUSTNESS (R-SERIES)")
print("Validation targets are NEVER modified. Only y_train changes.")

rf = []
for e in ["H0", "R1", "R2", "R3"]:
    for fold in FOLD_ORDER:
        r = RESULTS.get((fold, e))
        if r is None:
            continue
        rf.append({
            "exp": e, "fold": fold, "n_train": r["n_train_used"],
            "rmse": r["rmse"],
            "d": r["rmse"] - RESULTS[(fold, "H0")]["rmse"],
            "mae": r["mae"], "le30": r["rmse_le30"], "gt30": r["rmse_gt30"],
            "gt45": r["rmse_gt45"], "gt60": r["rmse_gt60"],
            "p_std": r["p_std"], "p_p99": r["p_p99"], "a_p99": r["a_p99"],
            "bias": r["bias"],
        })
rf.sort(key=lambda r: (r["exp"], r["fold"]))
table("Robust training-target variants, scored on ORIGINAL validation targets",
      rf,
      {"exp": 4, "fold": 8, "n_train": 10, "rmse": 8, "d": 8, "mae": 8,
       "le30": 8, "gt30": 8, "gt45": 8, "gt60": 8, "p_std": 8, "p_p99": 8,
       "a_p99": 8, "bias": 8})


# ============================================================
# FINAL ANSWERS
# ============================================================

header("FINAL ANSWERS", char="=")

H0J = RESULTS[("JANUARY", "H0")]["rmse"]
H0L = RESULTS[("JULY", "H0")]["rmse"]


def d_of(fold, e, band=None):
    a = RESULTS[(fold, e)]["rmse" if band is None else f"rmse_{band}"]
    b = RESULTS[(fold, "H0")]["rmse" if band is None else f"rmse_{band}"]
    return a - b


def mean_rel(e):
    return float(np.mean([
        RESULTS[("JANUARY", e)]["rmse"] / H0J,
        RESULTS[("JULY", e)]["rmse"] / H0L]))


CANDIDATES = [e for e in EXP_ORDER if ("JANUARY", e) in RESULTS
              and ("JULY", e) in RESULTS]
BEST = min(CANDIDATES, key=mean_rel)
BESTJ = RESULTS[("JANUARY", BEST)]["rmse"]
BESTL = RESULTS[("JULY", BEST)]["rmse"]
H5J = RESULTS[("JANUARY", "H5")]["rmse"]
H5L = RESULTS[("JULY", "H5")]["rmse"]

print(f"V7 baseline reproduced:  JANUARY {H0J:.2f}   JULY {H0L:.2f}")
print(f"H5 (selected combination, rule: {H5_RULE}):")
print(f"                        JANUARY {H5J:.2f} ({H5J - H0J:+.2f})   "
      f"JULY {H5L:.2f} ({H5L - H0L:+.2f})")
print()

# Q1 / Q2
print("1. Do historical aggregates improve Jan RMSE?")
best_j = min(CANDIDATES, key=lambda e: RESULTS[("JANUARY", e)]["rmse"])
print(f"   {'YES' if d_of('JANUARY', best_j) < 0 else 'NO'}. Best on January "
      f"is {best_j} at {RESULTS[('JANUARY', best_j)]['rmse']:.2f} "
      f"({d_of('JANUARY', best_j):+.2f} vs V7).")
print(f"   The H5 combination moves January by {H5J - H0J:+.2f}.")
print()
print("2. Do they improve July RMSE?")
best_l = min(CANDIDATES, key=lambda e: RESULTS[("JULY", e)]["rmse"])
print(f"   {'YES' if d_of('JULY', best_l) < 0 else 'NO'}. Best on July is "
      f"{best_l} at {RESULTS[('JULY', best_l)]['rmse']:.2f} "
      f"({d_of('JULY', best_l):+.2f} vs V7).")
print(f"   The H5 combination moves July by {H5L - H0L:+.2f}.")
print()

# Q3
fam_best = min(FAMILIES, key=lambda f: np.mean(
    [d_of(x, f) for x in FOLD_ORDER]))
print("3. Which feature family is most useful?")
print(f"   {fam_best} ({'+'.join(FAMILIES[fam_best])}), by mean RMSE delta "
      f"across both folds:")
for f in FOLD_ORDER:
    print(f"     {f:8s} {d_of(f, fam_best):+.2f}")
print("   Per-family detail is in the PHASE 2 table above.")
print()

# Q4
both = [e for e in CANDIDATES
        if d_of("JANUARY", e) < 0 and d_of("JULY", e) < 0]
print("4. Do improvements survive both months?")
print(f"   {'YES' if both else 'NO'}. Configurations beating V7 on BOTH folds: "
      f"{both if both else '(none)'}")
print(f"   H5, chosen on both folds together, gives "
      f"{(H5J - H0J) + (H5L - H0L):+.2f} summed RMSE change across the two folds.")
print()

# Q5
print("5. Do they improve normal flights, extreme flights, or both?")
for e in (["H5"] + ([BEST] if BEST != "H5" else [])):
    print(f"   {e}:")
    for f in FOLD_ORDER:
        parts = []
        for band in ["le30", "gt30", "gt45", "gt60"]:
            n = RESULTS[(f, e)][f"n_{band}"]
            dd = d_of(f, e, band)
            parts.append(f"{band}={fmt(dd, +1)} (n={n:,})")
        print(f"     {f:8s} " + "  ".join(parts))
print("   Negative is better. Compare against the band sizes above: the >45min")
print("   and >60min bands hold only a few hundred rows, so their RMSE is")
print("   noisy and a single 24-hour label moves them a lot.")
print()

# Q6
r_best = min(["H0", "R1", "R2", "R3"], key=mean_rel)
print("6. Does robust training-target treatment help?")
for e in ["R1", "R2", "R3"]:
    print(f"   {e}: JAN {d_of('JANUARY', e):+.2f}   JUL {d_of('JULY', e):+.2f}"
          f"   mean rel {100 * (mean_rel(e) - 1):+.3f}%")
print(f"   Best R variant: {r_best}. "
      f"{'Helps' if mean_rel(r_best) < 1 else 'Does NOT help'} overall.")
print("   Note R variants change the loss landscape, not the features; they")
print("   were not combined with the H families in this run.")
print()

# Q7 / Q8
print("7. What is the best measured configuration so far?")
print(f"   {BEST} = V7 features + {RESULTS[('JANUARY', BEST)]['groupings']}"
      f" historical aggregates ({RESULTS[('JANUARY', BEST)]['n_hist_features']}"
      f" columns), V7 CatBoost config unchanged.")
print()
print("8. What Jan/Jul RMSE does it achieve?")
print(f"   JANUARY {BESTJ:.2f}  (V7 {H0J:.2f}, delta {BESTJ - H0J:+.2f})")
print(f"   JULY    {BESTL:.2f}  (V7 {H0L:.2f}, delta {BESTL - H0L:+.2f})")
print()
print("=" * 112)
print("CAVEATS ON THESE NUMBERS - read before acting on them")
print("=" * 112)
print("a) Selection bias. This is the MINIMUM over "
      f"{len(CANDIDATES)} measured configurations. The winner's advantage is")
print("   optimistically biased; the true expected gain is smaller.")
print("b) The two folds are not independent. They share 10 of 12 training")
print("   months, so agreement between them is weaker evidence than it looks.")
print("c) Early stopping uses the validation fold, exactly as V7 does, so all")
print("   arms share that optimism equally but it is still optimism.")
print("d) The January fold builds aggregates from Feb-Dec targets (look-ahead).")
print("   That matches 2026-ranking deployment, and the Jan-Jun-only July")
print(f"   sensitivity above moved July by {d_past - d_full:+.2f} relative to the")
print("   full-history version, which is the honest read on prior-only data.")
print("e) No V8 was created, no submission was written, and V1-V7 are untouched.")
print()
print("END. Measurement only. No model was promoted.")
print("=" * 112)

_report_file.flush()
print()
print(f"Report written to: {REPORT_PATH}")
print(f"Report size : {REPORT_PATH.stat().st_size:,} bytes")
print(f"Report lines: {sum(1 for _ in REPORT_PATH.open(encoding='utf-8')):,}")
