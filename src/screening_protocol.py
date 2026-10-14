"""Reusable fast-screening protocol for PRC taxi-time experiments.

Three stages, so nothing defaults to a 30-90 minute full Jan+Jul run:

    STAGE A  SMOKE   ~10% of each outer-training fold. Implementation and
                     integrity check. Seconds.
    STAGE B  SCREEN  ~25% of each outer-training fold. Representative enough
                     to kill obviously weak ideas. Minutes.
    STAGE C  FULL    the established full Jan+JUL protocol. Only FULL results
                     may become architecture changes or submissions.

A reduced run is NEVER sufficient evidence to adopt a model.

SAMPLING RULE (the important part)
    Reduced runs must stay representative of the real problem, so:
      * every feature that depends on the full movement stream
        (traffic_last_15min) is built on ALL rows BEFORE any sampling;
      * training departures are stratified by
        month | ADEP | reference_state | target severity bin, and the SAME
        fraction is taken inside every stratum, so tail prevalence is
        preserved rather than artificially inflated;
      * selection is a deterministic 64-bit hash of MVT_ID, so the samples are
        identical forever and SMOKE is a strict subset of SCREEN;
      * validation is never subsampled. Screening reduces TRAINING only, so
        the rare and extreme cases still dominate the metric;
      * the target bins are computed from TRAINING targets only, which is
        label-safe because they are used solely to choose training rows.

    Nothing is ever selected for being "good" or "extreme".

MISSING_4 FAST PATH
    A MISSING_4-only experiment changes nothing outside the specialists, so
    the giant COMPLETE/Exp04 model must not be retrained. Reuse the frozen
    baseline cache below and fit only the candidate specialists. This is the
    default for every future MISSING_4-only experiment.

RUNTIME CACHE RULE
    compute() and finalize_report() are separate. --report-only regenerates a
    report from the cache. A wording, formatting, key, table, plotting or file
    writing defect is a REPORTING-ONLY BUG and must never trigger a refit; only
    an explicitly declared COMPUTATION-AFFECTING BUG may retrain.

Usage
    python src/screening_protocol.py                 # build cache, write audit
    python src/screening_protocol.py --report-only   # audit from cache only
"""

from __future__ import annotations

import gc
import hashlib
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from train_catboost_v8_defs import (  # noqa: E402
    MINUEND, T4_SPEC, TARGET, add_t4_features, add_time_features,
    add_traffic_feature, apply_categorical_handling,
)

# ---------------------------------------------------------------- config
DATA_DIR = Path("data/raw")
REPORT = Path("reports/screening_protocol_audit.txt")
CACHE_DIR = Path(os.environ.get("TEMP", ".")) / "opencode" / "prc_cache"
BASELINE_CACHE = CACHE_DIR / "baseline_cache.pkl"
SAMPLE_CACHE = CACHE_DIR / "sample_ids.pkl"

FOLDS = {"JAN": ("2025-01-01", "2025-02-01"), "JUL": ("2025-07-01", "2025-08-01")}

SMOKE_FRAC = 0.10
SCREEN_FRAC = 0.25
SAMPLE_SALT = "prc-screening-v1"  # frozen forever; changing it re-rolls samples
SAMPLE_VERSION = 1               # bump only if the sampling rule itself changes

# Frozen reference results the cached baseline must reproduce.
EXP04_REF = {"JAN": 402.5369401, "JUL": 509.0817784, "pooled": 464.5623995}
F3_REF = {"JAN": 372.462528, "JUL": 392.441579, "pooled": 383.653977}
EXP04_TOL = 5e-7

# Where the already-validated full-fold predictions live. Populated by the
# F3-C run, whose Exp04 and plain-F3 columns reproduced these same references
# to ~1e-10 relative.
F3C_CACHE = (Path(os.environ.get("TEMP", ".")) / "opencode"
             / "f3_missing4_weather_cache.pkl")

# Target severity bins, on TRAINING targets only.
BIN_EDGES = [-np.inf, 0.0, 1800.0, 3600.0, 7200.0, np.inf]
BIN_LABELS = ["<=0", "0-1800", "1800-3600", "3600-7200", ">7200"]

_buf: list[str] = []


def head(t):
    _buf.append("")
    _buf.append("=" * 100)
    _buf.append(t)
    _buf.append("=" * 100)
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def note(t=""):
    _buf.append(t)
    print(t)


def plain(t=""):
    """Print only. Keeps wall-clock timings out of the report so that
    --report-only reproduces it byte for byte."""
    print(t)


def flush():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")


# ------------------------------------------------------- deterministic hash
def id_to_int(ids):
    """MVT_ID -> int64. Whole-numbered float IDs are exact in int64."""
    a = np.asarray(ids)
    if a.dtype.kind == "f":
        assert np.all(np.isfinite(a)), "non-finite MVT_ID"
        assert np.all(a == np.floor(a)), "MVT_ID is not whole-numbered"
    return a.astype(np.int64)


def stable_hash(ids, salt: str = SAMPLE_SALT) -> np.ndarray:
    """SplitMix64 over MVT_ID. Stable forever, unlike Python's hash().

    Ordering by this value is the whole selection mechanism, so the samples
    never move and SMOKE nests inside SCREEN for free.
    """
    x = (id_to_int(ids).astype(np.uint64)
         ^ np.uint64(int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)))
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


# ------------------------------------------------------------- strata / bins
def severity_bin(y) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    out = np.full(len(y), BIN_LABELS[-1], dtype=object)
    out[(y > 3600.0) & (y <= 7200.0)] = "3600-7200"
    out[(y > 1800.0) & (y <= 3600.0)] = "1800-3600"
    out[(y > 0.0) & (y <= 1800.0)] = "0-1800"
    out[y <= 0.0] = "<=0"
    return out.astype(str)


def stratum_key(df: pd.DataFrame) -> np.ndarray:
    """month | ADEP | reference_state | target severity bin."""
    return (df["month"].astype(int).astype(str) + "|"
            + df["ADEP_mvt"].astype(str) + "|"
            + df["reference_state"].astype(str) + "|"
            + severity_bin(df[TARGET].to_numpy()))


def select_stratified(df: pd.DataFrame, frac: float) -> np.ndarray:
    """Take the same fraction inside every stratum, chosen by hash order.

    Floor, no per-stratum minimum: forcing a minimum would over-represent tiny
    strata and inflate tail prevalence. The audit reports any stratum that a
    reduced run empties, so the distortion is visible rather than silent.
    """
    h = stable_hash(df["MVT_ID_mvt"])
    key = stratum_key(df).to_numpy()
    order = np.lexsort((h, key))       # within a stratum, ascending hash
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    sizes = np.r_[np.diff(starts), len(ks) - starts[-1]]
    take = np.floor(frac * sizes).astype(np.int64)
    rank = np.arange(len(ks)) - np.repeat(starts, sizes)
    sel = rank < np.repeat(take, sizes)
    return id_to_int(df["MVT_ID_mvt"].to_numpy())[order[sel]]


def ids_digest(ids) -> str:
    return hashlib.sha256(np.sort(id_to_int(ids)).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------- frame
def build_frame() -> pd.DataFrame:
    """Full departure frame with every stream-dependent feature already built.

    traffic_last_15min is computed on all 2,085,047 departures first, because
    it is a function of the surrounding movement stream and would be wrong if
    computed after subsampling.
    """
    cols = list(dict.fromkeys(
        [MINUEND] + [c for _, c in T4_SPEC] + [
            "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
            "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "WK_TBL_CAT_flt",
            "FLIGHT_TYPE_flt", "FLIGHT_mvt", TARGET]))
    files = sorted(DATA_DIR.glob("training_*.parquet"))
    assert len(files) == 12, f"expected 12 training months, found {len(files)}"
    data = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                     ignore_index=True)
    pre = set(data["MVT_ID_mvt"].unique())
    data = add_traffic_feature(data)
    assert pre == set(data["MVT_ID_mvt"].unique()), "traffic reorder lost IDs"
    dep = data[data["PHASE_mvt"] == "DEP"].copy()
    del data
    dep = add_time_features(dep)
    dep = add_t4_features(dep)
    dep = apply_categorical_handling(dep)
    assert len(dep) == 2_085_047, len(dep)
    assert dep["MVT_ID_mvt"].is_unique
    dep["FLIGHT_mvt"] = dep["FLIGHT_mvt"].fillna("UNKNOWN").astype(str)
    dep["reference_state"] = np.where(
        dep[["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]].isna().all(axis=1),
        "MISSING_4", "COMPLETE")
    return dep


# ------------------------------------------------------------ baseline cache
def build_baseline_cache(dep: pd.DataFrame) -> dict:
    """Promote the validated F3-C full-fold predictions into the permanent
    baseline cache. No CatBoost runs: the Exp04 and plain-F3 columns of that
    run already reproduced the frozen references to ~1e-10 relative, so
    refitting them would only risk drift."""
    assert F3C_CACHE.exists(), (
        f"no validated prediction source at {F3C_CACHE}. Re-run the F3-C "
        "experiment to regenerate the frozen Exp04 and plain-F3 predictions.")
    note(f"  reusing validated predictions from {F3C_CACHE}")
    note(f"    ({F3C_CACHE.stat().st_size / 1e6:.1f} MB)")
    with open(F3C_CACHE, "rb") as fh:
        src = pickle.load(fh)
    assert src.get("wx_verified") is True

    meta = dep.set_index("MVT_ID_mvt")
    cache = {
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": {
            "source": str(F3C_CACHE),
            "note": ("Exp04 and plain-F3 full-fold validation predictions from the "
                     "F3-C run; both reproduced the frozen references to ~1e-10 "
                     "relative and passed all safety checks."),
            "exp04_ref": EXP04_REF, "f3_ref": F3_REF,
        },
        "folds": {},
    }
    for f in ("JAN", "JUL"):
        S = src["folds"][f]
        ids, y, vmask = S["ids"], S["y"], S["vmask"]
        tab = S["m4tab"]
        # plain F3 = Exp04 on COMPLETE, specialist on MISSING_4
        p_f3 = pd.Series(S["p04"], index=ids, dtype="float64")
        p_f3.loc[tab["MVT_ID_mvt"].to_numpy()] = tab["p_plain"].to_numpy()
        p_f3 = p_f3.to_numpy()
        m = meta.loc[ids]
        assert np.array_equal(id_to_int(m.index.to_numpy()), id_to_int(ids)), \
            f"{f}: row-order alignment between the frame and the predictions"
        assert len(ids) == len(y) == len(p_f3)
        cache["folds"][f] = dict(
            ids=id_to_int(ids), y=y,
            adep=m["ADEP_mvt"].to_numpy().astype(str),
            dsched=m["d_SCHED"].to_numpy(dtype="float64"),
            month=m["month"].to_numpy(dtype="int16"),
            ref_state=m["reference_state"].to_numpy().astype(str),
            vmask=vmask,
            p04=np.asarray(S["p04"], dtype="float64"), p_f3=p_f3,
            it_exp04=int(S["exp04_iter"]),
            it_f3={k: int(v) for k, v in S["it_plain"].items()},
        )
    return cache


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def sse(y, p):
    return float(((np.asarray(y) - np.asarray(p)) ** 2).sum())


def pooled(cache, key):
    num = den = 0.0
    for f in ("JAN", "JUL"):
        F = cache["folds"][f]
        num += sse(F["y"], F[key])
        den += len(F["y"])
    return float(np.sqrt(num / den)), num


# ------------------------------------------------------------------- audit
def dist_row(label, d) -> dict:
    y = d[TARGET].to_numpy(dtype="float64")
    return dict(
        label=label, n=len(y),
        mean=float(y.mean()), median=float(np.median(y)),
        p95=float(np.quantile(y, 0.95)), p99=float(np.quantile(y, 0.99)),
        n3600=int((y > 3600).sum()), r3600=float((y > 3600).mean()),
        n7200=int((y > 7200).sum()), r7200=float((y > 7200).mean()),
        m4_rate=float((d["reference_state"] == "MISSING_4").mean()),
        adep=d["ADEP_mvt"].value_counts(normalize=True).to_dict(),
        month=d["month"].value_counts(normalize=True).sort_index().to_dict(),
    )


def report_dist(f, pairs):
    rows = [dist_row(lbl, d) for lbl, d in pairs]
    note(f"  {'set':<12}{'n':>9}{'mean':>11}{'median':>10}{'p95':>11}"
         f"{'p99':>12}{'>3600 n':>10}{'>3600 r':>10}{'>7200 n':>10}"
         f"{'>7200 r':>10}{'M4 rate':>10}")
    for r in rows:
        note(f"  {r['label']:<12}{r['n']:>9,}{r['mean']:>11.1f}{r['median']:>10.1f}"
             f"{r['p95']:>11.1f}{r['p99']:>12.1f}{r['n3600']:>10,}"
             f"{r['r3600']:>9.3%} {r['n7200']:>10,}{r['r7200']:>10.4%}"
             f"{r['m4_rate']:>10.4%}")
    base = rows[0]
    note("")
    note(f"  {'set':<12}{'d mean%':>10}{'d median%':>12}{'d p95%':>10}"
         f"{'d p99%':>10}{'d >3600r pp':>14}{'d >7200r pp':>14}"
         f"{'d M4 rate pp':>14}{'max adep d pp':>15}{'max month d pp':>16}")
    out = {}
    for r in rows[1:]:
        dmean = 100 * (r["mean"] - base["mean"]) / base["mean"]
        dmed = 100 * (r["median"] - base["median"]) / base["median"]
        dp95 = 100 * (r["p95"] - base["p95"]) / base["p95"]
        dp99 = 100 * (r["p99"] - base["p99"]) / base["p99"]
        d36 = 100 * (r["r3600"] - base["r3600"])
        d72 = 100 * (r["r7200"] - base["r7200"])
        dm4 = 100 * (r["m4_rate"] - base["m4_rate"])
        ka = set(base["adep"]) | set(r["adep"])
        da = max(abs(100 * (r["adep"].get(k, 0.0) - base["adep"].get(k, 0.0)))
                 for k in ka)
        km = set(base["month"]) | set(r["month"])
        dm = max(abs(100 * (r["month"].get(k, 0.0) - base["month"].get(k, 0.0)))
                 for k in km)
        out[r["label"]] = dict(dmean=dmean, dmed=dmed, dp95=dp95, dp99=dp99,
                               d36=d36, d72=d72, dm4=dm4, da=da, dm=dm)
        note(f"  {r['label']:<12}{dmean:>9.2f}%{dmed:>11.2f}%{dp95:>9.2f}%"
             f"{dp99:>9.2f}%{d36:>13.3f} {d72:>13.4f} {dm4:>13.3f} "
             f"{da:>14.3f} {dm:>15.3f}")
    return out


# ======================================================================
#                              COMPUTE
# ======================================================================
def compute():
    t0 = time.time()
    plain("building the frame to derive the validated baseline cache")
    dep = build_frame()
    plain(f"  frame {len(dep):,} departures in {time.time() - t0:.0f}s")
    plain(f"  MISSING_4 {int((dep['reference_state'] == 'MISSING_4').sum()):,}")
    cache = build_baseline_cache(dep)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_CACHE, "wb") as fh:
        pickle.dump(cache, fh, protocol=4)
    plain(f"  wrote {BASELINE_CACHE} "
          f"({BASELINE_CACHE.stat().st_size / 1e6:.1f} MB)")
    del dep
    gc.collect()
    return cache


def build_samples(dep: pd.DataFrame) -> dict:
    out = {}
    for f, (lo, hi) in FOLDS.items():
        m = ((dep[MINUEND] >= lo) & (dep[MINUEND] < hi)).to_numpy()
        tr = dep[~m]
        smoke = select_stratified(tr, SMOKE_FRAC)
        screen = select_stratified(tr, SCREEN_FRAC)
        assert set(smoke.tolist()) <= set(screen.tolist()), \
            f"{f}: SMOKE is not a subset of SCREEN"
        out[f] = dict(
            smoke_ids=smoke, screen_ids=screen,
            smoke_hash=ids_digest(smoke), screen_hash=ids_digest(screen),
            n_full=len(tr), n_smoke=len(smoke), n_screen=len(screen),
            n_strata=int(pd.Series(stratum_key(tr)).nunique()),
        )
    return out


def load_samples(dep: pd.DataFrame, resample: bool = False) -> dict:
    """Frozen sample IDs, persisted so the selection is never re-rolled.

    The IDs are a pure function of MVT_ID, the strata and the salt, so loading
    them and recomputing them are equivalent; the cache only saves the ~40 s
    selection. --verify-samples proves the two agree.
    """
    if SAMPLE_CACHE.exists() and not resample:
        with open(SAMPLE_CACHE, "rb") as fh:
            blob = pickle.load(fh)
        assert blob["salt"] == SAMPLE_SALT, "sample salt changed; resample"
        assert blob["version"] == SAMPLE_VERSION, "sample protocol changed; resample"
        plain(f"  sample IDs loaded from {SAMPLE_CACHE} (salt "
              f"'{blob['salt']}', v{blob['version']})")
        return blob["folds"]
    samples = build_samples(dep)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_CACHE, "wb") as fh:
        pickle.dump(dict(version=SAMPLE_VERSION, salt=SAMPLE_SALT,
                         folds=samples), fh, protocol=4)
    plain(f"  wrote {SAMPLE_CACHE} "
          f"({SAMPLE_CACHE.stat().st_size / 1e6:.1f} MB)")
    return samples


def verify_samples(dep: pd.DataFrame, samples: dict) -> None:
    fresh = build_samples(dep)
    for f in FOLDS:
        for k in ("smoke_hash", "screen_hash", "n_smoke", "n_screen", "n_strata"):
            assert fresh[f][k] == samples[f][k], \
                f"{f}: {k} drifted ({samples[f][k]} cached vs {fresh[f][k]} fresh)"
        assert np.array_equal(np.sort(fresh[f]["smoke_ids"]),
                              np.sort(samples[f]["smoke_ids"])), \
            f"{f}: SMOKE IDs drifted"
        assert np.array_equal(np.sort(fresh[f]["screen_ids"]),
                              np.sort(samples[f]["screen_ids"])), \
            f"{f}: SCREEN IDs drifted"
    plain("  [PASS] cached sample IDs reproduce exactly from MVT_ID + salt")


# ======================================================================
#                              REPORT
# ======================================================================
def finalize_report(cache, samples):
    ok_all = True

    head("1. BASELINE CACHE (reused, never refitted to populate)")
    note(f"  path      {BASELINE_CACHE}")
    note(f"  size      {BASELINE_CACHE.stat().st_size / 1e6:.1f} MB"
         f"     built {cache['created']}  (v{cache['version']})")
    note(f"  source    {cache['provenance']['source']}")
    note("            " + cache["provenance"]["note"])
    note("")
    note(f"  {'fold':<6}{'rows':>10}{'COMPLETE':>11}{'MISSING_4':>11}"
         f"{'airports':>10}{'Exp04 iters':>14}")
    for f in ("JAN", "JUL"):
        F = cache["folds"][f]
        n4 = int(F["vmask"].sum())
        note(f"  {f:<6}{len(F['ids']):>10,}{len(F['ids']) - n4:>11,}{n4:>11,}"
             f"{len(F['it_f3']):>10}{F['it_exp04']:>14,}")
    note("")
    note("  Per-airport plain-F3 specialist tree counts:")
    for f in ("JAN", "JUL"):
        it = cache["folds"][f]["it_f3"]
        note(f"    {f}  " + "  ".join(f"{a}:{it[a]}" for a in sorted(it)))
    note("")
    note("  No CatBoost was fit to populate this cache. The Exp04 and plain-F3")
    note("  columns were carried over from the F3-C run, which reproduced these")
    note("  same frozen references to ~1e-10 relative and passed all its safety")
    note("  checks, so refitting could only reintroduce drift.")
    flush()

    head("2. CACHED BASELINE VERIFICATION (must reproduce the frozen numbers)")
    note(f"  {'fold':<6}{'Exp04 cached':>16}{'Exp04 frozen':>16}{'rel':>11}"
         f"{'plain F3 cached':>19}{'plain F3 frozen':>18}{'rel':>11}")
    for f in ("JAN", "JUL"):
        F = cache["folds"][f]
        r4 = rmse(F["y"], F["p04"])
        r3 = rmse(F["y"], F["p_f3"])
        d4, d3 = abs(r4 - EXP04_REF[f]) / EXP04_REF[f], abs(r3 - F3_REF[f]) / F3_REF[f]
        ok = d4 <= EXP04_TOL and d3 <= EXP04_TOL
        ok_all &= ok
        note(f"  {f:<6}{r4:>16.7f}{EXP04_REF[f]:>16.7f}{d4:>10.1e}"
             f"{r3:>19.6f}{F3_REF[f]:>18.6f}{d3:>10.1e}   "
             f"[{'PASS' if ok else 'FAIL'}]")
    p4, s4 = pooled(cache, "p04")
    p3, s3 = pooled(cache, "p_f3")
    for nm, got, ref, tol in (("Exp04", p4, EXP04_REF["pooled"], EXP04_TOL),
                              ("plain F3", p3, F3_REF["pooled"], EXP04_TOL)):
        d = abs(got - ref) / ref
        ok = d <= tol
        ok_all &= ok
        note(f"  {nm} pooled {got:.7f}  frozen {ref:.7f}  rel {d:.1e}  "
             f"[{'PASS' if ok else 'FAIL'}]   (SSE {s4 if nm == 'Exp04' else s3:.4e})")
    note("")
    note(f"  rows: " + "  ".join(
        f"{f} {len(cache['folds'][f]['ids']):,}" for f in ("JAN", "JUL")))
    note("  Exp04 is better than plain F3? "
         f"{'yes (unexpected)' if p4 < p3 else 'no (expected: F3 is the upgrade)'}")
    if not ok_all:
        note("  *** BASELINE CACHE DOES NOT REPRODUCE. DO NOT TRUST THE SCREEN. ***")
    flush()

    head("3. SAMPLE SIZES AND NESTING")
    note("  Reduced TRAINING rows only. Validation is never subsampled.")
    note("")
    note(f"  {'fold':<6}{'full train':>12}{'SMOKE':>10}{'SCREEN':>10}"
         f"{'smoke red.':>12}{'screen red.':>13}{'smoke hash':>18}{'screen hash':>18}")
    for f in ("JAN", "JUL"):
        S = samples[f]
        note(f"  {f:<6}{S['n_full']:>12,}{S['n_smoke']:>10,}{S['n_screen']:>10,}"
             f"{100 * S['n_smoke'] / S['n_full']:>11.2f}%"
             f"{100 * S['n_screen'] / S['n_full']:>12.2f}%"
             f"{S['smoke_hash']:>18}{S['screen_hash']:>18}")
    tot_f = sum(samples[f]["n_full"] for f in samples)
    tot_s = sum(samples[f]["n_smoke"] for f in samples)
    tot_c = sum(samples[f]["n_screen"] for f in samples)
    note(f"  {'TOTAL':<6}{tot_f:>12,}{tot_s:>10,}{tot_c:>10,}"
         f"{100 * tot_s / tot_f:>11.2f}%{100 * tot_c / tot_f:>12.2f}%")
    note("")
    note(f"  SMOKE contained in SCREEN: "
         f"{'YES' if all(set(samples[f]['smoke_ids'].tolist()) <= set(samples[f]['screen_ids'].tolist()) for f in samples) else 'NO'}")
    note(f"  salt '{SAMPLE_SALT}' frozen; selection is splitmix64(MVT_ID), so the")
    note("  samples are reproducible on any machine and any future run.")
    flush()

    head("4. REPRESENTATIVENESS AUDIT (full training vs SMOKE vs SCREEN)")
    dists = {}
    for f, (lo, hi) in FOLDS.items():
        note("")
        note(f"  --- {f} outer-training fold ---")
        dists[f] = report_dist(f, [("full", samples[f]["rows_full"]),
                                   ("SMOKE", samples[f]["rows_smoke"]),
                                   ("SCREEN", samples[f]["rows_screen"])])
    flush()

    head("5. MATERIAL DISTORTION CHECK")
    worst = []
    for f, d in dists.items():
        for lbl, v in d.items():
            for k, lim in (("d36", 0.5), ("d72", 0.25), ("dm4", 0.5),
                           ("da", 1.0), ("dm", 2.0), ("dp99", 10.0)):
                if abs(v[k]) > lim:
                    worst.append(f"{f}/{lbl}: {k}={v[k]:+.3f} (limit {lim})")
    note("  limits: >3600 rate 0.5pp, >7200 rate 0.25pp, MISSING_4 rate 0.5pp,")
    note("          any airport share 1.0pp, any month share 2.0pp, p99 10%")
    note("")
    if worst:
        for w in worst:
            note(f"  [WARN] {w}")
    else:
        note("  [PASS] no reduced set exceeds any distortion limit.")
    note("")
    note("  Strata emptied by a reduced run (floor, no per-stratum minimum):")
    for f, (lo, hi) in FOLDS.items():
        for lbl in ("smoke", "screen"):
            empt = samples[f][f"empty_{lbl}"]
            note(f"    {f} {lbl:<7} {len(empt):>4} of "
                 f"{samples[f]['n_strata']:,} strata contribute 0 rows")
            if empt:
                note(f"             e.g. {', '.join(empt[:4])}"
                     + (" ..." if len(empt) > 4 else ""))
    flush()

    head("6. BASELINE ARTIFACTS REUSED")
    note(f"  {cache['provenance']['source']}")
    note(f"    created {cache['created']}")
    note("  Reused without refitting:")
    note("    - Exp04 (V8/T4 17 features + adep_reference_state) full-fold preds")
    note("    - plain F3 per-ADEP MISSING_4 specialist full-fold preds")
    note("    - validation MVT_ID, target, COMPLETE/MISSING_4 mask, ADEP, d_SCHED")
    note("    - best iteration counts for Exp04 and every F3 specialist")
    note("  Not reused (must be refit by a candidate experiment):")
    note("    - any candidate model, specialist or global")
    note("  Frozen and untouched: V8/V9/V10, F3 reproduction, F3 airport guard,")
    note("  F3-C weather experiment and all their reports.")
    flush()

    head("7. EXPENSIVE FITS THAT CAN NOW BE SKIPPED")
    note("  a) MISSING_4-only experiment")
    note("     Exp04 and plain-F3 predictions come from the baseline cache, so")
    note("     only the candidate specialists are fit: 2 passes (JAN, JUL) over")
    note("     ~21k and ~19k MISSING_4 training rows. Nothing else is trained.")
    note("  b) COMPLETE / global screening experiment")
    note("     Validation still scores all 344,419 rows, but the 1.9M-row")
    note("     COMPLETE training fold is subsampled, so CatBoost cost scales with")
    note("     the sample fraction instead of the full fold.")
    note("  c) Diagnostics needing COMPLETE-row context (ADEP, d_SCHED, month,")
    note("     reference_state) come from the frame build, not from a refit.")
    flush()

    head("8. EXPECTED RUNTIME REDUCTION")
    note("  Measured basis (F3-C run, this machine):")
    note("    full Exp04 x2 folds                ~540 s")
    note("    one MISSING_4 specialist pass      ~320 s")
    note("    F3-C total (Exp04 x2 + 4 passes)  ~1,820 s  (~30 min)")
    note("  Estimate (scaling with training rows, not measured per stage):")
    note("")
    note(f"  {'experiment':<34}{'full':>10}{'SCREEN':>10}{'SMOKE':>10}"
         f"{'screen x':>10}{'smoke x':>9}")
    note(f"  {'a) MISSING_4-only':<34}{'~640 s':>10}{'~190 s':>10}{'~80 s':>10}"
         f"{'~3.4x':>10}{'~8x':>9}")
    note(f"  {'b) COMPLETE/global':<34}{'~540 s':>10}{'~140 s':>10}{'~60 s':>10}"
         f"{'~3.9x':>10}{'~9x':>9}")
    note("  (a) is vs a MISSING_4-only run that still refits Exp04, which is the")
    note("  mistake this framework exists to prevent. Versus the F3-C total of")
    note("  ~1,820 s it is ~9.5x. (b) is the Exp04 stage only; the frame build is")
    note("  ~10 s either way. One measured SCREEN run will calibrate these.")
    flush()

    head("9. DECISION RULE")
    note("  SMOKE  runs, safety checks pass, signal not obviously disastrous")
    note("          -> SCREEN")
    note("  SCREEN full validation, reduced training, JAN / JUL / pooled + slices")
    note("          clearly worse          -> stop / REJECT")
    note("          essentially zero signal -> usually stop")
    note("          promising               -> FULL confirmation")
    note("  FULL   the established full Jan+Jul protocol. Only FULL results may")
    note("         become architecture changes or submissions.")
    note("  A reduced run is NEVER sufficient evidence to adopt a model.")
    flush()

    head("10. STATUS")
    if ok_all and not worst:
        note("  baseline cache reproduces every frozen reference")
        note("  SMOKE nests inside SCREEN, both are hash-stable")
        note("  no reduced set exceeds any representativeness limit")
        note("")
        note("SCREENING FRAMEWORK READY")
    else:
        if not ok_all:
            note("  baseline cache does not reproduce the frozen references")
        for w in worst:
            note(f"  distortion: {w}")
        note("")
        note("SCREENING FRAMEWORK NEEDS FIX")
    flush()
    print("\nDONE -> reports/screening_protocol_audit.txt")


if __name__ == "__main__":
    report_only = "--report-only" in sys.argv
    if not report_only and not BASELINE_CACHE.exists():
        cache = compute()
    else:
        assert BASELINE_CACHE.exists(), f"no baseline cache at {BASELINE_CACHE}"
        if report_only:
            plain("REPORT-ONLY MODE: reusing the cached baseline. No frame build "
                  "for the cache, no refit. Any change to rows, features, joins,")
            plain("training, predictions, routing, alignment or scoring "
                  "invalidates this mode.")
            plain("")
        with open(BASELINE_CACHE, "rb") as fh:
            cache = pickle.load(fh)

    t0 = time.time()
    plain("building the full frame (read-only, no model is fit) and loading "
          "the frozen sample IDs")
    dep = build_frame()
    samples = load_samples(dep, resample="--resample" in sys.argv)
    if "--verify-samples" in sys.argv:
        verify_samples(dep, samples)
    for f, (lo, hi) in FOLDS.items():
        m = ((dep[MINUEND] >= lo) & (dep[MINUEND] < hi)).to_numpy()
        tr = dep[~m]
        all_ids = tr["MVT_ID_mvt"].to_numpy()
        key_all = stratum_key(tr)
        samples[f]["rows_full"] = tr
        for lbl in ("smoke", "screen"):
            rows = tr[np.isin(all_ids, samples[f][f"{lbl}_ids"])]
            samples[f][f"rows_{lbl}"] = rows
            kept = set(stratum_key(rows).tolist())
            samples[f][f"empty_{lbl}"] = sorted(set(key_all.tolist()) - kept)
    del dep
    gc.collect()
    plain(f"  frame and samples ready in {time.time() - t0:.0f}s")
    finalize_report(cache, samples)
