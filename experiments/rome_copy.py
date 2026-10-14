"""Rome (LIRF) schedule-copy model for MISSING_4 departures, 1 h < MVT - SCHED <= 6 h.

At LIRF, about half of the departures without a Network Manager record (AOBT_3,
EOBT_1, IOBT, LOBT all missing) that take off more than 1 h after schedule have an
official off-block time equal to the scheduled time (taxi = MVT - SCHED); the rest
have a normal taxi time. At the other nine airports this practically never happens.
Whether a flight is a "copy" depends strongly on the airline.

For those LIRF rows (1 h < MVT - SCHED <= 6 h; above 6 h nm_line.py already applies):
    p      = CatBoost probability of a copy (airline, operator, aircraft type, runway,
             stand area, market segment, flight type, MVT - SCHED, hour, weekday),
             3 seeds averaged, trained on 2025 LIRF MISSING_4 rows with MVT - SCHED > 1 h
    model  = p * (MVT - SCHED) + (1 - p) * mean non-copy taxi
    final  = 0.5 * current prediction + 0.5 * model          (equal weights, not fitted)
Validation trains WITHOUT the scored month. Idea inspired by team outgoing-boat's
copy classifier and Rome rule (github.com/EnioAguiar/prc-taxiout-2026); independent
re-implementation, Rome-only variant.

Needs nm_line.py in the same folder.
Validate on honest JAN/JUL corrector predictions (nm_line is applied first, as in V51):
    python -u experiments\\rome_copy.py --oof data\\processed\\corrector_oof_ctx.parquet
Build from an existing submission that already includes nm_line (e.g. V51):
    python -u experiments\\rome_copy.py --submission submissions\\vigorous-glove_v51.parquet --version N
"""

import argparse
import glob
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nm_line  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SUBS = ROOT / "submissions"
REPORT = ROOT / "reports" / "rome_copy.txt"
MID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
APT = "LIRF"
LO, HI = 3600, 6 * 3600
SEEDS = (0, 1, 2)
N_RANK = 344_841
REF = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]
CAT = ["airline", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "stand1",
       "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"]
NUM = ["ms", "hour", "dow"]
COLS = [MID, "PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt",
        "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt"] + REF
OUT = []


def A(s=""):
    OUT.append(str(s))
    print(s, flush=True)


def done():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(OUT) + "\n")


def stop(m):
    A(f"STOPPED: {m}")
    done()
    sys.exit(1)


def rmse(e):
    e = np.asarray(e, np.float64)
    return math.sqrt(float(np.dot(e, e)) / len(e))


def load(files, with_target):
    cols = COLS + ([TARGET] if with_target else [])
    d = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    d = d[d["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    T = d["MVT_TIME_UTC_mvt"]
    d["ms"] = (T - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["m4"] = d[REF].isna().all(axis=1)
    d["ym"] = T.dt.strftime("%Y-%m")
    d["hour"] = d["SCHED_TIME_UTC_mvt"].dt.hour
    d["dow"] = T.dt.dayofweek
    d["airline"] = d["FLIGHT_mvt"].astype(str).str.extract(r"^([A-Z]+)", expand=False)
    d["stand1"] = d["STAND_mvt"].astype(object).where(d["STAND_mvt"].notna(), "NA").astype(str).str[:1]
    for c in CAT:
        d[c] = d[c].astype(object).where(d[c].notna(), "NA").astype(str)
    d["ADEP_mvt"] = d["ADEP_mvt"].astype(str)
    if with_target:
        d["cp"] = ((d[TARGET] - d["ms"]).abs() <= 60).astype(int)
    return d


def train_pool(tr):
    return tr[(tr["ADEP_mvt"] == APT) & tr["m4"] & (tr["ms"] > LO) & tr[TARGET].notna()]


def target_mask(frame):
    return ((frame["ADEP_mvt"] == APT) & frame["m4"] & (frame["ms"] > LO)
            & (frame["ms"] <= HI)).to_numpy()


def model_pred(t, rows):
    """p*(MVT-SCHED) + (1-p)*non-copy mean, CatBoost averaged over SEEDS."""
    nc = float(t.loc[t["cp"] == 0, TARGET].clip(0, 3600).mean())
    p = np.zeros(len(rows))
    for s in SEEDS:
        m = CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, verbose=0,
                               random_seed=s, cat_features=CAT, thread_count=-1)
        m.fit(t[CAT + NUM], t["cp"])
        p += m.predict_proba(rows[CAT + NUM])[:, 1] / len(SEEDS)
    return np.maximum(p * rows["ms"].to_numpy() + (1 - p) * nc, 0.0), p, nc, len(t)


def validate(path):
    A("=" * 70)
    A(f"VALIDATION on honest JAN/JUL predictions: {path}")
    A(f"  rows: {APT}, MISSING_4, {LO // 3600} h < MVT-SCHED <= {HI // 3600} h; "
      "final = 0.5*current + 0.5*model; nm_line (>6 h) applied first")
    A("=" * 70)
    o = pd.read_parquet(path)
    pcol = "corr_pred" if "corr_pred" in o.columns else "pred"
    tr = load(sorted(glob.glob(str(RAW / "training_*.parquet"))), True)
    v = o[[MID, "fold", "y", pcol]].merge(tr.drop(columns=[TARGET]), on=MID, how="left")
    if v["ADEP_mvt"].isna().any():
        stop("some OOF ids not found in training files")
    cur = v[pcol].to_numpy(np.float64).copy()
    new = cur.copy()
    for fold, ym in (("JAN", "2025-01"), ("JUL", "2025-07")):
        rest = tr[tr["ym"] != ym]
        m = (v["fold"] == fold).to_numpy()
        cur[m], _ = nm_line.apply(cur[m], v[m], nm_line.fit_lines(rest))
        new[m] = cur[m]
        tm = m & target_mask(v)
        if tm.any():
            mp, p, nc, n = model_pred(train_pool(rest), v[tm])
            new[tm] = 0.5 * cur[tm] + 0.5 * mp
        A(f"  {fold}: training rows {n}, non-copy mean {nc:.0f} s, rows changed {int(tm.sum())}, "
          f"mean p {p.mean() if tm.any() else float('nan'):.2f}")
    res = {}
    for name, m in (("JAN", v["fold"] == "JAN"), ("JUL", v["fold"] == "JUL"),
                    ("POOLED", np.ones(len(v), bool))):
        m = np.asarray(m)
        b, c = rmse(v["y"][m] - cur[m]), rmse(v["y"][m] - new[m])
        res[name] = (b, c)
        A(f"  {name:<7} before {b:9.3f}   after {c:9.3f}   delta {c - b:+.3f}   (before = with nm_line)")
    tm = target_mask(v)
    g = ((v["y"] - cur) ** 2 - (v["y"] - new) ** 2).to_numpy()[tm]
    top = np.sort(g)[::-1][:20].sum()
    A(f"  changed rows {int(tm.sum())}: improved {int((g > 0).sum())}, worse {int((g < 0).sum())}")
    A(f"  SSE gain {g.sum():.4g}; without the 20 best rows {g.sum() - top:.4g} "
      f"({'still positive' if g.sum() - top > 0 else 'NEGATIVE - gain carried by few rows'})")
    ok = res["JAN"][1] < res["JAN"][0] and res["JUL"][1] < res["JUL"][0]
    A(f"  GATE (both months better): {'PASS' if ok else 'FAIL'}")


def build(sub_path, version):
    A("=" * 70)
    A(f"BUILD v{version} from {sub_path} (input must already include nm_line)")
    A("=" * 70)
    dest = SUBS / f"vigorous-glove_v{version}.parquet"
    if dest.exists():
        stop(f"{dest} exists; refusing to overwrite")
    tpl = pd.read_parquet(RAW / "submitting.parquet")
    s = pd.read_parquet(sub_path)
    if len(s) != N_RANK or set(s[MID]) != set(tpl[MID]):
        stop("submission ids do not match the template")
    tr = load(sorted(glob.glob(str(RAW / "training_*.parquet"))), True)
    rk = load([RAW / "ranking.parquet"], False)
    r = tpl[[MID]].merge(s[[MID, TARGET]], on=MID, how="left").merge(rk, on=MID, how="left")
    cur = r[TARGET].to_numpy(np.float64)
    new = cur.copy()
    tm = target_mask(r)
    mp, p, nc, n = model_pred(train_pool(tr), r[tm])
    new[tm] = 0.5 * cur[tm] + 0.5 * mp
    A(f"  training rows {n}, non-copy mean {nc:.0f} s")
    A(f"  ranking rows changed: {int(tm.sum())}; mean p {p.mean():.2f}")
    d = new[tm] - cur[tm]
    A(f"  change on those rows: median {np.median(d):+.0f} s, |max| {np.abs(d).max():.0f} s")
    out = pd.DataFrame({MID: r[MID].to_numpy(), TARGET: new})[list(tpl.columns)]
    checks = {
        "rows == 344,841": len(out) == N_RANK,
        "row order matches template": (out[MID].to_numpy() == tpl[MID].to_numpy()).all(),
        "unique ids": not out[MID].duplicated().any(),
        "no NaN/inf": bool(np.isfinite(out[TARGET]).all()),
        "no negatives": bool((out[TARGET] >= 0).all()),
        "only LIRF MISSING_4 rows changed": int((new != cur).sum()) <= int(tm.sum()),
    }
    for k, ok in checks.items():
        A(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    if not all(checks.values()):
        stop("integrity check failed")
    out.to_parquet(dest, index=False)
    A(f"  prediction min/median/max: {out[TARGET].min():.0f} / {out[TARGET].median():.0f} / "
      f"{out[TARGET].max():.0f}")
    A(f"  wrote {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="")
    ap.add_argument("--submission", default="")
    ap.add_argument("--version", type=int, default=None)
    a = ap.parse_args()
    if a.oof:
        validate(a.oof)
    if a.submission:
        if a.version is None:
            stop("--version N required")
        build(a.submission, a.version)
    done()


if __name__ == "__main__":
    main()
