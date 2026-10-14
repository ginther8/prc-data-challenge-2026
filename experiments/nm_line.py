"""Per-airport line for MISSING_4 departures with a large MVT - SCHED gap.

For departures without a Network Manager record (all four reference timestamps
AOBT_3 / EOBT_1 / IOBT / LOBT missing) whose takeoff is more than 6 h after the
scheduled time, the official off-block time is usually a copy of the schedule,
so taxi ~= MVT - SCHED. The prediction for those rows is replaced by a
per-airport line   taxi = a + b * (MVT - SCHED)   fitted on 2025 rows of the
same kind (airports with >= 20 such rows; others unchanged).
Validation fits the line WITHOUT the scored month. The 6 h threshold is taken
from team outgoing-boat (github.com/EnioAguiar/prc-taxiout-2026), not tuned here.

Validate on honest JAN/JUL predictions (corrector OOF saved with --save-oof):
    python -u experiments\\nm_line.py --oof data\\processed\\corrector_oof_ctx.parquet
Build from an existing submission:
    python -u experiments\\nm_line.py --submission submissions\\vigorous-glove_v15.parquet --version N
"""

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SUBS = ROOT / "submissions"
REPORT = ROOT / "reports" / "nm_line.txt"
MID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
THRESH = 6 * 3600
MIN_ROWS = 20
N_RANK = 344_841
REF = ["AOBT_3_flt", "EOBT_1_flt", "IOBT_flt", "LOBT_flt"]
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
    cols = [MID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"] + REF
    if with_target:
        cols.append(TARGET)
    d = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    d = d[d["PHASE_mvt"] == "DEP"].copy()
    d["ms"] = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["m4"] = d[REF].isna().all(axis=1)
    d["ym"] = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    d["ADEP_mvt"] = d["ADEP_mvt"].astype(str)
    return d


def fit_lines(train):
    t = train[train["m4"] & (train["ms"] > THRESH) & train[TARGET].notna()]
    lines = {}
    for ap, g in t.groupby("ADEP_mvt"):
        if len(g) >= MIN_ROWS:
            b, a = np.polyfit(g["ms"].to_numpy(float), g[TARGET].to_numpy(float), 1)
            lines[ap] = (a, b, len(g))
    return lines


def apply(pred, frame, lines):
    p = np.array(pred, dtype=np.float64, copy=True)
    hit = np.zeros(len(p), bool)
    for ap, (a, b, _) in lines.items():
        m = (frame["m4"].to_numpy() & (frame["ms"].to_numpy() > THRESH)
             & (frame["ADEP_mvt"].to_numpy() == ap))
        p[m] = a + b * frame["ms"].to_numpy()[m]
        hit |= m
    return np.maximum(p, 0.0), hit


def validate(path):
    A("=" * 70)
    A(f"VALIDATION on honest JAN/JUL predictions: {path}   (threshold {THRESH // 3600} h)")
    A("=" * 70)
    o = pd.read_parquet(path)
    pcol = "corr_pred" if "corr_pred" in o.columns else "pred"
    tr = load(sorted(glob.glob(str(RAW / "training_*.parquet"))), True)
    v = o.merge(tr[[MID, "ADEP_mvt", "ms", "m4"]].rename(columns={"ADEP_mvt": "apt"}), on=MID, how="left")
    v["ADEP_mvt"] = v["apt"]
    newp = np.array(v[pcol], dtype=np.float64, copy=True)
    hits = np.zeros(len(v), bool)
    for fold, ym in (("JAN", "2025-01"), ("JUL", "2025-07")):
        lines = fit_lines(tr[tr["ym"] != ym])
        m = (v["fold"] == fold).to_numpy()
        p, h = apply(v.loc[m, pcol], v[m], lines)
        newp[m] = p
        hits[m] = h
        A(f"  {fold}: lines for {len(lines)} airports; rows replaced {int(h.sum())}")
    res = {}
    for name, m in (("JAN", v["fold"] == "JAN"), ("JUL", v["fold"] == "JUL"),
                    ("POOLED", np.ones(len(v), bool))):
        m = np.asarray(m)
        b, c = rmse(v["y"][m] - v[pcol][m]), rmse(v["y"][m] - newp[m])
        res[name] = (b, c)
        A(f"  {name:<7} before {b:9.3f}   after {c:9.3f}   delta {c - b:+.3f}")
    g = (v["y"] - v[pcol]) ** 2 - (v["y"] - newp) ** 2
    A(f"  replaced rows: {int(hits.sum())}; of them improved {int((g[hits] > 0).sum())}, "
      f"worse {int((g[hits] < 0).sum())}")
    ok = res["JAN"][1] < res["JAN"][0] and res["JUL"][1] < res["JUL"][0]
    A(f"  GATE (both months better): {'PASS' if ok else 'FAIL'}")


def build(sub_path, version):
    A("=" * 70)
    A(f"BUILD v{version} from {sub_path}")
    A("=" * 70)
    dest = SUBS / f"vigorous-glove_v{version}.parquet"
    if dest.exists():
        stop(f"{dest} exists; refusing to overwrite")
    tpl = pd.read_parquet(RAW / "submitting.parquet")
    s = pd.read_parquet(sub_path)
    if len(s) != N_RANK or set(s[MID]) != set(tpl[MID]):
        stop("submission ids do not match the template")
    lines = fit_lines(load(sorted(glob.glob(str(RAW / "training_*.parquet"))), True))
    for ap, (a, b, n) in sorted(lines.items()):
        A(f"  {ap}: taxi = {a:8.1f} + {b:.4f} * (MVT - SCHED)   ({n} training rows)")
    rk = load([RAW / "ranking.parquet"], False)
    r = tpl[[MID]].merge(s[[MID, TARGET]], on=MID, how="left").merge(
        rk[[MID, "ADEP_mvt", "ms", "m4"]], on=MID, how="left")
    new, hit = apply(r[TARGET], r, lines)
    A(f"  ranking rows replaced: {int(hit.sum())}")
    if hit.any():
        d = new[hit] - r[TARGET].to_numpy()[hit]
        A(f"  change on those rows: median {np.median(d):+.0f} s, |max| {np.abs(d).max():.0f} s")
    out = pd.DataFrame({MID: r[MID].to_numpy(), TARGET: new})[list(tpl.columns)]
    checks = {
        "rows == 344,841": len(out) == N_RANK,
        "row order matches template": (out[MID].to_numpy() == tpl[MID].to_numpy()).all(),
        "unique ids": not out[MID].duplicated().any(),
        "no NaN/inf": bool(np.isfinite(out[TARGET]).all()),
        "no negatives": bool((out[TARGET] >= 0).all()),
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
