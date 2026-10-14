"""LOBT-window post-processing.

Fact (verified on all 2,062,577 departures of 2025 that have a LOBT):
    |BLOCK_TIME - LOBT| <= 3606 s   in 100 % of cases (max exactly 3606 s).
Since taxi = MVT - BLOCK, every prediction can be projected onto
    [MVT - LOBT - 3606, MVT - LOBT + 3606]
Rows without a LOBT are left unchanged. Predictions are then floored at 0.
(Idea credit: team outgoing-boat, github.com/EnioAguiar/prc-taxiout-2026, GPL-3.0;
this file is an independent re-implementation.)

Validate on honest JAN/JUL predictions (e.g. the corrector OOF saved with --save-oof):
    python -u experiments\\lobt_clip.py --oof data\\processed\\corrector_oof_v17base.parquet
Build a clipped submission from an existing one:
    python -u experiments\\lobt_clip.py --submission submissions\\vigorous-glove_v14.parquet --version 15
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
REPORT = ROOT / "reports" / "lobt_clip.txt"
MID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
WINDOW = 3606.0
N_RANK = 344_841
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


def times(files):
    cols = [MID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]
    d = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    d = d[d["PHASE_mvt"] == "DEP"]
    d["dl"] = (d["MVT_TIME_UTC_mvt"] - d["LOBT_flt"]).dt.total_seconds()   # NaN if no LOBT
    return d[[MID, "dl"]]


def clip(pred, dl):
    pred = np.asarray(pred, np.float64)
    dl = np.asarray(dl, np.float64)
    has = np.isfinite(dl)
    out = pred.copy()
    out[has] = np.clip(pred[has], dl[has] - WINDOW, dl[has] + WINDOW)
    return np.maximum(out, 0.0)


def validate(path):
    A("=" * 70)
    A(f"VALIDATION on honest JAN/JUL predictions: {path}")
    A("=" * 70)
    o = pd.read_parquet(path)
    pcol = "corr_pred" if "corr_pred" in o.columns else "pred"
    t = times(sorted(glob.glob(str(RAW / "training_*.parquet"))))
    o = o.merge(t, on=MID, how="left")
    o["clipped"] = clip(o[pcol], o["dl"])
    changed = int((np.abs(o["clipped"] - o[pcol]) > 1e-9).sum())
    res = {}
    for name, m in (("JAN", o["fold"] == "JAN"), ("JUL", o["fold"] == "JUL"),
                    ("POOLED", np.ones(len(o), bool))):
        m = np.asarray(m)
        b, c = rmse(o["y"][m] - o[pcol][m]), rmse(o["y"][m] - o["clipped"][m])
        res[name] = (b, c)
        A(f"  {name:<7} rows {m.sum():>8,}   before {b:9.3f}   after {c:9.3f}   delta {c - b:+.3f}")
    A(f"  rows changed: {changed:,}")
    ok = res["JAN"][1] <= res["JAN"][0] and res["JUL"][1] <= res["JUL"][0]
    A(f"  GATE (no month worse): {'PASS' if ok else 'FAIL'}")
    return ok


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
    t = times([RAW / "ranking.parquet"])
    r = tpl[[MID]].merge(s[[MID, TARGET]], on=MID, how="left").merge(t, on=MID, how="left")
    new = clip(r[TARGET], r["dl"])
    ch = np.abs(new - r[TARGET].to_numpy()) > 1e-9
    A(f"  ranking rows with a LOBT: {np.isfinite(r['dl']).mean():.1%}   rows changed: {int(ch.sum()):,}")
    if ch.any():
        d = (new - r[TARGET].to_numpy())[ch]
        A(f"  size of changes: median {np.median(np.abs(d)):.0f} s, max {np.abs(d).max():.0f} s")
    out = pd.DataFrame({MID: r[MID].to_numpy(), TARGET: new})[list(tpl.columns)]
    checks = {
        "rows == 344,841": len(out) == N_RANK,
        "row order matches template": (out[MID].to_numpy() == tpl[MID].to_numpy()).all(),
        "unique ids": not out[MID].duplicated().any(),
        "no NaN/inf": bool(np.isfinite(out[TARGET]).all()),
        "no negatives": bool((out[TARGET] >= 0).all()),
    }
    for k, v in checks.items():
        A(f"  [{'PASS' if v else 'FAIL'}] {k}")
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
