"""Equal-weight (50/50) average of two corrected pipelines, then nm_line (> 6 h rule).

Validate on the two honest JAN/JUL corrector OOF files (nm_line applied after averaging):
    python -u experiments\\blend2.py --oof data\\processed\\corrector_oof_ctx.parquet data\\processed\\corrector_oof_v32.parquet
Build from two FINAL submissions (each already LOBT-clipped + nm_line):
    python -u experiments\\blend2.py --submission submissions\\vigorous-glove_v51.parquet submissions\\vigorous-glove_v62.parquet --version N
Weights are fixed at 0.5 / 0.5 (not fitted). Needs nm_line.py in the same folder.
"""

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nm_line  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SUBS = ROOT / "submissions"
REPORT = ROOT / "reports" / "blend2.txt"
MID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
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


def read_oof(path):
    o = pd.read_parquet(path)
    c = "corr_pred" if "corr_pred" in o.columns else "pred"
    o = o[[MID, "fold", "y", c]].rename(columns={c: "p"})
    o[MID] = o[MID].astype("int64")
    return o


def validate(a_path, b_path):
    A("=" * 70)
    A(f"VALIDATION 50/50 blend:\n  A = {a_path}\n  B = {b_path}\n  (nm_line applied to every line below)")
    A("=" * 70)
    a, b = read_oof(a_path), read_oof(b_path)
    v = a.merge(b[[MID, "p"]], on=MID, suffixes=("_a", "_b"))
    if len(v) != len(a) or len(v) != len(b):
        stop(f"row mismatch: A {len(a)}, B {len(b)}, matched {len(v)}")
    A(f"  rows {len(v):,}; correlation of A and B predictions {np.corrcoef(v.p_a, v.p_b)[0, 1]:.4f}")
    tr = nm_line.load(sorted(glob.glob(str(RAW / "training_*.parquet"))), True)
    v = v.merge(tr[[MID, "ADEP_mvt", "ms", "m4"]], on=MID, how="left")
    preds = {"A": v.p_a.to_numpy(float), "B": v.p_b.to_numpy(float),
             "50/50": 0.5 * v.p_a.to_numpy(float) + 0.5 * v.p_b.to_numpy(float)}
    for k in preds:
        p = preds[k].copy()
        for fold, ym in (("JAN", "2025-01"), ("JUL", "2025-07")):
            m = (v["fold"] == fold).to_numpy()
            p[m], _ = nm_line.apply(p[m], v[m], nm_line.fit_lines(tr[tr["ym"] != ym]))
        preds[k] = p
    res = {}
    A(f"  {'':8}{'JAN':>10}{'JUL':>10}{'POOLED':>10}")
    for k, p in preds.items():
        j = (v["fold"] == "JAN").to_numpy()
        res[k] = (rmse(v.y[j] - p[j]), rmse(v.y[~j] - p[~j]), rmse(v.y - p))
        A(f"  {k:<8}{res[k][0]:10.3f}{res[k][1]:10.3f}{res[k][2]:10.3f}")
    g = (v.y - preds["A"]) ** 2 - (v.y - preds["50/50"]) ** 2
    top = np.sort(g.to_numpy())[::-1][:20].sum()
    A(f"  gain of 50/50 over A: SSE {g.sum():.4g}; share from top-20 rows {top / g.sum():.0%}"
      if g.sum() > 0 else "  50/50 is not better than A overall")
    ok = res["50/50"][0] < res["A"][0] and res["50/50"][1] < res["A"][1]
    A(f"  GATE (50/50 better than A in both months): {'PASS' if ok else 'FAIL'}")


def build(a_path, b_path, version):
    A("=" * 70)
    A(f"BUILD v{version} = 0.5 * {a_path} + 0.5 * {b_path}")
    A("=" * 70)
    dest = SUBS / f"vigorous-glove_v{version}.parquet"
    if dest.exists():
        stop(f"{dest} exists; refusing to overwrite")
    tpl = pd.read_parquet(RAW / "submitting.parquet")
    a, b = pd.read_parquet(a_path), pd.read_parquet(b_path)
    for s in (a, b):
        if len(s) != N_RANK or set(s[MID]) != set(tpl[MID]):
            stop("submission ids do not match the template")
    r = tpl[[MID]].merge(a[[MID, TARGET]], on=MID).merge(b[[MID, TARGET]], on=MID, suffixes=("_a", "_b"))
    new = np.maximum(0.5 * r[TARGET + "_a"].to_numpy(float) + 0.5 * r[TARGET + "_b"].to_numpy(float), 0)
    out = pd.DataFrame({MID: r[MID].to_numpy(), TARGET: new})[list(tpl.columns)]
    A(f"  correlation of A and B: {np.corrcoef(r[TARGET + '_a'], r[TARGET + '_b'])[0, 1]:.4f}")
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
    ap.add_argument("--oof", nargs=2, default=None)
    ap.add_argument("--submission", nargs=2, default=None)
    ap.add_argument("--version", type=int, default=None)
    a = ap.parse_args()
    if a.oof:
        validate(*a.oof)
    if a.submission:
        if a.version is None:
            stop("--version N required")
        build(*a.submission, a.version)
    done()


if __name__ == "__main__":
    main()
