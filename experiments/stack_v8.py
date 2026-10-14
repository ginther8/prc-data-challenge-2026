"""Stack two independent improvements over the same base (V4):

    V8 = V6 + (V7 - V4)

  V6 = V4 + our ADS-B correction        (leaderboard 300.6979)
  V7 = friend's improved base model     (leaderboard 321.0504)
  V4 = common base                      (leaderboard 327.7782)

So each flight gets V4's prediction, plus our ADS-B adjustment, plus the change
the friend's new model makes relative to V4. Predictions are floored at 0.
No training, no leaderboard tuning - just adding the two measured changes.

Run from the project root:
    python experiments/stack_v8.py --v4 <v4 file> --v6 <v6 file> --v7 <v7 file>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
N = 344_841
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "data" / "raw" / "submitting.parquet"
OUT = ROOT / "submissions" / "vigorous-glove_v8.parquet"


def load(path, name):
    d = pd.read_parquet(path)
    if len(d) != N or d[MID].duplicated().any() or not np.isfinite(d[TARGET]).all():
        sys.exit(f"FAIL: {name} has wrong rows / duplicate ids / non-finite values")
    return d.set_index(MID)[TARGET].rename(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", required=True)
    ap.add_argument("--v6", required=True)
    ap.add_argument("--v7", required=True)
    a = ap.parse_args()
    if OUT.exists():
        sys.exit(f"FAIL: {OUT} exists; refusing to overwrite")
    tpl = pd.read_parquet(TEMPLATE)
    v4, v6, v7 = load(a.v4, "v4"), load(a.v6, "v6"), load(a.v7, "v7")
    ids = tpl[MID]
    for s in (v4, v6, v7):
        if set(s.index) != set(ids):
            sys.exit(f"FAIL: {s.name} ids differ from the template")
    v4, v6, v7 = v4.loc[ids].to_numpy(), v6.loc[ids].to_numpy(), v7.loc[ids].to_numpy()

    ours, his = v6 - v4, v7 - v4
    print(f"our ADS-B change (V6-V4):  mean {ours.mean():+.1f}  mean|.| {np.abs(ours).mean():.1f}  "
          f"rows changed {np.mean(np.abs(ours) > 0.5):.1%}")
    print(f"friend's change (V7-V4):   mean {his.mean():+.1f}  mean|.| {np.abs(his).mean():.1f}  "
          f"rows changed {np.mean(np.abs(his) > 0.5):.1%}")
    both = (np.abs(ours) > 0.5) & (np.abs(his) > 0.5)
    if both.any():
        print(f"rows changed by both: {both.mean():.1%}   correlation of the two changes "
              f"on those rows: {np.corrcoef(ours[both], his[both])[0, 1]:+.3f}")

    v8 = np.clip(v4 + ours + his, 0.0, None)
    sub = pd.DataFrame({MID: ids.to_numpy(), TARGET: v8})
    sub = sub[list(tpl.columns)]
    checks = {
        "rows == 344,841": len(sub) == N,
        "columns match template": list(sub.columns) == list(tpl.columns),
        "row order matches template": (sub[MID].to_numpy() == tpl[MID].to_numpy()).all(),
        "no NaN/inf": bool(np.isfinite(sub[TARGET]).all()),
        "no negatives": bool((sub[TARGET] >= 0).all()),
    }
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    if not all(checks.values()):
        sys.exit("FAIL: integrity check")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(OUT, index=False)
    print(f"V8 min/median/max/mean: {v8.min():.0f} / {np.median(v8):.0f} / {v8.max():.0f} / {v8.mean():.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
