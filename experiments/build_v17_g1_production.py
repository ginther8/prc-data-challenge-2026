#!/usr/bin/env python3
"""
V17 PRODUCTION G1_basic  (COMPLETE route, 2026 ranking frame)
===========================================================

Builds ONLY the missing half of the V17 blend: the 2026 G1_basic COMPLETE
predictions. The V15 half already exists at handover/v15_queue_ranking.parquet
and is never refitted here.

G1_basic is defined to match the honest V16 screen exactly:
    exact C2 (RIG.C2_ALL_FEATURES, 27) + GF.G1_FEATURES (25 basic geometry)
No graph features (G2 is deliberately excluded).

Iteration selection is INNER ONLY (abs(MVT_ID_mvt) % 5 == 0), then refit on all
2025 COMPLETE rows of that ADEP with best_iteration + 1, exactly as V16 and V15
did. Jan/Jul models are NOT re-run.

BLOCK_TIME_UTC_mvt is excluded from the ranking projection before the read.
MISSING_4 is not predicted by G1 at all; it keeps the exact F3 value in Step 4.
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "reference" / "v8"))

import run_v5_c2_rigorous_janjul as RIG
import run_v5_missing4_substantial as B
import v16_geometry_features as GF
from create_a4_f3_v4_submission import augment  # noqa: F401  (module-level side effects kept out)

MID, TARGET, AP = B.MID, B.TARGET, "ADEP_mvt"
FORBIDDEN, MISSING_LABEL = B.FORBIDDEN, B.MISSING_LABEL
D_AOBT3, RESID_COL = RIG.D_AOBT3, RIG.RESID_COL
C2_FEATURES, C2_CATS = list(RIG.C2_ALL_FEATURES), list(RIG.C2_ALL_CATEGORICALS)
G1 = list(GF.G1_FEATURES)
FEATS = C2_FEATURES + G1

RAW = ROOT / "data" / "raw"
OUT = Path("data/processed/v17_g1_basic_ranking.parquet")
T0 = time.time()


def A(m=""):
    print(m, flush=True)


def drain():
    import sys as _s
    _s.stdout.flush()


from catboost import CatBoostRegressor  # noqa: E402
from train_catboost_v8_defs import (V7_FEATURES, T4_FEATURES, add_t4_features,  # noqa: E402
                                    add_time_features, add_traffic_feature,
                                    apply_categorical_handling)

A("=" * 78)
A("V17 PRODUCTION G1_basic  (2026 ranking, COMPLETE only)")
A("=" * 78)
A(f"features: {len(C2_FEATURES)} C2 + {len(G1)} G1 basic geometry = {len(FEATS)}")
A("  (G2 graph features deliberately excluded)")
A(f"{FORBIDDEN} excluded from the ranking projection before the read")
A("iteration selection: inner split abs(MVT_ID_mvt)%5==0 only")
A("")

# ---------------------------------------------------------------- ranking frame
load_cols = [c for c in pq.read_schema(RAW / "ranking.parquet").names
             if c != FORBIDDEN]
assert FORBIDDEN not in load_cols
A(f"ranking columns read: {len(load_cols)} ({FORBIDDEN} excluded)")
allr = pq.read_table(RAW / "ranking.parquet", columns=load_cols).to_pandas()
allr = add_traffic_feature(allr)
dep = allr[allr["PHASE_mvt"] == "DEP"].copy()
del allr
A(f"ranking departures: {len(dep):,}")
assert len(dep) == 344_841, len(dep)
assert int(dep[TARGET].notna().sum()) == 0
dep = apply_categorical_handling(add_t4_features(add_time_features(dep)))
rk = augment(pl.from_pandas(dep[[MID] + V7_FEATURES + T4_FEATURES
                               + ["FLIGHT_mvt", TARGET]]))
rk = RIG.add_c2_columns(rk)
assert all(c in rk.columns for c in RIG.C2_FEATURES)
r_state = rk["state"].to_numpy()
r_adep = rk[AP].to_numpy()
comp_mask = r_state == "COMPLETE"
n_comp, n_m4 = int(comp_mask.sum()), int((~comp_mask).sum())
A(f"ranking COMPLETE {n_comp:,}   MISSING_4 {n_m4:,}")
assert (n_comp, n_m4) == (339_551, 5_290), (n_comp, n_m4)

# ------------------------------------------------------------ geometry features
geos = GF._load_geometry()
tab = {a: GF._airport_tables(g) for a, g in geos.items()}
sub = rk.filter((pl.col("state") == "COMPLETE") & pl.col(AP).is_in(sorted(geos.keys())))
rows = list(zip(sub[MID].to_list(), sub[AP].to_list(),
                sub[GF.STAND].to_list(), sub[GF.RUNWAY].to_list()))
geo = GF.build_rows(rows, geos, tab)
assert geo[MID].n_unique() == geo.height
n0 = rk.height
rk = rk.join(geo.select([MID] + GF.G1_FEATURES), on=MID, how="left")
assert rk.height == n0, (n0, rk.height)
A(f"geometry features built for {geo.height:,} target-airport COMPLETE rows "
  f"of {n_comp:,}")
A(f"  runway coverage {100*sub.height/n_comp:.1f}%  "
  f"stand coverage {100*geo['geo_stand_size_code'].count()/max(geo.height,1):.1f}%")
cov_t = rk.filter(comp_mask)["geo_rwy_len_m"].count()
A(f"  G1 runway coverage on COMPLETE ranking rows: {100*cov_t/n_comp:.1f}%")
drain()

# ------------------------------------------------------------------ train 2025
base = B.prepare_base()
assert FORBIDDEN not in base.columns
base = RIG.add_c2_columns(base)
geo25 = pl.read_parquet("data/processed/v16_geometry_features.parquet")
assert geo25[MID].n_unique() == geo25.height
m0 = base.height
base = base.join(geo25.select([MID] + GF.G1_FEATURES), on=MID, how="left")
assert base.height == m0, (m0, base.height)
assert FORBIDDEN not in base.columns
t_cp = base.filter(pl.col("state") == "COMPLETE")
t_cp = t_cp.with_columns((pl.col(TARGET) - pl.col(D_AOBT3)).alias(RESID_COL))
A(f"2025 COMPLETE training rows: {t_cp.height:,}  "
  f"(geometry joined 1:1 from the V16 cache, G1 columns only)")
assert all(c in t_cp.columns for c in FEATS), "missing features on train frame"


def to_cb(frame):
    return RIG.to_cb(frame, FEATS, C2_CATS)


def probe(frame):
    iv = frame.filter(pl.col(MID).abs() % 5 == 0)
    it = frame.filter(~(pl.col(MID).abs() % 5 == 0))
    m = CatBoostRegressor(allow_writing_files=False, **RIG.PROBE_PARAMS)
    m.fit(to_cb(it), it[RESID_COL].to_numpy().astype(np.float64),
          cat_features=C2_CATS,
          eval_set=(to_cb(iv), iv[RESID_COL].to_numpy().astype(np.float64)),
          early_stopping_rounds=RIG.PROBE_PARAMS["early_stopping_rounds"],
          use_best_model=True)
    b = int(m.get_best_iteration())
    del m
    return b


def refit(frame, best):
    prm = dict(RIG.REFIT_BASE)
    prm["iterations"] = best + 1
    m = CatBoostRegressor(allow_writing_files=False, **prm)
    m.fit(to_cb(frame), frame[RESID_COL].to_numpy().astype(np.float64),
          cat_features=C2_CATS)
    return m


small = [a for a in sorted(set(r_adep[comp_mask].tolist()))
         if t_cp.filter(pl.col(AP) == a).height < RIG.PER_AIRPORT_MIN_ROWS]
A(f"airports below {RIG.PER_AIRPORT_MIN_ROWS} rows -> global fallback: "
  f"{sorted(small) if small else 'none'}")
A(f"  {'airport':<8}{'2025 COMPLETE':>14}{'best_iter':>11}{'rank rows':>12}"
  f"{'fit s':>8}")
gbest = gmodel = None
if small:
    gbest = probe(t_cp)
    gmodel = refit(t_cp, gbest)
pred = np.full(rk.height, np.nan, dtype=np.float64)
for ap in sorted(set(r_adep[comp_mask].tolist())):
    t1 = time.time()
    tri = t_cp.filter(pl.col(AP) == ap)
    sel = (r_adep == ap) & comp_mask
    if ap in small:
        best, model = gbest, gmodel
    else:
        best = probe(tri)
        model = refit(tri, best)
    fr = rk.filter(sel)
    res = np.asarray(model.predict(to_cb(fr)), dtype=np.float64)
    pred[sel] = res + fr[D_AOBT3].to_numpy().astype(np.float64)
    A(f"  {ap:<8}{tri.height:>14,}{best:>11}{int(sel.sum()):>12,}"
      f"{time.time()-t1:>8.0f}")
    drain()
    if ap not in small:
        del model
if gmodel is not None:
    del gmodel

assert np.isfinite(pred[comp_mask]).all(), "non-finite COMPLETE prediction"
assert np.isnan(pred[~comp_mask]).all(), "G1 must not predict MISSING_4"

OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
pl.DataFrame({MID: rk[MID].to_numpy(), "pred_g1_basic": pred}).write_parquet(OUT)
A()
A(f"COMPLETE predictions finite: {int(np.isfinite(pred).sum()):,}")
A(f"MISSING_4 left null for exact F3: {int(np.isnan(pred).sum()):,}")
A(f"written: {OUT}  rows={rk.height:,}")
A(f"runtime {time.time()-T0:.0f}s")
A("NO submission file was created and nothing was submitted.")
