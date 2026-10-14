"""Train the V8 candidate model: V7 features + the five validated T4 deltas.

Feature definitions and feature construction live in
train_catboost_v8_defs.py, which is the single source of truth shared with
create_submission_v8.py. See that module for the rationale, the rejected
clock / H1 feature families, and the iteration-count justification.

Trains on ALL 2,085,047 2025 departure rows. No Jan/Jul holdout is used for
the final fit. Raw targets are kept exactly as supplied: no clipping, no
filtering, no anomaly correction.

The final fit has no evaluation set, so early stopping and use_best_model are
deliberately NOT used; the iteration count is fixed in advance from the Jan/Jul
validation evidence (316 trees).

Output: models/v8_catboost.cbm
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from train_catboost_v8_defs import (
    CB_PARAMS, DATA_DIR, FEATURES, FORBIDDEN, MINUEND, MODEL_DIR, MODEL_PATH,
    N_TREES, TARGET, T4_FEATURES, T4_SOURCE_COLS, T4_SPEC, V7_CATEGORICALS,
    V7_FEATURES, VALIDATED_T4, VALIDATED_T4_BEST_ITER,
    add_t4_features, add_time_features, add_traffic_feature,
    apply_categorical_handling, verify_t4_deltas,
)

print("=" * 70)
print("V8 = V7 features + 5 validated T4 delta features")
print("=" * 70)
print(f"V7 features : {len(V7_FEATURES)}")
print(f"T4 features : {len(T4_FEATURES)}  {T4_FEATURES}")
print(f"total       : {len(FEATURES)}")
print(f"trees       : {N_TREES}")
print(f"CatBoost    : {CB_PARAMS}")
print(f"\nvalidated T4 (2025 holdouts, for reference only - NOT used to fit):")
for fold, rmse in VALIDATED_T4.items():
    print(f"  {fold}: RMSE {rmse}  best_iter "
          f"{VALIDATED_T4_BEST_ITER[fold]} -> {VALIDATED_T4_BEST_ITER[fold] + 1} trees")
print(f"chosen trees: {N_TREES} (midpoint of "
      f"{VALIDATED_T4_BEST_ITER['JAN'] + 1} and {VALIDATED_T4_BEST_ITER['JUL'] + 1})")

# ============================================================
# LOAD AND PREPARE THE 2025 TRAINING DATA
# ============================================================

print(f"\nLoading training data from {DATA_DIR}...")
training_files = sorted(DATA_DIR.glob("training_*.parquet"))
print(f"Training files found: {len(training_files)}")
assert len(training_files) == 12, f"expected 12 files, got {len(training_files)}"

data = pd.concat([pd.read_parquet(f) for f in training_files], ignore_index=True)
print(f"Total 2025 movements: {len(data):,}")
assert len(data) == 4_167_797, f"unexpected movement count {len(data):,}"

for col in T4_SOURCE_COLS:
    assert col in data.columns, f"missing T4 source column {col}"
print("[PASS] all T4 source columns present")

print("Creating 15-minute traffic feature on 2025 data...")
data = add_traffic_feature(data)

# Taxi-out prediction = departures only. Raw targets kept as supplied.
train = data[data["PHASE_mvt"] == "DEP"].copy()
print(f"2025 departure rows: {len(train):,}")
assert len(train) == 2_085_047, f"unexpected departures {len(train):,}"
assert train[TARGET].notna().all(), "some training departures have no target"

train = add_time_features(train)
train = add_t4_features(train)
train = apply_categorical_handling(train)

print(f"\nRaw training target (no cleaning applied):")
print(f"  mean   = {train[TARGET].mean():.2f}")
print(f"  median = {train[TARGET].median():.2f}")
print(f"  std    = {train[TARGET].std():.2f}")
print(f"  min    = {train[TARGET].min():.2f}")
print(f"  max    = {train[TARGET].max():.2f}")
print(f"  > 3600s = {int((train[TARGET] > 3600).sum()):,}")
print(f"  > 7200s = {int((train[TARGET] > 7200).sum()):,}")
print(f"  > 28800s= {int((train[TARGET] > 28800).sum()):,}")

print("\nT4 feature coverage over all training departures:")
for name, _ in T4_SPEC:
    s = train[name]
    print(f"  {name:9s} nan={int(s.isna().sum()):>7,} "
          f"({100 * s.isna().mean():6.3f}%)  min={s.min():10.1f}  "
          f"median={s.median():8.1f}  max={s.max():10.1f}")

print("\nT4 delta unit re-check (independent integer microsecond epoch path):")
verify_t4_deltas(train)
print("[PASS] all T4 deltas reproduced exactly, in seconds")

for f in FEATURES:
    assert f in train.columns, f"missing feature {f}"
    assert f not in FORBIDDEN, f"forbidden feature {f}"
print(f"[PASS] all {len(FEATURES)} V8 features present, none forbidden")

# ============================================================
# FIT
# ============================================================

model = CatBoostRegressor(iterations=N_TREES, **CB_PARAMS)

print(f"\nTraining CatBoost on all {len(train):,} 2025 departures "
      f"({N_TREES} trees)...")
model.fit(train[FEATURES], train[TARGET], cat_features=V7_CATEGORICALS)

print(f"Built trees: {model.tree_count_}")
assert model.tree_count_ == N_TREES, \
    f"tree count {model.tree_count_} != {N_TREES}"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save_model(str(MODEL_PATH))
print(f"Saved model: {MODEL_PATH}")

print("\nFeature importance (all 17):")
imp = pd.Series(model.get_feature_importance(), index=FEATURES)
print(imp.sort_values(ascending=False).round(4).to_string())
t4_share = 100 * imp[T4_FEATURES].sum() / imp.sum()
print(f"\nT4 share of importance: {t4_share:.1f}%")

print("\nV8 training complete. This script writes no submission.")
