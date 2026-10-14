# ADS-B correction pipeline (vigorous-glove, PRC Data Challenge 2026)

This pipeline adds an **ADS-B-based corrector** and two rule-based post-steps on
top of a base taxi-out model. The base model predicts `TAXITIME_SEC_mvt`; the
corrector learns the base model's remaining error from what the aircraft actually
did on the ground (open ADS-B data) and from the airport's recent traffic, and adds
a bounded correction.

```text
base model (V17: C2Q_CLEAN + G1 geometry, 50/50; F3 for MISSING_4 rows)  ──►  base prediction
ADS-B + geometry + airport-context features ──► corrector (CatBoost + LightGBM, 3 seeds)
                                              ──► correction (clipped ±1,800 s)
corrected = max(0, base prediction + correction)
          ──► LOBT window clip            (lobt_clip.py)
          ──► MISSING_4 per-airport line  (nm_line.py)
          ──► Rome schedule-copy model    (rome_copy.py)
          ──► submission
```

`BLOCK_TIME_UTC_mvt` is never loaded. No 2026 targets are used anywhere; all
training uses 2025 only.

---

## 1. Scripts (in `experiments/`), in run order

| # | Script | What it does | Output | Typical time |
|---|---|---|---|---|
| 1 | `rebuild_v8_prep_cache.py` | Builds the 2025 departure feature table with the V8 reference definitions | `data/processed/v8_prep_cache.parquet` | < 1 min |
| 2 | `run_janjul_v4_cached.py` | Honest V4 benchmark; per-row OOF cache for JAN/JUL 2025 (V4-style F3 used for MISSING_4 rows) | `data/processed/janjul_v4_cache_{JAN,JUL}.parquet` | ~4 h (laptop) |
| 3 | `adsb_extract.py` | Streams adsb.lol daily archives, keeps ground / low-altitude points within 7 km of the 10 airports (altitude relative to airport elevation); uses `PREFERRED_RELEASES.txt`, falls back to replicas and the next year's repository for year-boundary days; `--audit` / `--repair` | `data/external/adsb_v2/points_YYYY-MM-DD.parquet`, `manifest.jsonl` | ~1–2 min/day, 8 workers |
| 4 | `metar_download.py` | Airport weather (IEM ASOS/METAR) — tested, **not used** in the final corrector | `data/external/metar/metar_10apt.parquet` | minutes |
| 5 | `adsb_features.py` | Per-departure ADS-B features (v4): callsign → aircraft match (incl. IATA→ICAO), final ground segment, first movement, parking-spot distances, OSM stand distances, runway distances | `data/processed/adsb_features.parquet` | ~30–40 min (`--months all`) |
| 6 | `context_features.py` | Airport-context features from movements strictly **before** the departure's takeoff: arrival taxi-in (30/60 min, same runway), time since last arrival at the same stand, ADS-B taxi duration and MVT−AOBT_3 of earlier departures | `data/processed/context_features.parquet` | ~20 s |
| 6b | `gate_features.py` | Gate-retention features at each departure's AOBT_3: departures overdue at the gate (EOBT ≤ t < AOBT_3; all / same runway), active same-runway departures (AOBT_3 ≤ t < MVT), recent mean AOBT_3−SCHED, departures in the last hour | `data/processed/gate_features.parquet` | ~25 s |
| 7 | `adsb_corrector.py` | Validates (JAN↔JUL) and trains the corrector on the base model's honest OOF predictions; builds a submission | `reports/adsb_corrector.txt`, `submissions/vigorous-glove_vN.parquet` | ~6 min validation |
| 8 | `lobt_clip.py` | Projects each prediction onto the LOBT window `[MVT−LOBT−3606, MVT−LOBT+3606]` s, floor 0 | `reports/lobt_clip.txt`, new submission | seconds |
| 9 | `nm_line.py` | For MISSING_4 rows (AOBT_3, EOBT_1, IOBT, LOBT all missing) with MVT−SCHED > 6 h: per-airport line `taxi = a + b·(MVT−SCHED)` fitted on 2025 (airports with ≥ 20 such rows) | `reports/nm_line.txt`, new submission | seconds |
| 10 | `rome_copy.py` | LIRF only, MISSING_4 rows with 1 h < MVT−SCHED ≤ 6 h: CatBoost probability that the off-block time is a copy of the schedule (airline, operator, aircraft type, runway, stand area, …); final = 0.5·current + 0.5·(p·(MVT−SCHED) + (1−p)·non-copy mean) | `reports/rome_copy.txt`, new submission | ~1 min |
| – | `blend2.py` | 50/50 average of two pipelines (tested V17- vs V32-based: −0.12, rejected) | – | – |
| – | `adsb_direct.py` | ADS-B-first direct model + blend — tested, **not used** (fails July) | – | – |

Base-model inputs come from the base-model track (separate code):
honest OOF `data/processed/v19_quick_best_oof_{JAN,JUL}.parquet` (columns
`MVT_ID_mvt, fold, y, pred`) and the matching 2026 ranking file
`submissions/v17_lite_blend_ranking.parquet` (`MVT_ID_mvt, TAXITIME_SEC_mvt`).

---

## 2. Reproducing the current submission

```powershell
# one-off data preparation
python -u experiments\rebuild_v8_prep_cache.py
python -u experiments\run_janjul_v4_cached.py
python -u experiments\adsb_extract.py --workers 8
python -u experiments\adsb_features.py --months all
python -u experiments\context_features.py
python -u experiments\gate_features.py

# 1) corrector: validation (JAN <-> JUL, both months must improve)
python -u experiments\adsb_corrector.py --validate-only --gate --context --seeds 3 `
    --second-model lgbm --drop-features adsb_npts --m4-from-v4 `
    --base-oof data\processed\v19_quick_best_oof_JAN.parquet data\processed\v19_quick_best_oof_JUL.parquet `
    --save-oof data\processed\corrector_oof_gate.parquet

# 2) corrector: build on the 2026 ranking file
python -u experiments\adsb_corrector.py --version 70 --gate --context --seeds 3 `
    --second-model lgbm --drop-features adsb_npts --m4-from-v4 `
    --base-oof data\processed\v19_quick_best_oof_JAN.parquet data\processed\v19_quick_best_oof_JUL.parquet `
    --base-submission submissions\v17_lite_blend_ranking.parquet

# 3) LOBT window clip
python -u experiments\lobt_clip.py --oof data\processed\corrector_oof_gate.parquet
python -u experiments\lobt_clip.py --submission submissions\vigorous-glove_v70.parquet --version 71

# 4) MISSING_4 per-airport line (> 6 h)
python -u experiments\nm_line.py --oof data\processed\corrector_oof_gate.parquet
python -u experiments\nm_line.py --submission submissions\vigorous-glove_v71.parquet --version 72

# 5) Rome schedule-copy model
python -u experiments\rome_copy.py --oof data\processed\corrector_oof_gate.parquet
python -u experiments\rome_copy.py --submission submissions\vigorous-glove_v72.parquet --version 73
```

Useful corrector options:

| Option | Meaning |
|---|---|
| `--base-oof` / `--base-submission` | base model's honest OOF files and its 2026 ranking file |
| `--m4-from-v4` | MISSING_4 rows use the V4-style F3 OOF (matches the ranking file) |
| `--context` | add the airport-context features (`context_features.parquet`) |
| `--gate` | add the gate-retention features (`gate_features.parquet`) |
| `--seeds N` | average N seeds |
| `--second-model lgbm` | equal-weight average with a LightGBM corrector |
| `--drop-features a,b` | leave features out (e.g. `adsb_npts`, a receiver-density feature that drifts 2025 → 2026) |
| `--save-oof FILE` | save the corrected honest JAN/JUL predictions (input for steps 3 and 4) |
| `--weather`, `--extra-oof`, `--zero-unmatched`, `--max-trees` | tested options, not used in the final setup |
| `--version N` | output file `vigorous-glove_vN.parquet` (never overwrites) |

Upload (MinIO / S3 API, profile with the team's keys configured locally — keys are never stored in the repo):

```powershell
aws s3 cp submissions\vigorous-glove_vN.parquet s3://prc-2026-vigorous-glove/ --profile prc --endpoint-url https://s3.opensky-network.org
```

---

## 3. Validation protocol

- Folds: **JAN** = train on all other 2025 months, predict January 2025; **JUL** likewise
  (153,706 + 190,713 = 344,419 departures).
- Base predictions are honest OOF: tree counts chosen on an inner split
  (`abs(MVT_ID_mvt) % 5 == 0`) of the training months only.
- The corrector is validated **train JAN → score JUL** and **train JUL → score JAN**;
  pooled RMSE = sqrt(total squared error / 344,419).
- Rule-based post-steps are fitted without the scored month (the `nm_line.py` lines for
  January are fitted on Feb–Dec, etc.); the LOBT window has no fitted parameters.
- **Keep rule:** both months must improve (pooled gain ≥ 2 for new ideas);
  report the share of the gain from the top-20 rows.
- Leaderboard scores are used only as a final check, never for tuning.

Honest JAN/JUL results of the current pipeline:

| Step | JAN | JUL | Pooled |
|---|---|---|---|
| V17 base | 361.22 | 310.42 | 334.05 |
| + ADS-B corrector + context | 346.18 | 292.57 | 317.62 |
| + MISSING_4 line (> 6 h) | 338.09 | 283.21 | 308.91 |
| + Rome schedule-copy model | 336.48 | 280.73 | 306.86 |
| + gate-retention features in the corrector (V73) | 335.82 | 280.19 | 306.27 |

(Base row uses the V4-style F3 on MISSING_4 rows, as in the corrector. The LOBT clip
alone on the V17 OOF: 336.86 → 334.75.)

Tested and rejected (failed the both-months rule or no gain): METAR weather in the
corrector, ADS-B-first direct model blend, 6,000-tree corrector, schedule-copy
classifier mixture for all airports on top of LOBT clip + line rule (gain overlaps; January worse),
base V32 (V17 + residual layer) under the corrector (318.26 vs 317.62) and its 50/50 blend (−0.12, noise).
Note: the Rome model's local gain (−2.05) was concentrated in ~20 flights; it was uploaded as a
free attempt (ranking uses the best submission) and scored 267.92 on the leaderboard.

---

## 4. Submission history (leaderboard RMSE)

| Version | Content | Score |
|---|---|---|
| V4 | A4 + F3 | 327.78 |
| V6 | V4 + ADS-B corrector (v1 features) | 300.70 |
| V7 | C2 + F3 (base-model track) | 321.05 |
| V8 | V6 + (V7 − V4), simple stacking | 298.27 |
| V9 | corrector trained on C2+F3 OOF, parking-spot features | 295.52 |
| V10 | as V9 with the elevation-fixed extraction | 296.16 |
| V14 | V17 base + ADS-B corrector, 3 seeds | 288.91 |
| V15 | V14 setup + context features + LOBT clip | 281.22 |
| V51 | V15 + MISSING_4 per-airport line (> 6 h) | 272.19 |
| V52 | V51 + Rome (LIRF) schedule-copy model, 50/50 with V51 | 267.92 |
| **V73** | V52 pipeline with gate-retention features in the corrector (`--gate`) | **265.79** |

---

## 5. Data sources, ideas and attribution

| Source | Used for | Licence / terms |
|---|---|---|
| **adsb.lol** globe_history (github.com/adsblol) | ADS-B ground tracks | **ODbL 1.0**, © adsb.lol contributors |
| **OpenStreetMap** (Overpass, `aeroway=parking_position/gate`) | stand coordinates | **ODbL 1.0**, © OpenStreetMap contributors |
| **OurAirports** runways.csv / airports.csv | runway thresholds, airport reference points | public domain |
| **OpenFlights** airlines.dat | IATA → ICAO airline codes for callsign matching | **ODbL 1.0**, © OpenFlights contributors |
| **Iowa Environmental Mesonet** ASOS/METAR archive | weather (tested, not used) | free / open data service |
| PRC Data Challenge 2026 dataset | training / ranking data | challenge terms |

Ideas credited to team **outgoing-boat** (github.com/EnioAguiar/prc-taxiout-2026, GPL-3.0):
the airport-context and gate-retention features, the LOBT window (verified independently on all 2,062,577
2025 departures with a LOBT: max |BLOCK − LOBT| = 3606 s) and the per-airport line for
MISSING_4 rows above 6 h, and the schedule-copy idea behind the Rome model. All code here is an independent re-implementation; no code was copied.

Frozen geometry: `data/external/b4a_geometry/frozen_geometry.pkl` (+ `frozen_geometry_meta.json`
with source URLs and SHA-256 hashes).