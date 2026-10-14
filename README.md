# prc-data-challenge-2026 — team vigorous-glove

Our solution for the **PRC Data Challenge 2026**: predicting departure taxi-out time
(`TAXITIME_SEC_mvt` = take-off time − off-block time) for 10 European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM).
Ranking metric: RMSE on all January and July 2026 departures (344,841 flights).

**Best leaderboard score: 267.92 (submission V52).**

## How the solution works

```text
1. Base model (V17)
   - CatBoost tabular model
   - C2Q_CLEAN feature set + G1 geometry blend
   - F3 fallback for MISSING_4 rows

2. ADS-B residual corrector
   - CatBoost + LightGBM, 3 seeds
   - learns the remaining base-model error using open ADS-B ground tracks
   - adds airport-context features from recent arrivals/departures
   - bounded residual correction

3. Final post-processing
   - LOBT window clip
   - MISSING_4 per-airport line for flights with MVT−SCHED > 6 h
   - LIRF schedule-copy model for selected MISSING_4 flights
```

`BLOCK_TIME_UTC_mvt` is never used as a predictor. The final pipeline is trained using 2025 data only.

## Final pipeline

```text
V17 base
  → ADS-B + airport-context corrector
  → LOBT window clip
  → MISSING_4 per-airport line
  → Rome (LIRF) schedule-copy model
  → submission V52
```

## Leaderboard history

| Version | Content | Leaderboard RMSE |
|---|---|---:|
| V4 | first tuned base model | 327.78 |
| V6 | + ADS-B corrector | 300.70 |
| V15 | V17 base + context corrector + LOBT clip | 281.22 |
| V51 | + MISSING_4 per-airport line | 272.19 |
| **V52** | + Rome schedule-copy model | **267.92** |

## Repository layout

| Folder / file | Content |
|---|---|
| `src/` | base-model preprocessing, feature engineering and CatBoost training |
| `experiments/` | ADS-B extraction/features, airport context, corrector, validation and post-processing |
| `results/` | experiment summaries |
| `README_adsb_pipeline.md` | full run order, commands, validation protocol, results and data attribution |
| `data/` | not committed; competition data and downloaded external data are obtained separately |

For the complete reproduction procedure, see **[README_adsb_pipeline.md](README_adsb_pipeline.md)**.

## Validation

Every change was evaluated on honest held-out **January and July 2025** departures.
The scored month was excluded from the training data used to produce its predictions.
The ADS-B corrector was evaluated cross-month, and rule-based post-processing was also fitted without the scored month.

Final honest JAN/JUL validation for the current pipeline:

| Step | JAN RMSE | JUL RMSE | Pooled RMSE |
|---|---:|---:|---:|
| V17 base | 361.22 | 310.42 | 334.05 |
| + ADS-B corrector + context | 346.18 | 292.57 | 317.62 |
| + MISSING_4 line | 338.09 | 283.21 | 308.91 |
| + Rome schedule-copy model | 336.48 | 280.73 | 306.86 |

Leaderboard scores were used only as a final check, not as the main validation signal.

## Data

- **Competition data:** PRC Data Challenge 2026 dataset; place under the expected `data/` structure locally.
- **ADS-B:** adsb.lol globe history (ODbL 1.0), used for ground-track features.
- **OpenStreetMap:** airport stand / parking-position information where used (ODbL 1.0).
- **OurAirports:** runway and airport reference data (public domain).
- **OpenFlights:** airline-code reference data (ODbL 1.0).
- **Iowa Environmental Mesonet:** METAR/ASOS weather data was tested but is **not used** in the final corrector.

Competition data and downloaded external datasets are not committed to this repository.
See section 5 of **[README_adsb_pipeline.md](README_adsb_pipeline.md)** for exact attribution and usage.

## Reproducing V52

The exact run order and commands are documented in **[README_adsb_pipeline.md](README_adsb_pipeline.md)**.
In summary:

```text
1. build the 2025 base/OOF inputs
2. extract and featurize ADS-B data
3. build airport-context features
4. train and apply the CatBoost + LightGBM corrector
5. apply the LOBT clip
6. apply the MISSING_4 per-airport line
7. apply the Rome schedule-copy model
8. write the final V52 parquet submission
```

The final submission contains only the required prediction output and is uploaded separately from the source repository.

## Credits

Several modelling ideas were informed by team **outgoing-boat**'s public solution:
`github.com/EnioAguiar/prc-taxiout-2026` (GPL-3.0).

Ideas credited include airport-context features, the LOBT window, the per-airport MISSING_4 line, and the schedule-copy concept. All code in this repository is our own independent implementation.

## Licence

GPL-3.0 — see [LICENSE](LICENSE). External datasets retain their own licences.
