# prc-data-challenge-2026 — team vigorous-glove

Our solution for the **PRC Data Challenge 2026**: predicting departure taxi-out time
(`TAXITIME_SEC_mvt` = take-off time − off-block time) for 10 European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM).
Ranking metric: RMSE on all January and July 2026 departures (344,841 flights).

**Best leaderboard score: 267.92 (submission V52).**

## How the solution works

```text
1. Base model (CatBoost, tabular)        src/  ............ ~321 → V17 base
2. ADS-B corrector (CatBoost + LightGBM) experiments/ ..... learns the base model's remaining error
   from open ADS-B ground tracks (adsb.lol) and airport context (recent arrivals/departures)
3. Rule-based post-steps                 experiments/
   a. LOBT window clip       |off-block − LOBT| ≤ 3606 s holds for 100 % of 2025 flights
   b. MISSING_4 line         per-airport line for flights without NM times, > 6 h late
   c. Rome schedule-copy     LIRF flights whose off-block time is a copy of the schedule
```

| Version | Content | Leaderboard RMSE |
|---|---|---|
| V4 | first tuned base model | 327.78 |
| V6 | + ADS-B corrector | 300.70 |
| V15 | V17 base + corrector with context features + LOBT clip | 281.22 |
| V51 | + MISSING_4 per-airport line | 272.19 |
| **V52** | + Rome schedule-copy model | **267.92** |

## Repository layout

| Folder | Content |
|---|---|
| `src/` | base-model track: data exploration, feature experiments, CatBoost training scripts (`train_catboost_v*.py`) and submission builders |
| `experiments/` | ADS-B extraction and features, airport-context features, the corrector, post-processing rules, honest validation |
| `data/` | not in the repo (competition data and downloaded ADS-B; see below) |

Full run order, exact commands, validation protocol and all results:
**[README_adsb_pipeline.md](README_adsb_pipeline.md)**.

## Validation

Every change was judged on honest out-of-fold predictions for **January and July 2025**
(models trained without the scored month) and kept only if **both months improved**.
Leaderboard scores were used as a final check, not for tuning.

## Data

- Competition data: PRC Data Challenge 2026 (place in `data/raw/`; challenge terms apply).
- ADS-B: adsb.lol globe_history (ODbL 1.0), downloaded by `experiments/adsb_extract.py`.
- OpenStreetMap stands, OpenFlights airline codes (ODbL 1.0), OurAirports runways (public domain).

Details and attribution: section 5 of [README_adsb_pipeline.md](README_adsb_pipeline.md).

## Credits

Several ideas (airport-context features, LOBT window, per-airport line for flights without
NM times, schedule-copy behaviour) were inspired by team **outgoing-boat**'s public solution
(github.com/EnioAguiar/prc-taxiout-2026, GPL-3.0). All code here is our own implementation.

## Licence

GPL-3.0 — see [LICENSE](LICENSE). External data keeps its own licence.
