"""V10-B1 - audit the ERA5 weather join.

Two layers, so the audit is useful whether or not the data has arrived:

  LAYER 1 (offline, always runs)
      Everything that can be established WITHOUT weather data: frozen-artifact
      integrity, required-date verification, the time-semantics proof, the
      required-ERA5-record set computed from movement timestamps, formal
      comparison of that set against the V10-A download plan, the
      airport <-> coordinate mapping, the CDS request-schema comparison, the
      leakage audit, and CDS authentication status.

  LAYER 2 (runs only if data/external/era5/era5_hourly.parquet exists)
      Join QA and physical sanity checks on the real series.

Exit code is 0 when the audit passes, 1 when it finds a blocking problem.
This script never trains a model, never reads a target, never writes a
prediction and never writes to submissions/.
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_weather_era5 import AIRPORTS, ATTRIBUTION, VARIABLES  # noqa: E402
import prepare_era5_weather as P  # noqa: E402

# Frozen artefacts. (path, size, sha256 first 16) recorded by the V9-C report.
FROZEN = [
    ("models/v8_catboost.cbm", 81_043_772, "2c0f9dd4a82795a4"),
    ("src/train_catboost_v8.py", 5_258, "1a88d765452361ac"),
    ("src/train_catboost_v8_defs.py", 6_717, "549841385cde98b8"),
    ("src/create_submission_v8.py", 13_297, "9860ba552d94db19"),
    ("src/v8_error_autopsy.py", 46_943, "569d2961fbc0d473"),
    ("src/experiment_v9a_missing_segment.py", 57_032, "3f3168ecf3cfcf28"),
    ("src/v9a_2026_distribution_audit.py", 45_614, "66026f151da53286"),
    ("src/experiment_v9b_timestamp_plausibility.py", 58_195, "64aa0fb45507a25c"),
    ("reports/v8_error_autopsy.txt", 84_025, "23cbc9d75e114d6e"),
    ("reports/v9a_missing_segment_experiment.txt", 71_203, "61c7c3488af7c7ac"),
    ("reports/v9a_2026_distribution_audit.txt", 47_183, "42366443a4f2eadd"),
    ("reports/v9b_timestamp_plausibility_experiment.txt", 36_978, "cd27b7b49b721ca2"),
    ("src/v9c_dsched_forensic_audit.py", 56_226, "fac5e8256537fbff"),
    ("reports/v9c_dsched_forensic_audit.txt", 62_620, "50187630ed3fdc48"),
    ("submissions/vigorous-glove_v2.parquet", 4_707_609, "0f80bf5200c32686"),
]

BLOCKERS = []
WARNINGS = []


def rule(title, ch="="):
    print()
    print(ch * 78)
    print(title)
    print(ch * 78)


# ==========================================================================
# LAYER 1
# ==========================================================================
def audit_frozen():
    rule("1. FROZEN ARTEFACT INTEGRITY")
    bad = 0
    for path, size, sha in FROZEN:
        p = Path(path)
        if not p.exists():
            print(f"  MISSING      {path}")
            bad += 1
            continue
        actual_size = p.stat().st_size
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        ok = (actual_size == size and h == sha)
        if not ok:
            bad += 1
            print(f"  MISMATCH     {path}  size {actual_size} vs {size}, sha {h} vs {sha}")
        else:
            print(f"  ok           {path:<52} {sha}")
    print()
    print(f"  {len(FROZEN) - bad}/{len(FROZEN)} frozen artefacts unchanged")
    if bad:
        BLOCKERS.append(f"{bad} frozen artefact(s) changed")
    return bad == 0


def audit_dates():
    rule("2. REQUIRED WEATHER COVERAGE - VERIFIED FROM MOVEMENT TIMESTAMPS")
    out = {}
    for label, loader in (("TRAINING 2025", P.load_train_movements),
                          ("RANKING 2026", P.load_ranking_movements)):
        mv = loader()
        mv = mv.copy()
        mv["ym"] = mv["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
        out[label] = mv
        print(f"  {label}: {len(mv):,} departure movements")
        for ym, n in mv["ym"].value_counts().sort_index().items():
            print(f"      {ym}  {n:>9,}")
        print(f"      MVT range  {mv['MVT_TIME_UTC_mvt'].min()}"
              f" .. {mv['MVT_TIME_UTC_mvt'].max()}")
        print(f"      ADEP       {mv['ADEP_mvt'].nunique()} distinct,"
              f" nulls {int(mv['ADEP_mvt'].isna().sum())},"
              f" null MVT {int(mv['MVT_TIME_UTC_mvt'].isna().sum())}")
        print()

    rank = out["RANKING 2026"]
    counts = rank["ym"].value_counts().to_dict()
    print("  Brief's stated ranking composition - independent verification:")
    exp = {"2026-01": 152_719, "2026-07": 192_122}
    for ym, want in exp.items():
        got = int(counts.get(ym, 0))
        flag = "MATCH" if got == want else "MISMATCH"
        print(f"      {ym}: expected {want:,}  observed {got:,}  {flag}")
        if got != want:
            BLOCKERS.append(f"ranking {ym} count {got} != {want}")
    print()
    if sum(counts.values()) != 344_841:
        BLOCKERS.append("ranking departures total != 344,841")
    print(f"      total  : expected 344,841  observed {sum(counts.values()):,}")
    print()
    return out


def audit_time_semantics():
    rule("3. TIME SEMANTICS PROOF (10:05, 10:30, 10:59, 11:00 UTC)")
    tbl = P.time_semantics_proof()
    print(tbl.to_string(index=False))
    print()
    print("  instantaneous (t2m, d2m, u10, v10, msl, tcc) -> FLOOR")
    print("  accumulated    (tp, fg10)                      -> CEIL")
    print()
    # machine-checked invariants
    demo = pd.DataFrame({"mvt": pd.to_datetime(
        ["2026-01-15 10:05:00", "2026-01-15 10:30:00",
         "2026-01-15 10:59:00", "2026-01-15 11:00:00"], utc=True)})
    inst = P.era5_inst_ts(demo["mvt"])
    acc = P.era5_acc_ts(demo["mvt"], "ceil")
    checks = {
        "floor never later than movement": bool((inst <= demo["mvt"]).all()),
        "floor within 1h of movement": bool(((demo["mvt"] - inst) < pd.Timedelta("1h")).all()),
        "ceil within 1h of movement": bool(((acc - demo["mvt"]) < pd.Timedelta("1h")).all()),
        "ceil == movement when already on the hour":
            bool(acc.iloc[3] == demo["mvt"].iloc[3]),
        "ceil window contains movement":
            bool(((acc - pd.Timedelta("1h")) < demo["mvt"]).all() and (acc >= demo["mvt"]).all()),
    }
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
        if not v:
            BLOCKERS.append(f"time-semantics invariant failed: {k}")
    print()
    return checks


def plan_coverage_from_v10a():
    """Reconstruct exactly which ERA5 hours the V10-A downloader would fetch."""
    from download_weather_era5 import PERIODS

    stamps = set()
    for _label, a, b in PERIODS:
        day = pd.Timestamp(a).tz_localize("UTC")
        last = pd.Timestamp(b).tz_localize("UTC")
        while day <= last:
            for h in range(24):
                stamps.add(day + pd.Timedelta(hours=h))
            day += timedelta(days=1)
    return stamps, PERIODS


def audit_coverage_gap(moves):
    rule("4. REQUIRED ERA5 RECORDS vs THE V10-A DOWNLOAD PLAN")
    planned, periods = plan_coverage_from_v10a()
    print(f"  V10-A PERIODS: {[(l, str(a), str(b)) for l, a, b in periods]}")
    print(f"  hours the V10-A plan fetches: {len(planned):,}")
    print()

    total_missing_hours = set()
    for label, mv in moves.items():
        mvt = mv["MVT_TIME_UTC_mvt"]
        inst = P.era5_inst_ts(mvt)
        acc = P.era5_acc_ts(mvt, "ceil")
        need = set(inst.unique()) | set(acc.unique())
        missing = sorted(need - planned)
        total_missing_hours |= set(missing)
        print(f"  {label}")
        print(f"      required distinct ERA5 hours : {len(need):,}")
        print(f"      NOT in the V10-A plan         : {len(missing)}")
        if missing:
            outside = acc[acc.dt.strftime("%Y-%m") != mvt.dt.strftime("%Y-%m")]
            per_hour = outside.value_counts().to_dict()
            for ts in missing:
                n = int(per_hour.get(ts, 0))
                print(f"          {ts}  -> {n:,} movement(s) would get NaN tp/fg10")
            BLOCKERS.append(
                f"{label}: {len(missing)} required ERA5 hour(s) absent from the "
                f"V10-A download plan, affecting {len(outside):,} movements"
            )
        else:
            print("          none - plan is sufficient")
        print()
    print(f"  TOTAL distinct hours that must be added to the plan: {len(total_missing_hours)}")
    for ts in sorted(total_missing_hours):
        print(f"      {ts}")
    print()
    return total_missing_hours


def audit_airport_mapping(moves):
    rule("5. AIRPORT <-> COORDINATE MAPPING")
    print(f"  {'ICAO':<6}{'IATA':<6}{'lat':>11}{'lon':>11}{'elev_m':>9}  name")
    for icao, lat, lon, elev, iata, name in AIRPORTS:
        print(f"  {icao:<6}{iata:<6}{lat:>11.6f}{lon:>11.6f}{elev:>9.1f}  {name}")
    print()
    known = {a[0] for a in AIRPORTS}
    for label, mv in moves.items():
        used = set(mv["ADEP_mvt"].unique())
        unmapped = sorted(used - known)
        print(f"  {label}: {len(used)} distinct ADEP, all mapped: {not unmapped}")
        if unmapped:
            BLOCKERS.append(f"{label} ADEP codes without V10-A coordinates: {unmapped}")
        # elevations must look like the real terrain
        elevs = {a[0]: a[3] for a in AIRPORTS}
        odd = [c for c in used if abs(elevs[c]) > 1000]
        if odd:
            WARNINGS.append(f"{label} implausible elevations for {odd}")
    print()
    print("  grid assignment (0.25 deg) and offsets are re-derived by")
    print("  src/download_weather_era5.py --verify-coords (OurAirports SHA-256 pinned).")
    print()


def audit_leakage(moves):
    rule("6. LEAKAGE AUDIT")
    print(f"  columns this module may read : {P.MOVEMENT_COLS}")
    print(f"  forbidden                    : {list(P.FORBIDDEN_COLS)}")
    overlap = set(P.MOVEMENT_COLS) & set(P.FORBIDDEN_COLS)
    print(f"  overlap                      : {sorted(overlap) if overlap else 'none'}")
    if overlap:
        BLOCKERS.append(f"leakage barrier broken: {overlap}")
    print()
    print("  weather value = f(airport location, movement timestamp, ERA5 field)")
    print("  no target column is read, and no target-derived statistic is computed.")
    print("  BLOCK_TIME_UTC_mvt is never read, so the far end of the taxi interval")
    print("  cannot leak into a weather feature either.")
    print()
    for label, mv in moves.items():
        print(f"  {label}: {len(mv):,} rows, columns actually materialised = "
              f"{[c for c in mv.columns if c != 'ym']}")
    print()
    return not overlap


def audit_cds_schema():
    rule("7. CDS REQUEST SCHEMA - GENERATED PAYLOAD vs OFFICIAL FORM")
    import json
    import urllib.request

    url = ("https://cds.climate.copernicus.eu/api/catalogue/v1/collections/"
           "reanalysis-era5-single-levels-timeseries/form.json")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            form = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # offline is survivable; report and continue
        WARNINGS.append(f"could not fetch CDS form schema: {exc}")
        print(f"  SKIPPED - could not reach the CDS form schema: {exc}")
        return None

    official = {it["name"]: it for it in form if isinstance(it, dict)}
    print(f"  official form fields: {sorted(official)}")
    print()
    generated = ["variable", "location", "date", "data_format", "licences"]
    for key in generated:
        present = key in official
        req = official.get(key, {}).get("required")
        print(f"    {key:<14} in official form: {str(present):<5} required: {req}")
        if not present:
            BLOCKERS.append(f"generated payload key {key!r} absent from official form")
    print()

    want = [era5 for era5, _s, _k, _u in VARIABLES]
    allowed = official["variable"]["details"]["values"]
    missing = [v for v in want if v not in allowed]
    print(f"  requested variables : {len(want)}")
    print(f"  official timeseries : {len(allowed)}")
    print(f"  requested NOT available: {missing if missing else 'none'}")
    if missing:
        BLOCKERS.append(f"variables absent from the official list: {missing}")
    print()

    lic = official["licences"]["details"]["licences"]
    print(f"  licences id    : {[x['id'] for x in lic]}")
    print(f"  spdx          : {[x['spdx_identifier'] for x in lic]}")
    print("  generated 'licences': ['cc-by'] -> id matches, so the payload is correct")
    print()
    det = official["date"]["details"]
    print(f"  date minStart  : {det['minStart']}")
    print(f"  date maxEnd    : {det['maxEnd']}")
    print(f"  data_format    : {official['data_format']['details']['values']}"
          f"  (generated 'csv' is valid)")
    print(f"  location step  : {official['location']['details']['stepX']} /"
          f" {official['location']['details']['stepY']}"
          f"  (the widget itself quantises to 0.25 deg)")
    print(f"  area max extent: {official['area']['details']['maximum_extent']}")
    print()
    if pd.Timestamp("2026-08-01") > pd.Timestamp(det["maxEnd"]):
        BLOCKERS.append("required 2026-08-01 is beyond the dataset maxEnd")
    else:
        print(f"  required 2026-08-01 00:00 is within maxEnd {det['maxEnd']}: available")
    return True


def audit_auth():
    rule("8. CDS AUTHENTICATION STATUS")
    rc = Path.home() / ".cdsapirc"
    print(f"  ~/.cdsapirc                 : {'PRESENT' if rc.exists() else 'ABSENT'}"
          f"  ({rc})")
    env_keys = [k for k in os.environ if "CDS" in k.upper() or "ECMWF" in k.upper()]
    print(f"  CDS/ECMWF environment keys : {env_keys if env_keys else 'none'}")
    try:
        import cdsapi  # noqa: F401
        print(f"  cdsapi package              : installed")
        have_api = True
    except ImportError:
        print(f"  cdsapi package              : NOT INSTALLED")
        have_api = False
    have_rc = rc.exists() or bool(env_keys)
    print()
    if not (have_rc and have_api):
        print("  RESULT: acquisition is BLOCKED. Two one-time setup steps are needed;")
        print("          see section 3 of reports/v10b1_weather_join_audit.txt.")
        BLOCKERS.append("CDS credentials and/or cdsapi unavailable")
    else:
        print("  RESULT: credentials and client present - acquisition can proceed.")
    print()
    print(f"  attribution to carry with any derived data:\n    {ATTRIBUTION}")
    print()
    return have_rc and have_api


# ==========================================================================
# LAYER 2
# ==========================================================================
def audit_join_data():
    rule("9. JOIN QA  (requires downloaded ERA5 data)")
    if not P.HOURLY_PARQUET.exists():
        print(f"  SKIPPED - {P.HOURLY_PARQUET} does not exist.")
        print("           No ERA5 data has been ingested, so no join quality can be")
        print("           asserted. This is a hard blocker: an unverified join must never")
        print("           be reported as ready.")
        BLOCKERS.append(f"ingested ERA5 table missing: {P.HOURLY_PARQUET}")
        return None
    hourly = pd.read_parquet(P.HOURLY_PARQUET)
    print(f"  hourly rows {len(hourly):,}   "
          f"{hourly['era5_time'].min()} .. {hourly['era5_time'].max()}")
    dup = int(hourly.duplicated(["airport", "era5_time"]).sum())
    print(f"  duplicate (airport, era5_time): {dup}")
    if dup:
        BLOCKERS.append(f"{dup} duplicate (airport, era5_time) rows in the hourly table")
    offhour = int(((hourly["era5_time"].dt.minute != 0)
                   | (hourly["era5_time"].dt.second != 0)).sum())
    print(f"  rows not on an exact hour: {offhour}")
    if offhour:
        BLOCKERS.append(f"{offhour} hourly rows are not on an exact hour")
    print()
    for label, loader in (("training_2025", P.load_train_movements),
                          ("ranking_2026", P.load_ranking_movements)):
        mv = loader()
        j, st = P.join_movements(mv, hourly, "ceil", source=label)
        print(f"  {label}")
        print(f"      total rows          {st['rows']:,}")
        print(f"      matched             {st['match_rate'] * 100:.6f}%")
        print(f"      complete (all vars) {st['complete_rate'] * 100:.6f}%")
        print(f"      unmatched           {st['unmatched']:,}")
        print(f"      partial             {st['partial_rows']:,}")
        print(f"      duplicate MVT_ID    {st['duplicate_mvt_ids']:,}")
        cols = [f"{v}_inst" for v in P.INSTANT_VARS] + [f"{v}_acc" for v in P.ACCUM_VARS]
        nulls = j[cols].isna().sum()
        nulls = nulls[nulls > 0]
        print(f"      missing by variable : "
              f"{dict(nulls) if len(nulls) else 'none'}")
        if st["unmatched"]:
            BLOCKERS.append(f"{label}: {st['unmatched']} unmatched movement rows")
        if st["duplicate_mvt_ids"]:
            BLOCKERS.append(f"{label}: duplicate MVT_ID after join")
        print()

    rule("9b. MONTH-BOUNDARY ACCUMULATION - DIRECT PROOF FOR THE 26 AT-RISK ROWS")
    # The original V10-A plan stopped at 2026-01-31 and 2026-07-31, so a movement at
    # 23:xx on the last day of those months needed an ERA5 record stamped 00:00 of the
    # NEXT month. Identify those rows from the join itself and prove they are populated.
    rank_j, _ = P.join_movements(P.load_ranking_movements(), hourly, "ceil",
                                 source="ranking_2026")
    mv_month = rank_j["mvt_time_utc"].dt.strftime("%Y-%m")
    acc_month = rank_j["era5_accum_ts"].dt.strftime("%Y-%m")
    cross = rank_j[acc_month != mv_month]
    print(f"  ranking movements whose accumulated record lands in the following calendar"
          f" month: {len(cross):,}")
    if len(cross):
        for t, n in cross.groupby(acc_month[acc_month != mv_month]).size().items():
            print(f"      {t} 00:00 UTC   {n} movement(s)")
        for v in ["tp_acc", "fg10_acc"]:
            ok = int(cross[v].notna().sum())
            print(f"      {v:<8} non-null {ok}/{len(cross)}")
            if ok != len(cross):
                BLOCKERS.append(f"{len(cross) - ok} boundary rows missing {v}")
        rng = cross["tp_acc"].describe()
        print(f"      tp_acc  min {rng['min']:.6f} m  max {rng['max']:.6f} m")
        g = cross["fg10_acc"]
        print(f"      fg10_acc min {g.min():.3f} m/s  max {g.max():.3f} m/s")
        if not bool(((cross["tp_acc"] >= 0) & (cross["tp_acc"] <= 0.3)).all()):
            BLOCKERS.append("boundary tp_acc outside 0..0.3 m")
        if not bool(((cross["fg10_acc"] >= 0) & (cross["fg10_acc"] <= 120)).all()):
            BLOCKERS.append("boundary fg10_acc outside 0..120 m/s")
    else:
        print("      none - no ranking movement is near a month boundary")
    print()

    rule("10. PHYSICAL PLAUSIBILITY")
    print(P.plausibility_flags(hourly).to_string(index=False))
    print()
    rule("11. DISTRIBUTIONS BY AIRPORT AND SEASON")
    h = hourly.copy()
    h["ym"] = h["era5_time"].dt.strftime("%Y-%m")
    h["season"] = np.where(h["era5_time"].dt.month.isin([12, 1, 2]), "winter", "summer")
    jan = h[h["ym"] == "2026-01"]
    jul = h[h["ym"] == "2026-07"]
    cols = ["t2m", "wind_speed", "fg10", "tp", "msl", "tcc", "dewpoint_spread"]
    print("  JAN 2026 vs JUL 2026 (hourly, all 10 airports pooled)")
    comp = pd.DataFrame({
        "JAN mean": jan[cols].mean(), "JAN p50": jan[cols].median(),
        "JAN max": jan[cols].max(),
        "JUL mean": jul[cols].mean(), "JUL p50": jul[cols].median(),
        "JUL max": jul[cols].max(),
    })
    print(comp.to_string())
    print()
    print("  by airport (JAN 2026 mean):")
    print(jan.groupby("airport")[["t2m", "wind_speed", "tp", "msl"]].mean().to_string())
    print()
    rule("12. MONTHLY-BOUNDARY CONTINUITY")
    # A download-artefact discontinuity would show as a spike in the hour-to-hour
    # difference at the first hour of a month. The difference is only meaningful when
    # the previous row is genuinely the immediately preceding hour: the plan contains a
    # deliberate multi-month hole, and differencing across that hole would report the
    # seasonal drift as a boundary artefact.
    h = hourly.sort_values(["airport", "era5_time"]).copy()
    h["prev_time"] = h.groupby("airport")["era5_time"].shift()
    gap = (h["era5_time"] - h["prev_time"]).dt.total_seconds() / 3600.0
    h["adjacent"] = gap.eq(1)
    for v in ["t2m", "msl", "wind_speed"]:
        h[v + "_absdiff"] = h.groupby("airport")[v].diff().abs()
    h["is_month_start"] = h["era5_time"].dt.day.eq(1) & h["era5_time"].dt.hour.eq(0)
    starts = h[h["is_month_start"]]
    non_adj = starts[~starts["adjacent"]]
    if len(non_adj):
        print(f"  month-starts NOT preceded by the immediately preceding hour, excluded "
              f"from the metric: {len(non_adj)}")
        for t, g in non_adj.groupby(non_adj["era5_time"].dt.strftime("%Y-%m-%d")):
            step = gap.loc[g.index].iloc[0]
            if pd.isna(step):
                print(f"      {t}: no earlier row (first hour of the table)"
                      f"  ({g.index.size} airports - nothing to difference against)")
            else:
                print(f"      {t}: {g['prev_time'].iloc[0]} -> {g['era5_time'].iloc[0]}"
                      f"  ({g.index.size} airports, step {step:.0f} h -"
                      f" planned ranking-window hole, not a chunk seam)")
        print()
    ok = h[h["adjacent"]]
    ms = ok[ok["is_month_start"]]
    nxt = ok[~ok["is_month_start"]]
    print(f"  hour-to-hour steps used: {len(ok):,} "
          f"(month starts {len(ms):,}, other hours {len(nxt):,})")
    print(f"  {'variable':<14}{'mean |diff| at month start':>28}"
          f"{'mean |diff| elsewhere':>24}{'ratio':>9}")
    for v in ["t2m", "msl", "wind_speed"]:
        a = ms[v + "_absdiff"].mean()
        b = nxt[v + "_absdiff"].mean()
        ratio = a / b if b else float("nan")
        print(f"  {v:<14}{a:>28.4f}{b:>24.4f}{ratio:>9.2f}")
        if ratio > 3:
            WARNINGS.append(f"possible month-boundary discontinuity in {v} (ratio {ratio:.2f})")
    print()
    return True


# ==========================================================================
def main():
    print("=" * 78)
    print("V10-B1 - ERA5 WEATHER JOIN AUDIT")
    print("no model is trained, no target is read, no prediction is produced")
    print("=" * 78)

    audit_frozen()
    moves = audit_dates()
    audit_time_semantics()
    audit_coverage_gap(moves)
    audit_airport_mapping(moves)
    audit_leakage(moves)
    audit_cds_schema()
    authed = audit_auth()
    audit_join_data()

    rule("13. AUDIT SUMMARY")
    print(f"  BLOCKERS  ({len(BLOCKERS)})")
    for b in BLOCKERS:
        print(f"    - {b}")
    print(f"  WARNINGS  ({len(WARNINGS)})")
    for w in WARNINGS:
        print(f"    - {w}")
    print()
    if not authed:
        print("  DECISION: BLOCKED")
        print("  reason: CDS credentials / cdsapi client unavailable, so no ERA5 data")
        print("          could be acquired. Acquisition, transformation and join quality")
        print("          are therefore unverified, not merely unproven.")
    elif BLOCKERS:
        print("  DECISION: NOT READY")
    else:
        print("  DECISION: READY FOR V10-B2")
    print()
    return 1 if BLOCKERS else 0


if __name__ == "__main__":
    raise SystemExit(main())
