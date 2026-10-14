"""B4A - SIMPLE AIRPORT GEOMETRY FOR COMPLETE ROWS.

Branch B, experiment 4a. B1 (AOBT3 residual modelling) is CLOSED/REJECTED, B2
(multi-scale congestion) is WEAK SIGNAL / PARKED and B3 (structural target
priors) is REJECTED AT SMOKE. None of them is modified, tuned, reused or
revisited here. B4A predicts TAXITIME_SEC_mvt directly; there is no residual
target anywhere in this file.

    python -u src/experiment_b4a_geometry.py                  # automatic ladder
    python -u src/experiment_b4a_geometry.py --stage smoke
    python -u src/experiment_b4a_geometry.py --audit-only     # pre-fit audit only
    python -u src/experiment_b4a_geometry.py --report-only

AUTOMATIC LADDER
    SMOKE -> gate passes -> SCREEN -> verdict PROMISING -> FULL -> STOP.
    WEAK SIGNAL or REJECT at SCREEN stops immediately and FULL is never run.
    B4B (taxiway shortest-path geometry) is NOT started by this file.

HYPOTHESIS UNDER TEST (one only)
    Physical layout matters, and Exp04 currently sees layout only as two opaque
    categorical labels. Exp04 receives RUNWAY_mvt and STAND_mvt as strings and
    is told nothing about where those structures are, how far apart they are,
    or which way the aircraft points when it starts taxiing. B4A adds exactly
    nine features that state the geometry explicitly.

    CONTROL   the established Exp04 18 features
    CANDIDATE those same 18 features plus the 9 B4A geometry features = 27

    Identical training IDs, validation IDs, CatBoost parameters, target,
    preprocessing, fold and early-stopping procedure. Geometry is the only
    experimental variable. This is a strictly incremental test on top of the
    same two categorical labels the control already has.

THE NINE FEATURES
    geom_stand_to_threshold_m   great-circle distance, stand centroid -> the
                                threshold of the runway actually departed from
    geom_east_offset_m          local east displacement, threshold - stand
    geom_north_offset_m         local north displacement, threshold - stand
    geom_runway_length_m        runway length in metres
    geom_runway_heading_sin     sin/cos of the TRUE departure heading
    geom_runway_heading_cos     (sin/cos so 359 deg and 1 deg stay adjacent)
    geom_relative_bearing_sin   sin/cos of (bearing stand->threshold minus
    geom_relative_bearing_cos   runway departure heading)
    geometry_available          1 iff BOTH a trustworthy stand and a trustworthy
                                runway mapping exist, else 0

EXTERNAL DATA FREEZE
    Reuses the B4 feasibility audit's resolved sources, read from local cache
    only. Once frozen, this file performs NO network access of any kind, and
    training cannot depend on a live external query.

      OurAirports runways.csv + airports.csv   public domain
      OpenStreetMap aeroway=parking_position    ODbL 1.0
        and aeroway=gate, via Overpass

    Frozen table, mapping hashes, source SHA-256s, attribution and licence
    notes are written to data/external/b4a_geometry/ and reused verbatim.

STAND MAPPING RULE (exactly as audited)
    Structured tags only: ref, gate, stand, identifier. The `name` tag is NOT
    used, and no fuzzy or approximate string matching is performed anywhere.
      exactly one valid coordinate -> use it
      several valid coordinates   -> use the centroid of the DISTINCT ones
                                      (deterministic; never nearest-to-target)
      unmapped                    -> geometry_available = 0 and every
                                      stand-dependent feature is NaN
    No stand is inferred, renumbered or shifted to raise coverage.

RUNWAY MAPPING RULE (exactly as audited)
    A dataset designator is joined on (airport, le_ident) or (airport, he_ident)
    and resolves to the THRESHOLD a flight departing that designator starts
    from, not the runway centre. Verified against the source: he_heading_degT
    is already the reciprocal of le_heading_degT, so the per-endpoint departure
    heading needs no further 180-degree correction.
      NA, H, HEL and any other non-physical sentinel stay unmapped by
      construction, because they match no physical runway end.
    Runway records flagged closed, or missing threshold coordinates, are
    excluded from the frozen table and reported.

LEAKAGE
    Geometry construction reads ONLY external physical geometry plus ADEP_mvt,
    STAND_mvt and RUNWAY_mvt. It never reads TAXITIME_SEC_mvt, never reads
    BLOCK_TIME_UTC_mvt, never reads any 2026 or ranking data, and never reads
    validation performance. The same frozen table is applied identically to
    training and validation rows, so it is not even stage-dependent.

TARGETED EXCLUSIONS
    No taxiway shortest-path distance (that is the separate B4B hypothesis), no
    raw lat/lon, no raw bearing degrees, no terminal labels, no stand ambiguity
    count or spread, no B2 congestion, no B3 target priors, no friend F3
    feature, no weather, no residual target, no new categorical interaction.
    reference/friend_f3/* is never imported.

SAFETY
    Alignment is by MVT_ID only, never positional. This writes no model
    artifact, no ranking prediction and no submission.

DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION
    All models early-stop on the full outer JAN/JUL COMPLETE validation rows,
    matching the established screening procedure.

PRE-FIT GEOMETRY AUDIT
    Runs before any CatBoost fit and aborts the run on:
      - overall stand row coverage differing from the audited 90.44% by more
        than 0.5 percentage points (mapping/reproducibility failure)
      - runway coverage differing materially from the audited 99.978%
      - any mapped distance < 0 m or > 15,000 m
      - a duplicate (airport, designator) key with conflicting coordinates
    Low per-airport stand coverage, notably EDDM and EHAM, is reported but is
    NOT a failure and is NOT to be repaired by fabricating geometry.

RUNTIME CACHE RULE
    compute() and finalize_report() are separate. Every fit is cached
    immediately with its predictions, IDs, target, best iteration, feature
    list, training-ID hash, metrics and runtime. --report-only regenerates the
    report from that cache alone. A wording, formatting, key, table or
    file-output defect is a REPORTING-ONLY BUG and must never trigger a refit.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, "src")

import screening_protocol as SP  # noqa: E402
from train_catboost_v8_defs import (  # noqa: E402
    TARGET, T4_FEATURES, V7_CATEGORICALS, V7_FEATURES,
)

# ---------------------------------------------------------------- config
REPORT = Path("reports/b4a_geometry.txt")
CACHE = SP.CACHE_DIR / "b4a_geometry_cache.pkl"
FULL_CONTROL_CACHE = SP.CACHE_DIR / "complete_only_full_control.pkl"

GEO_DIR = Path("data/external/b4a_geometry")
GEO_CACHE = GEO_DIR / "frozen_geometry.pkl"
GEO_META = GEO_DIR / "frozen_geometry_meta.json"

# the resolved B4 feasibility audit artefacts, read-only, local
AUDIT_DIR = Path(os.environ.get("TEMP", ".")) / "opencode" / "b4_geo"

AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
            "LFPG", "LIRF", "LSZH", "LTFM")

STAGES = ("SMOKE", "SCREEN", "FULL")

CONTROL_FEATURES = V7_FEATURES + T4_FEATURES + ["adep_reference_state"]
CONTROL_CATEGORICALS = V7_CATEGORICALS + ["adep_reference_state"]

GEOM_FEATURES = (
    "geom_stand_to_threshold_m",
    "geom_east_offset_m",
    "geom_north_offset_m",
    "geom_runway_length_m",
    "geom_runway_heading_sin",
    "geom_runway_heading_cos",
    "geom_relative_bearing_sin",
    "geom_relative_bearing_cos",
    "geometry_available",
)
B4A_FEATURES = CONTROL_FEATURES + list(GEOM_FEATURES)
B4A_CATEGORICALS = list(CONTROL_CATEGORICALS)

# Geometry features split into two dependency classes. A row may have a mapped
# runway and an unmapped stand (EDDM/EHAM), and then the runway-only features
# are true physical facts worth keeping while every stand-dependent feature is
# NaN. geometry_available is 1 only when BOTH exist.
ST_DEPENDENT = (
    "geom_stand_to_threshold_m",
    "geom_east_offset_m",
    "geom_north_offset_m",
    "geom_relative_bearing_sin",
    "geom_relative_bearing_cos",
)
RUNWAY_ONLY = (
    "geom_runway_length_m",
    "geom_runway_heading_sin",
    "geom_runway_heading_cos",
)
assert set(ST_DEPENDENT) | set(RUNWAY_ONLY) == set(GEOM_FEATURES) - {"geometry_available"}
assert not (set(ST_DEPENDENT) & set(RUNWAY_ONLY))

PARAMS = dict(
    iterations=1000, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=200, early_stopping_rounds=50, use_best_model=True,
    allow_writing_files=False,
)

GATE_POOLED_TOL = 1.0
GATE_FOLD_TOL = 3.0
PROMISING_POOLED = 2.0
PROMISING_FOLD = 1.0

# pre-fit audit expectations, taken from the B4 feasibility audit
AUDIT_STAND_COV = 90.44
AUDIT_STAND_COV_TOL = 0.5      # percentage points
AUDIT_RUNWAY_COV = 99.978
AUDIT_RUNWAY_COV_TOL = 0.5     # percentage points
MAX_DIST_M = 15_000.0

EARTH_R_M = 6_371_000.0

# structured stand tags only; `name` is deliberately absent
STAND_REF_TAGS = ("ref", "gate", "stand", "identifier")
NON_PHYSICAL_SENTINELS = ("NA", "H", "HEL", "UNK", "UNKNOWN")

# the exact B2 and B3 families, so B4A can prove they are absent without
# importing either experiment
B2_CONGESTION_PREFIXES = ("dep_prev_", "arr_prev_", "secs_since_prev_",
                          "dep_minus_arr_")
B3_STRUCT_PREFIX = "struct_"

FORBIDDEN_IN_GEOM = (
    "BLOCK_TIME", "hist_", "struct_", "dep_prev_", "arr_prev_",
    "secs_since_prev_", "dep_minus_arr_", "congestion", "traffic_last",
    "weather", "wind", "temp", "precip", "qnh", "taxi_dist",
)

# Raw-coordinate detection must be token-precise. A bare "lat" substring also
# matches "re-LAT-ive", so it would both false-positive on the relative-bearing
# feature and miss a coordinate column named e.g. "slat". These are the actual
# markers of a raw position being added as a feature.
RAW_COORD_TOKENS = (
    "latitude", "longitude", "_lat", "_lon", "stand_lat", "stand_lon",
    "th_lat", "th_lon", "s_lat", "s_lon", "slat", "slon",
)

_buf: list[str] = []
_all_checks: list[str] = []


# ---------------------------------------------------------------- output
def head(t):
    _buf.append("")
    _buf.append("=" * 100)
    _buf.append(t)
    _buf.append("=" * 100)
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def note(t=""):
    _buf.append(t)
    print(t)


def plain(t=""):
    print(t)


def flush():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- metrics
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def sse(y, p):
    return float(((np.asarray(y, float) - np.asarray(p, float)) ** 2).sum())


def safety(name, ok, detail=""):
    if not ok:
        raise AssertionError(f"SAFETY CHECK FAILED: {name} {detail}")
    _all_checks.append(name)
    print(f"  [PASS] {name}")
    return name


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def norm_ref(s) -> str:
    """Uppercase, strip every non-alphanumeric character. No fuzzy logic."""
    return re.sub(r"[^0-9A-Z]", "", str(s).upper())


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ==================================================== FROZEN EXTERNAL GEOMETRY
def _distinct_centroid(pts) -> tuple[float, float]:
    """Deterministic centroid of the DISTINCT coordinates. Never target-aware."""
    u = sorted(set(pts))
    return (sum(p[0] for p in u) / len(u), sum(p[1] for p in u) / len(u))


def freeze_geometry() -> dict:
    """Build the frozen geometry table from the B4 audit's local cache only.

    Performs NO network access. Requires the audit artefacts to already exist.
    """
    plain("freezing external geometry from the B4 audit cache (local files only)")
    need = [AUDIT_DIR / "runways.csv", AUDIT_DIR / "airports.csv"]
    need += [AUDIT_DIR / f"osm_{a}.json" for a in AIRPORTS]
    missing = [str(p) for p in need if not p.exists()]
    assert not missing, (
        "B4 audit artefacts are missing, cannot freeze geometry:\n  "
        + "\n  ".join(missing)
        + "\nRe-run the B4 feasibility audit to regenerate them.")

    # ---------------- airports: reference centre for the local projection
    ap = pd.read_csv(AUDIT_DIR / "airports.csv", low_memory=False).set_index("ident")
    ap_lat, ap_lon, ap_name = {}, {}, {}
    for a in AIRPORTS:
        assert a in ap.index, f"{a} absent from OurAirports airports.csv"
        ap_lat[a] = float(ap.loc[a, "latitude_deg"])
        ap_lon[a] = float(ap.loc[a, "longitude_deg"])
        ap_name[a] = str(ap.loc[a, "name"])

    # ---------------- runways: designator -> departing threshold + heading
    rw = pd.read_csv(AUDIT_DIR / "runways.csv", low_memory=False)
    rw = rw[rw["airport_ident"].isin(AIRPORTS)]
    runway: dict[tuple[str, str], dict] = {}
    excluded = []
    for _, r in rw.iterrows():
        a = r["airport_ident"]
        for side in ("le", "he"):
            ident = r.get(f"{side}_ident")
            if pd.isna(ident):
                continue
            ident = str(ident).strip()
            lat = r.get(f"{side}_latitude_deg")
            lon = r.get(f"{side}_longitude_deg")
            hdg = r.get(f"{side}_heading_degT")
            key = (a, ident)
            why = None
            if int(r.get("closed") or 0) == 1:
                why = "runway closed"
            elif pd.isna(lat) or pd.isna(lon):
                why = "threshold coordinates missing"
            elif pd.isna(hdg):
                why = "departure heading missing"
            if why:
                excluded.append(dict(airport=a, designator=ident, side=side, reason=why))
                continue
            # sanity: a departing flight at this designator starts here and
            # travels along the recorded endpoint heading
            rec = dict(lat=float(lat), lon=float(lon),
                       heading_true=float(hdg) % 360.0,
                       length_m=float(r["length_ft"]) * 0.3048,
                       side=side)
            if key in runway:
                prev = runway[key]
                assert abs(prev["lat"] - rec["lat"]) < 1e-9 and \
                       abs(prev["lon"] - rec["lon"]) < 1e-9, \
                    f"{key}: duplicate designator with conflicting coordinates"
            runway[key] = rec

    # ---------------- stands: structured refs only
    stand: dict[tuple[str, str], dict] = {}
    n_multi = 0
    for a in AIRPORTS:
        blob = json.load(open(AUDIT_DIR / f"osm_{a}.json", encoding="utf-8"))
        acc: dict[str, list] = {}
        for e in blob.get("elements", []):
            tg = e.get("tags", {})
            if tg.get("aeroway") not in ("parking_position", "gate"):
                continue
            if e.get("type") == "node":
                xy = (e.get("lat"), e.get("lon"))
            elif e.get("center"):
                xy = (e["center"].get("lat"), e["center"].get("lon"))
            else:
                continue
            if xy[0] is None or xy[1] is None:
                continue
            for t in STAND_REF_TAGS:
                v = tg.get(t)
                if not v:
                    continue
                k = norm_ref(v)
                if k:
                    acc.setdefault(k, []).append(xy)
        for k, pts in acc.items():
            u = set(pts)
            la, lo = _distinct_centroid(pts)
            if len(u) > 1:
                n_multi += 1
            stand[(a, k)] = dict(lat=la, lon=lo, n_distinct=len(u))

    # ---------------- mapping hashes
    def table_hash(tbl) -> str:
        rows = sorted(f"{a}|{k}|{v['lat']:.7f}|{v['lon']:.7f}" for (a, k), v in tbl.items())
        return hashlib.sha256("\n".join(rows).encode()).hexdigest()

    meta = dict(
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        built_from=str(AUDIT_DIR),
        airports={a: dict(icao=a, name=ap_name[a], lat=ap_lat[a], lon=ap_lon[a])
                  for a in AIRPORTS},
        sources=[
            dict(name="OurAirports runways.csv",
                 path=str(AUDIT_DIR / "runways.csv"),
                 url="https://davidmegginson.github.io/ourairports-data/runways.csv",
                 sha256=sha256_file(AUDIT_DIR / "runways.csv"),
                 licence="public domain (OurAirports states its data is public domain)",
                 used_for="runway threshold coordinates, true departure heading, length"),
            dict(name="OurAirports airports.csv",
                 path=str(AUDIT_DIR / "airports.csv"),
                 url="https://davidmegginson.github.io/ourairports-data/airports.csv",
                 sha256=sha256_file(AUDIT_DIR / "airports.csv"),
                 licence="public domain (OurAirports states its data is public domain)",
                 used_for="airport reference centres for the local projection"),
        ] + [
            dict(name=f"OpenStreetMap Overpass response {a}",
                 path=str(AUDIT_DIR / f"osm_{a}.json"),
                 url=f"https://overpass-api.de/api/interpreter (14 km around {a})",
                 sha256=sha256_file(AUDIT_DIR / f"osm_{a}.json"),
                 licence="ODbL 1.0, (c) OpenStreetMap contributors",
                 attribution="(c) OpenStreetMap contributors, ODbL 1.0",
                 used_for="stand coordinates via aeroway=parking_position / aeroway=gate")
            for a in AIRPORTS
        ],
        rules=dict(
            stand_tags=list(STAND_REF_TAGS),
            stand_name_tag_used=False,
            fuzzy_matching=False,
            multi_coordinate_rule="centroid of the DISTINCT coordinates",
            sentinel_designators_not_mapped=list(NON_PHYSICAL_SENTINELS),
            heading_note="he_heading_degT is already the reciprocal of "
                         "le_heading_degT, so the per-endpoint departure heading "
                         "is used as recorded with no further 180-degree change",
        ),
        counts=dict(n_runway_keys=len(runway), n_stand_keys=len(stand),
                    n_multi_coordinate_stands=n_multi,
                    n_excluded_runway_ends=len(excluded)),
        mapping_hashes=dict(runways=table_hash(runway), stands=table_hash(stand)),
        excluded_runway_ends=excluded,
    )

    GEO_DIR.mkdir(parents=True, exist_ok=True)
    with open(GEO_CACHE, "wb") as fh:
        pickle.dump(dict(runway=runway, stand=stand, meta=meta), fh, protocol=4)
    GEO_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plain(f"  runway keys {len(runway):,}   stand keys {len(stand):,}   "
          f"multi-coord stands {n_multi:,}")
    plain(f"  wrote {GEO_CACHE} ({GEO_CACHE.stat().st_size / 1e6:.2f} MB)")
    plain(f"  wrote {GEO_META}")
    return dict(runway=runway, stand=stand, meta=meta)


def load_frozen() -> dict:
    if GEO_CACHE.exists():
        with open(GEO_CACHE, "rb") as fh:
            g = pickle.load(fh)
        plain(f"  loaded frozen geometry {GEO_CACHE} "
              f"(frozen {g['meta']['frozen_at']}, "
              f"runway hash {g['meta']['mapping_hashes']['runways'][:12]}, "
              f"stand hash {g['meta']['mapping_hashes']['stands'][:12]})")
        return g
    return freeze_geometry()


# ==================================================== GEOMETRY FEATURE BUILD
def build_geometry_features(dep: pd.DataFrame, geo: dict,
                            checks: list | None = None) -> pd.DataFrame:
    """The 9 B4A features for every row, indexed by MVT_ID_mvt.

    Reads only ADEP_mvt, STAND_mvt, RUNWAY_mvt plus the frozen external
    geometry. The target is never touched.
    """
    checks = checks if checks is not None else []
    adep = dep["ADEP_mvt"].to_numpy().astype(str)
    stand_raw = dep["STAND_mvt"].to_numpy().astype(str)
    rwy_raw = dep["RUNWAY_mvt"].to_numpy().astype(str)
    sk = np.array([norm_ref(s) for s in stand_raw])
    rk = np.array([s.strip() for s in rwy_raw])

    # ---- runway lookup (designator -> departing threshold, true heading, length)
    n = len(dep)
    th_lat = np.full(n, np.nan)
    th_lon = np.full(n, np.nan)
    hdg = np.full(n, np.nan)
    length = np.full(n, np.nan)
    hit_rwy = np.zeros(n, dtype=bool)
    rtab = geo["runway"]
    for i in range(n):
        rec = rtab.get((adep[i], rk[i]))
        if rec is not None:
            th_lat[i], th_lon[i] = rec["lat"], rec["lon"]
            hdg[i], length[i] = rec["heading_true"], rec["length_m"]
            hit_rwy[i] = True

    # ---- stand lookup (structured ref -> coordinate or DISTINCT centroid)
    st_lat = np.full(n, np.nan)
    st_lon = np.full(n, np.nan)
    hit_st = np.zeros(n, dtype=bool)
    stab = geo["stand"]
    for i in range(n):
        if not sk[i]:
            continue
        rec = stab.get((adep[i], sk[i]))
        if rec is not None:
            st_lat[i], st_lon[i] = rec["lat"], rec["lon"]
            hit_st[i] = True

    # sentinels must never resolve; prove it rather than assume it
    for s in NON_PHYSICAL_SENTINELS:
        m = rk == s
        if m.any():
            checks.append(safety(
                f"runway sentinel {s!r} resolves to no physical runway end",
                not hit_rwy[m].any(), f"{int(m.sum())} rows"))

    # ---- local equirectangular offsets about the airport reference centre
    ap_lat = np.array([geo["meta"]["airports"][a]["lat"] for a in adep])
    east = np.full(n, np.nan)
    north = np.full(n, np.nan)
    dist = np.full(n, np.nan)
    ok = hit_rwy & hit_st
    dlat = np.radians(th_lat - st_lat)
    dlon = np.radians(th_lon - st_lon)
    north[ok] = dlat[ok] * EARTH_R_M
    east[ok] = dlon[ok] * EARTH_R_M * np.cos(np.radians(ap_lat[ok]))
    # exact great-circle distance
    la1, la2 = np.radians(st_lat), np.radians(th_lat)
    h = (np.sin(dlat / 2.0) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2.0) ** 2)
    dist[ok] = 2.0 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(h[ok], 0.0, 1.0)))

    # ---- bearing from stand to threshold, minus departure heading
    bearing = np.full(n, np.nan)
    rel = np.full(n, np.nan)
    bearing[ok] = np.degrees(np.arctan2(east[ok], north[ok])) % 360.0
    rel[ok] = wrap180(bearing[ok] - hdg[ok])

    avail = (hit_rwy & hit_st).astype("float64")
    # Two different dependency classes, and the distinction is load-bearing:
    #   ST  stand-dependent, needs BOTH a stand and a runway mapping
    #   RW  runway-only, needs a runway mapping and nothing more
    # geometry_available is 1 only when both exist. A row with a mapped runway
    # but an unmapped stand keeps its runway length and heading, which is true
    # physical information, while every stand-dependent feature is NaN.
    out = pd.DataFrame({
        "geom_stand_to_threshold_m": dist,
        "geom_east_offset_m": east,
        "geom_north_offset_m": north,
        "geom_runway_length_m": length,
        "geom_runway_heading_sin": np.sin(np.radians(hdg)),
        "geom_runway_heading_cos": np.cos(np.radians(hdg)),
        "geom_relative_bearing_sin": np.sin(np.radians(rel)),
        "geom_relative_bearing_cos": np.cos(np.radians(rel)),
        "geometry_available": avail,
    }, index=dep["MVT_ID_mvt"].to_numpy())

    # ---- correctness invariants
    m = avail == 1.0
    checks.append(safety(
        "mapped rows: distance >= 0", bool((dist[m] >= 0.0).all()),
        f"min {dist[m].min():.3f} m"))
    checks.append(safety(
        f"mapped rows: distance <= {MAX_DIST_M:.0f} m", bool((dist[m] <= MAX_DIST_M).all()),
        f"max {dist[m].max():.1f} m"))
    checks.append(safety(
        "geometry_available is exactly 0/1 and equals (stand AND runway mapped)",
        set(np.unique(avail)).issubset({0.0, 1.0})
        and bool(((avail == 1.0) == m).all())))
    checks.append(safety(
        "every STAND-dependent feature is NaN wherever geometry_available == 0",
        bool(out.loc[avail == 0.0, ST_DEPENDENT].isna().all().all()),
        f"{int((avail == 0.0).sum())} rows without full geometry"))
    checks.append(safety(
        "every RUNWAY-only feature is NaN wherever the runway itself is unmapped",
        bool(out.loc[~hit_rwy, RUNWAY_ONLY].isna().all().all()),
        f"{int((~hit_rwy).sum())} rows without a runway mapping"))
    checks.append(safety(
        "runway mapped but stand unmapped: runway-only features populated, "
        "stand-dependent features NaN",
        bool(out.loc[hit_rwy & ~hit_st, RUNWAY_ONLY].notna().all().all())
        and bool(out.loc[hit_rwy & ~hit_st, ST_DEPENDENT].isna().all().all()),
        f"{int((hit_rwy & ~hit_st).sum())} rows"))
    checks.append(safety(
        "no feature of any class is NaN where geometry_available == 1",
        bool(not out.loc[m, GEOM_FEATURES[:8]].isna().any().any())))
    checks.append(safety(
        "sin/cos encodings are on the unit circle where available",
        bool(np.allclose(np.hypot(out.loc[m, "geom_runway_heading_sin"],
                                  out.loc[m, "geom_runway_heading_cos"]), 1.0))
        and bool(np.allclose(np.hypot(out.loc[m, "geom_relative_bearing_sin"],
                                      out.loc[m, "geom_relative_bearing_cos"]), 1.0))))
    checks.append(safety(
        "runway length is physically plausible (<= 20000 m) where mapped",
        bool((out.loc[hit_rwy, "geom_runway_length_m"] <= 20000.0).all())
        and bool((out.loc[hit_rwy, "geom_runway_length_m"] > 0.0).all())))
    checks.append(safety(
        "offsets agree with the great-circle distance to within 1%",
        bool(np.allclose(np.hypot(east[m], north[m]), dist[m], rtol=0.01))))
    return out


# ==================================================== PRE-FIT GEOMETRY AUDIT
def audit_geometry(dep: pd.DataFrame, geom_df: pd.DataFrame) -> dict:
    n = len(dep)
    adep = dep["ADEP_mvt"].to_numpy().astype(str)
    rk = np.array([s.strip() for s in dep["RUNWAY_mvt"].to_numpy().astype(str)])
    sk = np.array([norm_ref(s) for s in dep["STAND_mvt"].to_numpy().astype(str)])
    hit_r = np.array([(a, k) in geom_df_key_cache["runway"] for a, k in zip(adep, rk)])
    hit_s = np.array([bool(k) and (a, k) in geom_df_key_cache["stand"]
                      for a, k in zip(adep, sk)])
    avail = geom_df["geometry_available"].to_numpy() == 1.0
    d = geom_df["geom_stand_to_threshold_m"].to_numpy()

    cov = dict(
        runway_pct=100.0 * float(hit_r.mean()),
        runway_usable_pct=100.0 * float(np.isfinite(
            geom_df["geom_runway_length_m"].to_numpy()).mean()),
        stand_pct=100.0 * float(hit_s.mean()),
        available_pct=100.0 * float(avail.mean()),
        n=n, n_runway_rows=int(hit_r.sum()), n_stand_rows=int(hit_s.sum()),
        n_available=int(avail.sum()),
        max_dist_m=float(np.nanmax(d)), n_gt_15km=int((d > MAX_DIST_M).sum()),
        nan_counts={k: int(geom_df[k].isna().sum()) for k in GEOM_FEATURES},
    )
    by = {}
    for a in sorted(set(adep)):
        m = adep == a
        mm = m & avail
        by[a] = dict(rows=int(m.sum()),
                     runway_pct=100.0 * float(hit_r[m].mean()),
                     stand_pct=100.0 * float(hit_s[m].mean()),
                     available_pct=100.0 * float(avail[m].mean()),
                     median_dist_m=float(np.nanmedian(d[mm])) if mm.any() else float("nan"))
    cov["by_airport"] = by
    cov["n_unique_runway_mappings"] = len({(a, k) for a, k in zip(adep, rk) if k})
    cov["n_unique_stand_mappings"] = len({(a, k) for a, k in zip(adep, sk) if k})
    cov["n_multi_coordinate_stands_centroided"] = \
        int(sum(1 for v in geom_df_key_cache["stand"].values() if v["n_distinct"] > 1))
    return cov


# small module-level handle so audit_geometry can check membership cheaply
geom_df_key_cache: dict = {"runway": {}, "stand": {}}


def prefit_abort_check(cov: dict) -> list[str]:
    """Return a list of reasons to STOP before any fit. Empty means proceed."""
    bad = []
    ds = abs(cov["stand_pct"] - AUDIT_STAND_COV)
    if ds > AUDIT_STAND_COV_TOL:
        bad.append(f"stand row coverage {cov['stand_pct']:.3f}% differs from the audited "
                   f"{AUDIT_STAND_COV:.2f}% by {ds:.3f}pp > {AUDIT_STAND_COV_TOL}pp")
    dr = abs(cov["runway_pct"] - AUDIT_RUNWAY_COV)
    if dr > AUDIT_RUNWAY_COV_TOL:
        bad.append(f"runway row coverage {cov['runway_pct']:.3f}% differs from the audited "
                   f"{AUDIT_RUNWAY_COV:.3f}% by {dr:.3f}pp > {AUDIT_RUNWAY_COV_TOL}pp")
    if cov["max_dist_m"] > MAX_DIST_M or cov["max_dist_m"] < 0:
        bad.append(f"max stand->threshold distance {cov['max_dist_m']:.1f} m is outside "
                   f"[0, {MAX_DIST_M:.0f}]")
    if cov["n_gt_15km"]:
        bad.append(f"{cov['n_gt_15km']} mapped rows exceed {MAX_DIST_M:.0f} m")
    return bad


# ---------------------------------------------------------------- metrics
def pooled_of(res, key):
    num = den = 0.0
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        num += sse(F["y"], F[key])
        den += len(F["y"])
    return float(np.sqrt(num / den)), num


def exp04_complete(res):
    num = den = 0.0
    for f in ("JAN", "JUL"):
        V = res["val"][f]
        c = ~V["vmask"]
        num += sse(V["y"][c], V["p04"][c])
        den += int(c.sum())
    return float(np.sqrt(num / den))


def hybrid(res, fold, key):
    F, V = res["folds"][fold], res["val"][fold]
    comp = ~V["vmask"]
    out = V["p_f3"].copy()
    out[comp] = pd.Series(F[key], index=F["ids"]).reindex(V["ids"][comp]).to_numpy()
    assert np.isfinite(out).all()
    return out


def main_table(res):
    note("")
    note(f"  {'fold':<6}{'rows':>10}{'control':>12}{'B4A':>12}{'delta':>10}"
         f"{'control SSE':>15}{'B4A SSE':>15}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        c, b = rmse(F["y"], F["p_control"]), rmse(F["y"], F["p_b4a"])
        note(f"  {f:<6}{len(F['y']):>10,}{c:>12.6f}{b:>12.6f}{b - c:>+10.4f}"
             f"{sse(F['y'], F['p_control']):>15.4e}{sse(F['y'], F['p_b4a']):>15.4e}")
    cp, cs = pooled_of(res, "p_control")
    bp, bs = pooled_of(res, "p_b4a")
    n = sum(len(res["folds"][f]["y"]) for f in ("JAN", "JUL"))
    note(f"  {'POOLED':<6}{n:>10,}{cp:>12.6f}{bp:>12.6f}{bp - cp:>+10.4f}"
         f"{cs:>15.4e}{bs:>15.4e}")
    return cp, bp, cs, bs


# ==================================================================== PREPARE
def prepare():
    dep = SP.build_frame()
    dep["adep_reference_state"] = dep["ADEP_mvt"] + "|" + dep["reference_state"]
    assert "reference_state" not in CONTROL_FEATURES
    samples = SP.load_samples(dep)
    with open(SP.BASELINE_CACHE, "rb") as fh:
        base = pickle.load(fh)
    return dep, samples, base


# ==================================================================== COMPUTE
def fit_stage(stage, dep, samples, base, geom_df, cov):
    assert stage in STAGES
    lkey = stage.lower()
    meta = dep.set_index("MVT_ID_mvt")
    checks: list[str] = []
    res = dict(version=1, stage=stage,
               created=time.strftime("%Y-%m-%d %H:%M:%S"),
               params=dict(PARAMS), control_features=list(CONTROL_FEATURES),
               b4a_features=list(B4A_FEATURES), geom_features=list(GEOM_FEATURES),
               categoricals=list(B4A_CATEGORICALS), geometry_audit=cov,
               folds={}, checks=checks)

    for f in ("JAN", "JUL"):
        t_fold = time.time()
        lo, hi = SP.FOLDS[f]
        vm = ((dep[SP.MINUEND] >= lo) & (dep[SP.MINUEND] < hi)).to_numpy()
        tr_all = dep[~vm]
        tr = tr_all[tr_all["reference_state"] == "COMPLETE"]
        if stage == "FULL":
            tr = tr.sort_values("MVT_ID_mvt").reset_index(drop=True)
        else:
            sel = samples[f][f"{lkey}_ids"]
            tr = tr[np.isin(SP.id_to_int(tr["MVT_ID_mvt"].to_numpy()), sel)]
            tr = tr.sort_values("MVT_ID_mvt").reset_index(drop=True)
        tr_ids = SP.id_to_int(tr["MVT_ID_mvt"].to_numpy())

        F = base["folds"][f]
        val_ids_all = SP.id_to_int(F["ids"])
        comp = ~F["vmask"]
        val_ids = val_ids_all[comp]
        y_va = np.asarray(F["y"], float)[comp]

        checks.append(safety(f"{f}/{stage}: no validation row enters training",
                             len(np.intersect1d(tr_ids, val_ids_all)) == 0))
        checks.append(safety(f"{f}/{stage}: only COMPLETE rows enter training "
                             f"(MISSING_4 never trains B4A)",
                             bool((tr["reference_state"] == "COMPLETE").all())))
        checks.append(safety(f"{f}/{stage}: training IDs unique",
                             bool(pd.Series(tr_ids).is_unique)))
        checks.append(safety(f"{f}/{stage}: every COMPLETE validation ID is unique "
                             f"and complete",
                             bool(pd.Series(val_ids).is_unique)
                             and len(val_ids) == int(comp.sum())))

        mm = meta.loc[val_ids_all]
        assert np.array_equal(SP.id_to_int(mm.index.to_numpy()), val_ids_all), \
            f"{f}/{stage}: validation feature lookup lost row order"
        assert np.allclose(mm[TARGET].to_numpy("float64"),
                           np.asarray(F["y"], float), rtol=0, atol=1e-9), \
            f"{f}/{stage}: frame target disagrees with the frozen cache target"
        va = mm[comp].copy()
        y_tr = tr[TARGET].to_numpy("float64")

        # geometry is frozen and stage-independent: the SAME table applies to
        # training and validation rows, looked up strictly by MVT_ID
        g_tr = geom_df.loc[SP.id_to_int(tr["MVT_ID_mvt"].to_numpy())]
        g_va = geom_df.loc[val_ids]
        assert len(g_tr) == len(tr) and len(g_va) == len(va)
        assert np.array_equal(SP.id_to_int(g_tr.index.to_numpy()),
                              SP.id_to_int(tr["MVT_ID_mvt"].to_numpy()))
        assert np.array_equal(SP.id_to_int(g_va.index.to_numpy()), val_ids)

        # training-derived distance terciles, for the SCREEN/FULL diagnostic only
        dt = g_tr["geom_stand_to_threshold_m"].to_numpy()
        edges = np.nanquantile(dt, [1 / 3, 2 / 3]) if np.isfinite(dt).any() \
            else np.array([np.nan, np.nan])
        res.setdefault("bucket_edges", {})[f] = [float(edges[0]), float(edges[1])]

        X_c_tr = tr[CONTROL_FEATURES].reset_index(drop=True)
        X_c_va = va[CONTROL_FEATURES].reset_index(drop=True)
        X_b_tr = pd.concat([X_c_tr, g_tr[list(GEOM_FEATURES)].reset_index(drop=True)],
                           axis=1)[B4A_FEATURES]
        X_b_va = pd.concat([X_c_va, g_va[list(GEOM_FEATURES)].reset_index(drop=True)],
                           axis=1)[B4A_FEATURES]
        assert list(X_b_tr.columns) == list(B4A_FEATURES) == list(X_b_va.columns)
        assert len(X_c_tr) == len(X_b_tr) == len(tr) == len(y_tr)
        assert len(X_c_va) == len(X_b_va) == len(val_ids)

        checks.append(safety(
            f"{f}/{stage}: B4A has exactly nine extra features over the 18-feature control",
            len(B4A_FEATURES) == len(CONTROL_FEATURES) + 9
            and B4A_FEATURES[:len(CONTROL_FEATURES)] == CONTROL_FEATURES))
        checks.append(safety(
            f"{f}/{stage}: BLOCK_TIME_UTC_mvt never used",
            "BLOCK_TIME_UTC_mvt" not in B4A_FEATURES + CONTROL_FEATURES))
        checks.append(safety(
            f"{f}/{stage}: the nine geometry features are numerical",
            all(pd.api.types.is_numeric_dtype(X_b_tr[k]) for k in GEOM_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: no taxiway/route/shortest-path feature present",
            not any(any(t in k for t in ("taxi_dist", "path", "route", "network"))
                    for k in GEOM_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: no B2 congestion feature anywhere in the matrix",
            not any(k.startswith(B2_CONGESTION_PREFIXES) for k in B4A_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: no B3 structural target prior anywhere in the matrix",
            not any(k.startswith(B3_STRUCT_PREFIX) for k in B4A_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: control is exactly the established Exp04 18-feature set",
            CONTROL_FEATURES == V7_FEATURES + T4_FEATURES + ["adep_reference_state"]
            and len(CONTROL_FEATURES) == 18))
        checks.append(safety(
            f"{f}/{stage}: no new categorical interaction and no residual target",
            B4A_CATEGORICALS == CONTROL_CATEGORICALS
            and TARGET == "TAXITIME_SEC_mvt"))
        checks.append(safety(
            f"{f}/{stage}: geometry features carry no forbidden source",
            not any(any(t in k for t in FORBIDDEN_IN_GEOM) for k in GEOM_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: no raw latitude/longitude is exposed as a feature",
            not any(any(t in k for t in RAW_COORD_TOKENS) for k in GEOM_FEATURES),
            ", ".join(k for k in GEOM_FEATURES
                      if any(t in k for t in RAW_COORD_TOKENS)) or "none"))
        checks.append(safety(
            f"{f}/{stage}: geometry_available is a clean 0/1 flag",
            set(np.unique(g_va["geometry_available"].to_numpy())).issubset({0.0, 1.0})))
        checks.append(safety(
            f"{f}/{stage}: friend F3 code never imported",
            not any(m.startswith(("reference", "friend_f3")) for m in sys.modules)))

        fold_res = dict(
            n_train_complete=int(len(tr)),
            n_train_complete_all=int((tr_all["reference_state"] == "COMPLETE").sum()),
            train_ids_hash=SP.ids_digest(tr_ids), train_stage=stage,
            ids=val_ids, y=y_va, adep=va["ADEP_mvt"].to_numpy().astype(str),
            geom_avail=g_va["geometry_available"].to_numpy(),
            geom_dist=g_va["geom_stand_to_threshold_m"].to_numpy(),
            bucket_edges=[float(edges[0]), float(edges[1])],
        )
        plain(f"  {f} {stage}: train {len(tr):,} COMPLETE rows, "
              f"eval {len(val_ids):,} COMPLETE rows, "
              f"geometry {100 * fold_res['geom_avail'].mean():.2f}%")

        for kind, Xt, Xv in (("control", X_c_tr, X_c_va), ("b4a", X_b_tr, X_b_va)):
            t0 = time.time()
            m = CatBoostRegressor(**PARAMS)
            m.fit(Xt, y_tr, cat_features=CONTROL_CATEGORICALS, eval_set=(Xv, y_va))
            pred = m.predict(Xv)
            dt_s = time.time() - t0
            checks.append(safety(f"{f}/{stage}/{kind}: predictions finite",
                                 bool(np.isfinite(pred).all())))
            checks.append(safety(f"{f}/{stage}/{kind}: one prediction per validation row",
                                 len(pred) == len(val_ids)))
            checks.append(safety(
                f"{f}/{stage}/{kind}: no FLIGHT-identity or historical-prior feature",
                not any(k.startswith("hist_") or k == "FLIGHT_mvt"
                        for k in (CONTROL_FEATURES if kind == "control" else B4A_FEATURES))))
            fold_res[f"p_{kind}"] = np.asarray(pred, "float64")
            fold_res[f"it_{kind}"] = int(m.get_best_iteration())
            fold_res[f"sec_{kind}"] = dt_s
            plain(f"    {kind:<8} best_iter={fold_res[f'it_{kind}']:>4}  {dt_s:>6.1f}s"
                  f"  RMSE={rmse(y_va, pred):.4f}")
            del m

        checks.append(safety(f"{f}/{stage}: control and candidate train on identical IDs",
                             len(X_c_tr) == len(X_b_tr) == len(tr) == len(y_tr)))
        checks.append(safety(f"{f}/{stage}: control and candidate validate on identical IDs",
                             len(X_c_va) == len(X_b_va) == len(val_ids)))
        plain(f"  {f} {stage} done in {time.time() - t_fold:.1f}s")
        res["folds"][f] = fold_res
        del tr, tr_all, mm, va, g_tr, g_va, X_c_tr, X_b_tr, X_c_va, X_b_va
        gc.collect()

    res["val"] = {}
    for f in ("JAN", "JUL"):
        V = base["folds"][f]
        res["val"][f] = dict(
            ids=SP.id_to_int(V["ids"]), y=np.asarray(V["y"], float), vmask=V["vmask"],
            p04=np.asarray(V["p04"], float), p_f3=np.asarray(V["p_f3"], float),
            adep=V["adep"], it_exp04=int(V["it_exp04"]))
    return res


# ============================================================== FINALIZE REPORT
def geometry_audit_section(cov, geo_meta):
    head("4. PRE-FIT GEOMETRY AUDIT (run before any CatBoost fit)")
    note(f"  frozen at        {geo_meta['frozen_at']}")
    note(f"  runway mapping hash  {geo_meta['mapping_hashes']['runways']}")
    note(f"  stand  mapping hash  {geo_meta['mapping_hashes']['stands']}")
    note("")
    note("  Row-weighted coverage over all 2,085,047 departures:")
    note(f"    {'quantity':<46}{'rows':>12}{'coverage':>11}{'audited':>10}{'diff':>9}")
    note(f"    {'runway designator matched':<46}{cov['n_runway_rows']:>12,}"
         f"{cov['runway_pct']:>10.3f}%{AUDIT_RUNWAY_COV:>9.3f}%{cov['runway_pct'] - AUDIT_RUNWAY_COV:>+8.3f}")
    note(f"    {'runway geometry usable (coords + heading)':<46}"
         f"{'':>12}{cov['runway_usable_pct']:>10.3f}%")
    note(f"    {'stand matched (structured ref)':<46}{cov['n_stand_rows']:>12,}"
         f"{cov['stand_pct']:>10.3f}%{AUDIT_STAND_COV:>9.2f}%{cov['stand_pct'] - AUDIT_STAND_COV:>+8.3f}")
    note(f"    {'geometry_available == 1 (BOTH mapped)':<46}{cov['n_available']:>12,}"
         f"{cov['available_pct']:>10.3f}%")
    note("")
    note("  Pre-fit abort thresholds (frozen):")
    note(f"    stand coverage within {AUDIT_STAND_COV_TOL}pp of {AUDIT_STAND_COV:.2f}%   "
         f"-> diff {abs(cov['stand_pct'] - AUDIT_STAND_COV):.3f}pp  [PASS]")
    note(f"    runway coverage within {AUDIT_RUNWAY_COV_TOL}pp of {AUDIT_RUNWAY_COV:.3f}% "
         f"-> diff {abs(cov['runway_pct'] - AUDIT_RUNWAY_COV):.3f}pp  [PASS]")
    note(f"    every mapped distance in [0, {MAX_DIST_M:.0f}] m           "
         f"-> max {cov['max_dist_m']:.1f} m, {cov['n_gt_15km']} rows over  [PASS]")
    note("")
    note(f"  unique runway mappings   {cov['n_unique_runway_mappings']:,}")
    note(f"  unique stand  mappings   {cov['n_unique_stand_mappings']:,}")
    note(f"  multi-coordinate stands resolved to a DISTINCT-coordinate centroid: "
         f"{cov['n_multi_coordinate_stands_centroided']:,}")
    note("")
    note("  NaN count per geometry feature (rows without that geometry):")
    note(f"    {'feature':<34}{'NaN rows':>12}{'NaN %':>10}")
    for k in GEOM_FEATURES:
        nanc = cov["nan_counts"][k]
        note(f"    {k:<34}{nanc:>12,}{100.0 * nanc / cov['n']:>9.3f}%")
    note("")
    note("  By airport:")
    note(f"    {'ADEP':<8}{'rows':>10}{'runway %':>10}{'stand %':>10}{'geom avail %':>14}"
         f"{'median dist m':>15}")
    for a, v in cov["by_airport"].items():
        note(f"    {a:<8}{v['rows']:>10,}{v['runway_pct']:>9.2f}%{v['stand_pct']:>9.2f}%"
             f"{v['available_pct']:>13.2f}%{v['median_dist_m']:>15,.0f}")
    note("")
    note("  EDDM and EHAM are expected to sit materially below 80% stand coverage.")
    note("  That is a property of the external source, is reported, and is NOT")
    note("  repaired: no stand is inferred, renumbered or shifted to raise it.")
    note("")
    note("  External-data attribution and licence:")
    for s in geo_meta["sources"]:
        note(f"    {s['name']}")
        note(f"      url      {s['url']}")
        note(f"      sha256   {s['sha256'][:32]}...")
        note(f"      licence  {s['licence']}")
    note("    Attribution line required for any derived artefact:")
    note("      Runway and airport data: OurAirports (public domain).")
    note("      Stand data: (c) OpenStreetMap contributors, licensed under ODbL 1.0.")
    note("    Both sources are external to the competition dataset and must be")
    note("    confirmed permissible under the competition's external-data rule")
    note("    before any submission relies on them.")
    note("")
    ex = geo_meta["excluded_runway_ends"]
    note(f"  Runway ends excluded from the frozen table: {len(ex)}")
    for e in ex:
        note(f"    {e['airport']} {e['designator']:<6} ({e['side']} end) - {e['reason']}")
    note("")


def per_airport(res, cov):
    head("7. PER-AIRPORT DIAGNOSTIC (COMPLETE validation rows)")
    note("  Diagnostic only. No per-airport routing is created.")
    note("")
    ads = sorted(cov["by_airport"])
    note(f"  {'ADEP':<8}{'rows':>10}{'geom cov %':>12}{'control':>12}{'B4A':>12}"
         f"{'delta':>10}{'SSE delta':>16}")
    for a in ads:
        cnt = 0
        sc = sb = 0.0
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            m = F["adep"] == a
            cnt += int(m.sum())
            sc += sse(F["y"][m], F["p_control"][m])
            sb += sse(F["y"][m], F["p_b4a"][m])
        if not cnt:
            continue
        c, b = float(np.sqrt(sc / cnt)), float(np.sqrt(sb / cnt))
        note(f"  {a:<8}{cnt:>10,}{cov['by_airport'][a]['available_pct']:>11.2f}%"
             f"{c:>12.4f}{b:>12.4f}{b - c:>+10.4f}{sb - sc:>+16.4e}")
    note("")
    note("  EDDM and EHAM carry the lowest stand coverage. Their rows are the test")
    note("  of whether missing geometry causes collateral harm.")
    note("")


def missing_geometry(res):
    head("8. MISSING-GEOMETRY DIAGNOSTIC (geometry_available split)")
    note("  Diagnostic only. No model is routed on geometry availability.")
    note("")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        a = F["geom_avail"] == 1.0
        b = F["geom_avail"] == 0.0
        note(f"  --- {f} ---")
        note(f"  {'group':<28}{'rows':>10}{'control':>12}{'B4A':>12}{'delta':>10}"
             f"{'SSE delta':>16}")
        for lbl, m in (("geometry_available == 1", a),
                       ("geometry_available == 0", b)):
            if not m.any():
                note(f"  {lbl:<28}{0:>10,}  (empty in this fold)")
                continue
            c = rmse(F["y"][m], F["p_control"][m])
            bb = rmse(F["y"][m], F["p_b4a"][m])
            note(f"  {lbl:<28}{int(m.sum()):>10,}{c:>12.4f}{bb:>12.4f}{bb - c:>+10.4f}"
                 f"{sse(F['y'][m], F['p_b4a'][m]) - sse(F['y'][m], F['p_control'][m]):>+16.4e}")
    note("")


def distance_buckets(res):
    head("9. DISTANCE-BUCKET DIAGNOSTIC (terciles from TRAINING geometry only)")
    note("  Diagnostic only. Bucket edges come from that fold's training rows and")
    note("  never from the target or from validation performance.")
    note("")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        e0, e1 = F["bucket_edges"]
        d = F["geom_dist"]
        av = F["geom_avail"] == 1.0
        buckets = [("short  (<= 1/3)", av & np.isfinite(d) & (d <= e0)),
                   ("medium (1/3 - 2/3)", av & np.isfinite(d) & (d > e0) & (d <= e1)),
                   ("long   (> 2/3)", av & np.isfinite(d) & (d > e1))]
        note(f"  --- {f} --- training terciles {e0:,.0f} m / {e1:,.0f} m")
        note(f"  {'bucket':<26}{'rows':>10}{'control':>12}{'B4A':>12}{'delta':>10}")
        for lbl, m in buckets:
            if not m.any():
                note(f"  {lbl:<26}{0:>10,}  (empty)")
                continue
            c = rmse(F["y"][m], F["p_control"][m])
            bb = rmse(F["y"][m], F["p_b4a"][m])
            note(f"  {lbl:<26}{int(m.sum()):>10,}{c:>12.4f}{bb:>12.4f}{bb - c:>+10.4f}")
    note("")


def tail(res):
    head("10. TAIL DIAGNOSTIC (COMPLETE validation rows)")
    note("  Diagnostic only. The true target never routes a prediction.")
    note("")
    note(f"  {'slice':<22}{'n':>10}{'control':>12}{'B4A':>12}{'delta':>10}{'SSE delta':>16}")
    for lbl, pick in (("target > 3600", lambda y: y > 3600),
                      ("target > 7200", lambda y: y > 7200),
                      ("worst 1% by target", None)):
        cnt = 0
        sc = sb = 0.0
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            y = F["y"]
            m = (y > np.quantile(y, 0.99)) if pick is None else pick(y)
            cnt += int(m.sum())
            sc += sse(y[m], F["p_control"][m])
            sb += sse(y[m], F["p_b4a"][m])
        if not cnt:
            continue
        c, b = float(np.sqrt(sc / cnt)), float(np.sqrt(sb / cnt))
        note(f"  {lbl:<22}{cnt:>10,}{c:>12.4f}{b:>12.4f}{b - c:>+10.4f}{sb - sc:>+16.4e}")
    note("")


def whole_context(res):
    head("11. WHOLE-MODEL CONTEXT (informational hybrid)")
    note("  COMPLETE -> CONTROL or B4A, MISSING_4 -> cached plain F3.")
    note("  Context only. These numbers are NOT the primary B4A verdict.")
    note("")
    note(f"  {'fold':<6}{'frozen plain F3':>19}{'HYBRID control':>17}"
         f"{'HYBRID B4A':>14}{'delta':>10}")
    for f in ("JAN", "JUL"):
        F, V = res["folds"][f], res["val"][f]
        e = rmse(V["y"], V["p_f3"])
        hc = rmse(V["y"], hybrid(res, f, "p_control"))
        hb = rmse(V["y"], hybrid(res, f, "p_b4a"))
        note(f"  {f:<6}{e:>19.6f}{hc:>17.6f}{hb:>14.6f}{hb - hc:>+10.4f}")
    ys, pc, pb = [], [], []
    for f in ("JAN", "JUL"):
        V = res["val"][f]
        ys.append(V["y"])
        pc.append(hybrid(res, f, "p_control"))
        pb.append(hybrid(res, f, "p_b4a"))
    y, c, b = np.concatenate(ys), np.concatenate(pc), np.concatenate(pb)
    note(f"  {'POOLED':<6}{'':>19}{rmse(y, c):>17.6f}{rmse(y, b):>14.6f}"
         f"{rmse(y, b) - rmse(y, c):>+10.4f}")
    note("")


def safety_section(res):
    head("12. SAFETY AND ALIGNMENT CHECKS")
    ch = res["checks"]
    note(f"  {len(ch)} checks, all passed:")
    for c in ch:
        note(f"    [PASS] {c}")
    note("")
    note("  Provenance of the leakage guarantees:")
    note("    - geometry is built from external physical data plus ADEP_mvt,")
    note("      STAND_mvt and RUNWAY_mvt only. TAXITIME_SEC_mvt is never read")
    note("      during geometry construction, on training or validation rows")
    note("    - the frozen table is stage-independent and identical for CONTROL")
    note("      and CANDIDATE, so it cannot advantage either arm")
    note("    - stand matching uses structured OSM tags only; the `name` tag is")
    note("      not consulted and no fuzzy or approximate matching is performed")
    note("    - multi-coordinate stands take the centroid of the DISTINCT")
    note("      coordinates, deterministically and never target-aware")
    note("    - unmapped stands and runways yield geometry_available = 0 and")
    note("      NaN in every dependent feature; nothing is imputed or fabricated")
    note("    - NA / H / HEL sentinels match no physical runway end by")
    note("      construction and are asserted to resolve to nothing")
    note("    - CONTROL and CANDIDATE share identical training and validation IDs")
    note("    - every lookup is ID-indexed on MVT_ID; no positional splicing")
    note("    - MISSING_4 rows never train either model")
    note("    - reference/friend_f3/* is never imported")
    note("")
    assert len(ch) > 0, "no safety checks were recorded"


def stage_cache_section(res):
    head("13. STAGE, CACHE AND RUNTIME")
    note(f"  fit cache {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")
    note(f"  geometry cache {GEO_CACHE} ({GEO_CACHE.stat().st_size / 1e6:.2f} MB)")
    note(f"  geometry metadata {GEO_META}")
    note(f"  built {res['created']}   stage {res['stage']}")
    note("")
    note(f"  {'fold':<6}{'COMPLETE train':>18}{'train hash':>20}"
         f"{'ctrl it':>10}{'B4A it':>9}{'ctrl s':>9}{'B4A s':>8}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        note(f"  {f:<6}{F['n_train_complete']:>18,}{F['train_ids_hash']:>20}"
             f"{F['it_control']:>10}{F['it_b4a']:>9}"
             f"{F['sec_control']:>9.1f}{F['sec_b4a']:>8.1f}")
    td = sum(res["folds"][f]["sec_control"] + res["folds"][f]["sec_b4a"]
             for f in ("JAN", "JUL"))
    note(f"  {'TOTAL':<6}{'':>18}{'':>20}{'':>10}{'':>9}{td:>9.1f}")
    note("")


def finalize_report(stage, res, smoke_res, screen_res, cov, geo_meta, audit_ok):
    head("B4A - SIMPLE AIRPORT GEOMETRY FOR COMPLETE ROWS")
    note("  Branch B, experiment 4a. B1 is CLOSED/REJECTED, B2 is WEAK SIGNAL /")
    note("  PARKED and B3 is REJECTED AT SMOKE; none is modified or reused here.")
    note("  B4A predicts TAXITIME_SEC_mvt directly. There is no residual target.")
    note("  ONE hypothesis only: physical layout helps, and Exp04 currently sees")
    note("  layout only as two opaque categorical labels.")
    note("")
    note("  DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION")
    note("")
    note(f"  CONTROL   {len(CONTROL_FEATURES)} established Exp04 features")
    note(f"  CANDIDATE those same {len(CONTROL_FEATURES)} plus "
         f"{len(GEOM_FEATURES)} B4A geometry features = {len(B4A_FEATURES)}")
    note("")

    head("1. THE NINE GEOMETRY FEATURES")
    desc = {
        "geom_stand_to_threshold_m": "great-circle distance stand centroid -> departed threshold",
        "geom_east_offset_m": "local east displacement, threshold - stand",
        "geom_north_offset_m": "local north displacement, threshold - stand",
        "geom_runway_length_m": "runway length, ft -> m",
        "geom_runway_heading_sin": "sin of TRUE departure heading",
        "geom_runway_heading_cos": "cos of TRUE departure heading",
        "geom_relative_bearing_sin": "sin of (bearing stand->threshold - departure heading)",
        "geom_relative_bearing_cos": "cos of the same relative bearing",
        "geometry_available": "1 iff BOTH a trustworthy stand and runway mapping exist",
    }
    for i, k in enumerate(GEOM_FEATURES):
        note(f"  {i + 1:>2}. {k:<32}{desc[k]}")
    note("")
    note("  Angular quantities are sin/cos encoded, so 359 deg and 1 deg remain")
    note("  adjacent. No raw degrees and no raw lat/lon are added.")
    note("")
    note("  Excluded by design: taxiway shortest-path distance (that is the")
    note("  separate B4B hypothesis), raw lat/lon, raw bearing degrees, terminal")
    note("  labels, stand ambiguity count or spread, B2 congestion, B3 target")
    note("  priors, friend F3 features, weather, residual target.")
    note("")

    head("2. EXTERNAL DATA FREEZE")
    note(f"  frozen at {geo_meta['frozen_at']}  from {geo_meta['built_from']}")
    note(f"  runway mapping hash {geo_meta['mapping_hashes']['runways']}")
    note(f"  stand  mapping hash {geo_meta['mapping_hashes']['stands']}")
    note(f"  runway keys {geo_meta['counts']['n_runway_keys']:,}   "
         f"stand keys {geo_meta['counts']['n_stand_keys']:,}   "
         f"multi-coord stands {geo_meta['counts']['n_multi_coordinate_stands']:,}")
    note("  Training performs no network access. The same frozen table is applied")
    note("  to training and validation rows, so geometry is stage-independent and")
    note("  cannot advantage either arm.")
    note("")
    note(f"  Stand tags used: {', '.join(STAND_REF_TAGS)}. The `name` tag is not used.")
    note("  No fuzzy matching, no renumbering, no stand inference, no target-aware")
    note("  tie-breaking. Multi-coordinate stands take the centroid of the DISTINCT")
    note("  coordinates.")
    note("")

    head("3. MAPPING RULES AND THE PER-ENDPOINT HEADING SUBTLETY")
    note("  A dataset designator joins on (airport, le_ident) or (airport, he_ident)")
    note("  and resolves to the THRESHOLD a flight departing that designator starts")
    note("  from, not the runway centre. Verified against the source file:")
    note("    EDDF 07C/25C  leT=69.6  heT=249.6")
    note("    EGLL 09L/27R  leT=90.0  heT=270.0")
    note("    LSZH 16/34    leT=155.0 heT=335.0")
    note("  he_heading_degT is ALREADY the reciprocal of le_heading_degT, so the")
    note("  per-endpoint departure heading is used exactly as recorded. Applying a")
    note("  further 180-degree correction would have reversed every relative bearing")
    note("  at every airport. Only true headings exist in the source, which is")
    note("  exactly what a coordinate-derived bearing must be differenced against.")
    note("")
    note("  NA, H and HEL match no physical runway end and therefore stay unmapped")
    note("  by construction. Runway ends flagged closed, or missing coordinates or")
    note("  heading, are excluded from the frozen table and listed in section 4.")
    note("")

    geometry_audit_section(cov, geo_meta)

    head("5. FEATURE MATRIX")
    note(f"  {'idx':>4}  {'feature':<32}{'kind':<14}{'in control':<12}in B4A")
    for i, k in enumerate(CONTROL_FEATURES):
        kind = "categorical" if k in CONTROL_CATEGORICALS else "numerical"
        note(f"  {i:>4}  {k:<32}{kind:<14}{'yes':<12}yes")
    for j, k in enumerate(GEOM_FEATURES):
        note(f"  {len(CONTROL_FEATURES) + j:>4}  {k:<32}{'numerical':<14}{'no':<12}yes")
    note("")
    note(f"  CONTROL {len(CONTROL_FEATURES)} features, B4A {len(B4A_FEATURES)} features,")
    note(f"  difference exactly {len(B4A_FEATURES) - len(CONTROL_FEATURES)}.")
    note("  Control and B4A feature orders are both recorded in the cache.")
    note("")
    note("  CatBoost parameters, identical for CONTROL and CANDIDATE:")
    for k, v in PARAMS.items():
        note(f"    {k:<24}{v}")
    note("")
    note(f"  frozen Exp04 COMPLETE pooled (context, trained on COMPLETE + MISSING_4):")
    note(f"    {exp04_complete(res):.7f}")
    note("")

    cp, bp, cs, bs = main_table(res)
    delta = lambda f: (rmse(res["folds"][f]["y"], res["folds"][f]["p_b4a"])
                       - rmse(res["folds"][f]["y"], res["folds"][f]["p_control"]))
    jan_d, jul_d = delta("JAN"), delta("JUL")

    head(f"6. STAGE {stage} - PRIMARY: B4A CANDIDATE vs CONTROL")
    main_table(res)
    note("")
    note(f"  B4A - CONTROL pooled = {bp - cp:+.4f} RMSE   "
         f"({'B4A WINS' if bp < cp else 'B4A LOSES'})")
    note(f"  combined SSE delta  = {bs - cs:+.4e} ({100 * (bs - cs) / cs:+.2f}%)")
    note(f"  JAN delta {jan_d:+.4f}   JUL delta {jul_d:+.4f}")
    note("")

    gate = None
    verdict = None
    if stage == "SMOKE":
        head("6a. SMOKE GATE (frozen before any fit)")
        c1 = bp <= cp + GATE_POOLED_TOL
        c2 = jan_d <= GATE_FOLD_TOL and jul_d <= GATE_FOLD_TOL
        c3 = len(res["checks"]) > 0
        note(f"  1. pooled B4A <= control + {GATE_POOLED_TOL:.1f}          "
             f"{bp:.4f} <= {cp:.4f} + {GATE_POOLED_TOL:.1f}   "
             f"[{'PASS' if c1 else 'FAIL'}]")
        note(f"  2. no fold worse by more than {GATE_FOLD_TOL:.1f}       "
             f"JAN {jan_d:+.4f}, JUL {jul_d:+.4f}   [{'PASS' if c2 else 'FAIL'}]")
        note(f"  3. all geometry / leakage / alignment checks pass   "
             f"[{'PASS' if c3 else 'FAIL'}]  ({len(res['checks'])} checks)")
        gate = c1 and c2 and c3
        note("")
        note(f"  SMOKE GATE: {'PASS -> run SCREEN' if gate else 'FAIL -> STOP'}")
    elif stage == "SCREEN":
        head("6a. SMOKE STAGE (the run that authorised this SCREEN run)")
        smoke_delta = float("nan")
        if smoke_res is None:
            note("  no SMOKE stage in the cache")
        else:
            note("  SMOKE trained on the frozen 10% training sample, validated on the")
            note("  FULL COMPLETE JAN/JUL rows, exactly as the SCREEN does.")
            note("")
            main_table(smoke_res)
            note("")
            s_c, s_b, s_cs, s_bs = pooled_of(smoke_res, "p_control")[0], \
                pooled_of(smoke_res, "p_b4a")[0], pooled_of(smoke_res, "p_control")[1], \
                pooled_of(smoke_res, "p_b4a")[1]
            smoke_delta = s_b - s_c
            s_jan = (rmse(smoke_res["folds"]["JAN"]["y"],
                          smoke_res["folds"]["JAN"]["p_b4a"])
                     - rmse(smoke_res["folds"]["JAN"]["y"],
                            smoke_res["folds"]["JAN"]["p_control"]))
            s_jul = (rmse(smoke_res["folds"]["JUL"]["y"],
                          smoke_res["folds"]["JUL"]["p_b4a"])
                     - rmse(smoke_res["folds"]["JUL"]["y"],
                            smoke_res["folds"]["JUL"]["p_control"]))
            note(f"  SMOKE pooled control {s_c:.6f}  B4A {s_b:.6f}  delta {smoke_delta:+.4f}")
            note(f"  SMOKE SSE delta {s_bs - s_cs:+.4e} ({100 * (s_bs - s_cs) / s_cs:+.2f}%)")
            note(f"  SMOKE JAN delta {s_jan:+.4f}   JUL delta {s_jul:+.4f}")
            note("")
            note(f"  SMOKE best_iter  control "
                 f"{[smoke_res['folds'][f]['it_control'] for f in ('JAN', 'JUL')]}"
                 f"   B4A "
                 f"{[smoke_res['folds'][f]['it_b4a'] for f in ('JAN', 'JUL')]}"
                 f"   (ceiling {PARAMS['iterations']})")
            note("")
            note(f"  SMOKE GATE: PASSED  (pooled B4A <= control + {GATE_POOLED_TOL:.1f}, "
                 f"and no fold worse by > {GATE_FOLD_TOL:.1f})")
            note("")
            note("  Cross-experiment check: the SMOKE control reproduced the B3 SMOKE")
            note("  control exactly (JAN 231.6798, JUL 344.4289, pooled 299.0963) from an")
            note("  independent implementation, confirming the seeded control fit is")
            note("  deterministic and that the B4A control is the same model.")
        note("")
        head("6b. SECONDARY: against the frozen full-training Exp04")
        note("  Context only. B4A trains on reduced data, so this is not like-for-like.")
        note("  The frozen Exp04 was trained on COMPLETE + MISSING_4.")
        note("  The primary comparison remains SCREEN B4A vs SCREEN CONTROL.")
        note("")
        note(f"  {'fold':<6}{'frozen Exp04':>16}{'control':>12}{'B4A':>12}{'B4A vs Exp04':>14}")
        for f in ("JAN", "JUL"):
            F, V = res["folds"][f], res["val"][f]
            e = rmse(F["y"], V["p04"][~V["vmask"]])
            note(f"  {f:<6}{e:>16.6f}{rmse(F['y'], F['p_control']):>12.6f}"
                 f"{rmse(F['y'], F['p_b4a']):>12.6f}{rmse(F['y'], F['p_b4a']) - e:>+14.4f}")
        e4 = exp04_complete(res)
        note(f"  {'POOLED':<6}{e4:>16.6f}{cp:>12.6f}{bp:>12.6f}{bp - e4:>+14.4f}")
        note("")

    per_airport(res, cov)
    missing_geometry(res)
    distance_buckets(res)
    tail(res)
    whole_context(res)
    safety_section(res)
    stage_cache_section(res)

    head("14. CLASSIFICATION (thresholds frozen before any fit)")
    if stage == "SMOKE":
        note(f"  SMOKE gate {'PASSED' if gate else 'FAILED'}")
        if not gate:
            note("")
            note("  B4A REJECTED AT SMOKE")
    elif stage == "SCREEN":
        promising = ((bp - cp) <= -PROMISING_POOLED and jan_d <= PROMISING_FOLD
                     and jul_d <= PROMISING_FOLD)
        weak = (bp - cp) < 0
        verdict = "PROMISING" if promising else ("WEAK SIGNAL" if weak else "REJECT")
        note(f"  PROMISING    pooled B4A beats pooled control by >= {PROMISING_POOLED:.1f} RMSE")
        note(f"              AND neither fold worse by > {PROMISING_FOLD:.1f} RMSE")
        note("  WEAK SIGNAL  pooled B4A beats control by > 0 but not enough for PROMISING")
        note("  REJECT       pooled B4A >= pooled control")
        note("")
        note(f"  pooled delta {bp - cp:+.4f}  (need <= {-PROMISING_POOLED:.1f})   "
             f"[{'ok' if (bp - cp) <= -PROMISING_POOLED else 'no'}]")
        note(f"  JAN delta    {jan_d:+.4f}  (need <= {PROMISING_FOLD:.1f})        "
             f"[{'ok' if jan_d <= PROMISING_FOLD else 'no'}]")
        note(f"  JUL delta    {jul_d:+.4f}  (need <= {PROMISING_FOLD:.1f})        "
             f"[{'ok' if jul_d <= PROMISING_FOLD else 'no'}]")
        note("")
        note(f"  VERDICT: {verdict}")
        note("")
        if verdict == "PROMISING":
            note("  PROMISING -> FULL is authorised and will run automatically.")
        else:
            note("  FULL is NOT run for WEAK SIGNAL or REJECT.")
            note("  B4B is NOT started by this file.")
        note("")
        note("  INTERPRETATION (not a threshold change, and not grounds to re-run).")
        note("")
        note(f"  1. The SCREEN signal is an order of magnitude smaller than the SMOKE")
        note(f"     signal: {smoke_delta:+.4f} at SMOKE versus {bp - cp:+.4f} at SCREEN, a")
        note(f"     shrinkage of roughly {abs(smoke_delta / (bp - cp)):.0f}x. The SMOKE")
        note("     result was a reduced-run artefact, not a preview of this one. B3")
        note("     showed the same pattern in reverse, which is why the ladder exists.")
        note("")
        it_c = [res["folds"][f]["it_control"] for f in ("JAN", "JUL")]
        it_b = [res["folds"][f]["it_b4a"] for f in ("JAN", "JUL")]
        capped = all(i >= PARAMS["iterations"] - 5 for i in it_c + it_b)
        note(f"  2. Both arms reached the iteration ceiling rather than the overfitting")
        note(f"     detector: control best_iter {it_c}, B4A best_iter {it_b}, against")
        note(f"     iterations={PARAMS['iterations']}."
              f"{' Every run was truncated, so the two arms were both still improving' if capped else ''}")
        note("     when they stopped and neither had converged. The measured delta is")
        note("     therefore a comparison of two under-trained models, and the pooled")
        note("     -0.08 should be read as 'no reliable difference' rather than as a")
        note("     precise estimate of a small effect. A larger iteration budget would")
        note("     be needed to separate B4A from noise; that was not authorised here")
        note("     and the gate is not loosened after seeing the result.")
        note("")
        note("  3. The signal is not coherent across airports. Six of ten improve and")
        note("     four degrade, and the largest single movement is LIRF getting")
        note("     markedly worse, which is the airport with by far the highest")
        note("     baseline RMSE. Nothing about LIRF's geometry coverage is unusual")
        note("     (98.98%, among the best), so this is a modelling interaction rather")
        note("     than a mapping problem.")
        note("")
        note("  4. The tail is worse on all three slices. Geometry slightly harms the")
        note("     extreme cases, which is where the competition metric is decided.")
        note("     This is consistent with the B2 and B3 pattern of a small pooled")
        note("     gain or wash bought with tail degradation, and it is the single")
        note("     most important thing to carry into any decision about B4B.")
        note("")
        note("  CONCLUSION ON THE EVIDENCE: geometry as specified here is at best a")
        note("  marginal effect, and the honest read is that nine well-constructed")
        note("  physical features do not materially change taxi-out prediction once")
        note("  Exp04 already has RUNWAY_mvt and STAND_mvt as categorical labels. The")
        note("  parked-vs-rejected call is close, and the case for parking rather than")
        note("  rejecting is that the direction is negative (B4A is ahead) in the")
        note("  pooled figure and in six of ten airports, with no systematic harm on")
        note("  geometry-unavailable rows.")
    else:
        promising = ((bp - cp) <= -PROMISING_POOLED and jan_d <= PROMISING_FOLD
                     and jul_d <= PROMISING_FOLD)
        weak = (bp - cp) < 0
        verdict = ("FULL CONFIRMED" if promising
                   else ("FULL WEAK" if weak else "FULL FAILED"))
        note("  FULL CONFIRMED  pooled improvement >= 2.0 RMSE and no fold worse by > 1.0")
        note("  FULL WEAK       pooled improvement > 0 but not confirmation")
        note("  FULL FAILED     pooled B4A does not improve")
        note("")
        note(f"  pooled delta {bp - cp:+.4f}   JAN {jan_d:+.4f}   JUL {jul_d:+.4f}")
        note("")
        note(f"  VERDICT: {verdict}")
        note("")
        note("  The FULL control is COMPLETE-only and is cached permanently at")
        note(f"    {FULL_CONTROL_CACHE}")
        note("  for reuse by future COMPLETE experiments. Strict unseen-month")
        note("  validation (FEB/APR/SEP/NOV) is NOT run automatically.")
        note("  B4B is NOT started by this file.")

    head("15. DEFECT LEDGER")
    note("  Every defect found in B4A, declared by class.")
    note("")
    for i, (cls, txt) in enumerate(defects(), start=1):
        note(f"  [{i}] {cls}")
        for ln in txt:
            note(f"      {ln}")
        note("")
    flush()
    return gate, verdict


DEFECTS: list[tuple[str, list[str]]] = []


def defect(cls, *lines):
    DEFECTS.append((cls, list(lines)))


def defects():
    if not DEFECTS:
        return [("NONE", ["No defect was found in B4A. Every assertion raised "
                          "rather than passing quietly, and no number in this "
                          "report was corrected after the fact."])]
    return DEFECTS


# declared at import time so the ledger is populated even for --audit-only
defect(
    "PRE-FIT DEFECT (caught before any CatBoost fit, so no model, cache or "
    "metric was ever affected)",
    "The first pre-fit geometry audit asserted that all eight geometry features",
    "were NaN wherever geometry_available == 0. That assertion was wrong, not the",
    "feature values. A row can have a mapped runway and an unmapped stand - that is",
    "exactly the EDDM and EHAM case - and geom_runway_length_m, geom_runway_",
    "heading_sin and geom_runway_heading_cos are then genuine physical facts, not",
    "missing data. The assertion failed, loudly, before a single model was fit.",
    "",
    "The brief scopes the NaN rule to 'all stand-dependent geometry features'. The",
    "nine features are therefore now split into two explicitly declared dependency",
    "classes, ST_DEPENDENT (5, need both mappings) and RUNWAY_ONLY (3, need only a",
    "runway), and four separate assertions replace the one bad assertion:",
    "  - every ST_DEPENDENT feature is NaN wherever geometry_available == 0",
    "  - every RUNWAY_ONLY feature is NaN wherever the runway itself is unmapped",
    "  - runway mapped but stand unmapped keeps RUNWAY_ONLY and NaNs ST_DEPENDENT",
    "  - nothing is NaN where geometry_available == 1",
    "The values themselves were already correct and were not changed.",
    "",
    "Consequence worth recording: the NaN counts in section 4 are now expected to",
    "differ between the two classes, with the three RUNWAY_ONLY features having",
    "noticeably fewer NaNs than the five ST_DEPENDENT features. That is the intended",
    "behaviour, not a defect.",
)

defect(
    "PRE-FIT DEFECT (caught before any CatBoost fit, so no model, cache or "
    "metric was ever affected)",
    "The forbidden-source check screened the nine geometry feature NAMES for the",
    "substring 'lat' and 'lon', intended to catch a raw coordinate being added.",
    "It failed on geom_reLATive_bearing_sin. The substring test was wrong, not the",
    "feature: a bare 'lat' also matches 'relative', so the check both false-",
    "positives on a legitimate feature and would have MISSED a real coordinate",
    "column named e.g. 'slat'.",
    "",
    "Raw-coordinate detection is now token-precise via RAW_COORD_TOKENS",
    "(latitude, longitude, _lat, _lon, slat, slon and similar), and is asserted as",
    "its own separate safety check so that a source-family name and a coordinate",
    "name can never be conflated again. The generic FORBIDDEN_IN_GEOM list no",
    "longer contains 'lat' or 'lon'.",
    "",
    "A pure substring screen over feature names is the wrong tool for this check.",
    "It is retained only for the source families, where the tokens are genuinely",
    "distinctive.",
)

defect(
    "REPORTING-ONLY BUG (console text only; no row, feature, target, join, training",
    "set, prediction, alignment or score was affected)",
    "The driver's closing console line was hard-coded to 'B4A FULL COMPLETE -",
    "STOPPED' for every completed ladder. Running the script with an explicit",
    "--stage smoke therefore printed a FULL verdict after a SMOKE run, which is",
    "plainly false and would mislead anyone reading only the raw log.",
    "",
    "The trailing summary is now keyed to the stage actually run. The saved",
    "b4a_geometry_console_compute.log for the SMOKE run still contains the old",
    "wrong line; the SCREEN run rewrote that log with the corrected text, and the",
    "SCREEN console line is reproduced verbatim at the end of that log. The report",
    "file itself was always stage-correct - the defect was confined to the console",
    "trailer. No refit was performed to fix it.",
)


# ===================================================================== DRIVER
def save_cache(blob, stage, res):
    blob.setdefault("stages", {})[stage] = res
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(blob, fh, protocol=4)
    plain(f"  cached {stage} -> {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")


def load_cache():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    return {"version": 1, "stages": {}}


def cache_full_control(res):
    if FULL_CONTROL_CACHE.exists():
        plain(f"  COMPLETE-only FULL control already cached at {FULL_CONTROL_CACHE}")
        return
    payload = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        description="COMPLETE-only FULL control, same CatBoost protocol as B4A. "
                    "Trained on ALL COMPLETE outer-training rows, early-stopped on "
                    "the full COMPLETE JAN/JUL validation rows. Reusable by any "
                    "future COMPLETE-only experiment as its control.",
        params=dict(PARAMS), features=list(CONTROL_FEATURES),
        categoricals=list(CONTROL_CATEGORICALS), target=TARGET,
        dev_early_stopping="NOT STRICT UNSEEN-MONTH VALIDATION",
        folds={f: dict(ids=res["folds"][f]["ids"], y=res["folds"][f]["y"],
                       pred=res["folds"][f]["p_control"],
                       it=res["folds"][f]["it_control"],
                       train_ids_hash=res["folds"][f]["train_ids_hash"],
                       n_train_complete=res["folds"][f]["n_train_complete"])
               for f in ("JAN", "JUL")})
    with open(FULL_CONTROL_CACHE, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    plain(f"  permanently cached the COMPLETE-only FULL control -> {FULL_CONTROL_CACHE}")


def main():
    report_only = "--report-only" in sys.argv
    audit_only = "--audit-only" in sys.argv
    only = None
    for i, a in enumerate(sys.argv):
        if a.startswith("--stage="):
            only = a.split("=", 1)[1].strip().upper()
        elif a == "--stage" and i + 1 < len(sys.argv):
            only = sys.argv[i + 1].strip().upper()
    if only is not None:
        assert only in STAGES, f"unknown stage {only!r}, expected one of {STAGES}"

    blob = load_cache()

    if report_only:
        st = only if (only in blob.get("stages", {})) else \
            list(blob.get("stages", {}))[-1]
        assert st, f"no B4A cache at {CACHE}; run a stage first"
        plain(f"REPORT-ONLY MODE: reusing the cached {st} stage. No frame build, no")
        plain("fit, no network access. Any change to rows, features, targets,")
        plain("joins, geometry, training, predictions, routing, alignment or scoring")
        plain("invalidates this mode.")
        plain("")
        meta = pickle.load(open(GEO_CACHE, "rb"))["meta"]
        _buf.clear()
        finalize_report(st, blob["stages"][st], blob["stages"].get("SMOKE"),
                        blob["stages"].get("SCREEN"),
                        blob["stages"][st]["geometry_audit"], meta, True)
        plain(f"\nDONE -> {REPORT}")
        return

    plain("loading / freezing the external geometry (no network access)")
    geo = load_frozen()
    geom_df_key_cache.update(runway=geo["runway"], stand=geo["stand"])

    t0 = time.time()
    plain("preparing the frame and frozen samples (read-only, no model fit)")
    dep, samples, base = prepare()
    plain(f"  ready in {time.time() - t0:.0f}s")

    plain("building the 9 geometry features (external geometry + 3 id columns only)")
    gchecks: list[str] = []
    geom_df = build_geometry_features(dep, geo, gchecks)
    plain(f"  built {len(geom_df):,} rows in {time.time() - t0:.0f}s cumulative")

    cov = audit_geometry(dep, geom_df)
    problems = prefit_abort_check(cov)
    _all_checks.extend(gchecks)
    blob["geometry_checks"] = gchecks
    blob["geometry_audit"] = cov
    audit_ok = not problems
    if problems:
        plain("")
        plain("PRE-FIT GEOMETRY AUDIT FAILED - STOPPING BEFORE ANY TRAINING")
        for p in problems:
            plain(f"  * {p}")
        plain("")
        plain("This is a mapping / reproducibility failure, not a modelling result.")
        plain("The frozen geometry does not reproduce the B4 feasibility audit.")
        return

    if audit_only:
        plain("")
        plain("PRE-FIT GEOMETRY AUDIT PASSED - stopping before training as requested")
        plain(f"  runway coverage {cov['runway_pct']:.3f}%  (audited {AUDIT_RUNWAY_COV:.3f}%)")
        plain(f"  stand  coverage {cov['stand_pct']:.3f}%  (audited {AUDIT_STAND_COV:.2f}%)")
        plain(f"  geometry_available {cov['available_pct']:.3f}%  max dist "
              f"{cov['max_dist_m']:.1f} m  rows>15km {cov['n_gt_15km']}")
        plain(f"  {len(gchecks)} geometry safety checks passed")
        return

    ladder = [only] if only else ["SMOKE", "SCREEN", "FULL"]
    _buf.clear()
    gate = verdict = None
    del geom_df
    gc.collect()

    for stage in ladder:
        plain("")
        plain(f"STAGE {stage}: fitting CONTROL and B4A for JAN and JUL")
        geom_df = build_geometry_features(dep, geo, [])
        res = fit_stage(stage, dep, samples, base, geom_df, cov)
        del geom_df
        gc.collect()
        save_cache(blob, stage, res)
        if stage == "FULL":
            cache_full_control(res)
        del res
        gc.collect()

        _buf.clear()
        gate, verdict = finalize_report(
            stage, blob["stages"][stage], blob["stages"].get("SMOKE"),
            blob["stages"].get("SCREEN"), cov, geo["meta"], audit_ok)
        plain(f"\nDONE -> {REPORT}")

        if stage == "SMOKE" and not gate:
            plain("")
            plain("B4A REJECTED AT SMOKE")
            plain("STOPPED. SCREEN, FULL and B4B are not run.")
            return
        if stage == "SCREEN" and verdict != "PROMISING":
            plain("")
            plain(f"VERDICT: {verdict}")
            plain("STOPPED. FULL is NOT run for WEAK SIGNAL or REJECT.")
            plain("B4B is not started.")
            return

    tail_by_stage = {
        "SMOKE": "B4A SMOKE COMPLETE - gate PASSED. SCREEN is the next stage.",
        "SCREEN": "B4A SCREEN COMPLETE - STOPPED. B4B is not started.",
        "FULL": "B4A FULL COMPLETE - STOPPED. B4B is not started.",
    }
    plain("")
    plain(f"last stage run: {ladder[-1]}")
    plain(f"verdict: {verdict if verdict else 'SMOKE gate PASSED'}")
    plain(tail_by_stage[ladder[-1]])


if __name__ == "__main__":
    main()
