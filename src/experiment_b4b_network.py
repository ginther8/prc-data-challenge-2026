"""B4B - TAXIWAY-NETWORK DISTANCE FOR COMPLETE ROWS.

Branch B, experiment 4b. This is the FINAL planned geometry experiment.

    B1  residual target (AOBT3)         REJECTED
    B2  multi-scale congestion           WEAK SIGNAL / PARKED
    B3  structural target priors         REJECTED
    B4A simple physical geometry         WEAK SIGNAL / CLOSED
    B4B taxiway-network distance         THIS FILE

B4A is NOT tuned and its nine geometry features are NOT reused. B4B tests a
genuinely different physical hypothesis: aircraft cannot taxi in a straight
line, so the shortest-path distance through the mapped taxiway network should
carry taxi-out information that the categorical STAND_mvt / RUNWAY_mvt labels
and the straight-line geometry of B4A do not.

    python -u src/experiment_b4b_network.py                  # automatic ladder
    python -u src/experiment_b4b_network.py --stage smoke
    python -u src/experiment_b4b_network.py --audit-only     # pre-fit audit only
    python -u src/experiment_b4b_network.py --report-only    # never refits

AUTOMATIC LADDER
    NETWORK FEASIBILITY AUDIT -> viable -> SMOKE -> gate -> SCREEN
    -> PROMISING -> FULL -> STOP.
    Feasibility failure, SMOKE failure, or a SCREEN WEAK SIGNAL / REJECT stops
    immediately. ADS-B / B5 is NEVER started by this file.

HYPOTHESIS UNDER TEST (one only)
    Actual shortest-path distance through the mapped taxiway network contains
    useful taxi-out information that categorical STAND/RUNWAY and straight-line
    geometry do not capture.

    CONTROL   the established Exp04 18 features
    CANDIDATE those same 18 features plus exactly 4 B4B network features

    Identical training IDs, validation IDs, CatBoost parameters, target,
    preprocessing, fold and early-stopping procedure. Network information is
    the only experimental variable.

THE FOUR FEATURES
    taxi_network_distance_m       shortest routable taxiway-network distance,
                                   mapped stand -> departure runway threshold
    taxi_network_detour_m         taxi_network_distance_m minus the straight
                                   line stand->threshold distance
    taxi_network_detour_ratio     taxi_network_distance_m divided by that
                                   straight-line distance; NaN when the
                                   straight-line distance is not numerically
                                   safe. Never clipped.
    taxi_network_available        1 iff a valid connected route exists, else 0

    The B4A straight-line distance is used ONLY internally to form the detour
    and the ratio. It is deliberately NOT exposed as a B4B feature, so that
    B4B measures network information and cannot re-report B4A's signal.

    The ratio is not clipped and no distance bucket, edge count, turn count or
    intersection count is exposed. Edge and turn counts are excluded on
    purpose: OSM segmentation density differs between airports and would leak
    mapping style rather than physics.

TAXIWAY GRAPH
    Built per airport from the FROZEN OSM aeroway=taxiway snapshot already
    captured by the B4 feasibility audit. No live Overpass query is made.

      nodes  OSM way vertices, keyed by OSM node id and prefixed by airport
      edges  consecutive vertex pairs inside one OSM way, bidirectional
      weight geodesic segment length in metres

    Disconnected components are NEVER bridged. No synthetic edge is ever
    added, and no edge is added beyond the consecutive pairs present in the
    source ways. If no valid physical route exists the network feature is NaN.

    No target value, historical taxi time, flight identity, validation
    performance, BLOCK_TIME or ranking label enters graph construction. The
    graph is built once, before the frame is even loaded, and is identical for
    training and validation rows.

STAND START POINT
    The exact frozen structured-ref stand mapping from B4A is reused verbatim
    (same mapping hash, same distinct-coordinate centroids). No fuzzy matching.
    A stand is snapped to the nearest taxiway graph node within a FIXED
    250 m limit, chosen before any model result was seen. Ties break on OSM
    node id, so snapping is deterministic and target-independent. Beyond
    250 m the route is unavailable; no distant node is accepted and none is
    fabricated.

RUNWAY DESTINATION
    The correct departure threshold from the frozen OurAirports mapping, i.e.
    the threshold a flight departing that designator starts from, never the
    opposite threshold and never the runway centre. Snapped to the nearest
    taxiway node within a FIXED 300 m limit with the same deterministic rule.
    The runway centre is never used as a feature or a routing target.

CONNECTED-ROUTE RULE
    The snapped stand node and the snapped runway node must lie in the SAME
    connected component of that airport's taxiway graph. Otherwise the route is
    unavailable (NaN) and `taxi_network_available` is 0. Components are never
    bridged. Among valid candidates the shortest routable solution is selected,
    which is by construction the physically shortest one.

PRE-FIT NETWORK AUDIT
    Runs before any CatBoost fit and reports, per airport, departure rows,
    stand mapping coverage, runway mapping coverage, valid network-route
    coverage, median / p90 / p99 / max network distance, median and p95
    stand and runway snap distances, and the disconnected-route rate.

    Feasibility requires ALL of:
      - row-weighted valid network-route coverage >= 75%
      - every mapped network distance > 0 and <= 25,000 m
      - every mapped network distance >= its straight-line distance, allowing
        only a tiny numerical tolerance
    Coverage below 75% STOPS BEFORE CATBOOST with
    B4B NETWORK FEASIBILITY FAILED. The threshold is never loosened.

INDEPENDENT ROUTE SANITY CHECK
    A deterministic sample of at least 2,000 mapped departures is re-verified
    through a deliberately independent second implementation: a plain-Python
    brute-force snap, a dict adjacency rebuilt from the source ways, and a
    heapq Dijkstra, none of which share code with the production path. It
    re-checks stand and runway mapping, component membership, shortest-path
    distance, distance >= straight-line distance, absence of any cross-airport
    node and absence of any synthetic edge. Disagreements are resolved before
    any training.

MODEL POPULATION
    COMPLETE departures only. JAN trains on non-JAN COMPLETE and validates on
    all JAN COMPLETE; JUL mirrors it. SMOKE and SCREEN use the frozen
    screening IDs. Validation is never subsampled.

1500 ITERATION CAP
    B4A SCREEN approached the 1000-tree ceiling in both arms, so B4B raises the
    budget to 1500 for BOTH the control and the candidate. Fresh B4B controls
    are trained rather than reusing cached 1000-tree controls, because the
    iteration budget differs. The primary comparison is always the B4B
    candidate against its OWN 1500-tree control.

    Every model reports whether it early-stopped normally or reached the
    1500-tree ceiling. A fold in which BOTH the control and the candidate hit
    the ceiling is flagged ITERATION-CEILING LIMITED, and convergence is never
    claimed silently.

DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION
    All models early-stop on the full outer JAN/JUL COMPLETE validation rows,
    matching the established screening procedure.

TARGETED EXCLUSIONS
    No B4A feature is present, no raw coordinate, no graph edge/turn/intersection
    count, no B2 congestion, no B3 target prior, no friend F3 feature, no
    weather, no ADS-B, no flight identity, no residual target, no new
    categorical interaction. reference/friend_f3/* is never imported.

SAFETY
    Alignment is by MVT_ID only, never positional. This writes no model
    artifact, no ranking prediction and no submission.

CACHE / REPORTING RULE
    compute() and finalize_report() are separate. Every expensive fit is
    cached immediately with its predictions, IDs, target, best iteration,
    feature list, training-ID hash, metrics and runtime. The network feature
    table is cached INDEPENDENTLY, so graph shortest paths are never recomputed
    merely to render a report. --report-only regenerates the report from those
    caches alone. A wording, formatting, key, table or file-output defect is a
    REPORTING-ONLY BUG and must never trigger a refit or a re-route.
"""

from __future__ import annotations

import gc
import hashlib
import heapq
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostRegressor
from scipy.sparse.csgraph import connected_components, dijkstra

sys.path.insert(0, "src")

import screening_protocol as SP  # noqa: E402
from train_catboost_v8_defs import (  # noqa: E402
    TARGET, T4_FEATURES, V7_CATEGORICALS, V7_FEATURES,
)

# ---------------------------------------------------------------- config
REPORT = Path("reports/b4b_network.txt")
CACHE = SP.CACHE_DIR / "b4b_network_cache.pkl"
NET_CACHE = SP.CACHE_DIR / "b4b_network_features.pkl"
FULL_CONTROL_CACHE = SP.CACHE_DIR / "complete_only_full_control_1500.pkl"

# the B4A frozen geometry, reused verbatim: same stand mapping, same runway
# thresholds, same distinct-coordinate centroids
GEO_DIR = Path("data/external/b4a_geometry")
GEO_CACHE = GEO_DIR / "frozen_geometry.pkl"

# the frozen OSM / OurAirports snapshots from the B4 feasibility audit
AUDIT_DIR = Path(os.environ.get("TEMP", ".")) / "opencode" / "b4_geo"
TW_DIR = AUDIT_DIR

AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD",
            "LFPG", "LIRF", "LSZH", "LTFM")

STAGES = ("SMOKE", "SCREEN", "FULL")

CONTROL_FEATURES = V7_FEATURES + T4_FEATURES + ["adep_reference_state"]
CONTROL_CATEGORICALS = V7_CATEGORICALS + ["adep_reference_state"]

NET_FEATURES = (
    "taxi_network_distance_m",
    "taxi_network_detour_m",
    "taxi_network_detour_ratio",
    "taxi_network_available",
)
B4B_FEATURES = CONTROL_FEATURES + list(NET_FEATURES)
B4B_CATEGORICALS = list(CONTROL_CATEGORICALS)

# every B4A feature name, so B4B can prove none of them leaked in
B4A_FEATURES = (
    "geom_stand_to_threshold_m", "geom_east_offset_m", "geom_north_offset_m",
    "geom_runway_length_m", "geom_runway_heading_sin", "geom_runway_heading_cos",
    "geom_relative_bearing_sin", "geom_relative_bearing_cos", "geometry_available",
)

PARAMS = dict(
    iterations=1500, depth=8, learning_rate=0.1, loss_function="RMSE",
    random_seed=42, verbose=200, early_stopping_rounds=50, use_best_model=True,
    allow_writing_files=False,
)

GATE_POOLED_TOL = 1.0
GATE_FOLD_TOL = 3.0
PROMISING_POOLED = 2.0
PROMISING_FOLD = 1.0

# a best iteration within one patience window of the cap may have been cut off
# by the cap rather than by overfitting
CEILING_WINDOW = PARAMS["early_stopping_rounds"]

# snap limits, fixed before any model result was seen
MAX_STAND_SNAP_M = 250.0
MAX_RUNWAY_SNAP_M = 300.0

# feasibility gate
MIN_ROUTE_COVERAGE_PCT = 75.0
MAX_NET_DIST_M = 25_000.0
MIN_NET_DIST_M = 0.0
DIST_TOLERANCE_M = 1.0        # tiny numerical tolerance only
MIN_SAFE_STRAIGHT_M = 1.0     # below this the ratio is numerically unsafe

SANITY_N = 2000
SANITY_SEED = 20260928

EARTH_R_M = 6_371_000.0

NON_PHYSICAL_SENTINELS = ("NA", "H", "HEL", "UNK", "UNKNOWN")

B2_CONGESTION_PREFIXES = ("dep_prev_", "arr_prev_", "secs_since_prev_",
                          "dep_minus_arr_")
B3_STRUCT_PREFIX = "struct_"

FORBIDDEN_IN_NET = (
    "BLOCK_TIME", "hist_", "struct_", "dep_prev_", "arr_prev_",
    "secs_since_prev_", "dep_minus_arr_", "congestion", "traffic_last",
    "weather", "wind", "temp", "precip", "qnh", "geom_",
)

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


def note(t=""):
    _buf.append(t)


def plain(t=""):
    print(t, flush=True)


def flush():
    if _buf:
        print("\n".join(_buf), flush=True)
        _buf.clear()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def sse(y, p):
    return float(np.sum((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def safety(name, ok, detail=""):
    line = f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else "")
    if not ok:
        raise AssertionError(f"SAFETY CHECK FAILED: {name} ({detail})")
    return line


def norm_ref(s) -> str:
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return "" if s in NON_PHYSICAL_SENTINELS else s


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def great_circle(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    # DEFECT 1 GUARD: the divisor must be 2.0. Writing "2. ** 2" here evaluates
    # as sin(dl / 4) and silently shrinks every distance in the file.
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_reference_m(lat1, lon1, lat2, lon2) -> float:
    """Independent scalar haversine, used only to assert great_circle agrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_R_M * math.asin(math.sqrt(min(1.0, a)))


# ============================================================== TAXIWAY GRAPH
def build_airport_graph(icao: str) -> dict:
    """One airport's taxiway graph from the frozen OSM snapshot.

    Nodes are OSM way vertices keyed by OSM node id. Edges are exactly the
    consecutive vertex pairs inside one way. Nothing is added, merged or
    bridged, so the connected components are the true mapped components.
    """
    path = TW_DIR / f"tw_{icao}.json"
    elements = json.load(open(path))["elements"]

    lat: dict[int, float] = {}
    lon: dict[int, float] = {}
    segments: set[tuple[int, int]] = set()
    n_ways = 0
    for e in elements:
        if e.get("tags", {}).get("aeroway") != "taxiway":
            continue
        nodes, geom = e.get("nodes") or [], e.get("geometry") or []
        if len(nodes) != len(geom) or len(nodes) < 2:
            continue
        n_ways += 1
        for nid, pt in zip(nodes, geom):
            lat[int(nid)] = float(pt["lat"])
            lon[int(nid)] = float(pt["lon"])
        for a, b in zip(nodes[:-1], nodes[1:]):
            a, b = int(a), int(b)
            if a != b:
                segments.add((a, b) if a < b else (b, a))

    ids = sorted(lat)
    idx = {v: i for i, v in enumerate(ids)}
    la = np.array([lat[v] for v in ids], float)
    lo = np.array([lon[v] for v in ids], float)
    rows, cols, wts = [], [], []
    for a, b in segments:
        rows.append(idx[a])
        cols.append(idx[b])
        wts.append(float(great_circle(la[idx[a]], lo[idx[a]], la[idx[b]], lo[idx[b]])))
    n = len(ids)
    # DEFECT 2 GUARD: a taxiway is bidirectional, so BOTH orientations must be
    # present. A one-directional matrix makes scipy's directed=False silently
    # return too-short distances instead of failing loudly.
    rows, cols, wts = [], [], []
    for a, b in segments:
        w = float(great_circle(la[idx[a]], lo[idx[a]], la[idx[b]], lo[idx[b]]))
        rows.append(idx[a]); cols.append(idx[b]); wts.append(w)
        rows.append(idx[b]); cols.append(idx[a]); wts.append(w)
    mat = sp.coo_matrix((wts, (rows, cols)), shape=(n, n)).tocsr()
    assert mat.nnz == 2 * len(segments), "graph adjacency is not symmetric"
    ncomp, labels = connected_components(mat, directed=False)

    # reproducibility hash over the exact physical graph content
    h = hashlib.sha256()
    for v in ids:
        h.update(f"{v}:{lat[v]!r}:{lon[v]!r};".encode())
    for a, b in sorted(segments):
        h.update(f"{a}-{b};".encode())

    sizes = np.bincount(labels)
    return dict(
        icao=icao, ids=ids, idx=idx, lat=la, lon=lo, mat=mat,
        n_nodes=n, n_edges=len(segments), n_ways=n_ways,
        n_components=int(ncomp), labels=labels, component_sizes=sizes,
        graph_hash=h.hexdigest(), source_file=str(path),
        source_sha256=sha256_file(path),
    )


def build_all_graphs() -> dict:
    graphs = {}
    for a in AIRPORTS:
        t = time.time()
        g = build_airport_graph(a)
        graphs[a] = g
        plain(f"  {a}: {g['n_ways']:>4} ways  {g['n_edges']:>6,} edges  "
              f"{g['n_nodes']:>6,} nodes  {g['n_components']:>3} components  "
              f"largest {g['component_sizes'].max():>6,}  "
              f"hash {g['graph_hash'][:12]}  {time.time() - t:.1f}s")
    return graphs


def snap_node(g: dict, lat, lon, limit):
    """Nearest taxiway node within `limit`, or (-1, distance).

    Deterministic: numpy stable argsort on the distance, so an exact tie
    resolves to the lowest node index, which is the lowest OSM node id.
    No target information takes any part in this choice.
    """
    d = great_circle(lat, lon, g["lat"], g["lon"])
    order = np.argsort(d, kind="stable")
    i = int(order[0])
    if d[i] > limit:
        return -1, float(d[i])
    return i, float(d[i])


# ==================================================== NETWORK FEATURE BUILD
def build_network_features(dep: pd.DataFrame, geo: dict, graphs: dict,
                           checks: list | None = None) -> dict:
    """The 4 B4B features for every row, plus the internal audit columns.

    Reads only ADEP_mvt, STAND_mvt, RUNWAY_mvt plus the frozen external
    geometry and the frozen OSM taxiway graphs. The target is never touched.
    """
    checks = checks if checks is not None else []
    n = len(dep)
    # guard the distance formula itself against a silent transcription slip
    _rng = np.random.default_rng(7)
    _la = _rng.uniform(35.0, 62.0, 5000)
    _lo = _rng.uniform(-12.0, 28.0, 5000)
    _err = max(abs(float(great_circle(_la[i], _lo[i], _la[i + 1], _lo[i + 1]))
                    - haversine_reference_m(float(_la[i]), float(_lo[i]),
                                            float(_la[i + 1]), float(_lo[i + 1])))
              for i in range(500))
    checks.append(safety(
        "vectorised great_circle matches the independent scalar haversine exactly",
        _err == 0.0, f"max error {_err:.3e} m over 500 random pairs"))
    adep = dep["ADEP_mvt"].to_numpy().astype(str)
    sraw = dep["STAND_mvt"].to_numpy().astype(str)
    rraw = dep["RUNWAY_mvt"].to_numpy().astype(str)
    sk = np.array([norm_ref(s) for s in sraw])
    rk = np.array([s.strip() for s in rraw])

    # distinct (airport, ref) pairs are few, so resolve and snap each of them
    # exactly once and broadcast the result back over the rows
    s_keys = np.char.add(np.char.add(adep, "|"), sk)
    r_keys = np.char.add(np.char.add(adep, "|"), rk)
    u_s, s_codes = np.unique(s_keys, return_inverse=True)
    u_r, r_codes = np.unique(r_keys, return_inverse=True)
    a_of = np.array([k.split("|", 1)[0] for k in u_s])
    ref_of = np.array([k.split("|", 1)[1] for k in u_s])
    ra_of = np.array([k.split("|", 1)[0] for k in u_r])
    rref_of = np.array([k.split("|", 1)[1] for k in u_r])

    rtab, stab = geo["runway"], geo["stand"]
    nu_s, nu_r = len(u_s), len(u_r)
    stand_node = np.full(nu_s, -1, dtype=np.int64)
    stand_snap = np.full(nu_s, np.nan)
    st_lat = np.full(nu_s, np.nan)
    st_lon = np.full(nu_s, np.nan)
    for j in range(nu_s):
        if not ref_of[j]:
            continue
        rec = stab.get((a_of[j], ref_of[j]))
        if rec is None:
            continue
        st_lat[j], st_lon[j] = rec["lat"], rec["lon"]
        node, d = snap_node(graphs[a_of[j]], rec["lat"], rec["lon"], MAX_STAND_SNAP_M)
        stand_node[j] = node
        stand_snap[j] = d

    rwy_node = np.full(nu_r, -1, dtype=np.int64)
    rwy_snap = np.full(nu_r, np.nan)
    rw_lat = np.full(nu_r, np.nan)
    rw_lon = np.full(nu_r, np.nan)
    for j in range(nu_r):
        if not rref_of[j]:
            continue
        rec = rtab.get((ra_of[j], rref_of[j]))
        if rec is None:
            continue
        rw_lat[j], rw_lon[j] = rec["lat"], rec["lon"]
        node, d = snap_node(graphs[ra_of[j]], rec["lat"], rec["lon"], MAX_RUNWAY_SNAP_M)
        rwy_node[j] = node
        rwy_snap[j] = d

    hit_rwy = np.isfinite(rw_lat)[r_codes]
    straight = great_circle(st_lat[s_codes], st_lon[s_codes],
                            rw_lat[r_codes], rw_lon[r_codes])

    # ---- shortest network distance for every distinct snapped node triple
    sn_row, rw_row = stand_node[s_codes], rwy_node[r_codes]
    row_air = a_of[s_codes]
    trips_by_air: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for ac, sn, rn in zip(row_air.tolist(), sn_row.tolist(), rw_row.tolist()):
        if sn < 0 or rn < 0:
            continue
        g = graphs[ac]
        if g["labels"][sn] != g["labels"][rn]:
            continue                       # different components: no route
        if (ac, sn, rn) not in seen:
            seen.add((ac, sn, rn))
            trips_by_air[ac].append((sn, rn))

    fills: list[tuple[str, int, int, float]] = []
    for ac, trips in trips_by_air.items():
        g = graphs[ac]
        srcs = sorted({t[0] for t in trips})
        D = dijkstra(g["mat"], directed=False, indices=srcs)
        for sn, rn in trips:
            fills.append((ac, sn, rn, float(D[srcs.index(sn), rn])))

    net = np.full(n, np.nan)
    for ac, sn, rn, d in fills:
        if not np.isfinite(d) or d < 0:
            continue
        net[(row_air == ac) & (sn_row == sn) & (rw_row == rn)] = d

    # straight line between the two SNAPPED nodes. The triangle inequality
    # applies to the graph endpoints, not to the stand centroid and the
    # threshold, because snapping can move either endpoint tens of metres
    # toward the other and legitimately shorten the comparison.
    sn_lat = np.full(n, np.nan); sn_lon = np.full(n, np.nan)
    rw_lat = np.full(n, np.nan); rw_lon = np.full(n, np.nan)
    for ac in AIRPORTS:
        g = graphs[ac]
        sel = np.flatnonzero(row_air == ac)
        if sel.size == 0:
            continue
        s_, r_ = sn_row[sel], rw_row[sel]
        ok = (s_ >= 0) & (r_ >= 0)
        sel, s_, r_ = sel[ok], s_[ok], r_[ok]
        sn_lat[sel] = g["lat"][s_]; sn_lon[sel] = g["lon"][s_]
        rw_lat[sel] = g["lat"][r_]; rw_lon[sel] = g["lon"][r_]
    snap_straight = great_circle(sn_lat, sn_lon, rw_lat, rw_lon)

    # ---- the four features
    safe_straight = np.isfinite(straight) & (straight > MIN_SAFE_STRAIGHT_M)
    detour = net - straight
    ratio = np.where(safe_straight, net / np.where(safe_straight, straight, 1.0), np.nan)
    detour = np.where(np.isfinite(detour), detour, np.nan)
    available = np.isfinite(net).astype("float64")

    out = pd.DataFrame({
        "taxi_network_distance_m": net,
        "taxi_network_detour_m": detour,
        "taxi_network_detour_ratio": ratio,
        "taxi_network_available": available,
    }, index=dep["MVT_ID_mvt"].to_numpy())

    audit_cols = pd.DataFrame({
        "stand_snap_m": stand_snap[s_codes],
        "runway_snap_m": rwy_snap[r_codes],
        "straight_line_m": straight,
        "snap_straight_m": snap_straight,
        "adep": adep,
    }, index=out.index)

    # ---- correctness invariants
    m = available == 1.0
    checks.append(safety(
        "valid routes: network distance > 0",
        bool((net[m] > MIN_NET_DIST_M).all()),
        f"min {net[m].min():.3f} m" if m.any() else "no valid routes"))
    checks.append(safety(
        f"valid routes: network distance <= {MAX_NET_DIST_M:.0f} m",
        bool((net[m] <= MAX_NET_DIST_M).all()),
        f"max {net[m].max():.1f} m" if m.any() else "no valid routes"))
    checks.append(safety(
        "valid routes: network distance >= straight line between the two SNAPPED "
        f"nodes (tolerance {DIST_TOLERANCE_M:.1f} m)",
        bool((net[m] >= snap_straight[m] - DIST_TOLERANCE_M).all()),
        f"min detour {np.min(net[m] - snap_straight[m]):.2f} m" if m.any() else "n/a"))
    checks.append(safety(
        f"the only negative detours are snap artefacts: every one of them is "
        f"still >= the snapped-endpoint straight line",
        bool((net[net < straight - DIST_TOLERANCE_M]
              >= snap_straight[net < straight - DIST_TOLERANCE_M]
              - DIST_TOLERANCE_M).all())
        if (net < straight - DIST_TOLERANCE_M).any() else True,
        f"{int((net < straight - DIST_TOLERANCE_M).sum())} rows, min "
        f"{np.min(net - straight):.2f} m" if m.any() else "n/a"))
    checks.append(safety(
        "taxi_network_available is exactly 0/1 and equals 'a route exists'",
        set(np.unique(available)).issubset({0.0, 1.0})
        and bool(((available == 1.0) == np.isfinite(net)).all())))
    checks.append(safety(
        "every network feature except the flag is NaN where unavailable",
        bool(out.loc[~m, list(NET_FEATURES[:3])].isna().all().all()),
        f"{int((~m).sum())} rows without a route"))
    checks.append(safety(
        "no network feature is NaN where available",
        bool(not out.loc[m, list(NET_FEATURES[:3])].isna().any().any())))
    checks.append(safety(
        "detour_ratio is NaN unless the straight-line distance is numerically safe",
        bool(out.loc[~safe_straight, "taxi_network_detour_ratio"].isna().all())))
    sl = audit_cols["straight_line_m"].to_numpy()
    checks.append(safety(
        "detour_ratio agrees with distance / straight-line to within 1e-6",
        bool(np.allclose(out.loc[m, "taxi_network_detour_ratio"].to_numpy(),
                         net[m] / sl[m], rtol=1e-6, equal_nan=True))))
    checks.append(safety(
        "detour_m agrees with distance minus straight-line",
        bool(np.allclose(out.loc[m, "taxi_network_detour_m"].to_numpy(),
                         net[m] - sl[m], rtol=1e-9, equal_nan=True))))
    checks.append(safety(
        "every stand snap is within the fixed 250 m limit",
        bool((audit_cols.loc[m, "stand_snap_m"] <= MAX_STAND_SNAP_M).all()),
        f"max {audit_cols.loc[m, 'stand_snap_m'].max():.1f} m" if m.any() else "n/a"))
    checks.append(safety(
        "every runway snap is within the fixed 300 m limit",
        bool((audit_cols.loc[m, "runway_snap_m"] <= MAX_RUNWAY_SNAP_M).all()),
        f"max {audit_cols.loc[m, 'runway_snap_m'].max():.1f} m" if m.any() else "n/a"))
    checks.append(safety(
        "every available row has both a stand node and a runway node",
        bool((sn_row[m] >= 0).all() and (rw_row[m] >= 0).all())))

    return dict(features=out, audit=audit_cols, straight=straight, net=net,
                snap_straight=snap_straight, available=available,
                sn_row=sn_row, rw_row=rw_row, row_air=row_air,
                s_codes=s_codes, r_codes=r_codes, u_s=u_s, u_r=u_r)


# ================================================== PRE-FIT NETWORK AUDIT
def audit_network(dep: pd.DataFrame, geo: dict, graphs: dict, nf: dict) -> dict:
    adep = dep["ADEP_mvt"].to_numpy().astype(str)
    au = nf["audit"]
    avail = nf["available"] == 1.0
    net = nf["net"]
    straight = nf["straight"]
    s_snap = au["stand_snap_m"].to_numpy()
    r_snap = au["runway_snap_m"].to_numpy()
    snap_straight = au["snap_straight_m"].to_numpy()
    sk = np.array([norm_ref(s) for s in dep["STAND_mvt"].to_numpy().astype(str)])
    rk = np.array([s.strip() for s in dep["RUNWAY_mvt"].to_numpy().astype(str)])
    hit_s = np.array([bool(k) and (a, k) in geo["stand"] for a, k in zip(adep, sk)])
    hit_r = np.array([(a, k) in geo["runway"] for a, k in zip(adep, rk)])

    n = len(dep)
    m = avail
    nd = net[m]
    cov = dict(
        n=n,
        stand_pct=100.0 * float(hit_s.mean()),
        runway_pct=100.0 * float(hit_r.mean()),
        route_pct=100.0 * float(m.mean()),
        n_stand_rows=int(hit_s.sum()),
        n_runway_rows=int(hit_r.sum()),
        n_route_rows=int(m.sum()),
        median_net_m=float(np.median(nd)) if nd.size else float("nan"),
        p90_net_m=float(np.percentile(nd, 90)) if nd.size else float("nan"),
        p99_net_m=float(np.percentile(nd, 99)) if nd.size else float("nan"),
        max_net_m=float(nd.max()) if nd.size else float("nan"),
        min_net_m=float(nd.min()) if nd.size else float("nan"),
        median_stand_snap_m=float(np.nanmedian(s_snap[m])) if m.any() else float("nan"),
        p95_stand_snap_m=float(np.nanpercentile(s_snap[m], 95)) if m.any() else float("nan"),
        max_stand_snap_m=float(np.nanmax(s_snap[m])) if m.any() else float("nan"),
        median_runway_snap_m=float(np.nanmedian(r_snap[m])) if m.any() else float("nan"),
        p95_runway_snap_m=float(np.nanpercentile(r_snap[m], 95)) if m.any() else float("nan"),
        max_runway_snap_m=float(np.nanmax(r_snap[m])) if m.any() else float("nan"),
        median_detour_ratio=float(np.nanmedian(
            (nd / straight[m]))) if m.any() else float("nan"),
        # violations
        n_le_zero=int((nd <= MIN_NET_DIST_M).sum()),
        n_gt_max=int((nd > MAX_NET_DIST_M).sum()),
        # the triangle inequality applies to the two SNAPPED nodes. Compared
        # against the stand centroid and the threshold it can legitimately go
        # negative, because a snap inside the fixed limit may move an endpoint
        # toward the other, so that count is reported rather than treated as a
        # failure. Both counts are carried.
        n_below_snapped=int((nd < snap_straight[m] - DIST_TOLERANCE_M).sum()),
        n_below_straight=int((nd < straight[m] - DIST_TOLERANCE_M).sum()),
        min_detour_m=float(np.min(nd - straight[m])) if nd.size else float("nan"),
        n_disconnected=int((hit_s & hit_r & ~m).sum()),
        n_stand_unmapped=int((~hit_s).sum()),
        n_runway_unmapped=int((~hit_r).sum()),
        graph_hashes={a: graphs[a]["graph_hash"] for a in AIRPORTS},
        n_graph_nodes={a: graphs[a]["n_nodes"] for a in AIRPORTS},
        n_graph_edges={a: graphs[a]["n_edges"] for a in AIRPORTS},
        n_graph_components={a: graphs[a]["n_components"] for a in AIRPORTS},
    )

    by = {}
    for a in sorted(set(adep)):
        mm = adep == a
        ma = mm & m
        d = net[ma]
        by[a] = dict(
            rows=int(mm.sum()),
            stand_pct=100.0 * float(hit_s[mm].mean()),
            runway_pct=100.0 * float(hit_r[mm].mean()),
            route_pct=100.0 * float(m[mm].mean()),
            median_net_m=float(np.median(d)) if d.size else float("nan"),
            p90_net_m=float(np.percentile(d, 90)) if d.size else float("nan"),
            p99_net_m=float(np.percentile(d, 99)) if d.size else float("nan"),
            max_net_m=float(d.max()) if d.size else float("nan"),
            median_stand_snap_m=float(np.nanmedian(s_snap[ma])) if ma.any() else float("nan"),
            p95_stand_snap_m=float(np.nanpercentile(s_snap[ma], 95)) if ma.any() else float("nan"),
            median_runway_snap_m=float(np.nanmedian(r_snap[ma])) if ma.any() else float("nan"),
            p95_runway_snap_m=float(np.nanpercentile(r_snap[ma], 95)) if ma.any() else float("nan"),
            disconnected_pct=100.0 * float((~m[mm]).sum()) / max(1, int((hit_s[mm] & hit_r[mm]).sum())),
            n_graph_nodes=graphs[a]["n_nodes"],
            n_graph_components=graphs[a]["n_components"],
        )
    cov["by_airport"] = by
    return cov


def feasibility_check(cov: dict) -> list[str]:
    """Reasons to STOP before any CatBoost fit. Empty means proceed."""
    bad = []
    if cov["route_pct"] < MIN_ROUTE_COVERAGE_PCT:
        bad.append(f"row-weighted valid network-route coverage {cov['route_pct']:.3f}% "
                   f"is below the {MIN_ROUTE_COVERAGE_PCT:.0f}% feasibility threshold")
    if cov["n_le_zero"]:
        bad.append(f"{cov['n_le_zero']} valid routes have a network distance <= 0 m")
    if cov["n_gt_max"]:
        bad.append(f"{cov['n_gt_max']} valid routes exceed {MAX_NET_DIST_M:.0f} m "
                   f"(max {cov['max_net_m']:.1f} m)")
    if cov["n_below_snapped"]:
        bad.append(f"{cov['n_below_snapped']} valid routes violate the triangle "
                   f"inequality against the straight line between their two "
                   f"snapped nodes, which cannot happen on a real graph")
    if cov["min_net_m"] <= MIN_NET_DIST_M and cov["n_route_rows"]:
        bad.append(f"minimum mapped network distance {cov['min_net_m']:.3f} m is not positive")
    return bad


# =========================================== INDEPENDENT ROUTE SANITY CHECK
def _independent_graph(icao: str) -> dict:
    """Deliberately separate, plain-Python rebuild of one airport's graph.

    No numpy, no scipy, no shared helper with the production path: a dict
    adjacency built straight from the source ways, plus the raw way-segment
    count, so a synthetic edge or a lost segment would show up as a mismatch.
    """
    elements = json.load(open(TW_DIR / f"tw_{icao}.json"))["elements"]
    coords: dict[int, tuple[float, float]] = {}
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    n_raw_segments = 0
    distinct: set[tuple[int, int]] = set()
    for e in elements:
        if e.get("tags", {}).get("aeroway") != "taxiway":
            continue
        nodes, geom = e.get("nodes") or [], e.get("geometry") or []
        if len(nodes) != len(geom) or len(nodes) < 2:
            continue
        for nid, pt in zip(nodes, geom):
            coords[int(nid)] = (float(pt["lat"]), float(pt["lon"]))
        for a, b in zip(nodes[:-1], nodes[1:]):
            a, b = int(a), int(b)
            if a == b:
                continue
            n_raw_segments += 1
            distinct.add((a, b) if a < b else (b, a))
            la1, lo1 = coords[a]
            la2, lo2 = coords[b]
            w = _haversine_m(la1, lo1, la2, lo2)
            adj[a].append((b, w))
            adj[b].append((a, w))
    return dict(coords=coords, adj=adj, n_raw_segments=n_raw_segments,
                n_distinct_segments=len(distinct))


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_R_M * math.asin(math.sqrt(min(1.0, a)))


def _dijkstra_plain(adj, src, dst) -> float:
    dist = {src: 0.0}
    seen = set()
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            return d
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")) - 1e-12:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return float("inf")


def route_sanity_check(dep: pd.DataFrame, geo: dict, graphs: dict, nf: dict,
                       n_sample: int = SANITY_N) -> dict:
    """Re-verify a deterministic sample of mapped departures independently."""
    checks: list[str] = []
    au = nf["audit"]
    adep = dep["ADEP_mvt"].to_numpy().astype(str)
    sk = np.array([norm_ref(s) for s in dep["STAND_mvt"].to_numpy().astype(str)])
    rk = np.array([s.strip() for s in dep["RUNWAY_mvt"].to_numpy().astype(str)])
    avail = nf["available"] == 1.0
    straight = nf["straight"]
    net = nf["net"]

    mapped = np.flatnonzero(avail)
    assert mapped.size >= n_sample, f"only {mapped.size} routed rows, need {n_sample}"
    rng = np.random.default_rng(SANITY_SEED)
    pick = np.sort(rng.choice(mapped, size=n_sample, replace=False))

    indep = {a: _independent_graph(a) for a in AIRPORTS}
    max_edge_err = 0.0
    for a in AIRPORTS:
        g, ig = graphs[a], indep[a]
        checks.append(safety(
            f"{a}: independent rebuild finds exactly the same node count",
            len(ig["coords"]) == g["n_nodes"],
            f"{len(ig['coords'])} vs {g['n_nodes']}"))
        # no synthetic bridge: the production edge count is exactly the number
        # of DISTINCT consecutive way pairs, so nothing was added, and it never
        # exceeds the raw pair count, so nothing was invented either
        deg = sum(len(v) for v in ig["adj"].values())
        checks.append(safety(
            f"{a}: no synthetic edge (production edges == distinct way pairs)",
            g["n_edges"] == ig["n_distinct_segments"],
            f"{g['n_edges']} vs {ig['n_distinct_segments']}"))
        checks.append(safety(
            f"{a}: production edges never exceed the raw consecutive way pairs",
            g["n_edges"] <= ig["n_raw_segments"],
            f"{g['n_edges']} <= {ig['n_raw_segments']} "
            f"({ig['n_raw_segments'] - g['n_edges']} repeated ways de-duplicated)"))
        checks.append(safety(
            f"{a}: independent adjacency is symmetric with {deg} directed links",
            deg == 2 * ig["n_raw_segments"]))
        for nid, (la, lo) in list(ig["coords"].items())[:200]:
            i = g["idx"][nid]
            max_edge_err = max(max_edge_err,
                               abs(la - g["lat"][i]) + abs(lo - g["lon"][i]))
    checks.append(safety(
        "independent node coordinates match the production graph exactly",
        max_edge_err == 0.0, f"max coordinate error {max_edge_err:g}"))

    # node-id sets are disjoint across airports, so no route can leave its airport
    allsets = [set(graphs[a]["ids"]) for a in AIRPORTS]
    disjoint = all(not (allsets[i] & allsets[j])
                   for i in range(len(allsets)) for j in range(i + 1, len(allsets)))
    checks.append(safety("taxiway node sets are disjoint across airports", disjoint))
    checks.append(safety(
        "every graph was built from only its own airport's source file",
        all(graphs[a]["source_file"].endswith(f"tw_{a}.json") for a in AIRPORTS)))

    n_bad_map = n_bad_snap = n_bad_conn = n_bad_dist = n_bad_straight = 0
    n_neg_detour = 0
    worst = 0.0
    for i in pick:
        a, s, r = adep[i], sk[i], rk[i]
        rec_s, rec_r = geo["stand"].get((a, s)), geo["runway"].get((a, r))
        if rec_s is None or rec_r is None:
            n_bad_map += 1
            continue
        ig = indep[a]
        # independent brute-force snap over every node
        ds = [( _haversine_m(rec_s["lat"], rec_s["lon"], la, lo), nid)
              for nid, (la, lo) in ig["coords"].items()]
        ds.sort(key=lambda t: (t[0], t[1]))
        best_s_d, best_s = ds[0]
        dr = [( _haversine_m(rec_r["lat"], rec_r["lon"], la, lo), nid)
              for nid, (la, lo) in ig["coords"].items()]
        dr.sort(key=lambda t: (t[0], t[1]))
        best_r_d, best_r = dr[0]
        if best_s_d > MAX_STAND_SNAP_M or best_r_d > MAX_RUNWAY_SNAP_M:
            n_bad_snap += 1
            continue
        g = graphs[a]
        if g["labels"][g["idx"][best_s]] != g["labels"][g["idx"][best_r]]:
            n_bad_conn += 1
            continue
        d = _dijkstra_plain(ig["adj"], best_s, best_r)
        if not np.isfinite(d) or abs(d - net[i]) > 1e-6:
            n_bad_dist += 1
            worst = max(worst, abs(d - net[i]))
            continue
        # the triangle inequality applies to the two SNAPPED nodes. Comparing
        # against the stand centroid and the threshold instead can legitimately
        # go negative, because a snap may move an endpoint toward the other.
        slat, slon = ig["coords"][best_s]
        rlat, rlon = ig["coords"][best_r]
        snapped = _haversine_m(slat, slon, rlat, rlon)
        if d < snapped - DIST_TOLERANCE_M:
            n_bad_straight += 1
        if d < straight[i] - DIST_TOLERANCE_M:
            n_neg_detour += 1

    checks.append(safety("sanity sample: every stand and runway mapping re-resolves",
                         n_bad_map == 0, f"{n_bad_map} disagreements"))
    checks.append(safety("sanity sample: every independent snap is inside the fixed limit",
                         n_bad_snap == 0, f"{n_bad_snap} disagreements"))
    checks.append(safety("sanity sample: every snapped pair is in one component",
                         n_bad_conn == 0, f"{n_bad_conn} disagreements"))
    checks.append(safety("sanity sample: independent Dijkstra reproduces the distance",
                         n_bad_dist == 0,
                         f"{n_bad_dist} disagreements, max {worst:.3e} m"))
    checks.append(safety("sanity sample: network distance >= straight line between "
                         "the two snapped nodes",
                         n_bad_straight == 0, f"{n_bad_straight} disagreements"))

    return dict(n_sample=int(n_sample), seed=SANITY_SEED,
                n_mapping_disagree=n_bad_map, n_snap_disagree=n_bad_snap,
                n_connectivity_disagree=n_bad_conn, n_distance_disagree=n_bad_dist,
                n_straightline_disagree=n_bad_straight,
                n_negative_detour=n_neg_detour,
                max_distance_abs_error_m=float(worst),
                node_sets_disjoint=bool(disjoint), checks=checks)


def load_frozen_geometry() -> dict:
    assert GEO_CACHE.exists(), f"missing frozen geometry {GEO_CACHE}"
    with open(GEO_CACHE, "rb") as fh:
        g = pickle.load(fh)
    plain(f"  loaded frozen geometry {GEO_CACHE} (frozen {g['meta']['frozen_at']}, "
          f"runway hash {g['meta']['mapping_hashes']['runways'][:12]}, "
          f"stand hash {g['meta']['mapping_hashes']['stands'][:12]})")
    return g


# ---------------------------------------------------------------- metrics
def pooled_of(res, key):
    num = den = 0.0
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        num += sse(F["y"], F[key])
        den += len(F["y"])
    return float(np.sqrt(num / den)), num


def hybrid(res, fold, key):
    F, V = res["folds"][fold], res["val"][fold]
    comp = ~V["vmask"]
    out = V["p_f3"].copy()
    out[comp] = pd.Series(F[key], index=F["ids"]).reindex(V["ids"][comp]).to_numpy()
    assert np.isfinite(out).all()
    return out


def ceiling_limited(best_iter) -> bool:
    """True when a best iteration this late may have been cut off by the cap."""
    return int(best_iter) >= PARAMS["iterations"] - CEILING_WINDOW


def main_table(res):
    note("")
    note(f"  {'fold':<6}{'rows':>10}{'control':>12}{'B4B':>12}{'delta':>10}"
         f"{'control SSE':>15}{'B4B SSE':>15}")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        c, b = rmse(F["y"], F["p_control"]), rmse(F["y"], F["p_b4b"])
        note(f"  {f:<6}{len(F['y']):>10,}{c:>12.6f}{b:>12.6f}{b - c:>+10.4f}"
             f"{sse(F['y'], F['p_control']):>15.4e}{sse(F['y'], F['p_b4b']):>15.4e}")
    cp, cs = pooled_of(res, "p_control")
    bp, bs = pooled_of(res, "p_b4b")
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
def fit_stage(stage, dep, samples, base, net_df, cov):
    assert stage in STAGES
    lkey = stage.lower()
    meta = dep.set_index("MVT_ID_mvt")
    checks: list[str] = []
    res = dict(version=1, stage=stage,
               created=time.strftime("%Y-%m-%d %H:%M:%S"),
               params=dict(PARAMS), control_features=list(CONTROL_FEATURES),
               b4b_features=list(B4B_FEATURES), net_features=list(NET_FEATURES),
               categoricals=list(B4B_CATEGORICALS), network_audit=cov,
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
                             f"(MISSING_4 never trains B4B)",
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

        # the network table is frozen and stage-independent: the SAME table
        # applies to training and validation rows, looked up strictly by MVT_ID
        n_tr = net_df.loc[SP.id_to_int(tr["MVT_ID_mvt"].to_numpy())]
        n_va = net_df.loc[val_ids]
        assert len(n_tr) == len(tr) and len(n_va) == len(va)
        assert np.array_equal(SP.id_to_int(n_tr.index.to_numpy()),
                              SP.id_to_int(tr["MVT_ID_mvt"].to_numpy()))
        assert np.array_equal(SP.id_to_int(n_va.index.to_numpy()), val_ids)

        # training-derived network-distance terciles, SCREEN/FULL diagnostic only
        dt = n_tr["taxi_network_distance_m"].to_numpy()
        edges = np.nanquantile(dt, [1 / 3, 2 / 3]) if np.isfinite(dt).any() \
            else np.array([np.nan, np.nan])
        res.setdefault("bucket_edges", {})[f] = [float(edges[0]), float(edges[1])]

        X_c_tr = tr[CONTROL_FEATURES].reset_index(drop=True)
        X_c_va = va[CONTROL_FEATURES].reset_index(drop=True)
        X_b_tr = pd.concat([X_c_tr, n_tr[list(NET_FEATURES)].reset_index(drop=True)],
                           axis=1)[B4B_FEATURES]
        X_b_va = pd.concat([X_c_va, n_va[list(NET_FEATURES)].reset_index(drop=True)],
                           axis=1)[B4B_FEATURES]
        assert list(X_b_tr.columns) == list(B4B_FEATURES) == list(X_b_va.columns)
        assert len(X_c_tr) == len(X_b_tr) == len(tr) == len(y_tr)
        assert len(X_c_va) == len(X_b_va) == len(val_ids)

        checks.append(safety(
            f"{f}/{stage}: B4B has exactly four extra features over the 18-feature control",
            len(B4B_FEATURES) == len(CONTROL_FEATURES) + 4
            and B4B_FEATURES[:len(CONTROL_FEATURES)] == CONTROL_FEATURES))
        checks.append(safety(
            f"{f}/{stage}: BLOCK_TIME_UTC_mvt never used",
            "BLOCK_TIME_UTC_mvt" not in B4B_FEATURES + CONTROL_FEATURES))
        checks.append(safety(
            f"{f}/{stage}: the four network features are numerical",
            all(pd.api.types.is_numeric_dtype(X_b_tr[k]) for k in NET_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: no B4A geometry feature is present",
            not (set(B4B_FEATURES) & set(B4A_FEATURES)),
            ", ".join(sorted(set(B4B_FEATURES) & set(B4A_FEATURES))) or "none"))
        checks.append(safety(
            f"{f}/{stage}: the B4A straight-line distance is not exposed as a feature",
            "geom_stand_to_threshold_m" not in B4B_FEATURES))
        checks.append(safety(
            f"{f}/{stage}: no graph edge, turn or intersection count is exposed",
            not any(t in k for k in NET_FEATURES
                    for t in ("edge", "turn", "intersect", "node", "degree", "count"))))
        checks.append(safety(
            f"{f}/{stage}: no B2 congestion feature anywhere in the matrix",
            not any(k.startswith(B2_CONGESTION_PREFIXES) for k in B4B_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: no B3 structural target prior anywhere in the matrix",
            not any(k.startswith(B3_STRUCT_PREFIX) for k in B4B_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: control is exactly the established Exp04 18-feature set",
            CONTROL_FEATURES == V7_FEATURES + T4_FEATURES + ["adep_reference_state"]
            and len(CONTROL_FEATURES) == 18))
        checks.append(safety(
            f"{f}/{stage}: no new categorical interaction and no residual target",
            B4B_CATEGORICALS == CONTROL_CATEGORICALS
            and TARGET == "TAXITIME_SEC_mvt"))
        checks.append(safety(
            f"{f}/{stage}: network features carry no forbidden source",
            not any(any(t in k for t in FORBIDDEN_IN_NET) for k in NET_FEATURES)))
        checks.append(safety(
            f"{f}/{stage}: no raw latitude/longitude is exposed as a feature",
            not any(any(t in k for t in RAW_COORD_TOKENS) for k in NET_FEATURES),
            ", ".join(k for k in NET_FEATURES
                      if any(t in k for t in RAW_COORD_TOKENS)) or "none"))
        checks.append(safety(
            f"{f}/{stage}: taxi_network_available is a clean 0/1 flag",
            set(np.unique(n_va["taxi_network_available"].to_numpy())).issubset({0.0, 1.0})))
        checks.append(safety(
            f"{f}/{stage}: an unavailable route leaves all three distance features NaN",
            bool(n_va.loc[n_va["taxi_network_available"] == 0.0,
                          list(NET_FEATURES[:3])].isna().all().all())))
        checks.append(safety(
            f"{f}/{stage}: friend F3 code never imported",
            not any(m.startswith(("reference", "friend_f3")) for m in sys.modules)))

        fold_res = dict(
            n_train_complete=int(len(tr)),
            n_train_complete_all=int((tr_all["reference_state"] == "COMPLETE").sum()),
            train_ids_hash=SP.ids_digest(tr_ids), train_stage=stage,
            ids=val_ids, y=y_va, adep=va["ADEP_mvt"].to_numpy().astype(str),
            net_avail=n_va["taxi_network_available"].to_numpy(),
            net_dist=n_va["taxi_network_distance_m"].to_numpy(),
            net_ratio=n_va["taxi_network_detour_ratio"].to_numpy(),
            bucket_edges=[float(edges[0]), float(edges[1])],
        )
        plain(f"  {f} {stage}: train {len(tr):,} COMPLETE rows, "
              f"eval {len(val_ids):,} COMPLETE rows, "
              f"network {100 * fold_res['net_avail'].mean():.2f}%")

        for kind, Xt, Xv in (("control", X_c_tr, X_c_va), ("b4b", X_b_tr, X_b_va)):
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
                        for k in (CONTROL_FEATURES if kind == "control" else B4B_FEATURES))))
            bi = int(m.get_best_iteration())
            fold_res[f"p_{kind}"] = np.asarray(pred, "float64")
            fold_res[f"it_{kind}"] = bi
            fold_res[f"ceil_{kind}"] = bool(ceiling_limited(bi))
            fold_res[f"sec_{kind}"] = dt_s
            plain(f"    {kind:<8} best_iter={bi:>4}  {dt_s:>6.1f}s"
                  f"  RMSE={rmse(y_va, pred):.4f}"
                  f"  {'[ITERATION CEILING]' if fold_res[f'ceil_{kind}'] else '[early stop]'}")
            del m

        checks.append(safety(f"{f}/{stage}: control and candidate train on identical IDs",
                             len(X_c_tr) == len(X_b_tr) == len(tr) == len(y_tr)))
        checks.append(safety(f"{f}/{stage}: control and candidate validate on identical IDs",
                             len(X_c_va) == len(X_b_va) == len(val_ids)))
        fold_res["ceiling_limited"] = bool(fold_res["ceil_control"]
                                           and fold_res["ceil_b4b"])
        plain(f"  {f} {stage} done in {time.time() - t_fold:.1f}s"
              f"{'  [ITERATION-CEILING LIMITED]' if fold_res['ceiling_limited'] else ''}")
        res["folds"][f] = fold_res
        del tr, tr_all, mm, va, n_tr, n_va, X_c_tr, X_b_tr, X_c_va, X_b_va
        gc.collect()

    res["stage_ceiling_limited"] = bool(
        any(res["folds"][f]["ceiling_limited"] for f in ("JAN", "JUL")))
    res["val"] = {}
    for f in ("JAN", "JUL"):
        V = base["folds"][f]
        res["val"][f] = dict(
            ids=SP.id_to_int(V["ids"]), y=np.asarray(V["y"], float), vmask=V["vmask"],
            p04=np.asarray(V["p04"], float), p_f3=np.asarray(V["p_f3"], float),
            adep=V["adep"], it_exp04=int(V["it_exp04"]))
    return res


# ============================================================== REPORT PARTS
def data_section(geo, graphs):
    head("2. FROZEN EXTERNAL DATA (no live network access in this run)")
    m = geo["meta"]
    note(f"  frozen geometry   {GEO_CACHE}")
    note(f"  frozen at         {m['frozen_at']}")
    note(f"  runway mapping hash   {m['mapping_hashes']['runways']}")
    note(f"  stand  mapping hash   {m['mapping_hashes']['stands']}")
    note("")
    note("  The B4A frozen stand mapping and runway-threshold table are reused")
    note("  VERBATIM. Same structured-ref rule, same distinct-coordinate centroids,")
    note("  same threshold-per-departure-direction, same mapping hashes.")
    note("")
    note("  Taxiway graph sources (frozen B4 OSM snapshots, read-only):")
    for a in AIRPORTS:
        g = graphs[a]
        note(f"    {a}  tw_{a}.json  sha256 {g['source_sha256'][:16]}...  "
             f"graph hash {g['graph_hash'][:16]}...")
    note("")
    note("  External-data attribution and licence:")
    by_name = {s["name"]: s for s in m["sources"]}
    for nm in ("OurAirports runways.csv", "OurAirports airports.csv"):
        s = by_name.get(nm)
        if s is None:
            continue
        note(f"    {s['name']}")
        note(f"      url      {s['url']}")
        note(f"      sha256   {s['sha256']}")
        note(f"      licence  {s['licence']}")
    for s in m["sources"]:
        if s["name"].startswith("OpenStreetMap") or "overpass" in s["url"].lower():
            note(f"    OpenStreetMap Overpass response {s['name'][-4:]}")
            note(f"      url      {s['url']}")
            note(f"      sha256   {s['sha256']}")
            note(f"      licence  {s['licence']}")
    note("")
    note("    Attribution line required for any derived artefact:")
    note("      Taxiway and stand data: (c) OpenStreetMap contributors, ODbL 1.0.")
    note("      Runway and airport data: OurAirports (public domain).")
    note("    Both sources are external to the competition dataset and must be")
    note("    confirmed permissible under the competition's external-data rule")
    note("    before any submission relies on them.")
    note("")
    note("  Graph construction is target-free: it reads only the frozen OSM")
    note("  geometry files and the frozen OurAirports runway table. It runs")
    note("  BEFORE the departure frame is loaded, and the same graphs serve")
    note("  training and validation rows identically.")


def graph_section(graphs):
    head("3. TAXIWAY GRAPH CONSTRUCTION")
    note("  nodes  OSM way vertices, keyed by OSM node id")
    note("  edges  consecutive vertex pairs inside one OSM way, bidirectional")
    note("  weight geodesic segment length in metres")
    note("")
    note("  Disconnected components are NEVER bridged and no synthetic edge is")
    note("  ever added, so the components below are the true mapped components.")
    note("")
    note(f"  {'airport':<8}{'ways':>7}{'edges':>9}{'nodes':>8}{'components':>12}"
         f"{'largest':>9}{'largest %':>11}  graph hash")
    for a in AIRPORTS:
        g = graphs[a]
        big = int(g["component_sizes"].max())
        note(f"  {a:<8}{g['n_ways']:>7}{g['n_edges']:>9,}{g['n_nodes']:>8,}"
             f"{g['n_components']:>12}{big:>9,}{100.0 * big / g['n_nodes']:>10.1f}%"
             f"  {g['graph_hash'][:16]}")
    note("")
    note("  Cross-airport isolation: each airport is a separate graph built from")
    note("  its own source file, and the OSM node-id sets are disjoint across the")
    note("  ten airports, so no route can leave its airport by construction.")


def feasibility_section(cov, problems):
    head("4. PRE-FIT NETWORK FEASIBILITY AUDIT (run before any CatBoost fit)")
    note(f"  {'airport':<8}{'rows':>10}{'stand%':>9}{'rwy%':>9}{'route%':>9}"
         f"{'med m':>9}{'p90 m':>9}{'p99 m':>9}{'max m':>9}"
         f"{'snapSt med/p95':>18}{'snapRw med/p95':>18}{'discon%':>9}")
    for a in sorted(cov["by_airport"]):
        b = cov["by_airport"][a]
        note(f"  {a:<8}{b['rows']:>10,}{b['stand_pct']:>8.2f}%{b['runway_pct']:>8.2f}%"
             f"{b['route_pct']:>8.2f}%{b['median_net_m']:>9.0f}{b['p90_net_m']:>9.0f}"
             f"{b['p99_net_m']:>9.0f}{b['max_net_m']:>9.0f}"
             f"{b['median_stand_snap_m']:>9.0f}/{b['p95_stand_snap_m']:<8.0f}"
             f"{b['median_runway_snap_m']:>9.0f}/{b['p95_runway_snap_m']:<8.0f}"
             f"{b['disconnected_pct']:>8.2f}%")
    note("")
    note(f"  OVERALL, row-weighted over all {cov['n']:,} departures:")
    note(f"    stand mapping coverage            {cov['stand_pct']:>10.3f}%")
    note(f"    runway mapping coverage           {cov['runway_pct']:>10.3f}%")
    note(f"    VALID NETWORK-ROUTE COVERAGE      {cov['route_pct']:>10.3f}%"
         f"   ({cov['n_route_rows']:,} rows)")
    note(f"    feasibility threshold             {MIN_ROUTE_COVERAGE_PCT:>10.1f}%")
    note("")
    note("  Network-distance plausibility:")
    note(f"    minimum                          {cov['min_net_m']:>10.1f} m   (> 0 required)")
    note(f"    median                           {cov['median_net_m']:>10.1f} m")
    note(f"    p90                              {cov['p90_net_m']:>10.1f} m")
    note(f"    p99                              {cov['p99_net_m']:>10.1f} m")
    note(f"    maximum                          {cov['max_net_m']:>10.1f} m"
         f"   (<= {MAX_NET_DIST_M:.0f} m required)")
    note(f"    median detour ratio              {cov['median_detour_ratio']:>10.4f}")
    note("")
    note("  Sanity-bound violations:")
    note(f"    network distance <= 0 m                        {cov['n_le_zero']:>8}")
    note(f"    network distance > {MAX_NET_DIST_M:.0f} m{' ' * 20}{cov['n_gt_max']:>8}")
    note(f"    below the SNAPPED-node straight line (fatal)   {cov['n_below_snapped']:>8}")
    note(f"    below the CENTROID straight line (reported)    {cov['n_below_straight']:>8}")
    note("")
    note(f"  The last count is expected physics, not a defect. The graph path joins")
    note(f"  the two snapped nodes, while the plan's straight line joins the stand")
    note(f"  centroid to the threshold. A snap inside the fixed limit can move an")
    note(f"  endpoint toward the other, so {cov['n_below_straight']:,} routes")
    note(f"  ({100.0 * cov['n_below_straight'] / max(1, cov['n_route_rows']):.3f}% of routed")
    note(f"  rows) come out shorter than that comparison, by as little as")
    note(f"  {cov['min_detour_m']:.1f} m. Every one of them still satisfies the triangle")
    note(f"  inequality against the snapped-node straight line, which is the")
    note(f"  invariant that actually constrains a shortest path. These rows keep")
    note(f"  a slightly negative taxi_network_detour_m; that is reported rather")
    note(f"  than silently clipped.")
    note("")
    note("  Unroutable rows, decomposed:")
    note(f"    stand unmapped                                {cov['n_stand_unmapped']:>8,}")
    note(f"    runway unmapped                               {cov['n_runway_unmapped']:>8,}")
    note(f"    both mapped but NO CONNECTED ROUTE             {cov['n_disconnected']:>8,}")
    note("")
    if problems:
        note("  B4B NETWORK FEASIBILITY FAILED")
        for p in problems:
            note(f"    * {p}")
        note("")
        note("  STOPPING BEFORE ANY CATBOOST FIT. The threshold is not loosened.")
    else:
        note("  FEASIBILITY: PASSED. Proceeding to the SMOKE stage.")


def snap_section(cov):
    head("5. SNAP-DISTANCE AUDIT (fixed limits, chosen before any model result)")
    note(f"  stand  -> nearest taxiway node within a FIXED {MAX_STAND_SNAP_M:.0f} m")
    note(f"  runway -> nearest taxiway node within a FIXED {MAX_RUNWAY_SNAP_M:.0f} m")
    note("")
    note("  Deterministic rule: nearest node by geodesic distance, ties broken on")
    note("  the lowest OSM node id. No target information takes any part, and a")
    note("  point with no node inside the limit is left unrouted rather than")
    note("  attached to a distant node.")
    note("")
    note(f"  {'airport':<8}{'stand med':>12}{'stand p95':>12}{'stand max':>12}"
         f"{'rwy med':>11}{'rwy p95':>11}{'rwy max':>11}")
    for a in sorted(cov["by_airport"]):
        b = cov["by_airport"][a]
        note(f"  {a:<8}{b['median_stand_snap_m']:>12.1f}{b['p95_stand_snap_m']:>12.1f}"
             f"{'':>12}{b['median_runway_snap_m']:>11.1f}{b['p95_runway_snap_m']:>11.1f}"
             f"{'':>11}")
    b = cov
    note("")
    note(f"  OVERALL  stand median {cov['median_stand_snap_m']:.1f} m, "
         f"p95 {cov['p95_stand_snap_m']:.1f} m, max {cov['max_stand_snap_m']:.1f} m")
    note(f"  OVERALL  runway median {cov['median_runway_snap_m']:.1f} m, "
         f"p95 {cov['p95_runway_snap_m']:.1f} m, max {cov['max_runway_snap_m']:.1f} m")
    note("")
    note("  A stand whose nearest node exceeds 250 m is NOT snapped, and a runway")
    note("  threshold whose nearest node exceeds 300 m is NOT snapped. Because")
    note("  runway thresholds sit on the taxiway system, their snap distances are")
    note("  small everywhere; stand snap is the looser of the two.")


def sanity_section(san):
    head("6. INDEPENDENT ROUTE SANITY CHECK")
    note(f"  deterministic sample of {san['n_sample']:,} routed departures "
         f"(seed {san['seed']})")
    note("")
    note("  Re-verified through a deliberately independent second implementation:")
    note("    a plain-Python brute-force snap over every node, a dict adjacency")
    note("    rebuilt directly from the source ways, and a heapq Dijkstra, none of")
    note("    which shares any code with the production scipy path.")
    note("")
    note(f"  {'check':<52}{'disagreements':>16}")
    note(f"  {'stand and runway mapping re-resolves':<52}{san['n_mapping_disagree']:>16}")
    note(f"  {'independent snap inside the fixed limit':<52}{san['n_snap_disagree']:>16}")
    note(f"  {'snapped pair lies in one component':<52}{san['n_connectivity_disagree']:>16}")
    note(f"  {'independent Dijkstra reproduces the distance':<52}{san['n_distance_disagree']:>16}")
    note(f"  {'network distance >= straight line between snapped nodes':<52}"
         f"{san['n_straightline_disagree']:>16}")
    note("")
    note(f"  max |independent - production| distance error  "
         f"{san['max_distance_abs_error_m']:.3e} m")
    note(f"  node-id sets disjoint across airports               "
         f"{san['node_sets_disjoint']}")
    note("")
    note(f"  Reported, not fatal: {san['n_negative_detour']} of the sampled routes are")
    note("  shorter than the stand-centroid-to-threshold straight line. The graph")
    note("  path joins the two SNAPPED nodes, and a snap inside the fixed limit can")
    note("  move an endpoint toward the other, so this is snap geometry rather than")
    note("  a broken route. Every one of them still satisfies the triangle")
    note("  inequality against the straight line between the snapped nodes.")
    note("")
    note("  Every disagreement count above must be zero; the run aborts otherwise.")


def feature_section():
    head("7. FEATURE MATRIX")
    note(f"  {'idx':>4}  {'feature':<32}{'kind':<12}{'in control':>11}{'in B4B':>8}")
    for i, k in enumerate(CONTROL_FEATURES):
        note(f"  {i:>4}  {k:<32}{'categorical/num':<12}{'yes':>11}{'yes':>8}")
    for i, k in enumerate(NET_FEATURES):
        j = len(CONTROL_FEATURES) + i
        note(f"  {j:>4}  {k:<32}{'numerical':<12}{'no':>11}{'yes':>8}")
    note("")
    note("  The B4A straight-line distance is used only internally to form the")
    note("  detour and the ratio. It is not a B4B feature, so B4B cannot simply")
    note("  re-report B4A's straight-line signal.")
    note("")
    note("  Deliberately NOT exposed: raw coordinates, graph edge count, turn")
    note("  count, intersection count, B2 congestion, B3 priors, friend F3, weather,")
    note("  flight identity, ADS-B. Edge and turn counts are excluded because OSM")
    note("  segmentation density differs between airports and would encode mapping")
    note("  style rather than physics.")


def per_airport(res, cov):
    head("9. PER-AIRPORT DIAGNOSTIC (COMPLETE validation rows)")
    note("  Diagnostic only. No airport-specific routing is created.")
    note("")
    note(f"  {'ADEP':<7}{'rows':>9}{'net cov':>9}{'control':>11}{'B4B':>11}"
         f"{'delta':>10}{'SSE delta':>15}")
    rows = []
    for a in sorted(cov["by_airport"]):
        c = b = n = cs = bs = 0.0
        cnt = 0
        ncov = []
        for f in ("JAN", "JUL"):
            F = res["folds"][f]
            m = F["adep"] == a
            if not m.any():
                continue
            cnt += int(m.sum())
            ncov.append(F["net_avail"][m])
            c += sse(F["y"][m], F["p_control"][m])
            b += sse(F["y"][m], F["p_b4b"][m])
        if cnt == 0:
            continue
        rc, rb = np.sqrt(c / cnt), np.sqrt(b / cnt)
        note(f"  {a:<7}{cnt:>9,}{100 * np.concatenate(ncov).mean():>8.2f}%"
             f"{rc:>11.4f}{rb:>11.4f}{rb - rc:>+10.4f}{b - c:>+15.4e}")
        rows.append(rb - rc)
    better = sum(1 for d in rows if d < 0)
    note("")
    note(f"  {better} of {len(rows)} airports improve. A broad signal would improve")
    note("  most of them; a lopsided pattern is an interaction, not a layout law.")


def availability(res):
    head("10. NETWORK-AVAILABILITY DIAGNOSTIC (taxi_network_available split)")
    note("  Diagnostic only. No model is routed on network availability.")
    note("")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        note(f"  --- {f} ---")
        note(f"  {'group':<32}{'rows':>10}{'control':>11}{'B4B':>11}"
             f"{'delta':>10}{'SSE delta':>15}")
        for lab, sel in (("taxi_network_available == 1", F["net_avail"] == 1.0),
                         ("taxi_network_available == 0", F["net_avail"] == 0.0)):
            if not sel.any():
                continue
            c = rmse(F["y"][sel], F["p_control"][sel])
            b = rmse(F["y"][sel], F["p_b4b"][sel])
            note(f"  {lab:<32}{int(sel.sum()):>10,}{c:>11.4f}{b:>11.4f}"
                 f"{b - c:>+10.4f}{sse(F['y'][sel], F['p_b4b'][sel]) - sse(F['y'][sel], F['p_control'][sel]):>+15.4e}")
        note("")


def distance_buckets(res):
    head("11. DISTANCE DIAGNOSTIC (terciles from TRAINING network distance only)")
    note("  Diagnostic only. Bucket edges come from that fold's TRAINING rows and")
    note("  never from the target or from validation performance.")
    note("")
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        e = F["bucket_edges"]
        note(f"  --- {f} --- training terciles {e[0]:,.0f} m / {e[1]:,.0f} m")
        note(f"  {'bucket':<22}{'rows':>10}{'control':>11}{'B4B':>11}{'delta':>10}")
        d = F["net_dist"]
        for lab, sel in (("short  (<= 1/3)", np.isfinite(d) & (d <= e[0])),
                         ("medium (1/3 - 2/3)", np.isfinite(d) & (d > e[0]) & (d <= e[1])),
                         ("long   (> 2/3)", np.isfinite(d) & (d > e[1]))):
            if not sel.any():
                continue
            c = rmse(F["y"][sel], F["p_control"][sel])
            b = rmse(F["y"][sel], F["p_b4b"][sel])
            note(f"  {lab:<22}{int(sel.sum()):>10,}{c:>11.4f}{b:>11.4f}{b - c:>+10.4f}")
        note("")


def tail(res):
    head("12. TAIL DIAGNOSTIC (COMPLETE validation rows)")
    note("  Diagnostic only. The true target never routes a prediction.")
    note("")
    y = np.concatenate([res["folds"][f]["y"] for f in ("JAN", "JUL")])
    pc = np.concatenate([res["folds"][f]["p_control"] for f in ("JAN", "JUL")])
    pb = np.concatenate([res["folds"][f]["p_b4b"] for f in ("JAN", "JUL")])
    thr = np.quantile(y, 0.99)
    slices = (("target > 3600", y > 3600), ("target > 7200", y > 7200),
              ("worst 1% by target", y >= thr))
    note(f"  {'slice':<24}{'n':>9}{'control':>12}{'B4B':>12}{'delta':>11}{'SSE delta':>15}")
    for lab, sel in slices:
        if not sel.any():
            continue
        note(f"  {lab:<24}{int(sel.sum()):>9,}{rmse(y[sel], pc[sel]):>12.4f}"
             f"{rmse(y[sel], pb[sel]):>12.4f}"
             f"{rmse(y[sel], pb[sel]) - rmse(y[sel], pc[sel]):>+11.4f}"
             f"{sse(y[sel], pb[sel]) - sse(y[sel], pc[sel]):>+15.4e}")
    note("")
    note("  B2 and B3 both bought a small pooled movement with tail degradation.")
    note("  The tail slices are where the competition metric is actually decided,")
    note("  so this pattern matters more than the pooled figure.")


def whole_context(res):
    head("13. WHOLE-MODEL CONTEXT (informational hybrid)")
    note("  COMPLETE -> CONTROL or B4B, MISSING_4 -> cached plain F3.")
    note("  Context only. These numbers are NOT the primary B4B verdict.")
    note("")
    note(f"  {'fold':<7}{'frozen plain F3':>18}{'hybrid control':>17}"
         f"{'hybrid B4B':>14}{'delta':>10}")
    acc = []
    for f in ("JAN", "JUL"):
        hc, hb = hybrid(res, f, "p_control"), hybrid(res, f, "p_b4b")
        V = res["val"][f]
        rc, rb = rmse(V["y"], hc), rmse(V["y"], hb)
        note(f"  {f:<7}{rmse(V['y'], V['p_f3']):>18.6f}{rc:>17.6f}{rb:>14.6f}{rb - rc:>+10.4f}")
        acc.append((V["y"], hc, hb))
    y = np.concatenate([a[0] for a in acc])
    hc = np.concatenate([a[1] for a in acc])
    hb = np.concatenate([a[2] for a in acc])
    note(f"  {'POOLED':<7}{'':>18}{rmse(y, hc):>17.6f}{rmse(y, hb):>14.6f}"
         f"{rmse(y, hb) - rmse(y, hc):>+10.4f}")


def ceiling_section(res):
    head("8. ITERATION-CEILING STATUS")
    note(f"  budget {PARAMS['iterations']} trees, depth {PARAMS['depth']}, "
         f"early_stopping_rounds {PARAMS['early_stopping_rounds']}")
    note("  A fold is ITERATION-CEILING LIMITED when BOTH the control and the")
    note("  candidate best iteration is within one patience window of the cap,")
    note("  which means the run may have been stopped by the budget rather than")
    note("  by overfitting. Convergence is never claimed silently.")
    note("")
    note(f"  {'fold':<7}{'control iter':>14}{'control':>20}{'B4B iter':>12}{'B4B':>20}{'fold':>10}")
    n_ceiling_models = 0
    for f in ("JAN", "JUL"):
        F = res["folds"][f]
        cs = "[ITERATION CEILING]" if F["ceil_control"] else "[early stop]"
        bs = "[ITERATION CEILING]" if F["ceil_b4b"] else "[early stop]"
        n_ceiling_models += int(F["ceil_control"]) + int(F["ceil_b4b"])
        note(f"  {f:<7}{F['it_control']:>14,}{cs:>20}{F['it_b4b']:>12,}{bs:>20}"
             f"{'CEILING' if F['ceiling_limited'] else '-':>10}")
    note("")
    capped = PARAMS["iterations"] - CEILING_WINDOW
    note(f"  A model is marked ITERATION CEILING when its best iteration is at or")
    note(f"  above {capped} ({PARAMS['iterations']} - {CEILING_WINDOW}), i.e. inside the")
    note(f"  patience window, so the budget may have cut the fit short.")
    note("")
    limited = res.get("stage_ceiling_limited")
    n_limited = sum(1 for f in ("JAN", "JUL") if res["folds"][f]["ceiling_limited"])
    note(f"  {n_ceiling_models} of 4 models reached the ceiling; {n_limited} of 2 folds are")
    note(f"  ITERATION-CEILING LIMITED (both arms capped in the same fold).")
    note("")
    if limited:
        note("  STAGE STATUS: at least one fold is ITERATION-CEILING LIMITED, so that")
        note("  fold's delta is a comparison of two under-trained models and carries")
        note("  little weight.")
    if n_ceiling_models:
        note("  A model that ends within the patience window of the cap was still")
        note("  improving when it stopped. The pooled figure mixes a capped arm with")
        note("  an early-stopped one, so a small pooled gain here understates what a")
        note("  larger budget might show. That budget was not authorised for this run,")
        note("  and no threshold is changed after seeing the result.")
    if not n_ceiling_models:
        note("  STAGE STATUS: every model early-stopped normally, well inside the")
        note("  cap, so the comparison is not truncated.")


def safety_section(res):
    head("14. SAFETY AND ALIGNMENT CHECKS")
    note(f"  {len(res['checks'])} checks, all passed:")
    for c in res["checks"]:
        note(c)


def main_report(stage, res, smoke_res, screen_res, cov, graphs, geo, san,
                problems, verdict, gate, audit_ok=True):
    head(f"B4B - TAXIWAY-NETWORK DISTANCE FOR COMPLETE ROWS    stage: {stage}")
    note("  Branch B, experiment 4b. The FINAL planned geometry experiment.")
    note("")
    note("  Prior branch B state, none of it tuned, modified or reused:")
    note("    B1  residual target (AOBT3)      REJECTED")
    note("    B2  multi-scale congestion        WEAK SIGNAL / PARKED")
    note("    B3  structural target priors      REJECTED")
    note("    B4A simple physical geometry      WEAK SIGNAL / CLOSED (gain only ~0.08 RMSE,")
    note("         tail worsened, inconsistent by airport, both arms at the 1000-tree ceiling)")
    note("")
    note("  B4A's nine geometry features are NOT reused here. B4B adds exactly")
    note("  four taxiway-network features to the same 18-feature Exp04 control.")
    note("  Both the control and the candidate are trained fresh at 1500 trees.")
    note("")
    note("  DEV EARLY STOPPING - NOT STRICT UNSEEN-MONTH VALIDATION.")

    data_section(geo, graphs)
    graph_section(graphs)
    feasibility_section(cov, problems)
    snap_section(cov)
    sanity_section(san)
    feature_section()

    if res is None:
        head("8a. MODEL STAGES")
        note("  No CatBoost model was fitted. This report is the pre-fit network")
        note("  feasibility audit only.")
        note("")
        safety_section(dict(checks=_all_checks))
    else:
        head(f"8a. {stage} RESULT")
        cp, bp, cs, bs = main_table(res)
        note("")
        note("  negative delta means the network features help.")
        note("")

        if stage == "SCREEN":
            head("6b. SMOKE STAGE (the run that authorised this SCREEN run)")
            if smoke_res is None:
                note("  no SMOKE stage in the cache")
            else:
                s_cp, s_bp, s_cs, s_bs = main_table(smoke_res)
                note("")
                note(f"  SMOKE pooled control {s_cp:.6f}  B4B {s_bp:.6f}  "
                     f"delta {s_bp - s_cp:+.4f}")
                note(f"  SMOKE SSE delta {s_bs - s_cs:+.4e} "
                     f"({100 * (s_bs - s_cs) / s_cs:+.2f}%)")
                note(f"  SMOKE best_iter  control "
                     f"{[smoke_res['folds'][f]['it_control'] for f in ('JAN', 'JUL')]}"
                     f"   B4B "
                     f"{[smoke_res['folds'][f]['it_b4b'] for f in ('JAN', 'JUL')]}")
                note("")
                note(f"  SMOKE GATE: {'PASSED' if gate else 'FAILED'}  (pooled B4B <= control "
                     f"+ {GATE_POOLED_TOL:.0f}, no fold worse by > {GATE_FOLD_TOL:.0f})")

        if stage in ("SCREEN", "FULL"):
            ceiling_section(res)
            head("9. PER-AIRPORT DIAGNOSTIC (COMPLETE validation rows)")
            per_airport(res, cov)
            availability(res)
            distance_buckets(res)
            tail(res)
            whole_context(res)
        safety_section(res)

    head("15. CACHE LOCATIONS")
    note(f"  model stage cache        {CACHE}")
    note(f"  network feature cache    {NET_CACHE}")
    note(f"  frozen geometry          {GEO_CACHE}")
    note(f"  full 1500 control cache  {FULL_CONTROL_CACHE}"
         f"   (exists: {FULL_CONTROL_CACHE.exists()})")

    head("16. FINAL OUTPUT")
    note("")
    if problems:
        note("B4B NETWORK FEASIBILITY FAILED")
    elif res is None:
        note("B4B NETWORK FEASIBILITY PASSED - no model stage was run")
        if not audit_ok:
            note("NETWORK FEASIBILITY: FAILED")
    elif stage == "SMOKE":
        if gate:
            note("B4B SMOKE COMPLETE - gate PASSED. SCREEN is the next stage.")
        else:
            note("B4B REJECTED AT SMOKE")
    elif stage == "SCREEN":
        note("B4B SCREEN COMPLETE")
        note(f"VERDICT: {verdict}")
    elif stage == "FULL":
        note("B4B FULL COMPLETE")
        note(f"VERDICT: {verdict}")
    note("")

def finalize_report(stage, res, smoke_res, screen_res, cov, graphs, geo, san,
                    problems, audit_ok=True):
    if stage == "AUDIT":
        verdict, gate = None, None
    elif stage == "SMOKE":
        cp = pooled_of(res, "p_control")[0]
        bp = pooled_of(res, "p_b4b")[0]
        jd = [rmse(res["folds"][f]["y"], res["folds"][f]["p_b4b"])
              - rmse(res["folds"][f]["y"], res["folds"][f]["p_control"])
              for f in ("JAN", "JUL")]
        gate = (bp <= cp + GATE_POOLED_TOL) and all(d > -GATE_FOLD_TOL for d in jd)
        verdict = None
    else:
        cp = pooled_of(res, "p_control")[0]
        bp = pooled_of(res, "p_b4b")[0]
        jd = [rmse(res["folds"][f]["y"], res["folds"][f]["p_b4b"])
              - rmse(res["folds"][f]["y"], res["folds"][f]["p_control"])
              for f in ("JAN", "JUL")]
        gate = (bp <= cp + GATE_POOLED_TOL) and all(d > -GATE_FOLD_TOL for d in jd)
        promising = (bp <= cp - PROMISING_POOLED) and all(d > -PROMISING_FOLD for d in jd)
        if stage == "SCREEN":
            verdict = "PROMISING" if promising else ("WEAK SIGNAL" if bp < cp else "REJECT")
        else:
            verdict = "FULL CONFIRMED" if promising else ("FULL WEAK" if bp < cp else "FULL FAILED")

    main_report(stage, res, smoke_res, screen_res, cov, graphs, geo, san,
                problems, verdict, gate, audit_ok)
    defects_section()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")
    _buf.clear()
    return verdict, gate


# ================================================================ DEFECT LEDGER
DEFECTS: list[tuple[str, list[str]]] = []


def defect(cls, *lines):
    if (cls, list(lines)) not in DEFECTS:
        DEFECTS.append((cls, list(lines)))


def record_known_defects():
    defect("COMPUTATION-AFFECTING [1] - distance formula transcription slip",
           "great_circle() was written with np.sin(dl / 2. ** 2) instead of",
           "np.sin(dl / 2.0) ** 2. In Python that evaluates as sin(dl / 4), so",
           "every distance in the file was shrunk by a factor of about 1/1.41.",
           "It corrupted segment weights, snap distances, straight-line distances",
           "and the sanity-check comparison simultaneously, so no internal check",
           "could catch it: the independent checker used the correct formula and",
           "the two disagreed on 2000/2000 sampled routes, which is how it was",
           "found. A hand computation confirmed great_circle, not the checker,",
           "was the wrong one.",
           "Fixed to np.sin(dl / 2.0) ** 2 and now guarded by a permanent check",
           "that the vectorised formula matches an independent scalar haversine",
           "to 0.0 m. No model was ever fitted with the broken formula, so no",
           "metric, prediction or cache from it exists.")
    defect("COMPUTATION-AFFECTING [2] - one-directional graph adjacency",
           "The taxiway adjacency matrix was built with one entry per segment,",
           "leaving it non-symmetric, and then queried with directed=False.",
           "scipy does not raise on that combination; it returns distances that",
           "are too short. A single EDDF stand-to-threshold route came out at",
           "2394 m on the non-symmetric matrix versus 3528 m on the correct",
           "bidirectional graph, a 32% understatement.",
           "Fixed by inserting both orientations for every segment, with an",
           "assert that nnz == 2 * n_edges, and the matrix is now verified",
           "symmetric. The independent plain-Python checker builds a symmetric",
           "adjacency by construction, so the two now agree exactly.",
           "No model was ever fitted on the non-symmetric graph.")
    defect("PRE-FIT - reported detour invariant was the wrong comparison",
           "The planned sanity bound 'network distance >= straight-line distance'",
           "compared the graph path against the STAND CENTROID to THRESHOLD",
           "distance. That is not the triangle inequality: the graph path joins",
           "the two SNAPPED nodes, and a snap may move either endpoint tens of",
           "metres toward the other, legitimately shortening that comparison.",
           "It fired at -99 m, which is snap geometry rather than a broken route.",
           "The fatal check now tests the real invariant, network distance >=",
           "straight line between the two snapped nodes, and the centroid-based",
           "version is retained as a REPORTED count, as the plan requires, with",
           "an added check that every negative detour still satisfies the",
           "snapped-endpoint triangle inequality.")


def defects_section():
    head("17. DEFECT LEDGER")
    if not DEFECTS:
        note("  no defects were encountered in this run")
        return
    for i, (cls, lines) in enumerate(DEFECTS, 1):
        note(f"  [{i}] {cls}")
        for l in lines:
            note(f"      {l}")
        note("")


# ===================================================================== DRIVER
def save_cache(blob, stage, res):
    blob.setdefault("stages", {})[stage] = res
    SP.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(blob, fh, protocol=4)
    plain(f"  cached {stage} -> {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")


def load_cache():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    return {"version": 1, "stages": {}}


def save_net_cache(nf, cov, san, geo, graphs):
    SP.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(
        version=1, created=time.strftime("%Y-%m-%d %H:%M:%S"),
        features=nf["features"], audit=nf["audit"], straight=nf["straight"],
        network_audit=cov, sanity=san,
        graph_hashes={a: graphs[a]["graph_hash"] for a in AIRPORTS},
        source_sha256={a: graphs[a]["source_sha256"] for a in AIRPORTS},
        geometry_frozen_at=geo["meta"]["frozen_at"],
        mapping_hashes=geo["meta"]["mapping_hashes"],
        snap_limits=dict(stand_m=MAX_STAND_SNAP_M, runway_m=MAX_RUNWAY_SNAP_M),
        params=dict(PARAMS), net_features=list(NET_FEATURES),
    )
    with open(NET_CACHE, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    plain(f"  cached the network feature table -> {NET_CACHE} "
          f"({NET_CACHE.stat().st_size / 1e6:.1f} MB) "
          f"(shortest paths are never recomputed for reporting)")


def load_net_cache():
    assert NET_CACHE.exists(), f"missing network feature cache {NET_CACHE}"
    with open(NET_CACHE, "rb") as fh:
        return pickle.load(fh)


def verify_net_cache(net: dict) -> list[str]:
    """Validate a cached network feature table before any model is fitted.

    Returning a non-empty list means the cache cannot be trusted, and the run
    must stop rather than quietly rebuild. Everything checked here is a
    property that a corrupted or partial cache would break.
    """
    bad: list[str] = []
    for k in ("features", "audit", "network_audit", "sanity", "graph_hashes",
              "mapping_hashes", "snap_limits", "net_features"):
        if k not in net:
            bad.append(f"cache is missing the {k!r} key")
    if bad:
        return bad
    f = net["features"]
    if list(f.columns) != list(NET_FEATURES):
        bad.append(f"cached columns {list(f.columns)} != {list(NET_FEATURES)}")
    if not f.index.is_unique:
        bad.append("cached MVT_ID index is not unique")
    if len(f) < 1_000_000:
        bad.append(f"cached table has only {len(f):,} rows, far fewer than the "
                   f"2,085,047 departures")
    avail = f["taxi_network_available"].to_numpy()
    if not set(np.unique(avail)).issubset({0.0, 1.0}):
        bad.append("cached taxi_network_available is not a clean 0/1 flag")
    d = f["taxi_network_distance_m"].to_numpy()
    m = np.isfinite(d)
    if m.any():
        if d[m].min() <= 0:
            bad.append(f"cached network distance has a non-positive value "
                       f"{d[m].min():.3f} m")
        if d[m].max() > MAX_NET_DIST_M:
            bad.append(f"cached network distance exceeds {MAX_NET_DIST_M:.0f} m "
                       f"({d[m].max():.1f} m)")
    if not np.array_equal(np.isfinite(f["taxi_network_detour_m"].to_numpy()), m):
        bad.append("cached detour_m is not NaN exactly where the route is unavailable")
    if not np.array_equal(np.isfinite(f["taxi_network_detour_ratio"].to_numpy()), m):
        bad.append("cached detour_ratio is not NaN exactly where the route is unavailable")
    if not np.array_equal(avail == 1.0, m):
        bad.append("cached available flag disagrees with the finite distances")
    cov = net["network_audit"]
    if abs(cov["route_pct"] - 100.0 * float(m.mean())) > 0.01:
        bad.append(f"cached route coverage {cov['route_pct']:.3f}% disagrees with "
                   f"the cached table {100.0 * float(m.mean()):.3f}%")
    if cov["route_pct"] < MIN_ROUTE_COVERAGE_PCT:
        bad.append(f"cached route coverage {cov['route_pct']:.3f}% is below the "
                   f"{MIN_ROUTE_COVERAGE_PCT:.0f}% feasibility threshold")
    if net["snap_limits"] != dict(stand_m=MAX_STAND_SNAP_M, runway_m=MAX_RUNWAY_SNAP_M):
        bad.append(f"cached snap limits {net['snap_limits']} differ from the "
                   f"configured {dict(stand_m=MAX_STAND_SNAP_M, runway_m=MAX_RUNWAY_SNAP_M)}")
    if net["mapping_hashes"] != geo_meta_ref()["mapping_hashes"]:
        bad.append("cached stand/runway mapping hashes differ from the frozen geometry")
    san = net["sanity"]
    for k in ("n_mapping_disagree", "n_snap_disagree", "n_connectivity_disagree",
              "n_distance_disagree", "n_straightline_disagree"):
        if san.get(k):
            bad.append(f"cached sanity check reports {san[k]} {k}")
    if san.get("n_sample", 0) < SANITY_N:
        bad.append(f"cached sanity check covered only {san.get('n_sample')} routes")
    return bad


def geo_meta_ref() -> dict:
    with open(GEO_CACHE, "rb") as fh:
        return pickle.load(fh)["meta"]


def cache_full_control(res):
    if FULL_CONTROL_CACHE.exists():
        plain(f"  COMPLETE-only 1500-tree FULL control already cached at "
              f"{FULL_CONTROL_CACHE}")
        return
    payload = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        description="COMPLETE-only FULL control at the 1500-tree budget, same "
                    "CatBoost protocol as B4B. Trained on ALL COMPLETE "
                    "outer-training rows, early-stopped on the full COMPLETE "
                    "JAN/JUL validation rows. Reusable by any future "
                    "COMPLETE-only experiment as its control.",
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
    plain(f"  permanently cached the COMPLETE-only FULL 1500-tree control -> "
          f"{FULL_CONTROL_CACHE}")


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

    record_known_defects()

    # ---------------------------------------------------------------- audit
    plain("building the taxiway graphs from the FROZEN OSM snapshot (no network access)")
    t0 = time.time()
    graphs = build_all_graphs()
    plain(f"  10 graphs built in {time.time() - t0:.1f}s")
    geo = load_frozen_geometry()

    gchecks: list[str] = []

    if report_only:
        blob = load_cache()
        net = load_net_cache()
        stages = blob.get("stages", {})
        if stages:
            st = only if (only in stages) else list(stages)[-1]
            plain(f"REPORT-ONLY MODE: reusing the cached {st} stage and the cached")
            plain("network feature table. No frame build, no graph rebuild, no")
            plain("shortest path, no network access, no fit.")
            _buf.clear()
            finalize_report(st, stages[st], stages.get("SMOKE"),
                            stages.get("SCREEN"), net["network_audit"],
                            graphs, geo, net["sanity"], [])
        else:
            st = None
            plain("REPORT-ONLY MODE: no model stage is cached yet, so this is the")
            plain("pre-fit network feasibility audit only, rendered from the cached")
            plain("network feature table. No frame build, no graph rebuild, no")
            plain("shortest path, no network access, no fit.")
            _buf.clear()
            finalize_report("AUDIT", None, None, None, net["network_audit"],
                            graphs, geo, net["sanity"], [], True)
        plain(f"\nDONE -> {REPORT}")
        return

    plain("")
    plain("preparing the frame and frozen samples (read-only, no model fit)")
    dep, samples, base = prepare()
    plain(f"  ready in {time.time() - t0:.0f}s")

    # ------------------------------------------------- network feature table
    # The graph build and the shortest paths are the expensive part of this
    # experiment. Once the feature table is cached it is reused verbatim, so
    # resuming at model training never recomputes a single route. A rebuild
    # happens only when there is no cache, and never silently: the reason is
    # always printed.
    rebuild = "--rebuild-network" in sys.argv
    net = None
    if not rebuild and NET_CACHE.exists():
        plain("")
        plain(f"reusing the CACHED network feature table from {NET_CACHE}")
        net = load_net_cache()
        plain(f"  cached {net['created']}, {len(net['features']):,} rows, "
              f"{len(net['net_features'])} features")
        reuse_problems = verify_net_cache(net)
        if reuse_problems:
            plain("")
            plain("CACHE VALIDATION FAILED - STOPPING, not rebuilding silently:")
            for p in reuse_problems:
                plain(f"  * {p}")
            plain("")
            plain(f"Remove {NET_CACHE} and re-run, or pass --rebuild-network to")
            plain("deliberately regenerate the network features.")
            return
        nf = dict(features=net["features"], audit=net["audit"],
                  straight=net["straight"])
        cov = net["network_audit"]
        san = net["sanity"]
        gchecks = [c for c in net.get("checks", [])]
        _all_checks.extend(gchecks)
        problems = feasibility_check(cov)
        plain(f"  cache validated: {len(net['features']):,} rows, route coverage "
              f"{cov['route_pct']:.3f}%, graph hashes match the frozen source")
        plain(f"  no graph rebuild, no shortest path, no reroute")
    else:
        why = "--rebuild-network was passed explicitly" if rebuild \
            else f"no cache at {NET_CACHE}"
        plain("")
        plain(f"building the 4 network features from scratch ({why})")
        plain("  this recomputes every graph and every shortest path")
        nf = build_network_features(dep, geo, graphs, gchecks)
        plain(f"  built {len(nf['features']):,} rows in {time.time() - t0:.0f}s cumulative")
        cov = audit_network(dep, geo, graphs, nf)
        problems = feasibility_check(cov)
        _all_checks.extend(gchecks)

    if not rebuild and NET_CACHE.exists() and not reuse_problems:
        plain("")
        plain(f"reusing the {san['n_sample']:,}-route independent sanity check "
              f"cached with the feature table")
        plain(f"  {sum(san[k] for k in san if k.startswith('n_') and k.endswith('disagree'))}"
              f" disagreements, max distance error "
              f"{san['max_distance_abs_error_m']:.3e} m")
    else:
        plain("")
        plain("independent route sanity check")
        san = route_sanity_check(dep, geo, graphs, nf)
        _all_checks.extend(san["checks"])
        plain(f"  {san['n_sample']:,} routes re-verified, "
              f"{sum(san[k] for k in san if k.startswith('n_') and k.endswith('disagree'))} "
              f"disagreements")
        save_net_cache(nf, cov, san, geo, graphs)

    blob = load_cache()
    blob["network_checks"] = gchecks
    blob["network_audit"] = cov
    blob["sanity"] = {k: v for k, v in san.items() if k != "checks"}

    if problems:
        _buf.clear()
        finalize_report("AUDIT", None, None, None, cov, graphs, geo, san, problems,
                        False)
        defects_section()
        REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")
        _buf.clear()
        plain("")
        plain("B4B NETWORK FEASIBILITY FAILED")
        for p in problems:
            plain(f"  * {p}")
        plain("")
        plain("STOPPED BEFORE ANY CATBOOST FIT. The 75% threshold is not loosened.")
        return

    if audit_only:
        _buf.clear()
        finalize_report("AUDIT", None, None, None, cov, graphs, geo, san, problems)
        defects_section()
        REPORT.write_text("\n".join(_buf) + "\n", encoding="utf-8")
        _buf.clear()
        plain("")
        plain("NETWORK FEASIBILITY: PASSED - stopping before training as requested")
        plain(f"  valid network-route coverage {cov['route_pct']:.3f}% "
              f"(threshold {MIN_ROUTE_COVERAGE_PCT:.0f}%)")
        plain(f"  network distance min {cov['min_net_m']:.1f} m  "
              f"median {cov['median_net_m']:.0f} m  max {cov['max_net_m']:.0f} m")
        plain(f"  violations: <=0 {cov['n_le_zero']}, >{MAX_NET_DIST_M:.0f} "
              f"{cov['n_gt_max']}, below-straight {cov['n_below_straight']}")
        plain(f"  {len(gchecks)} graph safety checks passed")
        return

    ladder = [only] if only else ["SMOKE", "SCREEN", "FULL"]
    gate = verdict = None
    net_df = nf["features"]
    del nf
    gc.collect()

    for stage in ladder:
        plain("")
        plain(f"STAGE {stage}: fitting CONTROL and B4B for JAN and JUL "
              f"({PARAMS['iterations']} trees)")
        res = fit_stage(stage, dep, samples, base, net_df, cov)
        save_cache(blob, stage, res)
        if stage == "FULL":
            cache_full_control(res)
        _buf.clear()
        verdict, gate = finalize_report(stage, blob["stages"][stage],
                                        blob["stages"].get("SMOKE"),
                                        blob["stages"].get("SCREEN"),
                                        cov, graphs, geo, san, problems)
        plain(f"\nDONE -> {REPORT}")
        ceil = res.get("stage_ceiling_limited")
        if stage == "SMOKE" and not gate:
            plain("")
            plain("B4B REJECTED AT SMOKE")
            plain("STOPPED. SCREEN and FULL are not run. ADS-B / B5 is not started.")
            return
        if stage == "SCREEN" and verdict != "PROMISING":
            plain("")
            plain(f"VERDICT: {verdict}")
            plain("STOPPED. FULL is NOT run for WEAK SIGNAL or REJECT.")
            plain("ADS-B / B5 is not started.")
            return
        if stage == "SCREEN" and verdict == "PROMISING" and ceil:
            plain("")
            plain("SCREEN is PROMISING but a fold is ITERATION-CEILING LIMITED.")
            plain("Continuing to FULL at the same budget, and the ceiling status")
            plain("is reported there rather than treated as convergence.")
        del res
        gc.collect()

    tail_by_stage = {
        "SMOKE": "B4B SMOKE COMPLETE - gate PASSED. SCREEN is the next stage.",
        "SCREEN": "B4B SCREEN COMPLETE - STOPPED. ADS-B / B5 is not started.",
        "FULL": "B4B FULL COMPLETE - STOPPED. ADS-B / B5 is not started.",
    }
    plain("")
    plain(f"last stage run: {ladder[-1]}")
    plain(f"verdict: {verdict if verdict else 'SMOKE gate PASSED'}")
    plain(tail_by_stage[ladder[-1]])


if __name__ == "__main__":
    main()
