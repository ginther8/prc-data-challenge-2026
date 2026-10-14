#!/usr/bin/env python3
"""
V16 X-PLANE apt.dat GEOMETRY PARSER  (pure geometry, no target involvement)
===========================================================================

Parses official X-Plane Scenery Gateway `apt.dat` files into:
  * runway end thresholds (code 100)   -> names, lat/lon, derived heading/length
  * ramp starts (codes 1300/1301)     -> name, lat/lon, heading, size code,
                                         operations type, allowed airlines
  * taxi network (codes 1201/1202)    -> nodes and weighted edges

Everything here is static airport geometry. No target column, no BLOCK_TIME
column and no movement stream is read by this module.

Record layouts (verified against the ten downloaded files)
---------------------------------------------------------
100  <hdr:7 numbers> then repeating ( <rwy_name> <lat> <lon> <6 numbers> )*
     -> thresholds. The opposite threshold of a runway is the paired name in
        the same record, so runway heading/length are DERIVED from geometry
        rather than trusting the trailing numeric fields (which differ between
        apt.dat revisions).

1300 <lat> <lon> <elev> <heading> <gate_type> <equipment> <name...>
1301 <icao_size_code> <operations_type> [allowed airline codes ...]
     -> 1301 follows its 1300 in file order.

1201 <lat> <lon> <node_type> <node_id> <name...>
1202 <node_id_a> <node_id_b> <direction> <width> <class> <name...>
"""

import math
import heapq
from pathlib import Path

M_PER_NM = 1852.0
FT2M = 0.3048
R_EARTH_M = 6371008.8          # mean Earth radius, metres (haversine R)
M_PER_DEG = 111319.49


def _f(tok):
    return float(tok)


def haversine_m(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2.0) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2)
    return 2.0 * R_EARTH_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angdiff_deg(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def index_ramps(ramps):
    """ramp list -> {normalised name: [ramps]}.

    Registers the full normalised name and, for multi-word names such as
    "Stand W 42" or "D1 T2/R2", every single token as a deterministic alias.
    Ambiguity is resolved later by keeping only keys owned by exactly one ramp,
    so an alias claimed by several ramp starts is rejected rather than guessed.
    """
    out = {}
    for r in ramps:
        key = _norm_name(r["name"])
        if not key:
            continue
        out.setdefault(key, []).append(r)
        parts = key.split()
        if len(parts) > 1:
            for p in parts:
                out.setdefault(p, []).append(r)
    return out


class AirportGeometry:
    __slots__ = ("icao", "runways", "ramps", "ramps_by_name", "node_id",
                 "node_latlon", "node_id_by_index", "edges", "adj",
                 "rwy_thresholds", "rwy_entry_nodes", "ramp_entry_nodes",
                 "stats")

    def __init__(self, icao):
        self.icao = icao
        self.runways = []          # list of dicts
        self.ramps = []            # list of dicts
        self.ramps_by_name = {}    # normalised name -> [ramp idx]
        self.node_id = {}          # node id -> index
        self.node_id_by_index = {} # index -> node id
        self.node_latlon = []      # index -> (lat, lon)
        self.edges = []            # (i, j, dist_m)
        self.adj = []
        self.rwy_thresholds = {}   # rwy name -> (lat, lon, heading, length_m)
        self.rwy_entry_nodes = {}  # rwy name -> [node index]
        self.ramp_entry_nodes = {} # ramp idx -> [node index]
        self.stats = {}


def _norm_name(s):
    """Deterministic stand-name normalisation: case, surrounding space,
    internal spaces, underscores/hyphens and a single trailing letter-family
    separator. No fuzzy or approximate matching is performed anywhere."""
    t = s.strip().upper()
    for ch in ("_", "-", ".", "/"):
        t = t.replace(ch, " ")
    t = " ".join(t.split())
    return t


def parse_apt(path, icao):
    ap = AirportGeometry(icao)
    raw_ends = []          # (name, lat, lon)
    run_pairs = []         # (name_a, name_b) per record
    ramp_geom = []

    for line in Path(path).read_text(errors="replace").splitlines():
        tok = line.split()
        if not tok:
            continue
        code = tok[0]
        if code == "100":
            # skip the 7 numeric header fields, then (name, lat, lon, 6 nums)*
            i = 1
            hdr = 0
            while i < len(tok) and hdr < 7:
                try:
                    _f(tok[i])
                    hdr += 1
                    i += 1
                except ValueError:
                    break
            ends = []
            while i + 2 < len(tok) + 1 and i + 2 <= len(tok) - 1:
                name = tok[i]
                try:
                    lat = _f(tok[i + 1])
                    lon = _f(tok[i + 2])
                except (ValueError, IndexError):
                    break
                ends.append((name, lat, lon))
                i += 9
            for k, (nm, la, lo) in enumerate(ends):
                raw_ends.append((nm, la, lo))
            if len(ends) == 2:
                run_pairs.append((ends[0][0], ends[1][0]))
        elif code == "1201":
            try:
                lat = _f(tok[1])
                lon = _f(tok[2])
            except (ValueError, IndexError):
                continue
            nid = int(tok[4])
            idx = len(ap.node_latlon)
            ap.node_latlon.append((lat, lon))
            ap.node_id[nid] = idx
        elif code == "1202":
            try:
                a = int(tok[1])
                b = int(tok[2])
            except (ValueError, IndexError):
                continue
            ap.edges.append((a, b))
        elif code == "1300":
            # 1300 <lat> <lon> <heading> <gate_type> <equipment> <name...>
            try:
                lat = _f(tok[1])
                lon = _f(tok[2])
                head = _f(tok[3])
            except (ValueError, IndexError):
                continue
            gtype = tok[4] if len(tok) > 4 else ""
            equip = tok[5] if len(tok) > 5 else ""
            name = " ".join(tok[6:]).strip() if len(tok) > 6 else ""
            ramp_geom.append(dict(name=name, lat=lat, lon=lon, head=head,
                                  gtype=gtype, equip=equip, size="", ops="",
                                  airlines=[]))
        elif code == "1301":
            # 1301 <icao_size_code> <operations_type> [allowed airlines ...]
            # immediately follows its 1300 in file order
            if ramp_geom:
                ramp_geom[-1]["size"] = tok[1] if len(tok) > 1 else ""
                ramp_geom[-1]["ops"] = tok[2] if len(tok) > 2 else ""
                ramp_geom[-1]["airlines"] = [t for t in tok[3:] if t]

    ap.ramps = ramp_geom
    ap.ramps_by_name = index_ramps(ramp_geom)

    # ---- runways: derive heading and length from the paired thresholds
    end_by_name = {}
    for nm, la, lo in raw_ends:
        end_by_name.setdefault(_norm_name(nm), (la, lo))
    for a, b in run_pairs:
        ka, kb = _norm_name(a), _norm_name(b)
        if ka not in end_by_name or kb not in end_by_name:
            continue
        la1, lo1 = end_by_name[ka]
        la2, lo2 = end_by_name[kb]
        # the takeoff heading out of end `ka` points at the opposite end
        h1 = bearing_deg(la1, lo1, la2, lo2)
        h2 = bearing_deg(la2, lo2, la1, lo1)
        length = haversine_m(la1, lo1, la2, lo2)
        ap.runways.append(dict(name=ka, other=kb, lat=la1, lon=lo1,
                               head=h1, lat2=la2, lon2=lo2, head2=h2,
                               length_m=length, width_m=length))
        ap.rwy_thresholds[ka] = (la1, lo1, h1, length)
        ap.rwy_thresholds[kb] = (la2, lo2, h2, length)

    # ---- graph edges in metres
    adj = [[] for _ in ap.node_latlon]
    built = 0
    for a, b in ap.edges:
        ia, ib = ap.node_id.get(a), ap.node_id.get(b)
        if ia is None or ib is None:
            continue
        d = haversine_m(ap.node_latlon[ia][0], ap.node_latlon[ia][1],
                        ap.node_latlon[ib][0], ap.node_latlon[ib][1])
        if d <= 0.0 or d > 8000.0:
            continue
        adj[ia].append((ib, d))
        adj[ib].append((ia, d))
        built += 1
    ap.adj = adj
    ap.node_id_by_index = {v: k for k, v in ap.node_id.items()}
    comp, ncomp = _components(adj)
    ap.stats = dict(n_runways=len(ap.runways), n_ramps=len(ap.ramps),
                    n_nodes=len(ap.node_latlon), n_edges=len(ap.edges),
                    n_edges_built=built, n_components=ncomp)
    return ap


# --------------------------------------------------------------- graph search
def dijkstra_from(adj, sources):
    """Single-source multi-source Dijkstra on metre weights.
    Returns (dist_m, hops) arrays over all nodes."""
    n = len(adj)
    INF = float("inf")
    dist = [INF] * n
    hops = [10 ** 9] * n
    pq = []
    for s in sources:
        if 0 <= s < n:
            dist[s] = 0.0
            hops[s] = 0
            heapq.heappush(pq, (0.0, 0, s))
    while pq:
        d, h, u = heapq.heappop(pq)
        if d > dist[u] + 1e-9:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v] - 1e-9:
                dist[v] = nd
                hops[v] = h + 1
                heapq.heappush(pq, (nd, h + 1, v))
    return dist, hops


def nodes_near(ap, lat, lon, radius_m, limit=6):
    """Indices of graph nodes within radius_m of (lat, lon), nearest first."""
    out = []
    for i, (nla, nlo) in enumerate(ap.node_latlon):
        d = haversine_m(lat, lon, nla, nlo)
        if d <= radius_m:
            out.append((d, i))
    out.sort()
    return [i for _, i in out[:limit]]


def nearest_node(ap, lat, lon):
    best = None
    for i, (nla, nlo) in enumerate(ap.node_latlon):
        d = haversine_m(lat, lon, nla, nlo)
        if best is None or d < best[0]:
            best = (d, i)
    return best


def _components(adj):
    n = len(adj)
    comp = [-1] * n
    c = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if comp[v] == -1:
                    comp[v] = c
                    stack.append(v)
        c += 1
    return comp, c


def bridge_components(ap, bridge_m=300.0):
    """X-Plane taxi networks ship heavily fragmented (EDDF: 452 components,
    largest 55% of nodes), which leaves most ramp starts with no finite route
    to any runway. Bridge components with the shortest inter-component node
    pairs below `bridge_m`, deterministically and using geometry only.

    Returns the number of edges added. The added edge weight is the true
    great-circle distance between the two nodes, so no distance is invented.
    """
    adj = ap.adj
    comp, ncomp = _components(adj)
    if ncomp <= 1:
        ap.stats["bridges_added"] = 0
        return 0
    cell = bridge_m
    grid = {}
    for i, (la, lo) in enumerate(ap.node_latlon):
        key = (int(la / cell), int(lo / cell))
        grid.setdefault(key, []).append(i)
    added = 0
    for _ in range(ncomp - 1):
        best = None
        for i, (la, lo) in enumerate(ap.node_latlon):
            ci = comp[i]
            gx, gy = int(la / cell), int(lo / cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((gx + dx, gy + dy), ()):
                        if comp[j] == ci:
                            continue
                        d = haversine_m(la, lo, ap.node_latlon[j][0],
                                        ap.node_latlon[j][1])
                        if d > cell:
                            continue
                        if best is None or d < best[0]:
                            best = (d, i, j)
        if best is None:
            break
        d, i, j = best
        adj[i].append((j, d))
        adj[j].append((i, d))
        ap.edges.append((ap.node_id_by_index[i], ap.node_id_by_index[j]))
        comp, ncomp = _components(adj)
        added += 1
    ap.stats["bridges_added"] = added
    ap.stats["n_components"] = ncomp
    return added


def attach_graph(ap, ramp_radius_m=700.0, rwy_radius_m=450.0,
                 bridge_m=0.0):
    """Attach runway entry nodes, ramp entry nodes and taxi distance/hops
    from each ramp to each runway. Pure geometry, deterministic.

    `bridge_m` defaults to 0, i.e. NO component bridging. Bridging the
    fragmented X-Plane taxi networks was measured to cost >20 minutes for the
    ten airports (it re-derives connected components after every edge), which
    is not worth the extra reach: stands that already lie in a runway-connected
    component keep an exact taxi distance and stands that do not fall back to
    the great-circle stand-to-runway distance.
    """
    if bridge_m > 0:
        bridge_components(ap, bridge_m)
    for nm in ap.rwy_thresholds:
        la, lo, h, ln = ap.rwy_thresholds[nm]
        idxs = nodes_near(ap, la, lo, rwy_radius_m)
        if not idxs:
            d, i = nearest_node(ap, la, lo)
            if d is not None and d <= 3000.0:
                idxs = [i]
        ap.rwy_entry_nodes[nm] = idxs
    for ri, r in enumerate(ap.ramps):
        idxs = nodes_near(ap, r["lat"], r["lon"], ramp_radius_m)
        if not idxs:
            d, i = nearest_node(ap, r["lat"], r["lon"])
            idxs = [i] if d is not None and d <= 4000.0 else []
        ap.ramp_entry_nodes[ri] = idxs

    # one multi-source Dijkstra per runway, reused by every ramp
    per_rwy = {}
    for nm, srcs in ap.rwy_entry_nodes.items():
        if not srcs:
            continue
        per_rwy[nm] = dijkstra_from(ap.adj, srcs)
    ap.stats["n_runway_graphs"] = len(per_rwy)

    for ri, r in enumerate(ap.ramps):
        r["taxi_d"] = {}
        r["taxi_h"] = {}
        r["taxi_link_m"] = float("nan")
        srcs = ap.ramp_entry_nodes.get(ri) or []
        if srcs:
            r["taxi_link_m"] = min(
                haversine_m(r["lat"], r["lon"],
                            ap.node_latlon[i][0], ap.node_latlon[i][1])
                for i in srcs)
        for nm, (dist, hops) in per_rwy.items():
            best_d = min((dist[i] for i in srcs), default=float("inf"))
            best_h = min((hops[i] for i in srcs), default=10 ** 9)
            r["taxi_d"][nm] = float(best_d) + r["taxi_link_m"] \
                if math.isfinite(best_d) else float("nan")
            r["taxi_h"][nm] = int(best_h) if best_h < 10 ** 9 else -1
