#!/usr/bin/env python3
"""
V16 GEOMETRY FEATURE BUILDER
============================

Turns the parsed official `apt.dat` airport geometry into one geometry row per
`MVT_ID_mvt` departure, cached to
    data/processed/v16_geometry_features.parquet

Feature groups
--------------
G1 "basic"  runway / stand / airport geometry, no taxi graph
G2 "graph"  G1 plus the taxi-network distances and hop counts

Safety
------
* Only airport geometry is read: lat/lon, ramp names, runway names, taxi nodes.
* No target column, no BLOCK_TIME_UTC_mvt, no movement stream, no 2026 row.
* Stand identity uses EXACT normalised-name equality against a ramp start.
  Ambiguous keys (a name shared by more than one ramp start) are rejected
  rather than guessed, and unmatched rows keep every stand feature null while
  the stand-independent features are still produced.
"""

import json
import math
from pathlib import Path

import polars as pl

from v16_apt_geometry import (AirportGeometry, _norm_name, angdiff_deg,
                              attach_graph, bearing_deg, haversine_m,
                              index_ramps, parse_apt)

AP = "ADEP_mvt"
STAND = "STAND_mvt"
RUNWAY = "RUNWAY_mvt"
MID = "MVT_ID_mvt"

GEODIR = Path("data/external/apt_geometry")
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF",
            "LSZH", "LTFM"]

SIZE_CODE = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0}
OPS_CODE = {"none": 0.0, "general_aviation": 1.0, "airline": 2.0, "cargo": 3.0,
            "military": 4.0}

# --------------------------------------------------------------------- names
G1_FEATURES = [
    # runway identity and geometry
    "geo_rwy_len_m",
    "geo_rwy_head_deg",
    "geo_rwy_len_rank_pct",
    "geo_rwy_len_frac_of_max",
    "geo_rwy_is_longest",
    # stand identity and geometry. Unmatched / ambiguous stands are encoded by
    # a null in every stand column, so no constant match flag is carried.
    "geo_stand_rwy_dist_m",
    "geo_stand_rwy_head_err_deg",
    "geo_stand_rwy_head_err_signed_deg",
    "geo_stand_nn_m",
    "geo_stand_nn150_n",
    "geo_stand_nn300_n",
    "geo_stand_rwy_rank_pct",
    "geo_stand_size_code",
    "geo_stand_ops_code",
    "geo_stand_is_gate",
    "geo_stand_equip_n",
    "geo_stand_n_airlines",
    # stand-independent fallbacks, so airports without usable stand identity
    # still receive runway-relative geometry
    "geo_rwy_to_ramp_centroid_m",
    "geo_rwy_to_nearest_ramp_m",
    # airport level
    "geo_ap_n_ramps",
    "geo_ap_n_runways",
    "geo_ap_ramp_extent_m",
    "geo_ap_ramp_density",
    "geo_ap_n_taxi_nodes",
    "geo_ap_n_taxi_edges",
]
G2_FEATURES = [
    "geo_stand_taxi_rwy_m",
    "geo_stand_taxi_rwy_hops",
    "geo_stand_taxi_link_m",
    "geo_stand_taxi_rwy_fallback_m",
]
GEO_FEATURES = G1_FEATURES + G2_FEATURES
GEO_CATEGORICALS: list = []      # geometry is numeric only


def _load_geometry():
    """Parse every airport, attach the named-ramp overlay and the taxi graph."""
    manifest = {}
    mp = GEODIR / "provenance.json"
    if mp.exists():
        for p in json.load(open(mp)):
            manifest[p["airport"]] = (p.get("primary_scenery_id"),
                                      p.get("overlay_scenery_id"))
    geos = {}
    for a in AIRPORTS:
        prim, over = manifest.get(a, (None, None))
        pf = (GEODIR / "apt_primary" / f"{a}.dat")
        if not pf.exists():
            pf = GEODIR / "apt" / f"{a}.dat"
        g = parse_apt(str(pf), a)
        of = GEODIR / "apt_overlay" / f"{a}.dat"
        if of.exists():
            og = parse_apt(str(of), a)
            # The overlay pack is the one whose ramp names actually match the
            # observed STAND_mvt. It describes the same airport in the same
            # global coordinate frame (verified at LEBL: runway thresholds
            # agree within 4 m, every overlay ramp start is within 81 m of a
            # primary taxi node), so its ramp starts REPLACE the primary's.
            # Keeping both would double-count ramp density and neighbour
            # counts, since the two packs model the same physical apron.
            g.ramps = og.ramps
            g.ramps_by_name = index_ramps(og.ramps)
        attach_graph(g)
        geos[a] = g
    return geos


def _unambiguous_ramps(g):
    """Ramp index -> ramp dict, keeping only names owned by a single ramp."""
    out = {}
    for r in g.ramps:
        out.setdefault(id(r), []).append(r)
    uniq = {}
    for key, rs in g.ramps_by_name.items():
        if len(rs) == 1:
            uniq[key] = rs[0]
    return uniq


def _airport_tables(g):
    """Pre-compute per-airport reference geometry and per-stand statistics."""
    lats = [r["lat"] for r in g.ramps] or [g.node_latlon[0][0] if g.node_latlon else 0.0]
    lons = [r["lon"] for r in g.ramps] or [g.node_latlon[0][1] if g.node_latlon else 0.0]
    clat = sum(lats) / len(lats)
    clon = sum(lons) / len(lons)
    span = max(haversine_m(min(lats), min(lons), max(lats), max(lons)), 1.0)
    lengths = sorted(r["length_m"] for r in g.runways) or [1.0]
    lmax = max(lengths)
    n_stands = max(len(g.ramps), 1)
    # per-stand neighbour counts and per-runway distance rank
    for r in g.ramps:
        near = [haversine_m(r["lat"], r["lon"], o["lat"], o["lon"])
                for o in g.ramps if o is not r]
        near.sort()
        r["nn_m"] = near[0] if near else float("nan")
        r["nn150"] = float(sum(1 for d in near if d <= 150.0))
        r["nn300"] = float(sum(1 for d in near if d <= 300.0))
    for nm, (la, lo, hd, ln) in g.rwy_thresholds.items():
        ds = sorted(haversine_m(r["lat"], r["lon"], la, lo) for r in g.ramps)
        for r, d in zip(sorted(g.ramps, key=lambda x: haversine_m(x["lat"], x["lon"], la, lo)), ds):
            r.setdefault("rank", {})[nm] = (
                d, (ds.index(d) + 1) / max(len(ds), 1))
    return dict(clat=clat, clon=clon, extent=span, lmax=lmax, lengths=lengths,
                n_stands=n_stands)


def build_rows(rows, geos, tab):
    """rows: list of (mid, apt, stand, runway) -> dict of geometry columns."""
    n = len(rows)
    cols = {c: [None] * n for c in GEO_FEATURES}
    cache = {}
    for i, (mid, apt, stand, runway) in enumerate(rows):
        g = geos.get(apt)
        if g is None:
            continue
        t = tab[apt]
        rk = _norm_name(runway) if runway else ""
        sk = _norm_name(stand) if stand else ""
        key = (apt, sk, rk)
        ent = cache.get(key)
        if ent is None:
            ent = _entry(g, t, sk, rk)
            cache[key] = ent
        for c, v in ent.items():
            cols[c][i] = v
    return pl.DataFrame({MID: [r[0] for r in rows], **cols})


def _entry(g, t, stand_key, rwy_key):
    """All geometry values for one (airport, stand, runway) combination."""
    out = {c: None for c in GEO_FEATURES}
    out["geo_ap_n_ramps"] = float(len(g.ramps))
    out["geo_ap_n_runways"] = float(len(g.runways))
    out["geo_ap_ramp_extent_m"] = t["extent"]
    out["geo_ap_ramp_density"] = t["n_stands"] / max(t["extent"], 1.0) / 1e6
    out["geo_ap_n_taxi_nodes"] = float(g.stats.get("n_nodes", 0))
    out["geo_ap_n_taxi_edges"] = float(g.stats.get("n_edges_built", 0))
    rm = g.rwy_thresholds.get(rwy_key)
    if rm is not None:
        rlat, rlon, rhead, rlen = rm
        out["geo_rwy_len_m"] = rlen
        out["geo_rwy_head_deg"] = rhead
        lr = t["lengths"]
        out["geo_rwy_len_rank_pct"] = sum(1 for x in lr if x <= rlen) / max(len(lr), 1)
        out["geo_rwy_len_frac_of_max"] = rlen / max(t["lmax"], 1.0)
        out["geo_rwy_is_longest"] = 1.0 if rlen >= t["lmax"] - 1e-6 else 0.0
        out["geo_rwy_to_ramp_centroid_m"] = haversine_m(rlat, rlon, t["clat"],
                                                        t["clon"])
        near = min((haversine_m(rlat, rlon, r["lat"], r["lon"])
                    for r in g.ramps), default=None)
        out["geo_rwy_to_nearest_ramp_m"] = near
    ramp = g.ramps_by_name.get(stand_key)
    if ramp is None or len(ramp) != 1:
        return out                      # unmatched or ambiguous: no stand value
    r = ramp[0]
    out["geo_stand_nn_m"] = r.get("nn_m")
    out["geo_stand_nn150_n"] = r.get("nn150")
    out["geo_stand_nn300_n"] = r.get("nn300")
    out["geo_stand_size_code"] = SIZE_CODE.get(r.get("size", ""), None)
    out["geo_stand_ops_code"] = OPS_CODE.get(r.get("ops", ""), None)
    out["geo_stand_is_gate"] = 1.0 if r.get("gtype") == "gate" else 0.0
    out["geo_stand_equip_n"] = float(len([x for x in
                                          r.get("equip", "").split("|") if x]))
    out["geo_stand_n_airlines"] = float(len(r.get("airlines", [])))
    if rm is not None:
        rlat, rlon, rhead, rlen = rm
        d = haversine_m(r["lat"], r["lon"], rlat, rlon)
        brg = bearing_deg(r["lat"], r["lon"], rlat, rlon)
        err = angdiff_deg(brg, rhead)
        out["geo_stand_rwy_dist_m"] = d
        out["geo_stand_rwy_head_err_deg"] = err
        signed = (brg - rhead + 180.0) % 360.0 - 180.0
        out["geo_stand_rwy_head_err_signed_deg"] = signed
        rk = r.get("rank", {}).get(rwy_key)
        if rk is not None:
            out["geo_stand_rwy_rank_pct"] = rk[1]
        td = r.get("taxi_d", {}).get(rwy_key)
        th = r.get("taxi_h", {}).get(rwy_key)
        if td is not None and math.isfinite(td):
            out["geo_stand_taxi_rwy_m"] = td
        if th is not None and th >= 0:
            out["geo_stand_taxi_rwy_hops"] = float(th)
        tl = r.get("taxi_link_m")
        if tl is not None and math.isfinite(tl):
            out["geo_stand_taxi_link_m"] = tl
        if td is None or not math.isfinite(td):
            out["geo_stand_taxi_rwy_fallback_m"] = d
    return out


def build_cache(out_path=Path("data/processed/v16_geometry_features.parquet")):
    """Read every 2025 departure, build the geometry frame and cache it."""
    geos = _load_geometry()
    tab = {a: _airport_tables(g) for a, g in geos.items()}
    parts = []
    src = sorted(Path("data/raw").glob("training_2025-*.parquet"))
    if not src:
        raise SystemExit("no 2025 training shards found")
    for f in src:
        d = (pl.scan_parquet(f)
             .filter(pl.col("PHASE_mvt") == "DEP")
             .filter(pl.col(AP).is_in(AIRPORTS))
             .select([MID, AP, STAND, RUNWAY])
             .collect())
        rows = list(zip(d[MID].to_list(), d[AP].to_list(),
                        d[STAND].to_list(), d[RUNWAY].to_list()))
        parts.append(build_rows(rows, geos, tab))
        print(f"  {f.name}: {d.height:>9,} target-airport departures")
    out = pl.concat(parts, how="vertical")
    out = out.sort(MID)
    if out[MID].n_unique() != out.height:
        raise SystemExit("duplicate MVT_ID_mvt in geometry cache")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return out
