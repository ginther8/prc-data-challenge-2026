"""Per-departure ADS-B movement features from the extracted adsb.lol points.

Input : data/external/adsb_v2/points_YYYY-MM-DD.parquet   (from adsb_extract.py)
        data/raw/training_*.parquet                     (2025 departures)
        data/raw/ranking.parquet                        (2026 departures)
Output: data/processed/adsb_features.parquet  (one row per departure in the
        months that have ADS-B points; MVT_ID_mvt + adsb_* columns)

MATCHING
    Callsign = CALLSIGN_flt, falling back to FLIGHT_mvt when CALLSIGN_flt is
    missing (MISSING_4 rows). Points of the departure airport whose callsign
    equals it within [T-4h, T+10min] identify the aircraft (ICAO hex); all of
    that hex's points are then used. T = MVT_TIME_UTC_mvt (takeoff).

FEATURES (movement only - no airport identity, so they transfer when receiver
coverage changes between 2025 and 2026)
    adsb_matched     1 if a hex was found for the callsign
    adsb_ok          1 if the final ground segment reaches the runway
                     (last ground/low point within 300 s of T)
    adsb_est         T - start of the final ground segment (s). Segments are
                     split at gaps > 600 s (parked with transponder off).
    adsb_est_move    T - first point with gs >= 1 kt in that segment (s)
    adsb_gs0         ground speed at segment start (kt); low = pushback seen
    adsb_npts        points in the segment
    adsb_maxgap      largest gap inside the segment (s)
    adsb_tail        T - last ground/low point (s)
    adsb_stopped     seconds spent at gs < 1 kt inside the segment
    adsb_prev_gap    gap before the segment (s); large = parked before it
    adsb_nhex        number of hexes carrying the callsign in the window
  v2 additions (where the segment starts / how it moved):
    adsb_lead        s from segment start to first point with gs >= 1 kt
                     (transponder already on at the stand)
    adsb_dpark       metres from segment start to the nearest PARKING SPOT
                     seen that month at the airport (end of any ground track
                     at < 3 kt that is followed by a > 600 s gap)
    adsb_dpark_conf  metres to the nearest CONFIRMED parking spot (an aircraft
                     reappeared within 40 m of where its track stopped)
    adsb_down        metres from segment start to where the same aircraft's
                     previous ground segment ended (its own parking position)
    adsb_path        path length of the final segment (m)
    adsb_disp        straight-line distance segment start -> last point (m)
  v3 additions (OpenStreetMap stands / OurAirports runway thresholds from
  data/external/b4a_geometry/frozen_geometry.pkl, ODbL / public domain):
    adsb_d_stand     metres from the flight's OSM stand to the start of its
                     final ADS-B segment - how far it had already taxied when
                     first seen (late pickups: missed time grows with it)
    adsb_d_first_rwy metres from the segment start to the departure runway
                     threshold (distance still to go when first seen)
    geo_d_stand_rwy  metres stand -> runway threshold (straight line); needs no
                     ADS-B, so it is filled for every flight with both known
  v4 addition (OpenFlights airlines.dat, ODbL):
    adsb_match_type  how the ADS-B track was found: 0 none, 1 ATC callsign
                     (CALLSIGN_flt), 2 flight number as written (FLIGHT_mvt),
                     3 flight number with the IATA airline code converted to
                     ICAO (LH430 -> DLH430), 4 as 3 without leading zeros
  A segment that starts slowly AND near a parking spot is a true pushback
  (2025-07-15: 76% within 60 s, median error 20 s, vs 36% / 123 s otherwise).

No target, block time or AOBT is used here.

Run from the project root:
    python experiments/adsb_features.py                 # Jan+Jul 2025/2026
    python experiments/adsb_features.py --months all    # every month with points
    python experiments/adsb_features.py --adsb-dir data/external/adsb   # original extraction
"""

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
RAW = PROJECT_ROOT / "data" / "raw"
ADSB = PROJECT_ROOT / "data" / "external" / "adsb_v2"   # elevation-relative extraction
OUT = PROJECT_ROOT / "data" / "processed" / "adsb_features.parquet"
FEATURE_VERSION = 4
AIRLINES = PROJECT_ROOT / "data" / "external" / "openflights" / "airlines.dat"
GEOMETRY = PROJECT_ROOT / "data" / "external" / "b4a_geometry" / "frozen_geometry.pkl"
_GEO = None

MID = "MVT_ID_mvt"
WIN_BEFORE = 4 * 3600
WIN_AFTER = 600
SEG_GAP = 600
RUNWAY_TAIL = 300
LOW_ALT = 200
GROUND_GS = 50.0   # kt: nothing airborne moves this slowly within 7 km of a hub
ELEV_FT = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 14,
           "LEMD": 1998, "LFPG": 392, "LIRF": 13, "LSZH": 1416, "LTFM": 325}
FEATS = ["adsb_matched", "adsb_ok", "adsb_est", "adsb_est_move", "adsb_gs0",
         "adsb_npts", "adsb_maxgap", "adsb_tail", "adsb_stopped",
         "adsb_prev_gap", "adsb_nhex",
         "adsb_lead", "adsb_dpark", "adsb_dpark_conf", "adsb_down",
         "adsb_path", "adsb_disp",
         "adsb_d_stand", "adsb_d_first_rwy", "geo_d_stand_rwy",
         "adsb_match_type"]
PARK_GS = 3.0
CONF_M = 40.0
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def iata_to_icao():
    """IATA 2-char airline code -> list of ICAO 3-letter codes (active first)."""
    if not AIRLINES.exists():
        print(f"WARNING: {AIRLINES} not found - no IATA->ICAO callsign conversion", flush=True)
        return {}
    a = pd.read_csv(AIRLINES, header=None, keep_default_na=False,
                    names=["id", "name", "alias", "iata", "icao", "callsign", "country", "active"])
    a = a[a["iata"].str.fullmatch(r"[A-Z0-9]{2}") & a["icao"].str.fullmatch(r"[A-Z]{3}")]
    a = a.sort_values("active", ascending=False)          # 'Y' before 'N'
    out = {}
    for iata, icao in zip(a["iata"], a["icao"]):
        out.setdefault(iata, [])
        if icao not in out[iata]:
            out[iata].append(icao)
    print(f"airline codes: {len(out):,} IATA codes with ICAO equivalents", flush=True)
    return out


def candidates(callsign, flight, conv):
    """Ordered (callsign string, match type) candidates for one departure."""
    import re
    c = []
    if isinstance(callsign, str) and callsign.strip():
        c.append((callsign.strip(), 1))
    f = flight.strip() if isinstance(flight, str) else ""
    if f and f.upper() != "UNKNOWN":
        c.append((f, 2))
        m = re.fullmatch(r"([A-Z0-9]{2})(\d{1,4})([A-Z]{0,2})", f)
        if m:
            code, num, suf = m.groups()
            for icao in conv.get(code, []):
                c.append((icao + num + suf, 3))
                if num.lstrip("0") and num.lstrip("0") != num:
                    c.append((icao + num.lstrip("0") + suf, 4))
    seen, out = set(), []
    for x, t in c:
        if x not in seen:
            seen.add(x)
            out.append((x, t))
    return out


def load_departures():
    cols = [MID, "PHASE_mvt", "ADEP_mvt", "CALLSIGN_flt", "FLIGHT_mvt",
            "MVT_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt"]
    parts = []
    for f in sorted(glob.glob(str(RAW / "training_*.parquet"))) + \
            [str(RAW / "ranking.parquet")]:
        t = pq.read_table(f, columns=cols).to_pandas()
        parts.append(t[t["PHASE_mvt"] == "DEP"].drop(columns="PHASE_mvt"))
    d = pd.concat(parts, ignore_index=True)
    d["T"] = (d["MVT_TIME_UTC_mvt"] - EPOCH).dt.total_seconds()
    conv = iata_to_icao()
    d["cands"] = [candidates(c, f, conv) for c, f in
                  zip(d["CALLSIGN_flt"].astype(object), d["FLIGHT_mvt"].astype(object))]
    d["ym"] = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    return d


def load_points(days, apt):
    frames = []
    for day in days:
        p = ADSB / f"points_{day}.parquet"
        if p.exists():
            t = pq.read_table(p, filters=[("apt", "=", apt)]).to_pandas()
            if len(t):
                frames.append(t)
    if not frames:
        return None
    pts = pd.concat(frames, ignore_index=True)
    pts["flight"] = pts["flight"].astype(str).str.strip()
    elev = ELEV_FT.get(apt, 0)
    # on the ground: flagged, OR slow, OR near airport elevation (pressure altitude)
    pts["low"] = (pts["ground"] | (pts["gs"].fillna(999.0) <= GROUND_GS)
                  | (pts["alt"] < elev + LOW_ALT))
    return pts


def geometry():
    global _GEO
    if _GEO is None:
        import pickle
        if GEOMETRY.exists():
            with open(GEOMETRY, "rb") as fh:
                g = pickle.load(fh)
            _GEO = (g.get("stand", {}), g.get("runway", {}))
            print(f"geometry: {len(_GEO[0]):,} stands, {len(_GEO[1])} runway ends", flush=True)
        else:
            print(f"WARNING: {GEOMETRY} not found - stand/runway features left empty", flush=True)
            _GEO = ({}, {})
    return _GEO


def xy(lat, lon, lat0, lon0):
    return ((np.asarray(lon) - lon0) * 111000.0 * np.cos(np.radians(lat0)),
            (np.asarray(lat) - lat0) * 111000.0)


def parking_spots(pts, lat0, lon0):
    """Return (all_spots_xy, confirmed_spots_xy) from the airport's points."""
    p = pts[pts["low"]].sort_values(["hex", "t"], kind="mergesort")
    hx = p["hex"].to_numpy()
    t = p["t"].to_numpy()
    gs = np.nan_to_num(p["gs"].to_numpy(), nan=0.0)
    x, y = xy(p["lat"].to_numpy(), p["lon"].to_numpy(), lat0, lon0)
    same_next = np.r_[hx[1:] == hx[:-1], False]
    gap_next = np.r_[np.diff(t), np.inf]
    is_end = (~same_next) | (gap_next > SEG_GAP)
    spot = is_end & (gs < PARK_GS)
    all_xy = np.c_[x[spot], y[spot]]
    conf = is_end & same_next & (gap_next > SEG_GAP)
    idx = np.flatnonzero(conf)
    d = np.hypot(x[idx + 1] - x[idx], y[idx + 1] - y[idx]) if len(idx) else np.array([])
    idx = idx[d < CONF_M]
    conf_xy = np.c_[x[idx + 1], y[idx + 1]] if len(idx) else np.zeros((0, 2))
    return all_xy, conf_xy


class Nearest:
    """Nearest-neighbour distance on a 100 m grid (no scipy needed)."""

    def __init__(self, pts_xy, cell=100.0):
        self.cell = cell
        self.grid = {}
        for px, py in pts_xy:
            self.grid.setdefault((int(px // cell), int(py // cell)), []).append((px, py))
        self.grid = {k: np.array(v) for k, v in self.grid.items()}

    def dist(self, px, py, max_cells=3):
        if not self.grid:
            return np.nan
        cx, cy = int(px // self.cell), int(py // self.cell)
        best = np.inf
        for r in range(max_cells + 1):
            for i in range(cx - r, cx + r + 1):
                for j in range(cy - r, cy + r + 1):
                    if max(abs(i - cx), abs(j - cy)) != r:
                        continue
                    a = self.grid.get((i, j))
                    if a is not None:
                        best = min(best, float(np.min(np.hypot(a[:, 0] - px, a[:, 1] - py))))
            if best <= r * self.cell:
                break
        return best if np.isfinite(best) else float((max_cells + 1) * self.cell)


def features_for(deps, pts, apt=None):
    out = np.full((len(deps), len(FEATS)), np.nan)
    out[:, 0] = 0
    out[:, 1] = 0
    if len(deps) == 0:
        return out
    stands, runways = geometry()
    ap = str(deps["ADEP_mvt"].iloc[0])
    lat0 = float(pts["lat"].median()) if pts is not None and len(pts) else 0.0
    lon0 = float(pts["lon"].median()) if pts is not None and len(pts) else 0.0
    if pts is None or not len(pts):
        ref = [v for (a, _), v in stands.items() if a == ap]
        if ref:
            lat0, lon0 = ref[0]["lat"], ref[0]["lon"]
    sxy = np.full((len(deps), 2), np.nan)
    rxy = np.full((len(deps), 2), np.nan)
    for i, (st, rw) in enumerate(zip(deps["STAND_mvt"].astype(str).to_numpy(),
                                     deps["RUNWAY_mvt"].astype(str).to_numpy())):
        s_ = stands.get((ap, st))
        if s_:
            sxy[i] = xy(s_["lat"], s_["lon"], lat0, lon0)
        r_ = runways.get((ap, rw))
        if r_:
            rxy[i] = xy(r_["lat"], r_["lon"], lat0, lon0)
    i_geo = FEATS.index("geo_d_stand_rwy")
    out[:, i_geo] = np.hypot(sxy[:, 0] - rxy[:, 0], sxy[:, 1] - rxy[:, 1])
    if pts is None or len(deps) == 0:
        return out
    # hex index: sorted by hex, t
    all_xy, conf_xy = parking_spots(pts, lat0, lon0)
    near_all, near_conf = Nearest(all_xy), Nearest(conf_xy)
    pts = pts.sort_values(["hex", "t"], kind="mergesort").reset_index(drop=True)
    X_all, Y_all = xy(pts["lat"].to_numpy(), pts["lon"].to_numpy(), lat0, lon0)
    hx = pts["hex"].to_numpy()
    t_all = pts["t"].to_numpy()
    gs_all = pts["gs"].to_numpy()
    low_all = pts["low"].to_numpy()
    starts = np.flatnonzero(np.r_[True, hx[1:] != hx[:-1]])
    ends = np.r_[starts[1:], len(hx)]
    hex_pos = {hx[s]: (s, e) for s, e in zip(starts, ends)}
    # callsign index: callsign -> (times, hexes)
    cs_pts = pts.loc[pts["flight"] != "", ["flight", "t", "hex"]]
    cs_idx = {k: (g["t"].to_numpy(), g["hex"].to_numpy())
              for k, g in cs_pts.groupby("flight", sort=False)}

    i_mt = FEATS.index("adsb_match_type")
    out[:, i_mt] = 0
    for i, (cands, T) in enumerate(zip(deps["cands"], deps["T"].to_numpy())):
        hexes, mtype = None, 0
        for cs, t_ in cands:
            rec = cs_idx.get(cs)
            if rec is None:
                continue
            ct, ch = rec
            m = (ct >= T - WIN_BEFORE) & (ct <= T + WIN_AFTER)
            if m.any():
                hexes, mtype = np.unique(ch[m]), t_
                break
        if hexes is None:
            continue
        out[i, 0] = 1
        out[i, i_mt] = mtype
        out[i, 10] = len(hexes)
        best = None
        for h in hexes:
            s, e = hex_pos[h]
            tt, gg, ll = t_all[s:e], gs_all[s:e], low_all[s:e]
            sel = ll & (tt >= T - WIN_BEFORE) & (tt <= T + 30)
            if not sel.any():
                continue
            xx, yy = X_all[s:e][sel], Y_all[s:e][sel]
            tt, gg = tt[sel], gg[sel]
            tail = T - tt[-1]
            if best is None or abs(tail) < abs(best[2]):
                best = (tt, gg, tail, xx, yy)
        if best is None:
            continue
        tt, gg, tail, xx, yy = best
        cut = np.flatnonzero(np.diff(tt) > SEG_GAP)
        s0 = cut[-1] + 1 if len(cut) else 0
        seg_t, seg_g = tt[s0:], np.nan_to_num(gg[s0:], nan=0.0)
        mv = np.flatnonzero(seg_g >= 1.0)
        dts = np.diff(seg_t)
        out[i, 1] = 1.0 if tail < RUNWAY_TAIL else 0.0
        out[i, 2] = T - seg_t[0]
        out[i, 3] = T - seg_t[mv[0]] if len(mv) else np.nan
        out[i, 4] = seg_g[0]
        out[i, 5] = len(seg_t)
        out[i, 6] = dts.max() if len(dts) else 0.0
        out[i, 7] = tail
        out[i, 8] = float(dts[seg_g[:-1] < 1.0].sum()) if len(dts) else 0.0
        out[i, 9] = (tt[s0] - tt[s0 - 1]) if s0 > 0 else np.nan
        out[i, 11] = (seg_t[mv[0]] - seg_t[0]) if len(mv) else np.nan
        out[i, 12] = near_all.dist(xx[s0], yy[s0])
        out[i, 13] = near_conf.dist(xx[s0], yy[s0])
        out[i, 14] = (float(np.hypot(xx[s0] - xx[s0 - 1], yy[s0] - yy[s0 - 1]))
                      if s0 > 0 else np.nan)
        sx, sy = xx[s0:], yy[s0:]
        out[i, 15] = float(np.hypot(np.diff(sx), np.diff(sy)).sum()) if len(sx) > 1 else 0.0
        out[i, 16] = float(np.hypot(sx[-1] - sx[0], sy[-1] - sy[0]))
        out[i, 17] = float(np.hypot(sx[0] - sxy[i, 0], sy[0] - sxy[i, 1]))
        out[i, 18] = float(np.hypot(sx[0] - rxy[i, 0], sy[0] - rxy[i, 1]))
    return out


def main():
    global ADSB
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adsb-dir", default=str(ADSB),
                    help="folder with points_YYYY-MM-DD.parquet (default: adsb_v2; "
                         "use data/external/adsb for the original extraction)")
    ap.add_argument("--months", default="2025-01,2025-07,2026-01,2026-07",
                    help="comma-separated YYYY-MM, or 'all' for every month with points")
    args = ap.parse_args()
    ADSB = Path(args.adsb_dir)
    if not ADSB.is_absolute():
        ADSB = PROJECT_ROOT / ADSB
    print(f"ADS-B folder: {ADSB}", flush=True)
    t0 = time.time()
    days = sorted(p.stem.replace("points_", "") for p in ADSB.glob("points_*.parquet"))
    if args.months != "all":
        keep = {m.strip() for m in args.months.split(",") if m.strip()}
        days = [d for d in days if d[:7] in keep]
    if not days:
        print("FAIL: no ADS-B point files found in", ADSB)
        sys.exit(1)
    months = sorted({d[:7] for d in days})
    print(f"{len(days)} ADS-B days found, months: {months}", flush=True)
    deps = load_departures()
    deps = deps[deps["ym"].isin(months)].reset_index(drop=True)
    print(f"departures in those months: {len(deps):,}", flush=True)

    results = []
    for ym in months:
        mdays = [d for d in days if d.startswith(ym)]
        dm = deps[deps["ym"] == ym]
        # also keep departures whose day has points (partial months are fine)
        dm = dm[dm["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").isin(mdays)]
        for apt, g in dm.groupby("ADEP_mvt"):
            pts = load_points(mdays, apt)
            f = features_for(g, pts)
            r = pd.DataFrame(f, columns=FEATS)
            r.insert(0, MID, g[MID].to_numpy())
            results.append(r)
            good = (r.adsb_ok == 1) & (r.adsb_gs0 < 6) & (r.adsb_dpark < 40)
            print(f"  {ym} {apt}: {len(g):,} deps, matched {r.adsb_matched.mean():.1%}, "
                  f"reaches runway {r.adsb_ok.mean():.1%}, "
                  f"pushback-like {((r.adsb_ok == 1) & (r.adsb_gs0 < 6)).mean():.1%}, "
                  f"pushback at parking spot {good.mean():.1%}",
                  flush=True)
    out = pd.concat(results, ignore_index=True)
    if out[MID].duplicated().any():
        print("FAIL: duplicate MVT_ID_mvt in features")
        sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  rows={len(out):,}  ({time.time()-t0:.0f}s)")
    print("DONE")


if __name__ == "__main__":
    main()