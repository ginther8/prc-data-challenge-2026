"""Download airport weather observations (METAR) for the 10 challenge airports
from the Iowa Environmental Mesonet (IEM) ASOS/METAR archive, and save one
tidy parquet for feature building.

Source : https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
         (IEM, Iowa State University; free/open, scripted access is supported)
Period : 2025-01-01 .. 2026-08-01 (all 2025 training months + Jan/Jul 2026)
Output : data/external/metar/raw_<ICAO>.csv      (as downloaded)
         data/external/metar/metar_10apt.parquet (all stations, parsed)

Columns in the parquet:
    station, valid (UTC timestamp), tmpc, dwpc, spread_c (= tmpc - dwpc),
    relh, sknt (wind kt), gust (kt), vsby_km, ceiling_ft (lowest BKN/OVC/VV
    layer, NaN if none), wxcodes (present weather, e.g. FZFG, -SN, BR),
    frost_flag (tmpc <= 0 and spread <= 3), low_vis_flag (vsby < 1.5 km or
    ceiling < 300 ft)

Use for features (base model):
    join each departure to the latest observation at its airport at or before
    MVT_TIME_UTC_mvt (both sides UTC, timezone-aware!). A Discord team lost 19
    submissions to a silent timezone-aware vs naive join mismatch - check that
    the joined columns are not all null.

Run from the project root:
    python experiments/metar_download.py
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "data" / "external" / "metar"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
VARS = ["tmpf", "dwpf", "relh", "sknt", "gust", "vsby",
        "skyc1", "skyc2", "skyc3", "skyl1", "skyl2", "skyl3", "wxcodes"]


def fetch(station):
    params = [("station", station), ("tz", "Etc/UTC"), ("format", "onlycomma"),
              ("latlon", "no"), ("elev", "no"), ("missing", "M"), ("trace", "T"),
              ("direct", "no"), ("year1", "2025"), ("month1", "1"), ("day1", "1"),
              ("year2", "2026"), ("month2", "8"), ("day2", "1"),
              ("report_type", "3"), ("report_type", "4")]
    params += [("data", v) for v in VARS]
    for attempt in range(1, 7):
        try:
            r = requests.get(URL, params=params, timeout=300)
            if r.status_code == 200 and r.text.startswith("station"):
                return r.text
            print(f"  {station}: HTTP {r.status_code}, retrying", flush=True)
        except Exception as e:  # network hiccup
            print(f"  {station}: {type(e).__name__}: {e}, retrying", flush=True)
        time.sleep(10 * attempt)
    return None


def ceiling(row):
    best = np.nan
    for c, l in (("skyc1", "skyl1"), ("skyc2", "skyl2"), ("skyc3", "skyl3")):
        if str(row[c]) in ("BKN", "OVC", "VV") and pd.notna(row[l]):
            best = row[l] if np.isnan(best) else min(best, row[l])
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for st in AIRPORTS:
        raw_path = OUT / f"raw_{st}.csv"
        if raw_path.exists() and raw_path.stat().st_size > 1000:
            text = raw_path.read_text(encoding="utf-8")
            print(f"{st}: cached", flush=True)
        else:
            print(f"{st}: downloading ...", flush=True)
            text = fetch(st)
            if text is None:
                print(f"FAIL: {st} could not be downloaded")
                sys.exit(1)
            raw_path.write_text(text, encoding="utf-8")
            time.sleep(5)  # be polite to the free service
        df = pd.read_csv(io.StringIO(text), na_values=["M", "T"], low_memory=False)
        print(f"  {st}: {len(df):,} observations", flush=True)
        frames.append(df)

    m = pd.concat(frames, ignore_index=True)
    m = m[m["station"].astype(str).str.fullmatch(r"[A-Z]{4}")]      # drop stray header/comment lines
    m["valid"] = pd.to_datetime(m["valid"], utc=True, errors="coerce", format="mixed")
    bad = int(m["valid"].isna().sum())
    if bad:
        print(f"  dropped {bad} rows with unparseable timestamps", flush=True)
    m = m[m["valid"].notna()].copy()
    for c in ("tmpf", "dwpf", "relh", "sknt", "gust", "vsby",
              "skyl1", "skyl2", "skyl3"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["tmpc"] = (m["tmpf"] - 32) * 5 / 9
    m["dwpc"] = (m["dwpf"] - 32) * 5 / 9
    m["spread_c"] = m["tmpc"] - m["dwpc"]
    m["vsby_km"] = m["vsby"] * 1.609344
    m["ceiling_ft"] = m.apply(ceiling, axis=1)
    m["frost_flag"] = ((m["tmpc"] <= 0) & (m["spread_c"] <= 3)).astype("int8")
    m["low_vis_flag"] = ((m["vsby_km"] < 1.5) | (m["ceiling_ft"] < 300)).astype("int8")
    m["wxcodes"] = m["wxcodes"].fillna("")
    out = m[["station", "valid", "tmpc", "dwpc", "spread_c", "relh", "sknt", "gust",
             "vsby_km", "ceiling_ft", "wxcodes", "frost_flag", "low_vis_flag"]]
    out = out.sort_values(["station", "valid"]).drop_duplicates(["station", "valid"])
    out.to_parquet(OUT / "metar_10apt.parquet", index=False)

    print("\nper station: observations / months covered / share with temperature")
    for st, g in out.groupby("station"):
        months = g["valid"].dt.strftime("%Y-%m").nunique()
        print(f"  {st}: {len(g):7,}  months={months:2d}  tmpc present={g['tmpc'].notna().mean():.1%}"
              f"  frost obs={int(g['frost_flag'].sum()):5d}  low-vis obs={int(g['low_vis_flag'].sum()):5d}")
    print(f"wrote {OUT / 'metar_10apt.parquet'}  rows={len(out):,}")
    print("Attribution: weather observations from the Iowa Environmental Mesonet (IEM), "
          "Iowa State University.")
    print("DONE")


if __name__ == "__main__":
    main()
