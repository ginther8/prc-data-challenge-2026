"""V10-A - fetch ERA5 hourly single-level weather at the 10 competition airports.

This is a FEASIBILITY SCAFFOLD, not a production downloader. It is dry-run by
default and performs no network call to the Climate Data Store unless
``--execute`` is passed. V10-A was an audit: no weather was downloaded, no
CatBoost was trained, and the model is unchanged (V8/T4 remains champion).

WHAT THIS PULLS
    Dataset : "ERA5 hourly time-series data on single levels from 1940 to present"
              (CDS id ``reanalysis-era5-single-levels-timeseries``, DOI 10.24381/1cf1ad76)
    Why the time-series dataset and not the gridded one: a CDS request counts
    "fields" as n_variables x n_times x n_gridpoints, and the published limit for
    the gridded ``reanalysis-era5-single-levels`` is 120,000 fields per request.
    The 10 airports span a bounding box of 6,240 grid points, so a Europe-wide
    gridded pull would need ~3,600 requests. The time-series dataset's Location
    selection returns a time series for the NEAREST GRID POINT only, which puts
    one airport-year at 70,080 fields (58% of the limit). 20 requests, ~5 MB total.

LICENCE AND ATTRIBUTION
    ERA5 hourly data on single levels is released under CC-BY-4.0
    (SPDX: https://spdx.org/licenses/CC-BY-4.0), per the CDS catalogue entry and
    the CDS Terms of Use Article 4, which defers to each dataset's own licence.
    CC-BY-4.0 permits sharing, adaptation and commercial use, INCLUDING
    REDISTRIBUTION, on condition that appropriate credit is given, a link to the
    licence is provided, and any changes are indicated. The CDS Terms of Use
    Article 5 additionally forbids implying that ECMWF or the EU endorses the use.
    Required citation, also emitted by ATTRIBUTION below:
        Copernicus Climate Change Service (C3S), 2025: ERA5 hourly data on single
        levels from 1940 to present. DOI 10.24381/cds.adbb2d47.
        Hersbach, H., Bell, B., Berrisford, P., et al. (2020), The ERA5 global
        reanalysis. Quarterly Journal of the Royal Meteorological Society,
        146(730), 1999-2049. https://doi.org/10.1002/qj.3803

COORDINATE PROVENANCE
    Airport coordinates come from OurAirports (https://ourairports.com/data/,
    CSV at https://davidmegginson.github.io/ourairports-data/airports.csv).
    OurAirports states "All data is released to the Public Domain"; the mirror
    repository carries the Unlicense. NO COORDINATE IN THIS FILE IS GUESSED.
    The exact upstream copy used for V10-A was
        12,731,283 bytes
        sha256 2144a01fb74f20f80fcfc52a919e72e2f42ceb93aca456804a1cb9bb2d3ce79e
    OurAirports is regenerated nightly, so the coordinates below are FROZEN here
    rather than re-read at run time. --verify-coords re-derives them from a
    local airports.csv and asserts the match, so the pin stays auditable.

REPRODUCIBILITY AND CREDENTIALS
    A free CDS account is required. Credentials live in ~/.cdsapirc and are never
    written into this repository:
        url: https://cds.climate.copernicus.eu/api
        key: <PERSONAL-ACCESS-TOKEN>
    The account holder must also accept the dataset licence once in the web form
    ("Download data" tab, Terms of use section). Install the client with
    ``pip install "cdsapi>=0.7.7"``.

    Because the derived per-airport series are only ~5 MB and CC-BY-4.0 permits
    redistribution, the most reproducible arrangement for a competition pipeline
    is to COMMIT the derived CSVs and keep this script as the regeneration path.
    That removes the account dependency and the CDS queue from reproduction
    entirely. See the V10-A report, section 9.

DATA SEMANTICS THAT AFFECT THE JOIN (not optional reading)
    ERA5 hourly single levels mixes two kinds of variable on one time axis:
      * instantaneous analysis fields (2m temperature, 2m dewpoint, 10m u/v wind,
        mean sea level pressure, surface pressure, total cloud cover) are the
        model state VALID AT the timestamp;
      * accumulations and since-previous-post-processing extrema (total
        precipitation, 10m wind gust) are NOT valid at the timestamp - the value
        stamped T covers the hour ENDING at T, i.e. (T-1h, T].
    A single naive join therefore mis-times the accumulated variables by up to an
    hour relative to the instantaneous ones. The V10-A report (section 7) makes
    the resulting per-variable offsets explicit; this script deliberately does not
    perform the join, it only fetches the hourly series on a clean UTC axis.

NO VISIBILITY
    ERA5 hourly single levels does not publish a horizontal visibility variable
    (checked against the full 262-entry gridded variable list and the 23-entry
    time-series list). Fog/visibility is therefore proxied in V10-B via the
    2m temperature-minus-dewpoint depression and cloud base height, not fetched
    from a second source in this scaffold.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path

OUT_DIR = Path("data/external/era5")

DATASET = "reanalysis-era5-single-levels-timeseries"
ERA5_GRID_STEP = 0.25
PUBLISHED_FIELD_LIMIT = 120_000
NETCDF_FIELD_LIMIT = None

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
OURAIRPORTS_BYTES = 12_731_283
OURAIRPORTS_SHA256 = "2144a01fb74f20f80fcfc52a919e72e2f42ceb93aca456804a1cb9bb2d3ce79e"

ATTRIBUTION = (
    "Weather: Copernicus Climate Change Service (C3S), ERA5 hourly data on "
    "single levels from 1940 to present, DOI 10.24381/cds.adbb2d47, licensed "
    "CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). Method: Hersbach "
    "et al. (2020), QJRMS 146(730), 1999-2049, doi 10.1002/qj.3803. Airport "
    "coordinates: OurAirports, public domain, https://ourairports.com/data/. "
    "No endorsement by ECMWF or the EU is claimed or implied."
)

# icao, latitude, longitude, elevation_m, iata, name
# Lat/lon are reproduced at the full precision published by OurAirports so that
# --verify-coords is an exact byte-level pin rather than a tolerance comparison.
AIRPORTS = [
    ("EHAM", 52.308601, 4.76389, -3.4, "AMS", "Amsterdam Airport Schiphol"),
    ("EDDF", 50.026706, 8.55835, 110.9, "FRA", "Frankfurt Main Airport"),
    ("EDDM", 48.353802, 11.7861, 453.2, "MUC", "Munich Airport"),
    ("EGLL", 51.470748, -0.459909, 25.3, "LHR", "London Heathrow Airport"),
    ("LEBL", 41.2971, 2.07846, 3.7, "BCN", "Josep Tarradellas Barcelona-El Prat Airport"),
    ("LEMD", 40.493407, -3.572249, 609.0, "MAD", "Adolfo Suarez Madrid-Barajas Airport"),
    ("LFPG", 49.00896, 2.554117, 119.5, "CDG", "Charles de Gaulle International Airport"),
    ("LIRF", 41.804532, 12.251998, 4.0, "FCO", "Rome-Fiumicino Leonardo da Vinci International Airport"),
    ("LSZH", 47.458056, 8.548056, 431.9, "ZRH", "Zurich Airport"),
    ("LTFM", 41.274874, 28.732136, 99.1, "IST", "Istanbul Airport"),
]

# candidate variables, mapped to (era5 name, short name, kind, units)
#   kind "instantaneous" -> value is the state VALID AT the timestamp
#   kind "accumulated"    -> value stamped T covers the hour ENDING at T
VARIABLES = [
    ("10m_u_component_of_wind", "u10", "instantaneous", "m s-1"),
    ("10m_v_component_of_wind", "v10", "instantaneous", "m s-1"),
    ("10m_wind_gust_since_previous_post_processing", "fg10", "accumulated", "m s-1"),
    ("total_precipitation", "tp", "accumulated", "m"),
    ("2m_temperature", "t2m", "instantaneous", "K"),
    ("2m_dewpoint_temperature", "d2m", "instantaneous", "K"),
    ("mean_sea_level_pressure", "msl", "instantaneous", "Pa"),
    ("total_cloud_cover", "tcc", "instantaneous", "0-1"),
]

# (label, first day, last day) - the windows V10-A was asked to confirm
#
# V10-B1 CORRECTION: the last day of the 2026 periods is 2026-02-01 / 2026-08-01, not the
# last day of the month. The join aligns hour-ending variables (tp, fg10) with CEIL(MVT), so
# a movement in the 23:00 hour of a month needs the 00:00 record of the FOLLOWING month.
# Stopping on 2026-01-31 / 2026-07-31 left 15 + 11 = 26 ranking departures with NaN
# precipitation and gust. See reports/v10b1_weather_join_audit.txt section 4.
PERIODS = [
    ("2025", date(2025, 1, 1), date(2025, 12, 31)),
    ("2026-01", date(2026, 1, 1), date(2026, 2, 1)),
    ("2026-07", date(2026, 7, 1), date(2026, 8, 1)),
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def snap_to_grid(value, step=ERA5_GRID_STEP):
    return round(value / step) * step


def month_chunks(start, end):
    out = []
    cur = start.replace(day=1)
    while cur <= end:
        last_day = (cur.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        stop = min(last_day, end)
        out.append((cur, stop))
        # V10-B1 CORRECTION: advance from `cur`, not from `stop`. Advancing from `stop`
        # assumed `end` was always a month end: for a non-month-end `end` such as
        # 2026-02-01 the final chunk is (2026-02-01, 2026-02-01), so
        # (stop + 1 day).replace(day=1) == cur and this loop never terminated.
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def build_request(icao, lat, lon, start, stop, data_format="csv"):
    return {
        "variable": [v[0] for v in VARIABLES],
        "location": {"latitude": lat, "longitude": lon},
        "date": f"{start.isoformat()}/{stop.isoformat()}",
        "data_format": data_format,
        "licences": ["cc-by"],
    }


def count_fields(request, gridpoints=1):
    n_vars = len(request["variable"])
    days = request["date"].split("/")[1]
    y, m, d = (int(x) for x in days.split("-"))
    n_times = date(y, m, d).day * 24
    return n_vars * n_times * gridpoints


def verify_coords(airports_csv: Path):
    import csv as _csv

    frozen = {a[0]: (a[1], a[2]) for a in AIRPORTS}
    seen = set()
    problems = []
    with airports_csv.open(newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            ident = row["ident"]
            if ident in frozen:
                seen.add(ident)
                got = (float(row["latitude_deg"]), float(row["longitude_deg"]))
                if got != frozen[ident]:
                    problems.append(
                        f"{ident}: frozen {frozen[ident]} vs {airports_csv.name} {got}"
                    )
    for icao in sorted(set(frozen) - seen):
        problems.append(f"{icao}: not found in {airports_csv.name}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--execute", action="store_true",
                    help="actually call the CDS. Without this flag nothing is downloaded.")
    ap.add_argument("--data-format", default="csv", choices=["csv", "netcdf"],
                    help="csv needs no extra dependency and is the default")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--verify-coords", type=Path, metavar="AIRPORTS_CSV",
                    help="re-derive the frozen coordinates from a local OurAirports CSV and diff")
    ap.add_argument("--max-fields", type=int, default=PUBLISHED_FIELD_LIMIT)
    args = ap.parse_args(argv)

    print("=" * 78)
    print("V10-A ERA5 fetch scaffold")
    print("=" * 78)
    print(f"dataset     : {DATASET}")
    print(f"licence     : CC-BY-4.0  (see module docstring)")
    print(f"coordinates : OurAirports, public domain")
    print(f"              {OURAIRPORTS_BYTES} bytes, sha256 {OURAIRPORTS_SHA256[:16]}...")
    print(f"variables   : {len(VARIABLES)} "
          f"({sum(1 for v in VARIABLES if v[2] == 'instantaneous')} instantaneous, "
          f"{sum(1 for v in VARIABLES if v[2] == 'accumulated')} accumulated)")
    print()

    print("Airport grid assignment (0.25 deg ERA5 regular lat-lon grid)")
    print(f"  {'ICAO':5s} {'lat':>8s} {'lon':>9s} -> {'grid_lat':>8s} {'grid_lon':>8s} {'offset_km':>9s}")
    offsets = []
    for icao, lat, lon, elev, iata, name in AIRPORTS:
        glat, glon = snap_to_grid(lat), snap_to_grid(lon)
        for g in (glat, glon):
            scaled = g / ERA5_GRID_STEP
            if abs(scaled - round(scaled)) > 1e-9:
                print(f"  FAIL {icao}: {g} is not on the {ERA5_GRID_STEP} deg grid")
                return 2
        off = haversine_km(lat, lon, glat, glon)
        offsets.append(off)
        print(f"  {icao:5s} {lat:>8.4f} {lon:>9.4f} -> {glat:>8.2f} {glon:>8.2f} {off:>9.1f}")
    print(f"  offset to nearest grid point: min {min(offsets):.1f} km, "
          f"median {sorted(offsets)[len(offsets)//2]:.1f} km, max {max(offsets):.1f} km")
    print()

    if args.verify_coords:
        problems = verify_coords(args.verify_coords)
        if problems:
            print("COORDINATE VERIFICATION FAILED")
            for p in problems:
                print("  " + p)
            return 2
        print(f"coordinate verification PASSED against {args.verify_coords}")
        print()

    jobs = []
    for label, start, end in PERIODS:
        for a, b in month_chunks(start, end):
            jobs.append((label, a, b))
    total_fields = 0
    per_request = []
    for icao, lat, lon, *_ in AIRPORTS:
        glat, glon = snap_to_grid(lat), snap_to_grid(lon)
        for label, a, b in jobs:
            req = build_request(icao, glat, glon, a, b, args.data_format)
            f = count_fields(req)
            total_fields += f
            per_request.append(f)
            if f > args.max_fields:
                print(f"  ABORT: {icao} {a}..{b} = {f} fields > {args.max_fields}")
                return 2
    print(f"Request plan: {len(AIRPORTS)} airports x {len(jobs)} monthly chunks = "
          f"{len(AIRPORTS) * len(jobs)} requests")
    print(f"  fields per request = {len(VARIABLES)} vars x <=744 h x 1 grid point = "
          f"<= {max(per_request):,} ({max(per_request) / args.max_fields * 100:.1f}% of limit)")
    print(f"  all {len(per_request)} requests within the {args.max_fields}-field limit; "
          f"{total_fields:,} fields total")
    print()
    print("Example request that would be sent (validate against the dataset's")
    print("'Show API request code' button before the first real run):")
    ex = build_request("EHAM", snap_to_grid(52.3086), snap_to_grid(4.7639),
                       jobs[0][1], jobs[0][2], args.data_format)
    for k, v in ex.items():
        print(f"    {k!r}: {v!r}")
    print()

    if not args.execute:
        print("DRY RUN - no network call was made.")
        print("Pass --execute to download. That requires a CDS account, an accepted")
        print("licence in the web form, and 'pip install cdsapi>=0.7.7'.")
        print()
        print("Expected payload for the whole plan: ~819,840 values, ~3-5 MB.")
        print()
        print(ATTRIBUTION)
        return 0

    try:
        import cdsapi
    except ImportError:
        print('cdsapi is not installed. Run: pip install "cdsapi>=0.7.7"')
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    written = 0
    for icao, lat, lon, *_ in AIRPORTS:
        glat, glon = snap_to_grid(lat), snap_to_grid(lon)
        for label, a, b in jobs:
            ext = "csv" if args.data_format == "csv" else "nc"
            # V10-B1 CORRECTION: the chunk dates must be in the filename. The label alone is
            # constant across the 12 monthly chunks of a period ("2025"), so a
            # {icao}_{label}.csv target made chunk 2 onwards collide with chunk 1 and be
            # skipped by the `target.exists()` guard, silently downloading January only.
            target = args.out / f"{icao}_{label}_{a.isoformat()}_{b.isoformat()}.{ext}"
            if target.exists():
                print(f"  skip {target.name} (exists)")
                continue
            req = build_request(icao, glat, glon, a, b, args.data_format)
            print(f"  get {target.name}")
            client.retrieve(DATASET, req, str(target))
            written += 1
    print(f"\nwrote {written} file(s) to {args.out}")
    print("Next: verify each file has a contiguous hourly UTC index before joining,")
    print("and apply the per-variable-kind offsets described in the module docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
