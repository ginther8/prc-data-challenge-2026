"""Extract ground/low-altitude ADS-B points near the 10 challenge airports from
adsb.lol daily archives (github.com/adsblol/globe_history_YYYY, ODbL licence).

For each day: stream the split tar (.tar.aa, .tar.ab, ...) straight from GitHub,
parse every trace file in memory, keep only points within 7 km of one of the
10 airports that are on the ground or below airport elevation + 2,500 ft, and write one small
parquet. Nothing large is ever written to disk.

Output : data/external/adsb_v2/points_YYYY-MM-DD.parquet
         columns: hex, t (unix s), lat, lon, alt (-1 = ground), ground,
                  gs (kt), track, flight (callsign), apt
Manifest: data/external/adsb_v2/manifest.jsonl  (one line per finished day)

Resumable: a day whose parquet already exists is skipped.
Atomic   : each day is written to a .tmp file and renamed when complete.

Run from the project root:
    python experiments/adsb_extract.py --workers 3            # default 124 days
    python experiments/adsb_extract.py --workers 8            # after benchmark
    python experiments/adsb_extract.py --days 2025-07-15      # single day
    python experiments/adsb_extract.py --test                 # 3,000 files only
    python experiments/adsb_extract.py --audit                # check archives used
    python experiments/adsb_extract.py --audit --repair --workers 8   # fix mismatches

Archive choice: the day's entry in adsb.lol's PREFERRED_RELEASES.txt first,
then prod-0, prod-1, staging-0, prod-0tmp, staging-0tmp. A corrupt archive
(zlib/tar error part-way) falls through to the next replica.

Default days: Jan + Jul 2025 (training / validation) and Jan + Jul 2026
(ranking).

Attribution required if used in the final model:
    ADS-B data (c) adsb.lol contributors, Open Database License (ODbL) 1.0.
"""

import argparse
import statistics
import datetime as dt
import gzip
import json
import math
import os
import sys
import tarfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "external" / "adsb_v2"   # v2 = elevation-relative filter
MANIFEST = OUT_DIR / "manifest.jsonl"

AIRPORTS = {  # aerodrome reference points
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861),
    "EGLL": (51.4700, -0.4543), "EHAM": (52.3086, 4.7639),
    "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
    "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519),
}
RADIUS_KM = 7.0
# ADS-B altitude is PRESSURE altitude: on the ground it reads about the airport
# elevation +- ~800 ft depending on the weather. The limit is therefore relative
# to each airport's elevation (v1 used an absolute 1,500 ft, which silently
# dropped unflagged ground traffic at LEMD 1,998 ft, EDDM 1,487 ft, LSZH 1,416 ft).
ELEV_FT = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 14,
           "LEMD": 1998, "LFPG": 392, "LIRF": 13, "LSZH": 1416, "LTFM": 325}
ABOVE_ELEV_FT = 2500
MAX_ALT_ANY = max(ELEV_FT.values()) + ABOVE_ELEV_FT
SOURCES = ["prod-0", "prod-1", "staging-0", "prod-0tmp", "staging-0tmp"]  # fallback order
PREFERRED_URL = ("https://raw.githubusercontent.com/adsblol/globe_history_{y}/"
                 "main/PREFERRED_RELEASES.txt")
_PREFERRED = {}
MIN_DAY_BYTES = 1_000_000_000                  # smaller => try next source
CHUNK = 1 << 20

_COS = {k: math.cos(math.radians(v[0])) for k, v in AIRPORTS.items()}


def nearest_airport(lat, lon):
    for k, (a, b) in AIRPORTS.items():
        dy = (lat - a) * 111.0
        dx = (lon - b) * 111.0 * _COS[k]
        if dx * dx + dy * dy < RADIUS_KM * RADIUS_KM:
            return k
    return None


def default_days():
    days = []
    for y, m in ((2025, 1), (2025, 7), (2026, 1), (2026, 7)):
        d = dt.date(y, m, 1)
        while d.month == m:
            days.append(d.isoformat())
            d += dt.timedelta(days=1)
    return days


class ConcatStream:
    """File-like object reading several HTTP responses back to back."""

    def __init__(self, urls, session):
        self.urls = list(urls)
        self.session = session
        self.resp = None
        self.it = None
        self.buf = b""
        self.bytes_read = 0
        self._next()

    def _next(self):
        if self.resp is not None:
            self.resp.close()
        if not self.urls:
            self.resp, self.it = None, None
            return
        url = self.urls.pop(0)
        self.resp = self.session.get(url, stream=True, timeout=120)
        self.resp.raise_for_status()
        self.it = self.resp.iter_content(CHUNK)

    def read(self, n=-1):
        while (n < 0 or len(self.buf) < n) and self.it is not None:
            try:
                chunk = next(self.it)
                self.buf += chunk
                self.bytes_read += len(chunk)
            except StopIteration:
                self._next()
        if n < 0:
            out, self.buf = self.buf, b""
        else:
            out, self.buf = self.buf[:n], self.buf[n:]
        return out


def preferred(day, session):
    """URL list adsb.lol names as the preferred archive for this day, or None.
    Checks the day's own year and the next year's list (e.g. 2025-12-31 is
    published in globe_history_2026)."""
    for y in (day[:4], str(int(day[:4]) + 1)):
        hit = _preferred_in(y, day, session)
        if hit:
            return hit
    return None


def _preferred_in(y, day, session):
    if y not in _PREFERRED:
        _PREFERRED[y] = {}
        try:
            r = session.get(PREFERRED_URL.format(y=y), timeout=60)
            if r.status_code == 200:
                for line in r.text.splitlines():
                    urls = [u.strip() for u in line.split(",") if u.strip()]
                    if urls and "/v" in urls[0]:
                        tag = urls[0].split("/download/")[1].split("/")[0]
                        d = tag[1:11].replace(".", "-")
                        _PREFERRED[y][d] = (urls, tag)
        except Exception:
            pass
    return _PREFERRED[y].get(day)


def part_urls(day, source, session):
    urls, total, tag = _part_urls_repo(day, source, session, day[:4])
    if not urls:  # year-boundary days can live in the next year's repository
        urls, total, tag = _part_urls_repo(day, source, session, str(int(day[:4]) + 1))
    return urls, total, tag


def _part_urls_repo(day, source, session, repo_year):
    """Return the ordered list of existing part URLs for a day/source."""
    y = repo_year
    tag = f"v{day.replace('-', '.')}-planes-readsb-{source}"
    base = (f"https://github.com/adsblol/globe_history_{y}/releases/download/"
            f"{tag}/{tag}.tar")
    urls, total = [], 0
    for suffix in ("aa", "ab", "ac", "ad", "ae", "af"):
        r = session.head(base + "." + suffix, allow_redirects=True, timeout=60)
        if r.status_code != 200:
            break
        urls.append(base + "." + suffix)
        total += int(r.headers.get("Content-Length", 0))
    if not urls:  # some days are a single un-split .tar
        r = session.head(base, allow_redirects=True, timeout=60)
        if r.status_code == 200:
            urls = [base]
            total = int(r.headers.get("Content-Length", 0))
    return urls, total, tag


def candidate_archives(day, session):
    """Preferred archive first, then the other replicas (for corrupt archives)."""
    cands = []
    pref = preferred(day, session)
    if pref:
        cands.append((pref[0], 0, pref[1]))
    for src in SOURCES:
        urls, total, tag = part_urls(day, src, session)
        if urls and all(tag != c[2] for c in cands):
            cands.append((urls, total, tag))
    return cands


def extract_day(day, max_files=None):
    out_path = OUT_DIR / f"points_{day}.parquet"
    if out_path.exists():
        return {"day": day, "status": "skipped"}
    session = requests.Session()
    last_err = None
    for urls, total, tag in candidate_archives(day, session):
        try:
            return _extract_from(day, urls, total, tag, session, out_path, max_files)
        except Exception as e:  # corrupt archive (zlib/tar error) -> next replica
            last_err = f"{tag}: {type(e).__name__}: {e}"
            tmp = out_path.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()
            print(f"  {day}: {last_err} - trying next replica", flush=True)
    if last_err:
        raise RuntimeError(last_err)
    return {"day": day, "status": "no_release"}


def _old_source_choice(day, session):
    chosen = None
    for src in SOURCES:
        urls, total, tag = part_urls(day, src, session)
        if urls and total >= MIN_DAY_BYTES:
            chosen = (urls, total, tag)
            break
        if urls and chosen is None:
            chosen = (urls, total, tag)   # keep as fallback if all are small
    return chosen


def _extract_from(day, urls, total, tag, session, out_path, max_files):

    cols = {k: [] for k in ("hex", "t", "lat", "lon", "alt", "ground", "gs",
                            "track", "flight", "apt")}
    t0 = time.time()
    stream = ConcatStream(urls, session)
    nfiles = 0
    print(f"  {day}: started ({tag}, {len(urls)} parts, "
          f"{total/1e9:.2f} GB)", flush=True)
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or "trace_full" not in m.name:
                continue
            nfiles += 1
            raw = tf.extractfile(m).read()
            try:
                data = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b"
                                  else raw)
            except Exception:
                continue
            base = data.get("timestamp", 0)
            hx = data.get("icao", "")
            flight = ""
            for p in data.get("trace", []):
                if len(p) > 8 and isinstance(p[8], dict) and p[8].get("flight"):
                    flight = p[8]["flight"].strip()
                lat, lon, alt = p[1], p[2], p[3]
                if lat is None or lon is None:
                    continue
                ground = alt == "ground"
                if not ground and (alt is None or alt > MAX_ALT_ANY):
                    continue
                apt = nearest_airport(lat, lon)
                if apt is None:
                    continue
                if not ground and alt > ELEV_FT[apt] + ABOVE_ELEV_FT:
                    continue
                cols["hex"].append(hx)
                cols["t"].append(float(base + p[0]))
                cols["lat"].append(float(lat))
                cols["lon"].append(float(lon))
                cols["alt"].append(-1 if ground else int(alt))
                cols["ground"].append(ground)
                cols["gs"].append(float(p[4]) if p[4] is not None else float("nan"))
                cols["track"].append(float(p[5]) if p[5] is not None else float("nan"))
                cols["flight"].append(flight)
                cols["apt"].append(apt)
            if nfiles % 10000 == 0:
                print(f"  {day}: {nfiles:,} files, {len(cols['hex']):,} pts, "
                      f"{stream.bytes_read/1e9:.2f}/{total/1e9:.2f} GB, "
                      f"{time.time()-t0:.0f}s", flush=True)
            if max_files and nfiles >= max_files:
                break

    tmp = out_path.with_suffix(".tmp")
    pq.write_table(pa.table(cols), tmp)
    os.replace(tmp, out_path)
    per_apt = {}
    for a in cols["apt"]:
        per_apt[a] = per_apt.get(a, 0) + 1
    info = {"day": day, "status": "ok", "tag": tag, "parts": len(urls),
            "bytes": stream.bytes_read, "release_bytes": total,
            "trace_files": nfiles, "points": len(cols["hex"]),
            "points_per_airport": per_apt, "seconds": round(time.time() - t0, 1)}
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(info) + "\n")
    return info


def worker(args):
    day, max_files = args
    for attempt in range(1, 4):
        try:
            return extract_day(day, max_files)
        except Exception as e:  # network hiccup etc.
            tmp = OUT_DIR / f"points_{day}.tmp"
            if tmp.exists():
                tmp.unlink()
            err = f"{type(e).__name__}: {e}"
            if attempt == 3:
                return {"day": day, "status": "failed", "error": err,
                        "trace": traceback.format_exc()[-800:]}
            time.sleep(30 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--days", default="", help="comma-separated YYYY-MM-DD")
    ap.add_argument("--audit", action="store_true",
                    help="compare each extracted day's archive with adsb.lol's "
                         "PREFERRED_RELEASES.txt; with --repair, re-extract mismatches")
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="one day, first 3,000 trace files only")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.audit:
        import collections
        used = {}
        if MANIFEST.exists():
            for line in MANIFEST.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    used[r["day"]] = r
                except Exception:
                    pass
        session = requests.Session()
        days = sorted(p.stem.replace("points_", "") for p in OUT_DIR.glob("points_*.parquet"))
        wanted = ([d.strip() for d in args.days.split(",") if d.strip()] or days)
        bad, missing = [], [d for d in wanted if d not in days]
        for d in wanted:
            if d not in days:
                continue
            pref = preferred(d, session)
            tag = used.get(d, {}).get("tag", "?")
            if pref and pref[1] != tag:
                bad.append((d, tag, pref[1]))
        pts = [used[d]["points"] for d in days if d in used]
        med = statistics.median(pts) if pts else 0
        small = [d for d in days if d in used and used[d]["points"] < 0.5 * med]
        print(f"days extracted: {len(days)}   missing from request: {len(missing)}")
        print(f"days not from the preferred archive: {len(bad)}")
        for d, a, b in bad:
            print(f"  {d}: used {a}   preferred {b}")
        print(f"days with < 50% of median points ({med:,.0f}): {small or 'none'}")
        if args.repair and (bad or missing):
            for d, _, _ in bad:
                (OUT_DIR / f"points_{d}.parquet").unlink(missing_ok=True)
            redo = [d for d, _, _ in bad] + missing
            print(f"re-extracting {len(redo)} days with {args.workers} workers", flush=True)
            with Pool(args.workers) as pool:
                for info in pool.imap_unordered(worker, [(d, None) for d in redo]):
                    print(f"  {info['day']} {info['status']} {info.get('tag', '')} "
                          f"{info.get('error', '')}", flush=True)
        print("DONE")
        return

    if args.test:
        day = "2025-07-15"
        (OUT_DIR / f"points_{day}.parquet").unlink(missing_ok=True)
        info = extract_day(day, max_files=3000)
        print(json.dumps(info, indent=1))
        (OUT_DIR / f"points_{day}.parquet").unlink(missing_ok=True)
        print("TEST OK (test output deleted)")
        return

    days = ([d.strip() for d in args.days.split(",") if d.strip()]
            or default_days())
    todo = [d for d in days if not (OUT_DIR / f"points_{d}.parquet").exists()]
    print(f"{len(days)} days requested, {len(days)-len(todo)} already done, "
          f"{len(todo)} to process with {args.workers} workers", flush=True)
    t0 = time.time()
    done = 0
    failed = []
    with Pool(args.workers) as pool:
        for info in pool.imap_unordered(worker, [(d, None) for d in todo]):
            done += 1
            if info["status"] == "ok":
                print(f"[{done}/{len(todo)}] {info['day']} ok  "
                      f"{info['points']:,} pts  {info['trace_files']:,} files  "
                      f"{info['bytes']/1e9:.2f} GB  {info['seconds']:.0f}s  "
                      f"elapsed {(time.time()-t0)/60:.0f} min", flush=True)
            else:
                print(f"[{done}/{len(todo)}] {info['day']} {info['status']} "
                      f"{info.get('error', '')}", flush=True)
                if info["status"] in ("failed", "no_release"):
                    failed.append(info["day"])
    print(f"finished in {(time.time()-t0)/60:.0f} min; "
          f"failed/missing days: {failed or 'none'}")
    print("DONE")


if __name__ == "__main__":
    main()