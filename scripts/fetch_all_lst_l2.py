"""
Pre-fetch Sentinel-3 LST Level-2 data for analysis (scatter, cooling index, UHI).
Downloads real surface temperature from CDSE catalog.

Usage:
    uv run python scripts/fetch_all_lst_l2.py

Runs 4 downloads in parallel. Skips already cached files.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.modules import geography
from dashboard.modules.lst_l2 import fetch_lst_l2

YEAR = 2025
MONTHS = [5, 6, 7, 8, 9]  # Summer only — most relevant for heat analysis
SIZE = (128, 96)  # low res — analysis doesn't need full resolution
WORKERS = 4  # parallel downloads

_geojson = geography.load_regions_geojson()


def _fetch_one(args):
    name, bbox, month = args
    t0 = time.time()
    try:
        fetch_lst_l2(bbox, SIZE, YEAR, month)
        return name, month, time.time() - t0, None
    except Exception as e:
        return name, month, time.time() - t0, str(e)


def main():
    # Build task list
    areas = [("Czech Republic", geography.CZ_BBOX)]
    for name in geography.REGIONS:
        areas.append((name, geography.get_region_bbox_from_geojson(name, _geojson)))
    for name in geography.CITIES:
        areas.append((name, geography.get_city_bbox(name, buffer_deg=0.15)))

    tasks = [(name, bbox, month) for month in MONTHS for name, bbox in areas]
    total = len(tasks)
    print(f"Pre-fetching {total} LST L2 rasters ({WORKERS} parallel workers)")
    print(f"Areas: {len(areas)}, Months: {MONTHS}\n")

    done = 0
    failed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch_one, task): task for task in tasks}
        for future in as_completed(futures):
            name, month, elapsed, error = future.result()
            done += 1
            if error:
                failed += 1
                print(f"  [{done}/{total}] {name} {YEAR}-{month:02d} FAILED: {error}")
            else:
                print(f"  [{done}/{total}] {name} {YEAR}-{month:02d} — {elapsed:.1f}s")

    elapsed_total = time.time() - t_start
    print(f"\nDone! {done - failed}/{total} cached in {elapsed_total/60:.0f} min")
    if failed:
        print(f"  {failed} failed — re-run to retry")


if __name__ == "__main__":
    main()
