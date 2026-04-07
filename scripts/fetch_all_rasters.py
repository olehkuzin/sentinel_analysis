"""
Pre-fetch all rasters for 2025 and save to data/rasters/.
Run once — dashboard then loads from disk instantly.

Usage:
    uv run python scripts/fetch_all_rasters.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinelhub import BBox, CRS, bbox_to_dimensions

from dashboard.modules import geography, ingestion, preprocessing

YEAR = 2025
MONTHS = list(range(1, 13))

# Country overview: 500m resolution
CZ_BBOX = geography.CZ_BBOX
CZ_SIZE = geography.CZ_RASTER_SIZE[::-1]  # (W, H)

# Region/city resolution: 120m
RES = 120
MAX_DIM = 2500

_geojson = geography.load_regions_geojson()

# Download city boundaries if not present
_cities_gj = geography.load_cities_geojson()
if _cities_gj is None:
    print("City boundaries not found — downloading from OpenStreetMap...\n")
    _cities_gj = geography.download_cz_city_boundaries()
    print()


def _capped_size(bbox, resolution=RES):
    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    size = bbox_to_dimensions(sh_bbox, resolution=resolution)
    if size[0] > MAX_DIM or size[1] > MAX_DIM:
        scale = MAX_DIM / max(size)
        size = (int(size[0] * scale), int(size[1] * scale))
    return size


def fetch_pair(label: str, bbox, size, month: int):
    print(f"  {label} NDVI {YEAR}-{month:02d}...", end=" ", flush=True)
    t0 = time.time()
    ingestion.fetch_ndvi_raster(bbox, size, YEAR, month)
    print(f"{time.time()-t0:.1f}s")

    print(f"  {label} LST  {YEAR}-{month:02d}...", end=" ", flush=True)
    t0 = time.time()
    ingestion.fetch_lst_raster(bbox, size, YEAR, month)
    print(f"{time.time()-t0:.1f}s")


def main():
    regions = list(geography.REGIONS.keys())
    cities = list(geography.CITIES.keys())
    n_areas = 1 + len(regions) + len(cities)  # country + regions + cities
    total = len(MONTHS) * n_areas * 2
    print(f"Pre-fetching {total} rasters for {YEAR}")
    print(f"  12 months × ({1} country + {len(regions)} regions + {len(cities)} cities) × 2 vars")
    print(f"  Cache dir: {ingestion.RASTER_CACHE_DIR}\n")

    done = 0
    for month in MONTHS:
        print(f"=== Month {month:02d} ===")

        # Country
        fetch_pair("Country", CZ_BBOX, CZ_SIZE, month)
        done += 2

        # Regions
        for name in regions:
            bbox = geography.get_region_bbox_from_geojson(name, _geojson)
            size = _capped_size(bbox)
            fetch_pair(name, bbox, size, month)
            done += 2
            print(f"    [{done}/{total}]")

        # Cities (use real boundary bbox if available)
        for name in cities:
            if _cities_gj:
                bbox = geography.get_city_bbox_from_geojson(name, _cities_gj)
            else:
                bbox = geography.get_city_bbox(name, buffer_deg=0.15)
            size = _capped_size(bbox)
            fetch_pair(name, bbox, size, month)
            done += 2
            print(f"    [{done}/{total}]")

        print()

    print(f"Done! {total} rasters cached in {ingestion.RASTER_CACHE_DIR}")


if __name__ == "__main__":
    main()
