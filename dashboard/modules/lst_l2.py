"""
Sentinel-3 SLSTR Level-2 LST (Land Surface Temperature) downloader.

Downloads actual atmospherically-corrected LST from CDSE catalog,
giving real surface temperatures (e.g. 25-40°C in summer) instead of
brightness temperature from the Process API.

Requires CDSE_USERNAME and CDSE_PASSWORD in .env.
"""
import io
import os
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from dotenv import load_dotenv
from scipy.interpolate import griddata

load_dotenv()

_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "rasters"
_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_ZIPPER_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


def _get_token() -> str:
    resp = requests.post(_TOKEN_URL, data={
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": os.environ["CDSE_USERNAME"],
        "password": os.environ["CDSE_PASSWORD"],
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def _search_products(bbox: list[float], year: int, month: int, max_results: int = 10) -> list[dict]:
    """Search CDSE catalog for S3 LST L2 products covering the bbox."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2

    params = {
        "$filter": (
            "Collection/Name eq 'SENTINEL-3' and "
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            "and att/OData.CSC.StringAttribute/Value eq 'SL_2_LST___') and "
            f"ContentDate/Start gt {year}-{month:02d}-01T00:00:00.000Z and "
            f"ContentDate/Start lt {year}-{month:02d}-{last_day:02d}T23:59:59.999Z and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({center_lon} {center_lat})')"
        ),
        "$top": max_results,
        "$orderby": "ContentDate/Start desc",
    }
    resp = requests.get(_CATALOG_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("value", [])


def _download_nc(product_id: str, sen3_name: str, filename: str, token: str) -> bytes:
    """Download a single NetCDF file from a product."""
    url = f"{_ZIPPER_URL}({product_id})/Nodes({sen3_name})/Nodes({filename})/$value"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    resp.raise_for_status()
    return resp.content


def _regrid_to_bbox(
    lst_vals: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    bbox: list[float],
    size: tuple[int, int],
) -> np.ndarray:
    """Regrid irregular swath data to a regular grid matching bbox/size."""
    w, h = size  # (width, height)
    lon_min, lat_min, lon_max, lat_max = bbox

    # Target regular grid
    grid_lon = np.linspace(lon_min, lon_max, w)
    grid_lat = np.linspace(lat_max, lat_min, h)  # top-down
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

    # Flatten source data
    points = np.column_stack([lon.ravel(), lat.ravel()])
    values = lst_vals.ravel()

    # Remove NaN/invalid points
    valid = np.isfinite(values) & np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    valid &= values > 200  # must be > 200 K (filter fill values)
    points = points[valid]
    values = values[valid]

    # Interpolate to regular grid
    result = griddata(points, values, (grid_lon_2d, grid_lat_2d), method="nearest")
    result = result.astype(np.float32)

    # Mask pixels far from any source point (no data coverage)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    query_pts = np.column_stack([grid_lon_2d.ravel(), grid_lat_2d.ravel()])
    dists, _ = tree.query(query_pts)
    # If nearest source pixel is > 0.05° away (~5km), mark as no-data
    no_coverage = dists.reshape(h, w) > 0.05
    result[no_coverage] = np.nan

    return result


def fetch_lst_l2(
    bbox: list[float],
    size: tuple[int, int],
    year: int,
    month: int,
) -> np.ndarray:
    """
    Fetch real LST (°C) from Sentinel-3 SLSTR Level-2 product.

    Searches CDSE catalog, downloads the best product, filters clouds,
    regrids to the requested bbox/size, and caches to disk.

    Returns 2D float32 array in °C. NaN = no data / cloud.
    """
    # Check cache
    bbox_str = "_".join(f"{v:.4f}" for v in bbox)
    cache_key = f"lst_l2_{bbox_str}_{size[0]}x{size[1]}_{year}_{month:02d}.npy"
    cache_path = _CACHE_DIR / cache_key
    if cache_path.exists():
        return np.load(cache_path)

    w, h = size
    print(f"  Fetching LST L2 for {year}-{month:02d}...")

    # Search for products
    products = _search_products(bbox, year, month, max_results=30)
    if not products:
        print("  No S3 LST L2 products found!")
        return np.full((size[1], size[0]), np.nan, dtype=np.float32)

    token = _get_token()

    # Filter for daytime passes (08:00-12:00 UTC over Central Europe)
    daytime = []
    for p in products:
        # Product name format: S3x_SL_2_LST____YYYYMMDDTHHmmss_...
        name = p["Name"]
        try:
            hour = int(name[25:27])
            if 8 <= hour <= 12:
                daytime.append(p)
        except (IndexError, ValueError):
            daytime.append(p)  # keep if we can't parse
    if daytime:
        products = daytime

    # Composite: accumulate clear-sky LST from multiple products
    w, h = size
    accum = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.int32)

    for i, product in enumerate(products[:8]):  # try up to 8 daytime products
        product_id = product["Id"]
        sen3_name = product["Name"]
        print(f"    [{i+1}/{min(len(products),8)}] {sen3_name[:50]}...", end=" ", flush=True)

        try:
            lst_nc = _download_nc(product_id, sen3_name, "LST_in.nc", token)
            geo_nc = _download_nc(product_id, sen3_name, "geodetic_in.nc", token)
            flg_nc = _download_nc(product_id, sen3_name, "flags_in.nc", token)

            lst_ds = xr.open_dataset(io.BytesIO(lst_nc), engine="h5netcdf")
            geo_ds = xr.open_dataset(io.BytesIO(geo_nc), engine="h5netcdf")
            flg_ds = xr.open_dataset(io.BytesIO(flg_nc), engine="h5netcdf")

            lst = lst_ds["LST"].values
            lat = geo_ds["latitude_in"].values
            lon = geo_ds["longitude_in"].values
            cloud = flg_ds["cloud_in"].values

            # Mask clouds: use both binary flag and probability
            lst[cloud != 0] = np.nan
            lst[lst < 220] = np.nan
            # Also use cloud probability if available
            if "probability_cloud_dual_in" in flg_ds:
                prob = flg_ds["probability_cloud_dual_in"].values
                lst[prob > 10] = np.nan  # >10% cloud probability → mask

            # Regrid this product
            grid = _regrid_to_bbox(lst, lat, lon, bbox, size)
            valid = np.isfinite(grid) & (grid > 220)
            n_valid = np.sum(valid)
            accum[valid] += grid[valid] - 273.15
            count[valid] += 1
            print(f"{n_valid} px")

        except Exception as e:
            print(f"failed: {e}")
            continue

    # Compute mean composite
    result = np.full((h, w), np.nan, dtype=np.float32)
    has_data = count > 0
    result[has_data] = (accum[has_data] / count[has_data]).astype(np.float32)
    # Season-aware minimum filter (cloud contamination gives cold pixels)
    summer = month in (5, 6, 7, 8, 9)
    min_temp = 10.0 if summer else -10.0
    result[result < min_temp] = np.nan
    result[result > 55] = np.nan

    coverage = np.sum(has_data) / (h * w)
    if np.any(has_data):
        print(f"  Composite: {coverage:.0%} coverage, {np.nanmin(result):.1f} to {np.nanmax(result):.1f} °C, mean {np.nanmean(result):.1f} °C")
    else:
        print("  No valid LST data found")

    # Cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result)
    return result
