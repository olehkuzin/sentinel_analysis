"""
On-demand raster fetching via SentinelHubRequest with local disk cache.

First call fetches from Sentinel Hub and saves to data/rasters/.
Subsequent calls load from disk instantly.
Run scripts/fetch_all_rasters.py to pre-populate the cache.
"""
import calendar
import hashlib
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from sentinelhub import (
    BBox,
    CRS,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

# Reuse the CDSE collections already registered by statistics_api.py.
from dashboard.modules.statistics_api import CDSE_S2_L2A as _S2_L2A
from dashboard.modules.statistics_api import CDSE_S3_SLSTR as _S3_SLSTR

load_dotenv()

RASTER_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "rasters"

# ---------------------------------------------------------------------------
# Evalscripts
# ---------------------------------------------------------------------------

EVALSCRIPT_NDVI = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B08", "dataMask"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [NaN];
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [ndvi];
}
"""

EVALSCRIPT_LST = """
//VERSION=3
function setup() {
  return {
    input: ["S8", "S9", "dataMask"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  if (s.dataMask === 0 || s.S8 === 0 || s.S9 === 0) return [NaN];
  // Split-window correction: improves raw brightness temp
  var lst_k = s.S8 + 1.5 * (s.S8 - s.S9) + 0.5;
  return [lst_k - 273.15];
}
"""


def _build_config() -> SHConfig:
    """Load Copernicus Data Space credentials from .env."""
    config = SHConfig()
    config.sh_client_id = os.environ["SH_CLIENT_ID"]
    config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
        "/protocol/openid-connect/token"
    )
    return config


def _month_time_interval(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}")


def _cache_key(variable: str, bbox: list[float], size: tuple[int, int], year: int, month: int) -> str:
    """Deterministic filename for a raster request."""
    bbox_str = "_".join(f"{v:.4f}" for v in bbox)
    return f"{variable}_{bbox_str}_{size[0]}x{size[1]}_{year}_{month:02d}.npy"


def _load_cached(key: str) -> np.ndarray | None:
    path = RASTER_CACHE_DIR / key
    if path.exists():
        return np.load(path)
    return None


def _save_cached(key: str, arr: np.ndarray) -> None:
    RASTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RASTER_CACHE_DIR / key, arr)


def fetch_ndvi_raster(
    bbox: list[float],
    size: tuple[int, int],
    year: int,
    month: int,
) -> np.ndarray:
    """Fetch NDVI raster, loading from disk cache if available."""
    key = _cache_key("ndvi", bbox, size, year, month)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    config = _build_config()
    time_interval = _month_time_interval(year, month)
    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_NDVI,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=_S2_L2A,
                time_interval=time_interval,
                mosaicking_order="leastCC",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=sh_bbox,
        size=size,
        config=config,
    )
    data = request.get_data()[0]
    arr = np.squeeze(data).astype(np.float32)
    arr[arr == 0] = np.nan
    _save_cached(key, arr)
    return arr


def fetch_lst_raster(
    bbox: list[float],
    size: tuple[int, int],
    year: int,
    month: int,
) -> np.ndarray:
    """Fetch LST (brightness temp, °C) via Sentinel Hub Process API."""
    key = _cache_key("lst", bbox, size, year, month)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    config = _build_config()
    time_interval = _month_time_interval(year, month)
    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_LST,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=_S3_SLSTR,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=sh_bbox,
        size=size,
        config=config,
    )
    data = request.get_data()[0]
    arr = np.squeeze(data).astype(np.float32)
    arr[arr == 0] = np.nan
    _save_cached(key, arr)
    return arr
