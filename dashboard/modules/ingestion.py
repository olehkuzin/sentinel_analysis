"""
On-demand raster fetching via SentinelHubRequest.
Used only for map overlays (one raster per selected month/year/geography).
Results are cached by Streamlit via @st.cache_data(persist="disk").
"""
import calendar
import os

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
# Calling define_from again with the same definition would raise ValueError.
from dashboard.modules.statistics_api import CDSE_S2_L2A as _S2_L2A
from dashboard.modules.statistics_api import CDSE_S3_SLSTR as _S3_SLSTR

load_dotenv()

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
    input: ["S7"],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.S7 - 273.15];
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


def fetch_ndvi_raster(
    bbox: list[float],
    size: tuple[int, int],
    year: int,
    month: int,
) -> np.ndarray:
    """
    Fetch an NDVI raster for the given bbox, size (W, H), year, and month
    using Sentinel-2 L2A with least-cloud-cover mosaicking.

    Returns a 2D float32 array of shape (H, W) with values in [-1, 1].
    NaN = cloud-masked or no data.
    """
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
    data = request.get_data()[0]          # shape: (H, W, 1) or (H, W)
    arr = np.squeeze(data).astype(np.float32)
    arr[arr == 0] = np.nan                # sentinel value from some APIs
    return arr


def fetch_lst_raster(
    bbox: list[float],
    size: tuple[int, int],
    year: int,
    month: int,
) -> np.ndarray:
    """
    Fetch a Land Surface Temperature raster for the given bbox, size (W, H),
    year, and month using Sentinel-3 SLSTR (S7 band, Kelvin → Celsius).

    Returns a 2D float32 array of shape (H, W) in degrees Celsius.
    NaN = no data.
    """
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
    return arr
