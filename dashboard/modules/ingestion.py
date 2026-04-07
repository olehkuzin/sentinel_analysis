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
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

# SentinelHubRequest routes via config.sh_base_url, so the default collections
# work fine with the CDSE config — no define_from needed here (unlike the
# Statistical API which routes via the collection's own service_url).
_S2_L2A = DataCollection.SENTINEL2_L2A
_S3_SLSTR = DataCollection.SENTINEL3_SLSTR

load_dotenv()

# ---------------------------------------------------------------------------
# Evalscripts
# ---------------------------------------------------------------------------

EVALSCRIPT_NDVI = """
//VERSION=3
function setup() {
  return {
    input: [
      {bands: ["B04", "B08"], units: "REFLECTANCE"},
      {bands: ["SCL"], units: "DN"}
    ],
    output: [{bands: 1, sampleType: "FLOAT32"}]
  };
}
function evaluatePixel(samples) {
  // SCL: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=cirrus
  if ([3, 8, 9, 10].includes(samples.SCL)) return [NaN];
  let denom = samples.B08 + samples.B04;
  if (denom === 0) return [NaN];
  return [(samples.B08 - samples.B04) / denom];
}
"""

EVALSCRIPT_LST = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["S7"]}],
    output: [{bands: 1, sampleType: "FLOAT32"}]
  };
}
function evaluatePixel(samples) {
  if (samples.S7 === 0) return [NaN];
  return [samples.S7 - 273.15];
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
