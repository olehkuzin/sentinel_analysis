"""
SentinelHub Statistical API wrapper.

Uses SentinelHubStatistical to compute mean/min/max/stdev server-side per bbox
for a given time range — returns all monthly intervals in a single API call,
avoiding the need to download full rasters for time series.

Run scripts/fetch_all_timeseries.py once to populate data/timeseries/*.parquet.
The dashboard never calls this module at runtime.
"""
import calendar
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    SentinelHubStatistical,
    SHConfig,
    parse_time_interval,
)

# ---------------------------------------------------------------------------
# Copernicus Data Space collections
# DataCollection.SENTINEL2_L2A and SENTINEL3_SLSTR have service_url pointing to
# services.sentinel-hub.com (commercial). SentinelHubStatistical derives its
# endpoint from the collection's service_url, so we must redefine them for CDSE.
# ---------------------------------------------------------------------------
CDSE_URL = "https://sh.dataspace.copernicus.eu"

CDSE_S2_L2A = DataCollection.SENTINEL2_L2A.define_from(
    "CDSE_SENTINEL2_L2A",
    service_url=CDSE_URL,
)
CDSE_S3_SLSTR = DataCollection.SENTINEL3_SLSTR.define_from(
    "CDSE_SENTINEL3_SLSTR",
    service_url=CDSE_URL,
)

import math

from . import geography


def _compute_stats_size(bbox: list[float], target_res_m: int = 1000) -> tuple[int, int]:
    """
    Return (width, height) in pixels so the effective resolution is ~target_res_m
    metres/pixel. S2L2A limit is 1500 m/px; default 1000 m gives a safe margin.
    """
    lon_span = bbox[2] - bbox[0]
    lat_span = bbox[3] - bbox[1]
    centre_lat = (bbox[1] + bbox[3]) / 2
    lon_m = lon_span * 111_320 * math.cos(math.radians(centre_lat))
    lat_m = lat_span * 111_000
    width = max(32, int(lon_m / target_res_m))
    height = max(32, int(lat_m / target_res_m))
    return (width, height)

load_dotenv()

# ---------------------------------------------------------------------------
# Evalscripts — JavaScript VERSION=3, server-side band math
# ---------------------------------------------------------------------------

EVALSCRIPT_NDVI = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["B04", "B08"], units: "REFLECTANCE"}],
    output: [
      {id: "default", bands: 1, sampleType: "FLOAT32"},
      {id: "dataMask", bands: 1}
    ],
    mosaicking: "SIMPLE"
  };
}
function evaluatePixel(samples) {
  let denom = samples.B08 + samples.B04;
  if (denom === 0) return {default: [NaN], dataMask: [0]};
  return {
    default: [(samples.B08 - samples.B04) / denom],
    dataMask: [1]
  };
}
"""

EVALSCRIPT_LST = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["S7"]}],
    output: [
      {id: "default", bands: 1, sampleType: "FLOAT32"},
      {id: "dataMask", bands: 1}
    ]
  };
}
function evaluatePixel(samples) {
  if (samples.S7 === 0) return {default: [NaN], dataMask: [0]};
  return {
    default: [samples.S7 - 273.15],
    dataMask: [1]
  };
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


def _month_intervals(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """
    Generate (start, end) pairs — one per calendar month between
    start_date and end_date (inclusive), formatted as ISO date strings.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    intervals = []
    current = date(start.year, start.month, 1)
    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)
        intervals.append((current.isoformat(), month_end.isoformat()))
        # advance to first day of next month
        current = (date(current.year, current.month, last_day) + timedelta(days=1))
    return intervals


def _parse_stats_response(response: list[dict], variable: str) -> pd.DataFrame:
    """
    Parse raw SentinelHubStatistical response into a tidy DataFrame.
    Each interval becomes one row with date, mean, min, max, stdev, sample_count.
    Intervals with fewer than 10 valid samples are set to NaN.
    """
    rows = []
    for interval_data in response:
        interval = interval_data.get("interval", {})
        date_from = interval.get("from", "")[:10]  # keep only YYYY-MM-DD
        outputs = interval_data.get("outputs", {})
        default = outputs.get("default", {})
        bands = default.get("bands", {})
        b0 = bands.get("B0", {})
        stats = b0.get("stats", {})
        sample_count = stats.get("sampleCount", 0)
        no_data_count = stats.get("noDataCount", 0)
        valid_count = sample_count - no_data_count

        if valid_count < 10:
            rows.append({
                "date": pd.Timestamp(date_from),
                f"{variable}_mean": np.nan,
                f"{variable}_min": np.nan,
                f"{variable}_max": np.nan,
                f"{variable}_std": np.nan,
                f"{variable}_sample_count": valid_count,
            })
        else:
            rows.append({
                "date": pd.Timestamp(date_from),
                f"{variable}_mean": stats.get("mean"),
                f"{variable}_min": stats.get("min"),
                f"{variable}_max": stats.get("max"),
                f"{variable}_std": stats.get("stDev"),
                f"{variable}_sample_count": valid_count,
            })
    return pd.DataFrame(rows)


def fetch_statistical_timeseries(
    bbox: list[float],
    start_date: str,
    end_date: str,
    collection_name: str,
    variable: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch monthly aggregated statistics for a bbox from the Statistical API.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        start_date: ISO date string, e.g. "2020-01-01"
        end_date:   ISO date string, e.g. "2025-12-31"
        collection_name: "SENTINEL2_L2A" or "SENTINEL3_SLSTR"
        variable: "ndvi" or "lst" (used to name output columns)
        max_retries: number of retry attempts on failure

    Returns:
        DataFrame with columns: date, {variable}_mean, {variable}_min,
        {variable}_max, {variable}_std, {variable}_sample_count
    """
    config = _build_config()
    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    evalscript = EVALSCRIPT_NDVI if variable == "ndvi" else EVALSCRIPT_LST

    collection_map = {
        "SENTINEL2_L2A": CDSE_S2_L2A,
        "SENTINEL3_SLSTR": CDSE_S3_SLSTR,
    }
    collection = collection_map[collection_name]

    intervals = _month_intervals(start_date, end_date)

    # Build time_interval list for SentinelHubStatistical
    time_intervals = [(s, e) for s, e in intervals]

    all_rows = []
    for attempt in range(1, max_retries + 1):
        try:
            request = SentinelHubStatistical(
                aggregation=SentinelHubStatistical.aggregation(
                    evalscript=evalscript,
                    time_interval=(start_date, end_date),
                    aggregation_interval="P1M",  # monthly
                    size=_compute_stats_size(bbox),
                ),
                input_data=[
                    SentinelHubStatistical.input_data(
                        collection,
                        # Skip acquisitions with >80% cloud cover (S2 only; ignored by S3)
                        maxcc=0.8,
                    ),
                ],
                bbox=sh_bbox,
                config=config,
            )
            response = request.get_data()[0]
            # response is a dict with key "data" containing the intervals list
            data = response.get("data", [])
            df = _parse_stats_response(data, variable)
            return df
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Statistical API failed after {max_retries} attempts "
                    f"for bbox={bbox}, variable={variable}: {exc}"
                ) from exc
            wait = 2 ** attempt
            print(f"  Attempt {attempt} failed ({exc}), retrying in {wait}s...")
            time.sleep(wait)

    return pd.DataFrame()  # unreachable, satisfies type checker


def fetch_entity_timeseries(
    entity_name: str,
    entity_type: str,
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Fetch NDVI and LST monthly time series for a named geographic entity,
    merge them, and return the canonical TimeSeriesDF.

    Args:
        entity_name: "Czech Republic", a region name from REGIONS, or a city from CITIES
        entity_type: "country" | "region" | "city"
        start_date, end_date: ISO date strings

    Returns:
        DataFrame with columns: year, month, date, entity,
        ndvi_mean, ndvi_min, ndvi_max, ndvi_std,
        lst_mean, lst_min, lst_max, lst_std
    """
    # Resolve bbox
    if entity_type == "country":
        bbox = geography.CZ_BBOX
    elif entity_type == "region":
        bbox = geography.get_region_bbox(entity_name)
    elif entity_type == "city":
        bbox = geography.get_city_bbox(entity_name)
    else:
        raise ValueError(f"Unknown entity_type: {entity_type!r}")

    print(f"  Fetching NDVI for {entity_name} ...")
    ndvi_df = fetch_statistical_timeseries(
        bbox, start_date, end_date, "SENTINEL2_L2A", "ndvi"
    )

    print(f"  Fetching LST for {entity_name} ...")
    lst_df = fetch_statistical_timeseries(
        bbox, start_date, end_date, "SENTINEL3_SLSTR", "lst"
    )

    # Merge on date
    df = pd.merge(ndvi_df, lst_df, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "entity", entity_name)
    df.insert(0, "month", df["date"].dt.month)
    df.insert(0, "year", df["date"].dt.year)

    # Drop helper sample_count columns
    df = df.drop(
        columns=[c for c in df.columns if c.endswith("_sample_count")],
        errors="ignore",
    )

    return df
