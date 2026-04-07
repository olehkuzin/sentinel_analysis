"""
Page 2 — Regions
Raster map clipped to CZ borders with NUTS-3 region boundaries,
selected region highlighted. Regional statistics and time series.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from dashboard.modules import (
    aggregation,
    geography,
    ingestion,
    ndvi as ndvi_mod,
    lst as lst_mod,
    preprocessing,
    visualization,
)

st.set_page_config(page_title="Regions", page_icon="🗾", layout="wide")
st.title("Czech Republic — Regions")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
variable = st.sidebar.radio("Variable", ["NDVI", "LST"], key="reg_var")
selected_region = st.sidebar.selectbox("Region", list(geography.REGIONS.keys()), key="reg_sel")
map_month = st.sidebar.selectbox(
    "Map month",
    options=list(range(1, 13)),
    format_func=lambda m: [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ][m-1],
    index=6,
    key="reg_map_mo",
)

YEAR = 2025
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_ts():
    try:
        return aggregation.load_timeseries("regions")
    except FileNotFoundError as e:
        return str(e)

@st.cache_data(persist="disk")
def load_region_raster(region_name: str, year: int, month: int):
    """Fetch raster at high resolution for just the selected region's bbox."""
    geojson = geography.load_regions_geojson()
    bbox = geography.get_region_bbox_from_geojson(region_name, geojson)
    from sentinelhub import BBox, CRS, bbox_to_dimensions
    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    # 120m resolution — good quality, fast fetch (~5-8s per request)
    size = bbox_to_dimensions(sh_bbox, resolution=120)
    max_dim = 2500
    if size[0] > max_dim or size[1] > max_dim:
        scale = max_dim / max(size)
        size = (int(size[0] * scale), int(size[1] * scale))
    ndvi_arr = ingestion.fetch_ndvi_raster(bbox, size, year, month)
    lst_arr = ingestion.fetch_lst_raster(bbox, size, year, month)
    ndvi_arr = preprocessing.mask_invalid(ndvi_arr, -1.0, 1.0)
    lst_arr = preprocessing.mask_invalid(lst_arr, -20.0, 60.0)
    ndvi_arr, lst_arr = preprocessing.align_resolutions(ndvi_arr, lst_arr)
    return ndvi_arr, lst_arr, bbox

ts_data = load_ts()

if isinstance(ts_data, str):
    st.error(ts_data)
    st.stop()

ts_filtered = aggregation.filter_by_year_range(ts_data, YEAR, YEAR)
value_col = "ndvi_mean" if variable == "NDVI" else "lst_mean"

geojson = geography.load_regions_geojson()

# ---------------------------------------------------------------------------
# Row 1: Raster map with region borders (only selected highlighted)
# ---------------------------------------------------------------------------
st.subheader(f"{variable} — {selected_region} — {MONTH_NAMES[map_month-1]} {YEAR}")
with st.spinner(f"Fetching raster for {selected_region}..."):
    try:
        reg_ndvi, reg_lst, reg_bbox = load_region_raster(selected_region, YEAR, map_month)
        fmap = visualization.build_raster_map_with_borders(
            reg_ndvi, reg_lst, reg_bbox,
            layer=variable.lower(),
            geojson=geojson,
            selected_region=selected_region,
            mask_region=selected_region,
        )
        folium_static(fmap, width=900, height=480)
    except Exception as exc:
        st.error(f"Raster fetch failed: {exc}")

st.caption(
    "Satellite raster with NUTS-3 region borders. "
    "Use the sidebar to select a region for detail view below."
)

# ---------------------------------------------------------------------------
# Row 2: Selected region stats + time series
# ---------------------------------------------------------------------------
st.divider()
st.subheader(f"Detail — {selected_region}")

region_ts = aggregation.filter_by_entity(ts_filtered, selected_region)

detail_left, detail_right = st.columns([1, 2])

with detail_left:
    st.markdown(f"**Statistics — {MONTH_NAMES[map_month-1]} {YEAR}**")
    month_row = region_ts[region_ts["month"] == map_month].iloc[-1:] if not region_ts.empty else pd.DataFrame()
    if not month_row.empty:
        latest_row = month_row
    else:
        latest_row = pd.DataFrame()
    if not latest_row.empty:
        row = latest_row.iloc[0]
        unit = "" if variable == "NDVI" else " °C"
        st.metric("Mean", f"{row[value_col]:.3f}{unit}")
        st.metric("Min",  f"{row[f'{variable.lower()}_min']:.3f}{unit}" if f"{variable.lower()}_min" in row else "N/A")
        st.metric("Max",  f"{row[f'{variable.lower()}_max']:.3f}{unit}" if f"{variable.lower()}_max" in row else "N/A")
        if variable == "NDVI":
            st.info(ndvi_mod.interpret_value(row[value_col]))
        else:
            st.info(lst_mod.interpret_value(row[value_col]))
    else:
        st.info("No data for this region/period.")

with detail_right:
    fig = visualization.plot_timeseries(
        region_ts, variable=variable.lower(),
        title=f"{selected_region} — Monthly {variable}",
    )
    st.plotly_chart(fig, use_container_width=True)
