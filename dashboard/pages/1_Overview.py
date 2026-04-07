"""
Page 1 — Czech Republic Overview
Interactive raster map, NDVI/LST toggle, summary statistics,
time series (monthly & yearly), NDVI vs LST scatter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
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

st.set_page_config(page_title="CZ Overview", page_icon="🗺️", layout="wide")
st.title("Czech Republic Overview")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
variable = st.sidebar.radio("Variable", ["NDVI", "LST"], key="overview_var")
map_month = st.sidebar.selectbox(
    "Map month",
    options=list(range(1, 13)),
    format_func=lambda m: [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ][m-1],
    index=6,
    key="overview_map_mo",
)

YEAR = 2025
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_ts():
    try:
        return aggregation.load_timeseries("country")
    except FileNotFoundError as e:
        return str(e)

@st.cache_data
def load_raster(year: int, month: int):
    bbox = geography.CZ_BBOX
    size = geography.CZ_RASTER_SIZE[::-1]  # sentinelhub wants (W, H)
    ndvi_arr = ingestion.fetch_ndvi_raster(bbox, size, year, month)
    lst_arr = ingestion.fetch_lst_raster(bbox, size, year, month)
    ndvi_arr = preprocessing.mask_invalid(ndvi_arr, -1.0, 1.0)
    lst_arr = preprocessing.mask_invalid(lst_arr, -20.0, 60.0)
    ndvi_arr, lst_arr = preprocessing.align_resolutions(ndvi_arr, lst_arr)
    return ndvi_arr, lst_arr

@st.cache_data
def load_lst_l2(year: int, month: int):
    """Real LST from S3 Level-2 product for analysis (not for map display)."""
    from dashboard.modules.lst_l2 import fetch_lst_l2
    bbox = geography.CZ_BBOX
    size = (256, 128)  # lower res for speed — analysis doesn't need full resolution
    return fetch_lst_l2(bbox, size, year, month), bbox

ts_data = load_ts()

if isinstance(ts_data, str):
    st.error(ts_data)
    st.stop()

ts_filtered = aggregation.filter_by_year_range(ts_data, YEAR, YEAR)
cz_ts = aggregation.filter_by_entity(ts_filtered, "Czech Republic")

geojson = geography.load_regions_geojson()

# ---------------------------------------------------------------------------
# Row 1: Map + Statistics
# ---------------------------------------------------------------------------
map_col, stats_col = st.columns([3, 2])

with map_col:
    st.subheader(f"Map — {MONTH_NAMES[map_month-1]} {YEAR}")
    with st.spinner("Fetching raster from Sentinel Hub..."):
        try:
            ndvi_arr, lst_arr = load_raster(YEAR, map_month)
            fmap = visualization.build_raster_map_with_borders(
                ndvi_arr, lst_arr, geography.CZ_BBOX,
                layer=variable.lower(),
                geojson=geojson,
            )
            folium_static(fmap, width=700, height=460)
        except Exception as exc:
            st.error(f"Raster fetch failed: {exc}")

with stats_col:
    st.subheader("Statistics (selected map period)")
    try:
        arr = ndvi_arr if variable == "NDVI" else lst_arr
        stats = ndvi_mod.compute_stats(arr) if variable == "NDVI" else lst_mod.compute_stats(arr)
        unit = "" if variable == "NDVI" else " °C"

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean", f"{stats['mean']:.3f}{unit}" if not np.isnan(stats['mean']) else "N/A")
        c2.metric("Min",  f"{stats['min']:.3f}{unit}"  if not np.isnan(stats['min'])  else "N/A")
        c3.metric("Max",  f"{stats['max']:.3f}{unit}"  if not np.isnan(stats['max'])  else "N/A")

        st.plotly_chart(
            visualization.plot_stats_bar(stats, "Czech Republic", variable.lower()),
            use_container_width=True,
        )
        st.caption(
            f"Valid pixels: {stats['valid_pixels']:,} / "
            f"Coverage: {stats['coverage_pct']:.1f}%"
        )
        if variable == "NDVI":
            st.info(f"Vegetation class: {ndvi_mod.interpret_value(stats['mean'])}")
        else:
            st.info(f"Temperature class: {lst_mod.interpret_value(stats['mean'])}")
    except NameError:
        st.info("Load a raster using the sidebar controls.")

# ---------------------------------------------------------------------------
# Row 2: Vegetation cooling analysis
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Vegetation Cooling Analysis")

with st.spinner("Loading real surface temperature (S3 LST Level-2)..."):
    try:
        lst_l2, _ = load_lst_l2(YEAR, map_month)
        # Resample NDVI to match LST L2 resolution
        ndvi_resized = preprocessing.resample_to_shape(ndvi_arr, lst_l2.shape)

        cool_left, cool_right = st.columns(2)
        with cool_left:
            scatter_fig = visualization.plot_pixel_scatter(
                ndvi_resized, lst_l2,
                title=f"NDVI vs Surface Temp — {MONTH_NAMES[map_month-1]} {YEAR}",
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        with cool_right:
            cooling = visualization.compute_vegetation_cooling(ndvi_resized, lst_l2)
            st.markdown("**Vegetation Cooling Index**")
            c1, c2 = st.columns(2)
            c1.metric("Vegetated areas (NDVI>0.4)", f"{cooling['vegetated_mean']:.1f} °C" if np.isfinite(cooling['vegetated_mean']) else "N/A")
            c2.metric("Bare/urban areas (NDVI<0.2)", f"{cooling['bare_mean']:.1f} °C" if np.isfinite(cooling['bare_mean']) else "N/A")

            if np.isfinite(cooling['cooling_effect']):
                st.metric("Cooling effect of vegetation", f"{cooling['cooling_effect']:.1f} °C")
                st.caption(f"Vegetation is **{cooling['cooling_effect']:.1f}°C cooler** than bare/urban surfaces")
            st.metric("NDVI–Temperature correlation", f"r = {cooling['correlation']:.2f}" if np.isfinite(cooling['correlation']) else "N/A")
            st.caption(f"Based on {cooling['n_vegetated']:,} vegetated and {cooling['n_bare']:,} bare pixels | Data: Sentinel-3 LST Level-2")
    except Exception as e:
        st.warning(f"LST Level-2 analysis unavailable: {e}")

# ---------------------------------------------------------------------------
# Row 3: Time series
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Time Series — 2025")

fig = visualization.plot_timeseries(
    cz_ts,
    variable=variable.lower(),
    title=f"Czech Republic — Monthly {variable}",
)
st.plotly_chart(fig, use_container_width=True)
