"""
Page 3 — Cities
Raster map for selected city, statistics, time series, and Urban Heat Island comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
from sentinelhub import BBox, CRS, bbox_to_dimensions

from dashboard.modules import (
    aggregation,
    geography,
    ingestion,
    ndvi as ndvi_mod,
    lst as lst_mod,
    preprocessing,
    visualization,
)

st.set_page_config(page_title="Cities", page_icon="🏙️", layout="wide")
st.title("Czech Cities — Vegetation & Heat")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
variable = st.sidebar.radio("Variable", ["NDVI", "LST"], key="city_var")
all_city_names = list(geography.CITIES.keys())
detail_city = st.sidebar.selectbox("City", all_city_names, key="city_detail")
map_month = st.sidebar.selectbox(
    "Map month",
    options=list(range(1, 13)),
    format_func=lambda m: [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ][m-1],
    index=6,
    key="city_map_mo",
)

YEAR = 2025
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_city_ts():
    try:
        return aggregation.load_timeseries("cities")
    except FileNotFoundError as e:
        return str(e)

@st.cache_data
def load_country_ts():
    try:
        return aggregation.load_timeseries("country")
    except FileNotFoundError as e:
        return None

cities_geojson = geography.load_cities_geojson()

@st.cache_data
def load_city_raster(city_name: str, year: int, month: int):
    """Fetch raster at 120m for the selected city area."""
    if cities_geojson:
        bbox = geography.get_city_bbox_from_geojson(city_name, cities_geojson)
    else:
        bbox = geography.get_city_bbox(city_name, buffer_deg=0.15)
    sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
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

city_ts_data = load_city_ts()
country_ts_data = load_country_ts()

if isinstance(city_ts_data, str):
    st.error(city_ts_data)
    st.stop()

ts_filtered = aggregation.filter_by_year_range(city_ts_data, YEAR, YEAR)
value_col = "ndvi_mean" if variable == "NDVI" else "lst_mean"

# Stats per city for selected month
city_latest = (
    ts_filtered[
        (ts_filtered["year"] == YEAR) & (ts_filtered["month"] == map_month)
    ][["entity", value_col]]
    .copy()
)

# ---------------------------------------------------------------------------
# Row 1: Raster map for selected city
# ---------------------------------------------------------------------------
st.subheader(f"{variable} — {detail_city} — {MONTH_NAMES[map_month-1]} {YEAR}")
with st.spinner(f"Fetching raster for {detail_city}..."):
    try:
        city_ndvi, city_lst, city_bbox = load_city_raster(detail_city, YEAR, map_month)
        if cities_geojson:
            # Clip to actual city boundary
            fmap = visualization.build_raster_map_with_borders(
                city_ndvi, city_lst, city_bbox,
                layer=variable.lower(),
                geojson=cities_geojson,
                selected_region=detail_city,
                mask_region=detail_city,
            )
        else:
            # Fallback: circular clip
            city_info_map = geography.CITIES[detail_city]
            center = (city_info_map["lon"], city_info_map["lat"])
            fmap = visualization.build_city_raster_map(
                city_ndvi, city_lst, city_bbox, layer=variable.lower(),
                city_center=center,
            )
        folium_static(fmap, width=900, height=480)
    except Exception as exc:
        st.error(f"Raster fetch failed: {exc}")

st.caption(f"Satellite raster for {detail_city} area at 120m resolution.")

# ---------------------------------------------------------------------------
# Row 2: Detail view for selected city
# ---------------------------------------------------------------------------
st.divider()
st.subheader(f"Detail — {detail_city}")

city_info = geography.CITIES.get(detail_city, {})
detail_ts = aggregation.filter_by_entity(ts_filtered, detail_city)

col_info, col_ts, col_compare = st.columns([1, 2, 2])

with col_info:
    st.markdown(f"**Region:** {city_info.get('region','—')}")
    st.markdown(f"**Population:** {city_info.get('population',0):,}")

    month_row = detail_ts[detail_ts["month"] == map_month].iloc[-1:] if not detail_ts.empty else pd.DataFrame()
    latest_row = month_row if not month_row.empty else pd.DataFrame()
    if not latest_row.empty:
        row = latest_row.iloc[0]
        unit = "" if variable == "NDVI" else " °C"
        st.metric("Mean", f"{row[value_col]:.3f}{unit}")
        min_col = f"{variable.lower()}_min"
        max_col = f"{variable.lower()}_max"
        if min_col in row:
            st.metric("Min", f"{row[min_col]:.3f}{unit}")
        if max_col in row:
            st.metric("Max", f"{row[max_col]:.3f}{unit}")
        if variable == "NDVI":
            st.info(ndvi_mod.interpret_value(row[value_col]))

with col_ts:
    fig = visualization.plot_timeseries(
        detail_ts, variable=variable.lower(),
        title=f"{detail_city} — Monthly {variable}",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_compare:
    st.markdown(f"**City comparison — {MONTH_NAMES[map_month-1]} {YEAR}**")
    if not city_latest.empty:
        sorted_df = city_latest.sort_values(value_col, ascending=True).rename(
            columns={value_col: variable}
        )
        import plotly.graph_objects as go
        color = "#27AE60" if variable == "NDVI" else "#E74C3C"
        bar_fig = go.Figure(go.Bar(
            x=sorted_df[variable],
            y=sorted_df["entity"],
            orientation="h",
            marker_color=[
                "#C8F7C8" if e == detail_city else color
                for e in sorted_df["entity"]
            ],
        ))
        bar_fig.update_layout(
            xaxis_title=variable + ("" if variable == "NDVI" else " °C"),
            template="plotly_white",
            margin=dict(l=120, r=20, t=20, b=40),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 3: Vegetation cooling analysis
# ---------------------------------------------------------------------------
st.divider()
st.subheader(f"Vegetation Cooling — {detail_city}")

@st.cache_data
def load_city_lst_l2(city_name: str, year: int, month: int):
    from dashboard.modules.lst_l2 import fetch_lst_l2
    bbox = geography.get_city_bbox(city_name, buffer_deg=0.15)
    return fetch_lst_l2(bbox, (128, 96), year, month)

with st.spinner("Loading real surface temperature (S3 LST Level-2)..."):
    try:
        lst_l2 = load_city_lst_l2(detail_city, YEAR, map_month)
        ndvi_resized = preprocessing.resample_to_shape(city_ndvi, lst_l2.shape)

        cool_left, cool_right = st.columns(2)
        with cool_left:
            scatter_fig = visualization.plot_pixel_scatter(
                ndvi_resized, lst_l2,
                title=f"NDVI vs Surface Temp — {detail_city}",
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        with cool_right:
            cooling = visualization.compute_vegetation_cooling(ndvi_resized, lst_l2)
            c1, c2 = st.columns(2)
            c1.metric("Vegetated (NDVI>0.4)", f"{cooling['vegetated_mean']:.1f} °C" if np.isfinite(cooling['vegetated_mean']) else "N/A")
            c2.metric("Bare/urban (NDVI<0.2)", f"{cooling['bare_mean']:.1f} °C" if np.isfinite(cooling['bare_mean']) else "N/A")
            if np.isfinite(cooling['cooling_effect']):
                st.metric("Cooling effect", f"{cooling['cooling_effect']:.1f} °C")
                st.caption(f"Vegetation is **{cooling['cooling_effect']:.1f}°C cooler** than bare surfaces")
            st.metric("Correlation (r)", f"{cooling['correlation']:.2f}" if np.isfinite(cooling['correlation']) else "N/A")
    except Exception as e:
        st.warning(f"LST Level-2 analysis unavailable: {e}")

# ---------------------------------------------------------------------------
# Row 4: Urban Heat Island
# ---------------------------------------------------------------------------
if variable == "LST":
    st.divider()
    st.subheader("Urban Heat Island Intensity")
    st.markdown("Difference between city LST and Czech Republic mean LST for the same month.")

    if country_ts_data is not None and not isinstance(country_ts_data, str):
        country_filtered = aggregation.filter_by_year_range(country_ts_data, YEAR, YEAR)
        cz_latest = aggregation.filter_by_entity(country_filtered, "Czech Republic")
        cz_latest_row = cz_latest[
            (cz_latest["year"] == YEAR) & (cz_latest["month"] == map_month)
        ]
        country_lst_mean = float(cz_latest_row["lst_mean"].iloc[0]) if not cz_latest_row.empty else np.nan

        if not np.isnan(country_lst_mean):
            uhi_fig = visualization.plot_uhi_bar(
                city_latest.rename(columns={value_col: "lst_mean"}),
                country_lst_mean,
                title=f"UHI — {MONTH_NAMES[map_month-1]} {YEAR}",
            )
            st.plotly_chart(uhi_fig, use_container_width=True)
            st.caption(f"Country reference LST: {country_lst_mean:.1f} °C")
        else:
            st.info("Country reference LST not available for UHI calculation.")
    else:
        st.info("Country time series not available for UHI calculation.")
