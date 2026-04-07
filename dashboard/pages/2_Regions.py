"""
Page 2 — Regions
Choropleth map of 14 Czech NUTS-3 regions, regional statistics and time series.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

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
year_range = st.sidebar.slider("Year range", 2020, 2025, (2020, 2025), key="reg_yr")
map_year = st.sidebar.selectbox("Map year", list(range(2025, 2019, -1)), key="reg_map_yr")
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
    bbox = geography.get_region_bbox(region_name)
    # Scale size proportionally to region bbox area
    lon_span = bbox[2] - bbox[0]
    lat_span = bbox[3] - bbox[1]
    w = max(64, int(lon_span / (18.87 - 12.09) * geography.CZ_RASTER_SIZE[1]))
    h = max(64, int(lat_span / (51.06 - 48.55) * geography.CZ_RASTER_SIZE[0]))
    size = (w, h)  # (W, H) for sentinelhub
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

ts_filtered = aggregation.filter_by_year_range(ts_data, year_range[0], year_range[1])

# Latest stats for all regions → choropleth colour data
value_col = "ndvi_mean" if variable == "NDVI" else "lst_mean"
latest_year = ts_filtered["year"].max()
latest_month = ts_filtered[ts_filtered["year"] == latest_year]["month"].max()
choropleth_df = (
    ts_filtered[
        (ts_filtered["year"] == latest_year) & (ts_filtered["month"] == latest_month)
    ][["entity", value_col]]
    .rename(columns={value_col: value_col})  # keep name
    .copy()
)

geojson = geography.load_regions_geojson()

# ---------------------------------------------------------------------------
# Row 1: Choropleth map
# ---------------------------------------------------------------------------
st.subheader(f"Choropleth — {variable} ({['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][latest_month-1]} {latest_year})")
fmap = visualization.build_region_choropleth_map(
    geojson, choropleth_df, variable.lower(), selected_region=selected_region
)
st_folium(fmap, width=900, height=480, returned_objects=[])

st.caption(
    "Map shows the most recent available month across all regions. "
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
    st.markdown("**Summary statistics (selected period)**")
    latest_row = region_ts.dropna(subset=[value_col]).sort_values("date").iloc[-1:] if not region_ts.empty else pd.DataFrame()
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

    # Raster map for selected region
    st.markdown("**Raster map (selected month/year)**")
    with st.spinner("Fetching region raster..."):
        try:
            r_ndvi, r_lst, r_bbox = load_region_raster(selected_region, map_year, map_month)
            r_fmap = visualization.build_country_raster_map(
                r_ndvi, r_lst, r_bbox, layer=variable.lower()
            )
            st_folium(r_fmap, width=360, height=280, returned_objects=[])
        except Exception as exc:
            st.error(f"Raster fetch failed: {exc}")

with detail_right:
    tab1, tab2, tab3 = st.tabs(["Monthly time series", "Yearly", "All regions heatmap"])
    with tab1:
        fig = visualization.plot_timeseries(
            region_ts, variable=variable.lower(), show_range=True,
            title=f"{selected_region} — Monthly {variable}",
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        yearly = aggregation.aggregate_yearly(region_ts)
        fig2 = visualization.plot_yearly_bar(
            yearly, variable.lower(), selected_region,
        )
        st.plotly_chart(fig2, use_container_width=True)
    with tab3:
        fig3 = visualization.plot_monthly_heatmap(
            ts_filtered, variable=variable.lower(),
            title=f"All regions — Monthly {variable} heatmap",
        )
        st.plotly_chart(fig3, use_container_width=True)
