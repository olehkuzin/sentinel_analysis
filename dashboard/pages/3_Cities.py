"""
Page 3 — Cities
City-level marker map, statistics, time series, and Urban Heat Island comparison.
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

st.set_page_config(page_title="Cities", page_icon="🏙️", layout="wide")
st.title("Czech Cities — Vegetation & Heat")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
variable = st.sidebar.radio("Variable", ["NDVI", "LST"], key="city_var")
all_city_names = list(geography.CITIES.keys())
selected_cities = st.sidebar.multiselect(
    "Show cities", all_city_names, default=all_city_names, key="city_multi"
)
detail_city = st.sidebar.selectbox(
    "Detail view",
    selected_cities if selected_cities else all_city_names,
    key="city_detail",
)
year_range = st.sidebar.slider("Year range", 2020, 2025, (2020, 2025), key="city_yr")

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

@st.cache_data(persist="disk")
def load_city_raster(city_name: str, year: int, month: int):
    bbox = geography.get_city_bbox(city_name, buffer_deg=0.12)
    size = (256, 256)
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

ts_filtered = aggregation.filter_by_year_range(city_ts_data, year_range[0], year_range[1])
cities_subset = {k: v for k, v in geography.CITIES.items() if k in (selected_cities or all_city_names)}

# Latest stats per city → marker colouring
value_col = "ndvi_mean" if variable == "NDVI" else "lst_mean"
latest_year = ts_filtered["year"].max()
latest_month = ts_filtered[ts_filtered["year"] == latest_year]["month"].max()
city_latest = (
    ts_filtered[
        (ts_filtered["year"] == latest_year) & (ts_filtered["month"] == latest_month)
        & ts_filtered["entity"].isin(list(cities_subset.keys()))
    ][["entity", value_col]]
    .copy()
)

# ---------------------------------------------------------------------------
# Row 1: City marker map
# ---------------------------------------------------------------------------
st.subheader(
    f"City Map — {variable} "
    f"({['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][latest_month-1]} "
    f"{latest_year})"
)
fmap = visualization.build_city_marker_map(
    cities_subset, city_latest, variable.lower(), selected_city=detail_city
)
st_folium(fmap, width=900, height=480, returned_objects=[])

st.caption("Circle size ∝ city population. Color reflects the selected variable value.")

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

    latest_row = detail_ts.dropna(subset=[value_col]).sort_values("date").iloc[-1:] if not detail_ts.empty else pd.DataFrame()
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
        else:
            st.info(lst_mod.interpret_value(row[value_col]))

with col_ts:
    tab1, tab2 = st.tabs(["Monthly", "Yearly"])
    with tab1:
        fig = visualization.plot_timeseries(
            detail_ts, variable=variable.lower(), show_range=True,
            title=f"{detail_city} — Monthly {variable}",
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        yearly = aggregation.aggregate_yearly(detail_ts)
        fig2 = visualization.plot_yearly_bar(yearly, variable.lower(), detail_city)
        st.plotly_chart(fig2, use_container_width=True)

with col_compare:
    st.markdown("**City comparison (latest month)**")
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
                "#000000" if e == detail_city else color
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
# Row 3: Urban Heat Island
# ---------------------------------------------------------------------------
if variable == "LST":
    st.divider()
    st.subheader("Urban Heat Island Intensity")
    st.markdown("Difference between city LST and Czech Republic mean LST for the same month.")

    if country_ts_data is not None and not isinstance(country_ts_data, str):
        country_filtered = aggregation.filter_by_year_range(country_ts_data, year_range[0], year_range[1])
        cz_latest = aggregation.filter_by_entity(country_filtered, "Czech Republic")
        cz_latest_row = cz_latest[
            (cz_latest["year"] == latest_year) & (cz_latest["month"] == latest_month)
        ]
        country_lst_mean = float(cz_latest_row["lst_mean"].iloc[0]) if not cz_latest_row.empty else np.nan

        if not np.isnan(country_lst_mean):
            uhi_fig = visualization.plot_uhi_bar(
                city_latest.rename(columns={value_col: "lst_mean"}),
                country_lst_mean,
                title=f"UHI — {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][latest_month-1]} {latest_year}",
            )
            st.plotly_chart(uhi_fig, use_container_width=True)
            st.caption(f"Country reference LST: {country_lst_mean:.1f} °C")
        else:
            st.info("Country reference LST not available for UHI calculation.")
    else:
        st.info("Country time series not available for UHI calculation.")
