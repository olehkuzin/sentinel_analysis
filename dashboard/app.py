"""
Landing page — Czech Republic Vegetation & Urban Heat Dashboard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="CZ Vegetation & Heat Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from dashboard.modules import aggregation

@st.cache_data
def _load_country():
    try:
        return aggregation.load_timeseries("country")
    except FileNotFoundError:
        return None

country_df = _load_country()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; padding: 2.5rem 1rem 1.5rem;">
    <h1 style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.3rem;">
        🌿 Czech Republic
    </h1>
    <h2 style="font-size: 1.6rem; font-weight: 400; color: #555; margin-top: 0;">
        Vegetation & Urban Heat Dashboard
    </h2>
    <p style="font-size: 1.05rem; color: #777; max-width: 640px; margin: 1rem auto 0;">
        Monthly satellite analysis of <strong>vegetation health</strong> and
        <strong>land surface temperature</strong> across the Czech Republic —
        country-wide, 14 NUTS-3 regions, and 10 major cities.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Live KPI cards
# ---------------------------------------------------------------------------
if country_df is not None:
    latest = aggregation.latest_stats(country_df, "Czech Republic")
    ndvi_val = latest.get("ndvi_mean", np.nan)
    lst_val  = latest.get("lst_mean",  np.nan)
    date_val = str(latest.get("date", "—"))[:7]

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest data", date_val)
    k2.metric("CZ Mean NDVI", f"{ndvi_val:.3f}" if not np.isnan(ndvi_val) else "N/A",
              help="Normalized Difference Vegetation Index — higher = more vegetation")
    k3.metric("CZ Mean LST",  f"{lst_val:.1f} °C" if not np.isnan(lst_val) else "N/A",
              help="Land Surface Temperature from Sentinel-3 SLSTR")
    k4.metric("Coverage period", "2025", help="Monthly aggregates, Jan – latest available")

st.divider()

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
st.markdown("## Explore the dashboard")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
### 🗺️ Overview
National NDVI & LST raster map of the whole Czech Republic with interactive
Folium overlay. Monthly time series and NDVI vs LST scatter.

**→ Use for:** country-wide trend, seasonal patterns
""")

with c2:
    st.markdown("""
### 🗾 Regions
Choropleth across all **14 NUTS-3 kraje**. Drill into any region for a
high-resolution raster map, statistics, and monthly heatmap across all regions.

**→ Use for:** regional comparison, spotting hot-spots
""")

with c3:
    st.markdown("""
### 🏙️ Cities
City-level raster maps for **10 major cities** — Praha, Brno, Ostrava and more.
Urban Heat Island intensity chart comparing city LST to the national baseline.

**→ Use for:** urban heat analysis, city vs. country
""")

with c4:
    st.markdown("""
### 🤖 AI Assistant
Ask questions in plain language. Claude automatically fetches the right
satellite data and answers — NDVI values, heat recommendations, region
comparisons — without any manual setup.

**→ Use for:** quick insights, open-ended analysis
""")

st.divider()

# ---------------------------------------------------------------------------
# Data & methodology
# ---------------------------------------------------------------------------
st.markdown("## Data & methodology")

m1, m2 = st.columns(2)

with m1:
    st.markdown("""
**Satellite sources**

| Variable | Satellite | Resolution | Processing |
|----------|-----------|------------|------------|
| NDVI | Sentinel-2 L2A | 10 m | B08/B04, cloud-masked (SCL) |
| LST | Sentinel-3 SLSTR | ~1 km | S8/S9 split-window, K → °C |

Both datasets are accessed via the
[Copernicus Data Space](https://dataspace.copernicus.eu/) Sentinel Hub API.
Monthly statistics (mean, min, max, std) are pre-computed server-side using
the **Statistical API** and stored as Parquet files.
Rasters for map display are fetched on demand and cached to disk.
""")

with m2:
    st.markdown("""
**NDVI interpretation**

| Range | Meaning |
|-------|---------|
| < 0.0 | Water / urban surface |
| 0.0 – 0.2 | Bare soil / impervious |
| 0.2 – 0.4 | Sparse vegetation |
| 0.4 – 0.6 | Moderate vegetation |
| > 0.6 | Dense vegetation |

**LST interpretation**

| Range | Meaning |
|-------|---------|
| < 5 °C | Very cold |
| 5 – 12 °C | Cold |
| 12 – 20 °C | Moderate |
| 20 – 28 °C | Warm |
| > 28 °C | Very hot / heat stress |
""")

st.divider()

# ---------------------------------------------------------------------------
# Tech stack
# ---------------------------------------------------------------------------
st.markdown("## Built with")

t1, t2, t3, t4, t5 = st.columns(5)
t1.markdown("**Sentinel Hub**\nCopernicus Data Space API — raster & statistical")
t2.markdown("**Streamlit**\nMulti-page dashboard framework")
t3.markdown("**Folium**\nInteractive Leaflet maps with raster overlays")
t4.markdown("**Plotly**\nTime series, bar, scatter, heatmap charts")
t5.markdown("**Claude AI**\nOpenRouter · tool-use agent for satellite queries")