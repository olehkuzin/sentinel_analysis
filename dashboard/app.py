"""
Landing page for the Czech Republic Vegetation & Urban Heat Dashboard.
Streamlit's pages/ folder handles multi-page navigation automatically.
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
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Try to load pre-fetched data for KPI cards
# ---------------------------------------------------------------------------
from dashboard.modules import aggregation

@st.cache_data
def _load_country():
    try:
        return aggregation.load_timeseries("country")
    except FileNotFoundError:
        return None

country_df = _load_country()

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("Czech Republic Vegetation & Urban Heat")
st.markdown(
    "Satellite-based analysis using **Sentinel-2 L2A** (NDVI) "
    "and **Sentinel-3 SLSTR** (Land Surface Temperature) "
    "for the period **2020–2025**."
)

if country_df is None:
    st.warning(
        "**Data not yet downloaded.** Run the fetch script first:\n\n"
        "```\nuv run python scripts/fetch_all_timeseries.py\n```\n\n"
        "This downloads ~72 months of statistics for the whole country, "
        "14 regions, and 10 cities (~10–30 min). "
        "After that, launch the dashboard again."
    )
else:
    latest = aggregation.latest_stats(country_df, "Czech Republic")
    ndvi_latest = latest.get("ndvi_mean", np.nan)
    lst_latest = latest.get("lst_mean", np.nan)
    latest_date = latest.get("date", "—")

    st.markdown(f"**Latest data point:** {str(latest_date)[:10] if latest_date != '—' else '—'}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="CZ Mean NDVI",
            value=f"{ndvi_latest:.3f}" if not np.isnan(ndvi_latest) else "N/A",
            help="Normalized Difference Vegetation Index — higher = more vegetation",
        )
    with col2:
        st.metric(
            label="CZ Mean LST",
            value=f"{lst_latest:.1f} °C" if not np.isnan(lst_latest) else "N/A",
            help="Land Surface Temperature (Sentinel-3 SLSTR)",
        )
    with col3:
        st.metric(
            label="Data Coverage",
            value=f"2020–2025",
            help="Monthly aggregated statistics via SentinelHub Statistical API",
        )

st.divider()

st.markdown("""
### About this dashboard

| Page | Content |
|------|---------|
| **Czech Republic Overview** | National NDVI/LST map, summary statistics, time series 2020–2025 |
| **Regions** | Choropleth across 14 NUTS-3 regions, regional drill-down |
| **Cities** | City-level markers, Urban Heat Island comparison |

**Data sources**
- NDVI: [Sentinel-2 L2A](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) — 10 m spatial resolution, cloud-masked
- LST: [Sentinel-3 SLSTR](https://sentinel.esa.int/web/sentinel/missions/sentinel-3) — S7 band, Kelvin → Celsius
- Accessed via [Copernicus Data Space](https://dataspace.copernicus.eu/) (Sentinel Hub API)

**Architecture**
The data pipeline separates concerns cleanly:
- `scripts/fetch_all_timeseries.py` — one-time bulk download using the Statistical API
- `dashboard/modules/` — modular processing: NDVI, LST, aggregation, visualization
- `dashboard/pages/` — Streamlit multi-page app with `@st.cache_data` for rasters
""")
