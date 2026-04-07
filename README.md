# Czech Republic Vegetation & Urban Heat Dashboard

Satellite-based analysis of vegetation health (NDVI) and land surface temperature across the Czech Republic using Copernicus Sentinel data. Built with Streamlit.

## Features

- **Country Overview** — NDVI and temperature anomaly raster maps clipped to Czech borders with region boundaries overlay
- **14 Regions** — NUTS-3 regional drill-down with high-resolution (120m) rasters clipped to actual administrative boundaries
- **10 Cities** — City-level raster maps with population data and urban heat island analysis
- **Vegetation Cooling Analysis** — Pixel-level NDVI vs LST scatter plots, correlation metrics, and vegetation cooling index showing how much cooler green areas are
- **UHI Intensity** — Urban Heat Island comparison: city surface temperature vs national baseline

## Data Sources

| Variable | Satellite | Resolution | Product |
|----------|-----------|------------|---------|
| NDVI | Sentinel-2 L2A | 10 m | B04, B08 bands via Process API |
| Temperature (maps) | Sentinel-3 SLSTR | ~1 km | S8/S9 split-window brightness temp via Process API |
| Temperature (analysis) | Sentinel-3 SLSTR L2 | ~1 km | LST Level-2 product via CDSE Catalog download |
| Region boundaries | EUROSTAT GISCO | — | NUTS-3 2021 GeoJSON |
| City boundaries | OpenStreetMap | — | Nominatim API |

All satellite data accessed through [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure credentials

Create `.env` in project root:

```
SH_CLIENT_ID=<your-sentinel-hub-client-id>
SH_CLIENT_SECRET=<your-sentinel-hub-client-secret>
CDSE_USERNAME=<your-copernicus-dataspace-email>
CDSE_PASSWORD=<your-copernicus-dataspace-password>
```

- SH credentials: [Sentinel Hub Dashboard](https://shapps.dataspace.copernicus.eu/dashboard/)
- CDSE account: [Register here](https://identity.dataspace.copernicus.eu/auth/realms/CDSE/login-actions/registration)

### 3. Download geographic boundaries

```bash
uv run python -c "from dashboard.modules.geography import download_cz_nuts3_boundaries, download_cz_city_boundaries; download_cz_nuts3_boundaries(); download_cz_city_boundaries()"
```

### 4. Fetch time series data

```bash
uv run python scripts/fetch_all_timeseries.py
```

### 5. Pre-fetch raster data (optional, recommended)

```bash
uv run python scripts/fetch_all_rasters.py        # NDVI + brightness temp maps
uv run python scripts/fetch_all_lst_l2.py          # Real LST for analysis (parallel, ~30 min)
```

Without pre-fetching, rasters are downloaded on first dashboard load and cached to disk.

### 6. Run the dashboard

```bash
uv run streamlit run dashboard/app.py
```

## Project Structure

```
dashboard/
  app.py                    # Landing page
  pages/
    1_Overview.py           # Country-wide raster map + stats
    2_Regions.py            # Regional drill-down
    3_Cities.py             # City-level analysis + UHI
  modules/
    geography.py            # Regions/cities coordinates, GeoJSON loading
    ingestion.py            # Sentinel Hub Process API raster fetching + disk cache
    statistics_api.py       # Sentinel Hub Statistical API for time series
    aggregation.py          # Parquet loading, filtering, aggregation
    visualization.py        # Folium maps, Plotly charts, pixel scatter, cooling index
    preprocessing.py        # Masking, resampling, alignment
    ndvi.py                 # NDVI constants, classification, interpretation
    lst.py                  # LST constants and interpretation
    lst_l2.py               # Sentinel-3 LST Level-2 download from CDSE catalog
  data/
    rasters/                # Cached .npy raster files
    timeseries/             # Pre-computed .parquet time series
    cz_regions.geojson      # NUTS-3 boundaries
    cz_cities.geojson       # City boundaries from OSM

scripts/
  fetch_all_timeseries.py   # Pre-fetch monthly statistics for all areas
  fetch_all_rasters.py      # Pre-fetch NDVI + brightness temp rasters
  fetch_all_lst_l2.py       # Pre-fetch real LST L2 data (parallel)

notebooks/
  data_extraction.ipynb     # Original exploration notebook
```

## Key Findings

- **Vegetation cools urban areas by 2-5 °C** in summer months across Czech cities
- **Negative NDVI-temperature correlation** (r = -0.2 to -0.4) consistently observed
- **Urban Heat Island effect** clearly visible in city vs. country temperature comparison

## Tech Stack

- **Sentinel Hub** — Copernicus Data Space API for raster and statistical data
- **Streamlit** — Multi-page dashboard framework
- **Folium** — Interactive Leaflet maps with raster overlays
- **Plotly** — Time series, scatter, and comparison charts
- **Shapely + Rasterio** — Geospatial masking and boundary clipping
