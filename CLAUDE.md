# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Satellite remote sensing analysis of Prague's urban heat island effect using Sentinel Hub (Copernicus Data Space). Fetches NDVI (Sentinel-2) and Land Surface Temperature (Sentinel-3 SLSTR) data for July 2024, visualizes them, and saves arrays for further analysis.

## Setup

```bash
uv sync                  # install dependencies into .venv
# Create .env in project root with:
# SH_CLIENT_ID=<your-id>
# SH_CLIENT_SECRET=<your-secret>
```

## Running

Open `notebooks/data_extraction.ipynb` in VS Code (Jupyter extension) or run:
```bash
uv run jupyter notebook notebooks/data_extraction.ipynb
```

## Architecture

- **`src/config.py`** — shared constants: bounding box (Prague), resolution (60m), time interval. Not yet imported by the notebook (duplicated inline).
- **`notebooks/data_extraction.ipynb`** — main workflow: auth → define area → fetch NDVI → fetch LST → visualize → save `.npy` to `data/`.
- **Data flow**: Sentinel Hub API → numpy arrays (441×588 px at 60m) → `data/*.npy` files.

## Key Dependencies

- `sentinelhub` — API client for Copernicus Data Space (evalscripts, data collections)
- `rasterio` — raster I/O (available but not yet used in current notebooks)
- `folium` — interactive maps (available but not yet used)

## Sentinel Hub Specifics

- Base URL: `https://sh.dataspace.copernicus.eu`
- Token URL: `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`
- Data collections used: `SENTINEL2_L2A` (NDVI), `SENTINEL3_SLSTR` (LST)
- Evalscripts are JavaScript (VERSION=3) embedded as Python strings
