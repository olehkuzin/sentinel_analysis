"""
One-time data fetch script.

Downloads monthly NDVI and LST statistics for the Czech Republic,
its 14 regions, and 10 major cities using the SentinelHub Statistical API.
Results are saved to data/timeseries/*.parquet and loaded by the dashboard
at runtime — the dashboard never calls the API directly.

Usage:
    uv run python scripts/fetch_all_timeseries.py

Environment variables required in .env:
    SH_CLIENT_ID=<your-id>
    SH_CLIENT_SECRET=<your-secret>

Expected runtime: 10–30 minutes depending on API response times.
"""
import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from dashboard.modules import geography
from dashboard.modules import statistics_api

OUTPUT_DIR = Path("data/timeseries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2025-01-01"
END = "2025-12-31"


def fetch_and_save(entities: list[str], entity_type: str, filename: str) -> None:
    frames = []
    total = len(entities)
    for i, name in enumerate(entities, 1):
        print(f"[{i}/{total}] {entity_type}: {name}")
        try:
            df = statistics_api.fetch_entity_timeseries(name, entity_type, START, END)
            frames.append(df)
            print(f"  → {len(df)} rows fetched")
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping {name}")

    if frames:
        result = pd.concat(frames, ignore_index=True)
        out_path = OUTPUT_DIR / filename
        result.to_parquet(out_path, index=False)
        print(f"\nSaved {len(result)} rows to {out_path}\n")
    else:
        print(f"\nNo data fetched for {entity_type}.\n")


if __name__ == "__main__":
    print("=== Downloading NUTS-3 region boundaries ===")
    geography.download_cz_nuts3_boundaries()

    print("=== Fetching country-level time series ===")
    fetch_and_save(["Czech Republic"], "country", "country_ndvi_lst.parquet")

    print("=== Fetching region-level time series ===")
    fetch_and_save(list(geography.REGIONS.keys()), "region", "regions_ndvi_lst.parquet")

    print("=== Fetching city-level time series ===")
    fetch_and_save(list(geography.CITIES.keys()), "city", "cities_ndvi_lst.parquet")

    print("All done. Run the dashboard with:")
    print("    uv run streamlit run dashboard/app.py")
