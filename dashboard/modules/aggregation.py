"""
Time-series aggregation helpers operating on the canonical TimeSeriesDF.

TimeSeriesDF schema:
    year (int), month (int), date (Timestamp), entity (str),
    ndvi_mean, ndvi_min, ndvi_max, ndvi_std (float32),
    lst_mean,  lst_min,  lst_max,  lst_std  (float32)
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Parquet files are written by scripts/fetch_all_timeseries.py
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "timeseries"

_MISSING_MSG = (
    "Parquet file not found: {path}\n"
    "Run the fetch script first:\n"
    "    uv run python scripts/fetch_all_timeseries.py"
)


def load_timeseries(level: str) -> pd.DataFrame:
    """
    Load the pre-fetched time-series parquet for 'country', 'regions', or 'cities'.
    Raises FileNotFoundError with a helpful message if the file does not exist.
    """
    path = _DATA_DIR / f"{level}_ndvi_lst.parquet"
    if not path.exists():
        raise FileNotFoundError(_MISSING_MSG.format(path=path))
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def filter_by_entity(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Return rows where entity column matches the given name."""
    return df[df["entity"] == entity].copy()


def filter_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return rows for a single year."""
    return df[df["year"] == year].copy()


def filter_by_year_range(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Return rows where start <= year <= end."""
    return df[(df["year"] >= start) & (df["year"] <= end)].copy()


def aggregate_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group TimeSeriesDF by (entity, year) and average across the 12 months.
    Returns a DataFrame with one row per (entity, year) and columns:
        entity, year, ndvi_mean, ndvi_std, lst_mean, lst_std
    """
    agg = (
        df.groupby(["entity", "year"], sort=True)
        .agg(
            ndvi_mean=("ndvi_mean", "mean"),
            ndvi_std=("ndvi_std", "mean"),
            lst_mean=("lst_mean", "mean"),
            lst_std=("lst_std", "mean"),
        )
        .reset_index()
    )
    return agg


def get_monthly_climatology(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    For a single entity, compute the mean and std of NDVI/LST per calendar month
    across all years. Returns a 12-row DataFrame indexed by month (1–12).
    Useful as a background 'typical season' band on time-series charts.
    """
    sub = filter_by_entity(df, entity)
    clim = (
        sub.groupby("month")
        .agg(
            ndvi_clim_mean=("ndvi_mean", "mean"),
            ndvi_clim_std=("ndvi_mean", "std"),
            lst_clim_mean=("lst_mean", "mean"),
            lst_clim_std=("lst_mean", "std"),
        )
        .reset_index()
    )
    return clim


def latest_stats(df: pd.DataFrame, entity: str) -> dict:
    """
    Return the most recent non-NaN row for the given entity as a plain dict.
    Used for KPI cards on the landing page.
    """
    sub = filter_by_entity(df, entity).dropna(subset=["ndvi_mean", "lst_mean"])
    if sub.empty:
        return {}
    row = sub.sort_values("date").iloc[-1]
    return row.to_dict()
