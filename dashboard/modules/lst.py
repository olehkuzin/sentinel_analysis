"""
LST (Land Surface Temperature) domain logic.
All functions accept 2D float32 numpy arrays in degrees Celsius with NaN for no-data.
"""
import numpy as np

# Plotly colorscale: blue → yellow → red
LST_COLORSCALE = [
    [0.00, "#313695"],
    [0.25, "#74ADD1"],
    [0.50, "#FEE090"],
    [0.75, "#F46D43"],
    [1.00, "#A50026"],
]

LST_CMAP = "RdYlBu_r"
# SLSTR brightness temperature (not true surface temp).
# Relative patterns (urban heat islands, vegetation cooling) are valid.
LST_VMIN = 0.0
LST_VMAX = 16.0
LST_LABEL = "Brightness Temp (°C)"


def compute_stats(arr: np.ndarray) -> dict:
    """
    Compute descriptive statistics from a 2D LST array (Celsius).
    Returns dict with: mean, min, max, std, p25, p75, valid_pixels, coverage_pct.
    """
    flat = arr.ravel()
    valid = flat[~np.isnan(flat)]
    n_valid = len(valid)
    n_total = flat.size

    if n_valid == 0:
        return dict(
            mean=np.nan, min=np.nan, max=np.nan, std=np.nan,
            p25=np.nan, p75=np.nan, valid_pixels=0, coverage_pct=0.0,
        )
    return dict(
        mean=float(np.mean(valid)),
        min=float(np.min(valid)),
        max=float(np.max(valid)),
        std=float(np.std(valid)),
        p25=float(np.percentile(valid, 25)),
        p75=float(np.percentile(valid, 75)),
        valid_pixels=n_valid,
        coverage_pct=round(n_valid / n_total * 100, 1),
    )


def compute_uhi_intensity(city_arr: np.ndarray, rural_arr: np.ndarray) -> float:
    """
    Urban Heat Island intensity = mean(city LST) − mean(rural LST).
    Returns scalar in °C. NaN if either array has no valid data.
    """
    city_mean = np.nanmean(city_arr)
    rural_mean = np.nanmean(rural_arr)
    if np.isnan(city_mean) or np.isnan(rural_mean):
        return np.nan
    return float(city_mean - rural_mean)


def interpret_value(lst_mean: float) -> str:
    """Human-readable temperature category for a scalar mean LST (°C)."""
    if np.isnan(lst_mean):
        return "No data"
    if lst_mean < 5:
        return "Very Cold"
    if lst_mean < 12:
        return "Cold"
    if lst_mean < 20:
        return "Moderate"
    if lst_mean < 28:
        return "Warm"
    return "Very Hot"
