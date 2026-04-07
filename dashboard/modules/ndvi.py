"""
NDVI domain logic: statistics, classification, and display constants.
All functions accept 2D float32 numpy arrays with NaN for no-data.
"""
import numpy as np

# Plotly colorscale: red → yellow → green
NDVI_COLORSCALE = [
    [0.00, "#8B0000"],
    [0.20, "#FF4500"],
    [0.40, "#FFFF00"],
    [0.65, "#90EE90"],
    [1.00, "#006400"],
]

NDVI_CMAP = "RdYlGn"
NDVI_VMIN = -0.1
NDVI_VMAX = 0.9

NDVI_CLASS_LABELS = {
    0: "Water / No Data",
    1: "Bare Soil",
    2: "Sparse Vegetation",
    3: "Moderate Vegetation",
    4: "Dense Vegetation",
}


def compute_stats(arr: np.ndarray) -> dict:
    """
    Compute descriptive statistics from a 2D NDVI array.
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


def classify_ndvi(arr: np.ndarray) -> np.ndarray:
    """
    Classify into 5 integer classes:
      0 = Water/NoData (NaN or <0), 1 = Bare [0,0.2),
      2 = Sparse [0.2,0.4), 3 = Moderate [0.4,0.6), 4 = Dense >=0.6
    """
    out = np.zeros(arr.shape, dtype=np.int8)
    out[(arr >= 0.0) & (arr < 0.2)] = 1
    out[(arr >= 0.2) & (arr < 0.4)] = 2
    out[(arr >= 0.4) & (arr < 0.6)] = 3
    out[arr >= 0.6] = 4
    out[np.isnan(arr)] = 0
    return out


def interpret_value(ndvi_mean: float) -> str:
    """Human-readable vegetation label for a scalar mean NDVI."""
    if np.isnan(ndvi_mean):
        return "No data"
    if ndvi_mean < 0.0:
        return "Water / Urban Surface"
    if ndvi_mean < 0.2:
        return "Bare Soil / Impervious"
    if ndvi_mean < 0.4:
        return "Sparse Vegetation"
    if ndvi_mean < 0.6:
        return "Moderate Vegetation"
    return "Dense Vegetation"
