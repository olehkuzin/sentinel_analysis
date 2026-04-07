"""
Raw raster cleaning before statistics or display.
All functions operate on 2D numpy arrays and return new arrays (no in-place mutation).
"""
import numpy as np


def mask_invalid(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Set pixels outside [low, high] to NaN.
    Typical ranges: NDVI [-1, 1], LST [-20, 60].
    """
    out = arr.astype(np.float32, copy=True)
    out[(out < low) | (out > high)] = np.nan
    return out


def normalize_for_display(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Map arr to [0, 1] for colormap rendering.
    Values are first clipped to [vmin, vmax], then linearly scaled.
    """
    clipped = np.clip(arr, vmin, vmax)
    span = vmax - vmin
    if span == 0:
        return np.zeros_like(clipped)
    return (clipped - vmin) / span


def resample_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Resize a 2D array to target_shape (H, W) using nearest-neighbour index mapping.
    Preserves NaN values. Suitable for display downscaling.
    """
    src_h, src_w = arr.shape
    tgt_h, tgt_w = target_shape
    row_idx = (np.arange(tgt_h) * src_h / tgt_h).astype(int)
    col_idx = (np.arange(tgt_w) * src_w / tgt_w).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


def align_resolutions(ndvi: np.ndarray, lst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample LST to match the spatial shape of NDVI.
    Sentinel-2 is fetched at higher resolution than Sentinel-3 SLSTR;
    this makes both arrays spatially comparable for overlay.
    """
    if ndvi.shape == lst.shape:
        return ndvi, lst
    lst_resampled = resample_to_shape(lst, ndvi.shape)
    return ndvi, lst_resampled
