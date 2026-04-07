"""
All map and chart builders for the dashboard.

Map functions return folium.Map objects → render with streamlit_folium.st_folium().
Chart functions return plotly go.Figure objects → render with st.plotly_chart().
"""
import base64
import io
import math

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from branca.colormap import LinearColormap

from . import ndvi as ndvi_mod
from . import lst as lst_mod

matplotlib.use("Agg")  # non-interactive backend


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_raster_to_geojson(arr: np.ndarray, bbox: list[float], geojson: dict) -> np.ndarray:
    """Set pixels outside the GeoJSON boundary to NaN so raster follows CZ shape.
    Uses rasterio.features.geometry_mask for speed (~50ms vs ~3.5s)."""
    from rasterio.transform import from_bounds
    from rasterio.features import geometry_mask

    h, w = arr.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

    geometries = [feat["geometry"] for feat in geojson.get("features", [])]
    # geometry_mask returns True for pixels OUTSIDE the shapes
    mask = geometry_mask(geometries, out_shape=(h, w), transform=transform, invert=False)

    result = arr.copy()
    result[mask] = np.nan
    return result


def _mask_raster_circle(arr: np.ndarray, bbox: list[float], center: tuple[float, float], radius_deg: float) -> np.ndarray:
    """Mask raster to a circle around center (lon, lat) with given radius in degrees."""
    h, w = arr.shape
    lon_min, lat_min, lon_max, lat_max = bbox
    lons = np.linspace(lon_min, lon_max, w)
    lats = np.linspace(lat_max, lat_min, h)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # Elliptical correction: lon degrees are shorter at higher latitudes
    lat_center = center[1]
    lon_scale = np.cos(np.radians(lat_center))
    dist = np.sqrt(((lon_grid - center[0]) * lon_scale) ** 2 + (lat_grid - center[1]) ** 2)
    result = arr.copy()
    result[dist > radius_deg] = np.nan
    return result


def _array_to_png_b64(arr: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> str:
    """Convert a 2D float array to a base64-encoded PNG for folium ImageOverlay.
    NaN pixels become fully transparent (alpha=0)."""
    from PIL import Image

    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # Apply colormap → RGBA float array (H, W, 4)
    rgba = cmap(norm(np.clip(np.nan_to_num(arr, nan=vmin), vmin, vmax)))
    # Force NaN pixels to fully transparent
    nan_mask = np.isnan(arr)
    rgba[nan_mask, 3] = 0.0
    # Convert to uint8 and save via PIL for guaranteed alpha support
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _folium_base_map(center_lat: float, center_lon: float, zoom: int = 7) -> folium.Map:
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )
    return fmap


# ---------------------------------------------------------------------------
# Folium maps
# ---------------------------------------------------------------------------

def build_country_raster_map(
    ndvi_arr: np.ndarray | None,
    lst_arr: np.ndarray | None,
    bbox: list[float],
    layer: str,
) -> folium.Map:
    """
    Build a Folium map with an ImageOverlay of the NDVI or LST raster for the Czech Republic.
    A branca colorbar is added as a legend.

    Args:
        ndvi_arr: 2D float32 array or None
        lst_arr:  2D float32 array or None
        bbox:     [min_lon, min_lat, max_lon, max_lat]
        layer:    "ndvi" | "lst"
    """
    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2
    # Auto-fit zoom based on bbox span
    lon_span = bbox[2] - bbox[0]
    if lon_span > 5:
        zoom = 7
    elif lon_span > 2:
        zoom = 8
    elif lon_span > 1:
        zoom = 9
    else:
        zoom = 10
    fmap = _folium_base_map(center_lat, center_lon, zoom=zoom)

    bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]

    if layer == "ndvi" and ndvi_arr is not None:
        png_b64 = _array_to_png_b64(ndvi_arr, ndvi_mod.NDVI_CMAP, ndvi_mod.NDVI_VMIN, ndvi_mod.NDVI_VMAX)
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{png_b64}",
            bounds=bounds,
            opacity=0.75,
            interactive=False,
            name="NDVI",
        ).add_to(fmap)
        colorbar = LinearColormap(
            colors=["#8B0000", "#FF4500", "#FFFF00", "#90EE90", "#006400"],
            vmin=ndvi_mod.NDVI_VMIN,
            vmax=ndvi_mod.NDVI_VMAX,
            caption="NDVI",
        )
        colorbar.add_to(fmap)

    elif layer == "lst" and lst_arr is not None:
        # Convert to anomaly (deviation from area mean)
        area_mean = float(np.nanmean(lst_arr))
        anomaly = lst_arr - area_mean
        vmin, vmax = -4.0, 4.0  # ±4°C range
        png_b64 = _array_to_png_b64(anomaly, "RdBu_r", vmin, vmax)
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{png_b64}",
            bounds=bounds,
            opacity=0.75,
            interactive=False,
            name="Temperature anomaly",
        ).add_to(fmap)
        colorbar = LinearColormap(
            colors=["#313695", "#74ADD1", "#FFFFBF", "#F46D43", "#A50026"],
            vmin=vmin,
            vmax=vmax,
            caption="Temperature anomaly (°C vs area mean)",
        )
        colorbar.add_to(fmap)

    return fmap


def build_city_raster_map(
    ndvi_arr: np.ndarray | None,
    lst_arr: np.ndarray | None,
    bbox: list[float],
    layer: str,
    city_center: tuple[float, float],
    radius_deg: float = 0.13,
) -> folium.Map:
    """Raster clipped to a circle around the city center."""
    if ndvi_arr is not None:
        ndvi_arr = _mask_raster_circle(ndvi_arr, bbox, city_center, radius_deg)
    if lst_arr is not None:
        lst_arr = _mask_raster_circle(lst_arr, bbox, city_center, radius_deg)
    return build_country_raster_map(ndvi_arr, lst_arr, bbox, layer)


def build_raster_map_with_borders(
    ndvi_arr: np.ndarray | None,
    lst_arr: np.ndarray | None,
    bbox: list[float],
    layer: str,
    geojson: dict | None = None,
    selected_region: str | None = None,
    mask_region: str | None = None,
) -> folium.Map:
    """Raster overlay clipped to CZ/region borders with NUTS-3 lines on top.

    Args:
        mask_region: If set, clip raster to only this region's polygon.
                     If None, clip to the full country outline.
    """
    if geojson and geojson.get("features"):
        # Build the mask geojson: single region or whole country
        if mask_region:
            mask_gj = {
                "type": "FeatureCollection",
                "features": [
                    f for f in geojson["features"]
                    if f["properties"].get("name") == mask_region
                ],
            }
        else:
            mask_gj = geojson

        if mask_gj.get("features"):
            if ndvi_arr is not None:
                ndvi_arr = _mask_raster_to_geojson(ndvi_arr, bbox, mask_gj)
            if lst_arr is not None:
                lst_arr = _mask_raster_to_geojson(lst_arr, bbox, mask_gj)

    fmap = build_country_raster_map(ndvi_arr, lst_arr, bbox, layer)

    if geojson and geojson.get("features"):
        def _style(feature):
            is_sel = feature["properties"].get("name") == selected_region
            return {
                "fillOpacity": 0,
                "weight": 3 if is_sel else 1.5,
                "color": "#000000" if is_sel else "#222222",
            }
        # Borders only — no click/hover popups
        folium.GeoJson(
            geojson,
            style_function=_style,
            name="Region borders",
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap


def build_region_choropleth_map(
    geojson: dict,
    stats_df: pd.DataFrame,
    layer: str,
    selected_region: str | None = None,
) -> folium.Map:
    """
    Choropleth map coloring each region by mean NDVI or LST.

    Args:
        geojson:         GeoJSON FeatureCollection with feature.properties.name
        stats_df:        DataFrame with columns [entity, ndvi_mean | lst_mean]
        layer:           "ndvi" | "lst"
        selected_region: if set, highlight with a thicker border
    """
    fmap = _folium_base_map(49.75, 15.5, zoom=7)
    value_col = "ndvi_mean" if layer == "ndvi" else "lst_mean"

    if value_col not in stats_df.columns or stats_df.empty:
        return fmap

    choropleth = folium.Choropleth(
        geo_data=geojson,
        data=stats_df,
        columns=["entity", value_col],
        key_on="feature.properties.name",
        fill_color="RdYlGn" if layer == "ndvi" else "RdYlBu_r",
        fill_opacity=0.7,
        line_opacity=0.4,
        legend_name="NDVI" if layer == "ndvi" else "LST (°C)",
        nan_fill_color="lightgray",
        highlight=True,
    )
    choropleth.add_to(fmap)

    # Tooltip on hover
    style_func = lambda x: {
        "fillOpacity": 0,
        "weight": 3 if x["properties"]["name"] == selected_region else 0.5,
        "color": "#000000" if x["properties"]["name"] == selected_region else "#555555",
    }
    tooltip_df = stats_df.set_index("entity")
    folium.GeoJson(
        geojson,
        style_function=style_func,
        tooltip=folium.GeoJsonTooltip(
            fields=["name"],
            aliases=["Region:"],
            localize=True,
        ),
    ).add_to(fmap)

    return fmap


def build_city_marker_map(
    cities: dict,
    stats_df: pd.DataFrame,
    layer: str,
    selected_city: str | None = None,
) -> folium.Map:
    """
    CircleMarker map for Czech cities, colored by NDVI or LST value.

    Args:
        cities:    geography.CITIES dict
        stats_df:  DataFrame with columns [entity, ndvi_mean | lst_mean]
        layer:     "ndvi" | "lst"
        selected_city: highlighted with black border
    """
    fmap = _folium_base_map(49.75, 15.5, zoom=7)
    value_col = "ndvi_mean" if layer == "ndvi" else "lst_mean"

    if stats_df.empty or value_col not in stats_df.columns:
        for name, info in cities.items():
            folium.CircleMarker(
                location=[info["lat"], info["lon"]],
                radius=8,
                color="#888888",
                fill=True,
                fill_color="#888888",
                fill_opacity=0.6,
                popup=folium.Popup(name, max_width=200),
                tooltip=name,
            ).add_to(fmap)
        return fmap

    # Normalise values to [0,1] for colormap
    vals = stats_df.set_index("entity")[value_col]
    v_min = vals.min()
    v_max = vals.max()
    span = v_max - v_min if v_max != v_min else 1.0

    cmap = plt.get_cmap("RdYlGn" if layer == "ndvi" else "RdYlBu_r")

    for name, info in cities.items():
        val = vals.get(name, np.nan)
        if np.isnan(val):
            color = "#888888"
        else:
            rgba = cmap((val - v_min) / span)
            color = matplotlib.colors.to_hex(rgba)

        radius = max(6, min(18, int(math.log(info["population"] + 1) * 1.2)))
        border_color = "#000000" if name == selected_city else "#555555"
        border_weight = 3 if name == selected_city else 1

        popup_text = (
            f"<b>{name}</b><br>"
            f"Region: {info['region']}<br>"
            f"Population: {info['population']:,}<br>"
            f"{value_col.replace('_', ' ').title()}: "
            f"{val:.3f}" if not np.isnan(val) else "N/A"
        )
        folium.CircleMarker(
            location=[info["lat"], info["lon"]],
            radius=radius,
            color=border_color,
            weight=border_weight,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=f"{name}: {val:.3f}" if not np.isnan(val) else name,
        ).add_to(fmap)

    return fmap


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------

def plot_timeseries(
    df: pd.DataFrame,
    variable: str,
    title: str = "",
) -> go.Figure:
    """Line chart of monthly mean values."""
    if df.empty:
        return go.Figure().update_layout(title="No data available")

    mean_col = f"{variable}_mean"
    unit = "" if variable == "ndvi" else " °C"
    color = "#2ECC71" if variable == "ndvi" else "#E74C3C"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df[mean_col],
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=7),
        name="Mean",
    ))

    # Vertical year boundary lines
    years = df["date"].dt.year.unique()
    for y in sorted(years)[1:]:
        fig.add_vline(
            x=str(pd.Timestamp(f"{y}-01-01")),
            line_width=1,
            line_dash="dash",
            line_color="gray",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=variable.upper() + unit,
        xaxis=dict(tickformat="%b %Y"),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=40, b=50),
        template="plotly_white",
    )
    return fig


def plot_yearly_bar(
    df: pd.DataFrame,
    variable: str,
    entity: str,
    title: str = "",
) -> go.Figure:
    """Grouped bar chart comparing yearly means with stdev error bars."""
    if df.empty:
        return go.Figure().update_layout(title="No data available")

    mean_col = f"{variable}_mean"
    std_col = f"{variable}_std"
    unit = "" if variable == "ndvi" else " °C"
    color = "#27AE60" if variable == "ndvi" else "#C0392B"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["year"].astype(str),
        y=df[mean_col],
        error_y=dict(type="data", array=df[std_col].fillna(0), visible=True),
        marker_color=color,
        name=entity,
    ))
    fig.update_layout(
        title=title or f"{variable.upper()} by Year — {entity}",
        xaxis_title="Year",
        yaxis_title=variable.upper() + unit,
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig


def plot_monthly_heatmap(
    df: pd.DataFrame,
    variable: str,
    title: str = "",
) -> go.Figure:
    """
    Heatmap: rows = entities or years, columns = months 1–12.
    Pivots df on (entity, month) taking the mean across years.
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available")

    mean_col = f"{variable}_mean"
    pivot = df.pivot_table(index="entity", columns="month", values=mean_col, aggfunc="mean")
    pivot = pivot.reindex(columns=range(1, 13))

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    colorscale = "RdYlGn" if variable == "ndvi" else "RdYlBu_r"
    unit = "" if variable == "ndvi" else " °C"

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=month_labels,
        y=pivot.index.tolist(),
        colorscale=colorscale,
        colorbar=dict(title=variable.upper() + unit),
        hoverongaps=False,
    ))
    fig.update_layout(
        title=title or f"Monthly {variable.upper()} by Entity",
        xaxis_title="Month",
        yaxis_title="",
        template="plotly_white",
        margin=dict(l=120, r=20, t=40, b=50),
    )
    return fig


def plot_stats_bar(stats: dict, label: str, variable: str) -> go.Figure:
    """Horizontal range bar: min → p25 → mean → p75 → max."""
    unit = "" if variable == "ndvi" else " °C"
    keys = ["min", "p25", "mean", "p75", "max"]
    values = [stats.get(k, np.nan) for k in keys]

    color = "#27AE60" if variable == "ndvi" else "#C0392B"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values,
        y=[""] * len(values),
        mode="markers+lines",
        marker=dict(size=[8, 8, 14, 8, 8], color=color),
        line=dict(color=color, width=2),
        text=[f"{k}: {v:.3f}{unit}" if not np.isnan(v) else k for k, v in zip(keys, values)],
        hoverinfo="text",
    ))
    fig.update_layout(
        title=f"{variable.upper()} distribution — {label}",
        xaxis_title=variable.upper() + unit,
        height=120,
        margin=dict(l=10, r=10, t=30, b=30),
        template="plotly_white",
        showlegend=False,
    )
    return fig


def plot_ndvi_lst_scatter(
    df: pd.DataFrame,
    title: str = "",
) -> go.Figure:
    """
    Scatter: NDVI mean (x) vs LST mean (y), colored by month.
    Shows the vegetation-cooling relationship.
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available")

    month_labels = {
        1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec",
    }
    df = df.dropna(subset=["ndvi_mean", "lst_mean"])

    fig = go.Figure()
    for month, grp in df.groupby("month"):
        fig.add_trace(go.Scatter(
            x=grp["ndvi_mean"],
            y=grp["lst_mean"],
            mode="markers",
            marker=dict(size=9),
            name=month_labels.get(month, str(month)),
            text=grp.get("entity", pd.Series()),
            hovertemplate="NDVI: %{x:.3f}<br>LST: %{y:.1f} °C<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title=title or "NDVI vs LST (vegetation-cooling relationship)",
        xaxis_title="NDVI mean",
        yaxis_title="LST mean (°C)",
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=50),
        legend_title="Month",
    )
    return fig


def plot_uhi_bar(
    cities_stats: pd.DataFrame,
    country_lst_mean: float,
    title: str = "",
) -> go.Figure:
    """
    Horizontal bar chart of UHI intensity per city (city LST mean − country LST mean).
    Bars are sorted descending.
    """
    if cities_stats.empty:
        return go.Figure().update_layout(title="No data available")

    df = cities_stats.dropna(subset=["lst_mean"]).copy()
    df["uhi"] = df["lst_mean"] - country_lst_mean
    df = df.sort_values("uhi", ascending=True)

    colors = ["#E74C3C" if v >= 0 else "#3498DB" for v in df["uhi"]]
    fig = go.Figure(go.Bar(
        x=df["uhi"],
        y=df["entity"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f} °C" for v in df["uhi"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="black")
    fig.update_layout(
        title=title or "Urban Heat Island Intensity vs Country Mean",
        xaxis_title="LST anomaly (°C)",
        template="plotly_white",
        margin=dict(l=120, r=60, t=40, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Analysis: NDVI vs LST pixel scatter, vegetation cooling, UHI
# ---------------------------------------------------------------------------

def plot_pixel_scatter(
    ndvi_arr: np.ndarray,
    lst_arr: np.ndarray,
    title: str = "NDVI vs LST — Pixel-level",
    max_points: int = 5000,
) -> go.Figure:
    """Scatter plot of NDVI vs LST at pixel level with trendline.
    Shows that higher NDVI (vegetation) correlates with lower temperature."""
    # Flatten and pair valid pixels
    ndvi_flat = ndvi_arr.ravel()
    lst_flat = lst_arr.ravel()
    valid = np.isfinite(ndvi_flat) & np.isfinite(lst_flat)
    ndvi_v = ndvi_flat[valid]
    lst_v = lst_flat[valid]

    if len(ndvi_v) == 0:
        return go.Figure().update_layout(title="No valid pixel pairs")

    # Subsample for performance
    if len(ndvi_v) > max_points:
        idx = np.random.default_rng(42).choice(len(ndvi_v), max_points, replace=False)
        ndvi_v = ndvi_v[idx]
        lst_v = lst_v[idx]

    # Linear trendline
    mask = np.isfinite(ndvi_v) & np.isfinite(lst_v)
    if np.sum(mask) > 10:
        coeffs = np.polyfit(ndvi_v[mask], lst_v[mask], 1)
        trend_x = np.linspace(np.min(ndvi_v), np.max(ndvi_v), 100)
        trend_y = np.polyval(coeffs, trend_x)
        r = np.corrcoef(ndvi_v[mask], lst_v[mask])[0, 1]
    else:
        coeffs, trend_x, trend_y, r = None, None, None, 0

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=ndvi_v, y=lst_v,
        mode="markers",
        marker=dict(size=3, color=lst_v, colorscale="RdYlBu_r", opacity=0.5),
        name="Pixels",
        hovertemplate="NDVI: %{x:.2f}<br>Temp: %{y:.1f}°C<extra></extra>",
    ))

    if trend_x is not None:
        fig.add_trace(go.Scatter(
            x=trend_x, y=trend_y,
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name=f"Trend (r={r:.2f}, slope={coeffs[0]:.1f}°C/NDVI)",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="NDVI",
        yaxis_title="Surface Temp (°C)",
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=50),
        height=400,
    )
    return fig


def compute_vegetation_cooling(ndvi_arr: np.ndarray, lst_arr: np.ndarray) -> dict:
    """Compute how much cooler vegetated areas are vs bare/urban areas.

    Returns dict with:
        vegetated_mean: mean LST where NDVI > 0.4
        bare_mean: mean LST where NDVI < 0.2
        cooling_effect: bare_mean - vegetated_mean (positive = vegetation cools)
        correlation: Pearson r between NDVI and LST
    """
    valid = np.isfinite(ndvi_arr) & np.isfinite(lst_arr)
    ndvi_v = ndvi_arr[valid]
    lst_v = lst_arr[valid]

    vegetated = lst_v[ndvi_v > 0.4]
    bare = lst_v[ndvi_v < 0.2]

    veg_mean = float(np.mean(vegetated)) if len(vegetated) > 0 else np.nan
    bare_mean = float(np.mean(bare)) if len(bare) > 0 else np.nan
    cooling = bare_mean - veg_mean if np.isfinite(bare_mean) and np.isfinite(veg_mean) else np.nan

    r = float(np.corrcoef(ndvi_v, lst_v)[0, 1]) if len(ndvi_v) > 10 else np.nan

    return {
        "vegetated_mean": veg_mean,
        "bare_mean": bare_mean,
        "cooling_effect": cooling,
        "correlation": r,
        "n_vegetated": len(vegetated),
        "n_bare": len(bare),
    }
