"""
Geographic constants and helpers for the Czech Republic dashboard.
REGIONS and CITIES are the single source of truth for all spatial lookups.
"""
import json
from pathlib import Path
from sentinelhub import BBox, CRS


CZ_BBOX = [12.09, 48.55, 18.87, 51.06]
CZ_RASTER_SIZE = (570, 1200)  # (H, W) at ~500 m effective resolution

# 14 NUTS-3 regions (kraje) — bboxes are manually curated
REGIONS: dict[str, dict] = {
    "Praha": {
        "bbox": [14.22, 49.94, 14.71, 50.18],
        "centroid": [14.46, 50.06],
        "nuts3": "CZ010",
    },
    "Středočeský": {
        "bbox": [13.55, 49.58, 15.33, 50.36],
        "centroid": [14.65, 49.95],
        "nuts3": "CZ020",
    },
    "Jihočeský": {
        "bbox": [13.38, 48.55, 15.28, 49.59],
        "centroid": [14.40, 49.05],
        "nuts3": "CZ031",
    },
    "Plzeňský": {
        "bbox": [12.55, 49.15, 13.93, 50.02],
        "centroid": [13.30, 49.58],
        "nuts3": "CZ032",
    },
    "Karlovarský": {
        "bbox": [12.09, 49.89, 13.25, 50.44],
        "centroid": [12.75, 50.15],
        "nuts3": "CZ041",
    },
    "Ústecký": {
        "bbox": [13.19, 50.21, 14.91, 50.96],
        "centroid": [14.00, 50.58],
        "nuts3": "CZ042",
    },
    "Liberecký": {
        "bbox": [14.74, 50.57, 15.55, 51.06],
        "centroid": [15.05, 50.80],
        "nuts3": "CZ051",
    },
    "Královéhradecký": {
        "bbox": [15.50, 50.12, 16.62, 50.82],
        "centroid": [16.03, 50.45],
        "nuts3": "CZ052",
    },
    "Pardubický": {
        "bbox": [15.52, 49.60, 16.80, 50.22],
        "centroid": [16.15, 49.88],
        "nuts3": "CZ053",
    },
    "Kraj Vysočina": {
        "bbox": [15.15, 49.25, 16.55, 49.80],
        "centroid": [15.82, 49.52],
        "nuts3": "CZ063",
    },
    "Jihomoravský": {
        "bbox": [15.80, 48.62, 18.00, 49.65],
        "centroid": [16.85, 49.10],
        "nuts3": "CZ064",
    },
    "Olomoucký": {
        "bbox": [16.60, 49.40, 17.80, 50.12],
        "centroid": [17.25, 49.75],
        "nuts3": "CZ071",
    },
    "Zlínský": {
        "bbox": [17.33, 48.98, 18.46, 49.68],
        "centroid": [17.90, 49.32],
        "nuts3": "CZ072",
    },
    "Moravskoslezský": {
        "bbox": [17.40, 49.42, 18.87, 50.00],
        "centroid": [18.20, 49.68],
        "nuts3": "CZ080",
    },
}

CITIES: dict[str, dict] = {
    "Praha": {
        "lat": 50.0755, "lon": 14.4378, "population": 1_309_000, "region": "Praha",
    },
    "Brno": {
        "lat": 49.1951, "lon": 16.6068, "population": 382_000, "region": "Jihomoravský",
    },
    "Ostrava": {
        "lat": 49.8209, "lon": 18.2625, "population": 284_000, "region": "Moravskoslezský",
    },
    "Plzeň": {
        "lat": 49.7384, "lon": 13.3736, "population": 174_000, "region": "Plzeňský",
    },
    "Liberec": {
        "lat": 50.7663, "lon": 15.0543, "population": 104_000, "region": "Liberecký",
    },
    "Olomouc": {
        "lat": 49.5938, "lon": 17.2509, "population": 100_000, "region": "Olomoucký",
    },
    "České Budějovice": {
        "lat": 48.9747, "lon": 14.4742, "population": 94_000, "region": "Jihočeský",
    },
    "Hradec Králové": {
        "lat": 50.2092, "lon": 15.8328, "population": 92_000, "region": "Královéhradecký",
    },
    "Ústí nad Labem": {
        "lat": 50.6607, "lon": 14.0323, "population": 91_000, "region": "Ústecký",
    },
    "Pardubice": {
        "lat": 50.0343, "lon": 15.7812, "population": 91_000, "region": "Pardubický",
    },
}


def build_regions_geojson_from_bboxes() -> dict:
    """
    Returns a GeoJSON FeatureCollection where each feature is a rectangular Polygon
    derived from the REGIONS bbox. Adequate for choropleth coloring and tooltips.

    To use accurate NUTS-3 boundaries, replace the generated file at
    dashboard/data/cz_regions.geojson with the EUROSTAT GeoJSON, ensuring each
    feature has properties.name matching the keys in REGIONS.
    """
    features = []
    for name, data in REGIONS.items():
        min_lon, min_lat, max_lon, max_lat = data["bbox"]
        coords = [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]]
        features.append({
            "type": "Feature",
            "properties": {"name": name, "nuts3": data["nuts3"]},
            "geometry": {"type": "Polygon", "coordinates": coords},
        })
    return {"type": "FeatureCollection", "features": features}


def load_geojson(path: str) -> dict:
    """Load a GeoJSON file from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_region_bbox(region_name: str) -> list[float]:
    """Return [min_lon, min_lat, max_lon, max_lat] for the named region."""
    return REGIONS[region_name]["bbox"]


def get_region_bbox_from_geojson(region_name: str, geojson: dict, pad: float = 0.02) -> list[float]:
    """Compute bbox from actual GeoJSON geometry (more accurate than manual bbox).
    Adds `pad` degrees of padding on each side."""
    from shapely.geometry import shape
    for feat in geojson.get("features", []):
        if feat["properties"].get("name") == region_name:
            bounds = shape(feat["geometry"]).bounds  # (minx, miny, maxx, maxy)
            return [
                bounds[0] - pad,
                bounds[1] - pad,
                bounds[2] + pad,
                bounds[3] + pad,
            ]
    # Fallback to manual bbox
    return REGIONS[region_name]["bbox"]


def get_city_bbox(city_name: str, buffer_deg: float = 0.15) -> list[float]:
    """Return a square bbox around the city centroid with buffer_deg padding."""
    city = CITIES[city_name]
    lat, lon = city["lat"], city["lon"]
    return [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]


def bbox_to_sh_bbox(bbox: list[float]) -> BBox:
    """Wrap [min_lon, min_lat, max_lon, max_lat] in a sentinelhub BBox object."""
    return BBox(bbox=bbox, crs=CRS.WGS84)


# ---------------------------------------------------------------------------
# Real NUTS-3 boundary download
# ---------------------------------------------------------------------------

_GEOJSON_PATH = Path(__file__).resolve().parents[1] / "data" / "cz_regions.geojson"

# NUTS3 code → our canonical region name (keys in REGIONS)
_NUTS3_TO_NAME = {v["nuts3"]: k for k, v in REGIONS.items()}


def download_cz_nuts3_boundaries(save_path: Path | str | None = None) -> dict:
    """
    Download Czech Republic NUTS-3 boundaries from the EUROSTAT GISCO service,
    attach canonical region names matching our REGIONS keys, and save to disk.

    Args:
        save_path: where to save the GeoJSON (defaults to dashboard/data/cz_regions.geojson)

    Returns:
        GeoJSON FeatureCollection dict
    """
    import urllib.request

    save_path = Path(save_path) if save_path else _GEOJSON_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    url = (
        "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
        "NUTS_RG_01M_2021_4326_LEVL_3.geojson"
    )
    print(f"Downloading NUTS-3 boundaries from EUROSTAT …")
    with urllib.request.urlopen(url, timeout=60) as resp:
        all_nuts = json.loads(resp.read().decode("utf-8"))

    # Keep only Czech regions and attach our canonical names
    cz_features = []
    for feature in all_nuts["features"]:
        props = feature["properties"]
        if props.get("CNTR_CODE") != "CZ":
            continue
        nuts_id = props.get("NUTS_ID", "")
        canonical_name = _NUTS3_TO_NAME.get(nuts_id)
        if canonical_name is None:
            continue  # unknown region — skip
        feature["properties"]["name"] = canonical_name
        cz_features.append(feature)

    geojson = {"type": "FeatureCollection", "features": cz_features}

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    print(f"Saved {len(cz_features)} region boundaries to {save_path}")
    return geojson


_CITIES_GEOJSON_PATH = Path(__file__).resolve().parents[1] / "data" / "cz_cities.geojson"


def download_cz_city_boundaries(save_path: Path | str | None = None) -> dict:
    """
    Download city boundaries from OpenStreetMap Nominatim for each city in CITIES.
    Saves to dashboard/data/cz_cities.geojson.
    """
    import time
    import urllib.request

    save_path = Path(save_path) if save_path else _CITIES_GEOJSON_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    for name in CITIES:
        print(f"  Fetching boundary for {name}...", end=" ", flush=True)
        query = urllib.parse.quote(f"{name}, Czech Republic")
        url = (
            f"https://nominatim.openstreetmap.org/search?"
            f"q={query}&format=json&polygon_geojson=1&limit=1"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "sentinel-analysis-dashboard/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read().decode("utf-8"))

        if results and "geojson" in results[0]:
            geom = results[0]["geojson"]
            features.append({
                "type": "Feature",
                "properties": {"name": name},
                "geometry": geom,
            })
            print("OK")
        else:
            print("NOT FOUND")

        time.sleep(1.1)  # Nominatim rate limit: 1 req/sec

    geojson = {"type": "FeatureCollection", "features": features}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    print(f"Saved {len(features)} city boundaries to {save_path}")
    return geojson


def load_cities_geojson() -> dict | None:
    """Load city boundaries GeoJSON, or None if not downloaded yet."""
    if _CITIES_GEOJSON_PATH.exists():
        return load_geojson(str(_CITIES_GEOJSON_PATH))
    return None


def get_city_bbox_from_geojson(city_name: str, geojson: dict, pad: float = 0.02) -> list[float]:
    """Compute bbox from actual city GeoJSON geometry."""
    from shapely.geometry import shape
    for feat in geojson.get("features", []):
        if feat["properties"].get("name") == city_name:
            bounds = shape(feat["geometry"]).bounds
            return [bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad]
    # Fallback
    return get_city_bbox(city_name)


def load_regions_geojson() -> dict:
    """
    Load Czech region boundaries.  Uses the real NUTS-3 GeoJSON if it has been
    downloaded (dashboard/data/cz_regions.geojson), otherwise falls back to
    rectangle polygons derived from REGIONS bboxes.
    """
    if _GEOJSON_PATH.exists():
        with open(_GEOJSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    return build_regions_geojson_from_bboxes()
