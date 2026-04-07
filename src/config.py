# config.py
from sentinelhub import BBox, CRS

# Study area
PRAGUE_BBOX = BBox(
    bbox=[14.22, 49.94, 14.71, 50.18],
    crs=CRS.WGS84
)

# Resolution
RESOLUTION = 60  # meters

# Time range — peak summer for max heat island effect
TIME_INTERVAL = ("2024-07-01", "2024-07-31")
