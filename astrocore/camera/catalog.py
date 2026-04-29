"""
Folder scanner that builds a pandas DataFrame from canonically named frame files.

Usage::

    df = scan_folder("/data/frames")

    # Camera 3 frames with the longest exposure
    best = (
        df[df.camera_id == 3]
        .sort_values("exposure_us", ascending=False)
        .iloc[0]
    )
    print(best.path)

    # All Hydrogen-alpha frames
    ha = df[df.filter_name == "Ha"]

Columns returned
----------------
filename      str    bare filename, e.g. C3_Ha_1250e3us_22C_001.tif
path          Path   absolute path to the file
camera_id     int    camera identifier
filter_name   str    filter name (e.g. 'Ha', 'OIII', 'none')
exposure_us   int    exposure time in microseconds — use for sorting/comparison
exposure_s    float  exposure time in seconds — human-readable
temperature_c float  sensor temperature in °C (NaN if unknown)
frame_index   int    frame index within the sequence

Note: columns are named filter_name and frame_index (rather than filter and
index) because both of those are reserved attribute names in pandas.

Files that don't match the naming convention are silently skipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .naming import parse_filename

_COLUMNS = [
    "filename", "path", "camera_id", "filter_name",
    "exposure_us", "exposure_s", "temperature_c", "frame_index",
]


def scan_folder(path: str | Path) -> pd.DataFrame:
    """
    Scan *path* for .tif files and return a DataFrame of parsed frame metadata.

    The DataFrame is sorted by filename. Files that don't match the naming
    convention are silently ignored.
    """
    folder = Path(path)
    rows = []
    for f in sorted(folder.glob("*.tif")):
        parsed = parse_filename(f.name)
        if parsed is not None:
            parsed["filename"] = f.name
            parsed["path"] = f
            rows.append(parsed)

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["camera_id"] = df["camera_id"].astype(int)
    df["exposure_us"] = df["exposure_us"].astype(int)
    df["frame_index"] = df["frame_index"].astype(int)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    return df
