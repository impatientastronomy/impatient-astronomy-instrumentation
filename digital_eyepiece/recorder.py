"""
Session recorder — saves raw (pre-calibration) frames during a recording session.

A session folder is created when recording starts; each frame is written as a
uint16 TIFF with FrameMeta serialized as JSON in the ImageDescription tag.

save() is designed to be called from the grab worker thread so disk I/O never
blocks the render loop.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path

import tifffile

from astrocore.camera.base import Frame
from astrocore.camera.naming import frame_filename


class Recorder:
    """Manages one recording session: folder creation, naming, and TIFF writes."""

    def __init__(self, record_path: Path) -> None:
        self._record_path = record_path
        self._session_dir: Path | None = None
        self._counter: itertools.count = itertools.count(1)
        self.active: bool = False

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def start(self) -> None:
        """Create session folder and begin recording."""
        self._record_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_dir = self._record_path / ts
        self._session_dir.mkdir(exist_ok=True)
        self._counter = itertools.count(1)
        self.active = True

    def stop(self) -> None:
        self.active = False

    def save(self, frame: Frame) -> None:
        """Write one raw frame to disk. Thread-safe; no-op when not active."""
        if not self.active or self._session_dir is None:
            return
        index = next(self._counter)
        path = self._session_dir / frame_filename(frame.meta, index)
        tifffile.imwrite(
            str(path),
            frame.data,
            description=json.dumps(_meta_to_dict(frame.meta)),
        )


def _meta_to_dict(meta) -> dict:
    return {
        "camera_id":             meta.camera_id,
        "camera_model":          meta.camera_model,
        "timestamp":             meta.timestamp.isoformat() if meta.timestamp else None,
        "exposure_seconds":      meta.exposure_seconds,
        "gain":                  meta.gain,
        "offset":                meta.offset,
        "binning":               meta.binning,
        "temperature_c":         meta.temperature_c,
        "bayer_pattern":         meta.bayer_pattern,
        "roi_x":                 meta.roi_x,
        "roi_y":                 meta.roi_y,
        "roi_width":             meta.roi_width,
        "roi_height":            meta.roi_height,
        "flip":                  meta.flip,
        "telescope_description": meta.telescope_description,
        "focal_length_mm":       meta.focal_length_mm,
        "RA":                    meta.RA,
        "Dec":                   meta.Dec,
        "Lat":                   meta.Lat,
        "Lon":                   meta.Lon,
        "filter_id":             meta.filter_id,
        "extras":                meta.extras,
    }
