#!/usr/bin/env python3
"""
acquire_darks.py — Capture averaged dark calibration frames for all connected cameras.

All cameras are acquired simultaneously. For each exposure time, N frames are
captured, averaged, and saved as a single uint16 TIFF. The averaged sensor
temperature (rounded to the nearest °C) is embedded in the filename so that
FrameGrabber can select the closest match at runtime.

Darks are saved at full sensor size (no ROI) so they can be used with any
science ROI setting — FrameGrabber crops them to match at load time.

Existing dark files are overwritten without warning.

Output path: <cal_path>/darks/   (cal_path defined in configuration.yaml)

Usage
-----
    python acquire_darks.py -e 0.1 1 2 5 10
    python acquire_darks.py -e 2 5 -n 20
    python acquire_darks.py -e 0.1 --config /path/to/configuration.yaml
"""

import argparse
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from astrocore.camera.base import FrameMeta
from astrocore.camera.naming import frame_filename
from astrocore.camera.zwo_asi import FlipMode, ZwoAsiCamera, list_cameras
from astrocore.config.camera_config import load as load_config

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configuration.yaml"

_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-e", "--exposures",
        nargs="+", type=float, metavar="S", required=True,
        help="Exposure times in seconds, e.g.  -e 0.1 1 2 5 10",
    )
    p.add_argument(
        "-n", "--frames",
        type=int, default=10, metavar="N",
        help="Number of frames to average per exposure (default: 10)",
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, metavar="PATH",
        help=f"Path to configuration.yaml (default: {_DEFAULT_CONFIG.name})",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-camera worker
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _acquire_camera(
    usb_index: int,
    camera_id: int,
    model: str,
    exposures_s: list[float],
    n_frames: int,
    darks_dir: Path,
    cam_config,
    stop: threading.Event,
    errors: list,
) -> None:
    """Capture and save darks for one camera. Runs in its own thread."""
    tag = f"Cam{camera_id}"
    try:
        with ZwoAsiCamera(index=usb_index) as cam:
            # Apply settings from configuration so darks match science frames.
            if cam_config.gain is not None:
                cam.gain = cam_config.gain
            if cam_config.offset is not None:
                cam.offset = cam_config.offset
            cam.flip = FlipMode(cam_config.flip)
            # Full-sensor ROI at the configured bin — no x/y offset.
            cam.set_roi(x=0, y=0, width=None, height=None, bin=cam_config.bin)

            for exp_s in exposures_s:
                if stop.is_set():
                    return

                _log(f"  {tag}: capturing {n_frames}× {exp_s}s")
                cam.exposure_time = exp_s
                timeout = exp_s + 15.0

                raw_frames: list[np.ndarray] = []
                temps: list[float] = []

                for _ in range(n_frames):
                    if stop.is_set():
                        return
                    frame = cam.expose(is_dark=True, timeout=timeout)
                    raw_frames.append(frame.data.astype(np.float32))
                    if frame.meta.temperature_c is not None:
                        temps.append(frame.meta.temperature_c)

                averaged = (
                    np.mean(raw_frames, axis=0)
                    .clip(0, 65535)
                    .round()
                    .astype(np.uint16)
                )
                avg_temp = float(np.mean(temps)) if temps else None

                meta = FrameMeta(
                    camera_id    = camera_id,
                    camera_model = model,
                    timestamp    = datetime.now(tz=timezone.utc),
                    exposure_seconds = exp_s,
                    gain         = cam.gain,
                    offset       = cam.offset,
                    binning      = cam_config.bin,
                    temperature_c = avg_temp,
                    flip         = FlipMode(cam_config.flip).name,
                    filter_id    = 0,
                    # No roi_x/y — darks are full-sensor; FrameGrabber crops at load
                )
                filename = frame_filename(meta, index=1)
                tifffile.imwrite(str(darks_dir / filename), averaged)
                _log(f"  {tag}: saved {filename}")

    except Exception as exc:
        errors.append((camera_id, exc))
        _log(f"  Cam{camera_id}: ERROR — {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.frames < 1:
        print("Error: --frames must be at least 1.")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if config.cal_path is None:
        print("Error: cal_path is not set in configuration.yaml.")
        sys.exit(1)

    # Discover cameras
    print()
    cameras = list_cameras()
    if not cameras:
        print("No ZWO ASI cameras detected.")
        sys.exit(1)

    print(f"Connected cameras ({len(cameras)}):")
    for c in cameras:
        print(f"  USB {c['usb_index']}:  {c['model']}  ID={c['camera_id']}")

    unset = [c for c in cameras if c["camera_id"] == 0]
    if unset:
        noun = "camera has" if len(unset) == 1 else "cameras have"
        print(
            f"\n{len(unset)} {noun} no ID set "
            f"(USB index {[c['usb_index'] for c in unset]}). "
            "Run set_camera_id.py first."
        )
        sys.exit(1)

    darks_dir = config.cal_path / "darks"
    darks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExposures : {args.exposures} s")
    print(f"Frames    : {args.frames} per exposure")
    print(f"Output    : {darks_dir}")

    try:
        input("\nCap all cameras, then press Enter to start (Ctrl+C to abort)...")
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    print()

    # Launch one thread per camera
    stop  = threading.Event()
    errors: list = []
    threads: list[threading.Thread] = []

    for c in cameras:
        cam_config = config.get_config(c["camera_id"])
        t = threading.Thread(
            target  = _acquire_camera,
            args    = (
                c["usb_index"], c["camera_id"], c["model"],
                args.exposures, args.frames,
                darks_dir, cam_config, stop, errors,
            ),
            name   = f"cam{c['camera_id']}",
            daemon = True,
        )
        threads.append(t)

    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted — waiting for current frames to finish...")
        stop.set()
        for t in threads:
            t.join()

    print()
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for cam_id, exc in errors:
            print(f"  Camera {cam_id}: {exc}")
        sys.exit(1)

    total = len(cameras) * len(args.exposures)
    print(f"Done — {total} dark file(s) saved to {darks_dir}")


if __name__ == "__main__":
    main()
