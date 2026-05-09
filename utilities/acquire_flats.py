#!/usr/bin/env python3
"""
acquire_flats.py — Capture averaged flat calibration frames for all connected cameras.

All cameras are acquired simultaneously.  The script auto-adjusts the exposure
time per camera until the mean ADU is close to the target (default 30 000).
Once settled, N frames are captured, averaged, and converted to a per-pixel
scaling map normalised so that the brightest pixel = 1000:

    flat_map = uint16( 1000 * max(averaged) / averaged )

When applied during calibration the science frame is multiplied by
flat_map / 1000, boosting under-illuminated pixels to correct vignetting and
sensor non-uniformity.

Flats are saved at full sensor size (no ROI).  FrameGrabber matches them by
camera_id, filter_id, and binning — exposure and temperature in the filename
are ignored during matching.

Existing flat files are overwritten without warning.

Output path: <cal_path>/flats/   (cal_path defined in configuration.yaml)

Usage
-----
    python acquire_flats.py -f 0
    python acquire_flats.py -f 1 --target-adu 25000 -n 32
    python acquire_flats.py -f 3 --config /path/to/configuration.yaml
"""

import argparse
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astrocore.camera.base import FrameMeta
from astrocore.camera.catalog import scan_folder
from astrocore.camera.naming import frame_filename
from astrocore.camera.zwo_asi import FlipMode, ZwoAsiCamera, list_cameras
from astrocore.config.camera_config import load as load_config

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configuration.yaml"

_BIAS_EXP_US  = 100     # 100 µs — camera minimum exposure; bias dark match key
_MAX_EXP_S    = 30.0
_CONVERGE_TOL = 0.05    # accept within ±5% of target ADU
_MAX_SETTLE   = 10      # max auto-exposure iterations

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
        "-f", "--filter",
        type=int, required=True, metavar="N",
        help="Filter slot number (must match filters defined in configuration.yaml)",
    )
    p.add_argument(
        "--target-adu",
        type=int, default=30_000, metavar="ADU",
        help="Target mean ADU for the flat (default: 30000)",
    )
    p.add_argument(
        "-n", "--frames",
        type=int, default=16, metavar="N",
        help="Number of frames to average (default: 16)",
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


def _load_bias_dark(darks_dir: Path, camera_id: int, bin_: int, tag: str) -> np.ndarray:
    """
    Load the 100 µs dark from darks_dir for this camera/bin combination.
    Raises FileNotFoundError if no matching file exists.
    """
    if not darks_dir.exists():
        raise FileNotFoundError(
            f"Dark calibration directory not found: {darks_dir}. "
            "Run: uv run python utilities/acquire_darks.py -e 0.0001"
        )
    table = scan_folder(darks_dir)
    candidates = table[
        (table.camera_id == camera_id) &
        (table["bin"] == bin_) &
        (table.exposure_us == _BIAS_EXP_US)
    ]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"{tag}: no 100 µs dark found in {darks_dir} "
            f"for camera_id={camera_id} bin={bin_}. "
            "Run: uv run python utilities/acquire_darks.py -e 0.0001"
        )
    path = candidates.iloc[0]["path"]
    _log(f"  {tag}: loaded bias dark {Path(path).name}")
    return tifffile.imread(str(path)).astype(np.float32)


def _settle_exposure(
    cam: ZwoAsiCamera,
    target_adu: int,
    tag: str,
    stop: threading.Event,
    bias: np.ndarray,
) -> float:
    """
    Proportionally adjust exposure until bias-corrected mean ADU ≈ target.
    Raises ValueError if the needed exposure would fall below the 100 µs camera minimum.
    Returns the settled exposure in seconds.
    """
    exp_s = 1.0
    cam.exposure_time = exp_s
    mean_adu = 0.0

    for i in range(_MAX_SETTLE):
        if stop.is_set():
            return exp_s
        frame = cam.expose(timeout=exp_s + 5.0)
        corrected = np.maximum(frame.data.astype(np.float32) - bias, 0.0)
        mean_adu = float(np.mean(corrected))
        _log(f"  {tag}: settle {i + 1}/{_MAX_SETTLE}  exp={exp_s:.4f}s  mean={mean_adu:.0f} ADU")

        if mean_adu < 1.0:
            exp_s = min(exp_s * 4, _MAX_EXP_S)
        else:
            if abs(mean_adu - target_adu) / target_adu <= _CONVERGE_TOL:
                break
            needed = exp_s * target_adu / mean_adu
            if needed * 1e6 < _BIAS_EXP_US:
                raise ValueError(
                    f"flat source too bright — needed exposure "
                    f"{needed * 1e6:.1f} µs is below the 100 µs camera minimum"
                )
            exp_s = min(needed, _MAX_EXP_S)

        cam.exposure_time = exp_s

    _log(f"  {tag}: exposure settled → {exp_s:.4f}s  (mean {mean_adu:.0f} ADU)")
    return exp_s


def _acquire_camera(
    usb_index: int,
    camera_id: int,
    model: str,
    filter_id: int,
    n_frames: int,
    target_adu: int,
    flats_dir: Path,
    darks_dir: Path,
    cam_config,
    stop: threading.Event,
    errors: list,
) -> None:
    """Capture and save a flat for one camera. Runs in its own thread."""
    tag = f"Cam{camera_id}"
    try:
        bias = _load_bias_dark(darks_dir, camera_id, cam_config.bin, tag)

        with ZwoAsiCamera(index=usb_index) as cam:
            if cam_config.gain is not None:
                cam.gain = cam_config.gain
            if cam_config.offset is not None:
                cam.offset = cam_config.offset
            cam.flip = FlipMode(cam_config.flip)
            cam.set_roi(x=0, y=0, width=None, height=None, bin=cam_config.bin)

            exp_s = _settle_exposure(cam, target_adu, tag, stop, bias)
            if stop.is_set():
                return

            _log(f"  {tag}: capturing {n_frames} flat frames at {exp_s:.4f}s")
            timeout = exp_s + 15.0

            raw_frames: list[np.ndarray] = []

            for _ in range(n_frames):
                if stop.is_set():
                    return
                frame = cam.expose(timeout=timeout)
                corrected = np.maximum(frame.data.astype(np.float32) - bias, 0.0)
                raw_frames.append(corrected)

            averaged = np.mean(raw_frames, axis=0)  # bias-corrected

            max_val = float(np.max(averaged))
            if max_val <= 0:
                raise ValueError("averaged flat is all zeros")

            # Per-pixel scaling map: 1000 = unity gain at the brightest pixel.
            # Dim pixels receive values > 1000 to compensate for vignetting.
            flat_map = (
                (10000.0 * max_val / np.maximum(averaged, 1.0))
                .clip(0, 65535)
                .round()
                .astype(np.uint16)
            )

            meta = FrameMeta(
                camera_id        = camera_id,
                camera_model     = model,
                timestamp        = datetime.now(tz=timezone.utc),
                exposure_seconds = _BIAS_EXP_US / 1_000_000,  # fixed 100us in filename
                gain             = cam.gain,
                offset           = cam.offset,
                binning          = cam_config.bin,
                temperature_c    = 20.0,                       # fixed 20C in filename
                flip             = FlipMode(cam_config.flip).name,
                filter_id        = filter_id,
            )
            filename = frame_filename(meta, index=1)
            tifffile.imwrite(str(flats_dir / filename), flat_map)
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
    if args.target_adu < 1:
        print("Error: --target-adu must be at least 1.")
        sys.exit(1)

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if config.cal_path is None:
        print("Error: cal_path is not set in configuration.yaml.")
        sys.exit(1)

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

    filter_name = config.filters.get(args.filter, str(args.filter))
    flats_dir = config.cal_path / "flats"
    darks_dir = config.cal_path / "darks"
    flats_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFilter    : {args.filter} ({filter_name})")
    print(f"Target ADU: {args.target_adu}")
    print(f"Frames    : {args.frames} per camera")
    print(f"Output    : {flats_dir}")

    try:
        input("\nIlluminate flat source, then press Enter to start (Ctrl+C to abort)...")
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    print()

    stop   = threading.Event()
    errors: list = []
    threads: list[threading.Thread] = []

    for c in cameras:
        cam_config = config.get_config(c["camera_id"])
        t = threading.Thread(
            target = _acquire_camera,
            args   = (
                c["usb_index"], c["camera_id"], c["model"],
                args.filter, args.frames, args.target_adu,
                flats_dir, darks_dir, cam_config, stop, errors,
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

    print(f"Done — {len(cameras)} flat file(s) saved to {flats_dir}")


if __name__ == "__main__":
    main()
