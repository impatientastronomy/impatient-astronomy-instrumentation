---
name: Camera stack design decisions
description: Key decisions about the camera abstraction, naming convention, catalog, FrameGrabber, and live viewer pipeline
type: project
---

# Camera stack design decisions

## Canonical TIFF naming
`C{id}_{filter}_{exp}us_{temp}C_{index:03d}.tif` — parsed by `naming.py`, catalogued by `catalog.py`.
Columns: `filename, path, camera_id, filter_name, exposure_us, exposure_s, temperature_c, frame_index`.
`filter` and `index` were renamed to avoid conflicts with pandas built-in `.filter()` and `.index`.

## FrameGrabber API
Non-blocking state machine — call `grab_frame()` every loop tick. Returns `GrabStatus` (STARTED / WORKING / SUCCESS / TIMEOUT / FAILED). Calibration is lazy: scalar `0` = unloaded, `isinstance(arr, np.ndarray)` = loaded.

## DPC mask rationale
Dead pixel correction uses 2nd-neighbors (distance 2) to stay within the same Bayer color channel and avoid color contamination.

## Camera config (cameras.yaml)
Repo-root YAML with two sections: `models` (gain/offset keyed by SDK model string) and `cameras` (pattern/flip keyed by camera_id integer). Applied automatically in `ZwoAsiCamera.connect()` via `_apply_camera_config()`. Missing file is silently ignored.

**ASI1600MM Pro defaults:** gain=139, offset=21.

## zwoasi Python library API (corrected)
The `zwoasi` Camera object uses different method names than the underlying C SDK:
- `get_roi()` → `[start_x, start_y, width, height]` (NOT width/height/bins/type)
- `get_roi_format()` → `[width, height, bins, image_type]` (use this to get binning)
- `get_roi_start_position()` → `[start_x, start_y]`
- `set_roi_start_position(x, y)` (not `set_start_pos`)

## ZWO SDK startup ordering
The ZWO SDK resets the exposure control when binning is changed. Always set binning BEFORE setting exposure time, otherwise the first exposure runs at an unexpected duration, times out (5s default margin), and causes ~10s startup delay.

## opencv on macOS
`opencv-python-headless` is broken on macOS (references missing Linux `libxcb` libraries). Use `opencv-python` on macOS, `opencv-python-headless` on Linux/Pi. requirements.txt uses `sys_platform` markers. The SDL2 duplicate warning from both packages bundling SDL2 is harmless.

## live viewer (digital_eyepiece/main.py)
- `--gain` defaults to `None` — camera config value is used unless explicitly overridden
- `--bin` defaults to 2 — 2×2 binning reduces ASI1600 from 16MP to 4MP, eliminating lag
- Binning set before exposure time (SDK ordering requirement above)
- `grabber.pattern` set from `cam.info.bayer_pattern` (populated by camera config)
- `demosaic=grabber.pattern is not None` — auto-enables for color cameras

## Multi-camera support (digital_eyepiece/main.py)
Mirrors the MATLAB LiveViewer: all connected cameras are opened at startup into `cam_pool: dict[int, Camera]`.
The preferred/primary camera is opened via the `with cam_ctx as cam:` context manager; additional cameras
are opened with explicit `.connect()` and stored as `_extra_cams` for cleanup on exit.
Camera switching: `_switch_cam_config(name, camera_id)` calls `grabber.reset()`, runs `_apply_cam_config()`
on the target camera (gain/offset/pattern/ROI/meta), then swaps `grabber.cam`, clears cal cache, resets stacker.
Runtime code that needs the active camera uses `grabber.cam` (not the outer `cam` variable).
YAML camera name keys must be quoted strings (e.g. `"primary":`) so users can use names with spaces.
