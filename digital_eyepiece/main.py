"""
Digital eyepiece — live viewer.

Connects to the configured ZWO ASI camera, streams RAW16 frames, and
displays them in a pygame window with percentile stretching.

All hardware settings (gain, bin, ROI, flip, focal length, mount driver,
observer location) are read from configuration.yaml at the repo root.

Usage::

    python -m digital_eyepiece.main [--exposure SECONDS]

Mouse controls
--------------
Left-click          : toggle Stream / Stack mode (menu closed)
                      confirm menu selection (menu open)
Right-click         : open menu (menu closed) / close menu (menu open)
Scroll wheel        : zoom in/out
Middle-click        : slew mount to cursor (when mount connected + overlay visible)

Press Q or Escape to quit.
"""

import argparse
import importlib
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame

from astrocore.camera.catalog import scan_folder
from astrocore.camera.frame_grabber import FrameGrabber, GrabStatus
from astrocore.camera.virtual_cam import VirtualCamera
from astrocore.camera.zwo_asi import ZwoAsiCamera, list_cameras
from astrocore.config.camera_config import CameraConfig, Configuration, compute_hfov, load
from astrocore.display.skyoverlay import compute_overlay, load_catalog
from astrocore.mount.coord import radec_to_altaz
from astrocore.pipeline.stacker import ExposureSequence, Stacker
from astrocore.pipeline.streaming import StreamExposure
from digital_eyepiece.display import stretch_to_uint8, to_surface
from digital_eyepiece.input.dispatcher import InputDispatcher
from digital_eyepiece.input.menu import Menu, MenuItem
from digital_eyepiece.recorder import Recorder
from digital_eyepiece.view_state import FocusState, ViewMode, ViewState

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "astrocore" / "mount" / "skyChart.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STACKING_SEQUENCE = [
    0.1, 1, 2,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    10, 10, 10, 10, 10, 10,
    20,
]

WINDOW_W = 1280
WINDOW_H = 960

WINDOW_TITLE      = "Digital Eyepiece"
TARGET_FPS        = 60
CURSOR_HIDE_DELAY = 5.0
ALERT_DURATION    = 3.0

FOCUS_ROI_HALF = 100   # half-width of the 200×200 focus patch
_MENU_ROW_H    = 44    # pixel height of one menu row (shared by renderer and hover hit-test)

GREEN  = (0, 220, 0)
WHITE  = (220, 220, 220)
DIM    = (140, 140, 140)
RED    = (220, 60, 60)
AMBER  = (220, 180, 0)
BLACK  = (0, 0, 0)


# ---------------------------------------------------------------------------
# Menu construction
# ---------------------------------------------------------------------------

def _build_menu(
    state: ViewState,
    on_connect,
    on_disconnect,
    on_start_focus,
    telescope_configs: list[tuple[str, str]],   # [(config_name, description), …]
    on_select_config,                            # callable(config_name)
    active_desc_fn,                              # callable() → current description str
) -> Menu:
    brightness_submenu = [
        MenuItem("0.1×", action=lambda: setattr(state, "brightness", 0.1)),
        MenuItem("0.2×", action=lambda: setattr(state, "brightness", 0.2)),
        MenuItem("0.4×", action=lambda: setattr(state, "brightness", 0.4)),
        MenuItem("0.7×", action=lambda: setattr(state, "brightness", 0.7)),
        MenuItem("1×",   action=lambda: setattr(state, "brightness", 1.0)),
        MenuItem("2×",   action=lambda: setattr(state, "brightness", 2.0)),
        MenuItem("4×",   action=lambda: setattr(state, "brightness", 4.0)),
        MenuItem("7×",   action=lambda: setattr(state, "brightness", 7.0)),
        MenuItem("10×",  action=lambda: setattr(state, "brightness", 10.0)),
    ]

    def _mode_label() -> str:
        return "Start Streaming" if state.mode == ViewMode.ACCUMULATE else "Start Stacking"

    def _toggle_mode() -> None:
        state.mode = ViewMode.ACCUMULATE if state.mode == ViewMode.LIVE else ViewMode.LIVE

    # Telescope submenu — first item shows/selects the active telescope config
    if len(telescope_configs) > 1:
        selector_items = [
            MenuItem(desc, action=lambda cn=cn: on_select_config(cn))
            for cn, desc in telescope_configs
        ]
        selector_items.append(MenuItem("Back"))
        scope_item = MenuItem(active_desc_fn, submenu=selector_items)
    else:
        scope_item = MenuItem(active_desc_fn)   # informational only

    telescope_submenu = [
        scope_item,
        MenuItem("Connect",    action=on_connect),
        MenuItem("Disconnect", action=on_disconnect),
        MenuItem("Sync",       action=lambda: None),   # placeholder
        MenuItem("Park",       action=lambda: None),   # placeholder
        MenuItem("Back"),
    ]

    menu = Menu()
    menu.add(MenuItem("Cancel"))
    menu.add(MenuItem("Brightness", submenu=brightness_submenu))
    menu.add(MenuItem(_mode_label, action=_toggle_mode))
    menu.add(MenuItem("Focus", action=on_start_focus))
    menu.add(MenuItem("Telescope", submenu=telescope_submenu))
    menu.add(MenuItem("Quit", action=lambda: pygame.event.post(
        pygame.event.Event(pygame.QUIT)
    )))
    return menu


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _draw_cursor(surface: pygame.Surface, x: int, y: int) -> None:
    size = 10
    pygame.draw.line(surface, GREEN, (x - size, y), (x + size, y), 2)
    pygame.draw.line(surface, GREEN, (x, y - size), (x, y + size), 2)


def _render_menu(surface: pygame.Surface, menu: Menu) -> None:
    w, h = surface.get_size()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    font  = pygame.font.SysFont("monospace", 30)
    items = menu.current_items
    sel   = menu.selection_index
    row_h = _MENU_ROW_H
    total = len(items) * row_h
    y0    = (h - total) // 2

    for i, item in enumerate(items):
        color = GREEN if i == sel else WHITE
        label = font.render(item.label_text, True, color)
        x = w // 2 - label.get_width() // 2
        surface.blit(label, (x, y0 + i * row_h))


def _render_hud(surface: pygame.Surface, text: str) -> None:
    font  = pygame.font.SysFont("monospace", 18)
    label = font.render(text, True, GREEN)
    surface.blit(label, (8, 8))


def _render_focus_waiting(surface: pygame.Surface) -> None:
    font  = pygame.font.SysFont("monospace", 22)
    label = font.render("FOCUS — click to set point", True, GREEN)
    x = surface.get_width() // 2 - label.get_width() // 2
    surface.blit(label, (x, 8))


def _render_recording(surface: pygame.Surface) -> None:
    font  = pygame.font.SysFont("monospace", 22)
    label = font.render("● Recording", True, RED)
    x = surface.get_width() // 2 - label.get_width() // 2
    surface.blit(label, (x, 8))


def _render_top_right_badges(
    surface: pygame.Surface,
    badges: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """Render a stacked column of (text, color) badges anchored to the top-right corner."""
    font = pygame.font.SysFont("monospace", 22)
    y = 8
    for text, color in badges:
        label = font.render(text, True, color)
        x = surface.get_width() - label.get_width() - 8
        surface.blit(label, (x, y))
        y += label.get_height() + 4


def _render_alert(surface: pygame.Surface, text: str) -> None:
    font  = pygame.font.SysFont("monospace", 28)
    label = font.render(text, True, RED)
    w, h  = surface.get_size()
    x = w // 2 - label.get_width() // 2
    y = h // 2 - label.get_height() // 2
    bg = pygame.Surface((label.get_width() + 20, label.get_height() + 10))
    bg.fill(BLACK)
    surface.blit(bg, (x - 10, y - 5))
    surface.blit(label, (x, y))


def _render_hover_label(
    surface: pygame.Surface,
    table: list[dict],
    cursor_x: int,
    cursor_y: int,
    threshold: int = 25,
) -> None:
    if not table:
        return
    best = None
    best_d2 = threshold * threshold
    for entry in table:
        d2 = (entry["px"] - cursor_x) ** 2 + (entry["py"] - cursor_y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = entry
    if best is None:
        return

    name = best["name"] or best["type"]
    text = f"{name}  mag {best['mag']:.1f}  {best['type']}"
    font  = pygame.font.SysFont("monospace", 16)
    label = font.render(text, True, WHITE)

    x = cursor_x + 16
    y = cursor_y - 10
    w, h = surface.get_size()
    x = min(x, w - label.get_width() - 10)
    y = max(y, 4)

    bg = pygame.Surface((label.get_width() + 8, label.get_height() + 4), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 190))
    surface.blit(bg, (x - 4, y - 2))
    surface.blit(label, (x, y))


def _apply_brightness(image: np.ndarray, brightness: float) -> np.ndarray:
    if brightness == 1.0:
        return image
    return np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exposure", type=float, default=None,
                   help="Override streaming start exposure in seconds (default: auto)")
    p.add_argument("--vcam", metavar="SUBFOLDER",
                   help="Replay a recorded session instead of using live hardware. "
                        "SUBFOLDER is a timestamped folder under record_path in configuration.yaml.")
    p.add_argument("--vcam-accel", type=float, default=1.0, metavar="FACTOR",
                   help="Time-acceleration for vcam playback (default: 1.0 = real-time). "
                        "E.g. 10 plays a 5-second exposure back in 0.5 s.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # -- load configuration --------------------------------------------------
    try:
        config = load()
    except FileNotFoundError as exc:
        logging.warning("%s — using defaults", exc)
        config = None

    lat = config.latitude  if config else 38.44
    lon = config.longitude if config else -122.71

    # -- camera setup --------------------------------------------------------
    if args.vcam:
        if config is None or config.record_path is None:
            logging.error("--vcam requires record_path to be set in configuration.yaml")
            sys.exit(1)
        session_path = config.record_path / args.vcam
        if not session_path.is_dir():
            logging.error("Session folder not found: %s", session_path)
            sys.exit(1)
        # Peek at the catalog to learn camera_id, then resolve bayer_pattern from config
        _vcam_table = scan_folder(session_path)
        if len(_vcam_table) == 0:
            logging.error("No canonical .tif frames found in %s", session_path)
            sys.exit(1)
        _vcam_id = int(_vcam_table.camera_id.iloc[0])
        _vcam_cfg = config.get_config(_vcam_id) if config else None
        cam_ctx: VirtualCamera | ZwoAsiCamera = VirtualCamera(
            session_path,
            camera_id   = _vcam_id,
            bayer_pattern = _vcam_cfg.pattern if _vcam_cfg else None,
            t_accel     = args.vcam_accel,
        )
    else:
        cameras = list_cameras()
        if not cameras:
            logging.warning("No ZWO ASI cameras found. Is the camera plugged in?")
            sys.exit(1)
        preferred_id = config.preferred_camera_id if config else None
        usb_index = cameras[0]["usb_index"]
        for c in cameras:
            if c.get("camera_id") == preferred_id:
                usb_index = c["usb_index"]
                break
        cam_ctx = ZwoAsiCamera(index=usb_index)

    # -- catalog -------------------------------------------------------------
    catalog = load_catalog(str(_CATALOG_PATH))
    logging.info("Loaded %d catalog entries", len(catalog))

    # -- pygame init ---------------------------------------------------------
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(WINDOW_TITLE)
    pygame.mouse.set_visible(True)
    clock = pygame.time.Clock()

    state = ViewState()

    # Mutable refs so menu callbacks can update the active config
    cam_config_ref: list[CameraConfig] = [CameraConfig()]
    fov_ref:        list[float]        = [30.0]

    # Mount holder
    mount_holder: list = [None]

    def _connect_mount() -> None:
        driver = cam_config_ref[0].mount_driver
        if not driver:
            logging.warning("No mount_driver set in configuration.yaml for this config")
            return
        try:
            mod = importlib.import_module(f"astrocore.mount.{driver}")
            cls = getattr(mod, "Driver", None)
            if cls is None:
                logging.warning("Mount module %r has no Driver attribute", driver)
                return
            mount_holder[0] = cls()
            state.mount_connected = True
            logging.info("Mount connected via %s", driver)
        except Exception as exc:
            logging.warning("Mount connect failed: %s", exc)

    def _disconnect_mount() -> None:
        if mount_holder[0] is not None:
            mount_holder[0].disconnect()
            mount_holder[0] = None
        state.mount_connected = False

    with cam_ctx as cam:

        # Warn if the connected camera has no entry in configuration.yaml.
        # camera_id == 0 means the EEPROM was never written (run set_camera_id.py).
        _cam_configured = config is not None and cam.info.camera_id in config.camera_ids()
        if not _cam_configured:
            id_note = " (EEPROM not set)" if cam.info.camera_id == 0 else ""
            logging.warning(
                "Camera ID %d%s not found in configuration.yaml — running with defaults. "
                "Run set_camera_id.py to register this camera.",
                cam.info.camera_id, id_note,
            )

        # Apply configuration to camera
        if config is not None:
            cam_config = config.get_config(cam.info.camera_id)
            cam_config_ref[0] = cam_config

            if cam_config.bin > 1:
                cam.bin = cam_config.bin
            if cam_config.gain is not None:
                cam.gain = cam_config.gain
            if cam_config.offset is not None:
                cam.offset = cam_config.offset
            if cam_config.pattern is not None:
                from dataclasses import replace as _dc_replace
                cam._info = _dc_replace(cam.info, bayer_pattern=cam_config.pattern)

            # Stamp telescope info onto all subsequent frames
            cam.meta.telescope_description = cam_config.telescope_description
            cam.meta.focal_length_mm       = cam_config.focal_length_mm
            cam.meta.Lat = lat
            cam.meta.Lon = lon

            # Compute FOV from focal length + hardware pixel size
            roi_w = cam_config.effective_roi(
                cam.info.sensor_width_px, cam.info.sensor_height_px
            )[2]
            fov_ref[0] = compute_hfov(
                cam_config.focal_length_mm,
                cam.info.pixel_size_um,
                roi_w,
            ) or fov_ref[0]

            if config.cal_path is not None:
                pass  # cal_path applied to grabber below

        grabber = FrameGrabber(cam)
        grabber.pattern = cam.info.bayer_pattern
        if config is not None and config.cal_path is not None:
            grabber.cal_path = config.cal_path

        recorder: Recorder | None = (
            None if args.vcam
            else Recorder(config.record_path) if config is not None and config.record_path is not None
            else None
        )

        # Telescope config switching from menu
        def _on_select_config(config_name: str) -> None:
            if config is None:
                return
            new_cfg = config.get_config(cam.info.camera_id, config_name)
            cam_config_ref[0] = new_cfg
            roi_w = new_cfg.effective_roi(
                cam.info.sensor_width_px, cam.info.sensor_height_px
            )[2]
            fov_ref[0] = compute_hfov(
                new_cfg.focal_length_mm, cam.info.pixel_size_um, roi_w
            ) or fov_ref[0]
            cam.meta.telescope_description = new_cfg.telescope_description
            cam.meta.focal_length_mm       = new_cfg.focal_length_mm
            if new_cfg.gain is not None:
                cam.gain = new_cfg.gain
            logging.info("Switched to %s (focal length %.0f mm)",
                         new_cfg.telescope_description, new_cfg.focal_length_mm)

        telescope_configs = (
            config.telescope_descriptions(cam.info.camera_id) if config else []
        )

        def _start_focus_mode() -> None:
            if state.recording and recorder is not None:
                state.recording = False
                recorder.stop()
            state.focus_state = FocusState.WAITING
            state.mode = ViewMode.LIVE

        menu = _build_menu(
            state,
            on_connect      = _connect_mount,
            on_disconnect   = _disconnect_mount,
            on_start_focus  = _start_focus_mode,
            telescope_configs = telescope_configs,
            on_select_config  = _on_select_config,
            active_desc_fn    = lambda: cam_config_ref[0].telescope_description or "Telescope",
        )
        dispatcher = InputDispatcher(state, menu, zoom_step=1.2, zoom_min=1.0, zoom_max=8.0)

        actual_gain = cam.gain
        ae        = StreamExposure(start=args.exposure if args.exposure is not None else 0.1)
        stacker   = Stacker()
        stack_seq = ExposureSequence(STACKING_SEQUENCE)

        # Background frame-grab thread
        _frame_queue: queue.Queue = queue.Queue(maxsize=2)
        _exposure_ref = [int(ae.current * 1_000_000)]
        _focus_hardware_roi: bool = False   # True when ZWO hardware ROI is active for focus

        def _make_grab_worker(stop_event: threading.Event) -> threading.Thread:
            """Return a new grab thread bound to *stop_event*.

            Each call creates a fresh Event that is never cleared, so a thread that
            outlives its intended lifetime (join timeout) remains stopped — it holds
            a reference to its own (already-set) event and cannot be un-stopped by
            a subsequent _restart_grab() call.
            """
            def _worker() -> None:
                configured_us: int | None = None   # exposure last confirmed on camera
                while not stop_event.is_set():
                    exp_us = _exposure_ref[0]
                    # Only pass exposure_us when it differs from what is already
                    # configured.  Passing None skips the get_control_value SDK
                    # call that would otherwise happen on every WORKING iteration.
                    result = grabber.grab_frame(
                        exposure_us  = exp_us if exp_us != configured_us else None,
                        dark         = not _focus_hardware_roi and grabber.cal_path is not None,
                        flat         = not _focus_hardware_roi and grabber.cal_path is not None,
                        dpc          = False,
                        demosaic     = grabber.pattern is not None,
                    )
                    if result.status == GrabStatus.WORKING:
                        time.sleep(0.001)   # yield GIL; don't flood SDK with status polls
                        continue
                    if result.status == GrabStatus.STARTED:
                        configured_us = exp_us
                        continue
                    # SUCCESS, TIMEOUT, or FAILED — enqueue for the main loop
                    configured_us = exp_us
                    if (result.status == GrabStatus.SUCCESS
                            and recorder is not None
                            and result.raw_frame is not None):
                        recorder.save(result.raw_frame)
                    try:
                        _frame_queue.put_nowait(result)
                    except queue.Full:
                        pass
            return threading.Thread(target=_worker, daemon=True)

        _stop_grab = threading.Event()
        _grab_thread = _make_grab_worker(_stop_grab)
        _grab_thread.start()

        _focus_roi_saved: tuple[int, int, int, int] | None = None

        def _stop_and_reset_grab() -> None:
            """Signal the current grab thread to stop and wait for it to exit.

            The stop event is NOT cleared after join so that a thread which outlives
            the timeout remains permanently stopped (it holds a reference to its own
            set event).
            """
            _stop_grab.set()
            _grab_thread.join(timeout=5.0)
            grabber.reset()

        def _restart_grab() -> None:
            nonlocal _stop_grab, _grab_thread
            _stop_grab = threading.Event()          # new event; old thread keeps its (set) one
            _grab_thread = _make_grab_worker(_stop_grab)
            _grab_thread.start()

        def _enter_focus_active() -> None:
            nonlocal _focus_roi_saved, _focus_hardware_roi
            if not isinstance(cam, ZwoAsiCamera):
                return   # VirtualCamera: fall back to software crop
            _stop_and_reset_grab()
            saved = cam.get_roi()
            _focus_roi_saved = saved
            roi_x, roi_y, roi_w, roi_h = saved
            sensor_cx = roi_x + int(state.focus_center_x * roi_w)
            sensor_cy = roi_y + int(state.focus_center_y * roi_h)
            sensor_w  = cam.info.sensor_width_px
            sensor_h  = cam.info.sensor_height_px
            fx = max(0, min(sensor_w - 200, (sensor_cx - 100) & ~1))
            fy = max(0, min(sensor_h - 200, (sensor_cy - 100) & ~1))
            cam.set_roi(x=fx, y=fy, width=200, height=200)
            _focus_hardware_roi = True
            _restart_grab()

        def _exit_focus() -> None:
            nonlocal _focus_roi_saved, _focus_hardware_roi
            if _focus_hardware_roi and _focus_roi_saved is not None and isinstance(cam, ZwoAsiCamera):
                _stop_and_reset_grab()
                x, y, w, h = _focus_roi_saved
                cam.set_roi(x=x, y=y, width=w, height=h)
                _focus_hardware_roi = False
                _focus_roi_saved = None
                _restart_grab()
            state.focus_state = FocusState.OFF
            state.mode = ViewMode.LIVE

        last_surface: pygame.Surface | None = None
        frame_count  = 0
        fps_display  = 0.0
        t_last_frame = time.monotonic()
        prev_mode    = state.mode
        _cal_ok      = True   # updated from grab results; False → show Uncalibrated notice

        t_last_move = time.monotonic()
        cursor_pos  = (WINDOW_W // 2, WINDOW_H // 2)

        alert_timer = 0.0
        ov_table: list[dict] = []

        running = True
        while running:
            dt = clock.tick(TARGET_FPS) / 1000.0
            dispatcher.update(dt)
            if alert_timer > 0:
                alert_timer -= dt

            # -- events -------------------------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif state.focus_state != FocusState.OFF:
                        _exit_focus()
                    elif event.key == pygame.K_f:
                        if state.recording and recorder is not None:
                            state.recording = False
                            recorder.stop()
                        state.focus_state = FocusState.WAITING
                        state.mode = ViewMode.LIVE
                    elif event.key == pygame.K_r and recorder is not None:
                        if state.recording:
                            state.recording = False
                            recorder.stop()
                        else:
                            state.recording = True
                            recorder.start()

                elif event.type == pygame.MOUSEMOTION:
                    cursor_pos  = event.pos
                    t_last_move = time.monotonic()
                    dispatcher.on_mouse_move(*event.pos)
                    if state.menu_open:
                        items = menu.current_items
                        if items:
                            total = len(items) * _MENU_ROW_H
                            y0 = (WINDOW_H - total) // 2
                            cy = event.pos[1]
                            if y0 <= cy < y0 + total:
                                menu.set_selection((cy - y0) // _MENU_ROW_H)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if state.focus_state == FocusState.WAITING:
                        state.focus_center_x = event.pos[0] / WINDOW_W
                        state.focus_center_y = event.pos[1] / WINDOW_H
                        _enter_focus_active()
                        state.focus_state = FocusState.ACTIVE
                    elif state.focus_state == FocusState.ACTIVE:
                        _exit_focus()
                    else:
                        if event.button == 1:
                            dispatcher.on_left_click()
                        elif event.button == 2:
                            if dispatcher.on_middle_click():
                                alert_timer = ALERT_DURATION
                        elif event.button == 3:
                            dispatcher.on_right_click()

                elif event.type == pygame.MOUSEWHEEL:
                    if state.focus_state == FocusState.OFF:
                        dispatcher.on_scroll(event.y)

            # -- mode change --------------------------------------------------
            if state.mode != prev_mode:
                if state.mode == ViewMode.LIVE:
                    ae.reset()
                    if state.recording and recorder is not None:
                        state.recording = False
                        recorder.stop()
                elif state.mode == ViewMode.ACCUMULATE:
                    stacker.reset()
                    stack_seq.reset()
                prev_mode = state.mode

            # -- grab frame ---------------------------------------------------
            _exposure_ref[0] = (
                int(ae.current * 1_000_000)
                if state.mode == ViewMode.LIVE
                else int(stack_seq.current * 1_000_000)
            )
            try:
                result = _frame_queue.get_nowait()
            except queue.Empty:
                result = None

            if result is not None and result.status == GrabStatus.SUCCESS:
                now = time.monotonic()
                fps_display  = 1.0 / max(now - t_last_frame, 1e-6)
                t_last_frame = now
                frame_count += 1
                if state.focus_state == FocusState.OFF:
                    _cal_ok = result.calibrated

                if state.focus_state == FocusState.ACTIVE:
                    if _focus_hardware_roi:
                        # Camera is delivering the 200×200 ROI directly
                        data8 = stretch_to_uint8(result.frame.data)
                    else:
                        # VirtualCamera fallback: software crop from full frame
                        h, w = result.frame.data.shape[:2]
                        cx = int(state.focus_center_x * w)
                        cy = int(state.focus_center_y * h)
                        x1 = max(0, cx - FOCUS_ROI_HALF)
                        y1 = max(0, cy - FOCUS_ROI_HALF)
                        x2 = min(w, cx + FOCUS_ROI_HALF)
                        y2 = min(h, cy + FOCUS_ROI_HALF)
                        data8 = stretch_to_uint8(result.frame.data[y1:y2, x1:x2])
                    last_surface = pygame.transform.smoothscale(
                        to_surface(data8), (WINDOW_W, WINDOW_H)
                    )
                elif state.mode == ViewMode.LIVE:
                    ae.update(result.frame.data)
                    data8 = stretch_to_uint8(result.frame.data)
                    data8 = _apply_brightness(data8, state.brightness)
                    last_surface = pygame.transform.smoothscale(
                        to_surface(data8), (WINDOW_W, WINDOW_H)
                    )
                else:
                    if stacker.add_frame(result.frame.data, stack_seq.current):
                        stack_seq.advance()

            if state.mode == ViewMode.ACCUMULATE:
                display8 = stacker.get_display_frame()
                if display8 is not None:
                    display8 = _apply_brightness(display8, state.brightness)
                    img_surface  = to_surface(display8)
                    last_surface = pygame.transform.smoothscale(
                        img_surface, (WINDOW_W, WINDOW_H)
                    )

            # -- render -------------------------------------------------------
            if last_surface is not None:
                screen.blit(last_surface, (0, 0))
                if state.all_sky_mode:
                    dark = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
                    dark.fill((0, 0, 0, 180))
                    screen.blit(dark, (0, 0))
            else:
                screen.fill(BLACK)
                font  = pygame.font.SysFont("monospace", 24)
                label = font.render("Waiting for first frame...", True, DIM)
                screen.blit(label, (WINDOW_W // 2 - 150, WINDOW_H // 2))

            # Sky overlay
            if state.overlay_active and mount_holder[0] is not None:
                try:
                    ra_h, dec_deg = mount_holder[0].position
                    alt_deg, az_deg = radec_to_altaz(ra_h, dec_deg, lat, lon)
                    ov_arr, ov_table = compute_overlay(
                        catalog,
                        fov_deg     = fov_ref[0],
                        alt_deg     = alt_deg,
                        az_deg      = az_deg,
                        image_shape = (WINDOW_H, WINDOW_W),
                        lat_deg     = lat,
                        lon_deg     = lon,
                    )
                    ov_surf = pygame.image.frombuffer(
                        ov_arr.tobytes(), (WINDOW_W, WINDOW_H), "RGBA"
                    )
                    screen.blit(ov_surf, (0, 0))
                except Exception as exc:
                    logging.warning("Overlay error: %s", exc)

                _render_hover_label(screen, ov_table, *cursor_pos)

            # HUD
            if state.focus_state == FocusState.ACTIVE:
                hud = (
                    f"FOCUS  exp={ae.current:.4g}s  "
                    f"gain={actual_gain}  fps={fps_display:.1f}  "
                    f"— click or any key to exit"
                )
            elif state.mode == ViewMode.ACCUMULATE:
                hud = (
                    f"STACK  exp={stack_seq.current:.4g}s  "
                    f"gain={actual_gain}  "
                    f"frames={stacker.frame_count}  t={stacker.t_accum:.1f}s  "
                    f"{cam_config_ref[0].telescope_description}"
                )
            else:
                hud = (
                    f"LIVE  exp={ae.current:.4g}s  "
                    f"gain={actual_gain}  "
                    f"fps={fps_display:.1f}  frames={frame_count}  "
                    f"{cam_config_ref[0].telescope_description}"
                )
            _render_hud(screen, hud)
            if state.focus_state == FocusState.WAITING:
                _render_focus_waiting(screen)
                _draw_cursor(screen, *cursor_pos)
            if state.recording:
                _render_recording(screen)
            _top_right: list[tuple[str, tuple[int, int, int]]] = []
            if not _cam_configured:
                _top_right.append(("Unconfigured", AMBER))
            if not _cal_ok and state.focus_state == FocusState.OFF:
                _top_right.append(("Uncalibrated", AMBER))
            if _top_right:
                _render_top_right_badges(screen, _top_right)

            if state.menu_open:
                _render_menu(screen, menu)

            if alert_timer > 0:
                _render_alert(screen, "Caution: Mount is moving")

            cursor_visible = (
                state.menu_open
                or (
                    state.focus_state == FocusState.OFF
                    and (time.monotonic() - t_last_move) < CURSOR_HIDE_DELAY
                )
            )
            pygame.mouse.set_visible(cursor_visible)

            pygame.display.flip()

        _stop_grab.set()
        _grab_thread.join(timeout=5.0)

    pygame.quit()


if __name__ == "__main__":
    main()
