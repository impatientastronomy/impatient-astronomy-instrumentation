"""
Digital eyepiece — live viewer.

Connects to the configured ZWO ASI camera, streams RAW16 frames, and
displays them in a pygame window with live stacking and sky overlay.

Layout
------
  ┌─────┬──────── Top status bar ────────┬─────┐
  │     │  ★ mode · exp · gain · fps     │  ⚙  │
  ├─────┴────────────────────────────────┴─────┤
  │                                            │
  │              Central region                │
  │           (image displayed here)           │
  │                                            │
  ├─────┬──────── Bottom status bar ─────┬─────┤
  │     │  ☰ frames · t_accum · scope   │     │
  └─────┴────────────────────────────────┴─────┘

Icons open menus:  ★ → Action   ⚙ → Utilities   ☰ → Controls

Mouse
-----
Left-click  : over open menu item → select; off menu → cancel; no menu → Stack/Stream
Right-click : context menu (Slew here / Sync here / SkyMap / Focus / Exit SkyMap)
Right-hold  : pan image
Scroll      : zoom
Hover       : show object name when star overlay is active

Press Q or Escape to quit.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import queue
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import replace as _dc_replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import numpy as np
import pygame

import cv2

from astrocore.camera.catalog import scan_folder
from astrocore.camera.frame_grabber import FrameGrabber, GrabStatus
from astrocore.camera.virtual_cam import NullCamera, VirtualCamera
from astrocore.camera.zwo_asi import ZwoAsiCamera, list_cameras
from astrocore.config.camera_config import CameraConfig, Configuration, HotspotConfig, compute_hfov, load
from digital_eyepiece.gallery_server import GalleryServer
from astrocore.display.overlay_style import load_overlay_style
from astrocore.display.skyoverlay import compute_overlay, load_catalog, load_constellation_lines
from astrocore.display.moon_mapper import (
    MOON_ANGULAR_RADIUS_DEG, compute_moon_overlay, load_moon_catalog,
)
from astrocore.mount.coord import altaz_to_radec, radec_to_altaz
from astrocore.pipeline.stacker import ConstellationStacker, ExposureSequence
from astrocore.pipeline.streaming import StreamExposure
from digital_eyepiece.display import stretch_to_uint8, to_surface
from digital_eyepiece.input.dispatcher import InputDispatcher
from digital_eyepiece.input.menu import Menu, MenuItem
from digital_eyepiece.recorder import Recorder
from digital_eyepiece.view_state import FocusState, ViewMode, ViewState

_DATA_ROOT               = Path.home() / Path(__file__).parent.name   # ~/digital_eyepiece (cals/sessions/images)
_DEFAULT_CONFIG          = Path(__file__).parent / "config" / "configuration.yaml"  # repo-local, gitignored
_CATALOG_PATH            = Path(__file__).resolve().parent.parent / "astrocore" / "mount" / "skychart.csv"
_MOON_CATALOG_PATH       = Path(__file__).resolve().parent.parent / "astrocore" / "mount" / "moon_features.csv"
_OVERLAY_STYLE_PATH      = Path(__file__).resolve().parent.parent / "overlay_style.yaml"
_CONSTELLATION_PATH      = Path(__file__).resolve().parent.parent / "astrocore" / "mount" / "constellation_lines.csv"

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

FOCUS_ROI_HALF = 200

# Color palette — black background, dim grey UI elements
BLACK  = (0, 0, 0)
DIM    = (100, 100, 100)    # borders, inactive icons
GREY   = (160, 160, 160)    # text and icons
WHITE  = (210, 210, 210)    # highlighted text
GREEN  = (0, 200, 0)        # selected menu item
AMBER  = (200, 160, 0)      # warnings
RED    = (200, 50, 50)      # recording indicator / alerts

# Controls menu values
_EXPOSURE_STEPS: list[tuple[str, float | None]] = [
    ("Auto",   None),
    ("0.1ms",  0.0001), ("0.2ms", 0.0002), ("0.5ms", 0.0005),
    ("1ms",    0.001),  ("2ms",   0.002),  ("5ms",   0.005),
    ("10ms",   0.01),   ("20ms",  0.02),   ("50ms",  0.05),
    ("0.1s",   0.1),    ("0.2s",  0.2),    ("0.5s",  0.5),
    ("1s",     1.0),    ("2s",    2.0),    ("5s",    5.0),
    ("10s",    10.0),   ("20s",   20.0),
]
_BRIGHTNESS_STEPS: list[tuple[str, float]] = [
    ("0.1×", 0.1), ("0.2×", 0.2), ("0.3×", 0.3),
    ("0.5×", 0.5), ("0.7×", 0.7), ("1×",   1.0),
    ("1.2×", 1.2), ("1.5×", 1.5), ("2×",   2.0),
]
_SKY_STEPS: list[tuple[str, float]] = [
    ("0×",   0.0), ("0.1×", 0.1), ("0.2×", 0.2), ("0.3×", 0.3),
    ("0.5×", 0.5), ("0.7×", 0.7), ("1×",   1.0), ("1.2×", 1.2),
    ("1.5×", 1.5), ("2×",   2.0),
]

# ---------------------------------------------------------------------------
# Layout geometry
# ---------------------------------------------------------------------------

def _make_layout(w: int, h: int) -> dict:
    """
    Compute all screen regions from window dimensions.

    Status bars are Hd/20 tall and min(Hd,Wd) wide, centered horizontally.
    Corner squares fill the leftover space at each corner.
    """
    sh = h // 20
    sw = min(h, w)
    sx = (w - sw) // 2   # left edge of status bars
    cw = sx               # corner width (0 when w <= h)
    return dict(
        status_h  = sh,
        status_w  = sw,
        status_x  = sx,
        corner_w  = cw,
        top_bar   = pygame.Rect(sx, 0,      sw, sh),
        bot_bar   = pygame.Rect(sx, h - sh, sw, sh),
        central   = pygame.Rect(0,  sh,     w,  h - 2 * sh),
        corner_tl = pygame.Rect(0,      0,      cw, sh),
        corner_tr = pygame.Rect(w - cw, 0,      cw, sh),
        corner_bl = pygame.Rect(0,      h - sh, cw, sh),
        corner_br = pygame.Rect(w - cw, h - sh, cw, sh),
        # Icon rects — square tiles at the ends of the status bars
        icon_star    = pygame.Rect(sx,            0,      sh, sh),
        icon_gear    = pygame.Rect(sx + sw - sh,  0,      sh, sh),
        icon_sliders = pygame.Rect(sx,            h - sh, sh, sh),
    )


def _image_rect(cam_w: int, cam_h: int, central: pygame.Rect) -> pygame.Rect:
    """Scale the camera image to fit inside the central region without clipping."""
    cam_aspect     = cam_w / cam_h
    central_aspect = central.width / central.height
    if cam_aspect >= central_aspect:
        iw = central.width
        ih = int(iw / cam_aspect)
    else:
        ih = central.height
        iw = int(ih * cam_aspect)
    ix = central.x + (central.width  - iw) // 2
    iy = central.y + (central.height - ih) // 2
    return pygame.Rect(ix, iy, iw, ih)


def _apply_zoom_pan(surface: pygame.Surface, state: ViewState) -> pygame.Surface:
    """
    Return a new surface showing the zoomed and panned crop of `surface`.
    When zoom_level is 1.0 the original surface is returned unchanged.
    """
    zoom = state.zoom_level
    if zoom <= 1.0:
        return surface
    W, H = surface.get_size()
    crop_w = max(1, int(W / zoom))
    crop_h = max(1, int(H / zoom))
    cx = state.zoom_center_x * W
    cy = state.zoom_center_y * H
    x = max(0, min(W - crop_w, int(cx - crop_w / 2)))
    y = max(0, min(H - crop_h, int(cy - crop_h / 2)))
    cropped = surface.subsurface(pygame.Rect(x, y, crop_w, crop_h))
    return pygame.transform.smoothscale(cropped, (W, H))


def _screen_to_sensor_norm(
    mx: int, my: int,
    img_rect: pygame.Rect,
    state: ViewState,
) -> tuple[float, float]:
    """Map a screen pixel to sensor-normalised coords [0, 1], accounting for zoom/pan."""
    nx = (mx - img_rect.left) / img_rect.width
    ny = (my - img_rect.top)  / img_rect.height
    zoom = state.zoom_level
    half = 0.5 / zoom
    cx = max(half, min(1.0 - half, state.zoom_center_x))
    cy = max(half, min(1.0 - half, state.zoom_center_y))
    sx = cx + (nx - 0.5) / zoom
    sy = cy + (ny - 0.5) / zoom
    return max(0.0, min(1.0, sx)), max(0.0, min(1.0, sy))


# ---------------------------------------------------------------------------
# Menu builders
# ---------------------------------------------------------------------------

def _build_action_menu(
    state: ViewState,
    on_connect,
    on_disconnect,
    recorder,
    on_save,
    on_quit,
    mount_driver: str = "",
) -> Menu:
    def _mode_label() -> str:
        return "Start Streaming" if state.mode == ViewMode.ACCUMULATE else "Start Stacking"

    def _toggle_mode() -> None:
        state.mode   = ViewMode.ACCUMULATE if state.mode == ViewMode.LIVE else ViewMode.LIVE
        state.paused = False

    def _pause_label() -> str:
        return "Resume" if state.paused else "Pause"

    def _toggle_pause() -> None:
        state.paused = not state.paused

    def _toggle_record() -> None:
        if recorder is None:
            return
        if state.recording:
            state.recording = False
            recorder.stop()
        else:
            state.recording = True
            recorder.start()

    def _record_label() -> str:
        return "Stop Recording" if state.recording else "Record"

    def _connect_label() -> str:
        return "Disconnect" if state.mount_connected else "Connect"

    def _toggle_connect() -> None:
        if state.mount_connected:
            on_disconnect()
        else:
            on_connect()

    mount_submenu = [
        MenuItem(mount_driver or "no driver configured"),
        MenuItem(_connect_label, action=_toggle_connect),
        MenuItem("Park",         action=lambda: None),
        MenuItem("Back"),
    ]

    m = Menu()
    m.add(MenuItem(_mode_label,   action=_toggle_mode))
    m.add(MenuItem(_pause_label,  action=_toggle_pause))
    m.add(MenuItem("Mount",       submenu=mount_submenu))
    m.add(MenuItem(_record_label, action=_toggle_record))
    m.add(MenuItem("Save",        action=on_save))
    m.add(MenuItem("Quit",        action=on_quit))
    return m


def _build_utilities_menu(cam, cam_config_ref: list, on_set_dpc,
                          on_clear_images=None, on_moon_mode=None,
                          moon_mode_label_fn=None) -> Menu:
    bin_items = [
        MenuItem("1×", action=lambda: None),   # TODO: wire to cam.bin
        MenuItem("2×", action=lambda: None),
        MenuItem("Back"),
    ]
    temp_items = [
        MenuItem(f"{t}°C", action=lambda t=t: None)  # TODO: wire to cam TEC
        for t in (0, 5, 10, 15, 20)
    ]
    temp_items.append(MenuItem("Back"))

    camera_submenu = [
        MenuItem("Bin",         submenu=bin_items),
        MenuItem("Temperature", submenu=temp_items),
        MenuItem("Set DPC",     action=on_set_dpc),
        MenuItem("Back"),
    ]

    m = Menu()
    m.add(MenuItem("Camera", submenu=camera_submenu))
    if on_moon_mode is not None:
        label_fn = moon_mode_label_fn or (lambda: "Moon Map")
        m.add(MenuItem(label_fn, action=on_moon_mode))
    if on_clear_images is not None:
        m.add(MenuItem("Clear Images", action=on_clear_images))
    return m


def _build_context_menu(
    near_object: bool,
    object_name: str,
    on_focus,
    on_slew,
    on_sync,
    on_sky_map,
    mount_connected: bool,
    in_sky_map: bool = False,
    on_exit_sky_map=None,
    cam_select_items: list | None = None,
) -> Menu:
    m = Menu()
    if near_object:
        m.add(MenuItem(f"About {object_name}", action=lambda: None))
    if not in_sky_map:
        m.add(MenuItem("Focus here", action=on_focus))
    if mount_connected:
        m.add(MenuItem("Slew here", action=on_slew))
    if near_object:
        m.add(MenuItem("Sync here", action=on_sync if mount_connected else None))
    if not in_sky_map and cam_select_items is not None:
        submenu = list(cam_select_items)
        m.add(MenuItem("Cam Select", submenu=submenu))
    if in_sky_map:
        m.add(MenuItem("Exit SkyMap", action=on_exit_sky_map))
    return m


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_FONT_CACHE: dict[int, pygame.font.Font] = {}


def _font(size: int) -> pygame.font.Font:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.SysFont("monospace", size)
    return _FONT_CACHE[size]


def _draw_icon_star(surface: pygame.Surface, rect: pygame.Rect, color: tuple) -> None:
    """5-pointed star."""
    cx, cy  = rect.centerx, rect.centery
    r_out   = rect.height * 0.36
    r_in    = rect.height * 0.15
    pts = []
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pygame.draw.polygon(surface, color, pts)


def _draw_icon_gear(surface: pygame.Surface, rect: pygame.Rect, color: tuple) -> None:
    """Gear with 8 teeth."""
    cx, cy  = rect.centerx, rect.centery
    r_body  = int(rect.height * 0.28)
    r_tooth = int(rect.height * 0.38)
    r_hole  = int(rect.height * 0.12)
    pygame.draw.circle(surface, color, (cx, cy), r_body)
    for i in range(8):
        angle = math.radians(i * 45)
        x1 = cx + r_body  * math.cos(angle)
        y1 = cy + r_body  * math.sin(angle)
        x2 = cx + r_tooth * math.cos(angle)
        y2 = cy + r_tooth * math.sin(angle)
        pygame.draw.line(surface, color, (int(x1), int(y1)), (int(x2), int(y2)), max(3, rect.height // 14))
    pygame.draw.circle(surface, BLACK, (cx, cy), r_hole)


def _draw_icon_sliders(surface: pygame.Surface, rect: pygame.Rect, color: tuple) -> None:
    """Three horizontal lines with offset slider handles."""
    cx, cy   = rect.centerx, rect.centery
    hw       = int(rect.width  * 0.30)   # half-line width
    spacing  = int(rect.height * 0.22)
    handles  = [0.4, -0.1, 0.2]          # handle offset from center, normalized to hw
    r_handle = max(3, rect.height // 12)
    for i, ho in enumerate(handles):
        y  = cy + (i - 1) * spacing
        pygame.draw.line(surface, color, (cx - hw, y), (cx + hw, y), 2)
        hx = cx + int(ho * hw)
        pygame.draw.circle(surface, color, (hx, y), r_handle)
        pygame.draw.circle(surface, BLACK, (hx, y), r_handle - 2)


def _render_status_bars(
    surface: pygame.Surface,
    layout: dict,
    state: ViewState,
    hud_top: str,
    hud_bot: str,
    cam_configured: bool,
    cal_ok: bool,
    action_active: bool,
    controls_active: bool,
    utilities_active: bool,
) -> None:
    """Draw both status bars and their icons onto surface."""
    top = layout["top_bar"]
    bot = layout["bot_bar"]

    # Bars and corners: black background, dim border
    for r in (top, bot,
              layout["corner_tl"], layout["corner_tr"],
              layout["corner_bl"], layout["corner_br"]):
        pygame.draw.rect(surface, BLACK, r)
        pygame.draw.rect(surface, DIM,   r, 1)

    # Icons — bright when their menu is active
    star_col    = WHITE if action_active   else GREY
    sliders_col = WHITE if controls_active else GREY
    gear_col    = WHITE if utilities_active else GREY

    _draw_icon_star   (surface, layout["icon_star"],    star_col)
    _draw_icon_sliders(surface, layout["icon_sliders"], sliders_col)
    _draw_icon_gear   (surface, layout["icon_gear"],    gear_col)

    # HUD text in top bar (after the star icon)
    f = _font(max(9, (layout["status_h"] - 12) // 2))
    text_x = layout["icon_star"].right + 8
    text_y = top.y + (top.height - f.get_height()) // 2
    label = f.render(hud_top, True, GREY)
    surface.blit(label, (text_x, text_y))

    # Badge area in top bar (before gear icon)
    badges = []
    if not cam_configured:
        badges.append(("Unconfigured", AMBER))
    if not cal_ok:
        badges.append(("Uncalibrated", AMBER))
    if state.recording:
        badges.append(("● REC", RED))
    bx = layout["icon_gear"].left - 8
    for badge_text, badge_col in reversed(badges):
        bl = f.render(badge_text, True, badge_col)
        bx -= bl.get_width()
        surface.blit(bl, (bx, text_y))
        bx -= 12

    # HUD text in bottom bar (after the sliders icon)
    text_x = layout["icon_sliders"].right + 8
    text_y = bot.y + (bot.height - f.get_height()) // 2
    label = f.render(hud_bot, True, GREY)
    surface.blit(label, (text_x, text_y))


# --- vertical drop-down panel (Action + Utilities menus) --------------------

_MENU_ROW_H = 24
_MENU_PAD   = 6
_MENU_W     = 200


def _menu_panel_rect(anchor: str, layout: dict) -> pygame.Rect:
    """Return the bounding rect for an action/utilities menu panel."""
    central = layout["central"]
    h = 0   # computed dynamically when drawing
    if anchor == "left":
        return pygame.Rect(central.x, central.y, _MENU_W, central.height)
    else:
        return pygame.Rect(central.right - _MENU_W, central.y, _MENU_W, central.height)


def _render_vertical_menu(
    surface: pygame.Surface,
    menu: Menu,
    anchor: str,
    layout: dict,
    cursor_pos: tuple[int, int],
) -> None:
    """Render action or utilities menu as a drop-down panel."""
    items  = menu.current_items
    if not items:
        return

    n      = len(items)
    ph     = n * _MENU_ROW_H + 2 * _MENU_PAD
    central = layout["central"]
    sx, sw = layout["status_x"], layout["status_w"]
    pw     = _MENU_W

    if anchor == "left":
        px = sx
    else:
        px = sx + sw - pw
    py = central.y

    # Dim the rest of the image area
    overlay = pygame.Surface((central.width, central.height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (central.x, central.y))

    # Panel background
    panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 220))
    surface.blit(panel, (px, py))
    pygame.draw.rect(surface, DIM, pygame.Rect(px, py, pw, ph), 1)

    f   = _font(max(9, _MENU_ROW_H - 10))
    sel = menu.selection_index
    mx, my = cursor_pos

    for i, item in enumerate(items):
        ry = py + _MENU_PAD + i * _MENU_ROW_H
        row_rect = pygame.Rect(px, ry, pw, _MENU_ROW_H)
        hovered = row_rect.collidepoint(mx, my)
        if hovered:
            menu.set_selection(i)

        if i == sel or hovered:
            pygame.draw.rect(surface, (40, 40, 40), row_rect)

        color = GREEN if (i == sel or hovered) else GREY
        label = f.render(item.label_text, True, color)
        surface.blit(label, (px + _MENU_PAD, ry + (_MENU_ROW_H - label.get_height()) // 2))


def _vertical_menu_hit(
    pos: tuple[int, int],
    menu: Menu,
    anchor: str,
    layout: dict,
) -> int | None:
    """Return clicked item index, or None if outside the panel."""
    items   = menu.current_items
    if not items:
        return None
    n       = len(items)
    ph      = n * _MENU_ROW_H + 2 * _MENU_PAD
    central = layout["central"]
    sx, sw  = layout["status_x"], layout["status_w"]
    pw      = _MENU_W
    px      = sx if anchor == "left" else sx + sw - pw
    py      = central.y
    mx, my  = pos

    if not (px <= mx < px + pw and py <= my < py + ph):
        return None
    row = (my - py - _MENU_PAD) // _MENU_ROW_H
    if 0 <= row < n:
        return row
    return None


# --- Controls panel (sliders) ------------------------------------------------

_CTRL_ROW_H   = 36
_CTRL_LABEL_W = 108
_CTRL_VAL_W   = 54
_CTRL_PAD     = 6
_CTRL_TRACK_H = 4
_CTRL_THUMB_R = 7

_CONTROLS_ROWS = [
    ("Stream Exp",   "stream_exposure",  _EXPOSURE_STEPS),
    ("Stack Exp",    "stack_exposure",   _EXPOSURE_STEPS),
    ("Brightness",   "brightness",       _BRIGHTNESS_STEPS),
    ("Sky Sub",      "sky_subtraction",  _SKY_STEPS),
]


def _controls_panel_rect(layout: dict) -> pygame.Rect:
    """Controls panel rises from the bottom of the central region, within status bar bounds."""
    central = layout["central"]
    ph = len(_CONTROLS_ROWS) * _CTRL_ROW_H + 2 * _CTRL_PAD
    return pygame.Rect(layout["status_x"], central.bottom - ph, layout["status_w"], ph)


def _ctrl_track_x(pr: pygame.Rect) -> int:
    return pr.x + _CTRL_LABEL_W + _CTRL_VAL_W + _CTRL_PAD


def _ctrl_track_w(pr: pygame.Rect) -> int:
    return pr.right - _ctrl_track_x(pr) - _CTRL_PAD


def _ctrl_thumb_x(track_x: int, track_w: int, idx: int, n_steps: int) -> int:
    t = idx / max(1, n_steps - 1)
    return track_x + int(t * track_w)


def _render_controls_menu(
    surface: pygame.Surface,
    layout: dict,
    state: ViewState,
    cursor_pos: tuple[int, int],
) -> None:
    central = layout["central"]
    pr = _controls_panel_rect(layout)

    overlay = pygame.Surface((central.width, central.height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (central.x, central.y))

    panel = pygame.Surface((pr.width, pr.height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 220))
    surface.blit(panel, (pr.x, pr.y))
    pygame.draw.rect(surface, DIM, pr, 1)

    f_lbl    = _font(13)
    f_val    = _font(13)
    f_tip    = _font(11)
    mx, my   = cursor_pos
    track_x  = _ctrl_track_x(pr)
    track_w  = _ctrl_track_w(pr)

    for ri, (label, attr, steps) in enumerate(_CONTROLS_ROWS):
        cy = pr.y + _CTRL_PAD + ri * _CTRL_ROW_H + _CTRL_ROW_H // 2
        current = getattr(state, attr)
        cur_idx = next((i for i, (_, v) in enumerate(steps) if v == current), 0)

        # Label
        lbl = f_lbl.render(label + ":", True, GREY)
        surface.blit(lbl, (pr.x + _CTRL_PAD, cy - lbl.get_height() // 2))

        # Current value — always visible, left of track
        val_text = steps[cur_idx][0]
        vsurf = f_val.render(val_text, True, WHITE)
        vx = pr.x + _CTRL_LABEL_W + (_CTRL_VAL_W - vsurf.get_width()) // 2
        surface.blit(vsurf, (vx, cy - vsurf.get_height() // 2))

        # Track rail
        rail = pygame.Rect(track_x, cy - _CTRL_TRACK_H // 2, track_w, _CTRL_TRACK_H)
        pygame.draw.rect(surface, (50, 50, 50), rail, border_radius=2)

        # Filled portion (left of thumb)
        thumb_x = _ctrl_thumb_x(track_x, track_w, cur_idx, len(steps))
        if thumb_x > track_x:
            filled = pygame.Rect(track_x, cy - _CTRL_TRACK_H // 2,
                                 thumb_x - track_x, _CTRL_TRACK_H)
            pygame.draw.rect(surface, GREY, filled, border_radius=2)

        # Thumb
        pygame.draw.circle(surface, WHITE, (thumb_x, cy), _CTRL_THUMB_R)

        # Hover: ghost thumb + tooltip
        hover_zone = pygame.Rect(track_x - _CTRL_THUMB_R, cy - _CTRL_THUMB_R * 2,
                                 track_w + _CTRL_THUMB_R * 2, _CTRL_THUMB_R * 4)
        if hover_zone.collidepoint(mx, my):
            raw_t     = (mx - track_x) / max(1, track_w)
            hover_idx = max(0, min(len(steps) - 1, round(raw_t * (len(steps) - 1))))
            if hover_idx != cur_idx:
                ghost_x = _ctrl_thumb_x(track_x, track_w, hover_idx, len(steps))
                pygame.draw.circle(surface, DIM, (ghost_x, cy), _CTRL_THUMB_R - 1)
                tip = f_tip.render(steps[hover_idx][0], True, GREY)
                tip_x = max(pr.x + _CTRL_PAD,
                            min(pr.right - tip.get_width() - _CTRL_PAD,
                                ghost_x - tip.get_width() // 2))
                tip_y = cy - _CTRL_THUMB_R - tip.get_height() - 2
                surface.blit(tip, (tip_x, tip_y))


def _controls_hit(
    pos: tuple[int, int],
    layout: dict,
) -> tuple[str, object] | None:
    """Return (attr_name, new_value) if pos lands on a slider track, else None."""
    pr = _controls_panel_rect(layout)
    mx, my = pos
    if not pr.collidepoint(mx, my):
        return None

    ri = (my - pr.y - _CTRL_PAD) // _CTRL_ROW_H
    if ri < 0 or ri >= len(_CONTROLS_ROWS):
        return None
    _label, attr, steps = _CONTROLS_ROWS[ri]

    track_x = _ctrl_track_x(pr)
    track_w = _ctrl_track_w(pr)
    if mx < track_x or mx > track_x + track_w:
        return None

    raw_t = (mx - track_x) / max(1, track_w)
    idx   = max(0, min(len(steps) - 1, round(raw_t * (len(steps) - 1))))
    return (attr, steps[idx][1])


# --- Context menu -----------------------------------------------------------

_CTX_ROW_H = 22
_CTX_PAD   = 4
_CTX_W     = 150


def _render_context_menu(
    surface: pygame.Surface,
    menu: Menu,
    pos: tuple[int, int],
    cursor_pos: tuple[int, int],
    layout: dict,
) -> None:
    items = menu.current_items
    if not items:
        return
    ph  = len(items) * _CTX_ROW_H + 2 * _CTX_PAD
    sx, sw = layout["status_x"], layout["status_w"]
    px  = max(sx, min(pos[0], sx + sw - _CTX_W - 4))
    py  = min(pos[1], layout["bot_bar"].top - ph - 4)

    panel = pygame.Surface((_CTX_W, ph), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 230))
    surface.blit(panel, (px, py))
    pygame.draw.rect(surface, DIM, pygame.Rect(px, py, _CTX_W, ph), 1)

    f   = _font(max(9, _CTX_ROW_H - 10))
    sel = menu.selection_index
    mx, my = cursor_pos

    for i, item in enumerate(items):
        ry = py + _CTX_PAD + i * _CTX_ROW_H
        row_rect = pygame.Rect(px, ry, _CTX_W, _CTX_ROW_H)
        hovered = row_rect.collidepoint(mx, my)
        if hovered:
            menu.set_selection(i)
        if i == sel or hovered:
            pygame.draw.rect(surface, (40, 40, 40), row_rect)
        color = GREEN if (i == sel or hovered) else GREY
        label = f.render(item.label_text, True, color)
        surface.blit(label, (px + _CTX_PAD, ry + (_CTX_ROW_H - label.get_height()) // 2))


def _context_hit(
    pos: tuple[int, int],
    menu: Menu,
    menu_pos: tuple[int, int],
    layout: dict,
) -> int | None:
    items = menu.current_items
    if not items:
        return None
    ph  = len(items) * _CTX_ROW_H + 2 * _CTX_PAD
    sx, sw = layout["status_x"], layout["status_w"]
    px  = max(sx, min(menu_pos[0], sx + sw - _CTX_W - 4))
    py  = min(menu_pos[1], layout["bot_bar"].top - ph - 4)
    mx, my = pos
    if not (px <= mx < px + _CTX_W and py <= my < py + ph):
        return None
    row = (my - py - _CTX_PAD) // _CTX_ROW_H
    return row if 0 <= row < len(items) else None


# --- Misc overlays ----------------------------------------------------------

_DSO_HIT_MARGIN = 15   # px beyond the drawn radius that counts as a hit

def _render_constellation_labels(
    surface: pygame.Surface,
    table: list[dict],
    img_rect: pygame.Rect | None,
) -> None:
    """Render permanent constellation name labels from entries with type 'constellation_label'."""
    ox = img_rect.left if img_rect else 0
    oy = img_rect.top  if img_rect else 0
    f  = _font(10)
    con_color = (140, 150, 230)   # soft blue-grey to match constellation line color
    for entry in table:
        if entry.get("type") != "constellation_label":
            continue
        name = entry.get("name", "")
        if not name:
            continue
        sx = entry["px"] + ox
        sy = entry["py"] + oy
        label = f.render(name, True, con_color)
        lw, lh = label.get_size()
        # Centre the text on the centroid
        tx = sx - lw // 2
        ty = sy - lh // 2
        bg = pygame.Surface((lw + 4, lh + 2), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 120))
        surface.blit(bg,    (tx - 2, ty - 1))
        surface.blit(label, (tx, ty))


def _render_hover_label(
    surface: pygame.Surface,
    table: list[dict],
    cursor_x: int,
    cursor_y: int,
    img_rect: pygame.Rect | None = None,
    threshold: int = 25,
) -> None:
    if not table:
        return

    # Hit detection uses overlay-local coords (origin = img_rect top-left).
    # Label rendering uses screen coords (cursor_x / cursor_y unchanged).
    ox = img_rect.left if img_rect else 0
    oy = img_rect.top  if img_rect else 0
    lx = cursor_x - ox   # cursor in overlay-local space
    ly = cursor_y - oy

    # Non-stars (DSOs) take priority over stars when their regions overlap.
    best_dso  = None
    best_dso_d2 = float("inf")
    best_star  = None
    best_star_d2 = float("inf")

    for entry in table:
        if entry.get("type") == "constellation_label":
            continue
        d2 = (entry["px"] - lx) ** 2 + (entry["py"] - ly) ** 2
        is_star = entry.get("type") == "star"
        if is_star:
            if d2 < threshold * threshold and d2 < best_star_d2:
                best_star_d2 = d2
                best_star = entry
        else:
            hit_r = entry.get("radius", 0) + _DSO_HIT_MARGIN
            hit_r = max(hit_r, threshold)
            if d2 < hit_r * hit_r and d2 < best_dso_d2:
                best_dso_d2 = d2
                best_dso = entry

    best = best_dso if best_dso is not None else best_star
    if best is None:
        return

    name = best.get("name") or best.get("type") or ""
    mag  = best.get("mag")
    typ  = best.get("type") or ""
    text = f"{name}  mag {mag:.1f}  {typ}" if mag is not None else f"{name}  {typ}"
    f = _font(10)
    label = f.render(text, True, WHITE)
    x = min(cursor_x + 16, surface.get_width()  - label.get_width()  - 10)
    y = max(cursor_y - 10, 4)
    bg = pygame.Surface((label.get_width() + 8, label.get_height() + 4), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 190))
    surface.blit(bg,    (x - 4, y - 2))
    surface.blit(label, (x, y))


def _render_focus_waiting(surface: pygame.Surface, win_w: int, top_y: int) -> None:
    f = _font(12)
    label = f.render("FOCUS — click to set point", True, GREEN)
    surface.blit(label, (win_w // 2 - label.get_width() // 2, top_y + 8))


def _render_alert(surface: pygame.Surface, text: str, win_w: int, win_h: int) -> None:
    f = _font(16)
    label = f.render(text, True, RED)
    x = win_w // 2 - label.get_width() // 2
    y = win_h // 2 - label.get_height() // 2
    bg = pygame.Surface((label.get_width() + 20, label.get_height() + 10))
    bg.fill(BLACK)
    surface.blit(bg,    (x - 10, y - 5))
    surface.blit(label, (x, y))


def _draw_cursor(surface: pygame.Surface, x: int, y: int) -> None:
    size = 10
    pygame.draw.line(surface, GREEN, (x - size, y), (x + size, y), 2)
    pygame.draw.line(surface, GREEN, (x, y - size), (x, y + size), 2)


def _make_qr_surface(data: str, cell_px: int = 5) -> pygame.Surface | None:
    """Render a QR code as a pygame Surface. Returns None if qrcode not installed."""
    try:
        import qrcode as _qr
        qr = _qr.QRCode(
            error_correction=_qr.constants.ERROR_CORRECT_M,
            box_size=1, border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        matrix = qr.get_matrix()
    except ImportError:
        return None
    n = len(matrix)
    size = n * cell_px
    surf = pygame.Surface((size, size))
    surf.fill((255, 255, 255))
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surf, (0, 0, 0),
                                 (x * cell_px, y * cell_px, cell_px, cell_px))
    return surf


def _render_qr_overlay(
    surface: pygame.Surface,
    qr_surf: pygame.Surface,
    layout: dict,
    line1_text: str,
    line2_text: str,
    timer: float,
    total: float,
) -> None:
    """Draw the QR code + caption centred in the central region."""
    central = layout["central"]
    qs = qr_surf.get_width()
    padding = 12

    f = _font(11)
    line1 = f.render(line1_text, True, WHITE)
    line2 = f.render(line2_text, True, GREY) if line2_text else None

    panel_w = max(qs + 2 * padding, line1.get_width() + 2 * padding)
    panel_h = (qs + line1.get_height() + 3 * padding
               + ((line2.get_height() + padding) if line2 else 0))

    px = central.centerx - panel_w // 2
    py = central.centery - panel_h // 2

    # Fade out in the last second
    alpha = max(0, int(220 * min(1.0, timer)))
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, alpha))
    surface.blit(panel, (px, py))

    qx = px + (panel_w - qs) // 2
    qy = py + padding
    surface.blit(qr_surf, (qx, qy))
    surface.blit(line1, (px + (panel_w - line1.get_width()) // 2, qy + qs + padding))
    if line2:
        surface.blit(line2, (px + (panel_w - line2.get_width()) // 2,
                              qy + qs + padding + line1.get_height() + 4))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", metavar="PATH", default=None,
                   help="Path to configuration.yaml (overrides default search)")
    p.add_argument("--exposure", type=float, default=None,
                   help="Override streaming start exposure in seconds")
    p.add_argument("--vcam", metavar="SUBFOLDER",
                   help="Replay a recorded session instead of live hardware.")
    p.add_argument("--vcam-accel", type=float, default=1.0, metavar="FACTOR",
                   help="Time-acceleration for vcam playback (default: 1.0)")
    p.add_argument("--vmount", action="store_true",
                   help="Use a virtual mount for testing (no hardware required). "
                        "Connect via Action > Telescope > Connect.")
    p.add_argument("--windowed", action="store_true",
                   help="Run in a window instead of fullscreen.")
    p.add_argument("--window-size", metavar="WxH", default=None,
                   help="Windowed size override, e.g. 1280x720. Implies --windowed.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    try:
        _cfg_path = args.config or (_DEFAULT_CONFIG if _DEFAULT_CONFIG.exists() else None)
        config = load(config_path=_cfg_path, data_root=_DATA_ROOT)
    except FileNotFoundError as exc:
        logging.warning("%s — using defaults", exc)
        config = None

    lat = config.latitude  if config else 38.44
    lon = config.longitude if config else -122.71

    # -- camera setup --------------------------------------------------------
    if args.vcam:
        if config is None or config.record_path is None:
            logging.error("--vcam requires record_path in configuration.yaml")
            sys.exit(1)
        session_path = config.record_path / args.vcam
        if not session_path.is_dir():
            logging.error("Session folder not found: %s", session_path)
            sys.exit(1)
        _vcam_table = scan_folder(session_path)
        if len(_vcam_table) == 0:
            logging.error("No canonical .tif frames in %s", session_path)
            sys.exit(1)
        _vcam_id  = int(_vcam_table.camera_id.iloc[0])
        _vcam_cfg = config.get_config(_vcam_id) if config else None
        cam_ctx: VirtualCamera | ZwoAsiCamera = VirtualCamera(
            session_path,
            camera_id     = _vcam_id,
            bayer_pattern = _vcam_cfg.pattern if _vcam_cfg else None,
            t_accel       = args.vcam_accel,
        )
    else:
        cameras = list_cameras()
        if not cameras:
            logging.warning("No ZWO ASI cameras found — starting without camera.")
            cam_ctx = NullCamera()
        else:
            # Pick the first connected camera in YAML order; fall back to USB order.
            connected_ids = {c.get("camera_id", 0): c["usb_index"] for c in cameras}
            yaml_ids = config.camera_ids() if config else []
            start_id = next((cid for cid in yaml_ids if cid in connected_ids), None)
            usb_index = connected_ids.get(start_id, cameras[0]["usb_index"])
            cam_ctx = ZwoAsiCamera(index=usb_index)

    catalog      = load_catalog(str(_CATALOG_PATH))
    moon_catalog = load_moon_catalog(str(_MOON_CATALOG_PATH))
    try:
        constellation_lines = load_constellation_lines(
            str(_CONSTELLATION_PATH), str(_CATALOG_PATH))
    except Exception as exc:
        logging.warning("Failed to load constellation lines: %s", exc)
        constellation_lines = []

    # -- pygame init ---------------------------------------------------------
    pygame.init()

    info = pygame.display.Info()
    screen_w = info.current_w if info.current_w > 0 else WINDOW_W
    screen_h = info.current_h if info.current_h > 0 else WINDOW_H

    windowed = args.windowed or bool(args.window_size)

    if args.window_size:
        try:
            win_w, win_h = (int(v) for v in args.window_size.lower().split("x"))
        except ValueError:
            logging.error("--window-size must be WxH, e.g. 1280x720")
            sys.exit(1)
    elif windowed:
        # In windowed mode the OS menu bar / notch / dock eat into the available
        # height.  Reserve a platform-appropriate margin so the window fits.
        if sys.platform == "darwin":
            margin_h = 90   # menu bar (~38 px) + notch headroom + dock
        elif sys.platform == "win32":
            margin_h = 48   # taskbar
        else:
            margin_h = 40   # Pi / Linux desktop taskbar (~36 px)
        win_w = min(WINDOW_W, screen_w)
        win_h = min(WINDOW_H, screen_h - margin_h)
    else:
        win_w, win_h = screen_w, screen_h

    if windowed:
        screen = pygame.display.set_mode((win_w, win_h))
    else:
        screen = pygame.display.set_mode((win_w, win_h),
                                         pygame.FULLSCREEN | pygame.NOFRAME)

    pygame.display.set_caption(WINDOW_TITLE)
    pygame.mouse.set_visible(True)
    clock = pygame.time.Clock()

    layout = _make_layout(win_w, win_h)

    state = ViewState()

    cam_config_ref: list[CameraConfig] = [CameraConfig()]
    fov_ref:        list[float]        = [30.0]
    mount_holder:   list               = [None]
    overlay_style = load_overlay_style(
        _OVERLAY_STYLE_PATH,
        cam_config_ref[0].overlay_style,
    )

    def _open_menu(name: str) -> None:
        state.active_menu = name
        state.menu_open   = True

    def _close_menu() -> None:
        state.active_menu = None
        state.menu_open   = False
        action_menu.reset()
        utilities_menu.reset()
        if context_menu_ref[0] is not None:
            context_menu_ref[0].reset()

    def _connect_mount() -> None:
        if args.vmount:
            from astrocore.mount.virtual_mount import VirtualMount
            mount_holder[0] = VirtualMount(lat_deg=lat, lon_deg=lon)
            state.mount_connected = True
            state.mount_tracking  = True   # VirtualMount always tracks
            return
        driver = config.mount_driver if config else ""
        if not driver:
            return
        try:
            mod = importlib.import_module(f"astrocore.mount.{driver}")
            cls = getattr(mod, "Driver", None)
            if cls is None:
                return
            mount_holder[0] = cls()
            state.mount_connected = True
            try:
                state.mount_tracking = mount_holder[0].is_tracking
            except Exception:
                state.mount_tracking = False
            if not state.mount_tracking:
                logging.info("Mount connected but not tracking — overlay will use north horizon")
        except Exception as exc:
            logging.warning("Mount connect failed: %s", exc)

    def _disconnect_mount() -> None:
        if mount_holder[0] is not None:
            mount_holder[0].disconnect()
            mount_holder[0] = None
        state.mount_connected = False
        state.mount_tracking  = False

    with cam_ctx as cam:
        _cam_configured = config is not None and cam.info.camera_id in config.camera_ids()
        if not _cam_configured:
            id_note = " (EEPROM not set)" if cam.info.camera_id == 0 else ""
            logging.warning("Camera ID %d%s not in configuration.yaml", cam.info.camera_id, id_note)

        if config is not None:
            cam_config = config.get_config(cam.info.camera_id)
            cam_config_ref[0] = cam_config
            overlay_style = load_overlay_style(_OVERLAY_STYLE_PATH, cam_config.overlay_style)
            if cam_config.bin > 1:
                cam.bin = cam_config.bin
            if cam_config.gain is not None:
                cam.gain = cam_config.gain
            if cam_config.data_offset is not None:
                cam.offset = cam_config.data_offset
            if cam_config.pattern is not None:
                cam._info = _dc_replace(cam.info, bayer_pattern=cam_config.pattern)
            cam.meta.telescope_description = cam_config.telescope_description
            cam.meta.focal_length_mm       = cam_config.focal_length_mm
            cam.meta.Lat = lat
            cam.meta.Lon = lon
            roi_x, roi_y, roi_w, roi_h = cam_config.effective_roi(
                cam.info.sensor_width_px, cam.info.sensor_height_px)
            if (isinstance(cam, ZwoAsiCamera)
                    and (cam_config.cam_size[0] > 0 or cam_config.cam_size[1] > 0)):
                cam.set_roi(x=roi_x, y=roi_y, width=roi_w, height=roi_h)
            fov_ref[0] = compute_hfov(
                cam_config.focal_length_mm, cam.info.pixel_size_um, roi_w,
            ) or fov_ref[0]

        # -- open all other configured cameras so we can switch between them ----
        cam_pool: dict[int, ZwoAsiCamera | VirtualCamera] = {cam.info.camera_id: cam}
        _extra_cams: list[ZwoAsiCamera] = []
        if not args.vcam:
            for _uc in cameras:
                _eid = _uc.get("camera_id", 0)
                if _eid not in cam_pool:
                    _ec = ZwoAsiCamera(index=_uc["usb_index"])
                    try:
                        _ec.connect()
                        cam_pool[_eid] = _ec
                        _extra_cams.append(_ec)
                        logging.info("Opened secondary camera ID %d", _eid)
                    except Exception as _ex:
                        logging.warning("Could not open camera ID %d: %s", _eid, _ex)

        def _apply_cam_config(target_cam, cfg: CameraConfig) -> None:
            """Apply CameraConfig settings to a camera object."""
            if cfg.gain is not None:
                target_cam.gain = cfg.gain
            if cfg.data_offset is not None:
                target_cam.offset = cfg.data_offset
            if cfg.pattern is not None:
                target_cam._info = _dc_replace(target_cam.info, bayer_pattern=cfg.pattern)
            target_cam.meta.telescope_description = cfg.telescope_description
            target_cam.meta.focal_length_mm       = cfg.focal_length_mm
            target_cam.meta.Lat = lat
            target_cam.meta.Lon = lon

        def _switch_cam_config(name: str, camera_id: int) -> None:
            """Switch to a named config; if it belongs to a different physical camera, swap hardware."""
            if config is None:
                return
            new_cfg = config.get_config(camera_id, name)
            new_cam = cam_pool.get(camera_id)

            if new_cam is not None and new_cam is not grabber.cam:
                grabber.reset()
                if isinstance(new_cam, ZwoAsiCamera):
                    _apply_cam_config(new_cam, new_cfg)
                    if new_cfg.cam_size[0] > 0 or new_cfg.cam_size[1] > 0:
                        rx, ry, rw, rh = new_cfg.effective_roi(
                            new_cam.info.sensor_width_px, new_cam.info.sensor_height_px)
                        new_cam.set_roi(x=rx, y=ry, width=rw, height=rh)
                grabber.cam     = new_cam
                grabber.imDark  = 0
                grabber.imFlat  = 0
                grabber.imDPC   = 0
                grabber.pattern = new_cfg.pattern or None

            cam_config_ref[0] = new_cfg
            roi_x, roi_y, roi_w, roi_h = new_cfg.effective_roi(
                grabber.cam.info.sensor_width_px, grabber.cam.info.sensor_height_px)
            fov_ref[0] = compute_hfov(
                new_cfg.focal_length_mm, grabber.cam.info.pixel_size_um, roi_w
            ) or fov_ref[0]
            stacker.reset()
            state.paused = False
            _close_menu()

        grabber = FrameGrabber(cam)
        grabber.pattern = cam.info.bayer_pattern
        if config is not None and config.cal_path is not None:
            grabber.cal_path = config.cal_path

        recorder: Recorder | None = (
            None if args.vcam
            else Recorder(config.record_path)
            if config is not None and config.record_path is not None
            else None
        )

        # -- image save / gallery setup --------------------------------------
        image_path: Path | None = config.image_path if config else None
        hotspot: HotspotConfig  = config.hotspot    if config else HotspotConfig()

        if image_path is not None:
            image_path.mkdir(parents=True, exist_ok=True)
            GalleryServer(image_path, port=hotspot.port).start()

        # QR surface — WiFi join QR on Pi; gallery URL QR on Mac/Windows
        _is_pi = sys.platform == "linux"

        def _get_local_ip() -> str:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
                    _s.connect(("8.8.8.8", 80))
                    return _s.getsockname()[0]
            except Exception:
                return "localhost"

        if _is_pi:
            _qr_data  = hotspot.wifi_qr_data
            _qr_line1 = f"Join WiFi: {hotspot.ssid}  •  pw: {hotspot.password}"
            _qr_line2 = f"Then open: {hotspot.gallery_url}"
        else:
            _local_ip    = _get_local_ip()
            _gallery_url = f"http://{_local_ip}:{hotspot.port}"
            _qr_data  = _gallery_url
            _qr_line1 = f"Gallery: {_gallery_url}"
            _qr_line2 = ""

        _qr_surf: pygame.Surface | None = _make_qr_surface(_qr_data)
        _qr_timer: list[float] = [0.0]   # seconds remaining for QR overlay
        _QR_DURATION = 4.0
        _save_frame_ref:    list[np.ndarray | None] = [None]   # latest displayed uint8 frame
        _hotspot_started:   list[bool]             = [False]  # guest hotspot launched this session

        if _is_pi:
            _hs_log = open("/tmp/astro-hotspot-start.log", "w")
            subprocess.Popen(
                ["sudo", "/usr/local/bin/astro-hotspot-start"],
                stdin=subprocess.DEVNULL,
                stdout=_hs_log,
                stderr=_hs_log,
            )
            _hotspot_started[0] = True


        def _on_save() -> None:
            if image_path is None:
                logging.warning("image_path not set in configuration.yaml — cannot save")
                _close_menu()
                return

            save_frame = _save_frame_ref[0]
            if save_frame is None:
                logging.warning("No frame available to save yet")
                _close_menu()
                return

            # Scale stacked frame to the displayed image dimensions
            out_w, out_h = img_rect.width, img_rect.height
            scaled = cv2.resize(save_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            bgr = (cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
                   if scaled.ndim == 2
                   else scaled)  # already BGR from cv2 debayer pipeline

            # Text annotations
            exp_s   = _exposure_ref[0] / 1_000_000
            stats   = [
                f"exp {exp_s:.3g}s",
                f"total {stacker.t_accum:.0f}s",
                f"frames {stacker.frame_count}",
                time.strftime("%Y-%m-%d"),
            ]
            wm_text = "impatientastronomy.com"

            cv_font    = cv2.FONT_HERSHEY_SIMPLEX
            margin     = max(8, out_h // 80)
            stat_scale = max(0.35, out_h / 2000)
            wm_scale   = max(0.4,  out_h / 1800)

            def _shadowed(img, text, org, scale, thick):
                cv2.putText(img, text, (org[0] + 1, org[1] + 1),
                            cv_font, scale, (30, 30, 30), thick + 1, cv2.LINE_AA)
                cv2.putText(img, text, org,
                            cv_font, scale, (220, 220, 220), thick, cv2.LINE_AA)

            # Left side, vertically stacked from bottom up
            (_, line_h), baseline = cv2.getTextSize("A", cv_font, stat_scale, 1)
            line_step = line_h + baseline + max(2, out_h // 150)
            for i, line in enumerate(reversed(stats)):
                y = out_h - margin - i * line_step
                _shadowed(bgr, line, (margin, y), stat_scale, 1)

            # Bottom-centre watermark
            (ww, _), _ = cv2.getTextSize(wm_text, cv_font, wm_scale, 1)
            _shadowed(bgr, wm_text, ((out_w - ww) // 2, out_h - margin), wm_scale, 1)

            ts   = time.strftime("%Y-%m-%d_%H-%M-%S")
            path = image_path / f"{ts}.jpg"
            cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            logging.info("Saved %s", path)
            _qr_timer[0] = _QR_DURATION
            if _is_pi:
                if not _hotspot_started[0]:
                    subprocess.Popen(["sudo", "/usr/local/bin/astro-hotspot-start"])
                    _hotspot_started[0] = True
            else:
                webbrowser.open(f"http://localhost:{hotspot.port}")
            _close_menu()

        def _on_clear_images() -> None:
            if image_path is None:
                _close_menu()
                return
            deleted = 0
            for f in image_path.glob("*.jpg"):
                f.unlink()
                deleted += 1
            logging.info("Cleared %d image(s) from %s", deleted, image_path)
            _close_menu()

        def _on_set_dpc() -> None:
            # TODO: trigger DPC acquisition
            _close_menu()

        def _on_quit() -> None:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

        action_menu = _build_action_menu(
            state,
            on_connect    = _connect_mount,
            on_disconnect = _disconnect_mount,
            recorder      = recorder,
            on_save       = _on_save,
            on_quit       = _on_quit,
            mount_driver  = ("Virtual mount" if args.vmount
                             else config.mount_driver if config else ""),
        )

        def _on_moon_mode() -> None:
            state.moon_mode = not state.moon_mode
            _close_menu()

        utilities_menu = _build_utilities_menu(
            cam, cam_config_ref, _on_set_dpc,
            on_clear_images  = _on_clear_images,
            on_moon_mode     = _on_moon_mode,
            moon_mode_label_fn = lambda: "Moon Map [ON]" if state.moon_mode else "Moon Map",
        )

        context_menu_ref: list[Menu | None] = [None]

        dispatcher = InputDispatcher(
            state, action_menu,
            zoom_step       = 1.2,
            zoom_min        = 1.0,
            zoom_max        = config.max_zoom         if config else 5.0,
            sky_map_fov_min = config.sky_map.fov_min  if config else 10.0,
            sky_map_fov_max = config.sky_map.fov_max  if config else 60.0,
        )

        actual_gain = grabber.cam.gain
        ae        = StreamExposure(start=args.exposure if args.exposure is not None else 0.1)
        stacker = ConstellationStacker()
        stack_seq = ExposureSequence(STACKING_SEQUENCE)

        # Camera image dimensions — determined on first frame
        cam_w_ref: list[int] = [cam.info.sensor_width_px  or win_w]
        cam_h_ref: list[int] = [cam.info.sensor_height_px or win_h]
        img_rect   = _image_rect(cam_w_ref[0], cam_h_ref[0], layout["central"])
        dispatcher.set_img_rect(img_rect)

        _frame_queue: queue.Queue = queue.Queue(maxsize=2)
        _exposure_ref = [int(ae.current * 1_000_000)]
        _focus_hardware_roi: bool = False

        def _make_grab_worker(stop_event: threading.Event) -> threading.Thread:
            def _worker() -> None:
                configured_us: int | None = None
                while not stop_event.is_set():
                    exp_us = _exposure_ref[0]
                    result = grabber.grab_frame(
                        exposure_us = exp_us if exp_us != configured_us else None,
                        dark        = not _focus_hardware_roi and grabber.cal_path is not None,
                        flat        = not _focus_hardware_roi and grabber.cal_path is not None,
                        dpc         = False,
                        demosaic    = grabber.pattern is not None,
                        median      = True,
                    )
                    if result.status == GrabStatus.WORKING:
                        time.sleep(0.001)
                        continue
                    if result.status == GrabStatus.STARTED:
                        configured_us = exp_us
                        continue
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

        _stop_grab  = threading.Event()
        _grab_thread = None
        if not isinstance(cam, NullCamera):
            _grab_thread = _make_grab_worker(_stop_grab)
            _grab_thread.start()

        _focus_roi_saved: tuple[int, int, int, int] | None = None

        def _stop_and_reset_grab() -> None:
            _stop_grab.set()
            if _grab_thread is not None:
                _grab_thread.join(timeout=5.0)
            grabber.reset()

        def _restart_grab() -> None:
            nonlocal _stop_grab, _grab_thread
            _stop_grab  = threading.Event()
            _grab_thread = _make_grab_worker(_stop_grab)
            _grab_thread.start()

        def _enter_focus_active() -> None:
            nonlocal _focus_roi_saved, _focus_hardware_roi
            if not isinstance(grabber.cam, ZwoAsiCamera):
                return
            _stop_and_reset_grab()
            saved = grabber.cam.get_roi()
            _focus_roi_saved = saved
            roi_x, roi_y, roi_w, roi_h = saved
            sensor_cx = roi_x + int(state.focus_center_x * roi_w)
            sensor_cy = roi_y + int(state.focus_center_y * roi_h)
            sensor_w  = grabber.cam.info.sensor_width_px
            sensor_h  = grabber.cam.info.sensor_height_px
            fx = max(0, min(sensor_w - FOCUS_ROI_HALF * 2, (sensor_cx - FOCUS_ROI_HALF) & ~1))
            fy = max(0, min(sensor_h - FOCUS_ROI_HALF * 2, (sensor_cy - FOCUS_ROI_HALF) & ~1))
            grabber.cam.set_roi(x=fx, y=fy, width=FOCUS_ROI_HALF * 2, height=FOCUS_ROI_HALF * 2)
            _focus_hardware_roi = True
            _restart_grab()

        def _focus_here(cx: float, cy: float) -> None:
            """Immediately activate focus centred at normalised window coords (cx, cy)."""
            if state.recording and recorder is not None:
                state.recording = False
                recorder.stop()
            state.focus_center_x = cx
            state.focus_center_y = cy
            state.mode            = ViewMode.LIVE
            _enter_focus_active()
            state.focus_state     = FocusState.ACTIVE
            _close_menu()

        def _exit_focus() -> None:
            nonlocal _focus_roi_saved, _focus_hardware_roi
            if _focus_hardware_roi and _focus_roi_saved is not None and isinstance(grabber.cam, ZwoAsiCamera):
                _stop_and_reset_grab()
                x, y, w, h = _focus_roi_saved
                grabber.cam.set_roi(x=x, y=y, width=w, height=h)
                _focus_hardware_roi = False
                _focus_roi_saved    = None
                _restart_grab()
            state.focus_state = FocusState.OFF
            state.mode        = ViewMode.LIVE

        last_surface: pygame.Surface | None = None
        frame_count   = 0
        fps_display   = 0.0
        t_last_frame  = time.monotonic()
        prev_mode     = state.mode
        _cal_ok       = True

        t_last_move = time.monotonic()
        cursor_pos  = (win_w // 2, win_h // 2)

        alert_timer   = 0.0
        alert_message = ""
        ov_table: list[dict] = []

        # Overlay cache — recomputed only when inputs change or menu closes
        _ov_surf: pygame.Surface | None = None
        _ov_key:  tuple                 = ()

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
                        if state.active_menu:
                            _close_menu()
                        elif state.focus_state != FocusState.OFF:
                            _exit_focus()
                        else:
                            running = False
                    elif state.focus_state != FocusState.OFF:
                        _exit_focus()
                    elif event.key == pygame.K_r and recorder is not None:
                        if state.recording:
                            state.recording = False
                            recorder.stop()
                        else:
                            state.recording = True
                            recorder.start()
                    elif event.key == pygame.K_s:
                        _on_save()

                elif event.type == pygame.MOUSEMOTION:
                    cursor_pos  = event.pos
                    t_last_move = time.monotonic()
                    dispatcher.on_mouse_move(*event.pos, right_held=bool(event.buttons[2]))

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos

                    if state.focus_state == FocusState.ACTIVE and event.button == 1:
                        _exit_focus()

                    elif event.button == 1:
                        # --- Left-click routing ---
                        active = state.active_menu

                        if active == "action":
                            idx = _vertical_menu_hit(event.pos, action_menu, "left", layout)
                            if idx is not None:
                                action_menu.set_selection(idx)
                                if action_menu.select():
                                    _close_menu()
                            else:
                                _close_menu()

                        elif active == "utilities":
                            idx = _vertical_menu_hit(event.pos, utilities_menu, "right", layout)
                            if idx is not None:
                                utilities_menu.set_selection(idx)
                                if utilities_menu.select():
                                    _close_menu()
                            else:
                                _close_menu()

                        elif active == "controls":
                            hit = _controls_hit(event.pos, layout)
                            if hit is not None:
                                attr, val = hit
                                setattr(state, attr, val)
                                # Don't close controls menu on select — user likely wants to tweak
                            else:
                                _close_menu()

                        elif active == "context":
                            ctx = context_menu_ref[0]
                            if ctx is not None:
                                idx = _context_hit(event.pos, ctx, state.context_menu_pos,
                                                   layout)
                                if idx is not None:
                                    ctx.set_selection(idx)
                                    if ctx.select():
                                        _close_menu()
                                else:
                                    _close_menu()   # clicked outside menu
                            else:
                                _close_menu()

                        else:
                            # No menu open — check icon hits, then toggle
                            if layout["icon_star"].collidepoint(mx, my):
                                action_menu.reset()
                                _open_menu("action")
                            elif layout["icon_gear"].collidepoint(mx, my):
                                utilities_menu.reset()
                                _open_menu("utilities")
                            elif layout["icon_sliders"].collidepoint(mx, my):
                                _open_menu("controls")
                            else:
                                # Toggle Stream / Stack
                                if state.mode == ViewMode.LIVE:
                                    state.mode = ViewMode.ACCUMULATE
                                else:
                                    state.mode = ViewMode.LIVE

                    elif event.button == 3:
                        # --- Right button down: start drag/click tracking ---
                        dispatcher.on_right_button_down(mx, my)
                        if state.active_menu:
                            _close_menu()

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3 and dispatcher.on_right_button_up():
                        if not state.active_menu and state.focus_state == FocusState.ACTIVE:
                            m = Menu()
                            m.add(MenuItem("Exit focus mode", action=_exit_focus))
                            context_menu_ref[0] = m
                            state.context_menu_pos = event.pos
                            _open_menu("context")
                        elif not state.active_menu:
                            mx, my = event.pos
                            lx = mx - img_rect.left
                            ly = my - img_rect.top
                            near = None
                            for entry in ov_table:
                                d2 = (entry["px"] - lx) ** 2 + (entry["py"] - ly) ** 2
                                if d2 < 25 * 25:
                                    near = entry
                                    break

                            def _cursor_to_radec() -> tuple[float, float] | None:
                                """Convert the right-click screen position to RA/Dec."""
                                if mount_holder[0] is None:
                                    return None
                                try:
                                    ra_h, dec_deg_m = mount_holder[0].position
                                except Exception:
                                    return None
                                alt_m, az_m = radec_to_altaz(ra_h, dec_deg_m, lat, lon)
                                aspect_m = img_rect.height / img_rect.width
                                nx = (mx - img_rect.left) / img_rect.width
                                ny = (my - img_rect.top)  / img_rect.height
                                if state.all_sky_mode:
                                    fov = state.sky_map_fov
                                    d_az  = (state.zoom_center_x - 0.5) * fov
                                    d_alt = -(state.zoom_center_y - 0.5) * fov * aspect_m
                                else:
                                    fov   = fov_ref[0] / state.zoom_level
                                    d_az  = (state.zoom_center_x - 0.5) * fov_ref[0]
                                    d_alt = -(state.zoom_center_y - 0.5) * fov_ref[0] * aspect_m
                                center_alt = alt_m + d_alt
                                center_az  = az_m  + d_az
                                click_az   = center_az  + (nx - 0.5) * fov
                                click_alt  = center_alt - (ny - 0.5) * fov * aspect_m
                                return altaz_to_radec(click_alt, click_az, lat, lon)

                            def _do_slew() -> None:
                                nonlocal alert_timer, alert_message
                                coords = _cursor_to_radec()
                                if coords is None:
                                    return
                                try:
                                    mount_holder[0].slew_to(*coords)
                                    alert_message = "Caution: Mount is moving"
                                except Exception as exc:
                                    alert_message = f"Slew failed: {exc}"
                                alert_timer = ALERT_DURATION

                            def _do_sync() -> None:
                                nonlocal alert_timer, alert_message
                                if near is None or mount_holder[0] is None:
                                    return
                                ra_h    = near["ra_deg"] / 15.0   # degrees → hours
                                dec_deg = near["dec_deg"]
                                try:
                                    mount_holder[0].sync(ra_h, dec_deg)
                                    state.mount_tracking = True
                                    alert_message = "Mount synced"
                                except Exception as exc:
                                    alert_message = f"Sync failed: {exc}"
                                alert_timer = ALERT_DURATION

                            def _enter_sky_map() -> None:
                                state.all_sky_mode   = True
                                state.overlay_active = True
                                state.sky_map_fov    = (config.sky_map.fov_default
                                                        if config else 20.0)
                                state.zoom_center_x  = 0.5
                                state.zoom_center_y  = 0.5
                                # Snapshot the active camera's FOV for the FOV box
                                state.sky_map_cam_fov_h = fov_ref[0]
                                cw = cam_w_ref[0] if cam_w_ref[0] > 0 else 1
                                ch = cam_h_ref[0] if cam_h_ref[0] > 0 else 1
                                state.sky_map_cam_fov_v = fov_ref[0] * ch / cw
                                _close_menu()

                            def _exit_sky_map() -> None:
                                state.all_sky_mode      = False
                                state.overlay_active    = False
                                state.zoom_center_x     = 0.5
                                state.zoom_center_y     = 0.5
                                state.sky_map_cam_fov_h = None
                                state.sky_map_cam_fov_v = None
                                _close_menu()

                            _all_configs: list[tuple[str, int]] = []
                            if config:
                                for _cid in config.camera_ids():
                                    if _cid in cam_pool:
                                        for _n in config.config_names(_cid):
                                            _all_configs.append((_n, _cid))
                            _cam_sel_items: list[MenuItem] = [
                                MenuItem(n, action=lambda n=n, cid=cid: _switch_cam_config(n, cid))
                                for n, cid in _all_configs
                            ]
                            if state.mount_connected:
                                _cam_sel_items.append(
                                    MenuItem("SkyMap", action=_enter_sky_map))

                            if state.all_sky_mode:
                                context_menu_ref[0] = _build_context_menu(
                                    near_object     = near is not None,
                                    object_name     = near["name"] if near else "",
                                    on_focus        = lambda: _focus_here(*_screen_to_sensor_norm(mx, my, img_rect, state)),
                                    on_slew         = _do_slew,
                                    on_sync         = _do_sync,
                                    on_sky_map      = None,
                                    mount_connected = state.mount_connected,
                                    in_sky_map      = True,
                                    on_exit_sky_map = _exit_sky_map,
                                )
                            else:
                                context_menu_ref[0] = _build_context_menu(
                                    near_object       = near is not None,
                                    object_name       = near["name"] if near else "",
                                    on_focus          = lambda: _focus_here(*_screen_to_sensor_norm(mx, my, img_rect, state)),
                                    on_slew           = _do_slew,
                                    on_sync           = _do_sync,
                                    on_sky_map        = _enter_sky_map,
                                    mount_connected   = state.mount_connected,
                                    cam_select_items  = _cam_sel_items,
                                )
                            state.context_menu_pos = event.pos
                            _open_menu("context")

                elif event.type == pygame.MOUSEWHEEL:
                    if state.focus_state == FocusState.OFF:
                        if state.active_menu in (None, "controls"):
                            dispatcher.on_scroll(event.y, cursor_pos)   # zoom
                        else:
                            # Scroll navigates action / utilities menus
                            if state.active_menu == "action":
                                action_menu.scroll(-event.y)
                            elif state.active_menu == "utilities":
                                utilities_menu.scroll(-event.y)

            # -- mode change --------------------------------------------------
            if state.mode != prev_mode:
                state.paused = False
                if state.mode == ViewMode.LIVE:
                    ae.reset()
                    if state.recording and recorder is not None:
                        state.recording = False
                        recorder.stop()
                elif state.mode == ViewMode.ACCUMULATE:
                    stacker.reset()
                    stack_seq.reset()
                prev_mode = state.mode

            # -- exposure control ---------------------------------------------
            if state.mode == ViewMode.LIVE:
                if state.stream_exposure is not None:
                    _exposure_ref[0] = int(state.stream_exposure * 1_000_000)
                else:
                    _exposure_ref[0] = int(ae.current * 1_000_000)
            else:
                if state.stack_exposure is not None:
                    _exposure_ref[0] = int(state.stack_exposure * 1_000_000)
                else:
                    _exposure_ref[0] = int(stack_seq.current * 1_000_000)

            # -- grab frame ---------------------------------------------------
            try:
                result = _frame_queue.get_nowait()
            except queue.Empty:
                result = None

            _stack_dirty = False   # set True when a new frame is added to the stacker

            if result is not None and result.status == GrabStatus.SUCCESS and not state.paused:
                now = time.monotonic()
                fps_display  = 1.0 / max(now - t_last_frame, 1e-6)
                t_last_frame = now
                frame_count += 1
                fdata = result.frame.data

                # Update image rect from actual frame dimensions
                fh, fw = fdata.shape[:2]
                if fw != cam_w_ref[0] or fh != cam_h_ref[0]:
                    cam_w_ref[0] = fw
                    cam_h_ref[0] = fh
                    img_rect = _image_rect(fw, fh, layout["central"])
                    dispatcher.set_img_rect(img_rect)

                if state.focus_state == FocusState.OFF:
                    _cal_ok = result.calibrated

                # Skip expensive display processing while a menu is open —
                # last_surface stays frozen behind the opaque panel.
                if not state.active_menu:
                    if state.focus_state == FocusState.ACTIVE:
                        if _focus_hardware_roi:
                            data8 = stretch_to_uint8(fdata)
                        else:
                            h, w = fdata.shape[:2]
                            cx = int(state.focus_center_x * w)
                            cy = int(state.focus_center_y * h)
                            x1, y1 = max(0, cx - FOCUS_ROI_HALF), max(0, cy - FOCUS_ROI_HALF)
                            x2, y2 = min(w, cx + FOCUS_ROI_HALF), min(h, cy + FOCUS_ROI_HALF)
                            data8 = stretch_to_uint8(fdata[y1:y2, x1:x2])
                        last_surface = pygame.transform.smoothscale(
                            to_surface(data8), (img_rect.width, img_rect.height))
                        _save_frame_ref[0] = data8

                    elif state.mode == ViewMode.LIVE:
                        ae.update(fdata)
                        data8 = stretch_to_uint8(fdata, brightness=state.brightness)
                        last_surface = pygame.transform.smoothscale(
                            to_surface(data8), (img_rect.width, img_rect.height))
                        _save_frame_ref[0] = data8

                else:
                    # Menu open: still run AE so exposure stays current
                    if state.mode == ViewMode.LIVE:
                        ae.update(fdata)

                if state.mode == ViewMode.ACCUMULATE:
                    actual_exp_s = result.frame.meta.exposure_seconds
                    accepted = stacker.add_frame(fdata, actual_exp_s)
                    _stack_dirty = True
                    if state.stack_exposure is None and accepted:
                        stack_seq.advance()

            if state.mode == ViewMode.ACCUMULATE and _stack_dirty and not state.active_menu:
                stacker.process_stack(sky_sub_scale=state.sky_subtraction)

            if state.mode == ViewMode.ACCUMULATE and not state.active_menu:
                display8 = stacker.get_display_frame(brightness=state.brightness)
                if display8 is not None:
                    last_surface = pygame.transform.smoothscale(
                        to_surface(display8), (img_rect.width, img_rect.height))
                    _save_frame_ref[0] = display8

            # -- render -------------------------------------------------------
            screen.fill(BLACK)

            # Central region background
            pygame.draw.rect(screen, BLACK, layout["central"])

            if state.all_sky_mode:
                pass  # black background; sky overlay renders below
            elif last_surface is not None:
                screen.blit(_apply_zoom_pan(last_surface, state), img_rect.topleft)
            else:
                f = _font(14)
                msg = ("No camera connected"
                       if isinstance(cam, NullCamera)
                       else "Waiting for first frame...")
                label = f.render(msg, True, DIM)
                cx = layout["central"].centerx - label.get_width() // 2
                cy = layout["central"].centery - label.get_height() // 2
                screen.blit(label, (cx, cy))

            # Sky / Moon overlay
            if state.overlay_active:
                # Build a cache key from all overlay inputs.
                # compute_overlay is expensive; skip it when the key hasn't
                # changed (mount barely moves between frames) and always skip
                # it when a menu is open (overlay is hidden behind the panel).
                try:
                    if state.all_sky_mode and mount_holder[0] is not None:
                        ra_h, dec_deg = mount_holder[0].position
                        alt_deg, az_deg = radec_to_altaz(ra_h, dec_deg, lat, lon)
                        aspect = img_rect.height / img_rect.width
                        new_key = (
                            "sky",
                            round(alt_deg, 2), round(az_deg, 2),
                            round(state.sky_map_fov, 2),
                            round(state.zoom_center_x, 4),
                            round(state.zoom_center_y, 4),
                            img_rect.width, img_rect.height,
                        )
                        if new_key != _ov_key and not state.active_menu:
                            d_az  = (state.zoom_center_x - 0.5) * state.sky_map_fov
                            d_alt = -(state.zoom_center_y - 0.5) * state.sky_map_fov * aspect
                            ov_arr, ov_table = compute_overlay(
                                catalog,
                                fov_deg     = state.sky_map_fov,
                                alt_deg     = alt_deg + d_alt,
                                az_deg      = az_deg + d_az,
                                image_shape = (img_rect.height, img_rect.width),
                                lat_deg     = lat,
                                lon_deg     = lon,
                                style       = overlay_style,
                                constellation_lines = constellation_lines,
                                cam_fov_h_deg = state.sky_map_cam_fov_h,
                                cam_fov_v_deg = state.sky_map_cam_fov_v,
                                cam_alt_deg   = alt_deg,
                                cam_az_deg    = az_deg,
                            )
                            _ov_surf = pygame.image.frombuffer(
                                ov_arr.tobytes(), (img_rect.width, img_rect.height), "RGBA")
                            _ov_key = new_key

                    elif state.moon_mode:
                        cw, ch = img_rect.width, img_rect.height
                        moon_r_src = (MOON_ANGULAR_RADIUS_DEG / fov_ref[0]) * cw
                        moon_r_px  = moon_r_src * state.zoom_level
                        new_key = (
                            "moon",
                            round(state.zoom_level, 4),
                            round(state.zoom_center_x, 4),
                            round(state.zoom_center_y, 4),
                            cw, ch,
                        )
                        if new_key != _ov_key and not state.active_menu:
                            moon_cx = (0.5 - state.zoom_center_x) * state.zoom_level * cw + cw / 2
                            moon_cy = (0.5 - state.zoom_center_y) * state.zoom_level * ch + ch / 2
                            ov_arr, ov_table = compute_moon_overlay(
                                moon_catalog,
                                moon_cx         = moon_cx,
                                moon_cy         = moon_cy,
                                moon_r          = moon_r_px,
                                image_shape     = (ch, cw),
                                min_diameter_km = max(0.0, 20.0 * (50.0 / max(moon_r_px, 1))),
                            )
                            _ov_surf = pygame.image.frombuffer(
                                ov_arr.tobytes(), (img_rect.width, img_rect.height), "RGBA")
                            _ov_key = new_key

                    elif mount_holder[0] is not None:
                        ra_h, dec_deg = mount_holder[0].position
                        alt_deg, az_deg = radec_to_altaz(ra_h, dec_deg, lat, lon)
                        eff_fov = fov_ref[0] / state.zoom_level
                        aspect  = img_rect.height / img_rect.width
                        new_key = (
                            "normal",
                            round(alt_deg, 2), round(az_deg, 2),
                            round(eff_fov, 3),
                            round(state.zoom_center_x, 4),
                            round(state.zoom_center_y, 4),
                            img_rect.width, img_rect.height,
                        )
                        if new_key != _ov_key and not state.active_menu:
                            d_az  = (state.zoom_center_x - 0.5) * fov_ref[0]
                            d_alt = -(state.zoom_center_y - 0.5) * fov_ref[0] * aspect
                            ov_arr, ov_table = compute_overlay(
                                catalog,
                                fov_deg     = eff_fov,
                                alt_deg     = alt_deg + d_alt,
                                az_deg      = az_deg + d_az,
                                image_shape = (img_rect.height, img_rect.width),
                                lat_deg     = lat,
                                lon_deg     = lon,
                                style       = overlay_style,
                                constellation_lines = constellation_lines,
                            )
                            _ov_surf = pygame.image.frombuffer(
                                ov_arr.tobytes(), (img_rect.width, img_rect.height), "RGBA")
                            _ov_key = new_key

                    else:
                        _ov_surf = None

                except Exception as exc:
                    logging.warning("Overlay error: %s", exc)

                if _ov_surf is not None:
                    screen.blit(_ov_surf, img_rect.topleft)
                _render_constellation_labels(screen, ov_table, img_rect)
                _render_hover_label(screen, ov_table, *cursor_pos, img_rect=img_rect)

            # Menus
            if state.active_menu == "action":
                _render_vertical_menu(screen, action_menu, "left", layout, cursor_pos)
            elif state.active_menu == "utilities":
                _render_vertical_menu(screen, utilities_menu, "right", layout, cursor_pos)
            elif state.active_menu == "controls":
                _render_controls_menu(screen, layout, state, cursor_pos)
            elif state.active_menu == "context" and context_menu_ref[0] is not None:
                _render_context_menu(screen, context_menu_ref[0],
                                     state.context_menu_pos, cursor_pos,
                                     layout)

            # QR overlay (shown after Save)
            if _qr_timer[0] > 0:
                _qr_timer[0] -= dt
                if _qr_surf is not None:
                    _render_qr_overlay(screen, _qr_surf, layout,
                                       _qr_line1, _qr_line2,
                                       _qr_timer[0], _QR_DURATION)

            # Alert overlay
            if alert_timer > 0:
                _render_alert(screen, alert_message, win_w, win_h)

            # Status bars (drawn last so they sit on top)
            if state.focus_state == FocusState.ACTIVE:
                hud_top = (
                    f"FOCUS  exp={ae.current:.4g}s  fps={fps_display:.1f}"
                    f"  — click or any key to exit"
                )
                hud_bot = ""
            elif state.mode == ViewMode.ACCUMULATE:
                hud_top = (
                    f"Stacking  exp={stack_seq.current:.4g}s  t={stacker.t_accum:.1f}s"
                )
                hud_bot = (
                    f"frames={stacker.frame_count}  skipped={stacker.skipped_count}  "
                    f"{cam_config_ref[0].telescope_description}  "
                    f"hfov={fov_ref[0] / state.zoom_level:.1f}°"
                )
            else:
                hud_top = (
                    f"Streaming  exp={ae.current:.4g}s  fps={fps_display:.1f}"
                )
                hud_bot = (
                    f"frames={frame_count}  skipped=0  "
                    f"{cam_config_ref[0].telescope_description}  "
                    f"hfov={fov_ref[0] / state.zoom_level:.1f}°"
                )

            _render_status_bars(
                screen, layout, state,
                hud_top, hud_bot,
                cam_configured = _cam_configured,
                cal_ok         = _cal_ok,
                action_active    = state.active_menu == "action",
                controls_active  = state.active_menu == "controls",
                utilities_active = state.active_menu == "utilities",
            )

            cursor_visible = (
                state.active_menu is not None
                or (state.focus_state == FocusState.OFF
                    and (time.monotonic() - t_last_move) < CURSOR_HIDE_DELAY)
            )
            pygame.mouse.set_visible(cursor_visible)

            pygame.display.flip()

        _stop_grab.set()
        if _grab_thread is not None:
            _grab_thread.join(timeout=5.0)

        if _is_pi and _hotspot_started[0]:
            subprocess.run(["sudo", "/usr/local/bin/astro-hotspot-stop"],
                           check=False, timeout=5.0)

        for _ec in _extra_cams:
            try:
                _ec.disconnect()
            except Exception:
                pass

    pygame.quit()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
