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
Right-click : context menu
Right-hold  : drag image  (not yet implemented)
Scroll      : zoom
Middle-click: slew mount to cursor (mount connected + overlay visible)
Hover       : show object name when star overlay is active

Press Q or Escape to quit.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import numpy as np
import pygame

import cv2

from astrocore.camera.catalog import scan_folder
from astrocore.camera.frame_grabber import FrameGrabber, GrabStatus
from astrocore.camera.virtual_cam import VirtualCamera
from astrocore.camera.zwo_asi import ZwoAsiCamera, list_cameras
from astrocore.config.camera_config import CameraConfig, Configuration, HotspotConfig, compute_hfov, load
from digital_eyepiece.gallery_server import GalleryServer
from astrocore.display.skyoverlay import compute_overlay, load_catalog
from astrocore.display.moon_mapper import (
    MOON_ANGULAR_RADIUS_DEG, compute_moon_overlay, load_moon_catalog,
)
from astrocore.mount.coord import radec_to_altaz
from astrocore.pipeline.stacker import ExposureSequence, Stacker
from astrocore.pipeline.streaming import StreamExposure
from digital_eyepiece.display import stretch_to_uint8, to_surface
from digital_eyepiece.input.dispatcher import InputDispatcher
from digital_eyepiece.input.menu import Menu, MenuItem
from digital_eyepiece.recorder import Recorder
from digital_eyepiece.view_state import FocusState, ViewMode, ViewState

_CATALOG_PATH      = Path(__file__).resolve().parent.parent / "astrocore" / "mount" / "skyChart.csv"
_MOON_CATALOG_PATH = Path(__file__).resolve().parent.parent / "astrocore" / "mount" / "moon_features.csv"

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

FOCUS_ROI_HALF = 100

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


# ---------------------------------------------------------------------------
# Menu builders
# ---------------------------------------------------------------------------

def _build_action_menu(
    state: ViewState,
    on_connect,
    on_disconnect,
    on_start_focus,
    telescope_configs: list[tuple[str, str]],
    on_select_config,
    active_desc_fn,
    recorder,
    on_save,
    on_quit,
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

    if len(telescope_configs) > 1:
        scope_items = [
            MenuItem(desc, action=lambda cn=cn: on_select_config(cn))
            for cn, desc in telescope_configs
        ]
        scope_items.append(MenuItem("Back"))
        scope_entry = MenuItem(active_desc_fn, submenu=scope_items)
    else:
        scope_entry = MenuItem(active_desc_fn)

    telescope_submenu = [
        scope_entry,
        MenuItem("Connect",    action=on_connect),
        MenuItem("Disconnect", action=on_disconnect),
        MenuItem("Park",       action=lambda: None),
        MenuItem("Back"),
    ]

    m = Menu()
    m.add(MenuItem(_mode_label,   action=_toggle_mode))
    m.add(MenuItem(_pause_label,  action=_toggle_pause))
    m.add(MenuItem("Telescope",   submenu=telescope_submenu))
    m.add(MenuItem(_record_label, action=_toggle_record))
    m.add(MenuItem("Save",        action=on_save))
    m.add(MenuItem("Focus",       action=on_start_focus))
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
    mount_connected: bool,
) -> Menu:
    m = Menu()
    if near_object:
        m.add(MenuItem(f"About {object_name}", action=lambda: None))
    m.add(MenuItem("Focus", action=on_focus))
    m.add(MenuItem("Slew here", action=on_slew if mount_connected else None))
    if mount_connected:
        m.add(MenuItem("SkyMap", action=lambda: None))
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


# --- Controls panel (horizontal value selectors) ----------------------------

_CTRL_ROW_H    = 30
_CTRL_LABEL_W  = 110
_CTRL_VAL_W    = 36
_CTRL_PAD      = 5

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


def _render_controls_menu(
    surface: pygame.Surface,
    layout: dict,
    state: ViewState,
    cursor_pos: tuple[int, int],
) -> None:
    central = layout["central"]
    pr = _controls_panel_rect(layout)

    # Dim upper part of image
    overlay = pygame.Surface((central.width, central.height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (central.x, central.y))

    panel = pygame.Surface((pr.width, pr.height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 220))
    surface.blit(panel, (pr.x, pr.y))
    pygame.draw.rect(surface, DIM, pr, 1)

    f_lbl  = _font(max(9, _CTRL_ROW_H - 18))
    f_val  = _font(max(9, _CTRL_ROW_H - 20))
    mx, my = cursor_pos

    for ri, (label, attr, steps) in enumerate(_CONTROLS_ROWS):
        ry = pr.y + _CTRL_PAD + ri * _CTRL_ROW_H
        current = getattr(state, attr)

        # Row label
        lbl = f_lbl.render(label + ":", True, GREY)
        surface.blit(lbl, (pr.x + _CTRL_PAD, ry + (_CTRL_ROW_H - lbl.get_height()) // 2))

        # Value chips
        vx = pr.x + _CTRL_LABEL_W
        for vi, (vtext, vval) in enumerate(steps):
            chip = pygame.Rect(vx, ry + _CTRL_PAD, _CTRL_VAL_W - 2, _CTRL_ROW_H - 2 * _CTRL_PAD)
            selected = (current == vval)
            hovered  = chip.collidepoint(mx, my)

            bg = (60, 60, 60) if hovered else (30, 30, 30) if selected else BLACK
            pygame.draw.rect(surface, bg, chip)
            if selected or hovered:
                pygame.draw.rect(surface, GREY if selected else DIM, chip, 1)

            color  = WHITE if selected else GREY
            vlabel = f_val.render(vtext, True, color)
            surface.blit(vlabel, (
                chip.x + (chip.width  - vlabel.get_width())  // 2,
                chip.y + (chip.height - vlabel.get_height()) // 2,
            ))
            vx += _CTRL_VAL_W


def _controls_hit(
    pos: tuple[int, int],
    layout: dict,
) -> tuple[str, object] | None:
    """
    Return (attr_name, new_value) if pos is over a Controls panel value chip.
    Returns None if pos is outside the panel.
    """
    pr = _controls_panel_rect(layout)
    mx, my = pos
    if not pr.collidepoint(mx, my):
        return None

    ri = (my - pr.y - _CTRL_PAD) // _CTRL_ROW_H
    if ri < 0 or ri >= len(_CONTROLS_ROWS):
        return None
    _label, attr, steps = _CONTROLS_ROWS[ri]

    # Find which value chip was clicked
    vx = pr.x + _CTRL_LABEL_W
    for vi, (vtext, vval) in enumerate(steps):
        chip = pygame.Rect(vx, pr.y + _CTRL_PAD + ri * _CTRL_ROW_H,
                           _CTRL_VAL_W - 2, _CTRL_ROW_H - 2 * _CTRL_PAD)
        if chip.collidepoint(mx, my):
            return (attr, vval)
        vx += _CTRL_VAL_W
    return None


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

def _render_hover_label(
    surface: pygame.Surface,
    table: list[dict],
    cursor_x: int,
    cursor_y: int,
    threshold: int = 25,
) -> None:
    if not table:
        return
    best, best_d2 = None, threshold * threshold
    for entry in table:
        d2 = (entry["px"] - cursor_x) ** 2 + (entry["py"] - cursor_y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = entry
    if best is None:
        return

    name = best["name"] or best["type"]
    text = f"{name}  mag {best['mag']:.1f}  {best['type']}"
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
    hotspot: HotspotConfig,
    timer: float,
    total: float,
) -> None:
    """Draw the QR code + caption centred in the central region."""
    central = layout["central"]
    qs = qr_surf.get_width()
    padding = 12

    f = _font(11)
    line1 = f.render(f"Join WiFi: {hotspot.ssid}  •  pw: {hotspot.password}", True, WHITE)
    line2 = f.render("Then open your browser — gallery loads automatically", True, GREY)

    panel_w = max(qs + 2 * padding, line1.get_width() + 2 * padding)
    panel_h = qs + line1.get_height() + line2.get_height() + 4 * padding

    px = central.centerx - panel_w // 2
    py = central.centery - panel_h // 2

    # Fade out in the last second
    alpha = int(220 * min(1.0, timer))
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, alpha))
    surface.blit(panel, (px, py))

    qx = px + (panel_w - qs) // 2
    qy = py + padding
    surface.blit(qr_surf, (qx, qy))
    surface.blit(line1, (px + (panel_w - line1.get_width()) // 2, qy + qs + padding))
    surface.blit(line2, (px + (panel_w - line2.get_width()) // 2,
                          qy + qs + padding + line1.get_height() + 4))


def _apply_brightness(image: np.ndarray, brightness: float) -> np.ndarray:
    if brightness == 1.0:
        return image
    return np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exposure", type=float, default=None,
                   help="Override streaming start exposure in seconds")
    p.add_argument("--vcam", metavar="SUBFOLDER",
                   help="Replay a recorded session instead of live hardware.")
    p.add_argument("--vcam-accel", type=float, default=1.0, metavar="FACTOR",
                   help="Time-acceleration for vcam playback (default: 1.0)")
    p.add_argument("--fullscreen", action="store_true",
                   help="Run fullscreen at the native display resolution (default on Pi).")
    p.add_argument("--window-size", metavar="WxH", default=None,
                   help="Windowed size override, e.g. 1280x720. "
                        "Defaults to the display's current resolution.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

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
            logging.warning("No ZWO ASI cameras found.")
            sys.exit(1)
        preferred_id = config.preferred_camera_id if config else None
        usb_index = cameras[0]["usb_index"]
        for c in cameras:
            if c.get("camera_id") == preferred_id:
                usb_index = c["usb_index"]
                break
        cam_ctx = ZwoAsiCamera(index=usb_index)

    catalog      = load_catalog(str(_CATALOG_PATH))
    moon_catalog = load_moon_catalog(str(_MOON_CATALOG_PATH))

    # -- pygame init ---------------------------------------------------------
    pygame.init()

    info = pygame.display.Info()
    screen_w = info.current_w if info.current_w > 0 else WINDOW_W
    screen_h = info.current_h if info.current_h > 0 else WINDOW_H

    if args.window_size:
        try:
            win_w, win_h = (int(v) for v in args.window_size.lower().split("x"))
        except ValueError:
            logging.error("--window-size must be WxH, e.g. 1280x720")
            sys.exit(1)
    elif args.fullscreen:
        win_w, win_h = screen_w, screen_h
    else:
        # In windowed mode the OS menu bar / notch / dock eat into the available
        # height.  Reserve a platform-appropriate margin so the window fits.
        if sys.platform == "darwin":
            margin_h = 90   # menu bar (~38 px) + notch headroom + dock
        elif sys.platform == "win32":
            margin_h = 48   # taskbar
        else:
            margin_h = 0
        win_w = min(WINDOW_W, screen_w)
        win_h = min(WINDOW_H, screen_h - margin_h)

    if args.fullscreen:
        screen = pygame.display.set_mode((win_w, win_h),
                                         pygame.FULLSCREEN | pygame.NOFRAME)
    else:
        screen = pygame.display.set_mode((win_w, win_h))

    pygame.display.set_caption(WINDOW_TITLE)
    pygame.mouse.set_visible(True)
    clock = pygame.time.Clock()

    layout = _make_layout(win_w, win_h)

    state = ViewState()

    cam_config_ref: list[CameraConfig] = [CameraConfig()]
    fov_ref:        list[float]        = [30.0]
    mount_holder:   list               = [None]

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
        driver = cam_config_ref[0].mount_driver
        if not driver:
            return
        try:
            mod = importlib.import_module(f"astrocore.mount.{driver}")
            cls = getattr(mod, "Driver", None)
            if cls is None:
                return
            mount_holder[0] = cls()
            state.mount_connected = True
        except Exception as exc:
            logging.warning("Mount connect failed: %s", exc)

    def _disconnect_mount() -> None:
        if mount_holder[0] is not None:
            mount_holder[0].disconnect()
            mount_holder[0] = None
        state.mount_connected = False

    with cam_ctx as cam:
        _cam_configured = config is not None and cam.info.camera_id in config.camera_ids()
        if not _cam_configured:
            id_note = " (EEPROM not set)" if cam.info.camera_id == 0 else ""
            logging.warning("Camera ID %d%s not in configuration.yaml", cam.info.camera_id, id_note)

        if config is not None:
            cam_config = config.get_config(cam.info.camera_id)
            cam_config_ref[0] = cam_config
            if cam_config.bin > 1:
                cam.bin = cam_config.bin
            if cam_config.gain is not None:
                cam.gain = cam_config.gain
            if cam_config.data_offset is not None:
                cam.offset = cam_config.data_offset
            if cam_config.pattern is not None:
                from dataclasses import replace as _dc_replace
                cam._info = _dc_replace(cam.info, bayer_pattern=cam_config.pattern)
            cam.meta.telescope_description = cam_config.telescope_description
            cam.meta.focal_length_mm       = cam_config.focal_length_mm
            cam.meta.Lat = lat
            cam.meta.Lon = lon
            roi_w = cam_config.effective_roi(
                cam.info.sensor_width_px, cam.info.sensor_height_px)[2]
            fov_ref[0] = compute_hfov(
                cam_config.focal_length_mm, cam.info.pixel_size_um, roi_w,
            ) or fov_ref[0]

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

        def _on_select_config(config_name: str) -> None:
            if config is None:
                return
            new_cfg = config.get_config(cam.info.camera_id, config_name)
            cam_config_ref[0] = new_cfg
            roi_w = new_cfg.effective_roi(
                cam.info.sensor_width_px, cam.info.sensor_height_px)[2]
            fov_ref[0] = compute_hfov(
                new_cfg.focal_length_mm, cam.info.pixel_size_um, roi_w) or fov_ref[0]
            cam.meta.telescope_description = new_cfg.telescope_description
            cam.meta.focal_length_mm       = new_cfg.focal_length_mm
            if new_cfg.gain is not None:
                cam.gain = new_cfg.gain

        telescope_configs = (
            config.telescope_descriptions(cam.info.camera_id) if config else []
        )

        def _start_focus_mode() -> None:
            if state.recording and recorder is not None:
                state.recording = False
                recorder.stop()
            state.focus_state = FocusState.WAITING
            state.mode        = ViewMode.LIVE
            _close_menu()

        # -- image save / gallery setup --------------------------------------
        image_path: Path | None = config.image_path if config else None
        hotspot: HotspotConfig  = config.hotspot    if config else HotspotConfig()

        if image_path is not None:
            image_path.mkdir(parents=True, exist_ok=True)
            GalleryServer(image_path, port=hotspot.port).start()

        # QR surface built once from hotspot credentials (static)
        _qr_surf: pygame.Surface | None = _make_qr_surface(hotspot.wifi_qr_data)
        _qr_timer: list[float] = [0.0]   # seconds remaining for QR overlay
        _QR_DURATION = 10.0

        def _on_save() -> None:
            if image_path is None:
                logging.warning("image_path not set in configuration.yaml — cannot save")
                _close_menu()
                return
            ts   = time.strftime("%Y-%m-%d_%H-%M-%S")
            path = image_path / f"{ts}.jpg"
            # Capture the current screen (includes status bars)
            pixels = pygame.surfarray.array3d(screen)   # (W, H, 3) RGB
            pixels = pixels.swapaxes(0, 1)              # → (H, W, 3)
            bgr    = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            logging.info("Saved %s", path)
            _qr_timer[0] = _QR_DURATION
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
            on_connect       = _connect_mount,
            on_disconnect    = _disconnect_mount,
            on_start_focus   = _start_focus_mode,
            telescope_configs = telescope_configs,
            on_select_config  = _on_select_config,
            active_desc_fn    = lambda: cam_config_ref[0].telescope_description or "Telescope",
            recorder         = recorder,
            on_save          = _on_save,
            on_quit          = _on_quit,
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
            zoom_step = 1.2,
            zoom_min  = 1.0,
            zoom_max  = config.max_zoom if config else 5.0,
        )

        actual_gain = cam.gain
        ae        = StreamExposure(start=args.exposure if args.exposure is not None else 0.1)
        stacker   = Stacker()
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
        _grab_thread = _make_grab_worker(_stop_grab)
        _grab_thread.start()

        _focus_roi_saved: tuple[int, int, int, int] | None = None

        def _stop_and_reset_grab() -> None:
            _stop_grab.set()
            _grab_thread.join(timeout=5.0)
            grabber.reset()

        def _restart_grab() -> None:
            nonlocal _stop_grab, _grab_thread
            _stop_grab  = threading.Event()
            _grab_thread = _make_grab_worker(_stop_grab)
            _grab_thread.start()

        def _enter_focus_active() -> None:
            nonlocal _focus_roi_saved, _focus_hardware_roi
            if not isinstance(cam, ZwoAsiCamera):
                return
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
                        if state.active_menu:
                            _close_menu()
                        elif state.focus_state != FocusState.OFF:
                            _exit_focus()
                        else:
                            running = False
                    elif state.focus_state != FocusState.OFF:
                        _exit_focus()
                    elif event.key == pygame.K_f:
                        if state.recording and recorder is not None:
                            state.recording = False
                            recorder.stop()
                        state.focus_state = FocusState.WAITING
                        state.mode        = ViewMode.LIVE
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
                    dispatcher.on_mouse_move(*event.pos, right_held=bool(event.buttons[2]))

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos

                    if state.focus_state == FocusState.WAITING:
                        state.focus_center_x = mx / win_w
                        state.focus_center_y = my / win_h
                        _enter_focus_active()
                        state.focus_state = FocusState.ACTIVE

                    elif state.focus_state == FocusState.ACTIVE:
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
                                    ctx.select()
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

                    elif event.button == 2:
                        if dispatcher.on_middle_click():
                            alert_timer = ALERT_DURATION

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3 and dispatcher.on_right_button_up():
                        # was a click (not a drag) — open context menu
                        if not state.active_menu:
                            mx, my = event.pos
                            near = None
                            for entry in ov_table:
                                d2 = (entry["px"] - mx) ** 2 + (entry["py"] - my) ** 2
                                if d2 < 25 * 25:
                                    near = entry
                                    break

                            def _slew_to_cursor() -> None:
                                if mount_holder[0] is not None:
                                    alert_timer = ALERT_DURATION

                            context_menu_ref[0] = _build_context_menu(
                                near_object     = near is not None,
                                object_name     = near["name"] if near else "",
                                on_focus        = _start_focus_mode,
                                on_slew         = _slew_to_cursor,
                                mount_connected = state.mount_connected,
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

                elif state.mode == ViewMode.LIVE:
                    ae.update(fdata)
                    data8 = stretch_to_uint8(fdata)
                    data8 = _apply_brightness(data8, state.brightness)
                    last_surface = pygame.transform.smoothscale(
                        to_surface(data8), (img_rect.width, img_rect.height))

                else:
                    stacker.add_frame(fdata, stack_seq.current)
                    if state.stack_exposure is None:
                        stack_seq.advance()

            if state.mode == ViewMode.ACCUMULATE:
                display8 = stacker.get_display_frame(sky_sub_scale=state.sky_subtraction)
                if display8 is not None:
                    display8     = _apply_brightness(display8, state.brightness)
                    last_surface = pygame.transform.smoothscale(
                        to_surface(display8), (img_rect.width, img_rect.height))

            # -- render -------------------------------------------------------
            screen.fill(BLACK)

            # Central region background
            pygame.draw.rect(screen, BLACK, layout["central"])

            if last_surface is not None:
                screen.blit(_apply_zoom_pan(last_surface, state), img_rect.topleft)
                if state.all_sky_mode:
                    dark = pygame.Surface((img_rect.width, img_rect.height), pygame.SRCALPHA)
                    dark.fill((0, 0, 0, 180))
                    screen.blit(dark, img_rect.topleft)
            else:
                f = _font(14)
                label = f.render("Waiting for first frame...", True, DIM)
                cx = layout["central"].centerx - label.get_width() // 2
                cy = layout["central"].centery - label.get_height() // 2
                screen.blit(label, (cx, cy))

            # Sky / Moon overlay
            if state.overlay_active:
                ov_table: list[dict] = []
                try:
                    if state.moon_mode:
                        # Moon feature overlay — disk assumed centred in the source image
                        cw, ch = img_rect.width, img_rect.height
                        moon_r_src = (MOON_ANGULAR_RADIUS_DEG / fov_ref[0]) * cw
                        moon_r_px  = moon_r_src * state.zoom_level
                        moon_cx = (0.5 - state.zoom_center_x) * state.zoom_level * cw + cw / 2
                        moon_cy = (0.5 - state.zoom_center_y) * state.zoom_level * ch + ch / 2
                        ov_arr, ov_table = compute_moon_overlay(
                            moon_catalog,
                            moon_cx      = moon_cx,
                            moon_cy      = moon_cy,
                            moon_r       = moon_r_px,
                            image_shape  = (ch, cw),
                            min_diameter_km = max(0.0, 20.0 * (50.0 / max(moon_r_px, 1))),
                        )
                    elif mount_holder[0] is not None:
                        ra_h, dec_deg = mount_holder[0].position
                        alt_deg, az_deg = radec_to_altaz(ra_h, dec_deg, lat, lon)
                        eff_fov = fov_ref[0] / state.zoom_level
                        aspect  = img_rect.height / img_rect.width
                        d_az    = (state.zoom_center_x - 0.5) * fov_ref[0]
                        d_alt   = -(state.zoom_center_y - 0.5) * fov_ref[0] * aspect
                        ov_arr, ov_table = compute_overlay(
                            catalog,
                            fov_deg     = eff_fov,
                            alt_deg     = alt_deg + d_alt,
                            az_deg      = az_deg + d_az,
                            image_shape = (img_rect.height, img_rect.width),
                            lat_deg     = lat,
                            lon_deg     = lon,
                        )
                    else:
                        ov_arr = None
                    if ov_arr is not None:
                        ov_surf = pygame.image.frombuffer(
                            ov_arr.tobytes(), (img_rect.width, img_rect.height), "RGBA")
                        screen.blit(ov_surf, img_rect.topleft)
                except Exception as exc:
                    logging.warning("Overlay error: %s", exc)
                _render_hover_label(screen, ov_table, *cursor_pos)

            # Focus waiting crosshair
            if state.focus_state == FocusState.WAITING:
                _render_focus_waiting(screen, win_w, layout["central"].y)
                _draw_cursor(screen, *cursor_pos)

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
                    _render_qr_overlay(screen, _qr_surf, layout, hotspot,
                                       _qr_timer[0], _QR_DURATION)

            # Alert overlay
            if alert_timer > 0:
                _render_alert(screen, "Caution: Mount is moving", win_w, win_h)

            # Status bars (drawn last so they sit on top)
            if state.focus_state == FocusState.ACTIVE:
                hud_top = (
                    f"FOCUS  exp={ae.current:.4g}s  gain={actual_gain}  "
                    f"fps={fps_display:.1f}  — click or any key to exit"
                )
                hud_bot = ""
            elif state.mode == ViewMode.ACCUMULATE:
                hud_top = (
                    f"STACK  exp={stack_seq.current:.4g}s  gain={actual_gain}"
                )
                hud_bot = (
                    f"frames={stacker.frame_count}  skip={stacker.skipped_count}  "
                    f"t={stacker.t_accum:.1f}s  "
                    f"{cam_config_ref[0].telescope_description}"
                )
            else:
                hud_top = (
                    f"LIVE  exp={ae.current:.4g}s  gain={actual_gain}  "
                    f"fps={fps_display:.1f}"
                )
                hud_bot = (
                    f"frames={frame_count}  "
                    f"{cam_config_ref[0].telescope_description}"
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
                or state.focus_state == FocusState.WAITING
                or (state.focus_state == FocusState.OFF
                    and (time.monotonic() - t_last_move) < CURSOR_HIDE_DELAY)
            )
            pygame.mouse.set_visible(cursor_visible)

            pygame.display.flip()

        _stop_grab.set()
        _grab_thread.join(timeout=5.0)

    pygame.quit()


if __name__ == "__main__":
    main()
