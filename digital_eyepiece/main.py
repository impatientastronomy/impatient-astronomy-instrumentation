"""
Digital eyepiece — live viewer.

Connects to the first ZWO ASI camera found, streams RAW16 frames, and
displays them in a pygame window with percentile stretching.

Usage::

    python -m digital_eyepiece.main [--exposure SECONDS] [--gain VALUE]

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
import logging
import sys
import time

import numpy as np
import pygame

from astrocore.camera.frame_grabber import FrameGrabber, GrabStatus
from astrocore.camera.zwo_asi import ZwoAsiCamera, list_cameras
from astrocore.config import camera_config as _cam_cfg
from digital_eyepiece.display import stretch_to_uint8, to_surface
from digital_eyepiece.input.dispatcher import InputDispatcher
from digital_eyepiece.input.menu import Menu, MenuItem
from digital_eyepiece.view_state import ViewMode, ViewState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_TITLE      = "Digital Eyepiece"
TARGET_FPS        = 30
CURSOR_HIDE_DELAY = 5.0    # seconds before cursor auto-hides
ALERT_DURATION    = 3.0    # seconds the mount-slew alert is shown

GREEN  = (0, 220, 0)
WHITE  = (220, 220, 220)
DIM    = (140, 140, 140)
RED    = (220, 60, 60)
BLACK  = (0, 0, 0)


# ---------------------------------------------------------------------------
# Menu construction
# ---------------------------------------------------------------------------

def _build_menu(state: ViewState) -> Menu:
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

    def _mount_label() -> str:
        return "Disconnect Mount" if state.mount_connected else "Connect Mount"

    def _toggle_mount() -> None:
        state.mount_connected = not state.mount_connected   # placeholder

    menu = Menu()
    menu.add(MenuItem("Cancel"))
    menu.add(MenuItem("Brightness", submenu=brightness_submenu))
    menu.add(MenuItem(_mode_label, action=_toggle_mode))
    menu.add(MenuItem(_mount_label, action=_toggle_mount))
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

    # Semi-transparent dark background
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    font = pygame.font.SysFont("monospace", 30)
    items = menu.current_items
    sel   = menu.selection_index
    row_h = 44
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


def _apply_brightness(image: np.ndarray, brightness: float) -> np.ndarray:
    if brightness == 1.0:
        return image
    return np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exposure", type=float, default=1.0,
                   help="Exposure time in seconds (default: 1.0)")
    p.add_argument("--gain", type=int, default=None,
                   help="Camera gain (default: from cameras.yaml)")
    p.add_argument("--width", type=int, default=1280,
                   help="Window width in pixels (default: 1280)")
    p.add_argument("--height", type=int, default=960,
                   help="Window height in pixels (default: 960)")
    p.add_argument("--bin", type=int, default=2, choices=[1, 2, 3, 4],
                   help="Camera binning factor (default: 2)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    cameras = list_cameras()
    if not cameras:
        logging.warning("No ZWO ASI cameras found. Is the camera plugged in?")
        sys.exit(1)
    usb_index = cameras[0]["usb_index"]

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(WINDOW_TITLE)
    pygame.mouse.set_visible(False)   # we draw our own cursor
    clock = pygame.time.Clock()

    state      = ViewState()
    menu       = _build_menu(state)
    dispatcher = InputDispatcher(state, menu, zoom_step=1.2, zoom_min=1.0, zoom_max=8.0)

    with ZwoAsiCamera(index=usb_index) as cam:
        # Set binning first — SDK resets exposure when binning changes
        if args.bin > 1:
            cam.bin = args.bin
        cam.exposure_time = args.exposure
        if args.gain is not None:
            cam.gain = args.gain

        grabber = FrameGrabber(cam)
        grabber.pattern = cam.info.bayer_pattern
        try:
            cfg = _cam_cfg.load()
            if cfg.cal_path is not None:
                grabber.cal_path = cfg.cal_path
        except FileNotFoundError:
            pass

        exposure_us = int(args.exposure * 1_000_000)
        actual_gain = cam.gain

        last_surface: pygame.Surface | None = None
        frame_count  = 0
        fps_display  = 0.0
        t_last_frame = time.monotonic()

        # Cursor auto-hide
        t_last_move  = time.monotonic()
        cursor_pos   = (args.width // 2, args.height // 2)

        # Mount alert
        alert_timer  = 0.0

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

                elif event.type == pygame.MOUSEMOTION:
                    cursor_pos  = event.pos
                    t_last_move = time.monotonic()
                    dispatcher.on_mouse_move(*event.pos)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dispatcher.on_left_click()
                    elif event.button == 2:
                        if dispatcher.on_middle_click():
                            alert_timer = ALERT_DURATION
                    elif event.button == 3:
                        dispatcher.on_right_click()

                elif event.type == pygame.MOUSEWHEEL:
                    dispatcher.on_scroll(event.y)

            # -- grab frame ---------------------------------------------------
            result = grabber.grab_frame(
                exposure_us=exposure_us,
                dark=grabber.cal_path is not None,
                flat=grabber.cal_path is not None,
                dpc=False,
                demosaic=grabber.pattern is not None,
            )

            if result.status == GrabStatus.SUCCESS:
                now = time.monotonic()
                fps_display  = 1.0 / max(now - t_last_frame, 1e-6)
                t_last_frame = now
                frame_count += 1

                data8 = stretch_to_uint8(result.frame.data)
                data8 = _apply_brightness(data8, state.brightness)
                img_surface  = to_surface(data8)
                last_surface = pygame.transform.smoothscale(
                    img_surface, (args.width, args.height)
                )

            # -- render -------------------------------------------------------
            if last_surface is not None:
                screen.blit(last_surface, (0, 0))

                # All-sky mode: darken camera image
                if state.all_sky_mode:
                    dark = pygame.Surface((args.width, args.height), pygame.SRCALPHA)
                    dark.fill((0, 0, 0, 180))
                    screen.blit(dark, (0, 0))
            else:
                screen.fill(BLACK)
                font  = pygame.font.SysFont("monospace", 24)
                label = font.render("Waiting for first frame...", True, DIM)
                screen.blit(label, (args.width // 2 - 150, args.height // 2))

            # HUD
            mode_str = "STACK" if state.mode == ViewMode.ACCUMULATE else "LIVE"
            hud = (
                f"{mode_str}  exp={args.exposure:.2f}s  "
                f"gain={actual_gain}  bin={args.bin}×{args.bin}  "
                f"fps={fps_display:.1f}  frames={frame_count}"
            )
            _render_hud(screen, hud)

            # Menu
            if state.menu_open:
                _render_menu(screen, menu)

            # Mount alert
            if alert_timer > 0:
                _render_alert(screen, "Caution: Mount is moving")

            # Cursor: visible when mouse moved recently and menu is not open
            cursor_visible = (
                not state.menu_open
                and (time.monotonic() - t_last_move) < CURSOR_HIDE_DELAY
            )
            if cursor_visible:
                _draw_cursor(screen, *cursor_pos)

            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
