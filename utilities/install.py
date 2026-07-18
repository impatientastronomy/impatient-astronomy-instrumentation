#!/usr/bin/env python3
"""
install.py — set up the digital eyepiece environment on any platform.

Usage:
    python utilities/install.py          # Mac, Linux, Raspberry Pi, Windows

What it does:
    1. Checks that uv is installed and gives platform-specific install instructions if not
    2. Installs all Python dependencies via uv sync
    3. Creates ~/digital_eyepiece/{cals,sessions,images}/
    4. Copies digital_eyepiece/config/configuration-example.yaml to
       digital_eyepiece/config/configuration.yaml if no config exists yet
    5. macOS only: fixes the libSDL2 duplicate bundled by opencv and pygame
    6. Raspberry Pi only: installs the ZWO ASI udev rule and Waveshare display timings
    7. (step removed — USB WiFi adapter no longer needs a pinned name)

Safe to re-run — every step checks whether it has already been applied.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent
PROJECT_NAME = "digital_eyepiece"
DATA_ROOT    = Path.home() / PROJECT_NAME


# ---------------------------------------------------------------------------
# 1. Check for uv
# ---------------------------------------------------------------------------

def _check_uv() -> None:
    if shutil.which("uv") is not None:
        return
    print("Error: uv is not installed.")
    if sys.platform == "win32":
        print("  Install it with winget:")
        print("    winget install --id=astral-sh.uv")
        print("  or in PowerShell:")
        print("    irm https://astral.sh/uv/install.ps1 | iex")
    else:
        print("  Install it with:")
        print("    curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  Then open a new terminal so uv is on your PATH.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------

def _install_deps() -> None:
    print("Installing dependencies...")
    subprocess.run(["uv", "sync"], cwd=REPO_ROOT, check=True)


# ---------------------------------------------------------------------------
# 3 & 4. Data directory setup
# ---------------------------------------------------------------------------

def _setup_data_dirs() -> None:
    print(f"\nSetting up data directory at {DATA_ROOT} ...")
    for subdir in ("cals", "sessions", "images"):
        (DATA_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    config_dst = REPO_ROOT / "digital_eyepiece" / "config" / "configuration.yaml"
    config_src = REPO_ROOT / "digital_eyepiece" / "config" / "configuration-example.yaml"

    if config_dst.exists():
        print("  configuration.yaml already exists — skipping copy.")
    else:
        shutil.copy2(config_src, config_dst)
        print(f"  Created {config_dst}")
        print("  >>> Edit this file before running the eyepiece software. <<<")


# ---------------------------------------------------------------------------
# 5. macOS: fix libSDL2 duplicate between opencv and pygame
#
# Both packages bundle libSDL2-2.0.0.dylib. Loading both copies causes noisy
# "Class X is implemented in both..." warnings. Replacing cv2's copy with a
# symlink to pygame's copy means only one binary loads — no warnings.
# ---------------------------------------------------------------------------

def _fix_macos_sdl2() -> None:
    if sys.platform != "darwin":
        return

    lib_dir = REPO_ROOT / ".venv" / "lib"
    if not lib_dir.exists():
        return

    py_dir = next(
        (d for d in lib_dir.iterdir() if d.name.startswith("python3.")),
        None,
    )
    if py_dir is None:
        return

    site      = py_dir / "site-packages"
    pygame_sdl = site / "pygame" / ".dylibs" / "libSDL2-2.0.0.dylib"
    cv2_sdl    = site / "cv2"    / ".dylibs" / "libSDL2-2.0.0.dylib"

    if not pygame_sdl.exists() or not cv2_sdl.exists():
        return

    if cv2_sdl.is_symlink():
        print("libSDL2 symlink already in place, skipping.")
        return

    print("Fixing libSDL2 duplicate (macOS)...")
    cv2_sdl.unlink()
    cv2_sdl.symlink_to(pygame_sdl)
    print("  Symlinked cv2's libSDL2 → pygame's libSDL2")


# ---------------------------------------------------------------------------
# 6. Raspberry Pi: udev rule + Waveshare display timings
# ---------------------------------------------------------------------------

_DISPLAY_BLOCK = """\

# Waveshare 4inch HDMI LCD (C), 720x720 @ 60 Hz
hdmi_force_hotplug=1
config_hdmi_boost=10
hdmi_group=2
hdmi_mode=87
hdmi_timings=720 0 100 20 100 720 0 20 8 20 0 0 0 60 0 48000000 6
"""


def _setup_pi() -> bool:
    """Returns True if a reboot is required."""
    config_txt = Path("/boot/firmware/config.txt")
    if platform.machine() != "aarch64" or not config_txt.exists():
        return False

    print("\nRaspberry Pi detected — applying hardware configuration...")
    reboot_needed = False

    # ZWO ASI camera udev rule
    rules_src = REPO_ROOT / "astrocore" / "camera" / "asi_sdk" / "lib" / "asi.rules"
    rules_dst = Path("/etc/udev/rules.d/99-asi.rules")

    if rules_dst.exists():
        print("  ZWO ASI udev rule already installed, skipping.")
    else:
        print("  Installing ZWO ASI udev rule (requires sudo)...")
        subprocess.run(["sudo", "cp", str(rules_src), str(rules_dst)], check=True)
        subprocess.run(["sudo", "udevadm", "control", "--reload-rules"], check=True)
        subprocess.run(["sudo", "udevadm", "trigger"], check=True)
        print("  ZWO ASI udev rule installed.")

    # Waveshare display timings
    cfg_text = config_txt.read_text()
    if "Waveshare 4inch HDMI LCD" in cfg_text:
        print(f"  Display config already present in {config_txt}, skipping.")
    else:
        print(f"  Adding Waveshare display settings to {config_txt} (requires sudo)...")
        subprocess.run(
            ["sudo", "tee", "-a", str(config_txt)],
            input=_DISPLAY_BLOCK.encode(),
            stdout=subprocess.DEVNULL,
            check=True,
        )
        print("  Display settings added.")
        reboot_needed = True

    # USB current — allow full 1.6A USB output without USB-C PD negotiation.
    # Required when powering from a non-PD supply (e.g. a buck converter).
    if "usb_max_current_enable=1" in cfg_text:
        print("  USB max current already enabled, skipping.")
    else:
        print(f"  Enabling USB max current in {config_txt} (requires sudo)...")
        subprocess.run(
            ["sudo", "tee", "-a", str(config_txt)],
            input=b"\n# Allow full USB current without USB-C PD negotiation\nusb_max_current_enable=1\n",
            stdout=subprocess.DEVNULL,
            check=True,
        )
        print("  USB max current enabled.")
        reboot_needed = True

    return reboot_needed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _check_uv()
    _install_deps()
    _setup_data_dirs()
    _fix_macos_sdl2()
    reboot_needed = _setup_pi()

    print("\nInstallation complete.")

    if reboot_needed:
        print("\nConfiguration changes were made — reboot before continuing:")
        print("  sudo reboot")
        print("\nAfter rebooting, set up the guest hotspot:")
        print("  sudo bash utilities/setup_hotspot.sh")
        print("\nThen configure your mount to connect to the AstroEye hotspot")
        print("using the ZWO app (Network → Station Mode).")
    else:
        print("\nRun the digital eyepiece with:")
        print("  uv run python -m digital_eyepiece.main")


if __name__ == "__main__":
    main()
