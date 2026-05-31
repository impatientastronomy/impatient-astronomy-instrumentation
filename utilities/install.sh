#!/usr/bin/env bash
# install.sh — set up the Python environment for impatient-astronomy-instrumentation
#
# Usage:
#   bash utilities/install.sh
#
# What it does:
#   1. Installs all dependencies and astrocore (editable) via uv
#   2. On macOS: fixes the libSDL2 duplicate bundled by opencv and pygame

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Check for uv
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Install dependencies and astrocore
#    uv sync installs all dependencies from pyproject.toml + uv.lock and
#    installs astrocore itself in editable mode.
# ---------------------------------------------------------------------------
echo "Installing dependencies..."
uv sync

# ---------------------------------------------------------------------------
# 3. macOS: fix libSDL2 duplicate between opencv and pygame
#
# Both packages bundle libSDL2-2.0.0.dylib. Loading both copies causes noisy
# "Class X is implemented in both..." warnings. Replacing cv2's copy with a
# symlink to pygame's copy means only one binary loads — no warnings.
# ---------------------------------------------------------------------------
if [ "$(uname)" = "Darwin" ]; then
    PY_VER="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    PYGAME_SDL2="$REPO_ROOT/.venv/lib/python${PY_VER}/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib"
    CV2_SDL2="$REPO_ROOT/.venv/lib/python${PY_VER}/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib"

    if [ -f "$PYGAME_SDL2" ] && [ -f "$CV2_SDL2" ] && [ ! -L "$CV2_SDL2" ]; then
        echo "Fixing libSDL2 duplicate (macOS)..."
        ln -sf "$PYGAME_SDL2" "$CV2_SDL2"
        echo "  Symlinked cv2's libSDL2 → pygame's libSDL2"
    elif [ -L "$CV2_SDL2" ]; then
        echo "libSDL2 symlink already in place, skipping."
    fi
fi

# ---------------------------------------------------------------------------
# 4. Raspberry Pi: udev rule + display config (Linux/aarch64 only)
# ---------------------------------------------------------------------------
REBOOT_NEEDED=0

if [ "$(uname -m)" = "aarch64" ] && [ -f /boot/firmware/config.txt ]; then
    echo ""
    echo "Raspberry Pi detected — applying hardware configuration..."

    # ZWO ASI camera udev rule
    RULES_SRC="$REPO_ROOT/astrocore/camera/asi_sdk/lib/asi.rules"
    RULES_DST="/etc/udev/rules.d/99-asi.rules"
    if [ -f "$RULES_DST" ]; then
        echo "  ZWO ASI udev rule already installed, skipping."
    else
        echo "  Installing ZWO ASI udev rule (requires sudo)..."
        sudo cp "$RULES_SRC" "$RULES_DST"
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        echo "  ZWO ASI udev rule installed."
    fi

    # Waveshare 4inch HDMI LCD (C) display timings
    CONFIG=/boot/firmware/config.txt
    if grep -q "Waveshare 4inch HDMI LCD" "$CONFIG"; then
        echo "  Display config already present in $CONFIG, skipping."
    else
        echo "  Adding Waveshare display settings to $CONFIG (requires sudo)..."
        sudo tee -a "$CONFIG" > /dev/null << 'DISPLAY_EOF'

# Waveshare 4inch HDMI LCD (C), 720x720 @ 60 Hz
hdmi_force_hotplug=1
config_hdmi_boost=10
hdmi_group=2
hdmi_mode=87
hdmi_timings=720 0 100 20 100 720 0 20 8 20 0 0 0 60 0 48000000 6
DISPLAY_EOF
        echo "  Display settings added."
        REBOOT_NEEDED=1
    fi
fi

echo ""
echo "Installation complete. Run the digital eyepiece with:"
echo "  uv run python -m digital_eyepiece.main"

if [ "$REBOOT_NEEDED" -eq 1 ]; then
    echo ""
    echo "Display settings were added to /boot/firmware/config.txt."
    echo "Reboot before running the eyepiece:  sudo reboot"
fi
