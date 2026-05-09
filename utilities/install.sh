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

echo ""
echo "Installation complete. Run the digital eyepiece with:"
echo "  uv run python digital_eyepiece/main.py"
