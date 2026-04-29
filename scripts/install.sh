#!/usr/bin/env bash
# install.sh — set up the Python environment for impatient-astronomy-instrumentation
#
# Usage:
#   bash scripts/install.sh
#
# What it does:
#   1. Creates a .venv virtual environment (if not already present)
#   2. Installs all dependencies from requirements.txt
#   3. Installs the astrocore package in editable mode
#   4. On macOS: fixes the libSDL2 duplicate bundled by opencv and pygame

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Virtual environment
# ---------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

PYTHON=".venv/bin/python"
PIP=".venv/bin/pip"

# ---------------------------------------------------------------------------
# 2. Dependencies
# ---------------------------------------------------------------------------
echo "Installing dependencies..."
"$PIP" install --upgrade pip --quiet
"$PIP" install -r requirements.txt --quiet

# ---------------------------------------------------------------------------
# 3. Editable install of astrocore
# ---------------------------------------------------------------------------
echo "Installing astrocore (editable)..."
"$PIP" install -e . --quiet

# ---------------------------------------------------------------------------
# 4. macOS: fix libSDL2 duplicate between opencv and pygame
#
# Both packages bundle libSDL2-2.0.0.dylib. Loading both copies causes noisy
# "Class X is implemented in both..." warnings. Replacing cv2's copy with a
# symlink to pygame's copy means only one binary loads — no warnings.
# ---------------------------------------------------------------------------
if [ "$(uname)" = "Darwin" ]; then
    PYGAME_SDL2="$REPO_ROOT/.venv/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib"
    CV2_SDL2="$REPO_ROOT/.venv/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/cv2/.dylibs/libSDL2-2.0.0.dylib"

    if [ -f "$PYGAME_SDL2" ] && [ -f "$CV2_SDL2" ] && [ ! -L "$CV2_SDL2" ]; then
        echo "Fixing libSDL2 duplicate (macOS)..."
        ln -sf "$PYGAME_SDL2" "$CV2_SDL2"
        echo "  Symlinked cv2's libSDL2 → pygame's libSDL2"
    elif [ -L "$CV2_SDL2" ]; then
        echo "libSDL2 symlink already in place, skipping."
    fi
fi

echo ""
echo "Installation complete. Activate your environment with:"
echo "  source .venv/bin/activate"
