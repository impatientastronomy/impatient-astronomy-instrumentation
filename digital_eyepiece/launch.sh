#!/bin/bash
# Desktop-icon launcher for the digital eyepiece app.
#
# Logs stdout/stderr to ~/digital_eyepiece/launch.log rather than a terminal,
# since the app is meant to be started by double-clicking a desktop icon in
# the field with no keyboard attached -- if something goes wrong, the log
# can be read later (e.g. over SSH) instead of needing to see it live.
#
# Uses the .venv Python directly rather than `uv run`: a double-clicked
# desktop icon isn't guaranteed to inherit a shell environment where `uv`
# is on PATH, and this also skips uv's environment-resolution check on
# every launch (same reasoning as the boot-autostart entry in pi_setup.md).

cd "$(dirname "$0")/.." || exit 1
mkdir -p ~/digital_eyepiece
exec .venv/bin/python -m digital_eyepiece.main >> ~/digital_eyepiece/launch.log 2>&1
