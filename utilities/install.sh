#!/usr/bin/env bash
# install.sh — thin wrapper around the cross-platform Python installer.
# Kept for convenience on Mac and Raspberry Pi; Windows users run directly:
#   python utilities/install.py
exec python3 "$(dirname "$0")/install.py" "$@"
