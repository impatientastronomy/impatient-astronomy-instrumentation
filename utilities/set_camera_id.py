#!/usr/bin/env python3
"""
set_camera_id.py  —  Write a user-assigned ID to a ZWO ASI camera EEPROM.

The ID (1–8) is stored in the camera's EEPROM and survives power cycles and
USB reconnects.  Each physical camera should be assigned a unique ID once so
the software can identify it reliably regardless of USB port order.

When the new ID is not already in configuration.yaml, a default camera block
is appended to the file so the application has a starting configuration.
Use --force to reassign an ID that already appears in configuration.yaml;
in that case the YAML is not modified.

Usage
-----
    python set_camera_id.py <new_id>
    python set_camera_id.py <new_id> --index N     # by USB index (see camera list)
    python set_camera_id.py <new_id> --force       # reassign a taken ID; skip YAML
    python set_camera_id.py <new_id> --config PATH # custom config file
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

# Allow running from the repo root without installing astrocore.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astrocore.camera.zwo_asi import ZwoAsiCamera, list_cameras

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configuration.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("new_id", type=int,
                   help="Camera ID to write to EEPROM (1–8)")
    p.add_argument("--index", type=int, default=0, metavar="N",
                   help="USB index of the target camera (default: 0). "
                        "The camera list printed at startup shows each index.")
    p.add_argument("--force", action="store_true",
                   help="Write even if new_id is already in configuration.yaml. "
                        "The YAML is not modified when --force is used.")
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG, metavar="PATH",
                   help="Path to configuration.yaml "
                        f"(default: {_DEFAULT_CONFIG.name})")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _print_cameras(cameras: list[dict]) -> None:
    print(f"Connected cameras ({len(cameras)}):")
    for c in cameras:
        id_note = "  ← EEPROM not set" if c["camera_id"] == 0 else ""
        print(f"  USB {c['usb_index']}:  {c['model']}  ID={c['camera_id']}{id_note}")


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_raw(config_path: Path) -> dict:
    """Load configuration.yaml as a raw dict; return {} if file not found."""
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cameras_as_list(cameras_raw) -> list[dict]:
    """Normalise the cameras value to a list of dicts."""
    if isinstance(cameras_raw, dict):
        return [cameras_raw]
    if isinstance(cameras_raw, list):
        return [c for c in cameras_raw if isinstance(c, dict)]
    return []


def _id_of(block: dict) -> int:
    """Extract camera_id from a block, accepting both 'id' and legacy 'ID'."""
    return int(block.get("id", block.get("ID", 0)))


def _taken_ids(raw: dict) -> set[int]:
    return {_id_of(b) for b in _cameras_as_list(raw.get("cameras"))} - {0}


def _default_block(camera_id: int, cam: ZwoAsiCamera, raw: dict) -> dict:
    """Build a minimal default camera block for a newly identified camera."""
    model_defaults = (raw.get("models") or {}).get(cam.info.model, {})
    config: dict = {
        "telescope_description": f"{cam.info.model} — edit with telescope name",
        "focal_length_mm": 0,
        "flip": 0,
        "bin": 1,
    }
    if cam.info.bayer_pattern:
        config["pattern"] = cam.info.bayer_pattern
    if model_defaults.get("gain") is not None:
        config["gain"] = int(model_defaults["gain"])
    if model_defaults.get("offset") is not None:
        config["offset"] = int(model_defaults["offset"])
    return {"id": camera_id, "config_1": config}


def _append_camera_block(config_path: Path, new_block: dict, raw: dict) -> None:
    """
    Append new_block to the cameras section of configuration.yaml.

    Everything above the 'cameras:' line is preserved verbatim — all header
    comments, cal_path, filters, etc. remain untouched.  Only the cameras
    section is regenerated (as a YAML list), so inline comments within that
    section are not preserved.
    """
    cam_list = _cameras_as_list(raw.get("cameras"))
    cam_list.append(new_block)

    cameras_yaml = yaml.dump(
        {"cameras": cam_list},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        m = re.search(r"^cameras:", text, re.MULTILINE)
        header = text[: m.start()] if m else text
    else:
        header = ""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(header.rstrip("\n") + "\n" + cameras_yaml, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if not 1 <= args.new_id <= 8:
        print(f"Error: ID must be between 1 and 8 (got {args.new_id}).")
        sys.exit(1)

    # Show connected cameras
    print()
    cameras = list_cameras()
    if not cameras:
        print("No ZWO ASI cameras detected. Check USB connection.")
        sys.exit(1)
    _print_cameras(cameras)

    usb_indices = [c["usb_index"] for c in cameras]
    if args.index not in usb_indices:
        print(f"\nNo camera at USB index {args.index}. Available: {usb_indices}")
        sys.exit(1)

    target = next(c for c in cameras if c["usb_index"] == args.index)
    print(f"\nTarget : {target['model']}  (USB index {args.index}  "
          f"current ID={target['camera_id']})")
    print(f"New ID : {args.new_id}")

    # Conflict check
    raw   = _load_raw(args.config)
    taken = _taken_ids(raw)

    if args.new_id in taken:
        owner = next(
            (b for b in _cameras_as_list(raw.get("cameras")) if _id_of(b) == args.new_id),
            None,
        )
        owner_desc = ""
        if owner:
            first_cfg = next(
                (v for k, v in owner.items()
                 if k not in ("id", "ID") and isinstance(v, dict)),
                None,
            )
            if first_cfg:
                owner_desc = f" ({first_cfg.get('telescope_description', '')})"
        print(f"\nID {args.new_id} is already assigned in "
              f"{args.config.name}{owner_desc}.")
        if not args.force:
            print("Use --force to overwrite the EEPROM without modifying "
                  f"{args.config.name}.")
            sys.exit(1)
        print(f"--force: EEPROM will be written. {args.config.name} will not change.")

    # Write EEPROM and (if new ID) build the default YAML block
    print(f"\nWriting ID {args.new_id} to EEPROM...", end=" ", flush=True)
    with ZwoAsiCamera(index=args.index) as cam:
        cam.set_camera_id(args.new_id)
        verified = cam.info.camera_id
        if verified != args.new_id:
            print(f"FAILED (readback returned {verified}).")
            sys.exit(1)
        print(f"OK  (readback {verified} ✓)")

        new_block = (
            _default_block(args.new_id, cam, raw)
            if args.new_id not in taken
            else None
        )

    # Update YAML (skip if --force or if ID was already present)
    if new_block is not None:
        _append_camera_block(args.config, new_block, raw)
        print(f"\n{args.config.name} updated.")
        print("Note: inline comments in the cameras: section are not preserved "
              "by this tool — re-add any you need.")
        print(
            f"\nNext steps:\n"
            f"  1. Edit {args.config.name}: set telescope_description and "
            f"focal_length_mm for camera {args.new_id}.\n"
            f"  2. Collect calibration frames with acquire_darks.py and "
            f"acquire_flats.py."
        )
    else:
        print(f"\nDone. {args.config.name} was not changed.")


if __name__ == "__main__":
    main()
