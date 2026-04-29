"""
Metadata embedding and extraction for Frame objects.

embed_meta() serialises a FrameMeta as JSON and writes it into the first
row of the image using a near-black encoding:

    pixel value = 0x00XX   (high byte 0x00, low byte = ASCII character)

A printable ASCII character has a value of 32–126, so encoded pixels are
at most 126 out of 65535 — essentially invisible when viewing the image.
Unused pixels in the first row are left as 0x0000 (pure black).

extract_meta() reverses the process: it reads the low bytes of the first
row, strips nulls, and parses the JSON back into a FrameMeta.

These are standalone functions, not methods, so they can be used on any
Frame without needing a camera instance.

Typical use
-----------
Save with embedded metadata::

    saved_frame = embed_meta(frame)
    save_tiff(saved_frame.data, path)

Recover metadata from a saved image::

    raw_data = load_tiff(path)
    frame = Frame(data=raw_data, meta=FrameMeta(...))
    recovered_meta = extract_meta(frame)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from .base import Frame, FrameMeta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_meta(frame: Frame) -> Frame:
    """
    Return a new Frame with FrameMeta encoded into the first row.

    The original frame is not modified. If the JSON representation of the
    metadata is longer than the image width, it is truncated — which would
    prevent clean extraction later. In practice a 4144-pixel-wide sensor
    gives 4144 characters, far more than any realistic metadata JSON needs.
    """
    json_str = _meta_to_json(frame.meta)
    encoded_row = _encode_row(json_str, frame.data.shape[1])
    new_data = frame.data.copy()
    new_data[0, :] = encoded_row
    return Frame(data=new_data, meta=frame.meta)


def extract_meta(frame: Frame) -> FrameMeta | None:
    """
    Attempt to decode FrameMeta from the first row of a frame.

    Returns None if the first row does not contain valid embedded metadata,
    so callers can handle un-embedded frames gracefully.
    """
    try:
        json_str = _decode_row(frame.data[0, :])
        return _json_to_meta(json_str)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_row(text: str, width: int) -> np.ndarray:
    """Convert a string into a uint16 row using the 0x00XX encoding."""
    row = np.zeros(width, dtype=np.uint16)
    chars = text[:width]  # truncate if longer than the row
    for i, ch in enumerate(chars):
        row[i] = ord(ch) & 0x00FF
    return row


def _decode_row(row: np.ndarray) -> str:
    """Extract the string from a 0x00XX-encoded uint16 row."""
    low_bytes = (row & 0x00FF).astype(np.uint8)
    # Stop at the first null byte — that marks the end of the string
    chars = []
    for byte in low_bytes:
        if byte == 0:
            break
        chars.append(chr(byte))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _meta_to_json(meta: FrameMeta) -> str:
    d = asdict(meta)
    # datetime is not JSON-serialisable by default — convert to ISO string
    d["timestamp"] = meta.timestamp.strftime(_DATETIME_FORMAT)
    return json.dumps(d, separators=(",", ":"))  # compact: no extra spaces


def _json_to_meta(json_str: str) -> FrameMeta:
    d = json.loads(json_str)
    d["timestamp"] = datetime.strptime(d["timestamp"], _DATETIME_FORMAT).replace(
        tzinfo=timezone.utc
    )
    return FrameMeta(**d)
