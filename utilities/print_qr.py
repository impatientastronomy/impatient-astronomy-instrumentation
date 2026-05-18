#!/usr/bin/env python3
"""
print_qr.py — Generate a printable WiFi QR code for the AstroEye hotspot.

Reads hotspot credentials from configuration.yaml and writes a PNG that
can be printed and attached to the telescope.  Guests scan it to join the
hotspot; the captive portal then opens the image gallery automatically.

Usage::

    uv run python utilities/print_qr.py
    uv run python utilities/print_qr.py --output ~/Desktop/astro_qr.png
    uv run python utilities/print_qr.py --config /path/to/configuration.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astrocore.config.camera_config import load

try:
    import qrcode
    from qrcode.image.pil import PilImage
    import PIL  # noqa: F401
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("astro_qr.png"),
                   help="Output PNG path (default: astro_qr.png)")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    try:
        config = load(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    hs = config.hotspot
    data = hs.wifi_qr_data

    print(f"SSID     : {hs.ssid}")
    print(f"Password : {hs.password}")
    print(f"Gallery  : {hs.gallery_url}")
    print(f"QR data  : {data}")
    print()

    if not _HAS_PIL:
        print("Pillow is not installed — cannot write PNG.")
        print("Install it with:  uv add pillow")
        print()
        print("You can still generate the QR at https://qrfi.io/ using the data above.")
        sys.exit(1)

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(str(args.output))
    print(f"QR code saved to {args.output.resolve()}")
    print("Print this and attach it to the telescope.")


if __name__ == "__main__":
    main()
