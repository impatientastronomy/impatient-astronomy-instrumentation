"""
Gallery HTTP server for guest image sharing.

Runs as a daemon thread.  Guests connect to the Pi's WiFi hotspot (wlan1),
which triggers a captive portal that opens this server automatically.

Routes
------
GET /           → HTML image gallery (newest first, auto-refreshes)
GET /<file>     → serve a JPEG from image_path/
GET /favicon.ico → 204 No Content

Captive portal detection
------------------------
iOS, Android, and Windows each probe well-known URLs when joining a new
network.  This handler intercepts those probes and redirects to the gallery,
which causes the phone to display the "Sign in to Network" notification and
open the gallery without the guest typing a URL.
"""

from __future__ import annotations

import logging
import mimetypes
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

_CAPTIVE_PORTAL_PATHS = {
    # iOS / macOS
    "/hotspot-detect.html",
    "/library/test/success.html",
    # Android
    "/generate_204",
    "/gen_204",
    "/ncsi.txt",
    # Windows
    "/connecttest.txt",
    "/redirect",
}

_GALLERY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>AstroEye</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a0a0a; color: #bbb; font-family: sans-serif;
          padding: 10px; }}
  h1 {{ font-size: 1.1em; margin-bottom: 10px; color: #ddd; }}
  .grid {{ display: grid;
           grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
           gap: 8px; }}
  .grid a {{ display: block; }}
  .grid img {{ width: 100%; border-radius: 4px;
               border: 1px solid #222; display: block; }}
  .empty {{ color: #555; text-align: center; margin-top: 60px;
            font-size: 0.9em; }}
</style>
</head>
<body>
<h1>AstroEye</h1>
{body}
</body>
</html>
"""


class _GalleryHandler(BaseHTTPRequestHandler):
    image_path: Path = Path(".")   # overridden by GalleryServer

    def log_message(self, fmt, *args):
        logging.debug("gallery: " + fmt, *args)

    def do_GET(self):
        path = self.path.split("?")[0]

        # Captive portal probes → redirect to gallery
        if path in _CAPTIVE_PORTAL_PATHS:
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        if path in ("/", "/index.html"):
            self._serve_gallery()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._serve_file(path.lstrip("/"))

    def _serve_gallery(self):
        images = sorted(
            self.image_path.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if images:
            items = "\n".join(
                f'<a href="/{p.name}"><img src="/{p.name}" '
                f'alt="{p.stem}" loading="lazy"></a>'
                for p in images
            )
            body = f'<div class="grid">\n{items}\n</div>'
        else:
            body = '<p class="empty">No images saved yet.</p>'

        html = _GALLERY_HTML.format(body=body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_file(self, filename: str):
        target = self.image_path / filename
        if not target.exists() or not target.is_file():
            self.send_response(404)
            self.end_headers()
            return
        mime, _ = mimetypes.guess_type(str(target))
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{target.name}"')
        self.end_headers()
        self.wfile.write(data)


class GalleryServer:
    """
    Tiny HTTP server that serves the image gallery.

    Usage::

        server = GalleryServer(image_path, host="0.0.0.0", port=8080)
        server.start()   # launches daemon thread; stops automatically at exit
    """

    def __init__(self, image_path: Path, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._image_path = image_path
        self._host = host
        self._port = port
        self._server: socketserver.TCPServer | None = None

    def start(self) -> None:
        image_path = self._image_path

        class _Handler(_GalleryHandler):
            pass

        _Handler.image_path = image_path

        try:
            socketserver.TCPServer.allow_reuse_address = True
            self._server = socketserver.TCPServer((self._host, self._port), _Handler)
            t = threading.Thread(target=self._server.serve_forever, daemon=True)
            t.start()
            logging.info("Gallery server started at http://%s:%d", self._host, self._port)
        except OSError as exc:
            logging.warning("Gallery server could not start: %s", exc)
