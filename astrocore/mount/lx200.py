"""
LX200 telescope protocol driver over TCP/IP.

Implements the Meade LX200 command set used by many consumer goto mounts
(ZWO AM5, iOptron, SkyWatcher via SynScan, etc.).

Classes
-------
Lx200Mount  -- concrete Mount implementation using LX200 over a TCP socket
MountError  -- raised on protocol errors or communication failures
"""

from __future__ import annotations

import logging
import socket
import time

log = logging.getLogger(__name__)

from .base import Mount

_BUFSIZE      = 256
_DEFAULT_PORT = 4030


class MountError(Exception):
    """Raised when the mount reports an error or communication fails."""


class Lx200Mount(Mount):
    """
    Mount driver for LX200-compatible controllers over TCP/IP.

    Typical usage::

        with Lx200Mount("192.168.1.100") as mount:
            ra, dec = mount.position
            mount.slew_to(ra_target, dec_target)
            mount.wait_for_slew()
    """

    def __init__(
        self,
        host: str,
        port: int = _DEFAULT_PORT,
        timeout: float = 3.0,
    ) -> None:
        self._host    = host
        self._port    = port
        self._timeout = timeout
        self._sock: socket.socket | None = None

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._sock = socket.create_connection(
            (self._host, self._port), timeout=self._timeout
        )
        self._sock.settimeout(self._timeout)
        self._drain()

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def ping(self) -> bool:
        """Return True if the mount responds to a position query."""
        try:
            return bool(self._cmd(":GR#"))
        except OSError:
            return False

    # ── Mount ABC ─────────────────────────────────────────────────────────────

    @property
    def position(self) -> tuple[float, float]:
        """Return (ra_hours, dec_degrees) from the mount's current position."""
        return self._get_ra(), self._get_dec()

    def slew_to(self, ra_hours: float, dec_degrees: float) -> None:
        """
        Set target coordinates and begin slewing (non-blocking).

        Raises MountError if the target is unreachable (below horizon, etc.).
        """
        self._ensure()
        self._cmd1(f":Sr{_ra_str(ra_hours)}#")
        self._cmd1(f":Sd{_dec_str(dec_degrees)}#")
        result = self._cmd1(":MS#")
        if result != "0":
            raise MountError(f"Slew rejected by mount (code {result!r})")

    def abort(self) -> None:
        """Stop all slewing immediately."""
        self._ensure()
        self._cmdn(":Q#")

    def sync(self, ra_hours: float, dec_degrees: float) -> None:
        """Sync the mount's pointing model to the given coordinates."""
        self._ensure()
        self._cmd1(f":Sr{_ra_str(ra_hours)}#")
        self._cmd1(f":Sd{_dec_str(dec_degrees)}#")
        self._cmd(":CM#")

    def park(self) -> None:
        self._ensure()
        self._cmdn(":hP#")

    def unpark(self) -> None:
        self._ensure()
        self._cmdn(":hU#")
        self._cmdn(":Te#")   # AM5 requires explicit tracking enable after unpark

    @property
    def is_slewing(self) -> bool:
        """True while the mount is executing a slew.

        Uses a short-timeout :D# query. The AM5 does not implement :D#, so
        this will always return True on that mount — use wait_for_slew() instead
        of polling is_slewing directly when you need to block on completion.
        """
        self._ensure()
        if self._sock is None:
            raise MountError("Not connected")
        self._sock.sendall(b":D#")
        data = b""
        self._sock.settimeout(0.5)
        try:
            while b"#" not in data:
                chunk = self._sock.recv(_BUFSIZE)
                if chunk:
                    data += chunk
        except (socket.timeout, OSError):
            pass
        finally:
            self._sock.settimeout(self._timeout)
        status = data.decode(errors="replace").rstrip("#").strip()
        return status not in ("0", "")

    # ── convenience ───────────────────────────────────────────────────────────

    def wait_for_slew(self, timeout_s: float = 120.0, poll_s: float = 1.0) -> None:
        """
        Block until the mount stops moving.

        Polls position every poll_s seconds.  Waits until the mount has
        demonstrably moved from its starting position before checking for
        stability — this avoids a false-done if the mount hasn't started
        moving yet when the first poll fires.
        """
        t0           = time.monotonic()
        start        = self.position
        seen_movement = False
        prev         = start
        while True:
            if time.monotonic() - t0 > timeout_s:
                raise MountError(f"Slew timed out after {timeout_s:.0f}s")
            time.sleep(poll_s)
            cur = self.position
            total_delta = (
                (cur[0] - start[0]) ** 2 + (cur[1] - start[1]) ** 2
            ) ** 0.5
            if total_delta > 0.001:
                seen_movement = True
            if seen_movement:
                step_delta = (
                    (cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2
                ) ** 0.5
                if step_delta < 0.001:
                    return
            prev = cur

    def go_home(self) -> None:
        self._ensure()
        self._cmdn(":hF#")

    def set_home_here(self) -> None:
        """Mark the current position as the home position."""
        self._ensure()
        self._cmdn(":SZ#")

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_ra(self) -> float:
        self._ensure()
        return _parse_ra(self._cmd(":GR#"))

    def _get_dec(self) -> float:
        self._ensure()
        # The AM5 does not always respond with the expected '+DD*MM:SS#' format.
        # Root cause is unknown; retrying immediately always succeeds eventually.
        # No upper limit: a hard cap risks crashing on what is just a mount quirk.
        attempt = 0
        while True:
            response = self._cmd(":GD#")
            if "*" in response:
                return _parse_dec(response)
            attempt += 1
            log.warning("Malformed :GD# response (attempt %d): %r", attempt, response)
            time.sleep(0.1)

    def _ensure(self) -> None:
        if self._sock is None:
            self.connect()

    def _drain(self) -> None:
        """Discard any stale bytes in the socket receive buffer."""
        self._sock.setblocking(False)
        try:
            while self._sock.recv(_BUFSIZE):
                pass
        except (BlockingIOError, OSError):
            pass
        self._sock.setblocking(True)

    def _cmd(self, cmd: str) -> str:
        """Send command, read '#'-terminated response, return stripped string."""
        if self._sock is None:
            raise MountError("Not connected")
        self._sock.sendall(cmd.encode())
        data    = b""
        deadline = time.monotonic() + self._timeout
        while b"#" not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MountError(f"No response to {cmd!r} within {self._timeout}s")
            self._sock.settimeout(remaining)
            chunk = self._sock.recv(_BUFSIZE)
            if chunk:
                data += chunk
        return data.decode(errors="replace").rstrip("#").strip()

    def _cmd1(self, cmd: str) -> str:
        """Send command, read a single-byte response (no '#' terminator)."""
        if self._sock is None:
            raise MountError("Not connected")
        self._sock.sendall(cmd.encode())
        self._sock.settimeout(self._timeout)
        byte = self._sock.recv(1)
        return byte.decode(errors="replace").strip() if byte else ""

    def _cmdn(self, cmd: str) -> None:
        """Send command with no response expected."""
        if self._sock is None:
            raise MountError("Not connected")
        self._sock.sendall(cmd.encode())


# ── LX200 string codec ────────────────────────────────────────────────────────

def _ra_str(hours: float) -> str:
    """Format decimal RA hours as HH:MM:SS."""
    h = int(hours)
    m = int((hours - h) * 60)
    s = round(((hours - h) * 60 - m) * 60)
    if s == 60:
        s, m = 0, m + 1
    if m == 60:
        m, h = 0, (h + 1) % 24
    return f"{h:02d}:{m:02d}:{s:02d}"


def _dec_str(degrees: float) -> str:
    """Format decimal declination degrees as +DD*MM:SS."""
    sign = "+" if degrees >= 0 else "-"
    d    = abs(degrees)
    deg  = int(d)
    m    = int((d - deg) * 60)
    s    = round(((d - deg) * 60 - m) * 60)
    if s == 60:
        s, m = 0, m + 1
    if m == 60:
        m, deg = 0, deg + 1
    return f"{sign}{deg:02d}*{m:02d}:{s:02d}"


def _parse_ra(s: str) -> float:
    """Parse 'HH:MM:SS' to decimal hours."""
    h, m, sec = (float(x) for x in s.split(":"))
    return h + m / 60.0 + sec / 3600.0


def _parse_dec(s: str) -> float:
    """Parse '+DD*MM:SS' or '-DD*MM:SS' to decimal degrees."""
    sign    = -1.0 if s.startswith("-") else 1.0
    s       = s.lstrip("+-")
    deg_str, rest = s.split("*")
    min_str, sec_str = rest.split(":")
    return sign * (float(deg_str) + float(min_str) / 60.0 + float(sec_str) / 3600.0)
