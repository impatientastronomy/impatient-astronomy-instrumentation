"""
ZWO AM5 mount driver.

The AM5 speaks LX200 over a TCP connection on port 4030.  When operating as
a WiFi hotspot (the normal mode) it is always the gateway at x.x.x.1, so
auto-detection finds it on the first probe.
"""

from __future__ import annotations

import re
import socket
import subprocess

from .lx200 import Lx200Mount

_AM5_PORT    = 4030
_DETECT_TIMEOUT = 0.3   # seconds per IP probe during auto-detection


class ZwoAm5(Lx200Mount):
    """
    Mount driver for the ZWO AM5 harmonic equatorial mount.

    If host is None the driver scans the local subnet (en0 on macOS, eth0 on
    Linux) to find the mount automatically.

    Usage::

        with ZwoAm5() as mount:          # auto-detect IP
            ra, dec = mount.position
            mount.slew_to(target_ra, target_dec)
            mount.wait_for_slew()
    """

    def __init__(
        self,
        host: str | None = None,
        port: int = _AM5_PORT,
        timeout: float = 3.0,
    ) -> None:
        if host is None:
            host = _auto_detect_ip(port)
            if host is None:
                raise ConnectionError(
                    "ZWO AM5 not found on the local network. "
                    "Check that the mount is powered on and connected to WiFi."
                )
        super().__init__(host=host, port=port, timeout=timeout)


# ── IP auto-detection ─────────────────────────────────────────────────────────

def _all_interface_ips() -> list[str]:
    """
    Return all IPv4 addresses assigned to local network interfaces.

    Tries ifconfig (macOS + older Linux) then ip addr (newer Linux/Pi).
    Falls back to the UDP routing trick if neither is available.
    """
    for cmd in (["ifconfig"], ["ip", "addr"]):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            ips = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", out)
            ips = [ip for ip in ips if not ip.startswith("127.")]
            if ips:
                return ips
        except (OSError, subprocess.SubprocessError):
            pass

    # Last resort: routing trick (only finds the default-route interface)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        if not ip.startswith("127."):
            return [ip]
    except OSError:
        pass

    return []


def _local_subnets() -> list[str]:
    """Return /24 subnet prefixes for all active IPv4 interfaces."""
    subnets: set[str] = set()
    for ip in _all_interface_ips():
        parts = ip.split(".")
        subnets.add(f"{parts[0]}.{parts[1]}.{parts[2]}.")
    return list(subnets)


def _auto_detect_ip(port: int, max_host: int = 10) -> str | None:
    """
    Find the AM5 on the local network.

    Fast path: try the gateway (.1) on every local subnet — the AM5 is always
    the gateway when in hotspot mode, so this usually succeeds on the first probe.
    Fallback: scan .2 through .max_host on each subnet for station mode.
    """
    subnets = _local_subnets()
    if not subnets:
        return None

    for subnet in subnets:
        ip = f"{subnet}1"
        try:
            with socket.create_connection((ip, port), timeout=_DETECT_TIMEOUT):
                return ip
        except OSError:
            pass

    for subnet in subnets:
        for i in range(2, max_host + 1):
            ip = f"{subnet}{i}"
            try:
                with socket.create_connection((ip, port), timeout=_DETECT_TIMEOUT):
                    return ip
            except OSError:
                pass

    return None


# Convention: every mount module exposes Driver pointing to its concrete class.
Driver = ZwoAm5
