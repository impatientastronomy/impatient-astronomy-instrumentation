"""
ZWO AM5 mount driver.

The AM5 speaks LX200 over a TCP connection on port 4030.  Auto-detection
tries three strategies in order:

1. Hotspot mode (fast): AM5 is always the gateway at x.x.x.1.
2. ARP neighbours: probe recently seen hosts — catches station mode instantly.
3. Parallel subnet scan: scans .2–.254 on every local /24 (full fallback).
"""

from __future__ import annotations

import concurrent.futures
import re
import socket
import subprocess

from .lx200 import Lx200Mount

_AM5_PORT       = 4030
_DETECT_TIMEOUT = 0.3   # seconds per IP probe


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


# ── helpers ───────────────────────────────────────────────────────────────────

def _probe(ip: str, port: int) -> str | None:
    try:
        with socket.create_connection((ip, port), timeout=_DETECT_TIMEOUT):
            return ip
    except OSError:
        return None


def _all_interface_ips() -> list[str]:
    for cmd in (["ifconfig"], ["ip", "addr"]):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            ips = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", out)
            ips = [ip for ip in ips if not ip.startswith("127.")]
            if ips:
                return ips
        except (OSError, subprocess.SubprocessError):
            pass
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
    subnets: set[str] = set()
    for ip in _all_interface_ips():
        parts = ip.split(".")
        subnets.add(f"{parts[0]}.{parts[1]}.{parts[2]}.")
    return list(subnets)


def _arp_neighbors() -> list[str]:
    """Return IPs from the kernel ARP/neighbour table — populated by recent traffic."""
    try:
        out = subprocess.check_output(["ip", "neigh"], text=True, stderr=subprocess.DEVNULL)
        return re.findall(r"^(\d+\.\d+\.\d+\.\d+)", out, re.MULTILINE)
    except (OSError, subprocess.SubprocessError):
        return []


def _auto_detect_ip(port: int) -> str | None:
    subnets = _local_subnets()
    if not subnets:
        return None

    # 1. Hotspot mode: AM5 is always the gateway (.1)
    for subnet in subnets:
        if result := _probe(f"{subnet}1", port):
            return result

    # 2. ARP neighbours: finds the AM5 instantly in station mode if it has
    #    been seen recently (e.g. DHCP lease, prior ping, or ongoing traffic)
    for ip in _arp_neighbors():
        if result := _probe(ip, port):
            return result

    # 3. Full parallel scan of .2–.254 on every local subnet
    candidates = [f"{subnet}{i}" for subnet in subnets for i in range(2, 255)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
        futs = {ex.submit(_probe, ip, port): ip for ip in candidates}
        for fut in concurrent.futures.as_completed(futs):
            if (result := fut.result()) is not None:
                return result

    return None


# Convention: every mount module exposes Driver pointing to its concrete class.
Driver = ZwoAm5
