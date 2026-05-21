"""
VirtualMount — a software-only mount for testing without hardware.

Starts pointed at the north horizon (az=0°, alt=0°) at the observer's
location, converted to RA/Dec at construction time.  Behaves like a
tracking equatorial mount: position reports RA/Dec so the sky overlay
remains stable as sidereal time advances.

slew_to() and sync() update the stored RA/Dec immediately (no motion
delay) so the overlay responds to middle-click slews during testing.
"""

from __future__ import annotations

from astrocore.mount.base import Mount
from astrocore.mount.coord import altaz_to_radec


class VirtualMount(Mount):
    """Software stand-in for a real telescope mount."""

    def __init__(
        self,
        lat_deg: float = 38.44,
        lon_deg: float = -122.71,
        start_alt_deg: float = 0.0,
        start_az_deg:  float = 0.0,
    ) -> None:
        ra, dec = altaz_to_radec(start_alt_deg, start_az_deg, lat_deg, lon_deg)
        self._ra:  float = ra
        self._dec: float = dec

    # -- Mount ABC -------------------------------------------------------------

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    @property
    def position(self) -> tuple[float, float]:
        return (self._ra, self._dec)

    def slew_to(self, ra_hours: float, dec_degrees: float) -> None:
        self._ra  = ra_hours
        self._dec = dec_degrees

    def abort(self) -> None:
        pass

    def sync(self, ra_hours: float, dec_degrees: float) -> None:
        self._ra  = ra_hours
        self._dec = dec_degrees

    def park(self) -> None:
        pass

    def unpark(self) -> None:
        pass

    @property
    def is_slewing(self) -> bool:
        return False
