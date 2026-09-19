from abc import ABC, abstractmethod


class Mount(ABC):
    """Abstract base class for telescope mounts."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def position(self) -> tuple[float, float]:
        """Return current (ra_hours, dec_degrees)."""
        ...

    @abstractmethod
    def slew_to(self, ra_hours: float, dec_degrees: float) -> None:
        """Begin slewing to target equatorial coordinates (non-blocking)."""
        ...

    @abstractmethod
    def abort(self) -> None:
        """Immediately stop any in-progress slew."""
        ...

    @abstractmethod
    def sync(self, ra_hours: float, dec_degrees: float) -> None:
        """Sync the mount's pointing model to the given position."""
        ...

    @abstractmethod
    def park(self) -> None: ...

    @abstractmethod
    def unpark(self) -> None: ...

    @property
    @abstractmethod
    def is_slewing(self) -> bool:
        """True while the mount is executing a slew."""
        ...

    @property
    def is_tracking(self) -> bool:
        """True if the mount is actively tracking the sky.

        Default returns True so drivers that don't implement this are treated
        as always tracking.  Override in concrete classes that can query state.
        """
        return True

    @property
    def site_location(self) -> tuple[float, float] | None:
        """(lat_deg, lon_deg) the mount reports for its configured observing
        site, or None if the driver can't query it or none is configured.

        Longitude is positive-east, matching this codebase's convention
        (see astrocore.mount.coord) -- concrete drivers must convert from
        whatever convention their protocol uses.

        Default returns None so drivers that don't implement this are
        treated as not knowing their location; callers should fall back to
        a configured default. Override in concrete classes that can query it.
        """
        return None
