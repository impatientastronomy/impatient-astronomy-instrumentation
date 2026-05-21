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
