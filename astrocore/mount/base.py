from abc import ABC, abstractmethod


class Mount(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def slew_to(self, ra_hours: float, dec_degrees: float) -> None: ...

    @abstractmethod
    def sync(self, ra_hours: float, dec_degrees: float) -> None: ...

    @abstractmethod
    def abort(self) -> None: ...

    @property
    @abstractmethod
    def position(self) -> tuple[float, float]:
        """Return current (RA hours, Dec degrees)."""
        ...
