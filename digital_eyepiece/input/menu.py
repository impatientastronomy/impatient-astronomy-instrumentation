"""
Menu — a scroll-navigated, click-selected hierarchical menu.

The menu is always instantiated; visibility is controlled by
ViewState.menu_open. Items are added at startup based on which hardware
is connected. Submenus are entered by selecting a parent item and exited
by selecting 'Back' or calling back() programmatically.

Typical setup::

    menu = Menu()
    menu.add(MenuItem("Cancel"))                    # closes menu, no action
    menu.add(MenuItem("Brightness", submenu=[
        MenuItem("0.1×", action=lambda: set_brightness(0.1)),
        MenuItem("1×",   action=lambda: set_brightness(1.0)),
        MenuItem("10×",  action=lambda: set_brightness(10.0)),
        MenuItem("Back"),
    ]))
    menu.add(MenuItem("Save", action=save_images))

    # Hardware-conditional:
    if cooler:
        menu.add(MenuItem("Cam Temp", submenu=[...]))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class MenuItem:
    """
    A single menu entry.

    label may be a plain string or a zero-argument callable that returns a
    string — use a callable for items whose text changes with application
    state (e.g. "Start Stacking" / "Start Streaming").

    Exactly one of action or submenu should be set for interactive items.
    An item with neither (label only) acts as a separator or header.
    An item with action=None and no submenu closes the menu when selected
    (useful for a 'Cancel' or 'Back' entry).
    """
    label: str | Callable[[], str]
    action: Callable[[], None] | None = None
    submenu: list[MenuItem] | None = None

    @property
    def label_text(self) -> str:
        """Resolve the label, calling it if it is a callable."""
        return self.label() if callable(self.label) else self.label


class Menu:
    """
    Hierarchical menu navigated by scroll and confirmed by click.

    scroll(delta)  — move the highlighted item up (delta < 0) or down (delta > 0)
    select()       — activate the highlighted item; returns True if the menu
                     should close (leaf action called or no-op item selected)
    back()         — exit the current submenu level; returns True if already
                     at root (caller may choose to close the menu)
    reset()        — return to root and clear selection
    """

    def __init__(self) -> None:
        self._root: list[MenuItem] = []
        # Each stack entry is (items, selection_index)
        self._stack: list[tuple[list[MenuItem], int]] = []

    # -- building the menu --------------------------------------------------

    def add(self, item: MenuItem) -> None:
        """Append an item to the root menu."""
        self._root.append(item)

    # -- navigation ---------------------------------------------------------

    def scroll(self, delta: int) -> None:
        """Move selection by delta steps. Clamped to valid range, no wrap."""
        items, idx = self._top()
        new_idx = max(0, min(len(items) - 1, idx + delta))
        self._set_top(items, new_idx)

    def select(self) -> bool:
        """
        Activate the current item.

        Returns True if the menu should close (leaf item selected or a no-op
        item with no action and no submenu). Returns False if a submenu was
        entered.
        """
        items, idx = self._top()
        if not items:
            return True

        item = items[idx]

        if item.submenu:
            self._stack.append((item.submenu, 0))
            return False

        if item.action is not None:
            item.action()

        return True     # leaf selected or no-op (Cancel/Back): close menu

    def back(self) -> bool:
        """
        Exit the current submenu. Returns True if already at root
        (no submenu to exit — caller may close the menu).
        """
        if len(self._stack) <= 1:
            return True     # already at root
        self._stack.pop()
        return False

    def reset(self) -> None:
        """Return to root, clear selection."""
        self._stack = [(self._root, 0)]

    # -- read access --------------------------------------------------------

    @property
    def current_items(self) -> list[MenuItem]:
        """Items visible at the current menu level."""
        return self._top()[0]

    @property
    def selection_index(self) -> int:
        """Index of the highlighted item at the current level."""
        return self._top()[1]

    @property
    def current_item(self) -> MenuItem | None:
        """The currently highlighted MenuItem, or None if menu is empty."""
        items, idx = self._top()
        return items[idx] if items else None

    # -- internals ----------------------------------------------------------

    def _top(self) -> tuple[list[MenuItem], int]:
        if not self._stack:
            self._stack = [(self._root, 0)]
        return self._stack[-1]

    def _set_top(self, items: list[MenuItem], idx: int) -> None:
        if self._stack:
            self._stack[-1] = (items, idx)
        else:
            self._stack = [(items, idx)]
