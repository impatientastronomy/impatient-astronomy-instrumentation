"""
Tests for digital_eyepiece/input/menu.py.
Run with: pytest digital_eyepiece/tests/test_menu.py -v
"""

import pytest
from digital_eyepiece.input.menu import Menu, MenuItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _menu_with_items(*labels: str) -> Menu:
    m = Menu()
    for label in labels:
        m.add(MenuItem(label))
    return m


def _action_recorder():
    calls = []
    return calls, lambda: calls.append(True)


# ---------------------------------------------------------------------------
# Basic navigation
# ---------------------------------------------------------------------------

class TestNavigation:
    def test_initial_selection_is_zero(self):
        m = _menu_with_items("A", "B", "C")
        assert m.selection_index == 0

    def test_scroll_down_increases_index(self):
        m = _menu_with_items("A", "B", "C")
        m.scroll(1)
        assert m.selection_index == 1

    def test_scroll_up_decreases_index(self):
        m = _menu_with_items("A", "B", "C")
        m.scroll(2)
        m.scroll(-1)
        assert m.selection_index == 1

    def test_scroll_clamped_at_top(self):
        m = _menu_with_items("A", "B", "C")
        m.scroll(-5)
        assert m.selection_index == 0

    def test_scroll_clamped_at_bottom(self):
        m = _menu_with_items("A", "B", "C")
        m.scroll(99)
        assert m.selection_index == 2

    def test_scroll_by_multiple_steps(self):
        m = _menu_with_items("A", "B", "C", "D")
        m.scroll(3)
        assert m.selection_index == 3

    def test_current_item_reflects_selection(self):
        m = _menu_with_items("Alpha", "Beta")
        m.scroll(1)
        assert m.current_item.label == "Beta"


# ---------------------------------------------------------------------------
# Selection — leaf items
# ---------------------------------------------------------------------------

class TestLeafSelection:
    def test_select_noop_item_returns_true(self):
        m = _menu_with_items("Cancel")
        assert m.select() is True

    def test_select_calls_action(self):
        calls, action = _action_recorder()
        m = Menu()
        m.add(MenuItem("Save", action=action))
        m.select()
        assert calls == [True]

    def test_select_leaf_returns_true(self):
        calls, action = _action_recorder()
        m = Menu()
        m.add(MenuItem("Save", action=action))
        assert m.select() is True

    def test_select_second_item(self):
        calls, action = _action_recorder()
        m = Menu()
        m.add(MenuItem("Cancel"))
        m.add(MenuItem("Save", action=action))
        m.scroll(1)
        m.select()
        assert calls == [True]


# ---------------------------------------------------------------------------
# Submenus
# ---------------------------------------------------------------------------

class TestSubmenus:
    def test_select_parent_enters_submenu(self):
        sub = [MenuItem("0.5×"), MenuItem("1×"), MenuItem("2×")]
        m = Menu()
        m.add(MenuItem("Brightness", submenu=sub))
        result = m.select()
        assert result is False
        assert m.current_items is sub

    def test_navigation_inside_submenu(self):
        sub = [MenuItem("0.5×"), MenuItem("1×"), MenuItem("2×")]
        m = Menu()
        m.add(MenuItem("Brightness", submenu=sub))
        m.select()                # enter submenu
        m.scroll(2)
        assert m.current_item.label == "2×"

    def test_submenu_action_called(self):
        calls, action = _action_recorder()
        sub = [MenuItem("1×", action=action)]
        m = Menu()
        m.add(MenuItem("Brightness", submenu=sub))
        m.select()                # enter submenu
        result = m.select()       # select leaf
        assert result is True
        assert calls == [True]

    def test_back_exits_submenu(self):
        sub = [MenuItem("0.5×")]
        m = Menu()
        m.add(MenuItem("Brightness", submenu=sub))
        m.add(MenuItem("Save"))
        m.select()                # enter submenu
        at_root = m.back()
        assert at_root is False
        assert m.current_items is m._root

    def test_back_at_root_returns_true(self):
        m = _menu_with_items("Cancel", "Save")
        assert m.back() is True

    def test_reset_returns_to_root_and_clears_selection(self):
        sub = [MenuItem("A")]
        m = Menu()
        m.add(MenuItem("Parent", submenu=sub))
        m.select()                # enter submenu
        m.reset()
        assert m.current_items is m._root
        assert m.selection_index == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_menu_select_returns_true(self):
        m = Menu()
        assert m.select() is True

    def test_add_items_visible_immediately(self):
        m = Menu()
        m.add(MenuItem("A"))
        assert len(m.current_items) == 1
        m.add(MenuItem("B"))
        assert len(m.current_items) == 2
