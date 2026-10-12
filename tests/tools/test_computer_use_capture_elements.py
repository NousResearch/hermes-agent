"""What a capture puts in the model's context: one element list, the window's, without the app menu bar.

Regression for #135713. A capture returned every element twice (a JSON ``elements`` array and the same rows in
``summary``), and the app's menu-bar subtree filled the element cap ahead of window controls. On macOS that
subtree includes Apple menu > Recent Items, so recently opened file names reached the model, the auxiliary
vision prompt and the element spill file. The menu bar is now left out unless the caller passes ``menus=true``.

Driven from a cua-driver ``get_window_state`` payload through the real capture lane and response shaping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from tools.computer_use import tool
from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend_capture import _CaptureMixin

PRIVATE_FILE = "example-recent-document.pages"

# (index, parent_index, depth, role, label): a window with two controls, then the app menu bar whose
# Apple menu holds Recent Items -> a recently opened document. Shape and indices as cua-driver 0.34 emits.
TREE = [
    (0, None, 0, "AXWindow", "Calculator"),
    (1, 0, 1, "AXButton", "Add"),
    (2, 0, 1, "AXButton", "Equals"),
    (3, None, 0, "AXMenuBar", ""),
    (4, 3, 1, "AXMenuBarItem", "Apple"),
    (5, 4, 2, "AXMenu", ""),
    (6, 5, 3, "AXMenuItem", "Recent Items"),
    (7, 6, 4, "AXMenu", ""),
    (8, 7, 5, "AXMenuItem", PRIVATE_FILE),
]
MENU_LABELS = ("Apple", "Recent Items", PRIVATE_FILE)


def _structured_payload(tree=TREE) -> dict[str, Any]:
    return {"structuredContent": {"elements": [
        {"element_index": i, "role": role, "label": label, "depth": depth,
         **({"parent_index": parent} if parent is not None else {})}
        for i, parent, depth, role, label in tree]}}


def _markdown_payload(tree=TREE) -> dict[str, Any]:
    """The pre-structured driver shape: a markdown tree that nests by two-space indentation."""
    lines = [f'{"  " * depth}- [{i}] {role} "{label}"' for i, _, depth, role, label in tree]
    return {"data": "window state\n" + "\n".join(lines)}


class _Driver(_CaptureMixin):
    """The cua-driver capture lane with the driver call stubbed by a fixed get_window_state payload."""

    def __init__(self, payload: dict[str, Any], ax_max_elements: int = 0) -> None:
        self._payload, self._ax_max_elements = payload, ax_max_elements
        self._active_pid: Optional[int] = 42
        self._active_window_id: Optional[int] = 7
        self._session_id: Optional[str] = None
        self._last_app = ""

    def _resolve_capture_windows(self, mode, app, pid, window_id):
        return [{"app_name": "Calculator", "pid": 42, "window_id": 7, "title": "Calculator", "z_index": 1,
                 "off_screen": False}]

    def _set_active_target(self, target):
        self._active_pid, self._active_window_id = target["pid"], target["window_id"]

    def _gws_args(self):
        return {"pid": self._active_pid, "window_id": self._active_window_id,
                **({"max_elements": self._ax_max_elements} if self._ax_max_elements else {})}

    def _fetch_or_refetch(self, *args, **kwargs):
        return self._payload

    def capture(self, mode="som", app=None, pid=None, window_id=None):  # capture_after recaptures by exact target
        return super().capture("ax", app, pid, window_id)


def _capture(payload: dict[str, Any], **args: Any) -> tuple[str, dict[str, Any]]:
    raw = tool._do_capture(_Driver(payload, args.pop("ax_max_elements", 0)), "capture", {"mode": "ax", **args})
    return raw, json.loads(raw)


@pytest.mark.parametrize("payload", [_structured_payload(), _markdown_payload()], ids=["structured", "markdown"])
def test_default_capture_lists_the_window_once_and_no_menu_bar(payload):
    raw, body = _capture(payload)
    assert "elements" not in body  # the rows in `summary` are the only element list
    assert raw.count("'Add'") == 1 and raw.count("'Equals'") == 1
    assert not any(label in raw for label in MENU_LABELS)
    assert body["total_elements"] == 3
    assert "menu bar omitted (6 " in body["summary"] and "menus=true" in body["summary"]


def test_menus_true_lists_the_menu_bar_under_the_same_indices():
    _, body = _capture(_structured_payload(), menus=True)
    assert body["total_elements"] == len(TREE)
    assert "#6 AXMenuItem 'Recent Items'" in body["summary"] and f"#8 AXMenuItem '{PRIVATE_FILE}'" in body["summary"]
    assert "menu bar omitted" not in body["summary"]


def test_capture_after_follows_the_menus_choice():
    """Clicking a menu-bar item with capture_after=true must be able to show the menu it just opened."""
    driver: Any = _Driver(_structured_payload())  # a capture lane, not a full ComputerUseBackend
    click = lambda menus: json.loads(tool._maybe_follow_capture(
        driver, ActionResult(ok=True, action="click"), True, include_menus=menus))
    assert PRIVATE_FILE not in click(False)["summary"] and PRIVATE_FILE in click(True)["summary"]


def test_untagged_container_lines_keep_markdown_nesting_exact():
    """Only actionable rows carry [N]; an untagged line at the menu bar's level closes the menu bar."""
    md = "\n".join(['- [0] AXWindow "Calculator"', '- [3] AXMenuBar ""', '  - [4] AXMenuBarItem "Apple"',
                    '- AXGroup "toolbar"', '  - [9] AXButton "Add"'])
    _, body = _capture({"data": "window state\n" + md})
    assert "#9 AXButton 'Add'" in body["summary"] and "'Apple'" not in body["summary"]


@pytest.mark.parametrize("tree", [
    [(0, None, 0, "AXWindow", "Calculator"), (3, None, 0, "AXMenuBar", ""), (4, None, 0, "AXMenuBarItem", "Apple")],
    [(0, None, 0, "frame", "kcalc"), (3, 0, 1, "menu bar", ""), (4, 3, 2, "menu", "File")],
], ids=["no-tree-shape", "linux-menu-bar-inside-window"])
def test_menu_bar_is_kept_when_it_cannot_be_told_apart_from_the_window(tree):
    """Flat legacy trees give no subtree to drop, and AT-SPI/UIA menu bars sit inside the window: keep everything,
    and never claim an omission that did not happen."""
    _, body = _capture(_structured_payload(tree))
    assert body["total_elements"] == len(tree) and "menu bar omitted" not in body["summary"]


def test_spill_file_never_receives_the_menu_bar():
    """A long label forces the full-tree spill; the file is model-readable, so the menu bar stays out of it too."""
    tree = [*TREE[:2], (2, 0, 1, "AXStaticText", "x" * (tool._MAX_ELEMENT_LABEL_CHARS + 1)), *TREE[3:]]
    _, body = _capture(_structured_payload(tree))
    spill = Path(body["elements_file"]).read_text(encoding="utf-8")
    assert "Add" in spill and not any(label in spill for label in MENU_LABELS)


def test_walk_cap_is_still_reported_after_the_menu_bar_is_dropped():
    """The driver stopped at its bound; dropping menu rows afterwards must not hide that the tree is partial."""
    _, body = _capture(_structured_payload(), ax_max_elements=len(TREE))
    assert f"accessibility walk capped at {len(TREE)} elements" in body["summary"]


def test_every_listed_element_gets_a_row_with_its_bounds():
    """The summary is the only list now, so it carries every element up to the cap (it stopped at 40) with bounds."""
    tree = [(0, None, 0, "AXWindow", "Grid")] + [(i, 0, 1, "AXButton", f"cell {i}") for i in range(1, 61)]
    payload = _structured_payload(tree)
    for raw_el in payload["structuredContent"]["elements"]:
        raw_el["frame"] = {"x": 10, "y": 20, "w": 30, "h": 40}
    _, body = _capture(payload)
    rows = [line for line in body["summary"].splitlines() if line.startswith("  #")]
    assert len(rows) == 61 and rows[-1] == "  #60 AXButton 'cell 60' @ (10, 20, 30, 40)"
