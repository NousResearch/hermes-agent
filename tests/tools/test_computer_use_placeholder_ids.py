"""A zero-filled ``pid``/``window_id`` must not be read as exact targeting.

Several providers emit every declared schema property on every tool call,
filling unused optional integers with ``0``. ``capture()`` entered its
exact-target branch for any non-``None`` id, so those calls dropped the
caller's ``app=``, coerced both ids to ``None`` via ``_positive_int``, and
failed with a message pointing at pid/window_id rather than the real problem.
For that class of model ``capture(app=...)`` never worked at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def _backend_with_windows(windows: List[Dict[str, Any]]):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._session = MagicMock()
    backend._session.call_tool.return_value = {
        "data": "",
        "images": [],
        "structuredContent": {"windows": windows},
        "isError": False,
    }
    return backend


_WINDOWS = [
    {"app_name": "Fuwari", "pid": 100, "window_id": 1,
     "is_on_screen": True, "title": "menu bar", "z_index": 0},
    {"app_name": "Comet", "pid": 200, "window_id": 2,
     "is_on_screen": True, "title": "Comet Browser", "z_index": 1},
]


def test_app_scoped_capture_survives_placeholder_ids():
    """The bug: app= was discarded and the capture failed outright."""
    backend = _backend_with_windows(_WINDOWS)

    cap = backend.capture(mode="som", app="Comet", pid=0, window_id=0)

    assert cap.app == "Comet", (
        f"app= was dropped for a zero-filled call; got app={cap.app!r} "
        f"detail={cap.window_title!r}"
    )


def test_frontmost_capture_survives_placeholder_ids():
    """Frontmost capture is the same failure with no app= to drop."""
    backend = _backend_with_windows(_WINDOWS)

    cap = backend.capture(mode="som", pid=0, window_id=0)

    # Frontmost is the highest z_index window.
    assert cap.app == "Comet", f"frontmost capture failed; detail={cap.window_title!r}"


def test_real_exact_target_still_wins_over_app():
    """Guard: a genuine pid/window pair keeps its exact-targeting behaviour."""
    backend = _backend_with_windows(_WINDOWS)

    # Ids that match no enumerated window: only the exact branch can produce
    # this label, so it proves discovery was bypassed rather than consulted.
    cap = backend.capture(mode="som", app="Ghost", pid=999, window_id=99)

    assert cap.app == "Ghost", f"exact targeting was bypassed; detail={cap.window_title!r}"


def test_partial_target_still_reports_the_missing_half():
    """Guard: one real id and one placeholder is a caller error, not discovery."""
    backend = _backend_with_windows(_WINDOWS)

    cap = backend.capture(mode="som", app="Comet", pid=200, window_id=0)

    assert "requires both pid and window_id" in cap.window_title


def test_malformed_ids_are_not_treated_as_placeholders():
    """Guard: garbage must still reach the validation error, not be ignored."""
    backend = _backend_with_windows(_WINDOWS)

    cap = backend.capture(mode="som", app="Comet", pid="abc", window_id="def")

    assert "positive integer" in cap.window_title


def test_placeholder_predicate():
    from tools.computer_use.cua_backend_parse import _is_placeholder_id

    assert _is_placeholder_id(0) is True
    assert _is_placeholder_id("0") is True
    assert _is_placeholder_id(-1) is True
    assert _is_placeholder_id(200) is False
    assert _is_placeholder_id("200") is False
    # Garbage is not a placeholder: it must keep reaching the existing error.
    assert _is_placeholder_id("abc") is False
    assert _is_placeholder_id(None) is False
    assert _is_placeholder_id(1.5) is False
    # bools are ints in Python; False must not read as a placeholder id.
    assert _is_placeholder_id(False) is False
    assert _is_placeholder_id(True) is False
