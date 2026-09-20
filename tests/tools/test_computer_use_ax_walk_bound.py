"""Bound on the driver's accessibility-tree walk per capture (``computer_use.ax_max_elements``).

``_DEFAULT_MAX_ELEMENTS`` in ``tool.py`` caps the SURFACED element list; the walk that produces it was
unbounded, so a capture paid for every node in the target's tree before trimming. A bound is pure
latency: the bounded tree is a prefix of the unbounded one, so the elements the model sees are
unchanged. Disabled by 0 (driver default), tuned by config, and read through the real loader.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from tools.computer_use import cua_backend
from tools.computer_use.cua_backend_capture import _CaptureMixin


class _StubCapture(_CaptureMixin):
    """Capture-lane shell: enough state for ``_gws_args``, no driver, no session."""

    def __init__(self) -> None:
        self._active_pid: Optional[int] = 607
        self._active_window_id: Optional[int] = 382
        self._session_id: Optional[str] = None
        self._last_app = ""

    def _resolve_capture_windows(self, mode: str, app: Optional[str], pid: Optional[int],
                                 window_id: Optional[int]) -> List[Dict[str, Any]]:
        return [{"app_name": "Finder", "pid": 607, "window_id": 382, "title": "", "z_index": 1,
                 "off_screen": False}]


def _write_config(tmp_path, body: str) -> None:
    """A user config.yaml the backend readers reach through ``load_config`` (not a patched reader)."""
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")


class TestAxWalkBound:

    def test_default_is_a_finite_bound(self):
        """No config line: the walk is bounded, not unbounded. The whole point of the fix."""
        assert cua_backend._DEFAULT_AX_MAX_ELEMENTS > 0
        assert cua_backend._cua_configured_ax_max_elements() == cua_backend._DEFAULT_AX_MAX_ELEMENTS

    def test_configured_value_reaches_the_driver_args(self, tmp_path, monkeypatch):
        """End to end through the loader the backend uses: config.yaml -> get_window_state args."""
        _write_config(tmp_path, "computer_use:\n  ax_max_elements: 350\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert cua_backend._cua_configured_ax_max_elements() == 350
        assert _StubCapture()._gws_args()["max_elements"] == 350

    def test_zero_disables_the_bound(self, tmp_path, monkeypatch):
        """0 restores the driver default and must not leak a ``max_elements`` key into the payload."""
        _write_config(tmp_path, "computer_use:\n  ax_max_elements: 0\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert cua_backend._cua_configured_ax_max_elements() == 0
        assert "max_elements" not in _StubCapture()._gws_args()

    def test_unusable_value_fails_to_the_default_not_to_unbounded(self, tmp_path, monkeypatch):
        """A junk config line is a latency regression, never a silent revert to walking everything."""
        _write_config(tmp_path, "computer_use:\n  ax_max_elements: plenty\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert cua_backend._cua_configured_ax_max_elements() == cua_backend._DEFAULT_AX_MAX_ELEMENTS

    def test_negative_values_are_clamped(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "computer_use:\n  ax_max_elements: -5\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert cua_backend._cua_configured_ax_max_elements() == 0

    def test_bound_rides_alongside_the_target(self, tmp_path, monkeypatch):
        """The bound is an addition: pid/window_id addressing is untouched."""
        _write_config(tmp_path, "computer_use:\n  ax_max_elements: 250\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        args = _StubCapture()._gws_args()
        assert args["pid"] == 607 and args["window_id"] == 382
        assert args["max_elements"] == 250
