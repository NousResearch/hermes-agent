"""Bound on the driver's accessibility-tree walk per capture (``computer_use.ax_max_elements``).

``_DEFAULT_MAX_ELEMENTS`` in ``tool.py`` caps the SURFACED element list; the walk that produces it was
unbounded, so a capture paid for every node in the target's tree before trimming. A bound is pure
latency: the bounded tree is a prefix of the unbounded one, so the elements the model sees are
unchanged. Disabled by 0 (driver default), tuned by config, and read through the real loader.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


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


class TestCappedWalkHint:
    """When the driver's walk stopped at the bound, the spill file is not the full tree — say so."""

    @staticmethod
    def _summary(monkeypatch, n_elements: int, bound: int) -> str:
        from tools.computer_use import tool
        from tools.computer_use.backend import CaptureResult, UIElement
        monkeypatch.setattr(cua_backend, "_cua_configured_ax_max_elements", lambda: bound)
        monkeypatch.setattr(tool, "_spill_elements_to_file", lambda cap: "/tmp/elements.json")
        cap = CaptureResult(mode="ax", width=800, height=600,
                            elements=[UIElement(index=i + 1, role="AXButton", label=f"b{i}") for i in range(n_elements)])
        return "\n".join(tool._capture_summary_lines(tool._capture_view(cap, max_elements=5)))

    def test_capped_walk_is_named_and_full_is_dropped(self, monkeypatch):
        text = self._summary(monkeypatch, n_elements=8, bound=8)
        assert "accessibility walk capped at 8 elements; pass app= to narrow" in text
        assert "full element tree" not in text
        assert "element tree with untruncated labels saved to" in text

    def test_uncapped_walk_still_promises_the_full_tree(self, monkeypatch):
        text = self._summary(monkeypatch, n_elements=8, bound=50)
        assert "accessibility walk capped" not in text
        assert "full element tree with untruncated labels saved to" in text

    def test_zero_bound_never_claims_a_cap(self, monkeypatch):
        assert "accessibility walk capped" not in self._summary(monkeypatch, n_elements=8, bound=0)
