"""Regression tests for Windows HiDPI coordinate normalization (#94538).

cua-driver's Windows backend captures screenshots in physical pixels while
its input dispatch (SendInput) operates in logical units, so on a scaled
display (e.g. 150%) a click at screenshot coordinate ``[x, y]`` lands at
``[x * scale, y * scale]`` and misses the intended element. The Hermes
wrapper must divide coordinate inputs by the display scale factor
(DPI / 96) before dispatching them to the driver.

These tests pin that contract without needing a live cua-driver binary:
the DPI probe is patched and the MCP args the backend would send are
asserted directly. The Win32 ctypes probe chain (EnumWindows →
GetDpiForWindow → GetDpiForSystem) is exercised with a fake ``ctypes``
module, and the #94666 review findings (per-capture re-probe, retry of
failed probes, kill-switch truthiness, scroll passthrough, concurrency)
are covered in dedicated cases below.
"""

from __future__ import annotations

import sys
import threading
import types
from typing import Any, Dict, Optional

import pytest


class _FakeSession:
    """Minimal stand-in for ``_CuaDriverSession`` recording tool-call args."""

    def __init__(
        self,
        out: Optional[Dict[str, Any]] = None,
        *,
        scroll_coords: bool = True,
    ) -> None:
        self.out = out or {
            "isError": False,
            "data": {},
            "structuredContent": {"effect": "confirmed"},
        }
        self.scroll_coords = scroll_coords
        self.calls = []  # type: list[tuple[str, Dict[str, Any]]]

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0):
        self.calls.append((name, dict(args)))
        return self.out

    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        return self.scroll_coords and capability == "input.scroll.coordinates"

    def supports_input_property(self, tool: str, prop: str) -> bool:
        return False

    def _has_tool(self, name: str) -> bool:
        return True


class _ProbeSpy:
    """Stand-in for ``_win32_dpi_probe`` recording pids and replaying results.

    Each element is either ``None`` (probe failure) or an ``(hwnd, scale)``
    tuple, consumed in order — a failed probe is never retained.
    """

    def __init__(self, results):
        self.results = list(results)
        self.calls = []  # type: list[Optional[int]]

    def __call__(self, pid):
        self.calls.append(pid)
        if not self.results:
            return None
        return self.results.pop(0)


def _make_backend(session: _FakeSession):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = session
    backend._session_id = "hermes-session"
    backend._snapshot_tokens = {}
    backend._active_pid = 4242
    backend._active_window_id = 77
    backend._capture_dpi_scale = None
    backend._dpi_scale_lock = threading.Lock()
    return backend


def _last_call(session: _FakeSession):
    assert session.calls, "expected a cua-driver tool call"
    return session.calls[-1]


@pytest.fixture(autouse=True)
def _sanitize_env(monkeypatch):
    """The kill switch is env/config driven; keep each test opt-in clean."""
    monkeypatch.delenv("HERMES_CUA_NO_DPI_NORMALIZATION", raising=False)
    yield


def _patch_scale(monkeypatch, scale):
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_win32_dpi_probe", lambda pid: (None, scale))
    return cua_backend


# ── Coordinate normalization ──────────────────────────────────────────


def test_click_coordinates_are_divided_by_scale_on_windows_hidpi(monkeypatch):
    """150% DPI: screenshot-space [300, 450] must dispatch as [200, 300]."""
    _patch_scale(monkeypatch, 1.5)
    session = _FakeSession()
    backend = _make_backend(session)

    result = backend.click(x=300, y=450)

    assert result.ok
    _, args = _last_call(session)
    assert args["x"] == 200
    assert args["y"] == 300


def test_double_click_coordinates_are_normalized(monkeypatch):
    _patch_scale(monkeypatch, 1.5)
    session = _FakeSession()
    backend = _make_backend(session)

    backend.click(x=300, y=450, click_count=2)

    name, args = _last_call(session)
    assert name == "double_click"
    assert args["x"] == 200
    assert args["y"] == 300


def test_drag_coordinates_are_normalized(monkeypatch):
    _patch_scale(monkeypatch, 1.5)
    session = _FakeSession()
    backend = _make_backend(session)

    backend.drag(from_xy=(300, 450), to_xy=(600, 900))

    _, args = _last_call(session)
    assert (args["from_x"], args["from_y"]) == (200, 300)
    assert (args["to_x"], args["to_y"]) == (400, 600)


def test_scroll_coordinates_are_normalized(monkeypatch):
    _patch_scale(monkeypatch, 1.5)
    session = _FakeSession(scroll_coords=True)
    backend = _make_backend(session)

    backend.scroll(direction="down", amount=3, x=300, y=450)

    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (200, 300)


def test_normalization_rounds_to_nearest_integer(monkeypatch):
    """445 / 1.5 = 296.67 → 297, matching the issue's 285/1.5-style math."""
    _patch_scale(monkeypatch, 1.5)
    session = _FakeSession()
    backend = _make_backend(session)

    backend.click(x=445, y=285)

    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (297, 190)


def test_coordinates_unchanged_when_scale_is_unavailable(monkeypatch):
    """A driver/host that cannot report DPI must behave exactly as before."""
    _patch_scale(monkeypatch, None)
    session = _FakeSession()
    backend = _make_backend(session)

    backend.click(x=300, y=450)
    backend.drag(from_xy=(300, 450), to_xy=(600, 900))
    backend.scroll(direction="down", amount=3, x=300, y=450)

    _, click_args = session.calls[0]
    assert (click_args["x"], click_args["y"]) == (300, 450)
    _, drag_args = session.calls[1]
    assert (drag_args["from_x"], drag_args["from_y"]) == (300, 450)
    assert (drag_args["to_x"], drag_args["to_y"]) == (600, 900)
    _, scroll_args = session.calls[2]
    assert (scroll_args["x"], scroll_args["y"]) == (300, 450)


def test_coordinates_unchanged_at_100_percent_scale(monkeypatch):
    _patch_scale(monkeypatch, 1.0)
    session = _FakeSession()
    backend = _make_backend(session)

    backend.click(x=300, y=450)

    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (300, 450)


def test_scroll_coordinates_pass_through_untruncated_at_identity_scale(monkeypatch):
    """#94666 review: at scale 1.0/None scroll coordinates must pass through
    untouched — no int truncation (299.7 stays 299.7, not 299)."""
    _patch_scale(monkeypatch, None)
    session = _FakeSession()
    backend = _make_backend(session)

    backend.scroll(direction="down", amount=3, x=299.7, y=3.4)

    _, args = _last_call(session)
    assert args["x"] == 299.7 and isinstance(args["x"], float)
    assert args["y"] == 3.4 and isinstance(args["y"], float)


def test_env_kill_switch_disables_normalization(monkeypatch):
    _patch_scale(monkeypatch, 1.5)
    monkeypatch.setenv("HERMES_CUA_NO_DPI_NORMALIZATION", "1")
    session = _FakeSession()
    backend = _make_backend(session)

    backend.click(x=300, y=450)

    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (300, 450)


def test_config_kill_switch_disables_normalization(monkeypatch):
    from tools.computer_use import cua_backend

    _patch_scale(monkeypatch, 1.5)
    monkeypatch.setattr(
        cua_backend, "_computer_use_cfg", lambda: {"dpi_normalization": False}
    )
    session = _FakeSession()
    backend = _make_backend(session)

    backend.click(x=300, y=450)

    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (300, 450)


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE", "Yes", "anything", "2"])
def test_env_kill_switch_accepts_any_nonempty_truthy_value(monkeypatch, raw):
    """#94666 review: the docs say 'any truthy value'; the code must accept
    arbitrary non-empty spellings, not just the 4-value whitelist."""
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {})
    monkeypatch.setenv("HERMES_CUA_NO_DPI_NORMALIZATION", raw)
    assert cua_backend._cua_dpi_normalization_disabled() is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "   "])
def test_env_kill_switch_ignores_falsy_values(monkeypatch, raw):
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {})
    monkeypatch.setenv("HERMES_CUA_NO_DPI_NORMALIZATION", raw)
    assert cua_backend._cua_dpi_normalization_disabled() is False


def test_env_kill_switch_unset_keeps_normalization_enabled(monkeypatch):
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {})
    assert cua_backend._cua_dpi_normalization_disabled() is False


def test_element_index_clicks_are_not_scaled(monkeypatch):
    """Element clicks resolve server-side in the driver's own space."""
    _patch_scale(monkeypatch, 1.5)
    session = _FakeSession()
    backend = _make_backend(session)

    backend.click(element=3)

    _, args = _last_call(session)
    assert args["element_index"] == 3
    assert "x" not in args and "y" not in args


# ── #94666 review: cache invalidation / retry semantics ────────────────


def test_recapture_reprobes_and_never_reuses_stale_scale(monkeypatch):
    """#94666 review ①: the scale must match the capture. A re-capture, or a
    window moved to another monitor (same pid, different DPI), rebinds a
    fresh factor — the previous capture's scale is never reused."""
    from tools.computer_use import cua_backend

    spy = _ProbeSpy([(None, 1.25), (None, 1.5)])
    monkeypatch.setattr(cua_backend, "_win32_dpi_probe", spy)
    session = _FakeSession()
    backend = _make_backend(session)

    backend._dpi_bind_for_capture()  # first capture: monitor A at 125%
    backend.click(x=400, y=500)
    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (320, 400)

    backend._dpi_bind_for_capture()  # re-capture: window now on monitor B at 150%
    backend.click(x=300, y=450)
    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (200, 300)
    assert spy.calls == [4242, 4242]  # exactly one probe per capture


def test_failed_probe_is_retried_not_cached(monkeypatch):
    """#94666 review ②: a failed probe must not be pinned forever — the next
    action or capture probes again."""
    from tools.computer_use import cua_backend

    spy = _ProbeSpy([None, None, (None, 1.5)])
    monkeypatch.setattr(cua_backend, "_win32_dpi_probe", spy)
    session = _FakeSession()
    backend = _make_backend(session)

    backend._dpi_bind_for_capture()  # capture: probe fails
    backend.click(x=300, y=450)      # click re-probes, still fails → passthrough
    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (300, 450)

    backend._dpi_bind_for_capture()  # next capture: probe succeeds now
    backend.click(x=300, y=450)
    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (200, 300)
    assert spy.calls == [4242, 4242, 4242]


def test_focus_app_retarget_invalidates_capture_bound_scale(monkeypatch):
    """A target change outside capture() must not reuse the old capture's
    scale — the next coordinate action probes the new window."""
    from tools.computer_use import cua_backend

    spy = _ProbeSpy([(None, 1.25), (None, 2.0)])
    monkeypatch.setattr(cua_backend, "_win32_dpi_probe", spy)
    monkeypatch.setattr(
        cua_backend.CuaDriverBackend,
        "_load_windows",
        lambda self: [{"pid": 999, "window_id": 88, "app_name": "Other"}],
    )
    monkeypatch.setattr(
        cua_backend.CuaDriverBackend,
        "_match_windows_for_app",
        lambda self, windows, app: [{"pid": 999, "window_id": 88, "app_name": "Other"}],
    )
    session = _FakeSession()
    backend = _make_backend(session)

    backend._dpi_bind_for_capture()  # capture binds 1.25 for pid 4242
    backend.click(x=400, y=500)
    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (320, 400)

    result = backend.focus_app("Other")
    assert result.ok
    assert backend._active_pid == 999

    backend.click(x=400, y=500)  # new target: re-probe, never reuse 1.25
    _, args = _last_call(session)
    assert (args["x"], args["y"]) == (200, 250)
    assert spy.calls == [4242, 999]


def test_concurrent_coordinate_actions_are_safe(monkeypatch):
    """#94666 review (concurrency): parallel click/scroll threads must not
    race on the DPI scale binding (the old lock-free per-pid dict could
    raise RuntimeError 'dictionary changed size during iteration'). With
    the lock + scalar binding, exactly one probe runs and every dispatched
    coordinate is consistent."""
    from tools.computer_use import cua_backend

    probe_calls = []
    probe_lock = threading.Lock()

    def fake_probe(pid):
        with probe_lock:
            probe_calls.append(pid)
        return (None, 1.5)

    monkeypatch.setattr(cua_backend, "_win32_dpi_probe", fake_probe)
    session = _FakeSession()
    backend = _make_backend(session)
    errors = []

    def worker():
        try:
            for _ in range(20):
                backend.click(x=300, y=450)
                backend.scroll(direction="down", amount=3, x=300, y=450)
        except Exception as exc:  # pragma: no cover - failure signal
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(session.calls) == 8 * 20 * 2
    assert len(probe_calls) == 1  # bound once; every later call reads the binding
    for name, args in session.calls:
        assert (args["x"], args["y"]) == (200, 300)


# ── Win32 ctypes probe chain (fake ctypes module) ──────────────────────


class _FakeDWORD:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class _FakeWintypes:
    BOOL = bool
    HWND = int
    LPARAM = int
    DWORD = _FakeDWORD


class _FakeUser32:
    """Records calls and drives the EnumWindows/DPI chain deterministically."""

    def __init__(
        self,
        windows,
        *,
        dpi_window=None,
        dpi_system: int = 0,
        raise_window_dpi: bool = False,
    ) -> None:
        self.windows = list(windows)
        self.by_hwnd = {w["hwnd"]: w for w in self.windows}
        self.dpi_window = dict(dpi_window or {})
        self.dpi_system = dpi_system
        self.raise_window_dpi = raise_window_dpi
        self.enumerated = []  # hwnds visited by the EnumWindows callback
        self.dpi_window_calls = []
        self.dpi_system_calls = 0

    def IsWindowVisible(self, hwnd):
        return bool(self.by_hwnd[hwnd]["visible"])

    def GetWindowThreadProcessId(self, hwnd, proc_id_ref):
        proc_id_ref.value = self.by_hwnd[hwnd]["pid"]
        return 1

    def EnumWindows(self, callback, lparam):
        for window in self.windows:
            self.enumerated.append(window["hwnd"])
            if not callback(window["hwnd"], 0):
                break
        return True

    def GetDpiForWindow(self, hwnd):
        self.dpi_window_calls.append(hwnd)
        if self.raise_window_dpi:
            raise OSError("GetDpiForWindow unavailable")
        return self.dpi_window.get(hwnd, 0)

    def GetDpiForSystem(self):
        self.dpi_system_calls += 1
        return self.dpi_system


class _FakeCtypes:
    """Minimal ctypes surface the probe uses: WINFUNCTYPE, byref, windll."""

    def __init__(self, user32: _FakeUser32) -> None:
        self.wintypes = _FakeWintypes
        self.windll = types.SimpleNamespace(user32=user32)

    def WINFUNCTYPE(self, *restype):
        return lambda fn: fn  # keep the callback alive, pass it through

    def byref(self, obj):
        return obj


def _install_fake_ctypes(monkeypatch, user32):
    """Swap the real ctypes for a fake and pretend we're on win32."""
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "ctypes", _FakeCtypes(user32))
    return user32


def test_win32_hwnd_for_pid_stops_at_first_visible_match(monkeypatch):
    user32 = _FakeUser32(
        [
            {"hwnd": 11, "pid": 999, "visible": False},
            {"hwnd": 22, "pid": 4242, "visible": True},
            {"hwnd": 33, "pid": 4242, "visible": True},
        ]
    )
    _install_fake_ctypes(monkeypatch, user32)
    from tools.computer_use import cua_backend

    assert cua_backend._win32_hwnd_for_pid(user32, 4242) == 22
    assert user32.enumerated == [11, 22]  # enumeration stops at the match


def test_win32_hwnd_for_pid_returns_none_without_visible_match(monkeypatch):
    user32 = _FakeUser32(
        [
            {"hwnd": 11, "pid": 999, "visible": False},
            {"hwnd": 22, "pid": 555, "visible": True},
        ]
    )
    _install_fake_ctypes(monkeypatch, user32)
    from tools.computer_use import cua_backend

    assert cua_backend._win32_hwnd_for_pid(user32, 4242) is None
    assert user32.enumerated == [11, 22]


def test_win32_dpi_probe_prefers_get_dpi_for_window(monkeypatch):
    user32 = _FakeUser32(
        [{"hwnd": 22, "pid": 4242, "visible": True}],
        dpi_window={22: 144},
        dpi_system=96,
    )
    _install_fake_ctypes(monkeypatch, user32)
    from tools.computer_use import cua_backend

    assert cua_backend._win32_dpi_probe(4242) == (22, 1.5)
    assert user32.dpi_window_calls == [22]
    assert user32.dpi_system_calls == 0  # per-window DPI wins


def test_win32_dpi_probe_falls_back_to_system_dpi_when_window_dpi_zero(monkeypatch):
    user32 = _FakeUser32(
        [{"hwnd": 22, "pid": 4242, "visible": True}],
        dpi_window={22: 0},
        dpi_system=120,
    )
    _install_fake_ctypes(monkeypatch, user32)
    from tools.computer_use import cua_backend

    assert cua_backend._win32_dpi_probe(4242) == (22, 1.25)
    assert user32.dpi_window_calls == [22]
    assert user32.dpi_system_calls == 1


def test_win32_dpi_probe_falls_back_when_window_probe_raises(monkeypatch):
    user32 = _FakeUser32(
        [{"hwnd": 22, "pid": 4242, "visible": True}],
        dpi_system=96,
        raise_window_dpi=True,
    )
    _install_fake_ctypes(monkeypatch, user32)
    from tools.computer_use import cua_backend

    assert cua_backend._win32_dpi_probe(4242) == (22, 1.0)
    assert user32.dpi_system_calls == 1


def test_win32_dpi_probe_uses_system_dpi_only_when_no_window(monkeypatch):
    user32 = _FakeUser32([{"hwnd": 22, "pid": 999, "visible": True}], dpi_system=168)
    _install_fake_ctypes(monkeypatch, user32)
    from tools.computer_use import cua_backend

    assert cua_backend._win32_dpi_probe(4242) == (None, 1.75)
    assert user32.dpi_window_calls == []
    assert user32.dpi_system_calls == 1


def test_win32_dpi_probe_returns_none_when_dpi_missing(monkeypatch):
    user32 = _FakeUser32([], dpi_system=0)
    _install_fake_ctypes(monkeypatch, user32)
    from tools.computer_use import cua_backend

    assert cua_backend._win32_dpi_probe(4242) is None


def test_win32_dpi_probe_rejects_implausible_dpi(monkeypatch):
    user32 = _FakeUser32([], dpi_system=9600)  # 100x scale — probe garbage
    _install_fake_ctypes(monkeypatch, user32)
    from tools.computer_use import cua_backend

    assert cua_backend._win32_dpi_probe(4242) is None


def test_win32_dpi_scale_wraps_probe(monkeypatch):
    user32 = _FakeUser32(
        [{"hwnd": 22, "pid": 4242, "visible": True}],
        dpi_window={22: 144},
        dpi_system=96,
    )
    _install_fake_ctypes(monkeypatch, user32)
    from tools.computer_use import cua_backend

    assert cua_backend._win32_dpi_scale(4242) == 1.5


# ── Pure helpers ───────────────────────────────────────────────────────


def test_win32_dpi_scale_returns_none_off_windows(monkeypatch):
    """The raw probe must be a no-op on non-Windows hosts."""
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend.sys, "platform", "linux")
    assert cua_backend._win32_dpi_scale(4242) is None
    assert cua_backend._win32_dpi_probe(4242) is None


def test_normalize_coordinate_helper(monkeypatch):
    from tools.computer_use.cua_backend import _normalize_coordinate

    assert _normalize_coordinate(300, 1.5) == 200
    assert _normalize_coordinate(445, 1.5) == 297
    assert _normalize_coordinate(230, 1.25) == 184
    assert _normalize_coordinate(0, 2.0) == 0
    # Identity scale passes values through untouched — no int truncation.
    assert _normalize_coordinate(300, 1.0) == 300
    assert _normalize_coordinate(300, None) == 300
    assert _normalize_coordinate(299.7, 1.0) == 299.7
    assert _normalize_coordinate(299.7, None) == 299.7
    assert _normalize_coordinate(None, 1.0) is None
    assert _normalize_coordinate(None, None) is None
    assert _normalize_coordinate(None, 1.5) is None
