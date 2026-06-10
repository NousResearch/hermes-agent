"""Tests for the Windows UIA backend (tools/computer_use/windows_backend.py).

Stubbing strategy: windows_backend guards its win32-only imports in a
module-level try/except, so the module itself imports on any platform. The
pure-logic tests below only exercise code paths that fail fast (key-name
mapping, stale-element resolution, length caps) before any win32 API is
touched, so they run on Linux CI. Wiring tests stub the whole
tools.computer_use.windows_backend module in sys.modules, so they never need
win32 either. Anything that would hit live UIA/SendInput is skipped off
Windows.
"""

from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import patch

import pytest

from tools.computer_use.backend import UIElement


@pytest.fixture(autouse=True)
def _reset_backend():
    """Tear down the cached backend between tests."""
    from tools.computer_use.tool import reset_backend_for_tests
    reset_backend_for_tests()
    yield
    reset_backend_for_tests()


def _fresh_backend():
    from tools.computer_use.windows_backend import WindowsUIABackend
    return WindowsUIABackend()


# ---------------------------------------------------------------------------
# Pure logic — runs on every platform
# ---------------------------------------------------------------------------

class TestVkForKey:
    def test_cmd_aliases_to_ctrl(self):
        from tools.computer_use.windows_backend import _vk_for_key
        assert _vk_for_key("cmd") == 0x11
        assert _vk_for_key("ctrl") == 0x11

    def test_win_super_meta_map_to_windows_key(self):
        from tools.computer_use.windows_backend import _vk_for_key
        assert _vk_for_key("win") == 0x5B
        assert _vk_for_key("super") == 0x5B
        assert _vk_for_key("meta") == 0x5B

    def test_named_keys(self):
        from tools.computer_use.windows_backend import _vk_for_key
        assert _vk_for_key("enter") == 0x0D
        assert _vk_for_key("return") == 0x0D
        assert _vk_for_key("f5") == 0x74
        assert _vk_for_key("a") == 0x41
        assert _vk_for_key("backspace") == 0x08
        assert _vk_for_key("delete") == 0x2E

    def test_unknown_multichar_key_is_none(self):
        from tools.computer_use.windows_backend import _vk_for_key
        assert _vk_for_key("florp") is None
        assert _vk_for_key("") is None


class TestFailFastPaths:
    def test_key_with_unknown_token_fails_naming_it(self):
        res = _fresh_backend().key("ctrl+florp")
        assert not res.ok
        assert "florp" in res.message

    def test_click_with_stale_element_index_fails_with_recapture_hint(self):
        res = _fresh_backend().click(element=999)
        assert not res.ok
        assert "re-run" in res.message or "capture" in res.message

    def test_click_without_target_fails(self):
        res = _fresh_backend().click()
        assert not res.ok

    def test_resolve_point_returns_element_center(self):
        b = _fresh_backend()
        b._elements[1] = UIElement(index=1, role="Button", label="OK",
                                   bounds=(10, 20, 100, 50))
        x, y, what = b._resolve_point(1, None, None)
        assert (x, y) == (60, 45)
        assert "#1" in what

    def test_resolve_point_passes_coordinates_through(self):
        x, y, _ = _fresh_backend()._resolve_point(None, 123, 456)
        assert (x, y) == (123, 456)

    def test_type_text_rejects_over_20000_chars(self):
        res = _fresh_backend().type_text("a" * 20001)
        assert not res.ok
        assert "20000" in res.message

    def test_set_value_requires_known_element(self):
        b = _fresh_backend()
        assert not b.set_value("x").ok
        assert not b.set_value("x", element=7).ok


class TestAvailability:
    def test_unavailable_off_windows(self, monkeypatch):
        from tools.computer_use import windows_backend
        monkeypatch.setattr(sys, "platform", "linux")
        assert not windows_backend.windows_backend_available()

    def test_unavailable_when_imports_failed(self, monkeypatch):
        from tools.computer_use import windows_backend
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(windows_backend, "_IMPORT_ERROR", ImportError("nope"))
        assert not windows_backend.windows_backend_available()


# ---------------------------------------------------------------------------
# Wiring — selector, check_fn, blocked combos (stubbed module, any platform)
# ---------------------------------------------------------------------------

class _FakeWindowsBackend:
    instances: list = []

    def __init__(self):
        self.started = False
        _FakeWindowsBackend.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        pass


def _stub_windows_module(monkeypatch, available=True):
    mod = types.ModuleType("tools.computer_use.windows_backend")
    mod.WindowsUIABackend = _FakeWindowsBackend
    mod.windows_backend_available = lambda: available
    monkeypatch.setitem(sys.modules, "tools.computer_use.windows_backend", mod)
    return mod


class TestWiring:
    def test_env_selects_windows_backend_and_starts_it(self, monkeypatch):
        _FakeWindowsBackend.instances = []
        _stub_windows_module(monkeypatch)
        with patch.dict(os.environ, {"HERMES_COMPUTER_USE_BACKEND": "windows"}):
            from tools.computer_use.tool import _get_backend
            backend = _get_backend()
        assert isinstance(backend, _FakeWindowsBackend)
        assert backend.started

    def test_default_backend_is_windows_on_win32(self, monkeypatch):
        from tools.computer_use.tool import _default_backend_name
        monkeypatch.setattr(sys, "platform", "win32")
        assert _default_backend_name() == "windows"
        monkeypatch.setattr(sys, "platform", "darwin")
        assert _default_backend_name() == "cua"

    def test_check_requirements_false_when_backend_unavailable(self, monkeypatch):
        _stub_windows_module(monkeypatch, available=False)
        monkeypatch.setattr(sys, "platform", "win32")
        from tools.computer_use.tool import check_computer_use_requirements
        assert not check_computer_use_requirements()

    def test_check_requirements_true_when_backend_available(self, monkeypatch):
        _stub_windows_module(monkeypatch, available=True)
        monkeypatch.setattr(sys, "platform", "win32")
        from tools.computer_use.tool import check_computer_use_requirements
        assert check_computer_use_requirements()


class TestWindowsBlockedCombos:
    @pytest.mark.parametrize("keys", ["win+l", "ctrl+alt+delete", "alt+f4",
                                      "windows+l", "super+L"])
    def test_blocked_combo_rejected_before_backend_exists(self, keys, monkeypatch):
        _FakeWindowsBackend.instances = []
        _stub_windows_module(monkeypatch)
        with patch.dict(os.environ, {"HERMES_COMPUTER_USE_BACKEND": "windows"}):
            from tools.computer_use.tool import handle_computer_use
            result = handle_computer_use({"action": "key", "keys": keys})
        payload = json.loads(result)
        assert "error" in payload
        assert "blocked" in payload["error"]
        assert _FakeWindowsBackend.instances == []

    def test_plain_save_combo_is_not_blocked(self, monkeypatch):
        """ctrl+s must reach the backend (sanity check the block list scope)."""
        _FakeWindowsBackend.instances = []
        mod = _stub_windows_module(monkeypatch)

        class _KeyBackend(_FakeWindowsBackend):
            def key(self, keys):
                from tools.computer_use.backend import ActionResult
                return ActionResult(ok=True, action="key", message=f"pressed {keys}")

        mod.WindowsUIABackend = _KeyBackend
        with patch.dict(os.environ, {"HERMES_COMPUTER_USE_BACKEND": "windows"}):
            from tools.computer_use.tool import handle_computer_use
            result = handle_computer_use({"action": "key", "keys": "ctrl+s"})
        payload = json.loads(result)
        assert payload.get("ok") is True


# ---------------------------------------------------------------------------
# Live (Windows only) — no input injection, read-only against the real OS
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows")
class TestLiveReadOnly:
    def test_list_apps_returns_real_windows(self):
        b = _fresh_backend()
        b.start()
        apps = b.list_apps()
        assert isinstance(apps, list)
        for entry in apps:
            assert {"app", "pid", "windows", "window_count"} <= set(entry)

    def test_capture_ax_of_foreground_window(self):
        b = _fresh_backend()
        b.start()
        cap = b.capture(mode="ax")
        assert cap.mode == "ax"
        assert cap.png_b64 is None
