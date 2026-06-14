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


# ---------------------------------------------------------------------------
# Overlay client — gating and fail-safety (any platform)
# ---------------------------------------------------------------------------

class TestOverlayClient:
    def test_env_kill_switch_disables_overlay(self, monkeypatch):
        monkeypatch.setenv("HERMES_COMPUTER_USE_OVERLAY", "0")
        from tools.computer_use.windows_backend import _OverlayClient
        client = _OverlayClient()
        client.start()                       # must not spawn anything
        assert client._proc is None
        assert client.pid is None
        client.send({"cmd": "flash"})        # must be a silent no-op
        client.stop()

    def test_send_before_start_is_noop(self):
        from tools.computer_use.windows_backend import _OverlayClient
        client = _OverlayClient()
        client.send({"cmd": "click", "x": 1, "y": 2})  # no socket yet — no raise
        assert client.pid is None

    def test_backend_constructs_overlay_client(self):
        backend = _fresh_backend()
        assert hasattr(backend, "_overlay")
        # Overlay failures must never surface through backend actions: a dead
        # client swallows sends.
        backend._overlay._dead = True
        backend._overlay.send({"cmd": "flash"})


# ---------------------------------------------------------------------------
# Vision downscale helper (any platform; needs Pillow, a core dependency)
# ---------------------------------------------------------------------------

class TestShrinkCaptureForVision:
    @staticmethod
    def _png_bytes(w, h):
        pil = pytest.importorskip("PIL.Image")
        import io
        buf = io.BytesIO()
        pil.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
        return buf.getvalue()

    def test_oversized_image_is_downscaled(self):
        from PIL import Image
        import io
        from tools.computer_use.tool import _shrink_capture_for_vision
        raw = self._png_bytes(1920, 1080)
        out = _shrink_capture_for_vision(raw, ".png", max_dim=1456)
        img = Image.open(io.BytesIO(out))
        assert max(img.size) == 1456
        assert img.size == (1456, 819)       # aspect ratio preserved

    def test_small_image_passes_through_unchanged(self):
        from tools.computer_use.tool import _shrink_capture_for_vision
        raw = self._png_bytes(800, 600)
        assert _shrink_capture_for_vision(raw, ".png", max_dim=1456) is raw

    def test_garbage_bytes_return_unchanged(self):
        from tools.computer_use.tool import _shrink_capture_for_vision
        raw = b"not an image at all"
        assert _shrink_capture_for_vision(raw, ".png") is raw


# ---------------------------------------------------------------------------
# Hardening: vision-down fallback, stale-coordinate translation, idle guard
# ---------------------------------------------------------------------------

class TestVisionDownFallback:
    def test_capture_degrades_to_text_when_aux_vision_fails(self, monkeypatch):
        """Routing requested (text-only main) + vision down => AX text payload,
        never a multimodal envelope a text model can't consume."""
        from tools.computer_use import tool
        from tools.computer_use.backend import CaptureResult, UIElement
        monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: True)
        monkeypatch.setattr(tool, "_route_capture_through_aux_vision",
                            lambda cap, summary: None)
        # Must be >= 8x8 or _capture_response's provider-minimum check skips
        # the vision branch entirely before the fallback we're testing.
        import base64
        import io
        pil = pytest.importorskip("PIL.Image")
        buf = io.BytesIO()
        pil.new("RGB", (16, 16), (40, 40, 40)).save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        cap = CaptureResult(
            mode="som", width=800, height=600, png_b64=png_b64,
            elements=[UIElement(index=1, role="Button", label="OK",
                                bounds=(1, 2, 3, 4))],
            app="x.exe", window_title="W")
        resp = tool._capture_response(cap)
        assert isinstance(resp, str), "must be a text payload, not multimodal"
        body = json.loads(resp)
        assert body["vision_unavailable"] is True
        assert body["elements"][0]["index"] == 1
        assert "Element-index actions still work" in body["summary"]


class TestStaleCoordinateTranslation:
    def _backend_with_element(self, monkeypatch, new_rect):
        from tools.computer_use import windows_backend as wb
        from tools.computer_use.backend import UIElement
        b = wb.WindowsUIABackend()
        b._elements[1] = UIElement(index=1, role="Button", label="OK",
                                   bounds=(110, 220, 100, 50), window_id=777)
        b._capture_rect = (100, 200, 640, 480)
        monkeypatch.setattr(wb, "win32gui",
                            types.SimpleNamespace(IsWindow=lambda h: True),
                            raising=False)
        monkeypatch.setattr(wb, "_window_rect", lambda h: new_rect)
        return b

    def test_window_moved_translates_click_point(self, monkeypatch):
        b = self._backend_with_element(monkeypatch, (130, 250, 640, 480))
        x, y, what = b._resolve_point(1, None, None)
        assert (x, y) == (160 + 30, 245 + 50)   # center (160,245) + delta (30,50)
        assert "window moved" in what

    def test_window_unmoved_uses_cached_center(self, monkeypatch):
        b = self._backend_with_element(monkeypatch, (100, 200, 640, 480))
        x, y, _ = b._resolve_point(1, None, None)
        assert (x, y) == (160, 245)

    def test_window_resized_demands_recapture(self, monkeypatch):
        b = self._backend_with_element(monkeypatch, (100, 200, 800, 480))
        res = b.click(element=1)
        assert not res.ok
        assert "resized" in res.message


class TestIdleGuard:
    def test_zero_threshold_disables_guard(self, monkeypatch):
        from tools.computer_use import windows_backend as wb
        monkeypatch.setenv("HERMES_COMPUTER_USE_IDLE_WAIT", "0")
        wb._wait_for_user_idle()   # must return immediately, touch no win32

    def test_returns_once_user_is_idle(self, monkeypatch):
        from tools.computer_use import windows_backend as wb
        monkeypatch.setenv("HERMES_COMPUTER_USE_IDLE_WAIT", "1.5")
        monkeypatch.setattr(wb, "_seconds_since_user_input", lambda: 99.0)
        import time as _t
        t0 = _t.monotonic()
        wb._wait_for_user_idle()
        assert _t.monotonic() - t0 < 1.0


# ---------------------------------------------------------------------------
# Input-state safety: a failed pointer action must never strand modifiers
# (or, for drag, the mouse button) in the held-down state. Regression guard
# for the try/finally release in click/drag/scroll.
# ---------------------------------------------------------------------------

def _raise(*_a, **_k):
    raise RuntimeError("synthetic injection failure")


class TestInputStateReleasedOnFailure:
    def _backend(self, monkeypatch):
        from tools.computer_use import windows_backend as wb
        b = wb.WindowsUIABackend()
        b._elements[1] = UIElement(index=1, role="Button", label="OK",
                                   bounds=(110, 220, 100, 50), window_id=777)
        b._capture_rect = (100, 200, 640, 480)
        # Window present and unmoved -> _resolve_point yields the cached center.
        monkeypatch.setattr(wb, "win32gui",
                            types.SimpleNamespace(IsWindow=lambda h: True),
                            raising=False)
        monkeypatch.setattr(wb, "_window_rect", lambda h: (100, 200, 640, 480),
                            raising=False)
        monkeypatch.setattr(wb, "win32api",
                            types.SimpleNamespace(GetCursorPos=lambda: (5, 5)),
                            raising=False)
        monkeypatch.setattr(b, "_overlay",
                            types.SimpleNamespace(send=lambda *a, **k: None, pid=0))
        monkeypatch.setattr(b, "_ensure_target_foreground", lambda: None)
        # Tag the modifier down/up batches so we can assert both were sent.
        monkeypatch.setattr(b, "_with_modifiers",
                            lambda modifiers=None: (["DOWN"], ["UP"]))
        return wb, b

    def test_click_releases_modifiers_when_action_fails(self, monkeypatch):
        wb, b = self._backend(monkeypatch)
        sent = []
        monkeypatch.setattr(wb, "_send_inputs", lambda batch: sent.append(batch))
        monkeypatch.setattr(wb, "_mouse_move", lambda *a, **k: None)
        monkeypatch.setattr(wb, "_mouse_button", _raise)
        res = b.click(element=1, modifiers=["ctrl"])
        assert not res.ok
        assert sent == [["DOWN"], ["UP"]], "mods_up must run in the finally"

    def test_click_releases_modifiers_on_success(self, monkeypatch):
        wb, b = self._backend(monkeypatch)
        sent = []
        monkeypatch.setattr(wb, "_send_inputs", lambda batch: sent.append(batch))
        monkeypatch.setattr(wb, "_mouse_move", lambda *a, **k: None)
        monkeypatch.setattr(wb, "_mouse_button", lambda *a, **k: None)
        res = b.click(element=1, modifiers=["ctrl"])
        assert res.ok
        assert sent == [["DOWN"], ["UP"]]

    def test_drag_releases_button_and_modifiers_midway(self, monkeypatch):
        wb, b = self._backend(monkeypatch)
        sent, buttons = [], []
        calls = {"moves": 0}

        def _mv(*_a, **_k):
            calls["moves"] += 1
            if calls["moves"] == 2:        # 1 = move to start, 2 = first drag step
                raise RuntimeError("boom mid-drag")

        monkeypatch.setattr(wb, "_send_inputs", lambda batch: sent.append(batch))
        monkeypatch.setattr(wb, "_mouse_move", _mv)
        monkeypatch.setattr(wb, "_mouse_button",
                            lambda button, down: buttons.append((button, down)))
        res = b.drag(from_element=1, to_xy=(300, 400), modifiers=["alt"])
        assert not res.ok
        # Primary button was pressed, then released in the finally; mods released.
        assert ("left", True) in buttons
        assert buttons[-1] == ("left", False)
        assert sent == [["DOWN"], ["UP"]]

    def test_scroll_releases_modifiers_when_wheel_fails(self, monkeypatch):
        wb, b = self._backend(monkeypatch)
        sent = []
        monkeypatch.setattr(wb, "_send_inputs", lambda batch: sent.append(batch))
        monkeypatch.setattr(wb, "_mouse_move", lambda *a, **k: None)
        monkeypatch.setattr(wb, "_mouse_wheel", _raise)
        res = b.scroll(direction="down", element=1, modifiers=["shift"])
        assert not res.ok
        assert sent == [["DOWN"], ["UP"]]


# ---------------------------------------------------------------------------
# Shared tree-walk: capture (_walk_elements) and set_value (_control_at_index)
# consume one generator (_iter_interactable), so an element index resolves to
# the same control in both, discovery is breadth-first, and the filter is
# applied identically. Guards findings #2 (deque) and #3 (single walk).
# ---------------------------------------------------------------------------

class _FakeRect:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom


class _FakeCtrl:
    def __init__(self, role, name="", rect=(0, 0, 10, 10), enabled=True,
                 offscreen=False, children=None, automation_id="", patterns=None):
        self.ControlTypeName = role
        self.Name = name
        self.AutomationId = automation_id
        self.IsEnabled = enabled
        self.IsOffscreen = offscreen
        self.BoundingRectangle = _FakeRect(*rect)
        self._children = list(children or [])
        self._patterns = patterns or {}

    def GetChildren(self):
        return list(self._children)

    def GetPattern(self, pid):
        return self._patterns.get(pid)


class _FakeInitializer:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _fake_auto(root):
    return types.SimpleNamespace(
        ControlFromHandle=lambda hwnd: root,
        UIAutomationInitializerInThread=_FakeInitializer,
        PatternId=types.SimpleNamespace(ValuePattern=1, InvokePattern=2),
    )


_WIDE = (0, 0, 1000, 1000)


class TestSharedTreeWalk:
    def _install(self, monkeypatch, root):
        from tools.computer_use import windows_backend as wb
        monkeypatch.setattr(wb, "_auto", _fake_auto(root), raising=False)
        monkeypatch.setattr(wb, "_window_rect", lambda hwnd: _WIDE, raising=False)
        return wb, wb.WindowsUIABackend()

    def test_capture_and_set_value_resolve_same_control(self, monkeypatch):
        beta = _FakeCtrl("EditControl", name="Beta", rect=(10, 40, 60, 70))
        gamma = _FakeCtrl("ButtonControl", name="Gamma", offscreen=True)
        mid = _FakeCtrl("PaneControl", children=[gamma, beta])
        alpha = _FakeCtrl("ButtonControl", name="Alpha", rect=(10, 10, 60, 30))
        delta = _FakeCtrl("ButtonControl", name="Delta", enabled=False)
        root = _FakeCtrl("PaneControl", children=[alpha, mid, delta])
        wb, b = self._install(monkeypatch, root)

        els = b._walk_elements(123, _WIDE)
        # Offscreen Gamma + disabled Delta filtered; BFS order Alpha then Beta.
        assert [e.label for e in els] == ["Alpha", "Beta"]
        assert [e.index for e in els] == [1, 2]
        # Every advertised index re-resolves to the SAME control.
        for e in els:
            ctrl = b._control_at_index(123, e.index)
            assert ctrl is not None and ctrl.Name == e.label
        assert b._control_at_index(123, 99) is None

    def test_discovery_order_is_breadth_first(self, monkeypatch):
        # BFS yields the shallow button before the nested one; a LIFO queue or
        # DFS would invert these, so this pins deque.popleft() ordering.
        deep = _FakeCtrl("ButtonControl", name="Second")
        sub = _FakeCtrl("PaneControl", children=[deep])
        first = _FakeCtrl("ButtonControl", name="First", rect=(20, 20, 40, 40))
        root = _FakeCtrl("PaneControl", children=[first, sub])
        wb, b = self._install(monkeypatch, root)
        assert [e.label for e in b._walk_elements(1, _WIDE)] == ["First", "Second"]

    def test_text_node_counts_only_with_value_pattern(self, monkeypatch):
        # A Text control is non-interactable unless it exposes Value/Invoke —
        # the special case must apply identically in the shared walk.
        plain = _FakeCtrl("TextControl", name="label")
        editable = _FakeCtrl("TextControl", name="field", rect=(0, 20, 30, 40),
                             patterns={1: object()})  # PatternId.ValuePattern
        root = _FakeCtrl("PaneControl", children=[plain, editable])
        wb, b = self._install(monkeypatch, root)
        els = b._walk_elements(1, _WIDE)
        assert [e.label for e in els] == ["field"]
        assert b._control_at_index(1, 1).Name == "field"
