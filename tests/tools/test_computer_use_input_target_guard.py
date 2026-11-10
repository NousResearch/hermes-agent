"""Input actions must not silently deliver to a different app than requested.

Live QA (Aug 2026, KDE desktop): with kcalc as the sticky target,
``type(text="777", app="kate")`` reported ok:true and typed 777 into
KCALC — app= was silently dropped on every input action. The guard
refuses provable mismatches with a one-call fix instruction. Also covers
the unknown-action suggestion hint (model emitted "hotkey" for "key").
"""
import json
from types import SimpleNamespace

from tools.computer_use.tool import (
    _dispatch,
    _input_target_mismatch,
)

def _backend(last_app):
    return SimpleNamespace(_last_app=last_app)

# ── mismatch predicate ──────────────────────────────────────────────────

def test_clear_mismatch_returns_current_app():
    assert _input_target_mismatch(_backend("kcalc"), "kate") == "kcalc"

def test_same_app_no_mismatch():
    assert _input_target_mismatch(_backend("kate"), "kate") is None

def test_substring_variants_no_mismatch():
    # list_windows names are localized/variant; never refuse fuzzy matches.
    assert _input_target_mismatch(_backend("Google-chrome"), "chrome") is None
    assert _input_target_mismatch(_backend("chrome"), "Google-chrome") is None

def test_unknown_current_target_fails_open():
    assert _input_target_mismatch(_backend(None), "kate") is None
    assert _input_target_mismatch(_backend(""), "kate") is None

# ── dispatch guard ──────────────────────────────────────────────────────

def _dispatch_result(backend, action, args):
    return json.loads(_dispatch(backend, action, args))

def test_type_with_mismatched_app_refused():
    backend = _backend("kcalc")
    out = _dispatch_result(backend, "type", {"text": "777", "app": "kate"})
    assert out["ok"] is False
    assert out["code"] == "input_target_mismatch"
    assert "kcalc" in out["error"] and "kate" in out["error"]
    assert "capture(app='kate')" in out["error"]

def test_click_with_mismatched_app_refused():
    out = _dispatch_result(_backend("kcalc"), "click", {"element": 3, "app": "kate"})
    assert out["code"] == "input_target_mismatch"

def _fake_action_result(action="type_text"):
    from tools.computer_use.backend import ActionResult

    return ActionResult(ok=True, action=action, message="done")

def test_type_with_matching_app_reaches_backend():
    backend = _backend("kate")
    calls = {}

    def _type_text(text, **kw):
        calls["text"] = text
        return _fake_action_result()

    backend.type_text = _type_text
    out = _dispatch_result(backend, "type", {"text": "hi", "app": "kate"})
    assert calls["text"] == "hi"
    assert out.get("ok") is True

def test_type_without_app_unchanged():
    """Legacy flows that never pass app= keep working (fail open)."""
    backend = _backend("kcalc")
    backend.type_text = lambda text, **kw: _fake_action_result()
    out = _dispatch_result(backend, "type", {"text": "9"})
    assert out.get("ok") is True

def test_unqualified_capture_retargets_the_guard_to_the_captured_app(monkeypatch):
    """capture(app='kate') then capture() of kcalc: input sent with app='kate' would reach kcalc."""
    from tools.computer_use import cua_backend_capture
    from tools.computer_use.cua_backend_capture import _CaptureMixin

    windows = {"kate": {"app_name": "kate", "pid": 11, "window_id": 1},
               "kcalc": {"app_name": "kcalc", "pid": 22, "window_id": 2}}

    class _Backend(_CaptureMixin):
        _last_app = None

        def _resolve_capture_windows(self, mode, app, pid, window_id):
            return [{**windows[app or "kcalc"], "title": "", "z_index": 1, "off_screen": False}]

        def _set_active_target(self, target):
            self._active_pid, self._active_window_id = target["pid"], target["window_id"]

        def _capture_vision(self):
            return None, None, [], ""

    monkeypatch.setattr(cua_backend_capture, "_linux_x11_active_window_id", lambda: None)
    backend = _Backend()
    backend.capture("vision", app="kate")
    backend.capture("vision")
    assert _dispatch_result(backend, "click", {"element": 3, "app": "kate"})["code"] == "input_target_mismatch"

# ── unknown-action suggestions ──────────────────────────────────────────
