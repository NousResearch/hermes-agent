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
    _INPUT_ACTIONS,
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


def test_type_forwards_captured_element_to_backend():
    backend = _backend("kate")
    calls = {}

    def _type_text(text, **kw):
        calls["text"] = text
        calls.update(kw)
        return _fake_action_result()

    backend.type_text = _type_text
    out = _dispatch_result(
        backend,
        "type",
        {"text": "session-bound", "element": 7, "app": "kate"},
    )

    assert calls["text"] == "session-bound"
    assert calls["element"] == 7
    assert out.get("ok") is True


def test_type_without_app_unchanged():
    """Legacy flows that never pass app= keep working (fail open)."""
    backend = _backend("kcalc")
    backend.type_text = lambda text, **kw: _fake_action_result()
    out = _dispatch_result(backend, "type", {"text": "9"})
    assert out.get("ok") is True


def test_input_actions_set_matches_dispatch_branches():
    # Guard list must cover exactly the sticky-target input actions.
    assert _INPUT_ACTIONS == {
        "click", "double_click", "right_click", "middle_click",
        "drag", "scroll", "type", "key", "set_value",
    }


# ── unknown-action suggestions ──────────────────────────────────────────


def test_unknown_action_hotkey_suggests_key():
    out = _dispatch_result(_backend(None), "hotkey", {})
    assert "did you mean 'key'" in out["error"]


def test_unknown_action_without_suggestion_unchanged():
    out = _dispatch_result(_backend(None), "frobnicate", {})
    assert out["error"] == "unknown action 'frobnicate'"
    assert "did you mean" not in out["error"]
