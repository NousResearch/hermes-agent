"""Regression tests for the run_steps blocked-combo gate covering hotkey steps.

A step that names ``hotkey`` and carries ``key`` + ``modifiers`` is translated
to a hotkey chord by the driver translator, so the pre-approval gate must
derive the same combo from it — otherwise a blocked combination (e.g. the
Empty Trash chord) passes the gate and still executes.
"""

from __future__ import annotations

from tools.computer_use.tool import _reject_unsafe_steps


def test_hotkey_alias_step_with_key_modifiers_hits_blocked_gate():
    steps = [{"tool": "hotkey", "args": {"key": "backspace", "modifiers": ["cmd", "shift"]}}]
    result = _reject_unsafe_steps(steps)
    assert result is not None
    assert "blocked key combo" in result


def test_short_form_hotkey_step_hits_blocked_gate():
    steps = [{"hotkey": {"key": "backspace", "modifiers": ["cmd", "shift"]}}]
    result = _reject_unsafe_steps(steps)
    assert result is not None
    assert "blocked key combo" in result


def test_hotkey_step_with_native_keys_list_hits_blocked_gate():
    steps = [{"tool": "hotkey", "args": {"keys": ["cmd", "shift", "backspace"]}}]
    result = _reject_unsafe_steps(steps)
    assert result is not None
    assert "blocked key combo" in result


def test_safe_hotkey_step_still_passes_the_gate():
    steps = [{"tool": "hotkey", "args": {"key": "c", "modifiers": ["cmd"]}}]
    assert _reject_unsafe_steps(steps) is None


def test_key_alias_step_gate_unchanged():
    steps = [{"tool": "key", "args": {"key": "backspace", "modifiers": ["cmd", "shift"]}}]
    result = _reject_unsafe_steps(steps)
    assert result is not None
    assert "blocked key combo" in result
