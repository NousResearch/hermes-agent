"""Regression tests for macOS modifier-state flush after key injection (#93702).

Review #93702 raised three points on the flush mechanism:
1. The `~/bin/release-stuck-keys` helper is an arbitrary executable looked up
   in the user's home directory — provenance-gate it (only trusted when the
   macos-input-troubleshooting skill's source file is present) and resolve it
   once instead of per-action stat/spawn.
2. A synthetic F12 fallback can reach apps that bind F12 — the trade-off is
   documented; an in-process CGEvent re-post is not possible without Quartz
   bindings the backend does not ship.
3. `press_key` only flushed when `key is None` — a `press_key` carrying both a
   key and modifiers is also a chord and must flush symmetrically with hotkey.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from tools.computer_use import cua_backend as cb


@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(cb.sys, "platform", "darwin")


# ---------------------------------------------------------------------------
# _flush_modifier_after — pure decision table
# ---------------------------------------------------------------------------


class TestFlushModifierAfter:
    def test_non_darwin_never_flushes(self, monkeypatch):
        monkeypatch.setattr(cb.sys, "platform", "linux")
        assert cb._flush_modifier_after("hotkey", {"keys": ["cmd", "g"]}) is False
        assert cb._flush_modifier_after("click", {"modifiers": ["cmd"]}) is False

    def test_hotkey_always_flushes(self, darwin):
        assert cb._flush_modifier_after("hotkey", {"keys": ["cmd", "g"]}) is True
        assert cb._flush_modifier_after("hotkey", {"keys": ["ctrl"]}) is True
        # An empty keys list carries no modifiers — nothing to flush.
        assert cb._flush_modifier_after("hotkey", {"keys": []}) is False

    def test_press_key_with_modifiers_flushes_even_with_main_key(self, darwin):
        # Regression #93702-3: previously only `key is None` flushed, so
        # press_key(key='x', modifiers=['cmd']) skipped the flush while
        # carrying the same stuck-modifier risk as hotkey.
        assert cb._flush_modifier_after(
            "press_key", {"key": "x", "modifiers": ["cmd"]}
        ) is True
        assert cb._flush_modifier_after(
            "press_key", {"key": None, "modifiers": ["cmd"]}
        ) is True

    def test_press_key_without_modifiers_does_not_flush(self, darwin):
        assert cb._flush_modifier_after("press_key", {"key": "f12"}) is False
        assert cb._flush_modifier_after("press_key", {"key": "x"}) is False

    def test_click_drag_scroll_with_modifiers_flush(self, darwin):
        assert cb._flush_modifier_after("click", {"modifiers": ["cmd"]}) is True
        assert cb._flush_modifier_after("drag", {"modifier": "cmd"}) is True
        assert cb._flush_modifier_after("scroll", {"modifiers": ["ctrl"]}) is True

    def test_plain_click_does_not_flush(self, darwin):
        assert cb._flush_modifier_after("click", {}) is False
        assert cb._flush_modifier_after("type_text", {"text": "hi"}) is False


# ---------------------------------------------------------------------------
# _resolve_stuck_key_helper — provenance gate + cache
# ---------------------------------------------------------------------------


class TestResolveStuckKeyHelper:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        cb._resolve_stuck_key_helper.cache_clear()
        yield
        cb._resolve_stuck_key_helper.cache_clear()

    def test_missing_helper_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cb.os.path, "expanduser", lambda p: str(tmp_path / "release-stuck-keys"))
        assert cb._resolve_stuck_key_helper() is None

    def test_helper_without_skill_source_is_rejected(self, monkeypatch, tmp_path):
        # Regression #93702-1: an arbitrary executable dropped into ~/bin
        # must NOT be executed — only trusted when the skill's source file
        # (the provenance anchor) is still present.
        helper = tmp_path / "release-stuck-keys"
        helper.write_text("#!/bin/sh\necho hi\n")
        helper.chmod(0o755)

        def fake_expanduser(p):
            if "release-stuck-keys" in p and "skills" not in p:
                return str(helper)
            return str(tmp_path / "no-skill.swift")

        monkeypatch.setattr(cb.os.path, "expanduser", fake_expanduser)
        assert cb._resolve_stuck_key_helper() is None

    def test_helper_with_skill_source_is_trusted(self, monkeypatch, tmp_path):
        helper = tmp_path / "release-stuck-keys"
        helper.write_text("#!/bin/sh\necho hi\n")
        helper.chmod(0o755)
        skill_src = tmp_path / "skills" / "release-stuck-keys.swift"
        skill_src.parent.mkdir(parents=True)
        skill_src.write_text("// swift source\n")

        def fake_expanduser(p):
            if "skills" in p:
                return str(skill_src)
            return str(helper)

        monkeypatch.setattr(cb.os.path, "expanduser", fake_expanduser)
        assert cb._resolve_stuck_key_helper() == str(helper)


# ---------------------------------------------------------------------------
# _flush_stuck_modifiers — helper preferred, F12 fallback, silent failures
# ---------------------------------------------------------------------------


class TestFlushStuckModifiers:
    def test_helper_used_when_resolved(self, darwin, monkeypatch):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return None

        monkeypatch.setattr(cb, "_resolve_stuck_key_helper", lambda: "/trusted/release-stuck-keys")
        monkeypatch.setattr(cb.subprocess, "run", fake_run)
        backend = object.__new__(cb.CuaDriverBackend)
        backend._active_pid = 42
        backend._active_window_id = 7
        cb.CuaDriverBackend._flush_stuck_modifiers(backend)
        assert calls == [["/trusted/release-stuck-keys", "--fix"]]

    def test_f12_fallback_when_no_helper(self, darwin, monkeypatch):
        monkeypatch.setattr(cb, "_resolve_stuck_key_helper", lambda: None)
        backend = object.__new__(cb.CuaDriverBackend)
        backend._active_pid = 42
        backend._active_window_id = 7
        action_calls = []

        def fake_action(name, args):
            action_calls.append((name, args))
            return None

        backend._action = fake_action
        cb.CuaDriverBackend._flush_stuck_modifiers(backend)
        assert action_calls == [("press_key", {"pid": 42, "key": "f12", "window_id": 7})]

    def test_f12_fallback_skipped_without_active_pid(self, darwin, monkeypatch):
        monkeypatch.setattr(cb, "_resolve_stuck_key_helper", lambda: None)
        backend = object.__new__(cb.CuaDriverBackend)
        backend._active_pid = None
        backend._active_window_id = None
        backend._action = lambda name, args: (_ for _ in ()).throw(AssertionError("must not inject"))
        cb.CuaDriverBackend._flush_stuck_modifiers(backend)  # silent no-op

    def test_helper_exception_falls_back_to_f12(self, darwin, monkeypatch):
        def boom(cmd, **kw):
            raise OSError("helper vanished")

        monkeypatch.setattr(cb, "_resolve_stuck_key_helper", lambda: "/trusted/release-stuck-keys")
        monkeypatch.setattr(cb.subprocess, "run", boom)
        backend = object.__new__(cb.CuaDriverBackend)
        backend._active_pid = 42
        backend._active_window_id = None
        action_calls = []

        def fake_action(name, args):
            action_calls.append((name, args))
            return None

        backend._action = fake_action
        cb.CuaDriverBackend._flush_stuck_modifiers(backend)
        assert action_calls == [("press_key", {"pid": 42, "key": "f12"})]

    def test_non_darwin_is_noop(self, monkeypatch):
        monkeypatch.setattr(cb.sys, "platform", "linux")
        backend = object.__new__(cb.CuaDriverBackend)
        backend._action = lambda name, args: (_ for _ in ()).throw(AssertionError("must not inject"))
        cb.CuaDriverBackend._flush_stuck_modifiers(backend)
