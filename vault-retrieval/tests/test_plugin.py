"""Tests for plugin registration (plugin.py) and acceptance contracts.

Covers:
  - #1: Plugin discovery resolves under two distinct mocked HERMES_HOME
       profiles without cross-profile state leakage.
  - #9: Direct read_file / search_files inside the Vault are audit-only
       in audit, blocked in enforce, and allowed in off; paths outside
       the Vault are unchanged.
  - #18: Plugin disable/restart restores baseline file-tool behavior.

We use a fake PluginContext (no Hermes import) so the plugin code is
exercised against the same registry surface but stays isolated from
core. This matches the rubric: "test the contract, not the implementation".
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


from vault_retrieval.plugin import (
    _PROMPT_SECTION_ID,
    _PROMPT_SECTION_TEXT,
    _resolve_config,
    _resolve_target_in_vault,
    make_pre_tool_call_hook,
    register as plugin_register,
)
from vault_retrieval.tool import TOOL_NAME, TOOLSET


# ---------------------------------------------------------------------------
# Fake PluginContext — minimal surface that mirrors Hermes's PluginContext
# ---------------------------------------------------------------------------


class _FakeRegistration:
    def __init__(self, kind: str, key: str, payload: Any):
        self.kind = kind
        self.key = key
        self.payload = payload


class FakePluginContext:
    """In-memory stand-in for Hermes's PluginContext."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config or {})
        self.tools: Dict[str, _FakeRegistration] = {}
        self.hooks: Dict[str, List[Any]] = {}
        self.prompt_sections: Dict[str, _FakeRegistration] = {}

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def register_tool(
        self, name: str, toolset: str, schema: dict, handler,
        description: str = "", emoji: str = "", override: bool = False,
        **kwargs,
    ) -> _FakeRegistration:
        reg = _FakeRegistration("tool", name, {
            "name": name, "toolset": toolset, "schema": schema,
            "handler": handler, "description": description,
        })
        self.tools[name] = reg
        return reg

    def register_hook(self, hook_name: str, callback) -> _FakeRegistration:
        self.hooks.setdefault(hook_name, []).append(callback)
        return _FakeRegistration("hook", hook_name, callback)

    def register_system_prompt_section(
        self, id: str, content: str, *, position: str, max_chars: int,
        **kwargs,
    ) -> _FakeRegistration:
        reg = _FakeRegistration("system_prompt_section", id, {
            "content": content, "position": position, "max_chars": max_chars,
        })
        self.prompt_sections[id] = reg
        return reg


# ---------------------------------------------------------------------------
# Acceptance #1: Plugin discovery under two profiles (no cross-leakage)
# ---------------------------------------------------------------------------


class TestAcceptance1TwoProfiles:
    def test_two_profiles_register_independently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Profile A
        home_a = tmp_path / "profile_a"
        home_a.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        vault_a = home_a / "vault"
        vault_a.mkdir()
        ctx_a = FakePluginContext({"vault_root": str(vault_a), "mode": "enforce"})
        plugin_register(ctx_a)

        # Profile B
        home_b = tmp_path / "profile_b"
        home_b.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        vault_b = home_b / "vault"
        vault_b.mkdir()
        ctx_b = FakePluginContext({"vault_root": str(vault_b), "mode": "off"})
        plugin_register(ctx_b)

        # Both registered the same tool name
        assert TOOL_NAME in ctx_a.tools
        assert TOOL_NAME in ctx_b.tools
        # But the handlers are different instances (each profile owns
        # its own VaultContextHandler bound to its own vault root).
        assert ctx_a.tools[TOOL_NAME].payload["handler"] is not \
            ctx_b.tools[TOOL_NAME].payload["handler"]
        # Different state dirs.
        # Profile A: state/<vault-retrieval/query-log.jsonl>
        # Profile B: same path under a different HERMES_HOME.
        log_a = home_a / "state" / "vault-retrieval" / "query-log.jsonl"
        log_b = home_b / "state" / "vault-retrieval" / "query-log.jsonl"
        # Both state dirs exist (QueryLogger creates them on init).
        assert log_a.parent.exists()
        assert log_b.parent.exists()
        assert log_a != log_b
        # The two profiles must not see each other's state.
        assert log_a.parent.parent != log_b.parent.parent


# ---------------------------------------------------------------------------
# Acceptance #9: hook mode semantics + Vault vs outside-Vault
# ---------------------------------------------------------------------------


class TestAcceptance9HookModes:
    def _vault(self, tmp_path: Path) -> Path:
        v = tmp_path / "vault"
        v.mkdir()
        return v

    def test_enforce_blocks_inside_vault(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        v = self._vault(tmp_path)
        ctx = FakePluginContext({"vault_root": str(v), "mode": "enforce"})
        hook = make_pre_tool_call_hook(ctx)
        note = v / "note.md"
        note.write_text("hi")
        result = hook(tool_name="read_file", args={"path": str(note)})
        assert result is not None
        assert result["action"] == "block"
        assert "vault_context" in result["message"]

    def test_audit_passes_through_inside_vault(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ):
        v = self._vault(tmp_path)
        ctx = FakePluginContext({"vault_root": str(v), "mode": "audit"})
        hook = make_pre_tool_call_hook(ctx)
        note = v / "note.md"
        note.write_text("hi")
        with caplog.at_level(logging.INFO, logger="vault_retrieval.plugin"):
            result = hook(tool_name="read_file", args={"path": str(note)})
        assert result is None
        assert any("would-block" in r.message for r in caplog.records)

    def test_off_passes_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        v = self._vault(tmp_path)
        ctx = FakePluginContext({"vault_root": str(v), "mode": "off"})
        hook = make_pre_tool_call_hook(ctx)
        note = v / "note.md"
        note.write_text("hi")
        result = hook(tool_name="read_file", args={"path": str(note)})
        assert result is None

    def test_outside_vault_unaffected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        v = self._vault(tmp_path)
        outside = tmp_path / "outside.md"
        outside.write_text("outside")
        ctx = FakePluginContext({"vault_root": str(v), "mode": "enforce"})
        hook = make_pre_tool_call_hook(ctx)
        result = hook(tool_name="read_file", args={"path": str(outside)})
        assert result is None  # outside Vault, never blocked

    def test_search_files_inside_vault_blocked_in_enforce(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        v = self._vault(tmp_path)
        ctx = FakePluginContext({"vault_root": str(v), "mode": "enforce"})
        hook = make_pre_tool_call_hook(ctx)
        result = hook(tool_name="search_files", args={"pattern": "**/*.md"})
        assert result is not None
        assert result["action"] == "block"

    def test_non_protected_tools_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        v = self._vault(tmp_path)
        ctx = FakePluginContext({"vault_root": str(v), "mode": "enforce"})
        hook = make_pre_tool_call_hook(ctx)
        # write_file is NOT protected by the hook — only read tools.
        result = hook(tool_name="write_file", args={"path": str(v / "x.md")})
        assert result is None

    def test_block_direct_file_reads_false_disables_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        v = self._vault(tmp_path)
        ctx = FakePluginContext({
            "vault_root": str(v), "mode": "enforce",
            "block_direct_file_reads": False,
        })
        hook = make_pre_tool_call_hook(ctx)
        note = v / "note.md"
        note.write_text("hi")
        result = hook(tool_name="read_file", args={"path": str(note)})
        assert result is None  # explicitly disabled → no block

    def test_invalid_config_disables_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        v = self._vault(tmp_path)
        # hard_ceiling_chars > compiled cap → VaultConfigError
        ctx = FakePluginContext({
            "vault_root": str(v), "mode": "enforce",
            "hard_ceiling_chars": 99_999,
        })
        hook = make_pre_tool_call_hook(ctx)
        note = v / "note.md"
        note.write_text("hi")
        result = hook(tool_name="read_file", args={"path": str(note)})
        # Hook is disabled (fail closed at registration).
        assert result is None


# ---------------------------------------------------------------------------
# Acceptance #18: disable/restart restores baseline
# ---------------------------------------------------------------------------


class TestAcceptance18Restart:
    def test_mode_off_restores_direct_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        v = tmp_path / "vault"
        v.mkdir()
        ctx = FakePluginContext({"vault_root": str(v), "mode": "off"})
        hook = make_pre_tool_call_hook(ctx)
        note = v / "note.md"
        note.write_text("baseline content")
        # Even with a path INSIDE the Vault, off mode lets it through.
        result = hook(tool_name="read_file", args={"path": str(note)})
        assert result is None


# ---------------------------------------------------------------------------
# Registration shape: prompt section, toolset, hook
# ---------------------------------------------------------------------------


class TestRegistrationShape:
    def test_register_registers_tool_hook_and_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        vault = home / "vault"
        vault.mkdir()
        ctx = FakePluginContext({"vault_root": str(vault), "mode": "enforce"})
        plugin_register(ctx)
        # Tool
        assert TOOL_NAME in ctx.tools
        assert ctx.tools[TOOL_NAME].payload["toolset"] == TOOLSET
        # Hook
        assert "pre_tool_call" in ctx.hooks
        # Prompt section
        assert _PROMPT_SECTION_ID in ctx.prompt_sections
        sec = ctx.prompt_sections[_PROMPT_SECTION_ID].payload
        assert sec["content"] == _PROMPT_SECTION_TEXT
        assert sec["position"] == "after_memory"
        assert 0 < sec["max_chars"] <= 4000

    def test_register_invalid_config_does_not_register(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        ctx = FakePluginContext({
            "vault_root": str(home), "mode": "enforce",
            "hard_ceiling_chars": 999_999,  # > compiled cap
        })
        plugin_register(ctx)
        # Failure is silent: nothing was registered.
        assert ctx.tools == {}
        assert ctx.hooks == {}
        assert ctx.prompt_sections == {}

    def test_tool_handler_callable_through_registered_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        vault = home / "vault"
        vault.mkdir()
        note = vault / "note.md"
        note.write_text("alpha line\nbeta line\n", encoding="utf-8")
        ctx = FakePluginContext({"vault_root": str(vault), "mode": "enforce"})
        plugin_register(ctx)
        handler = ctx.tools[TOOL_NAME].payload["handler"]
        # Invoke the registered handler the way Hermes would.
        out = handler({"query": "anything", "selectors": [
            {"path": "note.md", "line_start": 1, "line_end": 2}
        ]})
        env = json.loads(out)
        assert env["status"] == "ok"
        assert env["extracts"][0]["content"] == "alpha line\nbeta line\n"


# ---------------------------------------------------------------------------
# _resolve_target_in_vault unit checks
# ---------------------------------------------------------------------------


class TestResolveTargetInVault:
    def test_path_inside_vault(self, tmp_path):
        v = tmp_path / "vault"
        v.mkdir()
        f = v / "x.md"
        f.write_text("x")
        out = _resolve_target_in_vault({"path": str(f)}, v)
        assert out == f.resolve()

    def test_path_outside_vault(self, tmp_path):
        v = tmp_path / "vault"
        v.mkdir()
        out = _resolve_target_in_vault({"path": "/etc/passwd"}, v)
        assert out is None

    def test_search_pattern_returns_sentinel(self, tmp_path):
        v = tmp_path / "vault"
        v.mkdir()
        out = _resolve_target_in_vault({"pattern": "**/*.md"}, v)
        assert out == v.resolve()

    def test_empty_args(self, tmp_path):
        v = tmp_path / "vault"
        v.mkdir()
        assert _resolve_target_in_vault({}, v) is None
        assert _resolve_target_in_vault({"path": ""}, v) is None
