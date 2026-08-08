#!/usr/bin/env python3
"""Tests for the opt-in persistent sub-agent soul injection.

Covers the ``delegation.persistent_agent`` feature: when enabled, a soul
contract read from disk (config path > HERMES_SUB_ENGINE_SOUL_PATH env >
<hermes_home>/profiles/sub-engine/SUB_ENGINE_SOUL.md) is prepended to every
delegated child's system prompt. When disabled (the default) behavior is
identical to upstream. The loader degrades gracefully when the file is absent,
empty, or unreadable.

Run with:  python -m pytest tests/tools/test_delegate_sub_engine_soul.py -v
"""

import logging
import os

import pytest

from hermes_constants import get_hermes_home
from tools.delegate_tool import (
    _build_child_system_prompt,
    _load_sub_engine_soul,
    _persistent_agent_config,
    _resolve_sub_engine_soul_path,
    _SUB_ENGINE_SOUL_RELPATH,
)

CONTRACT = (
    "# SUB_ENGINE_SOUL.md — mechanical sub-engine contract\n"
    "Every response is a <tool_execution>, <skill_execution_report>, or "
    "<mechanical_block> and nothing else."
)


@pytest.fixture
def soul_file(tmp_path, monkeypatch):
    """Write a contract file and point the env override at it."""
    path = tmp_path / "profiles" / "sub-engine" / "SUB_ENGINE_SOUL.md"
    path.parent.mkdir(parents=True)
    path.write_text(CONTRACT, encoding="utf-8")
    monkeypatch.setenv("HERMES_SUB_ENGINE_SOUL_PATH", str(path))
    return path


# ── path resolution ─────────────────────────────────────────────────────────

def test_resolve_default_path_uses_hermes_home(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_SUB_ENGINE_SOUL_PATH", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    resolved = _resolve_sub_engine_soul_path()
    assert "~" not in resolved
    assert resolved == str(get_hermes_home().joinpath(*_SUB_ENGINE_SOUL_RELPATH))
    assert resolved.endswith("profiles/sub-engine/SUB_ENGINE_SOUL.md")


def test_resolve_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom.md"
    monkeypatch.setenv("HERMES_SUB_ENGINE_SOUL_PATH", str(target))
    assert _resolve_sub_engine_soul_path() == str(target)


def test_resolve_explicit_config_path_wins_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SUB_ENGINE_SOUL_PATH", str(tmp_path / "env.md"))
    explicit = tmp_path / "from-config.md"
    assert _resolve_sub_engine_soul_path(str(explicit)) == str(explicit)


# ── loader ──────────────────────────────────────────────────────────────────

def test_load_returns_stripped_contents(soul_file):
    assert _load_sub_engine_soul() == CONTRACT
    soul_file.write_text(CONTRACT + "\n\n", encoding="utf-8")
    assert _load_sub_engine_soul() == CONTRACT


def test_load_missing_file_returns_none_and_warns(tmp_path, monkeypatch, caplog):
    missing = tmp_path / "nope" / "SUB_ENGINE_SOUL.md"
    monkeypatch.setenv("HERMES_SUB_ENGINE_SOUL_PATH", str(missing))
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        assert _load_sub_engine_soul() is None
    assert any("not found" in r.message for r in caplog.records)


def test_load_empty_file_returns_none_and_warns(soul_file, caplog):
    soul_file.write_text("   \n\t\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        assert _load_sub_engine_soul() is None
    assert any("empty" in r.message for r in caplog.records)


def test_load_unreadable_path_returns_none_and_warns(tmp_path, monkeypatch, caplog):
    as_dir = tmp_path / "SUB_ENGINE_SOUL.md"
    as_dir.mkdir()  # a directory at the path → OSError on open()
    monkeypatch.setenv("HERMES_SUB_ENGINE_SOUL_PATH", str(as_dir))
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        assert _load_sub_engine_soul() is None
    assert any(
        ("could not read" in r.message.lower()) or ("unexpected error" in r.message.lower())
        for r in caplog.records
    )


# ── the feature gate in _build_child_system_prompt ──────────────────────────

def test_disabled_by_default_no_injection_even_when_file_present(soul_file):
    # Default persistent_agent=False → upstream behavior, contract NOT injected.
    prompt = _build_child_system_prompt("Resolve a d20 attack", context="AC 15")
    assert CONTRACT not in prompt
    assert prompt.startswith("You are a focused subagent")
    assert "YOUR TASK:\nResolve a d20 attack" in prompt


def test_enabled_prepends_contract(soul_file):
    prompt = _build_child_system_prompt(
        "Resolve a d20 attack", context="AC 15", persistent_agent=True
    )
    assert prompt.startswith(CONTRACT)
    assert "You are a focused subagent" in prompt
    assert "YOUR TASK:\nResolve a d20 attack" in prompt
    assert "CONTEXT:\nAC 15" in prompt
    assert prompt.index(CONTRACT) < prompt.index("YOUR TASK:")


def test_enabled_but_missing_file_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SUB_ENGINE_SOUL_PATH", str(tmp_path / "absent.md"))
    prompt = _build_child_system_prompt("Do a thing", persistent_agent=True)
    assert CONTRACT not in prompt
    assert prompt.startswith("You are a focused subagent")
    assert "YOUR TASK:\nDo a thing" in prompt


def test_enabled_with_explicit_path_arg(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_SUB_ENGINE_SOUL_PATH", raising=False)
    p = tmp_path / "one-off.md"
    p.write_text(CONTRACT, encoding="utf-8")
    prompt = _build_child_system_prompt(
        "Task", persistent_agent=True, sub_engine_soul_path=str(p)
    )
    assert prompt.startswith(CONTRACT)


def test_enabled_orchestrator_role(soul_file):
    prompt = _build_child_system_prompt(
        "Coordinate research",
        role="orchestrator",
        child_depth=1,
        max_spawn_depth=2,
        persistent_agent=True,
    )
    assert prompt.startswith(CONTRACT)
    assert "Subagent Spawning (Orchestrator Role)" in prompt


# ── config read (_persistent_agent_config) ──────────────────────────────────

def test_config_off_by_default(monkeypatch):
    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {})
    assert _persistent_agent_config() == (False, None)


def test_config_enabled_with_path(monkeypatch):
    monkeypatch.setattr(
        "tools.delegate_tool._load_config",
        lambda: {"persistent_agent": True, "persistent_agent_soul_path": "/x/soul.md"},
    )
    assert _persistent_agent_config() == (True, "/x/soul.md")


def test_config_truthy_string_enables(monkeypatch):
    monkeypatch.setattr(
        "tools.delegate_tool._load_config", lambda: {"persistent_agent": "true"}
    )
    enabled, path = _persistent_agent_config()
    assert enabled is True and path is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
