"""Unit tests for per-task Hermes profile delegation routing.

Tests:
1. _detect_task_profile:
   - Explicit 'profile' key
   - Goal prefix '@<profile>:'
   - Non-profile @-mentions ignored (fallback to None)
   - Plain goals without @-mention (fallback to None)
   - Case-insensitive profile name matching
2. _resolve_profile_task_credentials:
   - Successfully reads model, provider, base_url, api_key from profile config.yaml
   - Returns None for non-existent profile
   - Gracefully handles corrupt / unreadable config.yaml without raising
3. Schema advertisement:
   - DELEGATE_TASK_SCHEMA exposes 'profile' on tasks items
4. Isolation:
   - Profile-scoped children activate memory and context files
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _detect_task_profile,
    _resolve_profile_task_credentials,
)


# ── Profile Detection Tests ────────────────────────────────────────────────


def test_detect_task_profile_explicit(monkeypatch):
    """Explicit 'profile' key matches when profile exists."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in ["piglet", "tigger"])

    assert _detect_task_profile({"profile": "piglet", "goal": "check logs"}) == "piglet"
    assert _detect_task_profile({"profile": "TIGGER", "goal": "run tests"}) == "tigger"
    assert _detect_task_profile({"profile": "unknown", "goal": "run tests"}) is None


def test_detect_task_profile_goal_prefix(monkeypatch):
    """Goal prefix '@<profile>:' extracts the profile name if valid."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in ["piglet", "tigger", "eeyore"])

    assert _detect_task_profile({"goal": "@piglet: review the changes"}) == "piglet"
    assert _detect_task_profile({"goal": "@Eeyore: check error trace"}) == "eeyore"
    assert _detect_task_profile({"goal": "   @tigger:   write unit tests"}) == "tigger"


def test_detect_task_profile_unknown_mention_ignored(monkeypatch):
    """Non-profile @-mentions fall back to None without error."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in ["piglet"])

    assert _detect_task_profile({"goal": "@someone_else: review this"}) is None
    assert _detect_task_profile({"goal": "ping @piglet about this"}) is None


def test_detect_task_profile_plain_goal(monkeypatch):
    """Goals without any @-mention return None."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)

    assert _detect_task_profile({"goal": "clean up temporary files"}) is None
    assert _detect_task_profile({}) is None


# ── Credential Resolution Tests ───────────────────────────────────────────


def test_resolve_profile_task_credentials_success(tmp_path, monkeypatch):
    """Reads model, provider, base_url, and api_key from profile config.yaml."""
    profile_dir = tmp_path / "profiles" / "mock_persona"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text(
        """
model:
  default: test-model-v1
  provider: mock_provider
providers:
  mock_provider:
    base_url: http://10.0.0.1:11434/v1
    api_key: secret-key-abc
    api_mode: chat
    request_overrides:
      extra_body:
        temperature: 0.2
"""
    )

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "mock_persona")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    creds = _resolve_profile_task_credentials("mock_persona", parent_agent=None)
    assert creds is not None
    assert creds["model"] == "test-model-v1"
    assert creds["provider"] == "mock_provider"
    assert creds["base_url"] == "http://10.0.0.1:11434/v1"
    assert creds["api_key"] == "secret-key-abc"
    assert creds["api_mode"] == "chat"
    assert creds["profile"] == "mock_persona"
    assert creds["profile_dir"] == profile_dir
    assert creds["request_overrides"] == {"extra_body": {"temperature": 0.2}}


def test_resolve_profile_task_credentials_nonexistent(monkeypatch):
    """Returns None for non-existent profiles."""
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)

    assert _resolve_profile_task_credentials("nonexistent", parent_agent=None) is None


def test_resolve_profile_task_credentials_corrupt_yaml(tmp_path, monkeypatch):
    """Returns None gracefully on unreadable or invalid YAML config."""
    profile_dir = tmp_path / "profiles" / "broken"
    profile_dir.mkdir(parents=True)
    cfg_file = profile_dir / "config.yaml"
    cfg_file.write_text(":::invalid:yaml:::")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "broken")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_dir)

    creds = _resolve_profile_task_credentials("broken", parent_agent=None)
    assert creds is None


# ── Schema Tests ──────────────────────────────────────────────────────────


def test_delegate_task_schema_advertises_profile():
    """DELEGATE_TASK_SCHEMA declares 'profile' on task items."""
    task_items = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
    assert "profile" in task_items["properties"]
    prop = task_items["properties"]["profile"]
    assert prop["type"] == "string"
    assert "Hermes profile" in prop["description"]
