"""Tests for tools/secrets_tool.py — secret lifecycle management."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hermes_constants import get_hermes_home
from tools.secrets_tool import secrets_handler, set_secret_capture_callback


def test_list_returns_only_key_names_not_values(monkeypatch):
    """Call list action, verify no values in output, only key names."""
    # Mock get_env_value to return a real value, but list should only show keys
    with patch("tools.secrets_tool.get_env_value") as mock_get:
        mock_get.side_effect = lambda k: "secret_value" if k == "OPENAI_API_KEY" else ""
        with patch(
            "tools.secrets_tool._discover_secret_keys",
            return_value=["OPENAI_API_KEY", "OTHER_KEY"],
        ):
            resp_json = secrets_handler(action="list")
            resp = json.loads(resp_json)

            assert "OPENAI_API_KEY" in resp["configured"]
            assert "secret_value" not in resp_json
            assert "OTHER_KEY" not in resp["configured"]


def test_list_shows_missing_for_skills(monkeypatch):
    """Create a skill with requires_secrets frontmatter, verify it shows up as missing."""
    hermes_home = get_hermes_home()
    skills_dir = hermes_home / "skills"
    skill_dir = skills_dir / "test_skill"
    skill_dir.mkdir(parents=True, exist_ok=True)

    (skill_dir / "SKILL.md").write_text("""---
name: test_skill
requires_secrets:
  - NEEDED_KEY
---
# Test Skill
""")

    with patch("tools.secrets_tool.get_env_value", return_value=""):
        resp = json.loads(secrets_handler(action="list"))

        assert "NEEDED_KEY" in resp["missing_for_skills"]
        assert any(
            s["skill"] == "test_skill" and "NEEDED_KEY" in s["missing"]
            for s in resp["skills_missing_secrets"]
        )


def test_check_configured_and_missing():
    """Set one env var, check two keys, verify one configured one missing."""

    def mock_get(key):
        return "val" if key == "CONFIGURED" else ""

    with patch("tools.secrets_tool.get_env_value", side_effect=mock_get):
        resp = json.loads(
            secrets_handler(action="check", keys=["CONFIGURED", "MISSING"])
        )

        assert "CONFIGURED" in resp["configured"]
        assert "MISSING" in resp["missing"]


def test_request_with_callback():
    """Set _secret_capture_callback, call request, verify callback was called with correct args."""
    callback = MagicMock(return_value={"stored": True})
    set_secret_capture_callback(callback)

    try:
        resp = json.loads(
            secrets_handler(
                action="request",
                key="NEW_KEY",
                description="My Key",
                instructions="Go to example.com",
            )
        )

        callback.assert_called_once_with(
            "NEW_KEY",
            "My Key (Go to example.com)",
            {"description": "My Key", "instructions": "Go to example.com"},
        )
        assert resp["stored"] is True
    finally:
        set_secret_capture_callback(None)


def test_request_cli_fallback(monkeypatch):
    """Without callback, test that request returns gateway_secret_prompt for gateway mode."""
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "true")

    resp = json.loads(secrets_handler(action="request", key="GATEWAY_KEY"))

    assert resp["stored"] is False
    assert "gateway_secret_prompt" in resp
    assert resp["gateway_secret_prompt"]["key"] == "GATEWAY_KEY"


def test_delete_clears_value():
    """Set a value, call delete, verify save_env_value called with empty string."""
    with patch("tools.secrets_tool.save_env_value") as mock_save:
        resp = json.loads(secrets_handler(action="delete", key="TO_DELETE"))

        assert resp["deleted"] is True
        mock_save.assert_called_once_with("TO_DELETE", "")


def test_inject_registers_passthrough():
    """Call inject with keys, verify env_passthrough has them registered."""
    with patch("tools.secrets_tool.register_env_passthrough") as mock_reg:
        resp = json.loads(secrets_handler(action="inject", keys=["KEY1", "KEY2"]))

        assert resp["injected"] == ["KEY1", "KEY2"]
        mock_reg.assert_called_once_with(["KEY1", "KEY2"])


def test_invalid_action():
    """Call with unknown action, verify error message."""
    resp = json.loads(secrets_handler(action="invalid"))
    assert "error" in resp
    assert "Invalid action" in resp["error"]


def test_request_without_key():
    """Call request without key, verify error message."""
    resp = json.loads(secrets_handler(action="request"))
    assert "error" in resp
    assert "key is required" in resp["error"]


def test_normalize_keys():
    """Test that duplicate keys are deduplicated."""
    # We test this via 'check' action which uses _normalize_keys
    with patch("tools.secrets_tool.get_env_value", return_value=""):
        resp = json.loads(
            secrets_handler(action="check", keys=["DUP", "DUP", "UNIQUE"], key="DUP")
        )
        # _normalize_keys(keys=["DUP", "DUP", "UNIQUE"], key="DUP") -> ["DUP", "UNIQUE"]
        assert len(resp["missing"]) == 2
        assert "DUP" in resp["missing"]
        assert "UNIQUE" in resp["missing"]
