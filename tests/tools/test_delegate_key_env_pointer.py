"""Verify that delegation credential pointers resolve without changing endpoint behavior.

These tests are needed because ``key_env`` and ``api_key_env`` are first-class
credential pointers and must work alongside explicit API keys and inheritance.
"""

from types import SimpleNamespace

import hermes_cli.config as hermes_config
import pytest

from tools.delegate_tool_config import _resolve_delegation_credentials


def _clear_possible_credentials(monkeypatch):
    for name in ("OPENAI_API_KEY", "LITELLM_MASTER_KEY", "POINTER_KEY_ENV"):
        monkeypatch.delenv(name, raising=False)


def _parent_agent():
    return SimpleNamespace(request_overrides=None)


def test_key_env_pointer_resolves_delegation_api_key(monkeypatch):
    """A delegation key_env pointer supplies the direct endpoint API key."""
    _clear_possible_credentials(monkeypatch)
    fake_key = "test-key-not-a-secret"
    monkeypatch.setattr(hermes_config, "get_env_value_prefer_dotenv", lambda name: fake_key)

    credentials = _resolve_delegation_credentials(
        {
            "model": "m",
            "provider": "openai-api",
            "base_url": "https://litellm.internal/v1",
            "key_env": "POINTER_KEY_ENV",
        },
        _parent_agent(),
    )

    assert credentials["api_key"] == fake_key
    assert credentials["base_url"] == "https://litellm.internal/v1"


def test_api_key_env_pointer_is_an_alias_for_key_env(monkeypatch):
    """The api_key_env spelling resolves the same way as key_env."""
    _clear_possible_credentials(monkeypatch)
    fake_key = "test-key-not-a-secret"
    monkeypatch.setattr(hermes_config, "get_env_value_prefer_dotenv", lambda name: fake_key)

    credentials = _resolve_delegation_credentials(
        {
            "model": "m",
            "provider": "openai-api",
            "base_url": "https://litellm.internal/v1",
            "api_key_env": "POINTER_KEY_ENV",
        },
        _parent_agent(),
    )

    assert credentials["api_key"] == fake_key
    assert credentials["base_url"] == "https://litellm.internal/v1"


def test_explicit_api_key_takes_precedence_over_key_env(monkeypatch):
    """An explicit API key wins when both credential forms are present."""
    _clear_possible_credentials(monkeypatch)
    env_lookups = []

    def record_env_lookup(name):
        env_lookups.append(name)
        return "test-key-not-a-secret"

    monkeypatch.setattr(
        hermes_config,
        "get_env_value_prefer_dotenv",
        record_env_lookup,
    )

    credentials = _resolve_delegation_credentials(
        {
            "model": "m",
            "provider": "openai-api",
            "base_url": "https://litellm.internal/v1",
            "api_key": "test-explicit-key",
            "key_env": "POINTER_KEY_ENV",
        },
        _parent_agent(),
    )

    assert credentials["api_key"] == "test-explicit-key"
    assert credentials["base_url"] == "https://litellm.internal/v1"
    assert "POINTER_KEY_ENV" not in env_lookups
def test_missing_api_key_and_pointer_preserve_inheritance_semantics(monkeypatch):
    """Without either credential form, the child still inherits its key."""
    _clear_possible_credentials(monkeypatch)

    credentials = _resolve_delegation_credentials(
        {
            "base_url": "https://litellm.internal/v1",
        },
        _parent_agent(),
    )

    assert credentials["api_key"] is None
    assert credentials["base_url"] == "https://litellm.internal/v1"


def test_empty_key_env_pointer_leaves_api_key_unset(monkeypatch):
    """An unreadable or empty pointer behaves like a missing API key."""
    _clear_possible_credentials(monkeypatch)
    monkeypatch.setattr(hermes_config, "get_env_value_prefer_dotenv", lambda name: "")

    credentials = _resolve_delegation_credentials(
        {
            "model": "m",
            "provider": "openai-api",
            "base_url": "https://litellm.internal/v1",
            "key_env": "POINTER_KEY_ENV",
        },
        _parent_agent(),
    )

    assert credentials["api_key"] is None
    assert credentials["base_url"] == "https://litellm.internal/v1"
