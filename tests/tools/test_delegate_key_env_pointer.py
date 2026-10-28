"""A delegation/auxiliary credential block may name its key with key_env/api_key_env.

Regression: a block that declared ``key_env`` but no inline ``api_key`` resolved to
``api_key=None``, which means "inherit the parent's key" downstream
(``_build_child_agent``: ``override_api_key or parent_api_key``). A child routed to a
different endpoint then sent the *parent session's* credential to that host and got
``401 {"error":"Invalid API key"}``.
"""

from unittest.mock import MagicMock

import pytest

from tools.delegate_tool import _resolve_delegation_credentials


def _parent(base_url="https://api.anthropic.com", api_key="«redacted:sk-…»"):
    parent = MagicMock()
    parent.base_url = base_url
    parent.api_key = api_key
    parent.request_overrides = None
    parent._client_kwargs = {"base_url": base_url, "api_key": api_key}
    parent.client = None
    return parent


def _cfg(**overrides):
    # provider="" keeps the direct-endpoint branch off the runtime-provider resolver,
    # so these tests never touch the network or a real credential store.
    cfg = {"model": "gemini-4-argon-high", "base_url": "http://127.0.0.1:8317/v1"}
    cfg.update(overrides)
    return cfg


@pytest.fixture
def pointer(monkeypatch):
    """Point ``key_env`` at a fixed value; returns the setter."""
    def _set(value):
        monkeypatch.setattr("hermes_cli.config.get_env_value_prefer_dotenv", lambda key: value)
    return _set


def test_key_env_resolves_when_api_key_absent(pointer):
    pointer("«redacted:sk-…»")
    creds = _resolve_delegation_credentials(_cfg(key_env="CLIPROXY_API_KEY"), _parent())
    assert creds["api_key"] == "«redacted:sk-…»"


def test_api_key_env_alias_resolves(pointer):
    pointer("sk-alias")
    creds = _resolve_delegation_credentials(_cfg(api_key_env="CLIPROXY_API_KEY"), _parent())
    assert creds["api_key"] == "sk-alias"


def test_inline_api_key_wins_and_pointer_is_not_read(monkeypatch):
    def _boom(key):
        raise AssertionError("pointer must not be read when an inline api_key is set")

    monkeypatch.setattr("hermes_cli.config.get_env_value_prefer_dotenv", _boom)
    creds = _resolve_delegation_credentials(
        _cfg(key_env="CLIPROXY_API_KEY", api_key="inline-key"), _parent())
    assert creds["api_key"] == "inline-key"


def test_pointer_that_resolves_empty_leaves_api_key_none(pointer):
    pointer("")
    creds = _resolve_delegation_credentials(_cfg(key_env="UNSET_VAR"), _parent(base_url="http://127.0.0.1:8317/v1"))
    assert creds["api_key"] is None


def test_pointer_does_not_change_base_url(pointer):
    pointer("k")
    creds = _resolve_delegation_credentials(_cfg(key_env="SOME_VAR"), _parent())
    assert creds["base_url"] == "http://127.0.0.1:8317/v1"
