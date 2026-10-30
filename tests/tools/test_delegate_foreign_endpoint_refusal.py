"""A route with base_url and no credential must not borrow the parent's key.

Regression (live): ``auxiliary.review`` set ``provider: cliproxy`` + ``base_url`` and no key;
the /review child inherited the parent session's Anthropic OAuth token and posted it to a
local OpenAI-compatible proxy, which answered ``401 {"error":"Invalid API key"}``.
"""

from unittest.mock import MagicMock

import pytest

from tools.delegate_tool import _resolve_delegation_credentials

CHILD_URL = "http://127.0.0.1:8317/v1"
PARENT_URL = "https://api.anthropic.com"


def _parent(base_url=PARENT_URL, api_key="«redacted:sk-…»"):
    parent = MagicMock()
    parent.base_url = base_url
    parent.api_key = api_key
    parent.request_overrides = None
    parent._client_kwargs = {"base_url": base_url, "api_key": api_key}
    parent.client = None
    return parent


def _cfg(**overrides):
    cfg = {"model": "gemini-4-argon-high", "base_url": CHILD_URL}
    cfg.update(overrides)
    return cfg


def test_refuses_unkeyed_route_to_a_different_endpoint():
    with pytest.raises(ValueError) as excinfo:
        _resolve_delegation_credentials(_cfg(), _parent())
    message = str(excinfo.value)
    assert "refusing to send it to a different host" in message
    assert CHILD_URL in message
    assert PARENT_URL in message
    assert "key_env" in message


def test_inline_api_key_is_never_refused():
    creds = _resolve_delegation_credentials(_cfg(api_key="k-123456"), _parent())
    assert creds["api_key"] == "k-123456"


def test_pointer_key_satisfies_the_check(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.get_env_value_prefer_dotenv", lambda key: "sk-cliproxy")
    creds = _resolve_delegation_credentials(_cfg(key_env="CLIPROXY_API_KEY"), _parent())
    assert creds["api_key"] == "sk-cliproxy"


def test_same_endpoint_as_parent_still_inherits():
    """A direct endpoint that IS the parent's endpoint keeps inherit-from-parent (documented)."""
    creds = _resolve_delegation_credentials(_cfg(base_url=PARENT_URL + "/"), _parent())
    assert creds["api_key"] is None


def test_unknown_parent_endpoint_does_not_refuse():
    """No provable parent endpoint (tests, embedders) → keep the old behaviour."""
    creds = _resolve_delegation_credentials(_cfg(), parent_agent=None)
    assert creds["api_key"] is None


def test_routes_for_turns_the_refusal_into_a_tool_error():
    """delegate_tool._routes_for already converts ValueError into (no routes, message);
    assert the /review path (credentials_cfg) is covered by that conversion."""
    from tools.delegate_tool import _routes_for

    routes, error = _routes_for(
        [{"goal": "Review recent work"}],
        {},
        {"model": "gemini-4-argon-high", "provider": "", "base_url": CHILD_URL,
         "api_key": "", "api_mode": "", "key_env": ""},
        _parent(),
    )
    assert routes == []
    assert error and "refusing to send it to a different host" in error
