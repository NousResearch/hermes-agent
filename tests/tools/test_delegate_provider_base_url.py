"""delegation.provider + delegation.base_url: the provider keeps its credentials.

A configured provider owns its key and transport; ``delegation.base_url`` only
moves the endpoint. Before the fix the base_url branch won outright, so a named
custom provider set alongside base_url came back as ``provider="custom"`` with
``api_key=None`` (the child then inherited the PARENT's key) and a URL-guessed
api_mode. Real config + env resolution against the temp HERMES_HOME.
"""

from pathlib import Path

import pytest
import hermes_yaml as yaml

from tools.delegate_tool import _resolve_delegation_credentials

OVERRIDE_URL = "http://127.0.0.1:8317/v1"


@pytest.fixture
def cliproxy_provider(monkeypatch):
    import os

    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "providers": {
            "cliproxy": {
                "base_url": "http://10.0.0.5:8317/v1",
                "key_env": "CLIPROXY_API_KEY",
                "api_mode": "anthropic_messages",
            },
        },
    }))
    monkeypatch.setenv("CLIPROXY_API_KEY", "cliproxy-secret")


def test_provider_with_base_url_keeps_provider_credentials(cliproxy_provider):
    creds = _resolve_delegation_credentials(
        {"model": "claude-sonnet-4-6", "provider": "cliproxy", "base_url": OVERRIDE_URL},
        parent_agent=None,
    )
    assert creds["base_url"] == OVERRIDE_URL
    assert creds["api_key"] == "cliproxy-secret"
    assert creds["api_mode"] == "anthropic_messages"
    assert creds["provider"] == "cliproxy"


def test_explicit_api_key_and_api_mode_still_win(cliproxy_provider):
    creds = _resolve_delegation_credentials(
        {"model": "m", "provider": "cliproxy", "base_url": OVERRIDE_URL,
         "api_key": "explicit-key", "api_mode": "chat_completions"},
        parent_agent=None,
    )
    assert creds["base_url"] == OVERRIDE_URL
    assert creds["api_key"] == "explicit-key"
    assert creds["api_mode"] == "chat_completions"
