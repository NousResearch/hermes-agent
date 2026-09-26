"""Third-party login sources use the existing schema and masked unlock path."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

from jsonschema import validate
import pytest

from tools import browser_vault_tool as tool
from tools.registry import registry


def test_unlock_accepts_plugin_name_and_rejects_ineligible_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = {"backend": "custom"}
    validate(args, tool.BROWSER_VAULT_UNLOCK_SCHEMA["parameters"])
    unlock = Mock()
    backend = SimpleNamespace(name="custom", display_name="Custom", needs_unlock=True,
                              is_unlocked=lambda: False, unlock=unlock)
    token_backend = SimpleNamespace(name="token_provider", needs_unlock=False)
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend, token_backend])
    monkeypatch.setattr("agent.vault_backends.unlock.can_prompt_here", lambda: True)
    prompt = Mock(return_value="synthetic-master")
    monkeypatch.setattr("agent.vault_backends.unlock.get_unlock_prompt_callback", lambda: prompt)

    raw = registry.dispatch("browser_vault_unlock", args)
    assert isinstance(raw, str) and json.loads(raw)["success"]
    assert "synthetic-master" not in raw
    unlock.assert_called_once_with("synthetic-master")
    prompt.reset_mock()
    unlock.reset_mock()
    for name in ("unregistered", "token_provider"):
        raw = registry.dispatch("browser_vault_unlock", {"backend": name})
        assert isinstance(raw, str) and not json.loads(raw)["success"]
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [])
    raw = registry.dispatch("browser_vault_unlock", args)
    assert isinstance(raw, str) and not json.loads(raw)["success"]
    prompt.assert_not_called()
    unlock.assert_not_called()
