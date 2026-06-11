"""Session persistence must not strip a custom provider's identity.

``_runtime_model_config`` persists the live agent's RESOLVED provider into
the session row's ``model_config`` JSON. For any named ``providers:`` /
``custom_providers:`` entry (e.g. one called "mimo-v2.5-pro"),
``agent.provider`` is the literal string "custom", so the entry name was
lost — and the api_key is deliberately never persisted. On ``session.resume``
or ``_reset_session_agent``, ``_stored_session_runtime_overrides`` fed
provider="custom" back into ``_make_agent`` →
``resolve_runtime_provider(requested="custom")``, which cannot match an entry
named "mimo-v2.5-pro". Depending on config the rebuild either raised
"No LLM provider configured. Run `hermes model`..." (resume failed) or
silently resolved placeholder credentials ("no-key-required") against the
patched-back base_url.

Fix: persist the REQUESTED/entry identity — ``_runtime_model_config`` maps
the agent's base_url back to the canonical ``custom:<name>`` menu key via
``find_custom_provider_identity``; ``_make_agent`` performs the same
recovery for rows persisted before the fix (and falls back to handing the
stored base_url to the direct-alias branch when no entry matches).

Related investigation: GH #44070 / PR #44099 (credential-pool base_url
pinning); same family of resolved-vs-requested identity loss.
"""

import json
import types
from unittest.mock import MagicMock, patch

import hermes_cli.runtime_provider as rp

MIMO_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_KEY = "sk-mimo-entry-key"

LEGACY_LIST_CONFIG = {
    "custom_providers": [
        {
            "name": "mimo-v2.5-pro",
            "base_url": MIMO_URL,
            "api_key": MIMO_KEY,
            "api_mode": "chat_completions",
        }
    ]
}

PROVIDERS_DICT_CONFIG = {
    "providers": {
        "mimo-v2.5-pro": {
            "api": MIMO_URL,
            "api_key": MIMO_KEY,
        }
    }
}


def _custom_agent(base_url=MIMO_URL):
    return types.SimpleNamespace(
        model="mimo-v2.5-pro",
        provider="custom",
        base_url=base_url,
        api_mode="chat_completions",
        reasoning_config=None,
        service_tier=None,
    )


class TestRuntimeModelConfigPersistsEntryIdentity:
    def test_persists_menu_key_instead_of_resolved_custom(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: LEGACY_LIST_CONFIG)

        from tui_gateway.server import _runtime_model_config

        config = _runtime_model_config(_custom_agent())

        assert config["provider"] == "custom:mimo-v2.5-pro"
        assert config["base_url"] == MIMO_URL
        # Credentials must keep coming from config/provider resolution,
        # never from the session DB.
        assert "api_key" not in config

    def test_persists_menu_key_for_providers_dict_entry(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: PROVIDERS_DICT_CONFIG)

        from tui_gateway.server import _runtime_model_config

        config = _runtime_model_config(_custom_agent())

        assert config["provider"] == "custom:mimo-v2.5-pro"

    def test_keeps_bare_custom_when_no_entry_matches(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: {})

        from tui_gateway.server import _runtime_model_config

        config = _runtime_model_config(_custom_agent())

        assert config["provider"] == "custom"

    def test_non_custom_provider_untouched(self, monkeypatch):
        def _boom():
            raise AssertionError("identity lookup must not run for built-ins")

        monkeypatch.setattr(rp, "load_config", _boom)

        from tui_gateway.server import _runtime_model_config

        agent = _custom_agent()
        agent.provider = "anthropic"
        agent.base_url = "https://api.anthropic.com"

        assert _runtime_model_config(agent)["provider"] == "anthropic"


def _make_agent_with_override(override, monkeypatch, config):
    """Run _make_agent through the REAL resolve_runtime_provider against a
    patched config, returning the kwargs AIAgent was constructed with."""
    monkeypatch.setattr(rp, "load_config", lambda: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    # Keep credential-pool resolution off the developer's real HERMES home.
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)

    fake_cfg = {"agent": {"system_prompt": ""}, "model": {"default": "unused"}}
    with (
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch("run_agent.AIAgent") as mock_agent,
    ):
        from tui_gateway.server import _make_agent

        _make_agent("sid-custom", "key-custom", model_override=override)

    return mock_agent.call_args.kwargs


class TestResumeRoundTrip:
    def test_round_trip_restores_entry_credentials(self, monkeypatch):
        """persist → stored-overrides → _make_agent resolves the entry's
        api_key again (the exact path that raised "No LLM provider
        configured" before the fix)."""
        monkeypatch.setattr(rp, "load_config", lambda: LEGACY_LIST_CONFIG)

        from tui_gateway.server import (
            _runtime_model_config,
            _stored_session_runtime_overrides,
        )

        model_config = _runtime_model_config(_custom_agent())
        row = {
            "model": "mimo-v2.5-pro",
            "model_config": json.dumps(model_config),
        }
        overrides = _stored_session_runtime_overrides(row)
        assert overrides["model_override"]["provider"] == "custom:mimo-v2.5-pro"

        kwargs = _make_agent_with_override(
            overrides["model_override"], monkeypatch, LEGACY_LIST_CONFIG
        )

        assert kwargs["provider"] == "custom"
        assert kwargs["base_url"] == MIMO_URL
        assert kwargs["api_key"] == MIMO_KEY

    def test_legacy_row_with_bare_custom_heals_via_base_url(self, monkeypatch):
        """Rows persisted BEFORE the fix stored provider="custom"; the
        rebuild must recover the entry identity from the stored base_url."""
        override = {
            "model": "mimo-v2.5-pro",
            "provider": "custom",
            "base_url": MIMO_URL,
            "api_mode": "chat_completions",
        }

        kwargs = _make_agent_with_override(override, monkeypatch, LEGACY_LIST_CONFIG)

        assert kwargs["base_url"] == MIMO_URL
        assert kwargs["api_key"] == MIMO_KEY

    def test_legacy_row_without_matching_entry_keeps_endpoint(self, monkeypatch):
        """No config entry owns the stored URL: the direct-alias branch must
        still receive the base_url so resolution targets the session's
        endpoint instead of raising auth_unavailable."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        override = {
            "model": "local-model",
            "provider": "custom",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_mode": "chat_completions",
        }

        kwargs = _make_agent_with_override(override, monkeypatch, {})

        assert kwargs["provider"] == "custom"
        assert kwargs["base_url"] == "http://127.0.0.1:8000/v1"
        assert kwargs["api_key"] == "no-key-required"
