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

import pytest

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


def _make_agent_with_override(override, monkeypatch, config, model_cfg=None):
    """Run _make_agent through the REAL resolve_runtime_provider against a
    patched config, returning the kwargs AIAgent was constructed with."""
    monkeypatch.setattr(rp, "load_config", lambda: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: model_cfg or {})
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


# --- Regression: bare "custom" WITHOUT a base_url (GH #44022 / #47714) ------
#
# The recurring Desktop/TUI "No LLM provider configured" regression. Every
# point-fix above recovers the entry identity from the persisted base_url —
# but a session can be persisted/restored with bare ``provider="custom"`` and
# NO base_url (the agent was built without one on the override). Then bare
# "custom" leaked through verbatim, ``resolve_runtime_provider("custom")``
# routed to the OpenRouter default URL with no api_key, and the next turn /
# resume failed with "No LLM provider configured". These tests lock the
# config-fallback recovery at all three leak sites so it cannot regress again.

NAMED_CONFIG = {
    "model": {"default": "mimo-v2.5-pro", "provider": "custom:mimo-v2.5-pro"},
    "custom_providers": [
        {
            "name": "mimo-v2.5-pro",
            "base_url": MIMO_URL,
            "api_key": MIMO_KEY,
            "api_mode": "chat_completions",
        }
    ],
}


class TestBareCustomNoBaseUrlHealsFromConfig:
    """A named custom provider must never escape as bare ``"custom"`` when the
    config identifies the active entry — even when no base_url survived."""

    def test_canonical_identity_recovers_from_config_when_no_base_url(
        self, monkeypatch
    ):
        monkeypatch.setattr(rp, "load_config", lambda: NAMED_CONFIG)
        monkeypatch.setattr(rp, "_get_model_config", lambda: NAMED_CONFIG["model"])

        # No base_url to reverse-lookup → must fall back to config.model.provider.
        assert (
            rp.canonical_custom_identity(base_url=None)
            == "custom:mimo-v2.5-pro"
        )


    def test_persist_recovers_entry_when_agent_has_no_base_url(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: NAMED_CONFIG)
        monkeypatch.setattr(rp, "_get_model_config", lambda: NAMED_CONFIG["model"])

        from tui_gateway.server import _runtime_model_config

        agent = _custom_agent(base_url="")  # the regression vector
        config = _runtime_model_config(agent)

        # Bare "custom" must NOT be persisted — it heals to the entry identity.
        assert config["provider"] == "custom:mimo-v2.5-pro"

    def test_restore_heals_bare_custom_row_without_base_url(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: NAMED_CONFIG)
        monkeypatch.setattr(rp, "_get_model_config", lambda: NAMED_CONFIG["model"])

        from tui_gateway.server import _stored_session_runtime_overrides

        # A poisoned row from before the fix: bare custom, no base_url.
        row = {
            "model": "mimo-v2.5-pro",
            "model_config": json.dumps(
                {"model": "mimo-v2.5-pro", "provider": "custom"}
            ),
            "billing_provider": "custom",
        }
        overrides = _stored_session_runtime_overrides(row)

        assert overrides["provider_override"] == "custom:mimo-v2.5-pro"
        assert overrides["model_override"]["provider"] == "custom:mimo-v2.5-pro"


    def test_make_agent_heals_bare_custom_no_base_url_end_to_end(self, monkeypatch):
        """The exact failing path: stored override has bare custom + no
        base_url; _make_agent must build the AIAgent with the named entry's
        endpoint + key, NOT the OpenRouter default with an empty key."""
        override = {
            "model": "mimo-v2.5-pro",
            "provider": "custom",
            "base_url": None,
            "api_mode": "chat_completions",
        }

        kwargs = _make_agent_with_override(
            override, monkeypatch, NAMED_CONFIG, model_cfg=NAMED_CONFIG["model"]
        )

        assert kwargs["base_url"] == MIMO_URL
        assert kwargs["api_key"] == MIMO_KEY
        assert "openrouter.ai" not in (kwargs.get("base_url") or "")

    def test_first_db_row_persists_entry_identity_not_bare_custom(self, monkeypatch):
        """The ORIGIN of poisoned rows: a fresh desktop session's first DB
        write (_ensure_session_db_row, before the agent is built) copies the
        composer override's RESOLVED provider. A named custom provider's
        resolved value is bare "custom" — persisting that verbatim seeds the
        unresumable row. It must be healed to ``custom:<name>`` here."""
        monkeypatch.setattr(rp, "load_config", lambda: NAMED_CONFIG)
        monkeypatch.setattr(rp, "_get_model_config", lambda: NAMED_CONFIG["model"])

        captured = {}

        class _DB:
            def create_session(self, key, **kwargs):
                captured.update(kwargs)

        from tui_gateway import server as srv

        monkeypatch.setattr(srv, "_get_db", lambda: _DB())
        monkeypatch.setattr(srv, "_resolve_model", lambda: "mimo-v2.5-pro")

        session = {
            "session_key": "agent:main:desktop:dm:abc",
            # composer override carrying the lossy resolved provider + no base_url
            "model_override": {"model": "mimo-v2.5-pro", "provider": "custom"},
        }
        srv._ensure_session_db_row(session)

        persisted = captured.get("model_config") or {}
        assert persisted.get("provider") == "custom:mimo-v2.5-pro"


# --- Regression: bare "custom" + no base_url + DIFFERENT default provider ----
#
# The config-provider fallback above only heals when ``config.model.provider``
# still points at the custom entry. A user whose global default is a built-in
# provider (e.g. Nous) but who switched THIS session to a self-hosted model
# gets no heal: the bare provider is dropped, resume falls back to the default
# provider, and the default provider's endpoint 404s with "Model '<x>' not
# found" (the b200/hermes-ultra-sft report). The stored MODEL NAME is the one
# session-scoped fact that still identifies the entry — these tests lock the
# model-name recovery tier.

ULTRA_URL = "http://b200-cluster:30090/v1"

ULTRA_CONFIG = {
    # Global default deliberately points at a BUILT-IN provider — the config
    # fallback must not fire; only the model lookup can recover the entry.
    "model": {"default": "some-nous-model", "provider": "nous"},
    "providers": {
        "hermes-ultra": {
            "api": ULTRA_URL,
            "api_key": "sk-ultra",
            "models": ["hermes-ultra-sft"],
        }
    },
}

ULTRA_LEGACY_CONFIG = {
    "model": {"default": "some-nous-model", "provider": "nous"},
    "custom_providers": [
        {
            "name": "hermes-ultra",
            "base_url": ULTRA_URL,
            "api_key": "sk-ultra",
            "model": "hermes-ultra-sft",
        }
    ],
}


class TestModelNameRecoversEntryIdentity:
    def test_identity_by_model_from_providers_dict_models_list(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: ULTRA_CONFIG)

        assert (
            rp.find_custom_provider_identity_by_model("hermes-ultra-sft")
            == "custom:hermes-ultra"
        )


# --- Regression: custom:<name> slug whose config entry was deleted (GH #75128) ---
#
# A session row persisted while a custom endpoint existed keeps
# ``provider: custom:<name>`` in ``model_config``. When that entry is later
# deleted from config (a dead HuggingFace endpoint, a retired gateway, …),
# restoring the stored identity fails agent init with "Unknown provider
# 'custom:<name>'" on Desktop/TUI while the CLI — which never restores
# session runtime — resumes the same session fine with the configured
# default. These tests lock the two-layer fallback: stored overrides are
# dropped at restore time, and _make_agent degrades to the configured
# default for any stale override that still reaches it.

DELETED_SLUG = "custom:deepseek-v4-0731"
DELETED_URL = (
    "https://q5dh1rfszfym23hj.us-east-2.aws.endpoints.huggingface.cloud/v1"
)

NO_CUSTOM_ENTRY_CONFIG = {
    # The dead endpoint's entry is gone; the global default is a built-in.
    "model": {"default": "deepseek/deepseek-v4-flash-0731", "provider": "nous"},
    "custom_providers": [],
}

DEFAULT_MIMO_CONFIG = {
    "model": {"default": "mimo-v2.5-pro", "provider": "custom:mimo-v2.5-pro"},
    "custom_providers": [
        {
            "name": "mimo-v2.5-pro",
            "base_url": MIMO_URL,
            "api_key": MIMO_KEY,
            "api_mode": "chat_completions",
        }
    ],
}

STALE_ROW = {
    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "model_config": json.dumps(
        {
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "provider": DELETED_SLUG,
            "base_url": DELETED_URL,
            "api_mode": "chat_completions",
            "reasoning_config": {"enabled": True, "effort": "max"},
            "service_tier": "normal",
        }
    ),
    "billing_provider": "custom",
}


class TestStaleCustomProviderSlugFallsBackToDefault:
    def test_stored_overrides_drop_deleted_slug(self, monkeypatch):
        """A deleted custom endpoint must not be restored: resume falls back
        to the configured default (CLI parity) instead of failing with
        'Unknown provider'."""
        monkeypatch.setattr(rp, "load_config", lambda: NO_CUSTOM_ENTRY_CONFIG)

        from tui_gateway.server import _stored_session_runtime_overrides

        overrides = _stored_session_runtime_overrides(STALE_ROW)

        assert "model_override" not in overrides
        assert "provider_override" not in overrides
        # Orthogonal session preferences survive the staleness gate.
        assert overrides["reasoning_config_override"] == {
            "enabled": True,
            "effort": "max",
        }

    def test_stored_overrides_keep_existing_entry(self, monkeypatch):
        """A still-configured custom entry keeps restoring (no regression)."""
        monkeypatch.setattr(rp, "load_config", lambda: LEGACY_LIST_CONFIG)

        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "mimo-v2.5-pro",
            "model_config": json.dumps(
                {
                    "model": "mimo-v2.5-pro",
                    "provider": "custom:mimo-v2.5-pro",
                    "base_url": MIMO_URL,
                    "api_mode": "chat_completions",
                }
            ),
            "billing_provider": "custom",
        }
        overrides = _stored_session_runtime_overrides(row)

        assert overrides["provider_override"] == "custom:mimo-v2.5-pro"
        assert overrides["model_override"]["provider"] == "custom:mimo-v2.5-pro"

    def test_stored_overrides_keep_builtin_provider(self, monkeypatch):
        """Built-in providers are untouched by the staleness gate."""
        monkeypatch.setattr(rp, "load_config", lambda: {})

        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "claude-sonnet-4-5",
            "model_config": json.dumps(
                {"model": "claude-sonnet-4-5", "provider": "anthropic"}
            ),
            "billing_provider": "anthropic",
        }
        overrides = _stored_session_runtime_overrides(row)

        assert overrides["provider_override"] == "anthropic"
        assert overrides["model_override"]["provider"] == "anthropic"

    def test_make_agent_falls_back_to_default_end_to_end(self, monkeypatch):
        """The exact failing path: a stale override slug reaches _make_agent;
        the agent must build with the configured default instead of failing
        with 'Unknown provider'."""
        override = {
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "provider": DELETED_SLUG,
            "base_url": DELETED_URL,
            "api_mode": "chat_completions",
        }
        fake_cfg = {
            "agent": {"system_prompt": ""},
            "model": {
                "default": "mimo-v2.5-pro",
                "provider": "custom:mimo-v2.5-pro",
            },
        }
        monkeypatch.setattr(rp, "load_config", lambda: DEFAULT_MIMO_CONFIG)
        monkeypatch.setattr(
            rp, "_get_model_config", lambda: DEFAULT_MIMO_CONFIG["model"]
        )
        monkeypatch.setattr(
            rp, "_try_resolve_from_custom_pool", lambda *a, **k: None
        )

        from tui_gateway import server as srv

        with (
            patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
            patch("tui_gateway.server._get_db", return_value=MagicMock()),
            patch("tui_gateway.server._load_reasoning_config", return_value=None),
            patch("tui_gateway.server._load_service_tier", return_value=None),
            patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
            patch("run_agent.AIAgent") as mock_agent,
        ):
            from tui_gateway.server import _make_agent

            _make_agent("sid-stale", "key-stale", model_override=override)

        kwargs = mock_agent.call_args.kwargs
        assert kwargs["model"] == "mimo-v2.5-pro"
        assert kwargs["provider"] == "custom"
        assert kwargs["base_url"] == MIMO_URL
        assert kwargs["api_key"] == MIMO_KEY
        assert DELETED_URL not in (kwargs.get("base_url") or "")


class TestResolveRuntimeOrDefault:
    def test_unknown_provider_falls_back_to_startup_default(self, monkeypatch):
        from hermes_cli.auth import AuthError

        from tui_gateway import server as srv

        def _boom(*args, **kwargs):
            raise AuthError(
                "Unknown provider 'custom:deepseek-v4-0731'",
                code="invalid_provider",
            )

        monkeypatch.setattr(srv, "_resolve_runtime_with_fallback", _boom)
        monkeypatch.setattr(
            srv, "_resolve_startup_runtime", lambda: ("default-model", "nous")
        )

        captured = {}

        def _fake_resolve(**kwargs):
            captured.update(kwargs)
            return {
                "provider": "nous",
                "base_url": "https://nous.inference.example",
                "api_key": "tok",
                "api_mode": "chat_completions",
            }

        monkeypatch.setattr(rp, "resolve_runtime_provider", _fake_resolve)

        resolution = srv._resolve_runtime_or_default(
            {"requested": DELETED_SLUG}, DELETED_SLUG
        )

        assert resolution.used_fallback is True
        assert resolution.selected_model == "default-model"
        assert resolution.runtime["provider"] == "nous"
        # The fallback must resolve the CONFIG default, not the stale slug.
        assert captured == {"requested": "nous", "target_model": "default-model"}

    def test_other_auth_errors_propagate(self, monkeypatch):
        """Credential/availability failures must stay loud — only the
        unknown-provider class degrades to the configured default."""
        from hermes_cli.auth import AuthError

        from tui_gateway import server as srv

        def _boom(*args, **kwargs):
            raise AuthError("no usable credentials", code="auth_unavailable")

        monkeypatch.setattr(srv, "_resolve_runtime_with_fallback", _boom)

        with pytest.raises(AuthError) as exc_info:
            srv._resolve_runtime_or_default({"requested": "nous"}, "nous")

        assert exc_info.value.code == "auth_unavailable"


