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
from hermes_state import SessionDB

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

    def test_make_agent_forwards_runtime_request_overrides(self, monkeypatch):
        config = {
            "model": {"default": "mimo-v2.5-pro", "provider": "custom:mimo-v2.5-pro"},
            "custom_providers": [{
                **LEGACY_LIST_CONFIG["custom_providers"][0],
                "model": "mimo-v2.5-pro",
                "extra_body": {"user": "proxy-user"},
            }],
        }

        kwargs = _make_agent_with_override(
            "mimo-v2.5-pro", monkeypatch, config, model_cfg=config["model"]
        )

        assert kwargs["request_overrides"] == {"extra_body": {"user": "proxy-user"}}


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


class TestStaleProviderNameFallsBack:
    """A session row stored under a provider that was renamed or removed must
    not sink agent init with "Unknown provider '<name>'": heal to the entry
    that still serves the stored model/base_url, else drop the provider so
    resume falls back to the configured default (or the user's pick)."""

    def test_stale_bare_name_heals_via_model(self, monkeypatch):
        """Registry serves mimo-v2.5-pro; the row still names the OLD slug —
        the exact shape of the renamed-provider report (oldone -> newone)."""
        monkeypatch.setattr(rp, "load_config", lambda: NAMED_CONFIG)
        monkeypatch.setattr(rp, "_get_model_config", lambda: NAMED_CONFIG["model"])

        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "mimo-v2.5-pro",
            "model_config": json.dumps(
                {"model": "mimo-v2.5-pro", "provider": "stale-provider"}
            ),
            "billing_provider": "custom",
        }
        overrides = _stored_session_runtime_overrides(row)

        assert overrides["provider_override"] == "custom:mimo-v2.5-pro"
        assert overrides["model_override"]["provider"] == "custom:mimo-v2.5-pro"

    def test_stale_prefixed_name_heals_and_drops_stale_base_url(self, monkeypatch):
        """Healing must also drop the snapshot's base_url so the registry URL
        (the renamed provider's current endpoint) is not overridden."""
        monkeypatch.setattr(rp, "load_config", lambda: NAMED_CONFIG)
        monkeypatch.setattr(rp, "_get_model_config", lambda: NAMED_CONFIG["model"])

        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "mimo-v2.5-pro",
            "model_config": json.dumps(
                {
                    "model": "mimo-v2.5-pro",
                    "provider": "custom:stale-provider",
                    "base_url": "https://old.invalid/v1",
                    "api_mode": "chat_completions",
                }
            ),
            "billing_provider": "custom:stale-provider",
        }
        overrides = _stored_session_runtime_overrides(row)

        assert overrides["provider_override"] == "custom:mimo-v2.5-pro"
        assert overrides["model_override"]["base_url"] is None

    def test_unrecoverable_provider_drops_to_default(self, monkeypatch):
        """No entry serves the stored model AND no configured default names a
        real entry → the provider is dropped; resume falls back to the
        configured default instead of failing the build."""
        config = {"custom_providers": NAMED_CONFIG["custom_providers"]}
        monkeypatch.setattr(rp, "load_config", lambda: config)
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})

        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "no-such-model",
            "model_config": json.dumps(
                {"model": "no-such-model", "provider": "dead-provider"}
            ),
            "billing_provider": "custom",
        }
        overrides = _stored_session_runtime_overrides(row)

        assert "provider_override" not in overrides
        assert overrides["model_override"]["provider"] is None

    def test_valid_provider_is_untouched(self, monkeypatch):
        """A live provider must round-trip unchanged — no healing, no drops."""
        monkeypatch.setattr(rp, "load_config", lambda: NAMED_CONFIG)
        monkeypatch.setattr(rp, "_get_model_config", lambda: NAMED_CONFIG["model"])

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
            "billing_provider": "custom:mimo-v2.5-pro",
        }
        overrides = _stored_session_runtime_overrides(row)

        assert overrides["provider_override"] == "custom:mimo-v2.5-pro"
        assert overrides["model_override"]["base_url"] == MIMO_URL


class TestOverridesHaveRoutableProvider:
    def test_gate_detects_stale_provider(self, monkeypatch):
        monkeypatch.setattr(rp, "load_config", lambda: NAMED_CONFIG)
        monkeypatch.setattr(rp, "_get_model_config", lambda: NAMED_CONFIG["model"])

        from tui_gateway.server import _overrides_have_routable_provider

        assert (
            _overrides_have_routable_provider(
                {"provider_override": "custom:mimo-v2.5-pro"}
            )
            is True
        )
        assert (
            _overrides_have_routable_provider(
                {"provider_override": "custom:stale-provider"}
            )
            is False
        )
        assert (
            _overrides_have_routable_provider(
                {"model_override": {"provider": None}}
            )
            is False
        )
        assert _overrides_have_routable_provider({}) is False


# --- Bot-Mode room plumbing sessions follow the profile's CURRENT config ------
#
# Room plumbing sessions are per-member scratch conversations inside a group
# chat (desktop Bot Mode). They must ALWAYS rebuild from the member profile's
# current config: restoring the stored model/provider pin from an old row is
# what left room bots stuck on a stale provider (e.g. "out of Nous credits"
# after the profile was switched to ollama-cloud) while the same bots worked
# fine in DMs. The stored-runtime restore stays intact for normal 1:1 chats.
#
# The contract is an EXPLICIT ``room_plumbing`` marker persisted in
# model_config (set by session.create/room consumers), with the hidden +
# "Group:" title shape kept as a legacy fallback for rows created by older
# desktop builds that never sent the marker.
#
# Regression: GH #89497 (room bots hang then report "out of Nous credits").


class TestRoomPlumbingRuntimeOverrides:
    def test_marked_row_returns_no_overrides(self):
        """A row carrying the room_plumbing marker never restores a stored
        provider pin — resume falls back to the profile's CURRENT config."""
        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "openai/gpt-5.6-luna-pro",
            "billing_provider": "nous",
            "model_config": json.dumps(
                {"model": "openai/gpt-5.6-luna-pro", "provider": "nous", "room_plumbing": True}
            ),
        }
        assert _stored_session_runtime_overrides(row) == {}

    def test_marked_row_dict_model_config(self):
        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "openai/gpt-5.6-luna-pro",
            "model_config": {"model": "openai/gpt-5.6-luna-pro", "provider": "nous", "room_plumbing": True},
        }
        assert _stored_session_runtime_overrides(row) == {}

    def test_legacy_group_title_shape_still_skipped(self):
        """Rows from older desktop builds (hidden + "Group:" title, no
        marker) keep the legacy guard: they also rebuild from current config."""
        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "title": "Group: Ceo, Product Designer, Cfo, COO, CTO, Coding",
            "hidden": 1,
            "model": "openai/gpt-5.6-luna-pro",
            "billing_provider": "nous",
            "model_config": json.dumps({"model": "openai/gpt-5.6-luna-pro", "provider": "nous"}),
        }
        assert _stored_session_runtime_overrides(row) == {}

    def test_normal_row_still_restores_stored_runtime(self):
        """The intended stored-runtime restore is untouched for normal 1:1
        chats: reopening an old chat shows the model it actually used."""
        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "title": "Analyze business idea gaps",
            "hidden": 0,
            "model": "glm-5.1",
            "billing_provider": "ollama-cloud",
            "model_config": json.dumps(
                {"model": "glm-5.1", "provider": "ollama-cloud", "service_tier": "normal"}
            ),
        }
        overrides = _stored_session_runtime_overrides(row)
        assert overrides["model_override"]["model"] == "glm-5.1"
        assert overrides["model_override"]["provider"] == "ollama-cloud"

    def test_hidden_normal_chat_untouched_by_legacy_shape(self):
        """A hidden NON-room chat (hidden without a "Group:" title) keeps the
        stored-runtime restore — the legacy shape is narrow on purpose."""
        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "title": "My hidden scratchpad",
            "hidden": 1,
            "model": "glm-5.1",
            "billing_provider": "ollama-cloud",
            "model_config": json.dumps({"model": "glm-5.1", "provider": "ollama-cloud"}),
        }
        overrides = _stored_session_runtime_overrides(row)
        assert overrides["model_override"]["model"] == "glm-5.1"


# --- Regression: bot DM stuck on a stale provider pin (GH #89497 class) ------
#
# Bot-Mode canonical chats (the ONE forever DM per bot) and room plumbing
# sessions are plugin-owned scratch conversations. They are created with the
# explicit ``follow_profile_config`` contract so resume ALWAYS rebuilds from
# the member profile's CURRENT config — restoring the stored model/provider
# pin from an old row is what left bot DMs stuck on a stale provider (e.g.
# "out of Nous credits" after the profile was switched to ollama-cloud) while
# the same bot worked fine in rooms. Normal 1:1 user chats keep the
# stored-runtime restore (opening an older chat must show the model it
# actually used).


class TestFollowProfileConfigRuntimeOverrides:
    def test_marked_row_returns_no_overrides(self):
        """A row carrying the follow_profile_config marker never restores a
        stored provider pin — resume falls back to the profile's CURRENT
        config."""
        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "openai/gpt-5.6-luna-pro",
            "billing_provider": "nous",
            "model_config": json.dumps(
                {
                    "model": "openai/gpt-5.6-luna-pro",
                    "provider": "nous",
                    "follow_profile_config": True,
                }
            ),
        }
        assert _stored_session_runtime_overrides(row) == {}

    def test_marked_row_dict_model_config_returns_no_overrides(self):
        """Same contract when model_config is already a dict (not JSON)."""
        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "openai/gpt-5.6-luna-pro",
            "model_config": {
                "model": "openai/gpt-5.6-luna-pro",
                "provider": "nous",
                "follow_profile_config": True,
            },
        }
        assert _stored_session_runtime_overrides(row) == {}

    def test_unmarked_row_still_restores_stored_runtime(self):
        """Normal 1:1 user chats keep the stored-runtime restore — the
        contract must not leak into ordinary sessions."""
        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "model": "openai/gpt-5.6-luna-pro",
            "billing_provider": "nous",
            "model_config": json.dumps(
                {"model": "openai/gpt-5.6-luna-pro", "provider": "nous"}
            ),
        }
        overrides = _stored_session_runtime_overrides(row)
        assert overrides["model_override"]["model"] == "openai/gpt-5.6-luna-pro"
        assert overrides["model_override"]["provider"] == "nous"

    def test_legacy_bot_chat_title_backfills_contract(self):
        """Canonical Bot Chats created BEFORE the marker existed carry no
        follow_profile_config, but they are still the plugin-owned forever-DM
        (identified by the exact title "Bot Chat"). They must also rebuild
        from the profile's CURRENT config — the live-report shape where every
        pre-existing Bot Chat stayed pinned to a deleted provider."""
        from tui_gateway.server import _stored_session_runtime_overrides

        for hidden in (0, 1):
            row = {
                "title": "Bot Chat",
                "hidden": hidden,
                "model": "openai/gpt-5.6-luna-pro",
                "billing_provider": "nous",
                "model_config": json.dumps(
                    {"model": "openai/gpt-5.6-luna-pro", "provider": "nous"}
                ),
            }
            assert _stored_session_runtime_overrides(row) == {}

    def test_bot_chat_prefix_title_is_not_backfilled(self):
        """Only the EXACT canonical title matches the legacy backfill — a
        user chat that merely mentions bots keeps its stored runtime."""
        from tui_gateway.server import _stored_session_runtime_overrides

        row = {
            "title": "Bot Chat ideas for my app",
            "hidden": 0,
            "model": "glm-5.1",
            "billing_provider": "ollama-cloud",
            "model_config": json.dumps(
                {"model": "glm-5.1", "provider": "ollama-cloud"}
            ),
        }
        overrides = _stored_session_runtime_overrides(row)
        assert overrides["model_override"]["model"] == "glm-5.1"

    def test_ensure_db_row_persists_contract_marker(self, monkeypatch):
        """_ensure_session_db_row stamps follow_profile_config into the row's
        model_config when the session carries the contract."""
        import tui_gateway.server as server

        captured = {}

        class FakeDB:
            def create_session(self, *args, **kwargs):
                captured["model_config"] = kwargs.get("model_config")
                return None

        monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
        monkeypatch.setattr(server, "_resolve_model", lambda: "glm-5.1")

        session = {
            "session_key": "key-1",
            "model_override": {"model": "glm-5.1", "provider": "ollama-cloud"},
            "follow_profile_config": True,
        }
        server._ensure_session_db_row(session)
        assert captured["model_config"].get("follow_profile_config") is True

    def test_ensure_db_row_omits_marker_without_contract(self, monkeypatch):
        """Sessions without the contract do NOT get the marker — normal chats
        keep the stored-runtime restore."""
        import tui_gateway.server as server

        captured = {}

        class FakeDB:
            def create_session(self, *args, **kwargs):
                captured["model_config"] = kwargs.get("model_config")
                return None

        monkeypatch.setattr(server, "_get_db", lambda: FakeDB())
        monkeypatch.setattr(server, "_resolve_model", lambda: "glm-5.1")

        session = {
            "session_key": "key-2",
            "model_override": {"model": "glm-5.1", "provider": "ollama-cloud"},
        }
        server._ensure_session_db_row(session)
        assert captured["model_config"].get("follow_profile_config") is None


# --- Regression: model column vs model_config desync (stale provider) ----------
#
# _runtime_model_config merges the agent's CURRENT identity onto the row's
# existing model_config. For model/provider it only SET the key when the agent
# attribute was truthy — so a falsy agent provider (agent inherits the profile
# default) left the PREVIOUS provider in the JSON while
# _persist_live_session_runtime updated the model column separately. Resume
# then read the fresh model from the column but the STALE provider/endpoint
# from model_config, silently routing the chat to the wrong provider (e.g. a
# VeniceAI/empero endpoint under a model that should run on Nous). The sibling
# CLI path (_persist_model_switch_to_session) already deletes stale keys with
# or-None; the gateway writer must drop them too, not merely omit the write.


def _agent_like(model="deepseek/deepseek-v4-flash-0731", provider=""):
    return types.SimpleNamespace(
        model=model,
        provider=provider,
        base_url="",
        api_mode="",
        reasoning_config=None,
        service_tier=None,
    )


class TestRuntimeModelConfigDropsStaleKeys:
    def test_falsy_provider_drops_stale_existing_provider(self):
        """Agent inherits the profile default (empty provider): the previously
        persisted provider must NOT survive the merge."""
        from tui_gateway.server import _runtime_model_config

        existing = {
            "model": "deepseek/deepseek-v4-flash-0731",
            "provider": "stealth-ox-alpha",  # stale from an earlier state
            "base_url": "https://api.venice.ai/api/v1",
            "api_mode": "chat_completions",
        }
        config = _runtime_model_config(_agent_like(), existing)

        assert config["model"] == "deepseek/deepseek-v4-flash-0731"
        assert "provider" not in config, config
        assert "base_url" not in config, config
        assert "api_mode" not in config, config

    def test_falsy_model_drops_stale_existing_model(self):
        """Mirror the provider rule: an empty agent model cannot keep the row's
        old model as its own."""
        from tui_gateway.server import _runtime_model_config

        agent = _agent_like(model="", provider="nous")
        existing = {"model": "meituan/longcat-2.0:free", "provider": "nous"}
        config = _runtime_model_config(agent, existing)

        assert "model" not in config, config
        assert config["provider"] == "nous"

    def test_truthy_provider_overwrites_stale_existing(self):
        from tui_gateway.server import _runtime_model_config

        existing = {
            "model": "deepseek/deepseek-v4-flash-0731",
            "provider": "stealth-ox-alpha",
            "base_url": "https://api.venice.ai/api/v1",
        }
        config = _runtime_model_config(_agent_like(provider="nous"), existing)

        assert config["provider"] == "nous"
        assert config["model"] == "deepseek/deepseek-v4-flash-0731"

    def test_resume_overrides_get_no_stale_provider(self):
        """End-to-end shape: a config merged from an empty-provider agent must
        NOT resurrect the stale endpoint on resume — the chat falls back to
        the row's billing provider (the profile default) instead of the stale
        VeniceAI/empero route."""
        from tui_gateway.server import (
            _runtime_model_config,
            _stored_session_runtime_overrides,
        )

        existing = {
            "model": "deepseek/deepseek-v4-flash-0731",
            "provider": "stealth-ox-alpha",
            "base_url": "https://api.venice.ai/api/v1",
        }
        config = _runtime_model_config(_agent_like(), existing)
        row = {
            "model": "deepseek/deepseek-v4-flash-0731",
            "model_config": json.dumps(config),
            "billing_provider": "nous",
        }
        overrides = _stored_session_runtime_overrides(row)

        assert overrides["model_override"]["model"] == "deepseek/deepseek-v4-flash-0731"
        # The stale endpoint identity is gone; resume routes through the
        # billing fallback to the profile's real provider.
        assert overrides["model_override"]["provider"] == "nous"
        assert overrides["provider_override"] == "nous"

    def test_real_db_persist_heals_desynced_row(self, tmp_path, monkeypatch):
        """A row already desynced (fresh model column + stale model_config
        provider) self-heals on the next live metadata persist."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="desync1", source="desktop", model="old-model")
        db.update_session_meta(
            "desync1",
            json.dumps(
                {
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "provider": "stealth-ox-alpha",
                    "base_url": "https://api.venice.ai/api/v1",
                }
            ),
            model="deepseek/deepseek-v4-flash-0731",
        )

        from tui_gateway.server import _runtime_model_config

        row = db.get_session("desync1")
        assert row is not None
        existing = json.loads(row["model_config"])
        merged = _runtime_model_config(_agent_like(), existing)
        db.update_session_meta("desync1", json.dumps(merged), model="deepseek/deepseek-v4-flash-0731")

        healed_row = db.get_session("desync1")
        assert healed_row is not None
        healed = json.loads(healed_row["model_config"])
        assert healed["model"] == "deepseek/deepseek-v4-flash-0731"
        assert "provider" not in healed, healed
        assert "base_url" not in healed, healed

    def test_existing_none_returns_only_agent_identity(self):
        """First write (no existing row): the merge starts from an empty dict
        and reflects only the agent's current identity — no stale keys, no
        crash on the None existing_config."""
        from tui_gateway.server import _runtime_model_config

        config = _runtime_model_config(_agent_like(provider="nous"), None)

        assert config == {"model": "deepseek/deepseek-v4-flash-0731", "provider": "nous"}


