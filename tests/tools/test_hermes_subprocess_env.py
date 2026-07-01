"""Tests for hermes_subprocess_env() — the centralized credential-safe env
builder for the non-terminal subprocess spawn surface.

Covers GHSA-m4m8-xjp4-5rmm / issue #29157: subprocesses spawned by the
gateway/browser/ACP/installer paths must not blindly inherit the operator's
full credential environment. Two tiers:

  * Tier 1 (_ALWAYS_STRIP_KEYS): gateway bot tokens, GitHub auth, infra
    secrets — stripped even when inherit_credentials=True.
  * Tier 2 (_HERMES_PROVIDER_ENV_BLOCKLIST): LLM provider/tool keys — stripped
    unless the caller opts into inherit_credentials=True.
"""

import os
from unittest.mock import patch

from tools.environments.local import (
    hermes_subprocess_env,
    _ALWAYS_STRIP_KEYS,
    _HERMES_PROVIDER_ENV_FORCE_PREFIX,
)


_TIER1_SAMPLE = {
    "GH_TOKEN": "ghp_secret",
    "TELEGRAM_BOT_TOKEN": "bot-token",
    "SLACK_APP_TOKEN": "xapp-secret",
    "MODAL_TOKEN_SECRET": "modal-secret",
    "HERMES_DASHBOARD_SESSION_TOKEN": "dash-secret",
}

_PROVIDER_SAMPLE = {
    "OPENAI_API_KEY": "sk-fake",
    "ANTHROPIC_API_KEY": "ant-fake",
    "OPENROUTER_API_KEY": "or-fake",
}

_SAFE_SAMPLE = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/user",
    "USER": "testuser",
    "MY_APP_VAR": "keep-me",
}


def _build(extra=None, *, inherit_credentials=False):
    env = dict(_SAFE_SAMPLE)
    if extra:
        env.update(extra)
    with patch.dict(os.environ, env, clear=True):
        return hermes_subprocess_env(inherit_credentials=inherit_credentials)


class TestStripByDefault:
    def test_provider_keys_stripped_by_default(self):
        result = _build(_PROVIDER_SAMPLE)
        for var in _PROVIDER_SAMPLE:
            assert var not in result, f"{var} leaked with inherit_credentials=False"

    def test_tier1_secrets_stripped_by_default(self):
        result = _build(_TIER1_SAMPLE)
        for var in _TIER1_SAMPLE:
            assert var not in result, f"{var} leaked (Tier-1) with inherit_credentials=False"

    def test_safe_vars_preserved(self):
        result = _build()
        assert result["HOME"] == "/home/user"
        assert result["USER"] == "testuser"
        assert "PATH" in result
        assert result["MY_APP_VAR"] == "keep-me"

    def test_force_prefix_hints_stripped(self):
        result = _build({f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}OPENAI_API_KEY": "sk-x"})
        assert f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}OPENAI_API_KEY" not in result
        assert "OPENAI_API_KEY" not in result

    def test_pythonutf8_set(self):
        result = _build()
        assert result.get("PYTHONUTF8") == "1"


class TestInheritCredentials:
    def test_provider_keys_preserved_when_inheriting(self):
        result = _build(_PROVIDER_SAMPLE, inherit_credentials=True)
        for var, val in _PROVIDER_SAMPLE.items():
            assert result.get(var) == val, f"{var} should survive inherit_credentials=True"

    def test_tier1_secrets_stripped_even_when_inheriting(self):
        """The whole point of Tier 1: gateway/GitHub/infra secrets never reach
        a child, even a model-driving CLI that legitimately needs provider keys."""
        result = _build({**_PROVIDER_SAMPLE, **_TIER1_SAMPLE}, inherit_credentials=True)
        for var in _TIER1_SAMPLE:
            assert var not in result, (
                f"{var} (Tier-1) must be stripped even with inherit_credentials=True"
            )
        # ...while provider keys survive.
        for var in _PROVIDER_SAMPLE:
            assert var in result

    def test_pythonutf8_set_when_inheriting(self):
        assert _build(inherit_credentials=True).get("PYTHONUTF8") == "1"

    def test_tool_and_messaging_secrets_stripped_when_inheriting(self):
        """Provider inheritance must not hand unrelated Hermes secrets to child CLIs.

        Codex/Copilot subprocesses may need LLM provider credentials, but they
        do not need platform bot credentials, webhook/API-server bearer keys,
        browser backend keys, or service-account JSON paths used by gateway
        adapters. Those are non-provider Hermes secrets and must stay Tier-1.
        """
        non_provider_secrets = {
            "MATRIX_ACCESS_TOKEN": "matrix-token",
            "TEAMS_CLIENT_SECRET": "teams-secret",
            "WEBHOOK_SECRET": "webhook-secret",
            "API_SERVER_KEY": "api-server-key",
            "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON": "/home/user/google-chat-sa.json",
            "BROWSERBASE_API_KEY": "browserbase-key",
            "FIRECRAWL_API_KEY": "firecrawl-key",
        }
        result = _build(
            {**_PROVIDER_SAMPLE, **non_provider_secrets},
            inherit_credentials=True,
        )

        for var in non_provider_secrets:
            assert var not in result, (
                f"{var} must be stripped even with inherit_credentials=True"
            )
        for var, val in _PROVIDER_SAMPLE.items():
            assert result.get(var) == val


class TestTierInvariants:
    def test_tier1_always_stripped_both_paths(self):
        """Behavioral invariant: every Tier-1 key is stripped on BOTH the
        default path and the inherit_credentials=True path. This is what
        guarantees no gap, regardless of whether the key also happens to be
        in the provider blocklist."""
        sample = {k: f"secret-{k}" for k in _ALWAYS_STRIP_KEYS}
        for inherit in (False, True):
            result = _build(sample, inherit_credentials=inherit)
            leaked = {k for k in _ALWAYS_STRIP_KEYS if k in result}
            assert not leaked, (
                f"Tier-1 keys leaked with inherit_credentials={inherit}: {sorted(leaked)}"
            )

    def test_tier1_covers_gateway_bot_token(self):
        assert "TELEGRAM_BOT_TOKEN" in _ALWAYS_STRIP_KEYS

    def test_tier1_covers_github_auth(self):
        assert {"GH_TOKEN", "GITHUB_TOKEN"} <= _ALWAYS_STRIP_KEYS

    def test_tier1_covers_infra_secrets(self):
        assert {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "DAYTONA_API_KEY"} <= _ALWAYS_STRIP_KEYS


class TestBrowserPassthroughPattern:
    def test_browser_keys_recoverable_after_strip(self):
        """Browser tool pattern: strip everything, then re-add the browser
        backend keys agent-browser actually needs."""
        from tools.browser_tool import _BROWSER_PASSTHROUGH_KEYS

        leaked = {
            "BROWSERBASE_API_KEY": "bb-key",
            "BROWSERBASE_PROJECT_ID": "bb-proj",
            "FIRECRAWL_API_KEY": "fc-key",
            "ANTHROPIC_API_KEY": "ant-should-go",
            "TELEGRAM_BOT_TOKEN": "bot-should-go",
        }
        with patch.dict(os.environ, {**_SAFE_SAMPLE, **leaked}, clear=True):
            env = hermes_subprocess_env(inherit_credentials=False)
            for key in _BROWSER_PASSTHROUGH_KEYS:
                if key in os.environ:
                    env[key] = os.environ[key]

        assert env["BROWSERBASE_API_KEY"] == "bb-key"
        assert env["FIRECRAWL_API_KEY"] == "fc-key"
        # Provider + gateway secrets must NOT come back.
        assert "ANTHROPIC_API_KEY" not in env
        assert "TELEGRAM_BOT_TOKEN" not in env


_INTERNAL_DYNAMIC_SAMPLE = {
    "AUXILIARY_VISION_API_KEY": "sk-vision",
    "AUXILIARY_VISION_BASE_URL": "http://internal:1234/v1",
    "AUXILIARY_WEB_EXTRACT_API_KEY": "sk-webx",
    "GATEWAY_RELAY_SECRET": "relay-secret",
    "GATEWAY_RELAY_DELIVERY_KEY": "relay-delivery",
}


class TestInternalDynamicSecrets:
    """AUXILIARY_*_API_KEY / _BASE_URL and GATEWAY_RELAY_* auth are stripped on
    BOTH paths — including inherit_credentials=True — since a model-driving CLI
    (codex/copilot) never needs them even when it needs provider keys."""

    def test_stripped_by_default(self):
        result = _build(_INTERNAL_DYNAMIC_SAMPLE)
        for var in _INTERNAL_DYNAMIC_SAMPLE:
            assert var not in result, f"{var} leaked with inherit_credentials=False"

    def test_stripped_even_when_inheriting(self):
        result = _build(
            {**_PROVIDER_SAMPLE, **_INTERNAL_DYNAMIC_SAMPLE},
            inherit_credentials=True,
        )
        for var in _INTERNAL_DYNAMIC_SAMPLE:
            assert var not in result, (
                f"{var} must be stripped even with inherit_credentials=True"
            )
        # ...while genuine provider keys survive so codex can authenticate.
        for var in _PROVIDER_SAMPLE:
            assert var in result

    def test_auxiliary_non_secrets_preserved(self):
        """AUXILIARY_*_PROVIDER / _MODEL routing config survives (not secrets)."""
        result = _build(
            {"AUXILIARY_VISION_PROVIDER": "openai", "AUXILIARY_VISION_MODEL": "gpt-4o"},
        )
        assert result.get("AUXILIARY_VISION_PROVIDER") == "openai"
        assert result.get("AUXILIARY_VISION_MODEL") == "gpt-4o"

    def test_gateway_relay_id_stripped_even_when_inheriting(self):
        """GATEWAY_RELAY_ID has no secret suffix (predicate skips it) but is
        gateway-identifying auth material provisioned alongside the relay
        secret. It's in _ALWAYS_STRIP_KEYS so it's stripped on the inherit path
        too — closes the codex/copilot leak the predicate alone would miss."""
        result = _build(
            {**_PROVIDER_SAMPLE, "GATEWAY_RELAY_ID": "relay-id"},
            inherit_credentials=True,
        )
        assert "GATEWAY_RELAY_ID" not in result
        # provider keys still flow (codex auth)
        for var in _PROVIDER_SAMPLE:
            assert var in result

    def test_relay_triplet_in_always_strip(self):
        assert {
            "GATEWAY_RELAY_ID", "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_DELIVERY_KEY",
        } <= _ALWAYS_STRIP_KEYS
