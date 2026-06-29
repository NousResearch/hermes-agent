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

import pytest

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


class TestHermesInternalDynamicSecrets:
    """AUXILIARY_<TASK>_* and GATEWAY_RELAY_* internal secrets are stripped in
    BOTH inherit_credentials modes.

    These credentials are injected into ``os.environ`` at runtime by the
    gateway and CLI but match no static ``PROVIDER_REGISTRY`` name, so they
    survive both Tier 1 (``_ALWAYS_STRIP_KEYS``) and Tier 2
    (``_HERMES_PROVIDER_ENV_BLOCKLIST``) unless the dynamic-secret predicate
    (:func:`_is_hermes_internal_secret`) strips them.  They are Tier-1
    gateway-managed secrets — not LLM provider credentials — so they must be
    stripped regardless of ``inherit_credentials``: even a user-blessed
    ``claude`` / ``codex`` CLI executor never legitimately needs a side-LLM
    key or relay-auth material.
    """

    _INTERNAL_SECRETS = {
        "AUXILIARY_VISION_API_KEY": "sk-aux-vision-secret-1234567890",
        "AUXILIARY_VISION_BASE_URL": "https://private-vision.example.com",
        "GATEWAY_RELAY_SECRET": "relay-shared-secret-value",
        "GATEWAY_RELAY_DELIVERY_KEY": "relay-delivery-key-value",
    }
    _NON_SECRET_AUXILIARY = {
        "AUXILIARY_VISION_MODEL": "gpt-4o",
        "AUXILIARY_VISION_PROVIDER": "openai",
    }

    def _run(self, *, inherit_credentials):
        env = {**_SAFE_SAMPLE, **self._INTERNAL_SECRETS, **self._NON_SECRET_AUXILIARY}
        with patch.dict(os.environ, env, clear=True):
            return hermes_subprocess_env(inherit_credentials=inherit_credentials)

    def test_auxiliary_api_key_stripped_when_not_inheriting(self):
        result = self._run(inherit_credentials=False)
        assert "AUXILIARY_VISION_API_KEY" not in result, (
            "AUXILIARY_*_API_KEY must be stripped with inherit_credentials=False"
        )

    def test_auxiliary_base_url_stripped_when_not_inheriting(self):
        result = self._run(inherit_credentials=False)
        assert "AUXILIARY_VISION_BASE_URL" not in result, (
            "AUXILIARY_*_BASE_URL must be stripped with inherit_credentials=False"
        )

    def test_gateway_relay_secret_stripped_when_not_inheriting(self):
        result = self._run(inherit_credentials=False)
        assert "GATEWAY_RELAY_SECRET" not in result, (
            "GATEWAY_RELAY_SECRET must be stripped with inherit_credentials=False"
        )

    def test_auxiliary_api_key_stripped_even_when_inheriting(self):
        result = self._run(inherit_credentials=True)
        assert "AUXILIARY_VISION_API_KEY" not in result, (
            "AUXILIARY_*_API_KEY is a Tier-1 internal secret and must be "
            "stripped even with inherit_credentials=True"
        )

    def test_auxiliary_base_url_stripped_even_when_inheriting(self):
        result = self._run(inherit_credentials=True)
        assert "AUXILIARY_VISION_BASE_URL" not in result, (
            "AUXILIARY_*_BASE_URL is a Tier-1 internal secret and must be "
            "stripped even with inherit_credentials=True"
        )

    def test_gateway_relay_secret_stripped_even_when_inheriting(self):
        result = self._run(inherit_credentials=True)
        assert "GATEWAY_RELAY_SECRET" not in result, (
            "GATEWAY_RELAY_SECRET is a Tier-1 gateway secret and must be "
            "stripped even with inherit_credentials=True — a model-driving "
            "CLI never needs relay-auth material"
        )

    def test_gateway_relay_delivery_key_stripped_even_when_inheriting(self):
        result = self._run(inherit_credentials=True)
        assert "GATEWAY_RELAY_DELIVERY_KEY" not in result

    @pytest.mark.parametrize("inherit_credentials", [False, True])
    def test_non_secret_auxiliary_vars_preserved(self, inherit_credentials):
        """Non-secret auxiliary model/provider selection survives in BOTH
        modes — they are config, not credentials, and a child reading the
        configured auxiliary model legitimately depends on them."""
        result = self._run(inherit_credentials=inherit_credentials)
        assert result.get("AUXILIARY_VISION_MODEL") == "gpt-4o", (
            "AUXILIARY_*_MODEL is non-secret config and must survive"
        )
        assert result.get("AUXILIARY_VISION_PROVIDER") == "openai", (
            "AUXILIARY_*_PROVIDER is non-secret config and must survive"
        )
