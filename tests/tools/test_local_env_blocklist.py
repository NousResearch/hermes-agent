"""Tests for subprocess env sanitization in LocalEnvironment.

Verifies that Hermes-managed provider, tool, and gateway env vars are
stripped from subprocess environments so external CLIs are not silently
misrouted or handed Hermes secrets.

See: https://github.com/NousResearch/hermes-agent/issues/1002
See: https://github.com/NousResearch/hermes-agent/issues/1264
"""

import os
import threading
from unittest.mock import MagicMock, patch

from tools.environments.local import (
    LocalEnvironment,
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _HERMES_PROVIDER_ENV_FORCE_PREFIX,
)


def _make_fake_popen(captured: dict):
    """Return a fake Popen constructor that records the env kwarg."""
    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        proc.stdout = MagicMock(__iter__=lambda s: iter([]), __next__=lambda s: (_ for _ in ()).throw(StopIteration))
        proc.stdin = MagicMock()
        return proc
    return fake_popen


def _run_with_env(extra_os_env=None, self_env=None):
    """Execute a command via LocalEnvironment with mocked Popen
    and return the env dict passed to the subprocess."""
    captured = {}
    fake_interrupt = threading.Event()
    test_environ = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/user",
        "USER": "testuser",
    }
    if extra_os_env:
        test_environ.update(extra_os_env)

    env = LocalEnvironment(cwd="/tmp", timeout=10, env=self_env)

    with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
         patch("subprocess.Popen", side_effect=_make_fake_popen(captured)), \
         patch("tools.terminal_tool._interrupt_event", fake_interrupt), \
         patch.dict(os.environ, test_environ, clear=True):
        env.execute("echo hello")

    return captured.get("env", {})


class TestProviderEnvBlocklist:
    """Provider env vars loaded from ~/.hermes/.env must not leak."""

    def test_blocked_vars_are_stripped(self):
        """OPENAI_BASE_URL and other provider vars must not appear in subprocess env."""
        leaked_vars = {
            "OPENAI_BASE_URL": "http://localhost:8000/v1",
            "OPENAI_API_KEY": "sk-fake-key",
            "OPENROUTER_API_KEY": "or-fake-key",
            "ANTHROPIC_API_KEY": "ant-fake-key",
            "LLM_MODEL": "anthropic/claude-opus-4-6",
        }
        result_env = _run_with_env(extra_os_env=leaked_vars)

        for var in leaked_vars:
            assert var not in result_env, f"{var} leaked into subprocess env"

    def test_registry_derived_vars_are_stripped(self):
        """Vars from the provider registry (ANTHROPIC_TOKEN, ZAI_API_KEY, etc.)
        must also be blocked — not just the hand-written extras."""
        registry_vars = {
            "ANTHROPIC_TOKEN": "ant-tok",
            "CLAUDE_CODE_OAUTH_TOKEN": "cc-tok",
            "ZAI_API_KEY": "zai-key",
            "Z_AI_API_KEY": "z-ai-key",
            "GLM_API_KEY": "glm-key",
            "KIMI_API_KEY": "kimi-key",
            "MINIMAX_API_KEY": "mm-key",
            "MINIMAX_CN_API_KEY": "mmcn-key",
            "DEEPSEEK_API_KEY": "deepseek-key",
            "NVIDIA_API_KEY": "nvidia-key",
        }
        result_env = _run_with_env(extra_os_env=registry_vars)

        for var in registry_vars:
            assert var not in result_env, f"{var} leaked into subprocess env"

    def test_non_registry_provider_vars_are_stripped(self):
        """Extra provider vars not in PROVIDER_REGISTRY must also be blocked."""
        extra_provider_vars = {
            "GOOGLE_API_KEY": "google-key",
            "MISTRAL_API_KEY": "mistral-key",
            "GROQ_API_KEY": "groq-key",
            "TOGETHER_API_KEY": "together-key",
            "PERPLEXITY_API_KEY": "perplexity-key",
            "COHERE_API_KEY": "cohere-key",
            "FIREWORKS_API_KEY": "fireworks-key",
            "XAI_API_KEY": "xai-key",
            "HELICONE_API_KEY": "helicone-key",
        }
        result_env = _run_with_env(extra_os_env=extra_provider_vars)

        for var in extra_provider_vars:
            assert var not in result_env, f"{var} leaked into subprocess env"

    def test_tool_and_gateway_vars_are_stripped(self):
        """Tool and gateway secrets/config must not leak into subprocess env."""
        leaked_vars = {
            "TELEGRAM_BOT_TOKEN": "bot-token",
            "TELEGRAM_HOME_CHANNEL": "12345",
            "DISCORD_HOME_CHANNEL": "67890",
            "SLACK_APP_TOKEN": "xapp-secret",
            "WHATSAPP_ALLOWED_USERS": "+15555550123",
            "SIGNAL_ACCOUNT": "+15555550124",
            "HASS_TOKEN": "ha-secret",
            "EMAIL_PASSWORD": "email-secret",
            "FIRECRAWL_API_KEY": "fc-secret",
            "BROWSERBASE_PROJECT_ID": "bb-project",
            "ELEVENLABS_API_KEY": "el-secret",
            "GITHUB_TOKEN": "ghp_secret",
            "GH_TOKEN": "gh_alias_secret",
            "GATEWAY_ALLOW_ALL_USERS": "true",
            "GATEWAY_ALLOWED_USERS": "alice,bob",
            "MODAL_TOKEN_ID": "modal-id",
            "MODAL_TOKEN_SECRET": "modal-secret",
            "DAYTONA_API_KEY": "daytona-key",
            "VERCEL_OIDC_TOKEN": "vercel-oidc-token",
            "VERCEL_TOKEN": "vercel-token",
            "VERCEL_PROJECT_ID": "vercel-project",
            "VERCEL_TEAM_ID": "vercel-team",
        }
        result_env = _run_with_env(extra_os_env=leaked_vars)

        for var in leaked_vars:
            assert var not in result_env, f"{var} leaked into subprocess env"

    def test_safe_vars_are_preserved(self):
        """Standard env vars (PATH, HOME, USER) must still be passed through."""
        result_env = _run_with_env()

        assert "HOME" in result_env
        assert result_env["HOME"] == "/home/user"
        assert "USER" in result_env
        assert "PATH" in result_env

    def test_self_env_blocked_vars_also_stripped(self):
        """Blocked vars in self.env are stripped; non-blocked vars pass through."""
        result_env = _run_with_env(self_env={
            "OPENAI_BASE_URL": "http://custom:9999/v1",
            "MY_CUSTOM_VAR": "keep-this",
        })

        assert "OPENAI_BASE_URL" not in result_env
        assert "MY_CUSTOM_VAR" in result_env
        assert result_env["MY_CUSTOM_VAR"] == "keep-this"


class TestForceEnvOptIn:
    """Callers can opt in to passing a blocked var via _HERMES_FORCE_ prefix."""

    def test_force_prefix_passes_blocked_var(self):
        """_HERMES_FORCE_OPENAI_API_KEY in self.env should inject OPENAI_API_KEY."""
        result_env = _run_with_env(self_env={
            f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}OPENAI_API_KEY": "sk-explicit",
        })

        assert "OPENAI_API_KEY" in result_env
        assert result_env["OPENAI_API_KEY"] == "sk-explicit"
        # The force-prefixed key itself must not appear
        assert f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}OPENAI_API_KEY" not in result_env

    def test_force_prefix_overrides_os_environ_block(self):
        """Force-prefix in self.env wins even when os.environ has the blocked var."""
        result_env = _run_with_env(
            extra_os_env={"OPENAI_BASE_URL": "http://leaked/v1"},
            self_env={f"{_HERMES_PROVIDER_ENV_FORCE_PREFIX}OPENAI_BASE_URL": "http://intended/v1"},
        )

        assert result_env["OPENAI_BASE_URL"] == "http://intended/v1"


class TestBlocklistCoverage:
    """Sanity checks that the blocklist covers all known providers."""

    def test_issue_1002_offenders(self):
        """Blocklist includes the main offenders from issue #1002."""
        must_block = {
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "LLM_MODEL",
        }
        assert must_block.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)

    def test_registry_vars_are_in_blocklist(self):
        """Every api_key_env_var and base_url_env_var from PROVIDER_REGISTRY
        must appear in the blocklist — ensures no drift."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        for pconfig in PROVIDER_REGISTRY.values():
            for var in pconfig.api_key_env_vars:
                assert var in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Registry var {var} (provider={pconfig.id}) missing from blocklist"
                )
            if pconfig.base_url_env_var:
                assert pconfig.base_url_env_var in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Registry base_url_env_var {pconfig.base_url_env_var} "
                    f"(provider={pconfig.id}) missing from blocklist"
                )

    def test_extra_auth_vars_covered(self):
        """Non-registry auth vars (ANTHROPIC_TOKEN, CLAUDE_CODE_OAUTH_TOKEN)
        must also be in the blocklist."""
        extras = {"ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"}
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)

    def test_non_registry_provider_vars_are_in_blocklist(self):
        extras = {
            "GOOGLE_API_KEY",
            "DEEPSEEK_API_KEY",
            "MISTRAL_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "PERPLEXITY_API_KEY",
            "COHERE_API_KEY",
            "FIREWORKS_API_KEY",
            "XAI_API_KEY",
            "HELICONE_API_KEY",
        }
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)

    def test_optional_tool_and_messaging_vars_are_in_blocklist(self):
        """Tool/messaging vars from OPTIONAL_ENV_VARS should stay covered."""
        from hermes_cli.config import OPTIONAL_ENV_VARS

        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"}:
                assert name in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Optional env var {name} (category={category}) missing from blocklist"
                )
            elif category == "setting" and metadata.get("password"):
                assert name in _HERMES_PROVIDER_ENV_BLOCKLIST, (
                    f"Secret setting env var {name} missing from blocklist"
                )

    def test_gateway_runtime_vars_are_in_blocklist(self):
        extras = {
            "TELEGRAM_HOME_CHANNEL",
            "TELEGRAM_HOME_CHANNEL_NAME",
            "DISCORD_HOME_CHANNEL",
            "DISCORD_HOME_CHANNEL_NAME",
            "DISCORD_REQUIRE_MENTION",
            "DISCORD_FREE_RESPONSE_CHANNELS",
            "DISCORD_AUTO_THREAD",
            "SLACK_HOME_CHANNEL",
            "SLACK_HOME_CHANNEL_NAME",
            "SLACK_ALLOWED_USERS",
            "WHATSAPP_ENABLED",
            "WHATSAPP_MODE",
            "WHATSAPP_ALLOWED_USERS",
            "SIGNAL_HTTP_URL",
            "SIGNAL_ACCOUNT",
            "SIGNAL_ALLOWED_USERS",
            "SIGNAL_GROUP_ALLOWED_USERS",
            "SIGNAL_HOME_CHANNEL",
            "SIGNAL_HOME_CHANNEL_NAME",
            "SIGNAL_IGNORE_STORIES",
            "HASS_TOKEN",
            "HASS_URL",
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
            "EMAIL_HOME_ADDRESS",
            "EMAIL_HOME_ADDRESS_NAME",
            "GATEWAY_ALLOWED_USERS",
            "GH_TOKEN",
            "GITHUB_APP_ID",
            "GITHUB_APP_PRIVATE_KEY_PATH",
            "GITHUB_APP_INSTALLATION_ID",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "DAYTONA_API_KEY",
            "VERCEL_OIDC_TOKEN",
            "VERCEL_TOKEN",
            "VERCEL_PROJECT_ID",
            "VERCEL_TEAM_ID",
        }
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)


class TestSanePathIncludesHomebrew:
    """Verify _SANE_PATH includes macOS Homebrew directories."""

    def test_sane_path_includes_homebrew_bin(self):
        from tools.environments.local import _SANE_PATH
        assert "/opt/homebrew/bin" in _SANE_PATH

    def test_sane_path_includes_homebrew_sbin(self):
        from tools.environments.local import _SANE_PATH
        assert "/opt/homebrew/sbin" in _SANE_PATH

    def test_make_run_env_appends_homebrew_on_minimal_path(self):
        """When PATH is minimal (no /usr/bin), _make_run_env should append
        _SANE_PATH which now includes Homebrew dirs."""
        from tools.environments.local import _make_run_env
        minimal_env = {"PATH": "/some/custom/bin"}
        with patch.dict(os.environ, minimal_env, clear=True):
            result = _make_run_env({})
        assert "/opt/homebrew/bin" in result["PATH"]
        assert "/opt/homebrew/sbin" in result["PATH"]

    def test_make_run_env_does_not_duplicate_on_full_path(self):
        """When PATH already has /usr/bin, _make_run_env should not append."""
        from tools.environments.local import _make_run_env
        full_env = {"PATH": "/usr/bin:/bin"}
        with patch.dict(os.environ, full_env, clear=True):
            result = _make_run_env({})
        # Should keep existing PATH unchanged
        assert result["PATH"] == "/usr/bin:/bin"
