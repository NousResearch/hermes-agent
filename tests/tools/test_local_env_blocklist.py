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

    def test_bedrock_bearer_token_is_stripped(self):
        """The Bedrock-specific bearer token is a Hermes inference secret
        (analogous to OPENAI_API_KEY) and must not leak into subprocesses.

        Regression for #32314: AWS_BEARER_TOKEN_BEDROCK leaked into terminal /
        execute_code children because the ``bedrock`` ProviderConfig declares
        ``api_key_env_vars=()`` (auth_type="aws_sdk") and the blocklist builder
        only consulted that field. The reporter caught it when ``opencode
        models`` run inside a Hermes terminal enumerated the entire Bedrock
        catalog off the leaked bearer token.
        """
        result_env = _run_with_env(extra_os_env={
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-bearer-secret",
        })

        assert "AWS_BEARER_TOKEN_BEDROCK" not in result_env, (
            "AWS_BEARER_TOKEN_BEDROCK leaked into subprocess env (see #32314)"
        )

    def test_general_aws_credential_chain_is_preserved(self):
        """The GENERAL AWS credential chain must STILL pass through to
        subprocesses — this is the no-regression guard for #32314.

        Per SECURITY.md §3.2 the local terminal is the user's trusted operator
        shell. A user running ``aws``/``terraform``/``cdk``/``boto3`` in the
        agent terminal must keep the same AWS access their own shell has.
        Stripping these would (a) break every user who does AWS work in the
        agent terminal — not just Bedrock users, since the registry is iterated
        unconditionally — and (b) be unrecoverable, because env_passthrough.py
        refuses to re-allow anything in _HERMES_PROVIDER_ENV_BLOCKLIST
        (GHSA-rhgp-j443-p4rf). Only the Bedrock inference bearer token is
        Hermes-managed; the rest belongs to the user.
        """
        general_chain = {
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "AWS_SESSION_TOKEN": "session-token",
            "AWS_PROFILE": "production",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_REGION": "us-east-1",
            "AWS_SHARED_CREDENTIALS_FILE": "/home/user/.aws/credentials",
            "AWS_CONFIG_FILE": "/home/user/.aws/config",
            "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/secrets/token",
            "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/example",
        }
        result_env = _run_with_env(extra_os_env=general_chain)

        for var, value in general_chain.items():
            assert result_env.get(var) == value, (
                f"{var} was stripped from subprocess env — this is a "
                f"capability regression (see #32314 discussion)"
            )

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
            "HERMES_DASHBOARD_SESSION_TOKEN": "dashboard-session-secret",
            "BROWSERBASE_PROJECT_ID": "bb-project",
            "ELEVENLABS_API_KEY": "el-secret",
            "GITHUB_TOKEN": "ghp_secret",
            "GH_TOKEN": "gh_alias_secret",
            "GATEWAY_ALLOW_ALL_USERS": "true",
            "GATEWAY_ALLOWED_USERS": "alice,bob",
            "MODAL_TOKEN_ID": "modal-id",
            "MODAL_TOKEN_SECRET": "modal-secret",
            "DAYTONA_API_KEY": "daytona-key",
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

    def test_bedrock_bearer_token_is_in_blocklist(self):
        """auth_type='aws_sdk' providers contribute their Hermes-managed
        inference token (the Bedrock bearer) to the blocklist, keyed off
        auth_type so any future SDK-cred provider is covered automatically."""
        assert "AWS_BEARER_TOKEN_BEDROCK" in _HERMES_PROVIDER_ENV_BLOCKLIST

    def test_general_aws_chain_not_in_blocklist(self):
        """The general AWS credential chain must NOT be in the blocklist —
        no-regression guard for #32314. These belong to the user's trusted
        operator shell (SECURITY.md §3.2), not to Hermes, and blocklisting
        them would be unrecoverable via env_passthrough (GHSA-rhgp-j443-p4rf).
        """
        general_chain = {
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_DEFAULT_REGION",
            "AWS_REGION",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONFIG_FILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_ROLE_ARN",
        }
        leaked_block = general_chain & _HERMES_PROVIDER_ENV_BLOCKLIST
        assert not leaked_block, (
            f"General AWS chain vars must stay inheritable, but these are "
            f"blocklisted: {sorted(leaked_block)} (capability regression, #32314)"
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
            "HERMES_DASHBOARD_SESSION_TOKEN",
            "GATEWAY_ALLOWED_USERS",
            "GH_TOKEN",
            "GITHUB_APP_ID",
            "GITHUB_APP_PRIVATE_KEY_PATH",
            "GITHUB_APP_INSTALLATION_ID",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "DAYTONA_API_KEY",
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
        """When PATH is minimal, _make_run_env appends missing sane entries."""
        from tools.environments.local import _SANE_PATH, _make_run_env
        minimal_env = {"PATH": "/some/custom/bin"}
        with patch.dict(os.environ, minimal_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert path_entries[0] == "/some/custom/bin"
        for entry in _SANE_PATH.split(":"):
            assert entry in path_entries

    def test_make_run_env_fills_missing_homebrew_when_usr_bin_present(self):
        """macOS launchd PATH can include /usr/bin while missing Homebrew."""
        from tools.environments.local import _make_run_env
        launchd_env = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
        with patch.dict(os.environ, launchd_env, clear=True):
            result = _make_run_env({})

        path_entries = result["PATH"].split(":")
        assert "/opt/homebrew/bin" in path_entries
        assert "/opt/homebrew/sbin" in path_entries

    def test_make_run_env_does_not_duplicate_existing_sane_entries(self):
        from tools.environments.local import _make_run_env
        existing_env = {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}
        with patch.dict(os.environ, existing_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert path_entries.count("/opt/homebrew/bin") == 1
        assert path_entries.count("/usr/local/bin") == 1
        assert path_entries.count("/usr/bin") == 1

    def test_make_run_env_real_launchd_path_gains_homebrew(self):
        """The literal macOS launchd PATH is the production trigger for #35613."""
        from tools.environments.local import _make_run_env
        launchd_env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
        with patch.dict(os.environ, launchd_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert "/opt/homebrew/bin" in path_entries
        assert "/opt/homebrew/sbin" in path_entries
        # Original entries keep their leading precedence.
        assert path_entries[:4] == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]

    def test_make_run_env_collapses_duplicate_caller_entries(self):
        """Duplicates already present in the caller PATH are de-duplicated."""
        from tools.environments.local import _make_run_env
        dup_env = {"PATH": "/usr/bin:/usr/bin:/custom/bin:/custom/bin:/bin"}
        with patch.dict(os.environ, dup_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert path_entries.count("/usr/bin") == 1
        assert path_entries.count("/custom/bin") == 1
        # First-occurrence order is preserved for the caller entries.
        assert path_entries[:3] == ["/usr/bin", "/custom/bin", "/bin"]

    def test_make_run_env_strips_empty_path_entries(self):
        """Leading/trailing/double colons (== CWD on POSIX) are dropped."""
        from tools.environments.local import _make_run_env
        empty_env = {"PATH": "/usr/bin::/bin:"}
        with patch.dict(os.environ, empty_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(":")
        assert "" not in path_entries
        assert "/usr/bin" in path_entries
        assert "/opt/homebrew/bin" in path_entries

    def test_make_run_env_leaves_windows_path_unchanged(self, monkeypatch):
        from tools.environments import local as local_mod
        from tools.environments.local import _make_run_env
        windows_env = {"PATH": r"C:\Windows\System32;C:\Program Files\Git\bin"}
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        with patch.dict(os.environ, windows_env, clear=True):
            result = _make_run_env({})
        assert result["PATH"] == windows_env["PATH"]

    def test_make_run_env_preserves_windows_mixed_case_path_key(self, monkeypatch):
        from tools.environments import local as local_mod
        from tools.environments.local import _make_run_env
        windows_env = {"Path": r"C:\Windows\System32;C:\Program Files\Git\bin"}
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        with patch.object(local_mod.os, "environ", windows_env):
            result = _make_run_env({})
        assert result["Path"] == windows_env["Path"]
        assert "PATH" not in result

class TestSessionEnvBridge:
    @staticmethod
    def _reset_session_context_vars():
        from gateway.session_context import _UNSET, _VAR_MAP

        saved = {var: var.get() for var in _VAR_MAP.values()}
        for var in _VAR_MAP.values():
            var.set(_UNSET)
        return saved

    @staticmethod
    def _restore_session_context_vars(saved):
        for var, value in saved.items():
            var.set(value)

    def test_make_run_env_explicit_empty_context_clears_stale_process_env(self):
        """Explicit empty ContextVars must not inherit stale process env values."""
        from gateway.session_context import (
            _UNSET,
            _VAR_MAP,
            clear_session_vars,
            set_session_vars,
        )
        from tools.environments.local import _make_run_env

        stale_env = {
            "PATH": "/usr/bin:/bin",
            "HERMES_SESSION_PLATFORM": "mattermost",
            "HERMES_SESSION_CHAT_ID": "ekum_fantasy_channel",
            "HERMES_SESSION_THREAD_ID": "ekum_fantasy_thread",
            "HERMES_SESSION_MESSAGE_ID": "ekum_fantasy_post",
        }
        tokens = set_session_vars(
            platform="mattermost",
            chat_id="trading_channel",
            thread_id="",
            session_key="mattermost:trading",
            message_id="trading_top_post",
        )
        try:
            with patch.dict(os.environ, stale_env, clear=True):
                result = _make_run_env({})
        finally:
            clear_session_vars(tokens)
            for var in _VAR_MAP.values():
                var.set(_UNSET)

        assert result["HERMES_SESSION_PLATFORM"] == "mattermost"
        assert result["HERMES_SESSION_CHAT_ID"] == "trading_channel"
        assert result["HERMES_SESSION_THREAD_ID"] == ""
        assert result["HERMES_SESSION_MESSAGE_ID"] == "trading_top_post"
        assert result["HERMES_SESSION_KEY"] == "mattermost:trading"

    def test_make_run_env_gateway_unset_context_strips_foreign_session_env(self):
        """Foreground env must not leak a foreign process-global session under the gateway."""
        from tools.environments.local import _make_run_env

        saved = self._reset_session_context_vars()
        try:
            stale_env = {
                "PATH": "/usr/bin:/bin",
                "_HERMES_GATEWAY": "1",
                "HERMES_SESSION_KEY": "agent:main:discord:thread:FOREIGN:X",
                "HERMES_SESSION_THREAD_ID": "FOREIGN_THREAD",
            }
            with patch.dict(os.environ, stale_env, clear=True):
                result = _make_run_env({})
        finally:
            self._restore_session_context_vars(saved)

        assert "HERMES_SESSION_KEY" not in result
        assert "HERMES_SESSION_THREAD_ID" not in result

    def test_make_run_env_gateway_partial_unset_strips_only_unset_session_vars(self):
        """A partially-bound gateway context keeps bound values and strips only _UNSET vars."""
        from gateway.session_context import _UNSET, _VAR_MAP, clear_session_vars, set_session_vars
        from tools.environments.local import _make_run_env

        tokens = set_session_vars(
            platform="discord",
            chat_id="CORRECT_CHAT",
            thread_id="CORRECT_THREAD",
            session_key="agent:main:discord:group:CORRECT_CHAT:1",
        )
        try:
            _VAR_MAP["HERMES_SESSION_THREAD_ID"].set(_UNSET)
            with patch.dict(os.environ, {
                "PATH": "/usr/bin:/bin",
                "_HERMES_GATEWAY": "1",
                "HERMES_SESSION_CHAT_ID": "FOREIGN_CHAT",
                "HERMES_SESSION_THREAD_ID": "FOREIGN_THREAD",
                "HERMES_SESSION_KEY": "agent:main:discord:thread:FOREIGN:X",
            }, clear=True):
                result = _make_run_env({})
        finally:
            clear_session_vars(tokens)
            for var in _VAR_MAP.values():
                var.set(_UNSET)

        assert result["HERMES_SESSION_CHAT_ID"] == "CORRECT_CHAT"
        assert result["HERMES_SESSION_KEY"] == "agent:main:discord:group:CORRECT_CHAT:1"
        assert "HERMES_SESSION_THREAD_ID" not in result

    def test_make_run_env_thread_without_copied_context_does_not_inherit_foreign_gateway_env(self):
        """Regression for fresh worker threads whose ContextVars are _UNSET."""
        from gateway.session_context import _UNSET, _VAR_MAP, clear_session_vars, set_session_vars
        from tools.environments.local import _make_run_env

        observed = []
        tokens = set_session_vars(
            platform="discord",
            chat_id="CORRECT_CHAT",
            session_key="agent:main:discord:group:CORRECT_CHAT:1",
        )
        try:
            with patch.dict(os.environ, {
                "PATH": "/usr/bin:/bin",
                "_HERMES_GATEWAY": "1",
                "HERMES_SESSION_KEY": "agent:main:discord:thread:FOREIGN:X",
                "HERMES_SESSION_THREAD_ID": "FOREIGN_THREAD",
            }, clear=True):
                def worker():
                    observed.append(_make_run_env({}))

                t = threading.Thread(target=worker)
                t.start()
                t.join(timeout=5)
                assert not t.is_alive()
        finally:
            clear_session_vars(tokens)
            for var in _VAR_MAP.values():
                var.set(_UNSET)

        assert len(observed) == 1
        assert "HERMES_SESSION_KEY" not in observed[0]
        assert "HERMES_SESSION_THREAD_ID" not in observed[0]

    def test_make_run_env_unset_context_preserves_cli_env_fallback(self):
        """Outside the gateway, CLI/cron still inherit HERMES_SESSION_* from os.environ."""
        from tools.environments.local import _make_run_env

        saved = self._reset_session_context_vars()
        try:
            cli_env = {
                "PATH": "/usr/bin:/bin",
                "HERMES_SESSION_KEY": "cli:session:kept",
            }
            with patch.dict(os.environ, cli_env, clear=True):
                result = _make_run_env({})
        finally:
            self._restore_session_context_vars(saved)

        assert result["HERMES_SESSION_KEY"] == "cli:session:kept"

    def test_sanitize_subprocess_env_explicit_empty_context_clears_stale_env(self):
        """Background subprocess env must use the same session overlay as foreground."""
        from gateway.session_context import (
            _UNSET,
            _VAR_MAP,
            clear_session_vars,
            set_session_vars,
        )
        from tools.environments.local import _sanitize_subprocess_env

        stale_env = {
            "PATH": "/usr/bin:/bin",
            "HERMES_SESSION_PLATFORM": "mattermost",
            "HERMES_SESSION_CHAT_ID": "ekum_fantasy_channel",
            "HERMES_SESSION_THREAD_ID": "ekum_fantasy_thread",
            "HERMES_SESSION_MESSAGE_ID": "ekum_fantasy_post",
        }
        tokens = set_session_vars(
            platform="mattermost",
            chat_id="trading_channel",
            thread_id="",
            session_key="mattermost:trading",
            message_id="trading_top_post",
        )
        try:
            result = _sanitize_subprocess_env(stale_env)
        finally:
            clear_session_vars(tokens)
            for var in _VAR_MAP.values():
                var.set(_UNSET)

        assert result["HERMES_SESSION_PLATFORM"] == "mattermost"
        assert result["HERMES_SESSION_CHAT_ID"] == "trading_channel"
        assert result["HERMES_SESSION_THREAD_ID"] == ""
        assert result["HERMES_SESSION_MESSAGE_ID"] == "trading_top_post"
        assert result["HERMES_SESSION_KEY"] == "mattermost:trading"

    def test_sanitize_subprocess_env_gateway_unset_context_strips_foreign_session_env(self):
        """Background env must use the same gateway _UNSET strip policy as foreground."""
        from tools.environments.local import _sanitize_subprocess_env

        saved = self._reset_session_context_vars()
        try:
            with patch.dict(os.environ, {"_HERMES_GATEWAY": "1"}, clear=True):
                result = _sanitize_subprocess_env({
                    "PATH": "/usr/bin:/bin",
                    "HERMES_SESSION_KEY": "agent:main:discord:thread:FOREIGN:X",
                    "HERMES_SESSION_THREAD_ID": "FOREIGN_THREAD",
                })
        finally:
            self._restore_session_context_vars(saved)

        assert "HERMES_SESSION_KEY" not in result
        assert "HERMES_SESSION_THREAD_ID" not in result

    def test_sanitize_subprocess_env_unset_context_preserves_cli_env_fallback(self):
        """Background CLI/cron subprocesses keep explicit process env fallback values."""
        from tools.environments.local import _sanitize_subprocess_env

        saved = self._reset_session_context_vars()
        try:
            with patch.dict(os.environ, {}, clear=True):
                result = _sanitize_subprocess_env({
                    "PATH": "/usr/bin:/bin",
                    "HERMES_SESSION_KEY": "cli:session:kept",
                })
        finally:
            self._restore_session_context_vars(saved)

        assert result["HERMES_SESSION_KEY"] == "cli:session:kept"
