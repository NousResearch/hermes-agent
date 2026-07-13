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

import pytest

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

    def test_vertex_credentials_path_is_stripped(self):
        """The Vertex AI service-account JSON path must not leak into
        subprocesses, even though it is filesystem path metadata rather
        than a bare API key.

        Regression: ``vertex`` authenticates via OAuth2 (service-account
        JSON / ADC), not PROVIDER_REGISTRY, and OPTIONAL_ENV_VARS marks
        VERTEX_CREDENTIALS_PATH as ``password=False`` (it's a path, not a
        secret string) with ``category="provider"`` — a category the
        registry-derived loop above never checks — so it fell through both
        blocklist sources. GOOGLE_APPLICATION_CREDENTIALS (the ADC fallback
        the adapter also reads) had the same gap. A leaked path discloses
        the on-disk location of a GCP service-account key to every spawned
        subprocess (terminal, codex/copilot app-server, browser workers).
        """
        result_env = _run_with_env(extra_os_env={
            "VERTEX_CREDENTIALS_PATH": "/home/user/.config/gcloud/sa-key.json",
            "GOOGLE_APPLICATION_CREDENTIALS": "/home/user/.config/gcloud/adc.json",
        })

        assert "VERTEX_CREDENTIALS_PATH" not in result_env
        assert "GOOGLE_APPLICATION_CREDENTIALS" not in result_env

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
            "TENKI_AUTH_TOKEN": "tenki-token",
            "TENKI_API_KEY": "tenki-key",
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


class TestActiveVenvMarkerStripping:
    """Active-virtualenv markers must not leak into terminal subprocesses (#23473).

    The gateway runs inside its own venv, so its process environment carries
    VIRTUAL_ENV (and possibly CONDA_PREFIX). If those leak into commands the
    agent runs against ANOTHER Python project, ``uv``/``poetry`` treat the
    inherited value as the active environment and build that project's deps
    into the Hermes venv path instead of the project's own ``.venv`` —
    silently clobbering the Hermes environment (and, when the other project
    pins a different Python, breaking the gateway outright). The Hermes venv
    stays reachable via PATH, so stripping the markers is safe.
    """

    def test_virtualenv_marker_stripped_end_to_end(self):
        result_env = _run_with_env(extra_os_env={
            "VIRTUAL_ENV": "/home/user/.hermes/hermes-agent/venv",
        })
        assert "VIRTUAL_ENV" not in result_env

    def test_conda_prefix_marker_stripped_end_to_end(self):
        result_env = _run_with_env(extra_os_env={
            "CONDA_PREFIX": "/opt/conda/envs/hermes",
        })
        assert "CONDA_PREFIX" not in result_env

    def test_make_run_env_strips_markers(self):
        from tools.environments.local import _make_run_env
        poison = {"VIRTUAL_ENV": "/venv", "CONDA_PREFIX": "/conda", "PATH": "/usr/bin"}
        with patch.dict(os.environ, poison, clear=True):
            result = _make_run_env({})
        assert "VIRTUAL_ENV" not in result
        assert "CONDA_PREFIX" not in result

    def test_sanitize_subprocess_env_strips_markers(self):
        from tools.environments.local import _sanitize_subprocess_env
        base = {"VIRTUAL_ENV": "/venv", "CONDA_PREFIX": "/conda", "HOME": "/home/user"}
        # Even an explicitly-passed extra marker is stripped.
        result = _sanitize_subprocess_env(base, {"VIRTUAL_ENV": "/also/venv"})
        assert "VIRTUAL_ENV" not in result
        assert "CONDA_PREFIX" not in result
        assert result.get("HOME") == "/home/user"

    def test_markers_constant_contents(self):
        from tools.environments.local import _ACTIVE_VENV_MARKER_VARS
        assert "VIRTUAL_ENV" in _ACTIVE_VENV_MARKER_VARS
        assert "CONDA_PREFIX" in _ACTIVE_VENV_MARKER_VARS


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
        must appear in the blocklist — ensures no drift.

        CLAUDE_CODE_OAUTH_TOKEN is the one deliberate exemption: it is owned
        by the user's Claude Code install, not Hermes (#55878).
        """
        from hermes_cli.auth import PROVIDER_REGISTRY

        exempt = {"CLAUDE_CODE_OAUTH_TOKEN"}
        for pconfig in PROVIDER_REGISTRY.values():
            for var in pconfig.api_key_env_vars:
                if var in exempt:
                    continue
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
        """Non-registry auth vars (ANTHROPIC_TOKEN) must also be in the
        blocklist."""
        extras = {"ANTHROPIC_TOKEN"}
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)

    def test_claude_code_oauth_token_is_inheritable(self):
        """CLAUDE_CODE_OAUTH_TOKEN is owned by the user's Claude Code install
        (subscription OAuth), not a Hermes inference credential. Stripping it
        made agent-spawned ``claude`` fall through to the shared Keychain /
        ~/.claude credential store and clobber the user's interactive login
        on auth failure (#55878). It must stay inheritable."""
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in _HERMES_PROVIDER_ENV_BLOCKLIST

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
            "TENKI_AUTH_TOKEN",
            "TENKI_API_KEY",
        }
        assert extras.issubset(_HERMES_PROVIDER_ENV_BLOCKLIST)


class TestSanePathIncludesHomebrew:
    """Verify _SANE_PATH includes macOS Homebrew directories."""

    @pytest.fixture(autouse=True)
    def _disable_hermes_bin_injection(self):
        """These tests assert the sane-path merge in isolation. Disable the
        hermes-install-dir prepend (a separate concern, covered by
        TestHermesBinDirOnPath) so a real ``hermes`` on the test runner's PATH
        doesn't shift the asserted PATH layout."""
        from tools.environments import local as local_mod
        saved = local_mod._HERMES_BIN_DIR
        local_mod._HERMES_BIN_DIR = None  # resolved -> no dir to inject
        yield
        local_mod._HERMES_BIN_DIR = saved

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


class TestHermesBinDirOnPath:
    """The hermes install dir is reachable in the terminal subshell PATH.

    Plugins shelling out to bare ``hermes`` via the terminal tool must work
    even when the gateway was launched without the hermes install dir on
    PATH (systemd, service managers, cron). See the discussion that motivated
    _resolve_hermes_bin_dir / _prepend_hermes_bin_dir.
    """

    def _reset_cache(self):
        from tools.environments import local as local_mod
        local_mod._HERMES_BIN_DIR = local_mod._SENTINEL

    def test_resolves_via_which(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        monkeypatch.setattr(local_mod.shutil, "which",
                            lambda name: "/opt/hermes/bin/hermes" if name == "hermes" else None)
        monkeypatch.setattr(local_mod.os.path, "isdir", lambda p: p == "/opt/hermes/bin")
        assert local_mod._resolve_hermes_bin_dir() == "/opt/hermes/bin"

    def test_resolves_via_sys_executable_dir(self, monkeypatch, tmp_path):
        from tools.environments import local as local_mod
        self._reset_cache()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "hermes").write_text("#!/bin/sh\n")
        monkeypatch.setattr(local_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(local_mod.sys, "argv", ["python"])
        monkeypatch.setattr(local_mod.sys, "executable", str(venv_bin / "python"))
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        assert local_mod._resolve_hermes_bin_dir() == str(venv_bin)

    def test_returns_none_when_unresolvable(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        monkeypatch.setattr(local_mod.shutil, "which", lambda name: None)
        monkeypatch.setattr(local_mod.sys, "argv", ["python"])
        monkeypatch.setattr(local_mod.sys, "executable", "/nonexistent/python")
        assert local_mod._resolve_hermes_bin_dir() is None

    def test_prepend_adds_missing_dir_at_front(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = "/opt/hermes/bin"
        out = local_mod._prepend_hermes_bin_dir("/usr/bin:/bin")
        assert out.split(os.pathsep)[0] == "/opt/hermes/bin"
        assert "/usr/bin" in out.split(os.pathsep)

    def test_prepend_is_idempotent(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = "/opt/hermes/bin"
        once = local_mod._prepend_hermes_bin_dir("/usr/bin:/bin")
        twice = local_mod._prepend_hermes_bin_dir(once)
        assert twice == once
        assert once.split(os.pathsep).count("/opt/hermes/bin") == 1

    def test_prepend_noop_when_unresolved(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = None
        assert local_mod._prepend_hermes_bin_dir("/usr/bin:/bin") == "/usr/bin:/bin"

    def test_make_run_env_injects_hermes_bin_dir(self, monkeypatch):
        """A gateway env missing the hermes dir gets it back in the subshell PATH."""
        from tools.environments import local as local_mod
        from tools.environments.local import _make_run_env
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = "/opt/hermes/bin"
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=True):
            result = _make_run_env({})
        entries = result["PATH"].split(os.pathsep)
        assert entries[0] == "/opt/hermes/bin"
        assert "/usr/bin" in entries


class TestHermesInternalDynamicSecrets:
    """Dynamically-named Hermes secrets injected at gateway/CLI startup must
    not leak into terminal subprocesses.

    The static ``_HERMES_PROVIDER_ENV_BLOCKLIST`` is name-based and derived
    from provider/tool registries, so it cannot enumerate:

    - ``AUXILIARY_<TASK>_API_KEY`` / ``AUXILIARY_<TASK>_BASE_URL`` — per-task
      side-LLM credentials bridged from ``config.yaml[auxiliary]`` by
      ``gateway/run.py`` and ``cli.py``.
    - ``GATEWAY_RELAY_*_SECRET`` / ``_KEY`` / ``_TOKEN`` — relay-auth material
      provisioned by ``gateway/relay``.

    ``_is_hermes_internal_secret`` is the single source of truth; every spawn
    path (``_sanitize_subprocess_env``, ``_make_run_env``,
    ``hermes_subprocess_env``, Docker forward filter, ``env_passthrough``)
    consults it. These tests exercise the terminal execute path + predicate.
    """

    def test_predicate_matches_auxiliary_api_key(self):
        from tools.environments.local import _is_hermes_internal_secret
        assert _is_hermes_internal_secret("AUXILIARY_VISION_API_KEY")
        assert _is_hermes_internal_secret("AUXILIARY_WEB_EXTRACT_API_KEY")
        assert _is_hermes_internal_secret("AUXILIARY_APPROVAL_API_KEY")
        # plugin-registered task names are covered by the pattern
        assert _is_hermes_internal_secret("AUXILIARY_MY_PLUGIN_TASK_API_KEY")

    def test_predicate_matches_auxiliary_base_url(self):
        from tools.environments.local import _is_hermes_internal_secret
        assert _is_hermes_internal_secret("AUXILIARY_VISION_BASE_URL")
        assert _is_hermes_internal_secret("AUXILIARY_COMPRESSION_BASE_URL")

    def test_predicate_matches_gateway_relay_auth(self):
        from tools.environments.local import _is_hermes_internal_secret
        assert _is_hermes_internal_secret("GATEWAY_RELAY_SECRET")
        assert _is_hermes_internal_secret("GATEWAY_RELAY_DELIVERY_KEY")
        assert _is_hermes_internal_secret("GATEWAY_RELAY_SESSION_TOKEN")

    def test_predicate_allows_auxiliary_non_secrets(self):
        """AUXILIARY_*_PROVIDER / _MODEL and GATEWAY_RELAY_* routing hints are
        NOT secrets and must remain visible so tooling that reads them works."""
        from tools.environments.local import _is_hermes_internal_secret
        assert not _is_hermes_internal_secret("AUXILIARY_VISION_PROVIDER")
        assert not _is_hermes_internal_secret("AUXILIARY_VISION_MODEL")
        assert not _is_hermes_internal_secret("GATEWAY_RELAY_URL")
        assert not _is_hermes_internal_secret("GATEWAY_RELAY_PLATFORMS")
        assert not _is_hermes_internal_secret("GATEWAY_RELAY_ID")  # not a secret suffix
        # unrelated vars pass through
        assert not _is_hermes_internal_secret("PATH")
        assert not _is_hermes_internal_secret("MY_APP_KEY")

    def test_auxiliary_secrets_stripped_from_subprocess(self):
        """AUXILIARY_*_API_KEY / _BASE_URL injected into os.environ must not
        reach the terminal subprocess, while _PROVIDER / _MODEL survive."""
        result_env = _run_with_env(extra_os_env={
            "AUXILIARY_VISION_API_KEY": "sk-vision-secret",
            "AUXILIARY_VISION_BASE_URL": "http://internal:1234/v1",
            "AUXILIARY_WEB_EXTRACT_API_KEY": "sk-webx-secret",
            "AUXILIARY_VISION_PROVIDER": "openai",
            "AUXILIARY_VISION_MODEL": "gpt-4o",
        })
        assert "AUXILIARY_VISION_API_KEY" not in result_env
        assert "AUXILIARY_VISION_BASE_URL" not in result_env
        assert "AUXILIARY_WEB_EXTRACT_API_KEY" not in result_env
        # Non-secret routing config is preserved.
        assert result_env.get("AUXILIARY_VISION_PROVIDER") == "openai"
        assert result_env.get("AUXILIARY_VISION_MODEL") == "gpt-4o"

    def test_gateway_relay_secret_stripped_from_subprocess(self):
        result_env = _run_with_env(extra_os_env={
            "GATEWAY_RELAY_SECRET": "relay-signing-secret",
            "GATEWAY_RELAY_DELIVERY_KEY": "relay-delivery-key",
            "GATEWAY_RELAY_URL": "https://relay.example.com",
        })
        assert "GATEWAY_RELAY_SECRET" not in result_env
        assert "GATEWAY_RELAY_DELIVERY_KEY" not in result_env
        # Non-secret routing hint stays visible.
        assert result_env.get("GATEWAY_RELAY_URL") == "https://relay.example.com"

    def test_auxiliary_secret_stripped_even_when_passthrough_registered(self):
        """A skill registering AUXILIARY_*_API_KEY as env_passthrough must NOT
        be able to tunnel it into a subprocess — the strip is unconditional."""
        with patch(
            "tools.env_passthrough.is_env_passthrough",
            side_effect=lambda name: name == "AUXILIARY_VISION_API_KEY",
        ):
            result_env = _run_with_env(extra_os_env={
                "AUXILIARY_VISION_API_KEY": "sk-vision-secret",
            })
        assert "AUXILIARY_VISION_API_KEY" not in result_env

    def test_make_run_env_strips_internal_secrets(self):
        """The foreground _make_run_env path strips the same dynamic secrets."""
        from tools.environments.local import _make_run_env
        with patch.dict(os.environ, {
            "PATH": "/usr/bin:/bin",
            "AUXILIARY_VISION_API_KEY": "sk-secret",
            "GATEWAY_RELAY_SECRET": "relay-secret",
            "AUXILIARY_VISION_PROVIDER": "openai",
        }, clear=True):
            run_env = _make_run_env({})
        assert "AUXILIARY_VISION_API_KEY" not in run_env
        assert "GATEWAY_RELAY_SECRET" not in run_env
        assert run_env.get("AUXILIARY_VISION_PROVIDER") == "openai"

    def test_gateway_relay_static_names_in_blocklist(self):
        """The static relay names are also added to the name-based blocklist so
        the exact-match path catches them independently of the predicate."""
        assert "GATEWAY_RELAY_SECRET" in _HERMES_PROVIDER_ENV_BLOCKLIST
        assert "GATEWAY_RELAY_DELIVERY_KEY" in _HERMES_PROVIDER_ENV_BLOCKLIST
        assert "GATEWAY_RELAY_ID" in _HERMES_PROVIDER_ENV_BLOCKLIST
