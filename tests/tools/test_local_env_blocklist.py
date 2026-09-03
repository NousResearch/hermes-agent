"""Tests for subprocess env sanitization in LocalEnvironment.

Verifies that Hermes-managed provider, tool, and gateway env vars are
stripped from subprocess environments so external CLIs are not silently
misrouted or handed Hermes secrets.

See: https://github.com/NousResearch/hermes-agent/issues/1002
See: https://github.com/NousResearch/hermes-agent/issues/1264
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.local import (
    LocalEnvironment,
    _HERMES_PROVIDER_ENV_BLOCKLIST,
    _HERMES_PROVIDER_ENV_FORCE_PREFIX,
)


def _running_venv_site_packages() -> Path:
    """Independently construct the host-native venv site-packages path."""
    if sys.platform == "win32":
        return Path(sys.prefix) / "Lib" / "site-packages"
    pyver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    return Path(sys.prefix) / "lib" / pyver / "site-packages"


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

    def test_bare_hermes_resolves_from_sanitized_subprocess_path(self):
        """Cron children can resolve Hermes even when the gateway PATH cannot."""
        from tools.environments.local import _sanitize_subprocess_env

        with patch(
            "tools.environments.local._resolve_hermes_bin_dir",
            return_value="/home/user/.local/bin",
        ):
            result = _sanitize_subprocess_env(
                {
                    "PATH": os.pathsep.join(["/usr/bin", "/bin"]),
                    "HOME": "/home/user",
                }
            )

        assert result["PATH"] == os.pathsep.join(
            ["/home/user/.local/bin", "/usr/bin", "/bin"]
        )

    def test_bare_hermes_path_does_not_duplicate_existing_install_dir(self):
        """The PATH repair is idempotent for already-correct environments."""
        from tools.environments.local import _sanitize_subprocess_env

        with patch(
            "tools.environments.local._resolve_hermes_bin_dir",
            return_value="/home/user/.local/bin",
        ):
            result = _sanitize_subprocess_env(
                {"PATH": os.pathsep.join(["/home/user/.local/bin", "/usr/bin"])}
            )

        assert result["PATH"] == os.pathsep.join(
            ["/home/user/.local/bin", "/usr/bin"]
        )

    def test_self_env_blocked_vars_also_stripped(self):
        """Blocked vars in self.env are stripped; non-blocked vars pass through."""
        result_env = _run_with_env(self_env={
            "OPENAI_BASE_URL": "http://custom:9999/v1",
            "MY_CUSTOM_VAR": "keep-this",
        })

        assert "OPENAI_BASE_URL" not in result_env
        assert "MY_CUSTOM_VAR" in result_env
        assert result_env["MY_CUSTOM_VAR"] == "keep-this"


class TestTerminalFirstPartyPlatformEnv:
    """BUZZ_* first-party platform credentials must reach terminal children —
    but ONLY in a Buzz agent context.

    Issue #78026: Buzz platform agents could not use the ``buzz`` CLI from the
    terminal tool because BUZZ_PRIVATE_KEY / BUZZ_AUTH_TAG / BUZZ_RELAY_URL
    (and the other BUZZ_* vars) are stripped by _HERMES_PROVIDER_ENV_BLOCKLIST
    and env_passthrough refuses to re-allow them (GHSA-rhgp-j443-p4rf).

    The carve-out is TERMINAL-ONLY and CONTEXT-GATED: it applies when the
    process is a Buzz-ACP managed agent (BUZZ_MANAGED_AGENT set by the
    buzz-acp harness, #76243) or the live session's platform is ``buzz``.
    Foreground (_make_run_env) and background/PTY (_sanitize_subprocess_env)
    children then get the BUZZ_* vars; execute_code, hermes_subprocess_env,
    docker, and env_passthrough registration stay sealed, and non-Buzz
    sessions/processes keep stripping the vars. The blocklist itself is NOT
    modified.
    """

    def test_make_run_env_preserves_buzz_vars(self):
        """Foreground terminal children get the BUZZ_* credentials when the
        process is a Buzz-managed agent (BUZZ_MANAGED_AGENT set)."""
        from tools.environments.local import _make_run_env

        buzz_vars = {
            "BUZZ_PRIVATE_KEY": "nsec1faketestkey",
            "BUZZ_AUTH_TAG": '["tag","data","kind","sig"]',
            "BUZZ_RELAY_URL": "https://mycommunity.communities.buzz.xyz",
        }
        with patch.dict(
            os.environ,
            {**buzz_vars, "BUZZ_MANAGED_AGENT": "1", "PATH": "/usr/bin:/bin"},
            clear=True,
        ):
            run_env = _make_run_env({})

        for var, value in buzz_vars.items():
            assert run_env.get(var) == value, (
                f"{var} missing from foreground terminal env (issue #78026)"
            )

    def test_sanitize_subprocess_env_preserves_buzz_vars(self, monkeypatch):
        """Background/PTY terminal children get the BUZZ_* credentials when
        the process is a Buzz-managed agent."""
        from tools.environments.local import _sanitize_subprocess_env

        monkeypatch.setenv("BUZZ_MANAGED_AGENT", "1")
        buzz_vars = {
            "BUZZ_PRIVATE_KEY": "nsec1faketestkey",
            "BUZZ_AUTH_TAG": '["tag","data","kind","sig"]',
            "BUZZ_RELAY_URL": "https://mycommunity.communities.buzz.xyz",
        }
        result = _sanitize_subprocess_env({**buzz_vars, "HOME": "/home/user"})

        for var, value in buzz_vars.items():
            assert result.get(var) == value, (
                f"{var} missing from background/PTY terminal env (issue #78026)"
            )

    def test_buzz_vars_stripped_without_buzz_context(self, monkeypatch):
        """NEGATIVE gate: with no Buzz context signal (no BUZZ_MANAGED_AGENT,
        session platform not buzz), the BUZZ_* credentials stay stripped from
        BOTH terminal scrub paths — a Telegram/CLI/cron session on a host that
        also runs a Buzz gateway must not see BUZZ_PRIVATE_KEY."""
        from gateway.session_context import _SESSION_PLATFORM
        from tools.environments.local import _make_run_env, _sanitize_subprocess_env

        monkeypatch.delenv("BUZZ_MANAGED_AGENT", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        buzz_vars = {
            "BUZZ_PRIVATE_KEY": "nsec1faketestkey",
            "BUZZ_AUTH_TAG": '["tag","data","kind","sig"]',
            "BUZZ_RELAY_URL": "https://mycommunity.communities.buzz.xyz",
        }
        for var, value in buzz_vars.items():
            monkeypatch.setenv(var, value)
        # Bind a non-buzz session platform (ContextVar-authoritative).
        token = _SESSION_PLATFORM.set("telegram")
        try:
            run_env = _make_run_env({})
            sanitized = _sanitize_subprocess_env({**buzz_vars, "HOME": "/home/user"})
        finally:
            _SESSION_PLATFORM.reset(token)

        for var in buzz_vars:
            assert var not in run_env, f"{var} leaked into non-Buzz foreground env"
            assert var not in sanitized, f"{var} leaked into non-Buzz background env"

    def test_session_platform_buzz_enables_carveout(self, monkeypatch):
        """A live gateway session whose platform is ``buzz`` gets the
        carve-out even without BUZZ_MANAGED_AGENT (native buzz gateway
        plugin path), via the concurrency-safe session ContextVar."""
        from gateway.session_context import _SESSION_PLATFORM
        from tools.environments.local import _make_run_env, _sanitize_subprocess_env

        monkeypatch.delenv("BUZZ_MANAGED_AGENT", raising=False)
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1faketestkey")
        token = _SESSION_PLATFORM.set("buzz")
        try:
            run_env = _make_run_env({})
            sanitized = _sanitize_subprocess_env(
                {"BUZZ_PRIVATE_KEY": "nsec1faketestkey", "HOME": "/home/user"}
            )
        finally:
            _SESSION_PLATFORM.reset(token)

        assert run_env.get("BUZZ_PRIVATE_KEY") == "nsec1faketestkey"
        assert sanitized.get("BUZZ_PRIVATE_KEY") == "nsec1faketestkey"

    def test_buzz_vars_stay_in_blocklist(self):
        """The carve-out is a scrub-path exemption, NOT a blocklist removal —
        BUZZ_* must remain blocked for every non-terminal surface (execute_code,
        hermes_subprocess_env, env_passthrough registration)."""
        assert {"BUZZ_PRIVATE_KEY", "BUZZ_AUTH_TAG", "BUZZ_RELAY_URL"} <= \
            _HERMES_PROVIDER_ENV_BLOCKLIST

    def test_buzz_vars_use_plain_value_under_multiplex_without_scope(self, monkeypatch):
        """First-party platform vars are the process's own env values: with
        multiplex active and NO profile secret scope installed, the terminal
        scrub paths must forward the plain env value — NOT raise
        UnscopedSecretError (the fail-closed regression where the webhook-
        filter script runner crashed instead of running without the var)."""
        from agent import secret_scope as ss
        from tools.environments.local import _make_run_env, _sanitize_subprocess_env

        monkeypatch.setenv("BUZZ_MANAGED_AGENT", "1")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec-plain-value")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        ss.set_multiplex_active(True)
        try:
            run_env = _make_run_env({})
            sanitized = _sanitize_subprocess_env(
                {"BUZZ_PRIVATE_KEY": "nsec-plain-value", "HOME": "/home/user"}
            )
        finally:
            ss.set_multiplex_active(False)

        assert run_env["BUZZ_PRIVATE_KEY"] == "nsec-plain-value"
        assert sanitized["BUZZ_PRIVATE_KEY"] == "nsec-plain-value"

    def test_buzz_vars_are_not_scope_resolved(self, monkeypatch):
        """First-party matches bypass the profile secret scope: a scope value
        for BUZZ_PRIVATE_KEY must NOT override the process env value — only
        skill/config passthrough names are scope-resolved."""
        from agent import secret_scope as ss
        from tools.environments.local import _make_run_env, _sanitize_subprocess_env

        monkeypatch.setenv("BUZZ_MANAGED_AGENT", "1")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec-process-env")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"BUZZ_PRIVATE_KEY": "nsec-scoped"})
        try:
            run_env = _make_run_env({})
            sanitized = _sanitize_subprocess_env(
                {"BUZZ_PRIVATE_KEY": "nsec-process-env", "HOME": "/home/user"}
            )
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert run_env["BUZZ_PRIVATE_KEY"] == "nsec-process-env"
        assert sanitized["BUZZ_PRIVATE_KEY"] == "nsec-process-env"


class TestTerminalFirstPartySnapshotIsolation:
    """BUZZ_* first-party vars must not persist in the shared terminal
    snapshot — a cross-profile leak under a multiplexed gateway.

    The terminal login-shell snapshot (init_session ``export -p`` dump and the
    per-command re-dump) captures the child env, which now includes
    BUZZ_PRIVATE_KEY. The exclusion set is derived from get_all_passthrough()
    plus backend-specific additions — and BUZZ_* can never be in it, because
    env_passthrough refuses blocklisted names (GHSA-rhgp-j443-p4rf). Without
    an exclusion, profile A's BUZZ_PRIVATE_KEY lands in hermes-snap-<id>.sh
    and profile B's later command on the same collapsed LocalEnvironment
    sources it. Fix: LocalEnvironment treats first-party terminal env names
    like profile-scoped passthrough names — excluded from the dump and
    save/restored per command.
    """

    def test_snapshot_exclusion_set_includes_first_party_names(self, monkeypatch):
        """Under multiplex, BUZZ_* names present in the env are added to the
        snapshot exclusion set, so the dump excludes them and _wrap_command
        save/restores them per command."""
        from agent import secret_scope as ss
        from tools.environments.local import LocalEnvironment

        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec-profile-a")
        env = LocalEnvironment.__new__(LocalEnvironment)
        env.env = {}
        env._snapshot_passthrough_names = set()
        ss.set_multiplex_active(True)
        try:
            excluded = env._snapshot_excluded_passthrough_names()
        finally:
            ss.set_multiplex_active(False)

        assert "BUZZ_PRIVATE_KEY" in excluded
        # The set is monotonic for the environment lifetime: the name stays
        # excluded (and unset-guarded per command) even once it leaves the env.
        assert "BUZZ_PRIVATE_KEY" in env._snapshot_passthrough_names

    def test_buzz_secret_never_reaches_second_profile_via_snapshot(self, monkeypatch, tmp_path):
        """Multiplex regression, end-to-end with real bash: (a) the snapshot
        file never contains profile A's BUZZ_PRIVATE_KEY, and (b) profile B
        sharing the same LocalEnvironment does not see profile A's
        BUZZ_PRIVATE_KEY in its terminal env."""
        import shutil
        if not shutil.which("bash"):
            pytest.skip("bash required")

        from agent import secret_scope as ss
        from tools.environments.local import LocalEnvironment

        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec-profile-a")
        ss.set_multiplex_active(True)
        env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
        try:
            # Profile A's command re-dumps the snapshot; the exclusion must
            # keep BUZZ_PRIVATE_KEY out of BOTH the initial dump and the
            # per-command re-dump.
            env.execute("true")

            snap = Path(env._snapshot_path)
            assert snap.exists()
            snap_text = snap.read_text(encoding="utf-8", errors="replace")
            assert "nsec-profile-a" not in snap_text
            assert "BUZZ_PRIVATE_KEY" not in snap_text

            # Profile B: no BUZZ_PRIVATE_KEY in its env, same LocalEnvironment
            # (same snapshot file). It must not see profile A's value.
            monkeypatch.delenv("BUZZ_PRIVATE_KEY")
            result = env.execute("printf '%s' \"${BUZZ_PRIVATE_KEY-unset}\"")
            assert "nsec-profile-a" not in result["output"]
            assert "unset" in result["output"]
        finally:
            env.cleanup()
            ss.set_multiplex_active(False)


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


def _make_directory_link(link: Path, target: Path) -> None:
    """Create a directory link without requiring symlink privileges.

    POSIX: Path.symlink_to.  Windows: try symlink_to first (works with
    Developer Mode enabled), then fall back to an unprivileged directory
    junction via `cmd /c mklink /J` -- junctions do not require the
    SeCreateSymbolicLinkPrivilege.  Raises the original error when no
    mechanism is available so callers can skip with a clear reason.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        if sys.platform != "win32":
            raise
    # Binary capture: on a localized Windows the junction message is in the
    # console code page (e.g. GBK), which would raise UnicodeDecodeError in
    # the reader thread under UTF-8 mode.  Only the exit code matters.
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(detail or f"mklink /J failed: {result.returncode}")


def _physical_repo_root(tmp_path: Path) -> Path:
    """Create the physical repo checkout directory for junction tests."""
    physical_root = tmp_path / "physical-home" / "hermes-agent"
    physical_root.mkdir(parents=True)
    return physical_root


class TestPythonpathSelectiveStrip:
    """PYTHONPATH Hermes-owned entry stripping (#74817).

    The Desktop Electron app injects the Hermes repo root and the Hermes
    venv's site-packages (Python 3.11) into PYTHONPATH.  When this leaks
    into subprocesses running a different Python (e.g. 3.13), 3.11 C
    extensions appear on sys.path and crash with ImportError.
    ``_strip_hermes_owned_pythonpath`` surgically removes only the
    entries Hermes itself owns (repo root, own venv site-packages),
    preserving user paths — including user paths whose names merely
    contain another Python version.
    """

    def test_owned_entries_stripped_matrix(self):
        """Exact Hermes-owned entries are removed; everything else survives
        verbatim (ordering, duplicates, empty components).

        Covers: the running venv's site-packages, the repo root (computed
        independently via parents[2] so an off-by-one in _hermes_repo_root
        cannot silently pass), duplicate Hermes entries, all-owned input
        (PYTHONPATH key removed), and mixed user/Hermes ordering with an
        empty component preserved.
        """
        from tools.environments.local import _strip_hermes_owned_pythonpath

        venv_sp = str(_running_venv_site_packages())
        local_file = Path(__import__("tools.environments.local", fromlist=["__file__"]).__file__).resolve()
        repo_root = str(local_file.parents[2])
        cases = [
            ([venv_sp, "/home/user/my-lib"], ["/home/user/my-lib"]),
            ([repo_root, "/home/user/my-lib"], ["/home/user/my-lib"]),
            ([venv_sp, "/user/lib", venv_sp, "/user/lib"], ["/user/lib", "/user/lib"]),
            ([venv_sp], None),  # all owned -> PYTHONPATH key removed
            (["/first/user/lib", repo_root, "", venv_sp, "/second/user/lib"],
             ["/first/user/lib", "", "/second/user/lib"]),
        ]
        for input_entries, expected in cases:
            env = {"PYTHONPATH": os.pathsep.join(input_entries)}
            _strip_hermes_owned_pythonpath(env)
            if expected is None:
                assert "PYTHONPATH" not in env
            else:
                assert env["PYTHONPATH"].split(os.pathsep) == expected

    @pytest.mark.parametrize("user_pp", [
        os.pathsep.join(["/opt/my-lib", "/another/path"]),
        "/nix/store/abc123-user-plugin/lib/python3.12/site-packages",
        os.pathsep.join(["/old/lib/python2.7/site-packages", "/home/user/lib"]),
        os.pathsep.join(["/opt/tools/python3.13/bin", "/opt/downloads/python3.13", "/custom/python3.13"]),
        os.pathsep.join([" /opt/user-lib ", "relative/../lib", "", "/opt/user-lib", "/opt/user-lib"]),
        os.pathsep.join(["/foo", "", "/bar"]),
        "",
    ])
    def test_non_owned_entries_preserved(self, user_pp):
        """Anything not proven Hermes-owned is preserved byte-for-byte.

        One invariant, one matrix: ordinary user paths, Nix store paths,
        other-major/minor-version site-packages, paths merely containing a
        pythonX.Y component, raw spellings (whitespace, relative segments,
        duplicates), empty components, and an empty PYTHONPATH all reduce to
        the same contract -- ownership is decided by provenance, never by
        path shape or version (P1/P2, #74817 follow-ups).
        """
        from tools.environments.local import _strip_hermes_owned_pythonpath
        env = {"PYTHONPATH": user_pp}
        _strip_hermes_owned_pythonpath(env)
        assert env.get("PYTHONPATH") == user_pp

    def test_non_owned_runtime_shaped_entries_preserved(self):
        """Runtime-derived user spellings are preserved: site-packages for a
        different interpreter version, a descendant of the Hermes venv
        site-packages, and direct/deeper children of the repo root.  The
        repo root is computed independently (parents[2] of this file) so an
        off-by-one in _hermes_repo_root cannot silently pass; no launcher
        injects a direct child as a standalone entry, so such paths are user
        paths by contract.
        """
        from tools.environments.local import _strip_hermes_owned_pythonpath
        import sys

        running_minor = sys.version_info[1]
        other_minor = running_minor + 1 if running_minor < 20 else running_minor - 1
        local_file = Path(__import__("tools.environments.local", fromlist=["__file__"]).__file__).resolve()
        real_repo_root = local_file.parents[2]
        inputs = [
            os.pathsep.join([
                f"/opt/other-venv/lib/python{sys.version_info[0]}.{other_minor}/site-packages",
                "/home/user/my-lib",
            ]),
            os.pathsep.join([str(_running_venv_site_packages() / "some-user-path"), "/home/user/my-lib"]),
            os.pathsep.join([str(real_repo_root / "tools"), "/home/user/my-lib"]),
            os.pathsep.join([str(real_repo_root / "tools" / "environments"), "/home/user/my-lib"]),
        ]
        for user_pp in inputs:
            env = {"PYTHONPATH": user_pp}
            _strip_hermes_owned_pythonpath(env)
            assert env["PYTHONPATH"] == user_pp

    def test_windows_backslash_paths(self):
        """Windows-style backslash paths are handled for Hermes-owned entries.

        On Windows, os.pathsep is ';'.  We mock it so the test runs
        correctly on POSIX CI.  On a POSIX host a backslash path is a
        single path component, so ``Path`` cannot identify it as
        Hermes-owned — the critical invariant is that user Windows paths
        (including site-packages paths for another Python version) are
        never destroyed.  On a real Windows host, Path splits on
        backslashes and Hermes venv site-packages entries are stripped
        by the same Hermes-owned check (covered by the Windows-only test
        below).
        """
        from tools.environments.local import _strip_hermes_owned_pythonpath
        import sys

        pyver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
        hermes_win = f"C:\\\\Users\\\\u\\\\.hermes\\\\hermes-agent\\\\venv\\\\lib\\\\{pyver}\\\\site-packages"
        user_win = "D:\\\\user\\\\lib"
        env = {
            "PYTHONPATH": ";".join([hermes_win, user_win]),
        }
        # Mock os.pathsep to ';' (Windows) just for the strip call.
        with patch("os.pathsep", ";"):
            _strip_hermes_owned_pythonpath(env)
        assert "PYTHONPATH" in env
        entries = env["PYTHONPATH"].split(";")
        # Both survive on POSIX: user paths must always be preserved, and
        # the Hermes-owned check cannot match a backslash path here.
        assert hermes_win in entries
        assert user_win in entries

    @pytest.mark.windows_only
    def test_windows_hermes_owned_paths_stripped(self):
        """On Windows, a Hermes venv site-packages entry written with
        backslashes is stripped by the same Hermes-owned check, while a
        user Windows path is preserved.  Windows-only: POSIX ``Path`` does
        not split on backslashes, so this cannot be meaningfully simulated
        on a POSIX host."""
        from tools.environments.local import _strip_hermes_owned_pythonpath

        venv_sp = str(_running_venv_site_packages())
        # Windows form: C:\...\venv\Lib\site-packages (backslashes)
        hermes_win = venv_sp
        user_win = "D:\\\\user\\\\lib"
        env = {
            "PYTHONPATH": ";".join([hermes_win, user_win]),
        }
        _strip_hermes_owned_pythonpath(env)
        entries = env["PYTHONPATH"].split(";")
        assert hermes_win not in entries
        assert user_win in entries

    def test_empty_pythonpath_unchanged(self):
        """An empty PYTHONPATH is a no-op (falsy -> early return)."""
        from tools.environments.local import _strip_hermes_owned_pythonpath
        env = {"PYTHONPATH": ""}
        _strip_hermes_owned_pythonpath(env)
        # Empty string is falsy, so the function returns early without
        # modifying the dict.  The key stays as-is (empty string).
        assert env.get("PYTHONPATH") == ""

    def test_empty_component_preserved(self):
        """An empty component means cwd and must survive unchanged."""
        from tools.environments.local import _strip_hermes_owned_pythonpath

        user_pp = os.pathsep.join(["/foo", "", "/bar"])
        env = {"PYTHONPATH": user_pp}

        _strip_hermes_owned_pythonpath(env)

        assert env["PYTHONPATH"] == user_pp

    def test_raw_user_spelling_preserved(self):
        """The sanitizer does not trim, normalize, or deduplicate user entries."""
        from tools.environments.local import _strip_hermes_owned_pythonpath

        user_pp = os.pathsep.join([
            " /opt/user-lib ",
            "relative/../lib",
            "",
            "/opt/user-lib",
            "/opt/user-lib",
        ])
        env = {"PYTHONPATH": user_pp}

        _strip_hermes_owned_pythonpath(env)

        assert env["PYTHONPATH"] == user_pp


    def test_base_python_sanitizer_uses_validated_separate_runtime_venv(self, tmp_path, monkeypatch):
        """A base interpreter strips the exact Windows runtime site-packages.

        This deliberately uses a synthetic Hermes venv separate from the test
        runner: sys.prefix represents base Python, while validated VIRTUAL_ENV
        identifies ``<repo>/venv`` as the Hermes runtime producer contract.
        """
        import tools.environments.local as local

        repo_root = tmp_path / "hermes-agent"
        runtime_venv = repo_root / "venv"
        runtime_sp = runtime_venv / "Lib" / "site-packages"
        runtime_sp.mkdir(parents=True)
        (runtime_venv / "pyvenv.cfg").write_text("version = 3.11\n", encoding="utf-8")
        base_prefix = tmp_path / "base-python"
        unrelated = "/custom/lib/python3.13/site-packages"

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", (repo_root,))
        monkeypatch.setattr(local, "_in_venv", False)
        monkeypatch.setattr(local, "_hermes_site_packages", None)
        monkeypatch.setattr(local.sys, "prefix", str(base_prefix))
        monkeypatch.setattr(local.sys, "base_prefix", str(base_prefix))

        env = {
            "VIRTUAL_ENV": str(runtime_venv),
            "PYTHONPATH": os.pathsep.join([str(runtime_sp), unrelated]),
        }
        result = local._sanitize_subprocess_env(env)

        assert Path(local.sys.prefix) == base_prefix
        assert runtime_venv != Path(local.sys.prefix)
        assert result["PYTHONPATH"] == unrelated
        assert "VIRTUAL_ENV" not in result

    def test_unrelated_virtual_env_is_not_runtime_provenance(self, tmp_path, monkeypatch):
        """An arbitrary inherited VIRTUAL_ENV cannot claim PYTHONPATH ownership."""
        import tools.environments.local as local

        repo_root = tmp_path / "hermes-agent"
        repo_root.mkdir()
        unrelated_venv = tmp_path / "user-venv"
        unrelated_sp = unrelated_venv / "Lib" / "site-packages"
        unrelated_sp.mkdir(parents=True)
        (unrelated_venv / "pyvenv.cfg").write_text("version = 3.13\n", encoding="utf-8")

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", (repo_root,))
        monkeypatch.setattr(local, "_in_venv", False)
        monkeypatch.setattr(local, "_hermes_site_packages", None)

        env = {
            "VIRTUAL_ENV": str(unrelated_venv),
            "PYTHONPATH": str(unrelated_sp),
        }
        local._strip_hermes_owned_pythonpath(env)

        assert env["PYTHONPATH"] == str(unrelated_sp)


    def test_no_pythonpath_key(self):
        """Missing PYTHONPATH key is a no-op."""
        from tools.environments.local import _strip_hermes_owned_pythonpath
        env = {"PATH": "/usr/bin"}
        _strip_hermes_owned_pythonpath(env)
        assert "PYTHONPATH" not in env


    @pytest.mark.parametrize("builder", [
        "_make_run_env",
        "_sanitize_subprocess_env",
        "hermes_subprocess_env",
    ])
    def test_builders_strip_hermes_venv_pythonpath(self, builder):
        """Every subprocess env builder applies the same sanitation contract:
        Hermes venv site-packages is stripped, user entries survive.
        """
        from tools.environments import local as local_mod

        venv_sp = str(_running_venv_site_packages())
        seed = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "PYTHONPATH": os.pathsep.join([venv_sp, "/home/user/my-lib"]),
        }
        with patch.dict(os.environ, seed, clear=True):
            if builder == "_make_run_env":
                result = local_mod._make_run_env({})
            elif builder == "_sanitize_subprocess_env":
                result = local_mod._sanitize_subprocess_env(dict(os.environ))
            else:
                result = local_mod.hermes_subprocess_env()
        pp = result.get("PYTHONPATH", "")
        entries = pp.split(os.pathsep) if pp else []
        assert venv_sp not in entries
        assert "/home/user/my-lib" in entries

    def test_scrub_child_env_strips_hermes_venv_pythonpath(self):
        """execute_code's _scrub_child_env path: after scrubbing, Hermes venv
        site-packages entries should be stripped when
        _strip_hermes_owned_pythonpath is applied (as the spawn path does),
        while user entries (even for another Python version) are preserved.
        """
        from tools.code_execution_tool import _scrub_child_env
        from tools.environments.local import _strip_hermes_owned_pythonpath

        venv_sp = str(_running_venv_site_packages())
        other_sp = "/opt/other-venv/lib/python3.99/site-packages"
        source = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "PYTHONPATH": os.pathsep.join([venv_sp, other_sp, "/home/user/my-lib"]),
        }
        scrubbed = _scrub_child_env(source)
        # The scrubber passes PYTHONPATH through (it's in _SAFE_ENV_PREFIXES).
        assert "PYTHONPATH" in scrubbed
        # Now apply the selective strip (as the spawn path does).
        _strip_hermes_owned_pythonpath(scrubbed)
        pp = scrubbed.get("PYTHONPATH", "")
        entries = pp.split(os.pathsep) if pp else []
        assert venv_sp not in entries
        assert other_sp in entries
        assert "/home/user/my-lib" in entries

    @pytest.mark.parametrize("same_env", [True, False])
    def test_execute_code_composition_strips_inherited_hermes_entries(self, same_env):
        """Integration: execute_code's real spawn path composes a clean PYTHONPATH.

        Seeds a contaminated inherited PYTHONPATH (Hermes repo root + Hermes
        venv site-packages + user entries) through os.environ and drives
        execute_code all the way to Popen.  Proves the #84500 conditional
        composition and the #82581 selective strip compose correctly:

        * inherited Hermes venv site-packages never survive into the sandbox;
        * the staging tmpdir stays the first entry;
        * the repo root is deliberately re-added exactly once for a same-env
          child (the single occurrence proves the inherited copy was stripped
          first) and stays absent for an external-environment child;
        * user entries survive after the controlled entries.
        """
        import tools.code_execution_tool as cet
        from tools.code_execution_tool import execute_code

        def _mock_handle_function_call(function_name, function_args, task_id=None, user_task=None):
            return '{"output": "mock", "exit_code": 0}'

        hermes_root = str(Path(cet.__file__).resolve().parents[1])
        venv_sp = str(_running_venv_site_packages())
        user_a = "/home/user/my-lib"
        user_b = "/opt/project/lib"
        captured = {}

        def _fake_popen(cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            captured["staging"] = os.path.dirname(cmd[1])
            proc = MagicMock()
            proc.stdout.read.return_value = b""
            proc.stderr.read.return_value = b""
            proc.wait.return_value = 0
            proc.returncode = 0
            proc.poll.return_value = 0
            return proc

        with patch("tools.code_execution_tool._load_config",
                   return_value={"mode": "strict"}), \
             patch("model_tools.handle_function_call",
                   side_effect=_mock_handle_function_call), \
             patch("tools.code_execution_tool._uses_hermes_python_environment",
                   return_value=same_env), \
             patch("subprocess.Popen", side_effect=_fake_popen), \
             patch.dict(os.environ, {
                 "PYTHONPATH": os.pathsep.join(
                     [hermes_root, venv_sp, user_a, user_b]),
             }):
            execute_code(code="pass", task_id="test-int", enabled_tools=[])

        assert "PYTHONPATH" in captured["env"], \
            "execute_code never reached Popen"
        parts = captured["env"]["PYTHONPATH"].split(os.pathsep)
        # Windows path comparison is case-insensitive: the inherited entries
        # and the re-added repo root can carry a different case than the
        # resolve()/abspath()-derived spellings used in this test (e.g. a
        # launcher-written lowercase PYTHONPATH).  Normalize with
        # os.path.normcase so a case-only difference never fails the
        # composition contract (identity on POSIX).
        norm_parts = [os.path.normcase(p) for p in parts]
        norm_staging = os.path.normcase(captured["staging"])
        norm_root = os.path.normcase(hermes_root)
        norm_venv = os.path.normcase(venv_sp)
        norm_user_a = os.path.normcase(user_a)
        norm_user_b = os.path.normcase(user_b)
        assert norm_parts[0] == norm_staging, \
            "staging tmpdir must be the first PYTHONPATH entry"
        assert norm_venv not in norm_parts, \
            "inherited Hermes venv site-packages must be stripped"
        assert norm_user_a in norm_parts and norm_user_b in norm_parts, \
            "user PYTHONPATH entries must survive"
        assert norm_parts.index(norm_user_a) > norm_parts.index(norm_staging), \
            "user entries must come after the staging tmpdir"
        if same_env:
            assert norm_parts.count(norm_root) == 1, \
                "repo root must be re-added exactly once for a same-env child"
            assert norm_parts.index(norm_user_a) > norm_parts.index(norm_root), \
                "user entries must come after the re-added repo root"
        else:
            assert norm_root not in norm_parts, \
                "repo root must stay absent for an external-env child"


    def test_repo_root_direct_child_preserved(self):
        """A direct child of the repo root (depth=1) is PRESERVED.

        Independent audit of every real launcher producer (Electron
        ``apps/desktop/electron/main.ts``,
        ``gateway/run.py::_ensure_windows_gateway_venv_imports``,
        ``cron/scheduler.py::_windows_cron_python_invocation``,
        ``tui_gateway/host_supervisor.py``) shows they all inject the exact
        repo root and/or the venv site-packages — none injects
        ``<repo>/tools`` or another direct child as an independent
        PYTHONPATH entry.  A user path that merely happens to live under
        the repo directory must therefore be preserved.
        """
        from tools.environments.local import _strip_hermes_owned_pythonpath

        local_file = Path(__import__("tools.environments.local", fromlist=["__file__"]).__file__).resolve()
        real_repo_root = local_file.parents[2]
        direct_child = str(real_repo_root / "tools")

        env = {
            "PYTHONPATH": os.pathsep.join([direct_child, "/home/user/my-lib"]),
        }
        _strip_hermes_owned_pythonpath(env)
        pp = env.get("PYTHONPATH", "")
        entries = pp.split(os.pathsep) if pp else []
        assert direct_child in entries
        assert "/home/user/my-lib" in entries

    def test_configured_home_alias_matches_launcher_output(self, tmp_path, monkeypatch):
        """The real producer spelling is derived and consumed end to end."""
        import tools.environments.local as local
        from hermes_cli.gateway_windows import _preserve_hermes_home_path

        physical_home = tmp_path / "physical-home"
        physical_root = _physical_repo_root(tmp_path)
        configured_home = tmp_path / "configured-home"
        try:
            _make_directory_link(configured_home, physical_home)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")
        monkeypatch.setenv("HERMES_HOME", str(configured_home))

        launcher_entry = Path(_preserve_hermes_home_path(physical_root))
        aliases = local._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )

        assert launcher_entry == configured_home / "hermes-agent"
        assert launcher_entry in aliases

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        nested_user_path = launcher_entry / "user-data"
        env = {
            "PYTHONPATH": os.pathsep.join([
                str(launcher_entry),
                str(nested_user_path),
                "/home/user/my-lib",
            ])
        }
        local._strip_hermes_owned_pythonpath(env)

        assert env["PYTHONPATH"].split(os.pathsep) == [
            str(nested_user_path),
            "/home/user/my-lib",
        ]

    def test_profile_rehome_keeps_junction_lexical_alias(self, tmp_path, monkeypatch):
        """Profile re-home must not lose the launcher's lexical repo-root spelling.

        The desktop/CLI spawn children with HERMES_HOME and PYTHONPATH in the
        configured (junction) spelling, but --profile / sticky active_profile
        re-home HERMES_HOME through resolve_profile_env() before the
        sanitizer loads.  Regression (junction + profile re-home): the alias
        builder must still recover the lexical root so the inherited lexical
        repo-root entry is stripped.
        """
        import tools.environments.local as local
        from hermes_cli.profiles import resolve_profile_env

        physical_home = tmp_path / "physical-home"
        physical_root = physical_home / "hermes-agent"
        physical_root.mkdir(parents=True)
        (physical_home / "profiles" / "coder").mkdir(parents=True)
        configured_home = tmp_path / "configured-home"
        try:
            _make_directory_link(configured_home, physical_home)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")

        # Launcher contract: the configured spelling is the env and the root.
        monkeypatch.setenv("HERMES_HOME", str(configured_home))
        lexical_root = configured_home / "hermes-agent"

        # Profile re-home keeps the configured spelling (physically identical
        # through the link; lexically the launcher spelling is preserved).
        assert Path(resolve_profile_env("default")) == configured_home
        assert Path(resolve_profile_env("coder")) == configured_home / "profiles" / "coder"

        # The sanitizer now runs under the re-homed (profile) HERMES_HOME.
        aliases = local._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home / "profiles" / "coder",
        )
        assert any(local._same_path(a, lexical_root) for a in aliases)

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        env = {"PYTHONPATH": os.pathsep.join([str(lexical_root), "/home/user/my-lib"])}
        local._strip_hermes_owned_pythonpath(env)
        assert env["PYTHONPATH"].split(os.pathsep) == ["/home/user/my-lib"]


    def test_repo_level_junction_recovers_lexical_alias(self, tmp_path, monkeypatch):
        """The repo itself may be a junction under the configured root
        (e.g. D:\\hermes\\hermes-agent -> C:\\...\\hermes-agent) while the
        editable import spelling resolves to the physical location.  The
        alias builder must recover the lexical spelling via exact-identity
        proof (strict resolve), not a name-based guess.
        """
        import tools.environments.local as local

        physical_root = _physical_repo_root(tmp_path)
        configured_home = tmp_path / "configured-home"
        configured_home.mkdir()
        # repo-level link: <configured-home>/hermes-agent -> physical repo
        try:
            _make_directory_link(configured_home / "hermes-agent", physical_root)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")

        lexical_root = configured_home / "hermes-agent"
        aliases = local._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )
        assert any(local._same_path(a, lexical_root) for a in aliases)

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        env = {"PYTHONPATH": os.pathsep.join([str(lexical_root), "/home/user/my-lib"])}
        local._strip_hermes_owned_pythonpath(env)
        assert env["PYTHONPATH"].split(os.pathsep) == ["/home/user/my-lib"]

    def test_same_named_non_owned_directories_preserved(self, tmp_path, monkeypatch):
        """Negative controls: a directory that merely shares the repo's name
        -- whether under the configured root or in an unrelated location --
        is never aliased or stripped.  Exact filesystem identity decides,
        not the name; no ownership provenance means no strip.
        """
        import tools.environments.local as local

        physical_root = _physical_repo_root(tmp_path)
        configured_home = tmp_path / "configured-home"
        (configured_home / "hermes-agent").mkdir(parents=True)
        unrelated = tmp_path / "user-tools" / "hermes-agent"
        unrelated.mkdir(parents=True)

        aliases = local._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )
        for lookalike in (configured_home / "hermes-agent", unrelated):
            assert not any(local._same_path(a, lookalike) for a in aliases)

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        for lookalike in (configured_home / "hermes-agent", unrelated):
            env = {"PYTHONPATH": os.pathsep.join([str(lookalike), "/home/user/my-lib"])}
            local._strip_hermes_owned_pythonpath(env)
            assert env["PYTHONPATH"].split(os.pathsep) == [str(lookalike), "/home/user/my-lib"]

    def test_profile_home_with_repo_level_junction(self, tmp_path, monkeypatch):
        """Profile re-home + repo-level junction together: the configured home
        is <root>/profiles/<name> while the repo is a link at <root>/hermes-agent.
        The root spelling must be derived (profiles -> grandparent) and then
        the lexical repo alias recovered from it.
        """
        import tools.environments.local as local

        physical_root = _physical_repo_root(tmp_path)
        configured_root = tmp_path / "configured-root"
        (configured_root / "profiles" / "coder").mkdir(parents=True)
        try:
            _make_directory_link(configured_root / "hermes-agent", physical_root)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")

        configured_home = configured_root / "profiles" / "coder"
        lexical_root = configured_root / "hermes-agent"
        aliases = local._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )
        assert any(local._same_path(a, lexical_root) for a in aliases)
        assert not any(local._same_path(a, configured_home / "hermes-agent") for a in aliases)

        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)
        env = {"PYTHONPATH": os.pathsep.join([str(lexical_root), "/home/user/my-lib"])}
        local._strip_hermes_owned_pythonpath(env)
        assert env["PYTHONPATH"].split(os.pathsep) == ["/home/user/my-lib"]

    def test_validated_runtime_venv_lexical_after_repo_recovery(self, tmp_path, monkeypatch):
        """uv-base gateway: once the lexical repo alias is recovered, a lexical
        VIRTUAL_ENV (<lexical repo>/venv) validates and its site-packages is
        stripped together with the repo root, while user entries survive.
        """
        import tools.environments.local as local

        physical_root = _physical_repo_root(tmp_path)
        venv_dir = physical_root / "venv"
        venv_dir.mkdir(parents=True)
        (venv_dir / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
        configured_home = tmp_path / "configured-home"
        configured_home.mkdir()
        try:
            _make_directory_link(configured_home / "hermes-agent", physical_root)
        except OSError as exc:
            pytest.skip(f"directory link unavailable on this host: {exc}")

        lexical_root = configured_home / "hermes-agent"
        aliases = local._build_hermes_repo_root_aliases(
            physical_root.resolve(),
            physical_root,
            configured_home,
        )
        assert any(local._same_path(a, lexical_root) for a in aliases)
        monkeypatch.setattr(local, "_hermes_repo_root_aliases", aliases)

        lexical_venv = lexical_root / "venv"
        validated = local._validated_runtime_venv({"VIRTUAL_ENV": str(lexical_venv)})
        assert validated is not None
        assert local._same_path(validated, lexical_venv)

        local._hermes_site_packages = None
        env = {"PYTHONPATH": os.pathsep.join([
            str(lexical_root),
            str(lexical_venv / "Lib" / "site-packages"),
            "/home/user/my-lib",
        ]), "VIRTUAL_ENV": str(lexical_venv)}
        local._strip_hermes_owned_pythonpath(env)
        assert env["PYTHONPATH"].split(os.pathsep) == ["/home/user/my-lib"]





class TestPythonhomeSanitized:
    """PYTHONHOME must not leak from the Hermes runtime into subprocesses.

    The gateway inherits/sets PYTHONHOME in its process environment; a child
    interpreter (system Python, another venv, cron no_agent scripts) that
    inherits it redirects its stdlib search to the Hermes venv and crashes
    with version-mismatch errors before importing anything (#75018).
    """

    @pytest.mark.parametrize("builder", [
        "_make_run_env",
        "_sanitize_subprocess_env",
        "hermes_subprocess_env",
        "build_subprocess_env",
    ])
    def test_builders_strip_pythonhome(self, builder):
        """The gateway's inherited PYTHONHOME must not reach any subprocess
        builder -- terminal, background/PTY, cron no_agent scripts, and
        execute_code children (#75018).
        """
        from tools.environments import local as local_mod

        seed = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "PYTHONHOME": "/opt/hermes-venv",
        }
        with patch.dict(os.environ, seed, clear=True):
            if builder == "_make_run_env":
                result = local_mod._make_run_env({})
            elif builder == "_sanitize_subprocess_env":
                result = local_mod._sanitize_subprocess_env(dict(os.environ))
            elif builder == "hermes_subprocess_env":
                result = local_mod.hermes_subprocess_env()
            else:
                result = local_mod.build_subprocess_env()
        assert "PYTHONHOME" not in result

    def test_pythonhome_removed_from_active_venv_markers(self):
        """PYTHONHOME is part of _ACTIVE_VENV_MARKER_VARS so all builders
        that iterate it drop the variable."""
        from tools.environments.local import _ACTIVE_VENV_MARKER_VARS
        assert "PYTHONHOME" in _ACTIVE_VENV_MARKER_VARS

    def test_build_subprocess_env_no_scrub_preserves_pythonhome(self):
        """``build_subprocess_env(scrub_secrets=False)`` is the documented
        byte-for-byte escape hatch: no key is removed, so PYTHONHOME (and
        everything else) survives there by contract, not by omission.

        Callers that explicitly opt out of scrubbing (git credential flows,
        secret CLIs) must not have their environment silently altered — this
        test pins that exception as intentional.
        """
        from tools.environments.local import build_subprocess_env
        base = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "PYTHONHOME": "/opt/hermes-venv",
            "VIRTUAL_ENV": "/opt/hermes-venv",
            "SERVICE_TOKEN": "s3cr3t",
        }
        result = build_subprocess_env(base, scrub_secrets=False)
        assert result.get("PYTHONHOME") == "/opt/hermes-venv"
        assert result.get("VIRTUAL_ENV") == "/opt/hermes-venv"
        assert result.get("SERVICE_TOKEN") == "s3cr3t"


class TestProfileScopedPassthrough:
    def test_make_run_env_uses_active_profile_for_passthrough(self, monkeypatch):
        """Allowlisted values must come from the routed profile, not os.environ."""
        from agent import secret_scope as ss
        from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
        from tools.environments.local import _make_run_env

        clear_env_passthrough()
        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-routed-profile"})
        try:
            result = _make_run_env({})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)
            clear_env_passthrough()

        assert result["SERVICE_TOKEN"] == "token-for-routed-profile"

    def test_make_run_env_omits_missing_scoped_passthrough(self, monkeypatch):
        """A missing routed secret must not fall back to the default profile."""
        from agent import secret_scope as ss
        from tools.env_passthrough import clear_env_passthrough, register_env_passthrough
        from tools.environments.local import _make_run_env

        clear_env_passthrough()
        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            result = _make_run_env({})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)
            clear_env_passthrough()

        assert "SERVICE_TOKEN" not in result


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
            "VERCEL_OIDC_TOKEN",
            "VERCEL_TOKEN",
            "VERCEL_PROJECT_ID",
            "VERCEL_TEAM_ID",
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


    def test_make_run_env_appends_homebrew_on_minimal_path(self, monkeypatch, tmp_path):
        """When PATH is minimal, _make_run_env appends missing sane entries.

        POSIX: the sane-path merge appends the Homebrew dirs.  Windows:
        _append_missing_sane_path_entries appends the private managed-uv dir
        at the tail (without reordering the native PATH), so the assertion is
        the input plus the managed uv dir.  Git Bash dir prepending is
        neutralised so the merged PATH layout is deterministic on every host.
        """
        from tools.environments import local as local_mod
        from tools.environments.local import _SANE_PATH, _make_run_env
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
        minimal_env = {"PATH": "/some/custom/bin"}
        # Pin get_hermes_home inside the env swap: clear=True blanks HOME /
        # USERPROFILE, so the in-function fallback (Path.home()) would raise
        # and the Windows branch of _append_missing_sane_path_entries would
        # silently skip the uv append. Pin it so the append and assertion agree.
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
             patch.dict(os.environ, minimal_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(os.pathsep)
        assert path_entries[0] == "/some/custom/bin"
        if sys.platform == "win32":
            # The agent PATH keeps the caller's entries and gains the private
            # managed-uv dir. Assert the relationship (uv present, original
            # entry retained) rather than the exact joined string so a future
            # extra appended entry does not break this test.
            assert str(tmp_path / "uv") in path_entries
        else:
            for entry in _SANE_PATH.split(os.pathsep):
                assert entry in path_entries


    @pytest.mark.macos_only
    def test_make_run_env_real_launchd_path_gains_homebrew(self):
        """The literal macOS launchd PATH is the production trigger for #35613.

        macOS-only: the regression is the launchd environment on macOS, and
        the sane-path merge is a documented passthrough on Windows.
        """
        from tools.environments.local import _make_run_env
        launchd_env = {"PATH": os.pathsep.join(["/usr/bin", "/bin", "/usr/sbin", "/sbin"])}
        with patch.dict(os.environ, launchd_env, clear=True):
            result = _make_run_env({})
        path_entries = result["PATH"].split(os.pathsep)
        assert "/opt/homebrew/bin" in path_entries
        assert "/opt/homebrew/sbin" in path_entries
        # Original entries keep their leading precedence.
        assert path_entries[:4] == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]


    @pytest.mark.windows_only
    def test_make_run_env_preserves_windows_mixed_case_path_key(self, monkeypatch, tmp_path):
        """Windows-only: ``_path_env_key`` looks for a case-insensitive PATH
        key only on Windows, so the mixed-case ``Path`` preservation this
        asserts is a genuinely Windows-native behaviour.

        The Git Bash dir prepend is neutralised so the assertion is about the
        key casing alone; the private managed-uv dir (``$HERMES_HOME/uv``) is
        the one thing appended at the tail. ``get_hermes_home`` is pinned,
        since the ``os.environ`` swap above blanks USERPROFILE and would make
        the in-function fallback raise (and silently skip the uv append).
        """
        from tools.environments import local as local_mod
        from tools.environments.local import _make_run_env
        windows_env = {"Path": r"C:\Windows\System32;C:\Program Files\Git\bin"}
        monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
             patch.object(local_mod.os, "environ", windows_env):
            result = _make_run_env({})
        parts = result["Path"].split(os.pathsep)
        # Every original Windows PATH entry is retained (not reordered/dropped)
        # and the private managed-uv dir is appended. Assert the relationship.
        for entry in windows_env["Path"].split(os.pathsep):
            assert entry in parts
        assert str(tmp_path / "uv") in parts
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


    def test_prepend_noop_when_unresolved(self, monkeypatch):
        from tools.environments import local as local_mod
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = None
        assert local_mod._prepend_hermes_bin_dir("/usr/bin:/bin") == "/usr/bin:/bin"

    def test_make_run_env_injects_hermes_bin_dir(self):
        """A gateway env missing the hermes dir gets it back in the subshell PATH.

        Platform-agnostic: ``_prepend_hermes_bin_dir`` uses ``os.pathsep`` on
        every host, so no platform flag is faked here."""
        from tools.environments import local as local_mod
        from tools.environments.local import _make_run_env
        self._reset_cache()
        local_mod._HERMES_BIN_DIR = "/opt/hermes/bin"
        with patch.dict(
            os.environ,
            {"PATH": os.pathsep.join(["/usr/bin", "/bin"])},
            clear=True,
        ):
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
