"""Tests for Singularity/Apptainer subprocess env isolation.

Apptainer/Singularity's ``exec`` inherits the calling process's FULL
environment into the container by default -- ``--cleanenv`` (passed to
``instance start``) is not repeated on the per-command ``exec`` that
``SingularityEnvironment._run_bash`` actually spawns. Unlike LocalEnvironment
(``_make_run_env``) and DockerEnvironment (explicit ``-e`` forwarding), the
Singularity backend never filtered the env at all, so every hermes provider
API key / session token / gateway secret in the host process's environment
was inherited by the container on every terminal command.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools.environments.singularity import (
    SingularityEnvironment,
    _filtered_container_env,
)
from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST


class TestFilteredContainerEnv:
    def test_strips_provider_api_keys(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-12345")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")

        result = _filtered_container_env({})

        assert "ANTHROPIC_API_KEY" not in result
        assert "OPENAI_API_KEY" not in result

    def test_strips_hermes_internal_secrets(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "dashboard-secret-xyz")
        monkeypatch.setenv("GH_TOKEN", "ghp_supersecrettoken")

        result = _filtered_container_env({})

        assert "HERMES_DASHBOARD_SESSION_TOKEN" not in result
        assert "GH_TOKEN" not in result

    def test_preserves_benign_vars(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setenv("LANG", "en_US.UTF-8")

        result = _filtered_container_env({})

        assert result.get("PATH") == "/usr/bin:/bin"
        assert result.get("LANG") == "en_US.UTF-8"

    def test_overrides_merge_over_os_environ(self, monkeypatch):
        monkeypatch.setenv("SOME_VAR", "from-os-environ")

        result = _filtered_container_env({"SOME_VAR": "from-overrides"})

        assert result["SOME_VAR"] == "from-overrides"

    def test_every_blocklisted_var_actually_stripped(self, monkeypatch):
        """Direct regression against the real blocklist, not just a sample."""
        for key in _HERMES_PROVIDER_ENV_BLOCKLIST:
            monkeypatch.setenv(key, "leaked-if-present")

        result = _filtered_container_env({})

        leaked = set(_HERMES_PROVIDER_ENV_BLOCKLIST) & result.keys()
        assert not leaked, f"blocklisted vars leaked into container env: {leaked}"


def _bare_singularity_env(env_overrides: dict | None = None) -> SingularityEnvironment:
    """Construct a SingularityEnvironment without running its real __init__
    (which starts an actual apptainer/singularity instance)."""
    instance = object.__new__(SingularityEnvironment)
    instance.executable = "apptainer"
    instance.instance_id = "hermes_test_instance"
    instance._instance_started = True
    instance.env = env_overrides or {}
    return instance


class TestRunBashEnvIsolation:
    def test_run_bash_passes_filtered_env_to_popen(self, monkeypatch):
        """The exact bug: _run_bash must not spawn with the unfiltered
        (inherited) environment -- every terminal command run under
        TERMINAL_ENV=singularity must be sanitized the same way Local and
        Docker already are."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-12345")

        captured = {}

        def fake_popen_bash(cmd, stdin_data=None, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return object()

        env = _bare_singularity_env()
        with patch("tools.environments.singularity._popen_bash", fake_popen_bash):
            env._run_bash("echo hi")

        assert "env" in captured["kwargs"], "_run_bash must pass env= explicitly"
        assert "ANTHROPIC_API_KEY" not in captured["kwargs"]["env"]
        assert captured["cmd"] == [
            "apptainer", "exec", "instance://hermes_test_instance",
            "bash", "-c", "echo hi",
        ]

    def test_run_bash_raises_when_instance_not_started(self):
        env = _bare_singularity_env()
        env._instance_started = False
        with pytest.raises(RuntimeError, match="not started"):
            env._run_bash("echo hi")
