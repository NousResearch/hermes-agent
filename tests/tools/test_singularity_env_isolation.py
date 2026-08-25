"""Tests for Singularity/Apptainer subprocess environment isolation.

Apptainer/Singularity control and ``exec`` processes would otherwise inherit
the calling process environment. Hermes routes command execution through the
shared ``_popen_bash`` sanitizer and supplies explicit sanitized mappings to
lifecycle and image-build subprocesses. These tests assert those production
boundaries without claiming native container-runtime verification.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools.environments.singularity import (
    SingularityEnvironment,
    _singularity_subprocess_env,
)
from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST


class TestFilteredContainerEnv:
    def test_strips_provider_api_keys(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-12345")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")

        result = _singularity_subprocess_env()

        assert "ANTHROPIC_API_KEY" not in result
        assert "OPENAI_API_KEY" not in result

    def test_strips_hermes_internal_secrets(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "dashboard-secret-xyz")
        monkeypatch.setenv("GH_TOKEN", "ghp_supersecrettoken")

        result = _singularity_subprocess_env()

        assert "HERMES_DASHBOARD_SESSION_TOKEN" not in result
        assert "GH_TOKEN" not in result

    def test_preserves_benign_vars(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setenv("LANG", "en_US.UTF-8")

        result = _singularity_subprocess_env()

        assert result.get("PATH", "").endswith("/usr/bin:/bin")
        assert result.get("LANG") == "en_US.UTF-8"

    def test_overrides_merge_over_os_environ(self, monkeypatch):
        from tools.environments.local import build_subprocess_env

        result = build_subprocess_env(base={"SOME_VAR": "from-overrides"})

        assert result["SOME_VAR"] == "from-overrides"

    def test_every_blocklisted_var_actually_stripped(self, monkeypatch):
        """Direct regression against the real blocklist, not just a sample."""
        for key in _HERMES_PROVIDER_ENV_BLOCKLIST:
            monkeypatch.setenv(key, "leaked-if-present")

        result = _singularity_subprocess_env()

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
    def test_run_bash_reaches_shared_filtered_env_at_popen(self, monkeypatch):
        """The shared ``_popen_bash`` boundary filters Singularity exec."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "redacted-provider-value")

        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return object()

        env = _bare_singularity_env()
        with patch("tools.environments.base.subprocess.Popen", fake_popen):
            env._run_bash("echo hi")

        assert "env" in captured["kwargs"]
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
