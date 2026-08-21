"""Regression tests for env_passthrough forwarding under APPTAINER_CLEANENV (#90298).

Apptainer strips every inherited variable at the container boundary when
``APPTAINER_CLEANENV`` is active unless it carries the ``APPTAINERENV_``
prefix. The singularity backend must forward each allowlisted
``terminal.env_passthrough`` entry under that prefix so the documented
passthrough exception survives cleanenv isolation — without forwarding
anything that is not allowlisted (isolation stays default-deny).
"""

import os

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def clean_passthrough_state():
    from tools import env_passthrough as ep

    ep.clear_env_passthrough()
    yield
    ep.clear_env_passthrough()


def _exec_env(monkeypatch, allowlist_names, environ):
    from tools.environments.singularity import _apptainer_exec_env

    for name in allowlist_names:
        monkeypatch.setenv(name, environ[name])
    with patch(
        "tools.env_passthrough._load_config_passthrough",
        return_value=frozenset(allowlist_names),
    ):
        return _apptainer_exec_env()


class TestApptainerExecEnv:
    def test_allowlisted_entry_gets_apptainerenv_prefix(self, monkeypatch):
        """The documented prefix is what survives APPTAINER_CLEANENV."""
        env = _exec_env(
            monkeypatch,
            ["MY_SERVICE_TOKEN"],
            {"MY_SERVICE_TOKEN": "tok-sandbox-forward"},
        )
        assert env["APPTAINERENV_MY_SERVICE_TOKEN"] == "tok-sandbox-forward"
        assert env["MY_SERVICE_TOKEN"] == "tok-sandbox-forward"

    def test_non_allowlisted_env_has_no_prefix(self, monkeypatch):
        """Only allowlisted names are forwarded — cleanenv stays default-deny."""
        monkeypatch.setenv("SECRET_NOT_LISTED", "should-not-be-prefixed")
        env = _exec_env(monkeypatch, [], {})
        assert not any(
            key.startswith("APPTAINERENV_") for key in env
        ), "non-allowlisted variables must not be forwarded across the cleanenv boundary"

    def test_allowlisted_but_unset_entry_is_skipped(self, monkeypatch):
        env = _exec_env(monkeypatch, ["MY_UNSET_TOKEN"], {"MY_UNSET_TOKEN": ""})
        # Simulate an entry with no value available anywhere: nothing to forward.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MY_UNSET_TOKEN", None)
            from tools.environments.singularity import _apptainer_exec_env
            with patch(
                "tools.env_passthrough._load_config_passthrough",
                return_value=frozenset({"MY_UNSET_TOKEN"}),
            ):
                env = _apptainer_exec_env()
        assert "APPTAINERENV_MY_UNSET_TOKEN" not in env

    def test_scope_resolves_value_over_host_environ(self, monkeypatch):
        """An active profile scope is authoritative for the forwarded value."""
        from agent import secret_scope as ss
        from tools.environments.singularity import _apptainer_exec_env

        monkeypatch.setenv("MY_SERVICE_TOKEN", "host-value")

        def _fake_config_passthrough():
            return frozenset({"MY_SERVICE_TOKEN"})

        with patch(
            "tools.env_passthrough._load_config_passthrough",
            return_value=frozenset({"MY_SERVICE_TOKEN"}),
        ), patch(
            "tools.env_passthrough.resolve_passthrough_value",
            side_effect=lambda name, fallback: "scoped-value"
            if name == "MY_SERVICE_TOKEN"
            else fallback,
        ):
            env = _apptainer_exec_env()
        assert env["APPTAINERENV_MY_SERVICE_TOKEN"] == "scoped-value"


class TestRunBashUsesForwardedEnv:
    def test_run_bash_passes_exec_env(self, monkeypatch):
        """_run_bash must hand the forwarded env to the spawned apptainer exec."""
        from tools.environments import singularity as sing

        env_obj = object.__new__(sing.SingularityEnvironment)
        env_obj._instance_started = True
        env_obj.executable = "apptainer"
        env_obj.instance_id = "hermes_test"

        captured = {}

        class _FakeProc:
            pass

        def _fake_popen(cmd, stdin_data=None, **kwargs):
            captured.update(kwargs)
            return _FakeProc()

        with patch.object(sing, "_popen_bash", side_effect=_fake_popen):
            env_obj._run_bash("echo hi")

        # Defuse BaseEnvironment.__del__ cleanup for the partial object.
        env_obj._instance_started = False

        assert "env" in captured, "_run_bash must pass an explicit env"
        assert isinstance(captured["env"], dict)
