import os
import subprocess
import shlex
import pytest

import tools.delegate_tool as delegate_tool


@pytest.fixture
def capture_subprocess(monkeypatch):
    """Patch delegate_tool.subprocess.run and capture calls.

    Returns a list collecting call dicts: {'cmd': cmd, 'kwargs': kwargs}
    and makes the fake run return subprocess.CompletedProcess(cmd, 0).
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return subprocess.CompletedProcess(cmd, 0)

    # Patch the subprocess.run used inside the delegate_tool module
    monkeypatch.setattr(delegate_tool.subprocess, "run", fake_run)
    return calls


def test_acp_args_model_wins(monkeypatch, capture_subprocess):
    monkeypatch.setenv("HERMES_MODEL", "env-model")
    cfg = {"model": "config-model"}
    acp_command = "mockcmd"
    acp_args = ["--acp", "--stdio", "--model", "acp-model"]

    calls = capture_subprocess
    delegate_tool._launch_subagent(acp_command, list(acp_args), cfg)

    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call["cmd"], list)
    assert call["cmd"][-2:] == ["--model", "acp-model"]
    assert call["kwargs"].get("env", {}).get("HERMES_MODEL") == "acp-model"


def test_hermes_model_env_wins_over_config(monkeypatch, capture_subprocess):
    monkeypatch.setenv("HERMES_MODEL", "env-model")
    cfg = {"model": "config-model"}
    acp_command = "mockcmd"
    acp_args = ["--acp", "--stdio"]

    calls = capture_subprocess
    delegate_tool._launch_subagent(acp_command, list(acp_args), cfg)

    assert len(calls) == 1
    call = calls[0]
    assert call["cmd"][-2:] == ["--model", "env-model"]
    assert call["kwargs"].get("env", {}).get("HERMES_MODEL") == "env-model"


def test_config_model_is_fallback(monkeypatch, capture_subprocess):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    cfg = {"model": "config-model"}
    acp_command = "mockcmd"
    acp_args = ["--acp", "--stdio"]

    calls = capture_subprocess
    delegate_tool._launch_subagent(acp_command, list(acp_args), cfg)

    assert len(calls) == 1
    call = calls[0]
    assert call["cmd"][-2:] == ["--model", "config-model"]
    assert call["kwargs"].get("env", {}).get("HERMES_MODEL") == "config-model"


def test_no_model_override_if_already_in_args(monkeypatch, capture_subprocess):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    cfg = {"model": "config-model"}
    acp_command = "mockcmd"
    acp_args = ["--acp", "--stdio", "--model=config-inline"]

    calls = capture_subprocess
    delegate_tool._launch_subagent(acp_command, list(acp_args), cfg)

    assert len(calls) == 1
    call = calls[0]
    # Should preserve the original --model=config-inline arg (no injection)
    assert "--model=config-inline" in call["cmd"]
    # And should set the env HERMES_MODEL to the parsed value
    assert call["kwargs"].get("env", {}).get("HERMES_MODEL") == "config-inline"


def test_no_model_set_at_all(monkeypatch, capture_subprocess):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    cfg = {}
    acp_command = "mockcmd"
    acp_args = ["--acp", "--stdio"]

    calls = capture_subprocess
    delegate_tool._launch_subagent(acp_command, list(acp_args), cfg)

    assert len(calls) == 1
    call = calls[0]
    # No --model injected
    assert "--model" not in " ".join(call["cmd"]) or ("--model=" not in " ".join(call["cmd"]))
    # Environment should not contain HERMES_MODEL
    assert call["kwargs"].get("env", {}).get("HERMES_MODEL") is None


@pytest.mark.parametrize("arg_form", [
    (["--model", "X"], "X"),
    (["-m", "Y"], "Y"),
    (["--model=Z"], "Z"),
])
def test_parses_model_forms(monkeypatch, capture_subprocess, arg_form):
    (form_args, expected_model) = arg_form
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    cfg = {}
    acp_command = "mockcmd"
    acp_args = ["--acp", "--stdio"] + form_args

    calls = capture_subprocess
    delegate_tool._launch_subagent(acp_command, list(acp_args), cfg)

    assert len(calls) == 1
    call = calls[0]
    # Ensure the model is present in the command as given
    joined = " ".join(call["cmd"]) if isinstance(call["cmd"], (list, tuple)) else str(call["cmd"])
    assert expected_model in joined
    assert call["kwargs"].get("env", {}).get("HERMES_MODEL") == expected_model
