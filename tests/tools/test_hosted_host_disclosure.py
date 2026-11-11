"""What a hosted tool call reveals about this host: a failed hosted terminal call returns no
traceback of this host's install. The local counterpart asserts behaviour there is unchanged."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.tools.test_sandbox_scope_conformance import A
from tests.tools.test_sandbox_scope_conformance import provider as _conformance_provider
from tools.environments import sandbox_conformance as c

provider = _conformance_provider  # the recording-sandbox fixture, by its name

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _host_install_paths() -> list[str]:
    return [p for p in {REPO_ROOT, sys.prefix, sys.base_prefix, sys.exec_prefix} if p and p != "/"]


# --------------------------------------------------------------------------------------------------
# A failed terminal call


@pytest.mark.parametrize("backend", ["local", "docker", "nosuchbackend_zz"])
def test_a_failed_hosted_terminal_call_returns_no_host_traceback(provider, monkeypatch, backend):
    monkeypatch.setenv("TERMINAL_ENV", backend)
    with c._bind(A):
        result = c.call_tool("terminal", {"command": "echo hi"}, task_id="t")
    assert result["status"] == "error" and result["exit_code"] == -1, result
    assert result["error"].startswith("Failed to execute command: "), result
    assert "traceback" not in result, result
    text = json.dumps(result)
    for path in _host_install_paths():
        assert path not in text, f"a hosted terminal result named the host path {path}: {result}"
    assert provider.created == []


def test_a_failed_local_terminal_call_still_returns_its_traceback(monkeypatch):
    from tools import terminal_tool
    monkeypatch.setenv("TERMINAL_ENV", "nosuchbackend_zz")
    result = json.loads(terminal_tool.terminal_tool("echo hi", task_id="t-local-fail"))
    assert result["status"] == "error" and "Unknown environment type" in result["error"]
    assert result["traceback"].startswith("Traceback (most recent call last):"), result
    assert list(result) == ["output", "exit_code", "error", "traceback", "status"]
