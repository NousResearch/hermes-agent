"""What a hosted tool call reveals about this host, or consults on it: a failed hosted terminal call
returns no traceback of this host's install, and the secret-file test a hosted read's path goes
through judges the sandbox path as written, never resolving it on this host. The local
counterparts assert behaviour there is unchanged."""

from __future__ import annotations

import base64
import contextlib
import json
import re
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


# --------------------------------------------------------------------------------------------------
# The secret-file test on a hosted read's path


SECRET = "Zq8vLmN3pXr7TtYw2KjH"
CONFIG = f"model:\n  api_key: {SECRET}\n"


def _serve_file(monkeypatch, content: str = CONFIG) -> None:
    """The recording sandbox answers ``read_file``'s one-round-trip probe with *content*."""
    original = c.RecordingSandboxEnvironment.execute

    def execute(self, command, **kwargs):
        result = original(self, command, **kwargs)
        found = re.search(r"echo (__HERMES_RF_[0-9a-f]+__)", command)
        if not found:
            return result
        mark, data = found.group(1) + "\n", content.encode()
        segments = [f"{len(data)}\n", base64.b64encode(data[:1000]).decode() + "\n", content,
                    f"{content.count(chr(10))}\n", "1\n" if content.endswith("\n") else "0\n"]
        return {"output": mark.join(segments) + mark + "0 0\n", "returncode": 0}

    monkeypatch.setattr(c.RecordingSandboxEnvironment, "execute", execute)


@contextlib.contextmanager
def _watching(*roots: Path):
    """Every host open, listing, stat, lstat and readlink under any of *roots* (one shared record)."""
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(c._watch_host(root)) for root in roots][-1]


@pytest.mark.parametrize("path", ["/workspace/config.yaml", "config.yaml"])
def test_a_hosted_config_yaml_read_never_resolves_its_path_on_this_host(provider, monkeypatch, tmp_path, path):
    host_cwd = tmp_path / "host-cwd"
    host_cwd.mkdir()
    monkeypatch.chdir(host_cwd)
    _serve_file(monkeypatch)
    from tools.file_tools import _resolve_path_for_task
    with c._bind(A):
        c.call_tool("terminal", {"command": "true"}, task_id="t")  # the sandbox is up
        # Where the read lands in the sandbox (a relative path: against the sandbox's working directory).
        in_sandbox = Path(str(_resolve_path_for_task(path, "t")))
        landed = in_sandbox.parent if in_sandbox.parent != Path("/") else in_sandbox
        roots = {Path("/workspace"), host_cwd} | ({landed} if in_sandbox.is_absolute() else set())
        with _watching(*roots) as touched:
            result = c.call_tool("read_file", {"path": path}, task_id="t")
            touched = sorted(set(touched))
    assert SECRET in result.get("content", ""), result  # arbitrary YAML: not a secret file
    assert not touched, f"a hosted read resolved {in_sandbox} on this host: {touched}"


def test_a_hosted_read_under_a_hermes_directory_is_still_a_secret_file(provider, monkeypatch):
    _serve_file(monkeypatch)
    with c._bind(A):
        secret = c.call_tool("read_file", {"path": "/workspace/.hermes/config.yaml"}, task_id="t")
        nested = c.call_tool("read_file", {"path": "/workspace/app/../.hermes/config.yaml"}, task_id="t")
    for result in (secret, nested):
        assert "api_key:" in result["content"] and SECRET not in result["content"], result


def test_a_local_config_yaml_under_the_hermes_home_is_still_recognized():
    from agent.redact import _is_under_hermes_home
    from hermes_constants import get_hermes_home
    from tools.file_tools import read_file_tool
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(CONFIG, encoding="utf-8")
    assert _is_under_hermes_home(str(home / "config.yaml"))
    assert not _is_under_hermes_home(str(home.parent / "elsewhere" / "config.yaml"))
    result = json.loads(read_file_tool(str(home / "config.yaml"), task_id="t-local-config"))
    assert SECRET not in json.dumps(result), result
