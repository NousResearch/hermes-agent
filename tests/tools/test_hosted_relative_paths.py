"""A hosted relative path is never anchored on this host: neither a ``~`` TERMINAL_CWD expanded against this
host's HOME nor the process working directory is sent to the sandbox or returned to the model. Outside a hosted
scope both keep their local meaning."""

from __future__ import annotations

import json
import posixpath

import pytest

from tests.tools.test_sandbox_scope_conformance import provider as _conformance_provider
from tests.tools.test_sandbox_scope_conformance import A
from tools.environments import sandbox_conformance as c

provider = _conformance_provider  # the recording-sandbox fixture, by its name


@pytest.fixture
def host_paths(monkeypatch, tmp_path):
    """A host HOME outside ``/home`` and ``/Users`` (a service account, a container data dir) and a host process
    cwd, neither of which a hosted call may name."""
    home, cwd = tmp_path / "hosthome-zz9", tmp_path / "host-proc-cwd"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    return str(home), str(cwd)


def _record_cwds(monkeypatch) -> tuple[list[str], list[tuple[str, str]]]:
    """The cwd each sandbox was created with, and ``(command, cwd)`` of each command it ran."""
    created: list[str] = []
    ran: list[tuple[str, str]] = []
    create, execute = c.RecordingSandboxProvider.create_environment, c.RecordingSandboxEnvironment.execute
    monkeypatch.setattr(c.RecordingSandboxProvider, "create_environment",
                        lambda self, **kw: created.append(kw["cwd"]) or create(self, **kw))
    monkeypatch.setattr(c.RecordingSandboxEnvironment, "execute",
                        lambda self, command, cwd="", **kw: ran.append((command, cwd)) or execute(self, command, cwd, **kw))
    return created, ran


def _sent(provider) -> str:
    return "\n".join(f"{call.command}\n{call.stdin}" for call in provider.calls)


def test_hosted_terminal_cwd_tilde_is_the_sandbox_home_never_this_hosts(provider, monkeypatch, host_paths):
    """``TERMINAL_CWD='~'`` names the sandbox's home. The recording backend declares none, so the unchanged container
    sanity check discards the unexpanded ``~`` for the backend default, and the terminal and the file tools both
    work there."""
    host_home, host_cwd = host_paths
    monkeypatch.setenv("TERMINAL_CWD", "~")
    created, ran = _record_cwds(monkeypatch)
    with c._bind(A):
        written = c.call_tool("write_file", {"path": "rel.txt", "content": "x\n"}, task_id="t")
        terminal = c.call_tool("terminal", {"command": "pwd"}, task_id="t")
    assert created == ["/root"], created
    assert written.get("resolved_path") == "/root/rel.txt", written
    assert [cwd for command, cwd in ran if command == "pwd"] == [posixpath.dirname(written["resolved_path"])], ran
    sent, returned = _sent(provider), json.dumps([written, terminal])
    for host_path in (host_home, host_cwd):
        assert host_path not in sent and host_path not in returned, (host_path, sent, returned)


@pytest.mark.parametrize(("default_cwd", "expected"), [("~", "/workspace/rel.txt"), (None, "/root/rel.txt")])
def test_hosted_relative_path_anchors_in_the_sandbox_default(provider, monkeypatch, host_paths, default_cwd, expected):
    """Controls: a ``~`` default the sandbox resolved (``/workspace``), and no declared default (``/root``)."""
    if default_cwd is not None:
        monkeypatch.setattr(c.RecordingSandboxProvider, "default_cwd", property(lambda self: default_cwd),
                            raising=False)
    with c._bind(A):
        written = c.call_tool("write_file", {"path": "rel.txt", "content": "x\n"}, task_id="t")
    assert written.get("resolved_path") == expected, written
    assert not any(host_path in _sent(provider) for host_path in host_paths)


def test_local_terminal_cwd_tilde_still_expands_to_this_hosts_home(monkeypatch, host_paths):
    """With no scope bound a ``~`` TERMINAL_CWD is this host's HOME, as it always was."""
    from tools.terminal_tool import _resolve_config_cwd

    host_home, _ = host_paths
    monkeypatch.setenv("TERMINAL_CWD", "~")
    assert _resolve_config_cwd("docker", False) == (host_home, None)


def test_local_relative_path_still_anchors_on_the_process_cwd(monkeypatch, host_paths):
    """With no scope bound and no configured cwd a relative write lands in the process cwd, as it always did."""
    import os

    import model_tools  # registers the tools dispatched through the registry
    from tools import file_tools, terminal_tool

    _, host_cwd = host_paths
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(file_tools, "_file_ops_cache", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    written = c.call_tool("write_file", {"path": "rel.txt", "content": "x\n"}, task_id="t")
    expected = os.path.join(os.path.realpath(host_cwd), "rel.txt")
    assert written.get("resolved_path") == expected, written
    with open(expected, encoding="utf-8") as fh:
        assert fh.read() == "x\n"
