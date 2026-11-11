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


@pytest.fixture
def unknown_sandbox_cwd(monkeypatch):
    """Shape A: the backend's default cwd is ``~`` and the sandbox declares no usable home (its environment reports a
    relative cwd), so no in-sandbox directory is known to anchor a relative path on."""
    monkeypatch.setattr(c.RecordingSandboxProvider, "default_cwd", property(lambda self: "~"), raising=False)
    monkeypatch.setattr(c.RecordingSandboxProvider, "create_environment",
                        lambda self, **kw: c.RecordingSandboxEnvironment(self, kw["sandbox"], "relative-cwd"))


_RELATIVE_CALLS = [
    ("write_file", {"path": "rel.txt", "content": "x\n"}, "'rel.txt'"),
    ("read_file", {"path": "rel.txt"}, "'rel.txt'"),
    ("patch", {"path": "rel.txt", "old_string": "x", "new_string": "y"}, "'rel.txt'"),
    ("search_files", {"pattern": "x", "path": "."}, "'.'"),
]


def test_hosted_relative_path_with_no_known_sandbox_cwd_is_sent_as_written(provider, host_paths, unknown_sandbox_cwd):
    """With no known sandbox directory a relative path is the sandbox's to anchor: every file tool sends it as written,
    never joined to this host's process cwd or HOME, refuses nothing, and looks nothing up on this host."""
    from pathlib import Path

    host_home, host_cwd = host_paths
    results, sent, looked_up = [], [], []
    with c._watch_host(Path(host_cwd)), c._watch_host(Path(host_home)) as touched, c._bind(A):
        # The write guards check this host's fixed approval / deny list on every write, whatever the path: what a
        # hosted write and patch of an ABSOLUTE sandbox path touch here is exactly that list.
        c.call_tool("write_file", {"path": "/workspace/abs.txt", "content": "x\n"}, task_id="t")
        c.call_tool("patch", {"path": "/workspace/abs.txt", "old_string": "x", "new_string": "y"}, task_id="t")
        fixed = set(touched)
        for name, args, _ in _RELATIVE_CALLS:
            seen, opened = len(provider.calls), len(touched)
            results.append(c.call_tool(name, args, task_id="t"))
            sent.append("\n".join(f"{call.command}\n{call.stdin}" for call in provider.calls[seen:]))
            looked_up.append(list(touched[opened:]))
    assert not any(host_cwd in t or "abs.txt" in t for t in fixed), fixed
    for (name, _, as_written), result, commands, host_touches in zip(_RELATIVE_CALLS, results, sent, looked_up):
        tolerated = fixed if name in ("write_file", "patch") else set()
        assert not [t for t in host_touches if t not in tolerated], (name, host_touches)
        assert as_written in commands, (name, commands)
        returned = json.dumps(result)
        assert "Pass an absolute path" not in returned, (name, result)
        for host_path in (host_home, host_cwd):
            assert host_path not in commands and host_path not in returned, (name, host_path, commands, result)


def test_hosted_relative_write_then_read_address_the_same_sandbox_file(provider, host_paths, unknown_sandbox_cwd):
    with c._bind(A):
        seen = len(provider.calls)
        written = c.call_tool("write_file", {"path": "rel.txt", "content": "x\n"}, task_id="t")
        wrote = [call.command for call in provider.calls[seen:]]
        seen = len(provider.calls)
        read = c.call_tool("read_file", {"path": "rel.txt"}, task_id="t")
        reads = [call.command for call in provider.calls[seen:]]
    assert "error" not in written and "error" not in read, (written, read)
    assert any("t='rel.txt'" in command for command in wrote), wrote
    assert reads and all("'rel.txt'" in command for command in reads), reads


def test_hosted_absolute_path_with_no_known_sandbox_cwd_still_works(provider, host_paths, unknown_sandbox_cwd):
    with c._bind(A):
        written = c.call_tool("write_file", {"path": "/workspace/abs.txt", "content": "x\n"}, task_id="t")
        read = c.call_tool("read_file", {"path": "/workspace/abs.txt"}, task_id="t")
    assert written.get("resolved_path") == "/workspace/abs.txt" and "error" not in read, (written, read)
    assert "'/workspace/abs.txt'" in _sent(provider)
    assert not any(host_path in _sent(provider) for host_path in host_paths)
