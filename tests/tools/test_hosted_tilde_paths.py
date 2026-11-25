"""A hosted ``~`` path names the agent's sandbox home (the home its backend declared), never this
host's HOME: a write and a read of the same ``~/x`` address one sandbox path, and nothing the tools
send or return carries the host home. With no declared home a ``~`` path is refused before anything
is sent. A local scope keeps the profile-home expansion."""

from __future__ import annotations

import json
import os

import pytest

from tests.tools.test_sandbox_scope_conformance import A, host_dir as _conformance_host_dir
from tests.tools.test_sandbox_scope_conformance import provider as _conformance_provider
from tools.environments import sandbox_conformance as c

provider = _conformance_provider  # the recording-sandbox fixture, by its name
host_dir = _conformance_host_dir

SANDBOX_HOME = "/home/tenant-sandbox"


@pytest.fixture
def host_home(monkeypatch, tmp_path):
    home = tmp_path / "hosthome-zz9"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return str(home)


@pytest.fixture
def declared_home(monkeypatch):
    monkeypatch.setattr(c.RecordingSandboxProvider, "default_cwd", property(lambda self: SANDBOX_HOME),
                        raising=False)
    return SANDBOX_HOME


def _sent(provider) -> str:
    return "\n".join(f"{call.command}\n{call.stdin}" for call in provider.calls)


def test_hosted_write_then_read_of_a_tilde_path_address_the_sandbox_home(provider, host_home, declared_home):
    with c._bind(A):
        written = c.call_tool("write_file", {"path": "~/notes.txt", "content": "x\n"}, task_id="t")
        read = c.call_tool("read_file", {"path": "~/notes.txt"}, task_id="t")
        searched = c.call_tool("search_files", {"pattern": "x", "path": "~/src"}, task_id="t")
        sensitive = c.call_tool("write_file", {"path": "~/../../etc/cron.d/x", "content": "x\n"}, task_id="t")
    target = f"{declared_home}/notes.txt"
    assert written.get("resolved_path") == target, written
    reads = [call.command for call in provider.calls if call.op == "execute" and "wc -c" in call.command]
    assert reads and all(target in cmd and "~/notes.txt" not in cmd for cmd in reads), reads
    assert any(f"{declared_home}/src" in call.command for call in provider.calls), "search did not use the home"
    assert host_home not in _sent(provider), "a hosted file tool sent the host HOME into the sandbox"
    assert host_home not in json.dumps([written, read, searched])
    assert "sensitive system path" in sensitive.get("error", ""), sensitive  # the guards still judge the path


def test_hosted_patch_of_a_tilde_path_addresses_the_sandbox_home(provider, host_home, declared_home):
    from tools.hosted_tool_policy import HOSTED_TOOL_ALLOWLIST
    assert "patch" in HOSTED_TOOL_ALLOWLIST
    with c._bind(A):
        patched = c.call_tool("patch", {"path": "~/x", "old_string": "a", "new_string": "b"}, task_id="t")
        v4a = c.call_tool("patch", {"mode": "patch", "patch": (
            "*** Begin Patch\n*** Add File: ~/added.txt\n+hello\n*** End Patch\n")}, task_id="t")
    sent = _sent(provider)
    assert f"{declared_home}/x" in sent and f"{declared_home}/added.txt" in sent, sent
    assert host_home not in sent and host_home not in json.dumps([patched, v4a])


def test_hosted_tilde_with_no_declared_home_is_refused_and_nothing_is_sent(provider, host_home):
    with c._bind(A):
        c.call_tool("terminal", {"command": "true"}, task_id="t")  # the sandbox is up, with no home
        seen = len(provider.calls)
        results = [
            c.call_tool("write_file", {"path": "~/notes.txt", "content": "x\n"}, task_id="t"),
            c.call_tool("read_file", {"path": "~/notes.txt"}, task_id="t"),
            c.call_tool("patch", {"path": "~/notes.txt", "old_string": "a", "new_string": "b"}, task_id="t"),
            c.call_tool("read_file", {"path": "~root/notes.txt"}, task_id="t"),
            c.call_tool("search_files", {"pattern": "x", "path": "~"}, task_id="t"),
            c.call_tool("patch", {"mode": "patch", "patch": (
                "*** Begin Patch\n*** Add File: ~/added.txt\n+hello\n*** End Patch\n")}, task_id="t"),
        ]
    for result in results:
        assert "absolute path" in str(result.get("error", "")), result
    assert "declared no home" in results[0]["error"] and "~root" in results[3]["error"]
    assert provider.calls[seen:] == [], "a refused ~ path reached the sandbox"
    assert host_home not in json.dumps(results)


@pytest.mark.platforms("posix")
def test_hosted_tilde_user_never_consults_the_host_passwd_database(provider, monkeypatch, declared_home):
    import pwd

    expanduser, getpwnam = os.path.expanduser, pwd.getpwnam

    def refuse(found: bool):
        if found:
            raise AssertionError("a hosted ~user path consulted this host's passwd database")
    monkeypatch.setattr(os.path, "expanduser", lambda p: refuse("~root" in os.fspath(p)) or expanduser(p))
    monkeypatch.setattr(pwd, "getpwnam", lambda name: refuse(name == "root") or getpwnam(name))
    with c._bind(A):
        results = [c.call_tool("write_file", {"path": "~root/x", "content": "x\n"}, task_id="t"),
                   c.call_tool("read_file", {"path": "~root/x"}, task_id="t"),
                   c.call_tool("patch", {"path": "~root/x", "old_string": "a", "new_string": "b"}, task_id="t")]
    for result in results:
        assert "'~user' paths are not supported" in str(result.get("error", "")), result
    assert not [call for call in provider.calls if "~root" in call.command]


def _record_v4a(monkeypatch) -> list[str]:
    """The V4A text each hosted patch hands the sandbox's file operations."""
    from tools.file_operations import ShellFileOperations
    sent: list[str] = []
    original = ShellFileOperations.patch_v4a
    monkeypatch.setattr(ShellFileOperations, "patch_v4a", lambda self, text: sent.append(text) or original(self, text))
    return sent


def _v4a(header: str, body: str = "") -> str:
    return f"*** Begin Patch\n*** {header}\n{body}*** End Patch\n"


_V4A_AS_WRITTEN = [
    _v4a("Update File: rel.txt", "@@\n-old\n+new\n"), _v4a("Add File: sub/added.txt", "+hi\n"),
    _v4a("Delete File: del.txt"), _v4a("Move File: src.txt -> dst.txt"),
    _v4a("Update File: /workspace/abs.txt", "@@\n-old\n+new\n"), _v4a("Add File: /workspace/added.txt", "+hi\n"),
    _v4a("Delete File: /workspace/del.txt"), _v4a("Move File: /workspace/src.txt -> /srv/dst.txt"),
]


@pytest.mark.parametrize("home", ["none", "declared", "tilde_default_cwd"])
def test_hosted_v4a_relative_and_absolute_headers_reach_the_sandbox_as_written(provider, monkeypatch, tmp_path, home):
    """Only a ``~`` header names the sandbox's home; every other header is the sandbox's to anchor, as it
    always was, and no host directory is put in its place."""
    if home == "declared":
        monkeypatch.setattr(c.RecordingSandboxProvider, "default_cwd", property(lambda self: SANDBOX_HOME),
                            raising=False)
    elif home == "tilde_default_cwd":
        monkeypatch.setattr(c.RecordingSandboxProvider, "default_cwd", property(lambda self: "~"), raising=False)
        monkeypatch.setattr(c.RecordingSandboxProvider, "create_environment",
                            lambda self, **kw: c.RecordingSandboxEnvironment(self, kw["sandbox"], "relative-cwd"))
        (tmp_path / "host-cwd").mkdir()
        monkeypatch.chdir(tmp_path / "host-cwd")
    sent = _record_v4a(monkeypatch)
    with c._bind(A):
        results = [c.call_tool("patch", {"mode": "patch", "patch": text}, task_id="t") for text in _V4A_AS_WRITTEN]
    assert sent == _V4A_AS_WRITTEN, (sent, results)
    assert os.getcwd() not in _sent(provider)


def test_hosted_v4a_tilde_headers_name_the_sandbox_home(provider, monkeypatch, host_home, declared_home):
    sent = _record_v4a(monkeypatch)
    patches = [_v4a("Update File: ~/t.txt", "@@\n-old\n+new\n"), _v4a("Delete File: ~/del.txt"),
               _v4a("Move File: ~/src.txt -> ~/dst.txt"), _v4a("Move File: ~/a.txt -> rel.txt")]
    with c._bind(A):
        results = [c.call_tool("patch", {"mode": "patch", "patch": text}, task_id="t") for text in patches]
    assert sent == [_v4a("Update File: /home/tenant-sandbox/t.txt", "@@\n-old\n+new\n"),
                    _v4a("Delete File: /home/tenant-sandbox/del.txt"),
                    _v4a("Move File: /home/tenant-sandbox/src.txt -> /home/tenant-sandbox/dst.txt"),
                    _v4a("Move File: /home/tenant-sandbox/a.txt -> rel.txt")], results
    assert host_home not in _sent(provider) and host_home not in json.dumps(results)


def test_hosted_v4a_tilde_headers_with_no_home_or_a_user_are_refused(provider, host_home):
    with c._bind(A):
        c.call_tool("terminal", {"command": "true"}, task_id="t")  # the sandbox is up, with no home
        seen = len(provider.calls)
        results = [c.call_tool("patch", {"mode": "patch", "patch": text}, task_id="t") for text in (
            _v4a("Update File: ~/t.txt", "@@\n-old\n+new\n"), _v4a("Delete File: ~/del.txt"),
            _v4a("Move File: rel.txt -> ~/dst.txt"), _v4a("Update File: ~root/t.txt", "@@\n-old\n+new\n"))]
    for result in results:
        assert "absolute path" in str(result.get("error", "")), result
    assert provider.calls[seen:] == [], "a refused ~ header reached the sandbox"


def _tilde_cwd(monkeypatch, anchor: str) -> None:
    """A ``~`` working directory from *anchor*; a task override goes through the real registration, which
    also records it as the session's cwd."""
    from tools import terminal_tool
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    if anchor == "terminal_cwd":
        monkeypatch.setenv("TERMINAL_CWD", "~")
    elif anchor == "padded_session_cwd":
        with c._bind(A):
            terminal_tool.record_session_cwd("t", " ~")  # recorded as given, unstripped
    else:
        with c._bind(A):
            terminal_tool.register_task_env_overrides("t", {"cwd": "~"})
            assert terminal_tool.get_session_cwd("t") == "~"


@pytest.mark.parametrize("anchor", ["terminal_cwd", "task_override", "padded_session_cwd"])
def test_hosted_tilde_cwd_with_no_declared_home_is_no_anchor(provider, monkeypatch, host_dir, host_home, anchor):
    """A ``~`` working directory the sandbox gives no meaning to anchors nothing: a plain relative path
    works as before, and nothing about it is looked up on this host."""
    _tilde_cwd(monkeypatch, anchor)
    (host_dir.parent / "elsewhere").mkdir()
    (host_dir / "workspace-link").symlink_to(host_dir.parent / "elsewhere", target_is_directory=True)
    monkeypatch.chdir(host_dir)
    with c._watch_host(host_dir) as touched, c._bind(A):
        results = [c.call_tool("read_file", {"path": "notes.txt"}, task_id="t"),
                   c.call_tool("write_file", {"path": "notes.txt", "content": "x\n"}, task_id="t"),
                   c.call_tool("write_file", {"path": "workspace-link/notes.txt", "content": "x\n"}, task_id="t"),
                   c.call_tool("patch", {"path": "notes.txt", "old_string": "a", "new_string": "b"}, task_id="t"),
                   c.call_tool("patch", {"mode": "patch", "patch": _v4a("Add File: added.txt", "+hi\n")}, task_id="t"),
                   c.call_tool("search_files", {"pattern": "x", "path": "."}, task_id="t")]
        touched = list(touched)
    assert not touched, f"a hosted tool touched the host filesystem: {touched}"
    for result in results:
        assert "~" not in str(result.get("error", "")) and "declared no home" not in json.dumps(result), result
    if anchor == "terminal_cwd":
        return  # where the terminal config puts a ``~`` TERMINAL_CWD is its own concern
    for written in results[1:3]:
        resolved = written.get("resolved_path")
        assert resolved and resolved.startswith("/"), written
        assert not resolved.startswith((str(host_dir), host_home)), written
    assert str(host_dir) not in _sent(provider) and host_home not in _sent(provider)


@pytest.mark.parametrize("anchor", ["terminal_cwd", "task_override"])
def test_hosted_tilde_cwd_with_a_declared_home_anchors_under_it(provider, monkeypatch, host_dir, host_home,
                                                               declared_home, anchor):
    _tilde_cwd(monkeypatch, anchor)
    monkeypatch.chdir(host_dir)
    with c._watch_host(host_dir) as touched, c._bind(A):
        read = c.call_tool("read_file", {"path": "notes.txt"}, task_id="t")
        written = c.call_tool("write_file", {"path": "sub/notes.txt", "content": "x\n"}, task_id="t")
        patched = c.call_tool("patch", {"path": "p.txt", "old_string": "a", "new_string": "b"}, task_id="t")
        touched = list(touched)
    assert not touched, f"a hosted tool touched the host filesystem: {touched}"
    assert written.get("resolved_path") == f"{declared_home}/sub/notes.txt", written
    sent = _sent(provider)
    assert f"{declared_home}/p.txt" in sent and "~" not in json.dumps([read, written, patched]), (sent, read, patched)
    assert host_home not in sent and str(host_dir) not in sent


def test_local_recorded_tilde_cwd_still_anchors_at_the_profile_home(monkeypatch, host_home):
    """Outside a hosted scope a recorded ``~`` session cwd is returned as recorded and expands to this host's
    profile home, as it always did."""
    from pathlib import Path
    from tools import file_tools_paths, terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr("hermes_constants.get_subprocess_home", lambda: host_home)
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    terminal_tool.record_session_cwd("t", "~")
    assert file_tools_paths._authoritative_workspace_root("t") == "~"
    assert file_tools_paths._resolve_path_for_task("notes.txt", "t") == Path(host_home).resolve() / "notes.txt"


def test_protected_instruction_check_never_resolves_a_hosted_path_on_the_host(provider, monkeypatch, host_dir):
    from tools import file_tools_write_guards as guards
    (host_dir / "workspace-link").symlink_to(host_dir.parent, target_is_directory=True)

    def unresolvable(*_args, **_kwargs):
        raise ValueError("unresolvable")
    with c._bind(A):
        no_home = guards._protected_instruction_reason("~/AGENTS.md", "t", enabled=True, extra_patterns=[])
        monkeypatch.setattr(guards, "_resolve_path_for_task", unresolvable)
        with c._watch_host(host_dir) as touched:
            linked = guards._protected_instruction_reason(str(host_dir / "workspace-link" / "AGENTS.md"), "t",
                                                          enabled=True, extra_patterns=[])
            touched = list(touched)
    assert no_home == "AGENTS.md" and linked == "AGENTS.md"
    assert not touched, f"the protected-file check resolved a hosted path on this host: {touched}"


def test_hosted_tilde_starts_the_sandbox_under_the_callers_task(provider, monkeypatch, declared_home):
    """Learning the sandbox's home brings up the CALLER's sandbox, with that task's overrides, and only for a
    path that starts with ``~``."""
    from tools import terminal_tool
    created = []
    original = c.RecordingSandboxProvider.create_environment
    monkeypatch.setattr(c.RecordingSandboxProvider, "create_environment",
                        lambda self, **kw: created.append(kw) or original(self, **kw))
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    with c._bind(A):
        terminal_tool.register_task_env_overrides("sess-1", {"recording_sandbox_image": "custom-image:v2",
                                                             "cwd": "/custom/work"})
        try:
            refused = c.call_tool("write_file", {"path": "/etc/passwd", "content": "x\n"}, task_id="sess-1")
            assert "sensitive system path" in refused.get("error", "") and created == [], created
            written = c.call_tool("write_file", {"path": "~/notes.txt", "content": "x\n"}, task_id="sess-1")
        finally:
            terminal_tool.clear_task_env_overrides("sess-1")
    assert written.get("resolved_path") == f"{declared_home}/notes.txt", written
    assert created and created[0]["image"] == "custom-image:v2" and created[0]["cwd"] == "/custom/work", created


def test_local_scope_tilde_still_expands_to_the_profile_home(monkeypatch, host_home):
    import contextlib
    from agent.runtime_scope import bind_state
    from hermes_constants import get_hermes_home
    from hermes_scope import Principal, Scope
    from tools.file_tools_paths import _expand_tilde

    monkeypatch.setattr("hermes_constants.get_subprocess_home", lambda: host_home)
    for scope in (None, Scope("local", str(get_hermes_home()), Principal("local"))):
        with bind_state(scope) if scope is not None else contextlib.nullcontext():
            assert _expand_tilde("~/notes.txt") == f"{host_home}/notes.txt"
            assert _expand_tilde("~") == host_home
