"""Conformance checks for a terminal backend that serves hosted scopes.

A hosted backend plugin runs these against its own provider; core runs them against
:class:`RecordingSandboxProvider`, a stub that records the identity it was asked for and every call
each sandbox receives (operation, command, stdin, the scope it came with), and runs nothing. Each
check binds hosted scopes through ``agent.runtime_scope.bind_state``, drives the real tools through
the tool registry, then asserts one isolation invariant:

* :func:`check_tools_address_own_sandbox` — terminal, file and code tools of agent A only ever
  address A's sandbox, and B's only B's, whatever task id each call carries;
* :func:`check_foreign_handle_refused` — an environment or long-lived process handle that names
  another agent is refused before the backend sees the call;
* :func:`check_file_tools_never_touch_host` — read, write, patch and search in a hosted scope never
  open, write or list anything on the host (a planted host file stays unread and unchanged, and an
  audit hook sees no ``open``/listing under its directory) and go to the sandbox instead;
* :func:`check_background_process_invisible` — a background process started by A is missing for B;
* :func:`check_host_backend_refused` — with a host backend configured, a hosted scope's tools
  fail closed instead of running here (backend-independent: run it with ``terminal.backend: local``).

The checks read the provider through ``created`` and ``calls`` (a list of :class:`Call`), so a
plugin's own backend can be checked by recording the same two things. They raise ``AssertionError``.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agent.terminal_env_provider import TerminalEnvironmentProvider
from hermes_scope import Principal, Scope
from tools.environments.sandbox_scope import CrossAgentHandleRefused, ProcessRef, ProcessStream, SandboxIdentity

STUB_BACKEND = "recording_sandbox"


def hosted_scope_for(agent: str, *, tenant: str = "acme", user: str = "u1") -> Scope:
    principal = Principal(user)
    return Scope(tenant, agent, principal, owner=principal, can_administer=True)


@dataclass(frozen=True)
class Call:
    sandbox: SandboxIdentity
    op: str
    scope: Optional[Scope]
    command: str = ""
    stdin: str = ""


class RecordingSandboxEnvironment:
    """A sandbox that runs nothing: every ``execute`` is recorded and exits 0 with no output, except
    the background-process launch wrapper, which is answered with a PID. Long-lived processes are an
    in-process echo (a ``cat`` child the stub owns)."""

    uses_host_filesystem = False

    def __init__(self, provider: RecordingSandboxProvider, identity: SandboxIdentity, cwd: str) -> None:
        self.provider, self.identity, self.cwd, self.timeout, self.env = provider, identity, cwd, 60, {}
        self._processes: dict[str, subprocess.Popen] = {}

    def _record(self, op: str, scope: Optional[Scope], command: str = "", stdin: Optional[str] = None) -> None:
        self.provider.calls.append(Call(self.identity, op, scope, command, stdin or ""))

    def execute(self, command: str, cwd: str = "", *, scope: Scope, stdin_data: Optional[str] = None,
                **_: Any) -> dict:
        self._record("execute", scope, command, stdin_data)
        launched = "nohup bash -lc" in command and "echo $! >" in command
        return {"output": "4242\n" if launched else "", "returncode": 0}

    def get_temp_dir(self, *, scope: Scope) -> str:
        return "/tmp"  # no-tmp: ok — the sandbox's own temp dir, never created on the host

    def fetch_realpath(self, remote_path: str, *, scope: Scope) -> Optional[str]:
        self._record("fetch_realpath", scope, remote_path)
        return remote_path

    def cleanup(self, *, scope: Scope) -> None:
        self._record("cleanup", scope)
        for proc in self._processes.values():
            proc.kill()

    def start_process(self, name: str, argv: list[str], *, child_env: Optional[dict] = None,
                      scope: Scope) -> ProcessStream:
        self._record("start_process", scope, " ".join(argv))
        proc = self._processes[name] = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        return self._stream(name, proc)

    def attach_process(self, name: str, *, scope: Scope) -> Optional[ProcessStream]:
        self._record("attach_process", scope, name)
        proc = self._processes.get(name)
        return self._stream(name, proc) if proc is not None and proc.poll() is None else None

    def stop_idle(self, *, scope: Scope) -> None:
        """Suspend the sandbox: its workspace is kept and it resumes on the next call."""
        self._record("stop_idle", scope)

    def stop_process(self, name: str, *, scope: Scope) -> bool:
        self._record("stop_process", scope, name)
        proc = self._processes.pop(name, None)
        if proc is None:
            return False
        proc.kill()
        proc.wait(timeout=5)
        return True

    @staticmethod
    def _stream(name: str, proc: subprocess.Popen) -> ProcessStream:
        assert proc.stdin is not None and proc.stdout is not None
        return ProcessStream(ref=ProcessRef("", name), stdin=proc.stdin, stdout=proc.stdout, poll=proc.poll)


class RecordingSandboxProvider(TerminalEnvironmentProvider):
    """A hosted backend that records every sandbox identity it is asked to create and every call its
    environments receive."""

    serves_hosted_scopes = True

    def __init__(self) -> None:
        self.created: list[SandboxIdentity] = []
        self.calls: list[Call] = []

    @property
    def name(self) -> str:
        return STUB_BACKEND

    def is_available(self) -> bool:
        return True

    def create_environment(self, *, cwd: str, timeout: int, task_id: str = "default", image=None,
                           container_config=None, **kwargs: Any) -> Any:
        sandbox, scope = kwargs["sandbox"], kwargs["scope"]
        assert task_id == sandbox.key, "a hosted sandbox is created by its identity, never a task id"
        assert SandboxIdentity(scope.tenant, scope.agent) == sandbox
        self.created.append(sandbox)
        return RecordingSandboxEnvironment(self, sandbox, "/workspace")


# --------------------------------------------------------------------------------------------------
# Driving the real tools

def _bind(scope: Scope):
    from agent.runtime_scope import bind_state
    return bind_state(scope)


def call_tool(name: str, args: dict, *, task_id: str) -> dict:
    from tools.registry import registry
    result = registry.dispatch(name, args, task_id=task_id)
    return json.loads(result) if isinstance(result, str) else result


def _identity(scope: Scope) -> SandboxIdentity:
    return SandboxIdentity(scope.tenant, scope.agent)


def _env_for(task_id: str) -> Any:
    from tools.terminal_tool_lifecycle import get_active_env
    env = get_active_env(task_id)
    assert env is not None, "the terminal call did not leave an environment"
    return env


def _marker(scope: Scope) -> str:
    return f"mark{uuid.uuid5(uuid.NAMESPACE_URL, f'{scope.tenant}/{scope.agent}').hex[:12]}"


def check_tools_address_own_sandbox(provider: RecordingSandboxProvider, a: Scope, b: Scope) -> None:
    """Every operation A's tools make reaches A's sandbox, and B's reach B's; a task id shared by
    both agents (or naming the other) changes nothing."""
    for scope, task in ((a, "shared-task"), (b, "shared-task"), (a, f"{b.agent}-task")):
        mark = _marker(scope)
        with _bind(scope):
            call_tool("terminal", {"command": f"echo terminal-{mark}"}, task_id=task)
            call_tool("write_file", {"path": "note.txt", "content": f"file-{mark}\n"}, task_id=task)
            call_tool("read_file", {"path": "note.txt"}, task_id=task)
            call_tool("search_files", {"pattern": "file-", "path": "."}, task_id=task)
            call_tool("execute_code", {"code": f"print('code-{mark}')"}, task_id=task)
    assert set(provider.created) == {_identity(a), _identity(b)}, provider.created
    assert len(provider.created) == 2, "an agent got a second sandbox"
    for call in provider.calls:
        if call.scope is not None:
            assert _identity(call.scope) == call.sandbox, f"{call.op} under {call.scope.agent} reached {call.sandbox}"
    for own, other in ((a, b), (b, a)):
        seen = " ".join(c.command + c.stdin for c in provider.calls if c.sandbox == _identity(own))
        assert f"terminal-{_marker(own)}" in seen, f"{own.agent}'s terminal command never reached its sandbox"
        assert _marker(other) not in seen, f"{other.agent}'s data reached {own.agent}'s sandbox"


def check_foreign_handle_refused(provider: RecordingSandboxProvider, a: Scope, b: Scope) -> None:
    """An environment handle or a process reference that names another agent is refused, and the
    backend never sees the call."""
    with _bind(a):
        call_tool("terminal", {"command": "true"}, task_id="t")
        env_a = _env_for("t")
        stream = env_a.start_process("mcp", ["srv"])
    seen = len(provider.calls)
    with _bind(b):
        attempts: list[Callable[[], Any]] = [
            lambda: env_a.execute("id"), lambda: env_a.attach_process(stream.ref),
            lambda: env_a.stop_process("mcp"), lambda: env_a.fetch_realpath("/etc")]
        for attempt in attempts:
            try:
                attempt()
            except CrossAgentHandleRefused:
                continue
            raise AssertionError("an operation through A's environment ran under B's scope")
        call_tool("terminal", {"command": "true"}, task_id="t")
        env_b = _env_for("t")
        try:
            env_b.attach_process(stream.ref)
        except CrossAgentHandleRefused:
            pass
        else:
            raise AssertionError("B's environment accepted a process reference naming A's sandbox")
    assert all(c.sandbox != _identity(a) for c in provider.calls[seen:]), "a refused call reached A's sandbox"
    with _bind(a):
        attached = env_a.attach_process(stream.ref)
        assert attached is not None
        attached.stdin.write(b"ping\n")
        attached.stdin.flush()
        assert attached.stdout.readline() == b"ping\n"
        assert env_a.stop_process(stream.ref) is True
        assert env_a.attach_process(stream.ref) is None


_HOST_OPENS: list[str] = []
_WATCHED: list[str] = []


def _audit(event: str, args: tuple) -> None:
    if _WATCHED and event in ("open", "os.listdir", "os.scandir") and args and isinstance(args[0], (str, Path)):
        if any(str(args[0]).startswith(root) for root in _WATCHED):
            _HOST_OPENS.append(f"{event} {args[0]}")


_audit_installed = False


def _watch_host(root: Path):
    """Record every host open / directory listing under *root* while the context is active (an
    audit hook; Python cannot remove one, so it is installed once and switched by ``_WATCHED``)."""
    import contextlib
    import sys
    global _audit_installed
    if not _audit_installed:
        sys.addaudithook(_audit)
        _audit_installed = True

    @contextlib.contextmanager
    def watching():
        _HOST_OPENS.clear()
        _WATCHED.append(str(root))
        try:
            yield _HOST_OPENS
        finally:
            _WATCHED.remove(str(root))
    return watching()


def _drive_file_tools(scope: Scope, host_dir: Path, path: str) -> list[dict]:
    with _bind(scope):
        return [
            call_tool("read_file", {"path": path}, task_id="t"),
            call_tool("write_file", {"path": path, "content": "sandbox copy\n"}, task_id="t"),
            call_tool("patch", {"path": path, "old_string": "sandbox", "new_string": "remote"}, task_id="t"),
            call_tool("search_files", {"pattern": "HOST-ONLY", "path": str(host_dir)}, task_id="t"),
            call_tool("terminal", {"command": f"cat {path}"}, task_id="t"),
            call_tool("execute_code", {"code": f"print(open({path!r}, encoding='utf-8-sig').read())"}, task_id="t"),
        ]


def check_file_tools_never_touch_host(provider: RecordingSandboxProvider, a: Scope, host_dir: Path) -> None:
    """A file planted on the host at the path the tools are asked for is never opened or listed, and
    stays byte-identical through a hosted read, write, patch, search, terminal and code call at that
    path; the operations go to the sandbox instead."""
    planted = host_dir / "planted.txt"
    marker = f"HOST-ONLY-{uuid.uuid4().hex}"
    planted.write_text(f"{marker}\n", encoding="utf-8")
    before = planted.stat()
    path = str(planted)
    with _watch_host(host_dir) as touched:
        outputs = _drive_file_tools(a, host_dir, path)
    assert not touched, f"a hosted tool touched the host filesystem: {touched}"
    assert marker not in json.dumps(outputs), "a hosted tool returned the host file's content"
    after = planted.stat()
    assert planted.read_text(encoding="utf-8-sig") == f"{marker}\n", "a hosted tool wrote the host file"
    assert (after.st_mtime_ns, after.st_size, after.st_ino) == (before.st_mtime_ns, before.st_size, before.st_ino)
    commands = [c.command for c in provider.calls if c.sandbox == _identity(a) and c.op == "execute"]
    assert any(path in c for c in commands), "the file tools did not address the path in the sandbox"
    assert any("sandbox copy" in c.stdin or "c2FuZGJveCBjb3B5" in c.stdin for c in provider.calls), \
        "write_file's content never reached the sandbox"


def check_host_backend_refused(a: Scope, host_dir: Path) -> None:
    """With a backend that runs on this host configured (``terminal.backend: local``), every tool of a
    hosted scope fails closed: nothing runs, and the planted host file is never opened or changed."""
    planted = host_dir / "planted.txt"
    marker = f"HOST-ONLY-{uuid.uuid4().hex}"
    planted.write_text(f"{marker}\n", encoding="utf-8")
    before = planted.stat()
    with _watch_host(host_dir) as touched:
        outputs = _drive_file_tools(a, host_dir, str(planted))
    assert not touched, f"a hosted tool touched the host filesystem: {touched}"
    assert marker not in json.dumps(outputs), "a hosted tool returned the host file's content"
    assert all(o.get("error") or o.get("success") is False or o.get("exit_code", 0) != 0 for o in outputs), outputs
    after = planted.stat()
    assert (after.st_mtime_ns, after.st_size, after.st_ino) == (before.st_mtime_ns, before.st_size, before.st_ino)


def check_background_process_invisible(provider: RecordingSandboxProvider, a: Scope, b: Scope) -> None:
    """A background process A started is missing for B: not listed, not pollable, not killable,
    not resolvable by id prefix."""
    with _bind(a):
        started = call_tool("terminal", {"command": "sleep 600", "background": True}, task_id="t")
    session_id = started.get("session_id")
    assert session_id, f"the background spawn failed: {started}"
    try:
        with _bind(b):
            listed = call_tool("process_manage", {"action": "list"}, task_id="t")
            assert session_id not in json.dumps(listed), "B's list shows A's process"
            for action in ("poll", "log", "kill"):
                result = call_tool("process_manage", {"action": action, "session_id": session_id}, task_id="t")
                assert result.get("status") == "not_found", f"B could {action} A's process: {result}"
            prefix = session_id.removeprefix("proc_")[:6]
            result = call_tool("process_manage", {"action": "poll", "session_id": prefix}, task_id="t")
            assert result.get("status") == "not_found", f"B resolved A's process by prefix: {result}"
        with _bind(a):
            assert session_id in json.dumps(call_tool("process_manage", {"action": "list"}, task_id="t"))
    finally:
        with _bind(a):
            call_tool("process_manage", {"action": "kill", "session_id": session_id}, task_id="t")


def _age_sandbox(a: Scope) -> str:
    from tools import terminal_tool
    key = _identity(a).key
    terminal_tool._last_activity[key] = 0.0  # idle since the epoch
    return key


def check_idle_sandbox_kept_while_working(provider: RecordingSandboxProvider, a: Scope) -> None:
    """A sandbox whose agent has a background process still running is never reaped as idle,
    whatever task id that process was started under."""
    from tools import terminal_tool
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs
    with _bind(a):
        started = call_tool("terminal", {"command": "sleep 600", "background": True}, task_id="t")
    assert started.get("session_id"), f"the background spawn failed: {started}"
    key = _age_sandbox(a)
    seen = len(provider.calls)
    try:
        _cleanup_inactive_envs(lifetime_seconds=300)
        assert key in terminal_tool._active_environments, "a sandbox with running work was reaped"
        assert not [c.op for c in provider.calls[seen:] if c.op in ("cleanup", "stop_idle")]
    finally:
        with _bind(a):
            call_tool("process_manage", {"action": "kill", "session_id": started["session_id"]}, task_id="t")


def check_idle_sandbox_suspended_not_destroyed(provider: RecordingSandboxProvider, a: Scope) -> None:
    """An idle sandbox is suspended (``stop_idle``), not destroyed, and the call reaches the backend
    under its owning agent's scope though the reaper runs with none bound."""
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs
    with _bind(a):
        call_tool("terminal", {"command": "true"}, task_id="t")
    _age_sandbox(a)
    seen = len(provider.calls)
    _cleanup_inactive_envs(lifetime_seconds=300)
    lifecycle = [c for c in provider.calls[seen:] if c.op in ("cleanup", "stop_idle")]
    assert [c.op for c in lifecycle] == ["stop_idle"], f"the idle sandbox was not suspended: {lifecycle}"
    assert lifecycle[0].scope == a and lifecycle[0].sandbox == _identity(a)


def _drain_all(registry: Any) -> list[dict]:
    events = []
    while not registry.completion_queue.empty():
        events.append(registry.completion_queue.get_nowait())
    return events


def check_process_notifications_private(provider: RecordingSandboxProvider, a: Scope, b: Scope) -> None:
    """Completion, watch and heartbeat events of A's background process carry A's sandbox identity,
    and a drain under B (even with the same session key, or no filter at all) neither delivers,
    consumes, suppresses nor drops them: they stay queued, and A's own drain receives them. Its live
    output never reaches the (scope-less) UI sink."""
    from tools.process_registry import ProcessSession, process_registry as registry
    key = _identity(a).key
    secret = f"PRIVATE-{_marker(a)}"
    session = ProcessSession(
        id=f"proc_{uuid.uuid4().hex[:12]}", command=f"echo {secret}", task_id=key, owner_task_id="t",
        session_key="shared", sandbox_key=key, notify_on_complete=True, exited=True,
        output_buffer=f"{secret}\n", watch_patterns=["PRIVATE"])
    registry._completions_restored = True
    _drain_all(registry)
    sunk: list[str] = []
    previous_sink, registry.on_output = registry.on_output, lambda s, chunk: sunk.append(chunk)
    try:
        with _bind(a):
            registry._publish_finished(session, True)
            session.exited = False
            registry._check_watch_patterns(session, f"{secret}\n")
            session._heartbeat_last, session.total_output_chars, session._heartbeat_total_at_last = 0.0, 10, 0
            registry._emit_heartbeat(session, 1.0)
            registry._emit_output(session, f"{secret}\n")
        queued = list(registry.completion_queue.queue)
        assert {e.get("type") for e in queued} >= {"completion", "watch_match", "heartbeat"}, queued
        lost = [e for e in queued if e.get("sandbox_key") != key]
        assert not lost, f"an event lost its producer identity: {lost}"
        assert not sunk, "a hosted process's live output reached a UI sink"
        from gateway.run import _drain_gateway_watch_events
        with _bind(b):
            for drain in (lambda: registry.drain_notifications(session_key="shared"),
                          lambda: registry.drain_notifications(),
                          lambda: _drain_gateway_watch_events(registry.completion_queue)):
                assert secret not in json.dumps(drain(), default=str), "B received A's process event"
        assert registry.drain_notifications() == [], "an unscoped drain received a hosted agent's event"
        assert len(registry.completion_queue.queue) == len(queued), "a foreign drain dropped A's events"
        with _bind(a):
            mine = registry.drain_notifications(session_key="shared")
        assert secret in json.dumps(mine, default=str) and registry.completion_queue.empty()
    finally:
        registry.on_output = previous_sink
        _drain_all(registry)


def checks() -> Iterator[tuple[str, Callable[..., None]]]:
    yield from (
        ("tools_address_own_sandbox", check_tools_address_own_sandbox),
        ("foreign_handle_refused", check_foreign_handle_refused),
        ("file_tools_never_touch_host", check_file_tools_never_touch_host),
        ("background_process_invisible", check_background_process_invisible),
        ("process_notifications_private", check_process_notifications_private),
        ("idle_sandbox_kept_while_working", lambda p, a, b: check_idle_sandbox_kept_while_working(p, a)),
        ("idle_sandbox_suspended_not_destroyed", lambda p, a, b: check_idle_sandbox_suspended_not_destroyed(p, a)),
    )
