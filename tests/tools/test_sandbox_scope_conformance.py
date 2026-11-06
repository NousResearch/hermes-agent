"""Sandbox conformance: a hosted scope's tools address only that agent's sandbox.

The checks live in ``tools/environments/sandbox_conformance.py`` so a hosted backend plugin runs the
same ones against its own provider; here they run against the recording stub, through the real tool
registry, with hosted scopes bound by ``agent.runtime_scope.bind_state``. Local behaviour is covered
by the existing terminal/file/process suites, which run with no hosted scope bound.
"""

from __future__ import annotations

import json

import pytest

from agent import terminal_env_registry as reg
from hermes_scope import Principal, Scope
from tools.environments import sandbox_conformance as conformance
from tools.environments.sandbox_scope import (
    CrossAgentHandleRefused, HostedSandboxRequired, SandboxIdentity, ScopedEnvironment, sandbox_identity,
)

A = Scope("acme", "agent-a", Principal("u1"), owner=Principal("u1"), can_administer=True)
B = Scope("acme", "agent-b", Principal("u1"), owner=Principal("u1"), can_administer=True)
OTHER_TENANT_A = Scope("globex", "agent-a", Principal("u1"), owner=Principal("u1"), can_administer=True)


@pytest.fixture
def provider(monkeypatch):
    import model_tools  # registers the tools the checks dispatch through the registry
    from tools import file_tools, process_registry, terminal_tool

    reg._reset_for_tests()
    stub = conformance.RecordingSandboxProvider()
    reg.register_provider(stub)
    monkeypatch.setenv("TERMINAL_ENV", conformance.STUB_BACKEND)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(file_tools, "_file_ops_cache", {})
    monkeypatch.setattr(process_registry.process_registry, "_running", {})
    monkeypatch.setattr(process_registry.process_registry, "_finished", {})
    yield stub
    reg._reset_for_tests()


@pytest.fixture
def host_dir(tmp_path):
    """A workspace-looking host directory, apart from the test's HERMES_HOME (Hermes reading its
    own state is not a tool touching the host)."""
    path = tmp_path / "host-workspace"
    path.mkdir()
    return path


@pytest.mark.parametrize("name", [name for name, _ in conformance.checks()])
def test_conformance(name, provider, host_dir):
    check = dict(conformance.checks())[name]
    if name in ("file_tools_never_touch_host", "media_never_read_on_host"):
        check(provider, A, host_dir)
    else:
        check(provider, A, B)


def test_identity_is_tenant_and_agent_only():
    turn = A.for_turn("session-1", "turn-1", fence=__import__("hermes_scope").Fence(1, 1))
    assert sandbox_identity(turn) == sandbox_identity(A) == SandboxIdentity("acme", "agent-a")
    assert sandbox_identity(OTHER_TENANT_A).key != sandbox_identity(A).key
    # Percent-quoting keeps the key injective across tenant/agent boundaries.
    assert SandboxIdentity("a/b", "c").key != SandboxIdentity("a", "b/c").key


def test_same_agent_in_another_tenant_gets_its_own_sandbox(provider):
    conformance.check_tools_address_own_sandbox(provider, A, OTHER_TENANT_A)


def test_a_hosted_scope_with_a_host_backend_runs_nothing_here(provider, monkeypatch, host_dir):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    conformance.check_host_backend_refused(A, host_dir)
    assert provider.created == []


def test_a_builtin_backend_is_refused_in_a_hosted_scope(monkeypatch, provider):
    from agent.runtime_scope import bind_state
    from tools.terminal_tool_backends import _create_environment

    with bind_state(A):
        with pytest.raises(HostedSandboxRequired):
            _create_environment("local", image="", cwd="/", timeout=10)


def test_an_environment_with_no_scope_bound_is_refused(provider):
    from agent.runtime_scope import bind_state

    with bind_state(A):
        conformance.call_tool("terminal", {"command": "true"}, task_id="t")
        env = conformance._env_for("t")
    assert isinstance(env, ScopedEnvironment)
    with pytest.raises(CrossAgentHandleRefused):
        env.execute("id")
    env.cleanup()  # the idle reaper runs with no scope bound and may still release it


def test_local_long_lived_process_round_trip(tmp_path):
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    stream = env.start_process("echo", ["cat"])
    stream.stdin.write(b"hello\n")
    stream.stdin.flush()
    assert stream.stdout.readline() == b"hello\n"
    with pytest.raises(ValueError, match="already running"):
        env.start_process("echo", ["cat"])
    attached = env.attach_process("echo")
    assert attached is not None and attached.stdout is stream.stdout
    assert env.stop_process("echo") is True
    assert env.attach_process("echo") is None and env.stop_process("echo") is False


def test_a_cached_environment_of_another_agent_is_refused_not_used(provider):
    """Even if another agent's sandbox ended up under this agent's cache key, it is refused and never
    reached: the guard checks the environment's own identity, not where it was found."""
    from agent.runtime_scope import bind_state
    from tools import terminal_tool

    with bind_state(B):
        conformance.call_tool("terminal", {"command": "true"}, task_id="t")
        env_b = conformance._env_for("t")
    from tools.terminal_tool_lifecycle import get_active_env

    terminal_tool._active_environments[sandbox_identity(A).key] = env_b
    seen = len(provider.calls)
    with bind_state(A):
        with pytest.raises(CrossAgentHandleRefused):
            get_active_env("t")  # refused at the lookup, before any caller holds it
        result = conformance.call_tool("terminal", {"command": "echo leak"}, task_id="t")
    assert "leak" not in result.get("output", "")
    assert all(call.sandbox != sandbox_identity(B) for call in provider.calls[seen:])


def test_another_agents_process_never_makes_a_prefix_ambiguous(provider):
    """Process ids resolve by unique prefix. Another agent's process sharing the prefix is invisible:
    it neither resolves nor turns this agent's own lookup into an ambiguous miss (which would reveal
    that it exists)."""
    from agent.runtime_scope import bind_state
    from tools.process_registry import ProcessSession, process_registry

    own = ProcessSession(id="proc_abcdef000001", command="own", sandbox_key=sandbox_identity(B).key, exited=True)
    other = ProcessSession(id="proc_abcdef000002", command="other", sandbox_key=sandbox_identity(A).key, exited=True)
    process_registry._finished.update({own.id: own, other.id: other})
    with bind_state(B):
        assert process_registry.get("abcdef") is own
        assert process_registry.get(other.id) is None
    assert process_registry.get("abcdef") is None  # a local activity sees neither agent's


@pytest.mark.platforms("posix")
def test_local_long_lived_process_dies_with_hermes_until_stopped(tmp_path, monkeypatch):
    """A long-lived local process is owned (its group is covered by the owner-death supervisor, so a
    SIGKILLed Hermes takes it down) from start until ``stop_process`` releases it."""
    from hermes_cli import process_identity
    from tools.environments.local import LocalEnvironment

    owned, released = [], []
    monkeypatch.setattr(process_identity, "own_process_groups", lambda pgids: owned.extend(pgids))
    monkeypatch.setattr(process_identity, "release_process_group_once_empty",
                        lambda pgid, **kw: released.append(pgid) or True)
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    stream = env.start_process("cat", ["cat"])
    pid = stream.poll and env._long_lived()["cat"].pid
    assert owned == [pid] and released == []
    assert env.stop_process("cat") is True
    assert released == [pid]


def test_every_hosted_operation_receives_the_bound_scope():
    """A backend whose operations take ``**kwargs`` still receives the scope; one that cannot take it
    at all is refused when its sandbox is created, never silently handed no authority."""
    from agent.runtime_scope import bind_state
    from tools.environments.sandbox_scope import require_hosted_interface

    class KwargsBackend:
        def execute(self, command, **kwargs):
            return kwargs

        def get_temp_dir(self, **kwargs):
            return kwargs

        cleanup = start_process = attach_process = stop_process = stop_idle = execute

    env = ScopedEnvironment(KwargsBackend(), sandbox_identity(A), owner=A)
    with bind_state(A):
        assert env.execute("true")["scope"] == A
        assert env.get_temp_dir()["scope"] == A
    require_hosted_interface(KwargsBackend(), "kwargs")

    class Blind(KwargsBackend):
        def execute(self, command, cwd=""):
            return {}

    with pytest.raises(HostedSandboxRequired, match="execute must accept scope"):
        require_hosted_interface(Blind(), "blind")

    class Partial:
        def execute(self, command, *, scope):
            return {}

    with pytest.raises(HostedSandboxRequired, match="missing cleanup"):
        require_hosted_interface(Partial(), "partial")


def test_a_hosted_provider_without_the_interface_is_refused_at_creation(provider, monkeypatch):
    class NoScope:
        def execute(self, command, cwd=""):
            raise AssertionError("an environment without the interface was used")

    monkeypatch.setattr(provider, "create_environment", lambda **kw: NoScope())
    from agent.runtime_scope import bind_state
    with bind_state(A):
        result = conformance.call_tool("terminal", {"command": "id"}, task_id="t")
    assert "hosted operation interface" in json.dumps(result)


def test_running_background_work_keeps_its_sandbox_alive(provider):
    conformance.check_idle_sandbox_kept_while_working(provider, A)


def test_an_idle_sandbox_is_suspended_under_its_owners_scope(provider):
    conformance.check_idle_sandbox_suspended_not_destroyed(provider, A)


def test_cleanup_stops_every_long_lived_process(tmp_path):
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    env.start_process("one", ["cat"])
    env.start_process("two", ["cat"])
    procs = list(env._long_lived().values())
    env.cleanup()
    assert all(proc.poll() is not None for proc in procs), "cleanup left a long-lived child alive"
    assert all(proc.stdin.closed and proc.stdout.closed for proc in procs)
    assert env._long_lived() == {} and env.attach_process("one") is None


@pytest.mark.platforms("posix")
def test_stopping_an_exited_leader_still_stops_its_descendants(tmp_path):
    """The leader exits but leaves a child in its group: stop_process still stops that child and
    releases the group only once it is empty."""
    import os
    import signal
    from tests.waiting import wait_until
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    pid_file = tmp_path / "child.pid"
    env.start_process("leader", ["bash", "-c", f"sleep 600 & echo $! > {pid_file}; exit 0"])
    leader = env._long_lived()["leader"]
    wait_until(lambda: leader.poll() is not None and pid_file.exists() and pid_file.read_text().strip())
    child = int(pid_file.read_text())
    try:
        assert env.stop_process("leader") is False  # the leader itself was no longer running
        wait_until(lambda: not _alive(child), timeout=10)
    finally:
        if _alive(child):
            os.kill(child, signal.SIGKILL)


def _alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as fh:
        return fh.read().split(") ", 1)[1][0] != "Z"


def test_a_tui_poller_leaves_another_agents_event_for_it():
    """The TUI notification poller under B, sharing A's session key, hands A's event back rather than
    rendering or consuming it."""
    from agent.runtime_scope import bind_state
    from tools.process_registry import process_registry
    from tui_gateway.session_notifications import _notif_handle_event

    evt = {"type": "completion", "session_id": "proc_tui_private", "session_key": "shared",
           "sandbox_key": sandbox_identity(A).key, "command": "echo A-PRIVATE", "output": "A-PRIVATE"}
    rendered, deferred = [], []
    with bind_state(B):
        assert _notif_handle_event("sid", {"session_key": "shared"}, evt, set(), process_registry,
                                   lambda e: rendered.append(e) or "text", deferred) is True
    assert deferred == [evt] and rendered == []


def test_a_recovered_hosted_process_stays_its_agents(tmp_path, monkeypatch):
    """The crash-recovery checkpoint keeps a hosted process's sandbox identity, so after a restart it
    is still invisible to every other activity."""
    import os
    from unittest.mock import patch
    from agent.runtime_scope import bind_state
    from tools.process_registry import ProcessRegistry, ProcessSession

    registry = ProcessRegistry()
    key = sandbox_identity(A).key
    registry._running["proc_hosted_ckpt"] = ProcessSession(
        id="proc_hosted_ckpt", command="sleep 600", task_id=key, sandbox_key=key, pid=os.getpid(),
        pid_scope="host", host_start_time=1.0)
    checkpoint = tmp_path / "procs.json"
    with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
        registry._write_checkpoint()
        restarted = ProcessRegistry()
        monkeypatch.setattr(restarted, "_host_pid_is_ours", lambda *_a: True)
        monkeypatch.setattr(restarted, "_is_host_pid_alive", lambda *_a: True)
        monkeypatch.setattr(restarted, "_refresh_detached_session", lambda s: s)
        assert restarted.recover_from_checkpoint() == 1
    assert restarted.get("proc_hosted_ckpt") is None
    with bind_state(B):
        assert restarted.get("proc_hosted_ckpt") is None
    with bind_state(A):
        assert restarted.get("proc_hosted_ckpt") is not None


def test_media_is_never_read_on_host_with_a_host_backend_configured(provider, monkeypatch, host_dir):
    """With ``terminal.backend: local`` a hosted scope still never reads media on this host: vision,
    video and image-generation sources resolve in the sandbox (which a host backend cannot provide,
    so they fail closed)."""
    import asyncio
    from agent.runtime_scope import bind_state
    from tools.image_generation_tool import _confine_source_images
    from tools.image_source import ImageResolutionError, ResolveContext, resolve_image_source
    from tools.vision_tools import _materialize_video

    monkeypatch.setenv("TERMINAL_ENV", "local")
    image, video = host_dir / "photo.png", host_dir / "clip.mp4"
    image.write_bytes(conformance._PNG)
    video.write_bytes(b"\0\0\0\x18ftypmp42" + b"\0" * 64)
    with conformance._watch_host(host_dir) as touched, bind_state(A):
        with pytest.raises(ImageResolutionError):
            asyncio.run(resolve_image_source(str(image), ResolveContext(task_id="t")))
        with pytest.raises(ValueError):
            asyncio.run(_materialize_video(str(video), "t", []))
        _, _, error = _confine_source_images(str(image), [str(image)], "t")
        assert error is not None
    assert not touched, f"media resolution touched the host: {touched}"


def test_local_scopes_build_byte_identical_environments(monkeypatch):
    """Unscoped and with the local scope bound, the subprocess environment map and the built-in
    backends' constructor arguments are byte-identical to the same calls with the hosted-scope
    machinery switched off entirely, and the environment is the backend's own object (no guard)."""
    import types
    from contextlib import nullcontext
    from agent.runtime_scope import bind_state
    from tools import terminal_tool_backends as backends
    from tools.environments import local, sandbox_scope

    attrs = {"local": "_LocalEnvironment", "docker": "_DockerEnvironment", "ssh": "_SSHEnvironment"}

    def build() -> bytes:
        maps = []
        for options in ({}, {"scrub_secrets": False}, {"inherit_profile_home": False, "extra": {"FOO": "x"}}):
            base = {"PATH": "/usr/bin", "FOO": "bar", "LANG": "C", "OPENAI_API_KEY": "synthetic-not-secret"}
            maps.append(sorted(local.build_subprocess_env(base, **options).items()))
        built = []
        for backend, attr in attrs.items():
            captured, sentinel = [], types.SimpleNamespace()
            monkeypatch.setattr(backends, attr, lambda *a, _c=captured, _s=sentinel, **k: _c.append((a, k)) or _s)
            env = backends._create_environment(backend, "image", "/workspace", 10, task_id="t",
                                               ssh_config={"host": "test", "user": "user"})
            built.append((captured, env is sentinel))
        return repr((maps, built)).encode()

    real_hosted_scope = sandbox_scope.hosted_scope
    for scope in (None, Scope("local", "default", Principal("local"))):
        with bind_state(scope) if scope else nullcontext():
            with_machinery = build()
            monkeypatch.setattr(sandbox_scope, "hosted_scope", lambda: None)
            without = build()
            monkeypatch.setattr(sandbox_scope, "hosted_scope", real_hosted_scope)
        assert with_machinery == without, f"local behaviour changed under scope {scope}"
        assert b"False)" not in with_machinery, "a local backend came back wrapped"


def test_an_acquired_stream_is_refused_under_another_scope():
    """start_process / attach_process hand back proxies: once acquired, a stream's pipes and poll
    stay A's, whoever holds the object (round-2 finding 16)."""
    import io
    from agent.runtime_scope import bind_state
    from tools.environments.sandbox_scope import ProcessRef, ProcessStream

    class Backend:
        def start_process(self, *args, **kwargs):
            return ProcessStream(ProcessRef("", "service"), io.BytesIO(), io.BytesIO(b"A-PRIVATE"), lambda: None)

        attach_process = start_process

    env = ScopedEnvironment(Backend(), sandbox_identity(A), owner=A)
    with bind_state(A):
        stream, attached = env.start_process("service", ["cat"]), env.attach_process("service")
    for s in (stream, attached):
        with bind_state(B):
            for op in (lambda: s.stdout.read(), lambda: s.stdin.write(b"x"), s.poll, lambda: s.stdout.close(),
                       lambda: list(s.stdout), lambda: s.stdout.closed):
                with pytest.raises(CrossAgentHandleRefused):
                    op()
        with pytest.raises(CrossAgentHandleRefused):  # no scope at all is not the owner either
            s.stdout.read()
    with bind_state(A):
        assert stream.stdout.read() == b"A-PRIVATE" and stream.poll() is None


def _buffered_stream_env():
    import io
    from tools.environments.sandbox_scope import ProcessRef, ProcessStream

    class Backend:
        def start_process(self, *a, **k):
            return ProcessStream(ProcessRef("", "service"), io.BufferedWriter(io.BytesIO()),
                                 io.BufferedReader(io.BytesIO(b"A-PRIVATE\nsecond\n")), lambda: None)

    return ScopedEnvironment(Backend(), sandbox_identity(A), owner=A)


def test_foreign_buffered_stream_raw_io_is_refused():
    """A standard BufferedReader's ``raw`` (or any nested I/O object / descriptor) is not reachable
    through the proxy, so nothing acquired under A can be read under B."""
    from agent.runtime_scope import bind_state
    env = _buffered_stream_env()
    with bind_state(A):
        stream = env.start_process("service", ["cat"])
        for pipe in (stream.stdin, stream.stdout):
            for escape in ("raw", "buffer", "fileno", "detach", "peek", "seek", "_raw", "name", "mode"):
                assert not hasattr(pipe, escape), escape
            with pytest.raises(AttributeError):
                pipe.raw = None
        read1, readline = stream.stdout.read1, stream.stdout.readline  # bound methods, kept for later
    with bind_state(B):
        for op in (read1, readline, lambda: stream.stdout.readinto(bytearray(4)), lambda: stream.stdout.readlines(),
                   lambda: stream.stdin.writelines([b"x"]), lambda: stream.stdin.flush(),
                   lambda: stream.stdout.readable(), lambda: stream.stdin.writable(),
                   lambda: stream.stdout.closed, lambda: iter(stream.stdout)):
            with pytest.raises(CrossAgentHandleRefused):
                op()
    with bind_state(A):
        assert stream.stdout.readline() == b"A-PRIVATE\n" and list(stream.stdout) == [b"second\n"]
        assert stream.stdin.write(b"ok") == 2 and stream.stdin.writable() and not stream.stdout.closed
