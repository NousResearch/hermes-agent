"""Sandbox conformance: a hosted scope's tools address only that agent's sandbox.

The checks live in ``tools/environments/sandbox_conformance.py`` so a hosted backend plugin runs the
same ones against its own provider; here they run against the recording stub, through the real tool
registry, with hosted scopes bound by ``agent.runtime_scope.bind_state``. Local behaviour is covered
by the existing terminal/file/process suites, which run with no hosted scope bound.
"""

from __future__ import annotations

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
    if name == "file_tools_never_touch_host":
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
