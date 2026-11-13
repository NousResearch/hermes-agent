"""Hosted-scope gaps from review: a tenant-named script read on the host by the gateway lifecycle
guard, file-tool bookkeeping shared by agents that reuse a task id, a background notification
promised to a hosted agent that no consumer can deliver, the full tool list still offered to a
hosted caller, and session kernels that outlive the boundary that should reap them. Each runs the
real tool or registry under a hosted scope with the recording sandbox."""

from __future__ import annotations

import json
from typing import Any

from agent.runtime_scope import bind_state
from tests.tools.test_sandbox_scope_conformance import A, B
from tests.tools.test_sandbox_scope_conformance import provider as _conformance_provider
from hermes_agent.tools.environments import sandbox_conformance as c

provider = _conformance_provider


def test_hosted_terminal_never_reads_a_named_script_on_the_host(provider, monkeypatch, tmp_path):
    import hermes_agent.tools.process_registry as processes
    monkeypatch.setattr(processes, "_is_supervised_gateway_process", lambda: True)
    host_dir = tmp_path / "host-workspace"  # apart from the test's HERMES_HOME
    host_dir.mkdir()
    script = host_dir / "deploy.sh"
    script.write_text("#!/bin/sh\nsystemctl --user restart hermes-gateway\n")
    with c._watch_host(host_dir) as touched, bind_state(A):
        result = c.call_tool("terminal", {"command": f"bash {script}"}, task_id="t")
    assert not touched, f"the hosted command's script was read on this host: {touched}"
    assert "Blocked" not in str(result.get("error")), result
    assert any(f"bash {script}" in call.command for call in provider.calls
               if call.sandbox == c._identity(A) and call.op == "execute")


def test_hosted_background_process_is_not_promised_a_notification(provider):
    with bind_state(A):
        result = c.call_tool("terminal", {"command": "sleep 5", "background": True, "notify_on_complete": True,
                                          "watch_patterns": ["done"]}, task_id="t")
        try:
            assert result.get("notify_on_complete") is False and result.get("notify_unsupported"), result
            assert "watch_patterns" not in result, result
        finally:
            c.call_tool("process_manage", {"action": "kill", "session_id": result.get("session_id")}, task_id="t")


def test_hosted_agents_sharing_a_task_id_keep_their_own_file_bookkeeping(provider):
    agents = [c.hosted_scope_for(f"agent-{n}") for n in range(4)]
    for tool, args in (("search_files", {"pattern": "needle", "path": "/workspace"}),
                       ("read_file", {"path": "/workspace/notes.txt"})):
        for scope in agents:  # each agent's first call; on a shared tracker the 4th is "repeated"
            with bind_state(scope):
                result = c.call_tool(tool, args, task_id="default")
            assert "BLOCKED" not in json.dumps(result) and "_warning" not in result, (tool, scope.agent, result)


def test_hosted_scope_is_offered_only_its_allowlist(provider):
    from hermes_agent.model_tools import get_tool_definitions
    from hermes_agent.tools.hosted_tool_policy import HOSTED_TOOL_ALLOWLIST

    local = {t["function"]["name"] for t in get_tool_definitions(quiet_mode=True)}  # fills the memo first
    with bind_state(A):
        hosted = {t["function"]["name"] for t in get_tool_definitions(quiet_mode=True)}
    assert hosted and hosted <= HOSTED_TOOL_ALLOWLIST, hosted - HOSTED_TOOL_ALLOWLIST
    assert local - HOSTED_TOOL_ALLOWLIST, "the local selection should still offer non-sandbox tools"


def test_clear_session_finds_a_hosted_agents_local_kernels():
    from hermes_agent.tools.code_kernel import _KERNELS, shutdown_kernels_for_owner

    torn = []
    key = ("sandbox:acme/agent-a:session-key", "standard", "python3", "/workspace", frozenset())
    stub: Any = type("Stub", (), {"key": key, "owner": key[0], "attached": 0,
                                  "teardown": lambda self: torn.append(key)})()
    _KERNELS[key] = stub
    try:
        with bind_state(A):
            shutdown_kernels_for_owner("session-key")
        assert torn == [key] and not _KERNELS, "the session boundary missed a hosted agent's kernel"
    finally:
        _KERNELS.pop(key, None)


def test_clear_session_finds_a_hosted_agents_remote_kernels():
    from hermes_agent.tools.code_kernel_remote import _REMOTE_KERNELS, shutdown_remote_kernels_for_owner

    killed = []
    key = ("sandbox:acme/agent-a:session-key", "remote", "stub", "t", frozenset())
    stub: Any = type("Stub", (), {"key": key, "owner": key[0], "attached": 0,
                                  "kill": lambda self: killed.append(key)})()
    _REMOTE_KERNELS[key] = stub
    try:
        with bind_state(A):
            shutdown_remote_kernels_for_owner("session-key")
        assert killed == [key] and not _REMOTE_KERNELS, "the session boundary missed a hosted agent's remote kernel"
    finally:
        _REMOTE_KERNELS.pop(key, None)


def test_local_shutdown_does_not_kill_colon_suffix_owner():
    """A session close matches its owner exactly: ``room`` is not ``telegram:room``, locally or
    inside a sandbox (where the sandbox key, not a bare suffix, qualifies the owner)."""
    from hermes_agent.tools.code_kernel import _KERNELS, shutdown_kernels_for_owner

    torn: list[str] = []
    keys = []
    for owner in ("room", "telegram:room"):
        key = (owner, "probe")
        keys.append(key)
        _KERNELS[key] = type("Stub", (), {"teardown": lambda self, o=owner: torn.append(o)})()
    try:
        shutdown_kernels_for_owner("room")
        assert torn == ["room"], torn
    finally:
        for key in keys:
            _KERNELS.pop(key, None)


def test_session_close_does_not_touch_another_agents_kernel(provider):
    """Two hosted agents share a session key; a session close under one agent resolves that
    agent's sandbox-qualified owner and must not run teardown in the other agent's sandbox."""
    from hermes_agent.tools.code_kernel_remote import (RemoteKernel, _REMOTE_KERNELS, _kernel_key,
                                          shutdown_remote_kernels_for_owner)
    from hermes_agent.tools.environments.sandbox_scope import ScopedEnvironment, sandbox_identity

    reached = []

    class Backend:
        def execute(self, command, cwd=None, timeout=30, scope=None):
            reached.append((getattr(scope, "agent", None), command))
            return {"output": ""}

    keys = []
    try:
        for scope in (A, B):
            from hermes_agent.tools.approval_context import reset_current_session_key, set_current_session_key
            from hermes_agent.tools.code_kernel import _resolve_owner
            token = set_current_session_key("shared-session")
            try:
                with bind_state(scope):
                    owner = _resolve_owner("t")
            finally:
                reset_current_session_key(token)
            key = _kernel_key(owner, "recording", "t", frozenset())
            keys.append(key)
            _REMOTE_KERNELS[key] = RemoteKernel(
                env=ScopedEnvironment(Backend(), sandbox_identity(scope), owner=scope),
                env_type="recording", kernel_dir="/workspace/k", pid="4242",
                rpc_token="tok", owner=owner, owner_scope=scope)
        with bind_state(B):
            shutdown_remote_kernels_for_owner("shared-session")
        assert not any(agent == "agent-a" for agent, _ in reached), reached
        assert "agent-b" in {agent for agent, _ in reached}, reached
    finally:
        for key in keys:
            _REMOTE_KERNELS.pop(key, None)


def test_another_agents_eviction_still_stops_the_owning_kernels_service(provider):
    from hermes_agent.tools.code_kernel_remote import RemoteKernel
    from hermes_agent.tools.environments.sandbox_scope import ScopedEnvironment

    reached = []

    class _Inner:
        def execute(self, command, cwd=None, timeout=30, scope=None):
            reached.append((command, scope))
            return ""

    identity = c._identity(A)
    kernel = RemoteKernel(env=ScopedEnvironment(_Inner(), identity, owner=A), env_type="stub",
                          kernel_dir="/workspace/k", pid="4242", rpc_token="tok",
                          owner="sandbox:acme/agent-a:session-key")
    kernel.owner_scope = A
    with bind_state(B):
        kernel.kill()
    assert any("4242" in command for command, _ in reached), reached
