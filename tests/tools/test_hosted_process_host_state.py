"""A hosted agent's process activity leaves this host's process files alone: the running-process
checkpoint (``processes.json``) and the completed-process receipts under ``logs/process-results``.
Its remote kernel's commands run under the turn that issues them."""
from __future__ import annotations

import json
import os
import time

from tests.tools.test_sandbox_scope_conformance import A
from tests.tools.test_sandbox_scope_conformance import provider as _conformance_provider
from hermes_agent.tools.environments import sandbox_conformance as conformance

provider = _conformance_provider


def _state(path):
    """Existence, mtime and bytes: a rewrite with identical content still changes the mtime."""
    return (path.stat().st_mtime_ns, path.read_bytes()) if path.exists() else None


def test_a_hosted_background_start_and_kill_leave_the_host_checkpoint_alone(provider):
    from hermes_agent.hermes_constants import get_hermes_home
    ckpt = get_hermes_home() / "processes.json"
    before = _state(ckpt)
    with conformance._bind(A):
        started = conformance.call_tool("terminal", {"command": "sleep 600", "background": True}, task_id="t")
        session_id = started.get("session_id")
        assert session_id, f"the background spawn failed: {started}"
        polled = conformance.call_tool("process_manage", {"action": "poll", "session_id": session_id}, task_id="t")
        assert polled.get("session_id") == session_id, polled
        killed = conformance.call_tool("process_manage", {"action": "kill", "session_id": session_id}, task_id="t")
        assert killed.get("status") in ("killed", "already_exited"), killed
    assert _state(ckpt) == before, "a hosted process start or kill rewrote this host's processes.json"


def test_hosted_process_manage_does_not_load_or_prune_host_receipts(provider, monkeypatch):
    from hermes_agent.gateway.session_context import scoped_current_session_id
    from hermes_agent.hermes_constants import get_hermes_home
    from hermes_agent.tools import process_registry_results as results

    directory = get_hermes_home() / "logs" / "process-results"
    directory.mkdir(parents=True, exist_ok=True)
    record = {key: None for key in results._RESULT_FIELDS}
    fresh, expired = directory / "proc_hostfresh.json", directory / "proc_hostexpired.json"
    for path in (fresh, expired):
        fresh_record = {**record, "id": path.stem, "command": "echo HOST", "parent_session_id": "sess-1",
                        "output": "HOST-RECEIPT", "exit_code": 0}
        path.write_text(json.dumps(fresh_record), encoding="utf-8")
    old = time.time() - results.RESULT_RETENTION_SECONDS - 3600
    os.utime(expired, (old, old))
    before = {path: _state(path) for path in (fresh, expired)}

    read = []
    real_paths = results._result_paths
    monkeypatch.setattr(results, "_result_paths", lambda: read.append(1) or real_paths())
    with scoped_current_session_id("sess-1"), conformance._bind(A):
        for args in ({"action": "list"}, {"action": "poll", "session_id": "proc_hostfresh"},
                     {"action": "poll", "session_id": "proc_hostf"}):
            out = conformance.call_tool("process_manage", args, task_id="t")
            assert "HOST-RECEIPT" not in json.dumps(out), out
    assert not read, "a hosted process_manage read this host's process receipts"
    assert {path: _state(path) for path in (fresh, expired)} == before, "a hosted call pruned host receipts"


def test_a_local_background_start_still_writes_the_checkpoint(tmp_path):
    from hermes_agent.hermes_constants import get_hermes_home
    from hermes_agent.tools.process_registry import ProcessRegistry
    ckpt = get_hermes_home() / "processes.json"
    registry = ProcessRegistry()
    session = registry.spawn_local("sleep 30", cwd=str(tmp_path), task_id="t")
    try:
        assert session.id in json.dumps(json.loads(ckpt.read_text(encoding="utf-8")))
    finally:
        registry.kill_process(session.id)


def test_a_remote_kernel_command_on_a_later_turn_runs_under_that_turns_scope():
    """The kernel record keeps the scope of the turn that spawned it; a later turn of the same
    sandbox runs its kernel commands under its own scope (and fence), not the spawning turn's."""
    from hermes_agent.agent.runtime_scope import bind_state
    from hermes_agent.hermes_scope import Fence
    from hermes_agent.tools.code_kernel_remote import RemoteKernel
    from hermes_agent.tools.environments.sandbox_scope import ScopedEnvironment, sandbox_identity

    turn1, turn2 = A.for_turn("s", "turn-1", Fence(1, 1)), A.for_turn("s", "turn-2", Fence(1, 2))
    reached = []

    class Backend:
        def execute(self, command, cwd=None, timeout=30, scope=None):
            reached.append(scope)
            return {"output": ""}

    kernel = RemoteKernel(env=ScopedEnvironment(Backend(), sandbox_identity(A), owner=turn1), env_type="stub",
                          kernel_dir="/workspace/k", pid="4242", rpc_token="tok",
                          owner="sandbox:acme/agent-a:s", owner_scope=turn1)
    with bind_state(turn2):
        kernel.sh("true")
    assert reached and reached[-1] is turn2, f"a turn-2 kernel command ran under {reached[-1]!r}"
