"""MCP's spawn-delta filter never adopts the owner-death supervisor.

The supervisor is a direct child of this process, started lazily by the first owned spawn of any
family. When that happens between MCP's before/after child snapshots, the delta contains it; adopting
it would let MCP teardown group-kill the one process that reaps every other owned child.
"""

import subprocess
import sys

import pytest

from hermes_cli import process_identity
from tools import mcp_tool_lifecycle

pytestmark = pytest.mark.platforms("posix")


def test_an_owned_spawn_racing_mcp_discovery_does_not_hand_mcp_the_supervisor(monkeypatch):
    monkeypatch.setattr(process_identity, "_SUPERVISOR_IDLE_EXIT_S", 0.0)
    monkeypatch.setattr(process_identity, "_supervisor", None)
    monkeypatch.setattr(process_identity, "_supervised_pgids", set())
    monkeypatch.setattr(process_identity, "register_child", lambda *a, **k: True)
    before = mcp_tool_lifecycle._snapshot_child_pids()
    # Another family's first owned spawn lands inside MCP's before/after window: it starts the supervisor.
    worker = process_identity.spawn_owned([sys.executable, "-c", "import sys; sys.stdin.read()"],
                                          purpose="test", stdin=subprocess.PIPE)
    mcp_server = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        supervisor = process_identity._supervisor
        assert supervisor is not None
        delta = mcp_tool_lifecycle._snapshot_child_pids() - before
        assert supervisor.pid in delta, "precondition: the supervisor is in MCP's spawn delta"
        assert supervisor.pid not in mcp_tool_lifecycle._filter_mcp_children(delta)
        assert mcp_server.pid in mcp_tool_lifecycle._filter_mcp_children(delta)
    finally:
        for proc in (worker, mcp_server):
            proc.stdin.close()
            proc.wait(timeout=10)
        process_identity.release_process_group_once_empty(worker.pid, proc=worker, timeout=5)
        assert process_identity._supervisor is None, "releasing the last group stops the supervisor"


def test_a_supervisor_whose_spawn_has_not_returned_yet_is_still_not_an_mcp_child(monkeypatch):
    """``_supervisor`` is assigned only after the spawn returns; a scan in that gap must still skip it."""
    monkeypatch.setattr(process_identity, "_supervisor", None)
    supervisor = process_identity._spawn_supervisor()  # running, but not yet published as _supervisor
    assert supervisor is not None
    try:
        assert mcp_tool_lifecycle._filter_mcp_children({supervisor.pid}) == set()
    finally:
        supervisor.stdin.close()
        supervisor.wait(timeout=10)
