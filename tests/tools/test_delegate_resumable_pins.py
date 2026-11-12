"""Durable resumable subagents, review round 2: a running continuation pins where it is now, a tree the child made
stays pinned after it leaves that repository, a late record write never recreates a removed profile, and boot
recovery names the continuable children on the ledger row that reports them (independent completions on or off).

Real SQLite, real git, real sweep, real dispatch and boot recovery against a temp HERMES_HOME; only the model is
scripted.
"""

import json
import os
import sqlite3
import stat
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _parent, _ScriptedModel
from tests.tools.test_delegate_resumable import _git
from tests.waiting import wait_until

_DEAD_PID = 2 ** 22 + 7


def _repo(path):
    path.mkdir()
    _git("init", "-q", "-b", "main", cwd=path)
    (path / "a").write_text("a")
    _git("add", "a", cwd=path)
    _git("commit", "-qm", "base", cwd=path)
    return path


def _run(parent, sid):
    child = SimpleNamespace(session_id="child", model="m", _parent_session_id=parent.session_id,
                            _delegate_resume={"goal": "g", "route": SimpleNamespace(request={})})
    return SimpleNamespace(child=child, parent_agent=parent, subagent_id=sid, worktree_info=None, child_task_id="t")


def _completions(delegation_ids):
    """Wait for every listed unit's completion event (one drain: waiting per id would drop a sibling's event)."""
    from tools.process_registry import process_registry
    pending = set(delegation_ids)

    def drained():
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "async_delegation":
                pending.discard(evt.get("delegation_id"))
        return not pending
    wait_until(drained, timeout=20, what=f"completions of {sorted(pending)}")


def _backdate(*paths):
    old = time.time() - 30 * 3600
    for p in paths:
        os.utime(p, (old, old))


def test_a_late_record_write_never_recreates_a_removed_profile_and_stays_owner_only(tmp_path):
    from hermes_constants import get_hermes_home, profile_tombstone_path
    from tools import delegate_tool_resumable as r
    removed = get_hermes_home() / "profiles" / "removed"
    tomb = profile_tombstone_path(removed)
    tomb.parent.mkdir(parents=True, exist_ok=True)
    tomb.write_text("deleted")

    r._write(removed / "state.db", lambda conn: conn.execute("SELECT 1").fetchone())
    assert not removed.exists()

    live = get_hermes_home() / "state.db"
    r._write(live, lambda conn: conn.execute("SELECT 1").fetchone())
    if os.name != "nt":
        assert stat.S_IMODE(live.stat().st_mode) == 0o600


