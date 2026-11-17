"""A resumable record's stored pins are discovered fresh: the sweep's memo of a running child's pins must not
decide what boot recovery or a dead-owner reclaim persists for good (a tree added within the memo's TTL)."""

import json
import os
import sqlite3
import time

import pytest

from tests.tools.test_delegate_resumable import _git
from tests.tools.test_delegate_resumable_pins import _DEAD_PID, _repo


@pytest.mark.parametrize("path", ["boot", "reclaim"])
def test_pins_persisted_for_a_dead_owner_include_a_tree_added_after_the_sweep_memo(tmp_path, path):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tools import delegate_tool_resumable as r
    repo = _repo(tmp_path / "repo")
    db = get_hermes_home() / "state.db"
    SessionDB(db_path=db).close()
    record = {"session_id": "s", "cwd": str(repo), "start_cwd": str(repo), "owners": ["p"],
              "worktrees_at_start": [os.path.realpath(repo)], "owner_pid": _DEAD_PID, "owner_started_at": 1.0}
    conn = sqlite3.connect(db)  # health: allow HX015 -- test seeds a temp-home store
    with conn:
        r._upsert(conn, "sa-dead", record, state="running", pinned=True, expires_at=time.time() + 3600)
    conn.close()
    r._live_pins(record)  # a sweep in this process asked a moment ago
    late = tmp_path / "late"
    _git("worktree", "add", "-q", str(late), "-b", "late", cwd=repo)  # the child adds a tree, cwd unchanged, dies

    if path == "boot":
        pins = r.plan_recovery(db, lambda pid, started: False)["sa-dead"][2]
    else:
        from unittest.mock import patch
        with patch("tools.async_delegation._owner_liveness", return_value=lambda pid, started: False):
            pins = r._reclaim_dead_owner(db, "sa-dead", record)["pin_paths"]
    assert os.path.realpath(late) in pins
