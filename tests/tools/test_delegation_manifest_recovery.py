"""Regression for #146414: crash recovery must also settle the live manifest."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation as ledger
from tools.delegation_live_log import create_live_transcripts, live_transcript_root, update_manifest_statuses


@pytest.mark.parametrize("record_finished_child", [False, True])
def test_dead_owner_recovery_settles_only_its_manifest_tasks(record_finished_child):
    home = get_hermes_home()
    producer = r'''
import os, time
from tools import async_delegation as ledger
from tools.delegation_live_log import create_live_transcripts
tasks = [{"goal": "finished"}, {"goal": "unfinished"}, {"goal": "other unit"}]
did, writers, paths = create_live_transcripts(tasks, delegation_id="deleg_abcdef01")
ledger._persist_dispatch({
    "delegation_id": did + "-1", "dispatched_at": time.time(),
    "is_batch": True, "goals": [t["goal"] for t in tasks], "task_indexes": [0, 1],
    "task_transcripts": {str(i): paths[i] for i in (0, 1)},
})
if os.environ["RECORD_FINISHED_CHILD"] == "1":
    ledger.record_unit_child(did + "-1", {"task_index": 0, "status": "completed", "summary": "done"})
os._exit(0)
'''
    subprocess.run(
        [sys.executable, "-c", producer], cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "HERMES_HOME": str(home), "RECORD_FINISHED_CHILD": str(int(record_finished_child))},
        capture_output=True, text=True, check=True, timeout=30,
    )
    manifest_path = live_transcript_root() / "deleg_abcdef01" / "manifest.json"
    assert [t["status"] for t in json.loads(manifest_path.read_text(encoding="utf-8-sig"))["tasks"]] == ["running"] * 3
    ledger._persist_dispatch({
        "delegation_id": "deleg_abcdef01-2", "dispatched_at": time.time(),
        "is_batch": True, "goals": ["finished", "unfinished", "other unit"], "task_indexes": [2],
    })

    assert ledger.recover_abandoned_delegations() == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert [t["status"] for t in manifest["tasks"]] == [
        "completed" if record_finished_child else "unknown", "unknown", "running",
    ]
    assert "completed" not in manifest
    assert ledger.get_durable_delegation("deleg_abcdef01-1")["state"] == "unknown"
    before = manifest_path.read_bytes()
    assert ledger.recover_abandoned_delegations() == 0
    assert manifest_path.read_bytes() == before

    update_manifest_statuses("deleg_abcdef01", [{"task_index": 2, "status": "completed"}])
    settled = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert settled["completed"]
    assert settled["tasks"][2]["status"] == "completed"


def test_live_owner_keeps_manifest_running():
    did, _, _ = create_live_transcripts([{"goal": "still working"}])
    ledger._persist_dispatch({"delegation_id": did, "dispatched_at": time.time(), "goal": "still working"})
    manifest_path = live_transcript_root() / did / "manifest.json"
    before = manifest_path.read_bytes()
    assert ledger.recover_abandoned_delegations() == 0
    assert manifest_path.read_bytes() == before


def test_recovery_stays_in_ledger_profile_and_keeps_known_outcomes(tmp_path):
    paths = []
    for profile in ("a", "b"):
        home = tmp_path / profile
        home.mkdir()
        token = set_hermes_home_override(home)
        try:
            did, _, _ = create_live_transcripts([{"goal": "finished"}, {"goal": "lost"}],
                                               delegation_id="deleg_abcdef02")
            update_manifest_statuses(did, [{"task_index": 0, "status": "completed"}], completed=False)
            ledger._persist_dispatch({"delegation_id": did, "dispatched_at": time.time(),
                                      "is_batch": True, "goals": ["finished", "lost"]})
            with ledger._transaction() as conn:
                conn.execute("UPDATE async_delegations SET owner_pid=0")
            paths.append(live_transcript_root() / did / "manifest.json")
        finally:
            reset_hermes_home_override(token)
    before_b = paths[1].read_bytes()
    for profile, expected in (("a", 1), ("b", 1), ("a", 0)):
        token = set_hermes_home_override(tmp_path / profile)
        try:
            assert ledger.recover_abandoned_delegations() == expected
        finally:
            reset_hermes_home_override(token)
        if profile == "a":
            manifest = json.loads(paths[0].read_text(encoding="utf-8-sig"))
            assert [t["status"] for t in manifest["tasks"]] == ["completed", "unknown"]
            assert manifest["completed"]
            if expected:
                assert paths[1].read_bytes() == before_b
