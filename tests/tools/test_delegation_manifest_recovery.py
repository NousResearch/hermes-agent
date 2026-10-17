"""Durable ledger, not a forgotten live transcript, owns delegation status."""
import json
import time

import pytest

from tools import async_delegation as ad
from tools import delegation_live_log as dll
from tools.delegation_live_log import create_live_transcripts, live_transcript_root, update_manifest_statuses


def _dispatch(delegation_id, goals):
    tasks = [{"goal": goal} for goal in goals]
    live_id, writers, paths = create_live_transcripts(tasks, delegation_id=delegation_id)
    assert live_id == delegation_id and len(writers) == len(goals)
    record = {
        "delegation_id": delegation_id,
        "session_key": "session-owner", "parent_session_id": "session-owner",
        "goal": goals[0], "goals": goals, "is_batch": len(goals) > 1,
        "task_indexes": list(range(len(goals))),
        "task_transcripts": {str(i): path for i, path in enumerate(paths)},
        "status": "running", "dispatched_at": time.time() - 100,
    }
    ad._persist_dispatch(record)
    return live_transcript_root() / delegation_id / "manifest.json"


def _manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_dead_owner_recovery_retires_only_its_running_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = _dispatch("deleg_aabbcc01", ["review only"])
    log = path.parent / "task-0.log"
    before = log.read_bytes()
    monkeypatch.setattr(ad, "_owner_liveness", lambda: lambda *_: False)

    assert ad.recover_abandoned_delegations() == 1
    assert _manifest(path)["tasks"][0]["status"] == "unknown"
    assert log.read_bytes() == before
    assert ad.recover_abandoned_delegations() == 0
    once = path.read_bytes()
    assert ad.recover_abandoned_delegations() == 0
    assert path.read_bytes() == once  # duplicate sweep cannot repeat an effect
    with ad._connect() as conn:
        assert conn.execute("select state,delivery_state from async_delegations where delegation_id=?",
                            ("deleg_aabbcc01",)).fetchone() == ("unknown", "pending")


def test_dead_batch_owner_preserves_finished_child_and_marks_only_unfinished_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = _dispatch("deleg_aabbcc02", ["done", "unfinished"])
    ad.record_unit_child("deleg_aabbcc02", {"task_index": 0, "status": "completed", "summary": "verified"})
    monkeypatch.setattr(ad, "_owner_liveness", lambda: lambda *_: False)

    assert ad.recover_abandoned_delegations() == 1
    assert [task["status"] for task in _manifest(path)["tasks"]] == ["completed", "unknown"]
    with ad._connect() as conn:
        payload = json.loads(conn.execute("select result_json from async_delegations where delegation_id=?",
                                          ("deleg_aabbcc02",)).fetchone()[0])
    assert [r["status"] for r in payload["results"]] == ["completed", "unknown"]


def test_restart_reconciles_old_terminal_row_without_replaying_work(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = _dispatch("deleg_aabbcc03", ["read-only old review"])
    ad._persist_completion({"delegation_id": "deleg_aabbcc03", "status": "unknown"},
                           {"status": "unknown", "error": "owner died"})
    assert _manifest(path)["tasks"][0]["status"] == "running"

    from queue import Queue
    queue = Queue()
    assert ad.restore_undelivered_completions(queue) == 1
    assert _manifest(path)["tasks"][0]["status"] == "unknown"
    with ad._connect() as conn:
        assert conn.execute("select state from async_delegations where delegation_id=?",
                            ("deleg_aabbcc03",)).fetchone()[0] == "unknown"


def test_grouped_unit_reconciles_shared_call_manifest_without_touching_sibling(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    call_id = "deleg_aabbcc05"
    _, _, paths = create_live_transcripts([{"goal": "live sibling"}, {"goal": "dead child"}],
                                          delegation_id=call_id)
    manifest = live_transcript_root() / call_id / "manifest.json"
    ad._persist_dispatch({
        "delegation_id": call_id + "-2", "session_key": "session-owner",
        "parent_session_id": "session-owner", "goal": "dead child",
        "goals": ["dead child"], "is_batch": True, "task_indexes": [1],
        "task_transcripts": {"1": paths[1]}, "status": "running",
        "dispatched_at": time.time() - 100,
    })
    monkeypatch.setattr(ad, "_owner_liveness", lambda: lambda *_: False)

    assert ad.recover_abandoned_delegations() == 1
    assert [task["status"] for task in _manifest(manifest)["tasks"]] == ["running", "unknown"]
    assert "completed" not in _manifest(manifest)  # sibling is still live


def test_late_sidecar_writer_cannot_permanently_override_terminal_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = _dispatch("deleg_aabbcc06", ["review"])
    ad._persist_completion({"delegation_id": "deleg_aabbcc06", "status": "unknown"},
                           {"status": "unknown", "error": "owner died"})
    ad.recover_abandoned_delegations()
    assert _manifest(path)["tasks"][0]["status"] == "unknown"
    # A late sidecar-only write must not become a permanent false result.
    update_manifest_statuses("deleg_aabbcc06", [{"task_index": 0, "status": "completed"}])
    assert _manifest(path)["tasks"][0]["status"] == "completed"
    ad.recover_abandoned_delegations()
    assert _manifest(path)["tasks"][0]["status"] == "unknown"


def test_terminal_projection_survives_unavailable_liveness_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = _dispatch("deleg_aabbcc07", ["settled before restart"])
    ad._persist_completion({"delegation_id": "deleg_aabbcc07", "status": "unknown"},
                           {"status": "unknown", "error": "owner died"})
    monkeypatch.setattr(ad, "_owner_liveness", lambda: None)
    assert ad.recover_abandoned_delegations() == 0
    assert _manifest(path)["tasks"][0]["status"] == "unknown"


def test_normal_status_writer_never_exposes_partial_manifest_on_write_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = _dispatch("deleg_aabbcc08", ["review"])
    original_write_text = type(path).write_text

    def interrupted_direct_write(self, text, *args, **kwargs):
        if self == path:
            with self.open("w", encoding="utf-8") as stream:
                stream.write("{")  # simulate interruption after truncating the live JSON
            raise OSError("writer lost during a direct manifest write")
        return original_write_text(self, text, *args, **kwargs)

    monkeypatch.setattr(type(path), "write_text", interrupted_direct_write)
    update_manifest_statuses("deleg_aabbcc08", [{"task_index": 0, "status": "completed"}])
    assert _manifest(path)["tasks"][0]["status"] == "completed"
    assert not list(path.parent.glob(".manifest.*"))


@pytest.mark.parametrize("writer", ["normal", "recovery"])
@pytest.mark.parametrize("stage", ["dump", "fsync", "replace"])
def test_failed_staged_write_preserves_live_json_and_cleans_staging(tmp_path, monkeypatch, writer, stage):
    """Exercise the *actual* staged seam, not only the retired direct-write path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    delegation_id = "deleg_aabbcc09"
    path = _dispatch(delegation_id, ["read-only review"])
    before = path.read_bytes()
    failures = []

    def fail_stage(*args, **kwargs):
        failures.append(stage)
        if stage == "dump":
            args[1].write("{")  # a truncated staging file, never the live manifest
        raise OSError("simulated loss during staged write")

    if stage == "dump":
        monkeypatch.setattr(dll.json, "dump", fail_stage)
    else:
        monkeypatch.setattr(dll.os, stage, fail_stage)
    if writer == "normal":
        update_manifest_statuses(delegation_id, [{"task_index": 0, "status": "completed"}])
    else:
        assert dll.reconcile_terminal_manifest(delegation_id, {0: "unknown"}) is False

    assert failures == [stage], "the failure must reach the staged writer"
    assert path.read_bytes() == before
    assert _manifest(path)["tasks"][0]["status"] == "running"
    assert not list(path.parent.glob(".manifest.staged.*"))


def test_live_owner_is_never_reclassified_by_reconciler(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = _dispatch("deleg_aabbcc04", ["long-running useful work"])
    monkeypatch.setattr(ad, "_owner_liveness", lambda: lambda *_: True)

    assert ad.recover_abandoned_delegations() == 0
    assert _manifest(path)["tasks"][0]["status"] == "running"
    with ad._connect() as conn:
        assert conn.execute("select state from async_delegations where delegation_id=?",
                            ("deleg_aabbcc04",)).fetchone()[0] == "running"


def test_suffixed_row_pointing_outside_parent_leaves_manifest_untouched(tmp_path, monkeypatch):
    """A suffixed row may not strip to a guessed parent: its own ``task_transcripts`` must
    name that parent's canonical file. Forged refs (another delegation's directory) write
    nothing — both live manifests stay byte-identical (PR #130937 review finding)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parent, other = "deleg_aabbcc10", "deleg_aabbcc11"
    create_live_transcripts([{"goal": "live parent review"}], delegation_id=parent)
    _, _, other_paths = create_live_transcripts([{"goal": "unrelated live review"}], delegation_id=other)
    parent_manifest = live_transcript_root() / parent / "manifest.json"
    other_manifest = live_transcript_root() / other / "manifest.json"
    before_parent, before_other = parent_manifest.read_bytes(), other_manifest.read_bytes()
    # A real suffixed id: this unit row's transcripts name the OTHER delegation's file, so
    # stripping "-1" and writing to the guessed parent would corrupt a manifest it never owned.
    ad._persist_dispatch({
        "delegation_id": "deleg_aabbcc10-1", "session_key": "session-owner",
        "parent_session_id": "session-owner", "goal": "forged unit", "goals": ["forged unit"],
        "is_batch": True, "task_indexes": [0], "task_transcripts": {"0": other_paths[0]},
        "status": "running", "dispatched_at": time.time() - 100,
    })
    monkeypatch.setattr(ad, "_owner_liveness", lambda: lambda *_: False)

    assert ad.recover_abandoned_delegations() == 1  # the abandoned row itself still settles
    assert _manifest(parent_manifest)["tasks"][0]["status"] == "running"  # refusal: no write
    assert parent_manifest.read_bytes() == before_parent  # byte-identical: no partial write
    assert other_manifest.read_bytes() == before_other
    assert not list(parent_manifest.parent.glob(".manifest.*"))  # no staging residue
    assert not list(other_manifest.parent.glob(".manifest.*"))
    assert ad.recover_abandoned_delegations() == 0  # repeated sweeps keep refusing
    assert parent_manifest.read_bytes() == before_parent


def test_suffixed_row_pointing_inside_parent_still_applies_status(tmp_path, monkeypatch):
    """The binding check discriminates rather than blanket-skipping suffixed rows: a row
    whose own ``task_transcripts`` name the parent's canonical file still projects its
    terminal status there (PR #130937 review finding)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parent = "deleg_aabbcc12"
    _, _, paths = create_live_transcripts([{"goal": "settled unit"}], delegation_id=parent)
    manifest = live_transcript_root() / parent / "manifest.json"
    ad._persist_dispatch({
        "delegation_id": "deleg_aabbcc12-1", "session_key": "session-owner",
        "parent_session_id": "session-owner", "goal": "settled unit", "goals": ["settled unit"],
        "is_batch": True, "task_indexes": [0], "task_transcripts": {"0": paths[0]},
        "status": "running", "dispatched_at": time.time() - 100,
    })
    monkeypatch.setattr(ad, "_owner_liveness", lambda: lambda *_: False)

    assert ad.recover_abandoned_delegations() == 1
    assert _manifest(manifest)["tasks"][0]["status"] == "unknown"  # bound row applies
    once = manifest.read_bytes()
    assert ad.recover_abandoned_delegations() == 0
    assert manifest.read_bytes() == once
