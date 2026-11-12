"""A secondary-profile sidecar session: the /undo file line and /rollback use the SESSION's profile store, not
the launch profile's, so the line names the edit and restoring the listed checkpoint brings the file back."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_agent.tools.checkpoint_manager import CheckpointManager
from hermes_agent.tui_gateway import server


def test_undo_hint_then_rollback_restore_round_trip_in_the_sessions_profile(tmp_path):
    launch, profile, project = tmp_path / "launch", tmp_path / "profiles" / "work", tmp_path / "proj"
    for path in (launch, profile, project / ".git"):
        path.mkdir(parents=True)
    (project / "app.js").write_text("hello\n")
    mgr = CheckpointManager(enabled=True)
    start = time.time()
    history = [{"role": "user", "content": "read", "timestamp": start - 60},
               {"role": "assistant", "content": "ok", "timestamp": start - 59},
               {"role": "user", "content": "edit app.js", "timestamp": start}]
    token = set_hermes_home_override(str(profile))  # the turn ran under its profile: its store took the snapshot
    try:
        assert mgr.ensure_checkpoint(str(project), "before write")
    finally:
        reset_hermes_home_override(token)
    (project / "app.js").write_text("edited\n")
    history.append({"role": "assistant", "content": "edited", "timestamp": time.time()})
    sid = "undo-hint-sid"
    server._sessions[sid] = {
        "agent": SimpleNamespace(_checkpoint_mgr=mgr), "session_key": "", "history": history,
        "history_lock": threading.Lock(), "history_version": 0, "running": False, "attached_images": [],
        "cwd": str(project), "profile_home": str(profile)}
    token = set_hermes_home_override(str(launch))  # the RPC thread holds the launch profile
    try:
        undo = server._methods["session.undo"]("rid", {"session_id": sid})["result"]
        listed = server._methods["rollback.list"]("rid", {"session_id": sid})["result"]
        restored = server._methods["rollback.restore"]("rid", {"session_id": sid, "hash": "1"})["result"]
        kept = [m["content"] for m in server._sessions[sid]["history"]]
    finally:
        reset_hermes_home_override(token)
        server._sessions.pop(sid, None)
    assert undo["removed"] == 2 and undo["undo_hint"]["files"] == ["app.js"], undo
    assert "Run /rollback" in undo["undo_hint"]["text"], undo
    assert len(listed["checkpoints"]) == 1, listed
    assert restored["success"] is True, restored
    assert (project / "app.js").read_text() == "hello\n"
    assert kept == ["read", "ok"]
    assert not (launch / "checkpoints").exists()  # nothing landed in the launch profile's store
