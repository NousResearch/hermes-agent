"""Sidecar ``session.undo`` names the ``/rollback`` of the SESSION's profile store, not the launch profile's."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.checkpoint_manager import CheckpointManager
from tui_gateway import server


def test_session_undo_hint_reads_the_session_profiles_checkpoints(tmp_path):
    launch, profile, project = tmp_path / "launch", tmp_path / "profiles" / "work", tmp_path / "proj"
    for path in (launch, profile, project / ".git"):
        path.mkdir(parents=True)
    (project / "app.js").write_text("hello\n")
    mgr = CheckpointManager(enabled=True)
    start = time.time() - 60
    token = set_hermes_home_override(str(profile))  # the turn ran under its profile: its store took the snapshot
    try:
        assert mgr.ensure_checkpoint(str(project), "before write")
    finally:
        reset_hermes_home_override(token)
    (project / "app.js").write_text("edited\n")
    history = [{"role": "user", "content": "read", "timestamp": start},
               {"role": "assistant", "content": "ok", "timestamp": start + 1},
               {"role": "user", "content": "edit app.js", "timestamp": start + 10},
               {"role": "assistant", "content": "edited", "timestamp": time.time() + 1}]
    server._sessions["undo-hint-sid"] = {
        "agent": SimpleNamespace(_checkpoint_mgr=mgr), "session_key": "", "history": history,
        "history_lock": threading.Lock(), "history_version": 0, "running": False, "attached_images": [],
        "cwd": str(project), "profile_home": str(profile)}
    token = set_hermes_home_override(str(launch))  # the RPC thread holds the launch profile
    try:
        result = server._methods["session.undo"]("rid", {"session_id": "undo-hint-sid"})["result"]
    finally:
        reset_hermes_home_override(token)
        server._sessions.pop("undo-hint-sid", None)
    assert result["removed"] == 2
    assert result["undo_hint"]["number"] == 1 and result["undo_hint"]["files"] == ["app.js"], result
