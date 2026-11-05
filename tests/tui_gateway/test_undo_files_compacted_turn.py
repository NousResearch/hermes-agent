"""The TUI sidecar's ``session.undo`` finds a turn's file writes after compaction restated its row.

Compaction replaces the in-flight user row with a restatement stamped at compaction time but keeps
its ``message_uid``; the turn's writes are keyed by the ORIGINAL stamp. The sidecar must address the
rewound turn by its durable row (as the shared owner's ``rollback.undo_files`` does), not by the
restated row's timestamp.
"""
import pytest

from tests.tui_gateway import test_composite_carrier_rewind as carrier
from tests.tui_gateway.test_composite_carrier_rewind import _composite_carrier, _session_undo
from tools import checkpoint_manager as cm

session_fixture = pytest.fixture(name="carrier_session")(carrier.carrier_session.__wrapped__)


def test_sidecar_undo_reverts_the_files_of_a_turn_compaction_restated(carrier_session, tmp_path, monkeypatch):
    db, install = carrier_session
    sid, key, session = install([{"role": "user", "content": "REAL ASK"}])
    row = db._read_one("SELECT id, timestamp, message_uid FROM messages WHERE session_id=? AND role='user'", (key,))
    restated = db.append_message(key, "user", _composite_carrier()["content"])
    db.append_message(key, "assistant", "done")
    with db._lock:
        db._conn.execute("UPDATE messages SET active=0 WHERE id=?", (row["id"],))
        db._conn.execute("UPDATE messages SET timestamp=?, message_uid=? WHERE id=?",
                         (row["timestamp"] + 50, row["message_uid"], restated))
        db._conn.commit()
    session["history"] = db.get_messages_as_conversation(key)
    session["agent"]._session_messages = session["history"]

    monkeypatch.setattr(cm, "CHECKPOINT_BASE", tmp_path / "checkpoints")
    work = tmp_path / "project"
    work.mkdir()
    path = work / "app.py"
    path.write_text("original\n")
    mgr = cm.CheckpointManager(enabled=True)
    mgr.begin_turn(row["timestamp"], key)  # the turn began under the original row's stamp
    mgr.ensure_checkpoint(str(work), "before write_file")
    mgr.note_pending_write(str(path))
    path.write_text("agent edit\n")
    mgr.record_agent_write(str(path))
    session["agent"]._checkpoint_mgr = mgr
    session["agent"].session_id = key
    session["cwd"] = str(work)

    response = _session_undo(sid)

    assert response["result"]["removed"] > 0, response
    assert path.read_text() == "original\n", response
    assert any("app.py" in line for line in response["result"]["files"]), response
