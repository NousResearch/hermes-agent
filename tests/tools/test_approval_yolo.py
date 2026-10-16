"""The session /yolo contract every surface shares (``tools.approval_yolo``): a toggle persists before it flips
the live flag, a fresh process re-arms it from the stored copy, a row created after the toggle records it, and a
compression rotation publishes the continuation row with it (the CLI manual /compress dropped it before)."""

import pytest

from hermes_state import SessionDB
from tools import approval
from tools.approval_yolo import restore_session_yolo, toggle_session_yolo, transfer_session_yolo, with_session_yolo


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    database = SessionDB(db_path=tmp_path / "state.db")
    yield database
    for key in ("s1", "s2"):
        approval.clear_session(key)
    database.close()


def test_toggle_restore_and_rotation_round_trip_through_the_row(db):
    # Toggled before the row exists: the lazily created row records it.
    assert toggle_session_yolo("s1", persist=lambda on: db.set_session_yolo("s1", on)) is True
    db.create_session("s1", source="cli", model_config=with_session_yolo({"max_iterations": 5}, "s1"))
    approval.clear_session("s1")  # a new process: the in-memory set starts empty
    assert restore_session_yolo("s1", SessionDB.session_yolo_enabled(db.get_session("s1"))) is True
    assert approval.is_session_yolo_enabled("s1")

    # A rotation's continuation row is created carrying the live flag, which then moves to the new key.
    db.create_session("s2", source="cli", model_config=with_session_yolo({}, "s1"), parent_session_id="s1")
    transfer_session_yolo("s1", "s2")
    assert approval.is_session_yolo_enabled("s2") and not approval.is_session_yolo_enabled("s1")
    approval.clear_session("s2")
    assert restore_session_yolo("s2", SessionDB.session_yolo_enabled(db.get_session("s2"))) is True

    # After a restart only the stored copy is ON; a toggle switches it OFF and the OFF persists.
    approval.clear_session("s2")
    assert toggle_session_yolo("s2", persisted=True, persist=lambda on: db.set_session_yolo("s2", on)) is False
    assert not SessionDB.session_yolo_enabled(db.get_session("s2"))
    assert restore_session_yolo("s2", False) is False and not approval.is_session_yolo_enabled("s2")

    # A failing write never blocks the live flip; a frozen process --yolo is never mirrored into the session set.
    def broken(_on):
        raise OSError("disk full")
    assert toggle_session_yolo("s2", persist=broken) is True and approval.is_session_yolo_enabled("s2")
    approval.clear_session("s2")
    approval._YOLO_MODE_FROZEN = True
    try:
        assert restore_session_yolo("s2", True) is False and not approval.is_session_yolo_enabled("s2")
    finally:
        approval._YOLO_MODE_FROZEN = False


def test_compression_rotation_publishes_the_child_row_with_the_live_flag(db):
    import types
    from agent import conversation_compression as cc
    db.create_session("s1", source="cli")
    db.append_message("s1", "user", "hello")
    db.append_message("s1", "assistant", "world")
    approval.enable_session_yolo("s1")
    agent = types.SimpleNamespace(
        _session_db=db, session_id="s1", platform="cli", model="m", _session_init_model_config={"max_iterations": 5},
        working_directory=None, _memory_manager=None, context_compressor=types.SimpleNamespace(),
        _flush_messages_to_session_db=lambda *a, **k: None, _persist_user_message_idx=None,
        _session_messages=None, _gateway_session_key=None, _cached_system_prompt="sys")
    cc._publish_rotated_compaction(
        agent, [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}],
        [{"role": "user", "content": "[handoff]"}],
        new_system_prompt="sys", lease=types.SimpleNamespace(holder=None, ttl=60.0, watermark=None),
        old_session_id="s1", compressed_user_turn_outcome="none")
    assert agent.session_id != "s1"
    assert SessionDB.session_yolo_enabled(db.get_session(agent.session_id))
    approval.clear_session(agent.session_id)
