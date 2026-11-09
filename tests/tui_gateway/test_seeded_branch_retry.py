"""Seed-copy failures must preserve the fresh child's proven ownership on retry."""

import errno

import pytest

from hermes_state import SessionDB
from tests.tui_gateway import test_stranded_session_adoption as adoption_cases
from tests.tui_gateway.test_protocol import _stub_session_create_dependencies

gateway = adoption_cases.gateway


def _prepare_branch(mod, db, monkeypatch, messages):
    ensure, seed = mod._ensure_session_db_row, mod._seed_branch_row
    _stub_session_create_dependencies(mod, monkeypatch)
    monkeypatch.setattr(mod, "_ensure_session_db_row", ensure)
    monkeypatch.setattr(mod, "_seed_branch_row", seed)
    monkeypatch.setattr(mod, "_new_session_key", lambda: "seed-retry-child")
    db.create_session("seed-retry-parent", source="desktop", system_prompt="cached parent prompt")
    db.append_messages_batch("seed-retry-parent", messages)


def _create_branch(mod):
    return mod.handle_request({
        "id": "create", "method": "session.branch_stored",
        "params": {"parent_session_id": "seed-retry-parent", "source": "desktop"},
    })


@pytest.mark.parametrize("fault", ["before_copy", "after_chunk", "title", "lazy_after_chunk"])
def test_branch_disk_full_retries_owned_row_without_losing_or_duplicating_seed(gateway, monkeypatch, fault):
    mod, db, _ = gateway
    messages = [{"role": "user", "content": f"seed {i}"} for i in range(501)]
    _prepare_branch(mod, db, monkeypatch, messages)
    append, title = db.append_messages_batch, db.set_auto_title

    def fail_copy(key, rows, **kwargs):
        if key == "seed-retry-child":
            if fault == "after_chunk":
                append(key, rows[:500])
            raise OSError(errno.ENOSPC, "injected seed-copy disk full")
        return append(key, rows, **kwargs)

    def fail_title(key, value, **kwargs):
        if key == "seed-retry-child":
            raise OSError(errno.ENOSPC, "injected title disk full")
        return title(key, value, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(db, "set_auto_title" if fault == "title" else "append_messages_batch",
                      fail_title if fault == "title" else fail_copy)
        created = _create_branch(mod)
    assert "result" in created, created
    record = mod._sessions[created["result"]["session_id"]]
    child_before = db.get_session("seed-retry-child")
    assert child_before is not None
    expected_before = {"before_copy": 0, "after_chunk": 500, "title": 501, "lazy_after_chunk": 0}[fault]
    assert len(db.get_messages("seed-retry-child")) == expected_before

    if fault == "lazy_after_chunk":
        def fail_retry_copy(key, rows, **kwargs):
            if key == "seed-retry-child":
                append(key, rows[:500])
                raise OSError(errno.ENOSPC, "injected partial lazy seed disk full")
            return append(key, rows, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(db, "append_messages_batch", fail_retry_copy)
            blocked = mod._persist_session_row_for_submit("blocked", record, "blocked followup")
        assert blocked and blocked.get("error")
        assert len(db.get_messages("seed-retry-child")) == 500

    for text in ("first followup", "second followup"):
        assert mod._persist_session_row_for_submit(text, record, text) is None

    child_after = db.get_session("seed-retry-child")
    assert child_after["started_at"] == child_before["started_at"]
    assert child_after["system_prompt"] == "cached parent prompt"
    assert child_after["parent_session_id"] == "seed-retry-parent"
    assert [row["content"] for row in db.get_messages("seed-retry-child")] == [
        *(f"seed {i}" for i in range(501)), "first followup", "second followup",
    ]
    assert [row["content"] for row in db.get_messages("seed-retry-parent")] == [
        f"seed {i}" for i in range(501)
    ]


@pytest.mark.parametrize("mode", ["retry", "local_collision", "foreign_collision", "discard_false",
                                  "discard_failed", "discard_unknown"])
def test_branch_compensation_does_not_authorize_an_unproven_row(gateway, monkeypatch, mode):
    mod, db, _ = gateway
    _prepare_branch(mod, db, monkeypatch, [{"role": "user", "content": "seed"}])
    discard = db.discard_unadmitted_session

    def fail_copy(*args, **kwargs):
        raise RuntimeError("injected seed-copy failure")

    def fail_discard(key):
        if mode == "discard_unknown":
            discard(key)
        raise RuntimeError("discard outcome unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(db, "append_messages_batch", fail_copy)
        if mode == "discard_false":
            patch.setattr(db, "discard_unadmitted_session", lambda key: False)
        elif mode in ("discard_failed", "discard_unknown"):
            patch.setattr(db, "discard_unadmitted_session", fail_discard)
        created = _create_branch(mod)
    assert "result" in created, created
    record = mod._sessions[created["result"]["session_id"]]
    sibling = None
    try:
        if mode == "local_collision":
            db.create_session("seed-retry-child", source="cli")
            db.append_message("seed-retry-child", "user", "unrelated")
        elif mode == "foreign_collision":
            profile = db.db_path.parent / "profiles" / "science"
            profile.mkdir()
            sibling = SessionDB(profile / "state.db")
            sibling.create_session("seed-retry-child", source="cli")
            sibling.append_message("seed-retry-child", "user", "foreign")
        stores = (db,) if sibling is None else (db, sibling)

        def snapshot():
            return [[tuple(row) for row in store._conn.execute("SELECT * FROM " + table + " ORDER BY rowid")]
                    for store in stores for table in ("sessions", "messages", "state_meta")]

        before = snapshot()
        reply = mod._persist_session_row_for_submit("submit", record, "followup")
        if mode == "retry":
            assert reply is None
            assert [row["content"] for row in db.get_messages("seed-retry-child")] == ["seed", "followup"]
        else:
            assert reply and reply.get("error"), reply
            assert snapshot() == before
    finally:
        if sibling is not None:
            sibling.close()


@pytest.mark.parametrize("cut", [0, 250])
def test_branch_seed_retry_honors_a_committed_rewind(gateway, monkeypatch, cut):
    mod, db, _ = gateway
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"seed {i}"}
                for i in range(501)]
    _prepare_branch(mod, db, monkeypatch, messages)
    append = db.append_messages_batch

    def fail_after_chunk(key, rows, **kwargs):
        if key == "seed-retry-child":
            append(key, rows[:500])
            raise OSError(errno.ENOSPC, "injected partial seed disk full")
        return append(key, rows, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(db, "append_messages_batch", fail_after_chunk)
        created = _create_branch(mod)
    assert "result" in created, created
    sid = created["result"]["session_id"]
    record = mod._sessions[sid]
    before = db.get_messages("seed-retry-child")
    assert len(before) == 500
    with record["history_lock"]:
        error, _ = mod._truncate_history_for_submit("rewind", sid, record, {
            "truncate_before_row_id": before[cut]["id"],
            "confirm_truncate": True,
            "confirm_deep_truncate": True,
            "confirm_empty_truncate": True,
        }, None)
    assert error is None, error
    assert mod._persist_session_row_for_submit("submit", record, "replacement") is None
    assert [row["content"] for row in db.get_messages("seed-retry-child")] == [
        *(f"seed {i}" for i in range(cut)), "replacement",
    ]
    archived = db._conn.execute(
        "SELECT content FROM messages WHERE session_id=? AND active=0 ORDER BY id",
        ("seed-retry-child",)).fetchall()
    assert [row["content"] for row in archived] == [f"seed {i}" for i in range(cut, 500)]
    assert [row["content"] for row in db.get_messages("seed-retry-parent")] == [
        f"seed {i}" for i in range(501)
    ]
