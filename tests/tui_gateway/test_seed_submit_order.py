"""A failed deferred branch seed must precede every accepted submit row."""

import errno

import pytest

from tests.tui_gateway.test_seeded_branch_retry import _create_branch, _prepare_branch
from tests.tui_gateway import test_stranded_session_adoption as adoption_cases

gateway = adoption_cases.gateway


@pytest.mark.parametrize("disk_full, error_code", [(False, 5071), (True, 5070)])
def test_deferred_seed_chunk_failure_refuses_submit_and_retries_once(gateway, monkeypatch, disk_full, error_code):
    mod, db, _ = gateway
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"seed {i}"}
                for i in range(501)]
    _prepare_branch(mod, db, monkeypatch, messages)
    parent_before = db.get_messages("seed-retry-parent", include_inactive=True)

    def defer_seed(*args, **kwargs):
        raise OSError(errno.ENOSPC, "defer initial seed")

    with monkeypatch.context() as patch:
        patch.setattr(db, "append_messages_batch", defer_seed)
        created = _create_branch(mod)
    assert "result" in created, created
    record = mod._sessions[created["result"]["session_id"]]
    child_before = db.get_session("seed-retry-child")
    assert db.get_messages("seed-retry-child") == []
    append_transaction = db._append_messages_in_transaction
    chunks = []

    def fail_second_chunk(conn, key, rows, **kwargs):
        if key == "seed-retry-child":
            chunks.append(len(rows))
            if len(chunks) == 2:
                assert len(db.get_messages(key)) == 500
                if disk_full:
                    raise OSError(errno.ENOSPC, "second chunk disk full")
                raise RuntimeError("second chunk failed")
        return append_transaction(conn, key, rows, **kwargs)

    record["running"] = True
    with monkeypatch.context() as patch:
        patch.setattr(db, "_append_messages_in_transaction", fail_second_chunk)
        refused = mod._persist_session_row_for_submit("blocked", record, "rejected followup")
    committed_prefix = db.get_messages("seed-retry-child")
    assert chunks[:2] == [500, 1]
    assert refused and refused.get("error"), (refused, [row["content"] for row in committed_prefix[-3:]])
    assert refused["error"]["code"] == error_code
    assert record["running"] is False
    assert not record.get("_branch_seed_persisted")
    assert record.get("_branch_seed_pending") is not None
    assert [row["content"] for row in committed_prefix] == [f"seed {i}" for i in range(500)]

    for text in ("first accepted followup", "second accepted followup"):
        assert mod._persist_session_row_for_submit(text, record, text) is None
    rows = db.get_messages("seed-retry-child", include_inactive=True)
    assert [row["content"] for row in rows] == [
        *(f"seed {i}" for i in range(501)), "first accepted followup", "second accepted followup",
    ]
    assert [row["id"] for row in rows[:500]] == [row["id"] for row in committed_prefix]
    assert record.get("_branch_seed_persisted")
    assert "_branch_seed_pending" not in record
    assert db.get_session("seed-retry-child")["started_at"] == child_before["started_at"]
    assert db.get_messages("seed-retry-parent", include_inactive=True) == parent_before


def test_transient_missing_seed_store_refuses_before_recovered_user_write(gateway, monkeypatch):
    from contextlib import contextmanager

    mod, db, _ = gateway
    _prepare_branch(mod, db, monkeypatch, [{"role": "user", "content": "seed"}])

    def defer_seed(*args, **kwargs):
        raise OSError(errno.ENOSPC, "defer initial seed")

    with monkeypatch.context() as patch:
        patch.setattr(db, "append_messages_batch", defer_seed)
        created = _create_branch(mod)
    assert "result" in created, created
    record = mod._sessions[created["result"]["session_id"]]
    real_session_db = mod._session_db
    opens = []

    @contextmanager
    def temporarily_unavailable(session):
        opens.append(session["session_key"])
        if len(opens) == 1:
            yield None
        else:
            with real_session_db(session) as owner:
                yield owner

    with monkeypatch.context() as patch:
        patch.setattr(mod, "_session_db", temporarily_unavailable)
        refused = mod._persist_session_row_for_submit("blocked", record, "rejected followup")
    rows = db.get_messages("seed-retry-child")
    assert refused and refused.get("error"), (refused, [row["content"] for row in rows])
    assert refused["error"]["code"] == 5071
    assert rows == []
    assert record.get("_branch_seed_pending") is not None
    assert mod._persist_session_row_for_submit("retry", record, "accepted followup") is None
    assert [row["content"] for row in db.get_messages("seed-retry-child")] == ["seed", "accepted followup"]


@pytest.mark.parametrize("mode", ["unseeded", "empty_history", "no_key"])
def test_optional_store_without_pending_seed_keeps_best_effort_submit(gateway, monkeypatch, mode):
    from contextlib import nullcontext

    mod, db, _ = gateway
    _prepare_branch(mod, db, monkeypatch, [{"role": "user", "content": "seed"}])
    created = _create_branch(mod)
    assert "result" in created, created
    record = mod._sessions[created["result"]["session_id"]]
    record.pop("_branch_seed_persisted", None)
    if mode == "unseeded":
        record["seeded"] = False
    elif mode == "empty_history":
        record["history"] = []
    else:
        record["session_key"] = ""
    monkeypatch.setattr(mod, "_get_db", lambda: None)
    monkeypatch.setattr(mod, "_session_db", lambda session: nullcontext(None))
    monkeypatch.setattr(mod, "_db_error", None)
    assert mod._persist_session_row_for_submit("optional", record, "followup") is None
    assert "_branch_seed_pending" not in record
