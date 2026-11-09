"""Pending seed evidence follows committed transcript replacements, not stale history."""
import errno
from types import SimpleNamespace

import pytest

from tests.tui_gateway.test_seeded_branch_retry import _create_branch, _prepare_branch
from tests.tui_gateway import test_stranded_session_adoption as adoption_cases

gateway = adoption_cases.gateway


@pytest.fixture
def pending_branch(gateway, monkeypatch, request):
    mod, db, _ = gateway
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"seed {i}"}
                for i in range(501)]
    messages[-1]["role"] = getattr(request, "param", "user")
    _prepare_branch(mod, db, monkeypatch, messages)
    append = db.append_messages_batch

    def partial(key, rows, **kwargs):
        if key == "seed-retry-child":
            append(key, rows[:500])
            raise OSError(errno.ENOSPC, "injected partial seed disk full")
        return append(key, rows, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(db, "append_messages_batch", partial)
        created = _create_branch(mod)
    assert "result" in created, created
    sid = created["result"]["session_id"]
    record = mod._sessions[sid]
    assert len(db.get_messages("seed-retry-child")) == 500
    return mod, db, sid, record


def _contents(db):
    return [row["content"] for row in db.get_messages("seed-retry-child")]


@pytest.mark.parametrize("pending_branch", ["assistant"], indirect=True)
@pytest.mark.parametrize("failed", [False, True])
def test_durable_undo_retires_seed_only_after_commit(pending_branch, monkeypatch, failed):
    mod, db, _, record = pending_branch
    pending = record["_branch_seed_pending"]
    if failed:
        with monkeypatch.context() as patch:
            def refuse(*args, **kwargs):
                raise OSError(errno.ENOSPC, "injected rewind failure")
            patch.setattr(db, "rewind_user_turn", refuse)
            with record["history_lock"], pytest.raises(OSError):
                mod._rewind_active_session_history(record, 125)
        assert record["_branch_seed_pending"] is pending
        expected = 501
    else:
        with record["history_lock"]:
            mod._rewind_active_session_history(record, 125)
        expected = 250
    assert mod._persist_session_row_for_submit("submit", record, "replacement") is None
    assert _contents(db) == [*(f"seed {i}" for i in range(expected)), "replacement"]
    assert len(db.get_messages("seed-retry-parent")) == 501


def test_failed_truncate_keeps_pending_seed_retryable(pending_branch, monkeypatch):
    mod, db, sid, record = pending_branch
    pending = record["_branch_seed_pending"]
    before = _contents(db)
    row_id = db.get_messages("seed-retry-child")[250]["id"]

    def refuse(*args, **kwargs):
        raise OSError(errno.ENOSPC, "injected truncate failure")

    with monkeypatch.context() as patch:
        patch.setattr(db, "replace_messages", refuse)
        with record["history_lock"]:
            error, _ = mod._truncate_history_for_submit("rewind", sid, record, {
                "truncate_before_row_id": row_id, "confirm_truncate": True,
                "confirm_deep_truncate": True,
            }, None)
    assert error and error.get("error")
    assert _contents(db) == before
    assert record["_branch_seed_pending"] is pending
    assert mod._persist_session_row_for_submit("submit", record, "replacement") is None
    assert _contents(db) == [*(f"seed {i}" for i in range(501)), "replacement"]


def test_changed_history_refuses_pending_seed_before_append(pending_branch, monkeypatch):
    mod, db, _, record = pending_branch
    before = _contents(db)
    record["history_version"] += 1
    record["history"] = record["history"][:250]
    appended = []
    append = db.append_messages_batch

    def observed(*args, **kwargs):
        appended.append(args)
        return append(*args, **kwargs)

    monkeypatch.setattr(db, "append_messages_batch", observed)
    reply = mod._persist_session_row_for_submit("submit", record, "replacement")
    assert reply and reply.get("error"), reply
    assert not appended
    assert _contents(db) == before


@pytest.mark.parametrize("mode", ["success", "failure", "unknown"])
def test_reset_cancels_seed_only_after_success_without_clearing_guards(pending_branch, monkeypatch, mode):
    mod, db, sid, record = pending_branch
    pending = record["_branch_seed_pending"]
    fresh = record["_fresh_generated_key"]
    if mode == "unknown":
        record["_seed_compensation_unverified"] = True

    def rebuild(*args, **kwargs):
        if mode == "failure":
            raise RuntimeError("injected rebuild failure")
        return SimpleNamespace()

    monkeypatch.setattr(mod, "_rebuild_session_agent", rebuild)
    monkeypatch.setattr(mod, "_session_info", lambda *args: {})
    monkeypatch.setattr(mod, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_restart_slash_worker", lambda *args: None)
    if mode == "failure":
        with pytest.raises(RuntimeError, match="rebuild failure"):
            mod._reset_session_agent(sid, record)
        assert record["_branch_seed_pending"] is pending
    else:
        mod._reset_session_agent(sid, record)
        assert record["history"] == []
    assert record["_fresh_generated_key"] is fresh
    reply = mod._persist_session_row_for_submit("submit", record, "replacement")
    if mode == "unknown":
        assert record["_seed_compensation_unverified"] is True
        assert reply and reply.get("error")
        assert _contents(db) == [f"seed {i}" for i in range(500)]
    else:
        assert reply is None, reply
        expected = 501 if mode == "failure" else 500
        assert _contents(db) == [*(f"seed {i}" for i in range(expected)), "replacement"]


@pytest.mark.parametrize("mode", ["settled", "disk_full", "unsettled", "unknown", "preview"])
def test_manual_compression_settles_seed_before_running(pending_branch, monkeypatch, mode):
    from agent import conversation_compression_manual as manual

    mod, db, _, record = pending_branch
    record["agent"] = SimpleNamespace()
    if mode == "unknown":
        record["_seed_compensation_unverified"] = True
    called = []

    def compress(agent, history, request, **kwargs):
        called.append(list(_contents(db)))
        if not request.preview:
            assert len(_contents(db)) == 501, "compression ran ahead of pending seed"
        return manual.CompressResult(
            "preview" if request.preview else "compressed", history, list(history), 1, 1, request)

    def refuse(*args, **kwargs):
        if mode == "disk_full":
            raise OSError(errno.ENOSPC, "injected seed retry disk full")
        raise RuntimeError("injected seed retry failure")

    monkeypatch.setattr(manual, "compress_now", compress)
    monkeypatch.setattr(mod, "_get_usage", lambda *args: {})
    if mode in {"disk_full", "unsettled"}:
        monkeypatch.setattr(db, "append_messages_batch", refuse)
    if mode in {"disk_full", "unsettled", "unknown"}:
        with pytest.raises((OSError, RuntimeError, ValueError)):
            mod._compress_session_history(record)
        assert not called
        assert len(_contents(db)) == 500
    else:
        mod._compress_session_history(record, "--preview" if mode == "preview" else None)
        assert len(called) == 1
        if mode == "preview":
            assert len(_contents(db)) == 500
            assert "_branch_seed_pending" in record
        else:
            assert len(_contents(db)) == 501
            assert "_branch_seed_pending" not in record
            assert mod._persist_session_row_for_submit("submit", record, "replacement") is None
            assert _contents(db) == [*(f"seed {i}" for i in range(501)), "replacement"]


def test_pending_seed_serializes_with_competing_rewind(pending_branch, monkeypatch):
    import threading

    mod, db, sid, record = pending_branch
    entered, release, decision, blocked = (threading.Event() for _ in range(4))
    errors = []
    append = db.append_messages_batch
    row_id = db.get_messages("seed-retry-child")[250]["id"]

    def paused(key, rows, **kwargs):
        if key == "seed-retry-child":
            entered.set()
            assert release.wait(10), "seed write was never released"
        return append(key, rows, **kwargs)

    def seed():
        try:
            mod._persist_branch_seed(record)
        except BaseException as exc:
            errors.append(exc)

    def rewind():
        acquired = record["history_lock"].acquire(blocking=False)
        if not acquired:
            blocked.set()
        decision.set()
        try:
            if not acquired:
                record["history_lock"].acquire()
            error, _ = mod._truncate_history_for_submit("rewind", sid, record, {
                "truncate_before_row_id": row_id, "confirm_truncate": True,
                "confirm_deep_truncate": True,
            }, None)
            assert error is None, error
        except BaseException as exc:
            errors.append(exc)
        finally:
            record["history_lock"].release()

    monkeypatch.setattr(db, "append_messages_batch", paused)
    writer = threading.Thread(target=seed)
    mutator = threading.Thread(target=rewind)
    writer.start()
    try:
        assert entered.wait(5), "seed never reached its write"
        mutator.start()
        assert decision.wait(5), "rewind never attempted history ownership"
        assert blocked.is_set(), "rewind acquired history while seed was writing"
    finally:
        release.set()
        writer.join(10)
        if mutator.ident is not None:
            mutator.join(10)
    assert not writer.is_alive() and not mutator.is_alive()
    assert not errors
    assert _contents(db) == [f"seed {i}" for i in range(250)]
