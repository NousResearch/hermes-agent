"""Warm resume must return a coherent draft or report that the runtime closed."""

import asyncio
import threading
from contextlib import closing

import pytest
import pytest_asyncio

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session
from tests.tui_gateway.test_resume_lock_contention import _SignallingLock
from tests.tui_gateway.test_ws_concurrent_reads import Socket, rpc
from tests.waiting import async_wait_until
from tui_gateway import server, ws


@pytest_asyncio.fixture
async def chats(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(ws, "_start_backend_liveness", lambda: None)
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "_ensure_lease_watcher", lambda: None)
    monkeypatch.setattr(server, "resolve_skin", dict)
    db = open_persistence(profile_scope(home=tmp_path))
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    with closing(db):
        for name in ("a", "b"):
            cwd = tmp_path / name
            cwd.mkdir()
            server._sessions[name] = server._deferred_session_record(
                f"chat-{name}", cols=80, cwd=str(cwd), lease=None,
                history=[{"role": "user", "content": name}], source="desktop",
            )
        sock = Socket()
        serving = asyncio.create_task(ws.handle_ws(sock))
        try:
            yield sock, db
        finally:
            sock.inbound.put_nowait(None)
            await asyncio.wait_for(serving, 10)
            for sid in list(server._sessions):
                server._close_session_by_id(sid, end_reason="test_cleanup")


def send(sock, rid, method, key, **params):
    frame = rpc(rid, method)
    frame["params"] = {"session_id": key, **params}
    sock.inbound.put_nowait(frame)


def reply(sock, rid):
    return next((frame for frame in sock.sent if frame.get("id") == rid), None)


@pytest.mark.asyncio
async def test_full_draft_waits_for_its_own_history_but_not_other_chats(chats):
    sock, _ = chats
    draft = server._sessions["a"]
    lock, attempted = draft["history_lock"], threading.Event()
    draft["history_lock"] = _SignallingLock(lock, attempted)
    with lock:
        send(sock, "draft", "session.resume", "chat-a")
        await async_wait_until(lambda: attempted.is_set() or reply(sock, "draft"))
        send(sock, "other", "session.resume", "chat-b", omit_messages=True)
        other = (await async_wait_until(lambda: reply(sock, "other")))["result"]
        assert (other["session_id"], other["stored_session_id"]) == ("b", "chat-b")
        draft["history"].append({"role": "assistant", "content": "second"})
    result = (await async_wait_until(lambda: reply(sock, "draft")))["result"]
    assert result["session_id"] == "a"
    assert result["message_count"] == 2
    assert [(m["role"], m["text"]) for m in result["messages"]] == [("user", "a"), ("assistant", "second")]


@pytest.mark.asyncio
@pytest.mark.parametrize("close", [False, True], ids=["still-live", "closed"])
async def test_warm_reply_after_metadata_read(chats, monkeypatch, close):
    sock, db = chats
    for key in ("chat-a", "chat-b"):
        seed_session(db, key, "desktop")
    a = server._sessions["a"]
    started, release = threading.Event(), threading.Event()
    original = server._fallback_session_info

    def gated_info(session):
        info = original(session)
        if info["cwd"] == a["cwd"]:
            started.set()
            assert release.wait(20), "test did not release A's info"
        return info

    monkeypatch.setattr(server, "_fallback_session_info", gated_info)
    send(sock, "warm", "session.resume", "chat-a", omit_messages=True)
    try:
        await async_wait_until(started.is_set)
        send(sock, "other", "session.resume", "chat-b", omit_messages=True)
        other = (await async_wait_until(lambda: reply(sock, "other")))["result"]
        assert (other["session_id"], other["resumed"]) == ("b", "chat-b")
        if close:
            send(sock, "close", "session.close", "a")
            assert (await async_wait_until(lambda: reply(sock, "close")))["result"]["closed"]
            assert "a" not in server._sessions
        assert reply(sock, "warm") is None
    finally:
        release.set()
    late = await async_wait_until(lambda: reply(sock, "warm"))
    if close:
        assert "result" not in late
        assert late["error"]["code"] == 4007
    else:
        assert (late["result"]["session_id"], late["result"]["info"]["cwd"]) == ("a", a["cwd"])
