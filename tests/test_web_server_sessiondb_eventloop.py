import asyncio
import contextvars
import threading

import pytest

from hermes_cli import web_server


def test_sessiondb_runner_owns_connection_on_worker_and_preserves_context(monkeypatch):
    loop_thread = threading.get_ident()
    request_marker = contextvars.ContextVar("request_marker", default="missing")
    request_marker.set("dashboard-request")
    events: list[tuple[str, int, str]] = []

    class _DB:
        def close(self):
            events.append(("close", threading.get_ident(), request_marker.get()))

    def _open(profile=None):
        assert profile == "work"
        events.append(("open", threading.get_ident(), request_marker.get()))
        return _DB()

    def _operation(db):
        assert isinstance(db, _DB)
        events.append(("work", threading.get_ident(), request_marker.get()))
        return "done"

    monkeypatch.setattr(web_server, "_open_session_db_for_profile", _open)

    result = asyncio.run(web_server._run_session_db("work", _operation))

    assert result == "done"
    assert [event[0] for event in events] == ["open", "work", "close"]
    worker_threads = {event[1] for event in events}
    assert len(worker_threads) == 1
    assert loop_thread not in worker_threads
    assert {event[2] for event in events} == {"dashboard-request"}


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: web_server.get_sessions(limit=1),
        lambda: web_server.search_sessions(q="needle", limit=1),
        lambda: web_server.bulk_delete_sessions_endpoint(
            web_server.BulkDeleteSessions(ids=["one"])
        ),
        lambda: web_server.count_empty_sessions_endpoint(),
        lambda: web_server.delete_empty_sessions_endpoint(),
        lambda: web_server.get_session_stats(),
        lambda: web_server.get_session_detail("session"),
        lambda: web_server.get_session_latest_descendant("session"),
        lambda: web_server.get_session_messages("session"),
        lambda: web_server.delete_session_endpoint("session"),
        lambda: web_server.rename_session_endpoint(
            "session", web_server.SessionRename(title="renamed")
        ),
        lambda: web_server.export_session_endpoint("session"),
        lambda: web_server.prune_sessions_endpoint(
            web_server.SessionPrune(dry_run=True)
        ),
        lambda: web_server.get_usage_analytics(),
        lambda: web_server.get_models_analytics(),
    ],
)
def test_sessiondb_handlers_route_through_shared_runner(monkeypatch, invoke):
    sentinel = object()
    calls = []

    async def _run(profile, operation):
        calls.append((profile, operation))
        return sentinel

    monkeypatch.setattr(web_server, "_run_session_db", _run)

    assert asyncio.run(invoke()) is sentinel
    assert len(calls) == 1


def test_bulk_delete_sessiondb_work_runs_off_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    db_threads: list[int] = []

    class _DB:
        def delete_sessions(self, ids):
            db_threads.append(threading.get_ident())
            assert ids == ["one", "two"]
            return 2

        def close(self):
            db_threads.append(threading.get_ident())

    monkeypatch.setattr(web_server, "_open_session_db_for_profile", lambda profile=None: _DB())

    result = asyncio.run(
        web_server.bulk_delete_sessions_endpoint(
            web_server.BulkDeleteSessions(ids=["one", "two"])
        )
    )

    assert result == {"ok": True, "deleted": 2}
    assert db_threads
    assert all(thread_id != loop_thread for thread_id in db_threads)
