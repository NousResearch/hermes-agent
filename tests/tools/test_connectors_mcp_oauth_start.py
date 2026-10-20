"""Regression tests for the card OAuth ``start()`` ownership handoff in
``tools/connectors/mcp_oauth.py``.

The carded path replaces ``_ACTIVE[(hermes_home, server_name)]`` with the new
flow and cancels the older attempt. ``probe_with_rollback.undo`` skips its
restore whenever a *different* flow owns the slot, so a new flow that never
became viable must not remain the owner: the older attempt's rollback has to
stay able to put the pre-attempt token snapshot back.
"""

import asyncio
import threading

import pytest

from hermes_constants import get_hermes_home
from tools.connectors import mcp_oauth
from tools.mcp_dashboard_oauth import DashboardOAuthFlow
from tui_gateway import mcp_oauth_sessions


@pytest.fixture(autouse=True)
def _clean_oauth_state():
    mcp_oauth._ACTIVE.clear()
    mcp_oauth_sessions._sessions.clear()
    yield
    mcp_oauth._ACTIVE.clear()
    mcp_oauth_sessions._sessions.clear()


def _home() -> str:
    return str(get_hermes_home().expanduser().resolve(strict=False))


def _cfg() -> dict:
    return {"url": "https://mcp.example.com/mcp", "auth": "oauth"}


def _flow(server_name: str, flow_id: str) -> DashboardOAuthFlow:
    return DashboardOAuthFlow(flow_id, server_name, None, _home(), "")


class _FakeHttpd:
    def __init__(self):
        self.shutdown_calls = 0
        self.server_close_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1

    def server_close(self):
        self.server_close_calls += 1


def _seed_older(server_name: str = "srv") -> DashboardOAuthFlow:
    older = _flow(server_name, "older-flow")
    older.backup = {"saved": "tokens"}
    mcp_oauth._ACTIVE[(_home(), server_name)] = older
    return older


def test_a_failed_receiver_bind_leaves_the_older_attempt_in_place(monkeypatch):
    """choose_callback_receiver raises before the swap: _ACTIVE keeps the older
    flow, it is never canceled, and its undo() still restores the snapshot."""
    older = _seed_older()

    class _Storage:
        instances = []

        def __init__(self, server_name):
            self.server_name = server_name
            self.restored = []
            _Storage.instances.append(self)

        def snapshot(self):
            return {"baseline": self.server_name}

        def restore(self, backup):
            self.restored.append(backup)

    class _Manager:
        def __init__(self):
            self.restored_entries = []

        def remove(self, server_name, hermes_home=None):
            return {"previous": server_name}

        def restore_entry(self, server_name, entry, hermes_home=None):
            self.restored_entries.append((server_name, entry))

    manager = _Manager()
    probe_started = threading.Event()
    probe_gate = threading.Event()

    def fake_probe(server_name, cfg, connect_timeout=None, details=None):
        probe_started.set()
        probe_gate.wait(10)
        raise RuntimeError("probe failed")

    monkeypatch.setattr("tools.mcp_oauth.HermesTokenStorage", _Storage)
    monkeypatch.setattr("tools.mcp_oauth_manager.get_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", fake_probe)

    worker_errors = []

    def older_worker():
        try:
            mcp_oauth.probe_with_rollback("srv", _cfg(), _home(), older, False)
        except Exception as exc:  # the probe failure we injected
            worker_errors.append(exc)

    thread = threading.Thread(target=older_worker, daemon=True)
    thread.start()
    assert probe_started.wait(5)

    monkeypatch.setattr(
        mcp_oauth,
        "choose_callback_receiver",
        lambda flow, cfg, uri: (_ for _ in ()).throw(ValueError("bad redirect URI")),
    )
    with pytest.raises(ValueError):
        mcp_oauth.start("srv", cfg=_cfg(), url_timeout=5)

    try:
        assert mcp_oauth._ACTIVE[(_home(), "srv")] is older
        assert not older.cancelled
    finally:
        probe_gate.set()
    thread.join(5)
    assert worker_errors, "the older worker should have observed the probe failure"

    assert _Storage.instances[0].restored == [{"baseline": "srv"}]
    assert manager.restored_entries == [("srv", {"previous": "srv"})]


def test_a_failed_session_registration_restores_the_older_attempt(monkeypatch):
    """register_flow raises after the swap: _ACTIVE must hand ownership back so
    the older flow's undo() does not see a dead replacement as the owner."""
    older = _seed_older()
    httpd = _FakeHttpd()
    captured = []

    def receiver(flow, cfg, uri):
        captured.append(flow)
        return httpd

    monkeypatch.setattr(mcp_oauth, "choose_callback_receiver", receiver)

    def boom(flow, httpd=None):
        raise RuntimeError("session table exploded")

    monkeypatch.setattr(mcp_oauth_sessions, "register_flow", boom)

    with pytest.raises(RuntimeError, match="session table exploded"):
        mcp_oauth.start("srv", cfg=_cfg(), url_timeout=5)

    assert mcp_oauth._ACTIVE[(_home(), "srv")] is older
    assert not older.cancelled, "the older attempt must only be canceled once the replacement runs"
    assert captured[0].snapshot().get("status") == "error"
    assert httpd.shutdown_calls == 1
    assert httpd.server_close_calls == 1


def test_a_failed_worker_start_drops_ownership_when_there_was_no_older(monkeypatch):
    """Thread.start() raises with no older attempt: _ACTIVE must not keep the
    dead flow as the owner of the slot."""
    httpd = _FakeHttpd()
    captured = []

    def receiver(flow, cfg, uri):
        captured.append(flow)
        return httpd

    monkeypatch.setattr(mcp_oauth, "choose_callback_receiver", receiver)

    real_thread = threading.Thread

    class _Unstartable(real_thread):
        def start(self):
            raise RuntimeError("cannot spawn worker")

    monkeypatch.setattr(mcp_oauth.threading, "Thread", _Unstartable)

    with pytest.raises(RuntimeError, match="cannot spawn worker"):
        mcp_oauth.start("srv", cfg=_cfg(), url_timeout=5)

    assert (_home(), "srv") not in mcp_oauth._ACTIVE
    assert captured[0].snapshot().get("status") == "error"
    assert httpd.shutdown_calls == 1
    assert httpd.server_close_calls == 1


def test_a_successful_replacement_still_owns_the_active_slot(monkeypatch):
    """Once the new worker is running it owns the slot: the older attempt is
    canceled and its backup is inherited so rollback stays suppressed."""
    older = _seed_older()
    httpd = _FakeHttpd()
    release = threading.Event()

    monkeypatch.setattr(
        mcp_oauth, "choose_callback_receiver", lambda flow, cfg, uri: httpd
    )

    def worker(hermes_home, server_name, cfg, reconnect_live, *, flow, on_done=None, **_kw):
        try:
            asyncio.run(
                flow.publish_authorization_url(
                    "https://as.example.com/authorize?state=xyz"
                )
            )
            release.wait(10)
        finally:
            flow.mark_worker_done()
            with mcp_oauth._COMMIT_GUARD:
                if mcp_oauth._ACTIVE.get((hermes_home, server_name)) is flow:
                    mcp_oauth._ACTIVE.pop((hermes_home, server_name), None)
            if on_done is not None:
                on_done()

    monkeypatch.setattr(mcp_oauth, "run_worker", worker)

    try:
        attempt = mcp_oauth.start("srv", cfg=_cfg(), url_timeout=5)
        assert attempt.auth_url.startswith("https://as.example.com/authorize")
        assert mcp_oauth._ACTIVE[(_home(), "srv")] is attempt.flow
        assert attempt.flow.flow_id in mcp_oauth_sessions._sessions
        assert older.cancelled
        assert attempt.flow.inherited_backup == {"saved": "tokens"}
    finally:
        release.set()
    attempt.flow._worker_done.wait(5)
    assert (_home(), "srv") not in mcp_oauth._ACTIVE
    # finish_flow keeps the outcome record but releases the listener.
    assert httpd.shutdown_calls == 1 and httpd.server_close_calls == 1


def test_e2e_a_failed_authorize_retry_keeps_the_first_attempt_alive(monkeypatch):
    """Real manage_connections authorize -> _CatalogBackend -> mcp_oauth.start on the
    no-card path: a retry whose receiver setup fails must fail its own target only,
    leaving the first attempt as _ACTIVE owner, uncanceled, and a healthy retry must
    still take the slot."""
    import json

    import tools.connectors.mcp as connectors_mcp
    from tools.connectors.mcp import _CatalogBackend
    from tools.connectors.tool import manage_connections

    monkeypatch.setattr(connectors_mcp, "_catalog_names", lambda: ["srv"])
    monkeypatch.setattr(connectors_mcp, "_configured_names", lambda: ["srv"])
    monkeypatch.setattr(
        "hermes_cli.mcp_config._get_mcp_servers", lambda: {"srv": _cfg()}
    )

    first_httpd = _FakeHttpd()
    release = threading.Event()

    def worker(hermes_home, server_name, cfg, reconnect_live, *, flow, on_done=None, **_kw):
        try:
            asyncio.run(
                flow.publish_authorization_url("https://as.example.com/authorize?state=a")
            )
            release.wait(10)
        finally:
            flow.mark_worker_done()
            with mcp_oauth._COMMIT_GUARD:
                if mcp_oauth._ACTIVE.get((hermes_home, server_name)) is flow:
                    mcp_oauth._ACTIVE.pop((hermes_home, server_name), None)
            if on_done is not None:
                on_done()

    monkeypatch.setattr(mcp_oauth, "run_worker", worker)
    monkeypatch.setattr(
        mcp_oauth, "choose_callback_receiver", lambda flow, cfg, uri: first_httpd
    )

    def authorize():
        return json.loads(
            manage_connections(
                {"action": "authorize", "connectors": [{"name": "srv", "mcp": True}]},
                mcp_backend=_CatalogBackend(),
                session_id="e2e",
            )
        )

    first_out = authorize()
    first = mcp_oauth._ACTIVE[(_home(), "srv")]
    assert first_out["status"] == "initiated"
    assert first_out["targets"][0]["state"] == "initiated"
    assert first_out["targets"][0]["connect_url"].startswith(
        "https://as.example.com/authorize"
    )

    monkeypatch.setattr(
        mcp_oauth,
        "choose_callback_receiver",
        lambda flow, cfg, uri: (_ for _ in ()).throw(RuntimeError("port bind failed")),
    )
    failed_out = authorize()
    assert failed_out["targets"][0]["state"] == "failed"
    assert mcp_oauth._ACTIVE[(_home(), "srv")] is first
    assert not first.cancelled
    assert first_httpd.shutdown_calls == 0

    # The failed attempt did not wedge the slot: a healthy retry still replaces and owns it.
    monkeypatch.setattr(
        mcp_oauth, "choose_callback_receiver", lambda flow, cfg, uri: _FakeHttpd()
    )
    third_out = authorize()
    assert third_out["targets"][0]["state"] == "initiated"
    third = mcp_oauth._ACTIVE[(_home(), "srv")]
    assert third is not first
    assert first.cancelled

    release.set()
    first._worker_done.wait(5)
    third._worker_done.wait(5)
    assert (_home(), "srv") not in mcp_oauth._ACTIVE


def test_a_serve_thread_failure_closes_the_bound_receiver(monkeypatch):
    """If the serve_forever thread cannot start, _start_loopback_receiver must
    release the socket it already bound instead of leaking it."""
    import http.server

    closed = []

    class _RecordingHTTPServer(http.server.HTTPServer):
        def server_close(self):
            closed.append(self)
            super().server_close()

    monkeypatch.setattr(mcp_oauth.http.server, "HTTPServer", _RecordingHTTPServer)

    real_thread = threading.Thread

    class _Unstartable(real_thread):
        def start(self):
            raise RuntimeError("cannot spawn serve thread")

    monkeypatch.setattr(mcp_oauth.threading, "Thread", _Unstartable)

    flow = _flow("srv", "unstarted-receiver")
    with pytest.raises(RuntimeError, match="cannot spawn serve thread"):
        mcp_oauth.choose_callback_receiver(flow, _cfg(), None)

    assert len(closed) == 1
    assert (_home(), "srv") not in mcp_oauth._ACTIVE
