"""Session finalize must clear the owner's approval/kernel state (TUI-owned only).

The gateway's /stop and /new paths call ``approval.clear_session`` so a finished
conversation cannot leak a session-persistent execute_code kernel (see #88637).
A TUI/Desktop session that ends via ``_finalize_session`` took no such path —
kernels survived until the NEXT execute_code in the process ran the lazy idle
sweep, orphaning one live interpreter per finished conversation.

Contract: when the TUI owns the session lifecycle, finalize clears approval
state (killing that owner's kernels) with the session's own key; when another
backend (gateway) owns the lease, finalize must NOT clear it — a viewer tab
must never kill the gateway's kernels.
"""

import threading

from tui_gateway import server


def _session(session_key: str) -> dict:
    return {
        "active_session_lease": None,
        "agent": None,
        "history": [],
        "history_lock": threading.Lock(),
        "profile_home": None,
        "session_key": session_key,
        "slash_worker": None,
        "source": "desktop",
    }


def _install_stubs(monkeypatch, *, db_source: str | None = "desktop"):
    """Stub out finalize's external work; keep the lifecycle-guard + db-source logic real."""
    cleared: list[str] = []

    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "tools.approval.clear_session",
        lambda key: cleared.append(key),
    )

    class _FakeDB:
        def __init__(self, source: str | None):
            self.source = source

        def get_session(self, target: str) -> dict | None:
            if self.source is None:
                return None
            return {"id": target, "source": self.source}

        def end_session(self, *_a, **_k):
            pass

    import contextlib

    @contextlib.contextmanager
    def _profile_db(_session: dict):
        yield _FakeDB(db_source)

    monkeypatch.setattr(server, "_session_db", _profile_db)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    return cleared


def test_tui_owned_finalize_clears_approval_state_for_its_owner(monkeypatch):
    cleared = _install_stubs(monkeypatch, db_source="desktop")

    server._finalize_session(
        _session("20260907_120000_abcdef"), end_reason="ws_orphan_reap"
    )

    assert cleared == ["20260907_120000_abcdef"]


def test_user_close_also_clears_approval_state(monkeypatch):
    cleared = _install_stubs(monkeypatch, db_source="desktop")

    server._finalize_session(
        _session("20260907_120000_abcdef"), end_reason="tui_close"
    )

    assert cleared == ["20260907_120000_abcdef"]


def test_viewer_finalize_does_not_clear_gateway_owned_session(monkeypatch):
    """A gateway-originated session (TUI is a viewer) must keep its kernels."""
    cleared = _install_stubs(monkeypatch, db_source="telegram")

    server._finalize_session(
        _session("20260907_120000_abcdef"), end_reason="tui_shutdown"
    )

    assert cleared == []


def test_finalize_without_db_row_still_clears_when_tui_owns(monkeypatch):
    cleared = _install_stubs(monkeypatch, db_source=None)

    server._finalize_session(
        _session("20260907_120000_abcdef"), end_reason="idle_timeout"
    )

    assert cleared == ["20260907_120000_abcdef"]
