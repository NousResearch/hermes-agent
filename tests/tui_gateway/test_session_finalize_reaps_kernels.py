"""Session finalize disposes the owner's execute_code kernels (TUI-owned only).

Regression for #105213. The gateway's /stop and /new paths call ``approval.clear_session``,
which tears down the session-persistent kernels owned by that session key (#88637). A
TUI/Desktop session ending via ``_finalize_session`` never did, so each finished conversation
left a live interpreter behind until the idle reaper's ``kernel_idle_timeout`` expired.

Contract: when the TUI owns the lifecycle, finalize disposes that session's kernels and approval
state; when the messaging gateway owns the session (the TUI is a viewer), both are left alone.
"""

import contextlib
import threading

import pytest

from tools import approval, code_kernel
from tui_gateway import server

SESSION_KEY = "20260907_120000_abcdef"


def _session() -> dict:
    return {"active_session_lease": None, "agent": None, "history": [], "history_lock": threading.Lock(),
            "profile_home": None, "session_key": SESSION_KEY, "slash_worker": None, "source": "desktop"}


@pytest.fixture
def live_kernel(monkeypatch):
    """A registered (unspawned) kernel owned by SESSION_KEY, plus per-session YOLO state."""
    kernel = code_kernel.SessionKernel((SESSION_KEY, "project", "python", "/tmp", ()))
    monkeypatch.setitem(code_kernel._REGISTRY.kernels, kernel.key, kernel)
    approval.enable_session_yolo(SESSION_KEY)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    yield kernel
    approval.clear_session(SESSION_KEY)


def _stored_source(monkeypatch, source):
    class _FakeDB:
        def get_session(self, target):
            return None if source is None else {"id": target, "source": source}

        def end_session(self, *_a, **_k):
            pass

    @contextlib.contextmanager
    def _profile_db(_session):
        yield _FakeDB()

    monkeypatch.setattr(server, "_session_db", _profile_db)


@pytest.mark.parametrize("source, end_reason", [
    ("desktop", "tui_close"), ("desktop", "ws_orphan_reap"), (None, "idle_timeout")])
def test_tui_owned_finalize_disposes_the_sessions_kernels(monkeypatch, live_kernel, source, end_reason):
    _stored_source(monkeypatch, source)

    server._finalize_session(_session(), end_reason=end_reason)

    assert live_kernel.key not in code_kernel._REGISTRY.kernels
    assert live_kernel.stop_event.is_set()
    assert not approval.is_session_yolo_enabled(SESSION_KEY)


def test_viewer_finalize_keeps_a_gateway_owned_sessions_kernels(monkeypatch, live_kernel):
    _stored_source(monkeypatch, "telegram")

    server._finalize_session(_session(), end_reason="tui_shutdown")

    assert code_kernel._REGISTRY.kernels.get(live_kernel.key) is live_kernel
    assert not live_kernel.stop_event.is_set()
    assert approval.is_session_yolo_enabled(SESSION_KEY)
