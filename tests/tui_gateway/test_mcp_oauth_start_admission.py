"""Starting the same server twice must reserve its slot before binding a listener."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tui_gateway import mcp_oauth_sessions as sessions


def test_concurrent_starts_reserve_server_before_listener_setup(monkeypatch, tmp_path):
    entered = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(sessions, "_sessions", {})
    monkeypatch.setattr(sessions, "run_worker", lambda *args, **kwargs: None)

    def receiver(flow, cfg, redirect):
        if not entered.is_set():
            entered.set()
            assert release.wait(10)
        flow.authorization_url = "https://auth.example.invalid/authorize?state=test"
        return None

    monkeypatch.setattr(sessions, "choose_callback_receiver", receiver)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(sessions.start_flow, str(tmp_path), "example", {})
        try:
            assert entered.wait(10)
            with pytest.raises(RuntimeError, match="already in progress"):
                sessions.start_flow(str(tmp_path), "example", {})
        finally:
            release.set()
            first.result(timeout=10)
    assert len(sessions._sessions) == 1


def test_listener_failure_releases_reservation(monkeypatch, tmp_path):
    monkeypatch.setattr(sessions, "_sessions", {})
    monkeypatch.setattr(sessions, "run_worker", lambda *args, **kwargs: None)

    def broken_receiver(*args):
        raise OSError("listener unavailable")

    monkeypatch.setattr(sessions, "choose_callback_receiver", broken_receiver)
    with pytest.raises(OSError, match="listener unavailable"):
        sessions.start_flow(str(tmp_path), "example", {})
    assert sessions._sessions == {}

    def working_receiver(flow, cfg, redirect):
        flow.authorization_url = "https://auth.example.invalid/authorize?state=retry"
        return None

    monkeypatch.setattr(sessions, "choose_callback_receiver", working_receiver)
    result = sessions.start_flow(str(tmp_path), "example", {})
    assert result["auth_url"].endswith("state=retry")
