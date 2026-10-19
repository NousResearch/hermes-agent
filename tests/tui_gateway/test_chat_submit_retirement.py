"""Accepted Desktop prompts that stop before worker admission release their journal claim."""

import threading

import pytest

from gateway.chat_execution_recovery import ChatExecutionStore
from tui_gateway import server


@pytest.mark.parametrize("early_exit", ["stop", "build_failure"])
def test_registered_prompt_early_exit_allows_physical_retirement(monkeypatch, tmp_path, early_exit):
    store = ChatExecutionStore(tmp_path)
    assert store.start_local_worker("session-key", "prior-turn") == 1
    assert store.worker_finished("session-key", 1)
    threads = []

    class DeferredThread:
        def __init__(self, target=None, daemon=None, **_options):
            self.target = target
            threads.append(self)

        def start(self):
            pass

        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

    ready = threading.Event()
    session = {"agent": None, "agent_ready": ready, "session_key": "session-key",
               "profile_home": str(tmp_path), "history": [], "history_lock": threading.Lock(),
               "history_version": 0, "running": False, "attached_images": [],
               "image_counter": 0, "cols": 80, "slash_worker": None,
               "show_reasoning": False, "tool_progress_mode": "all"}
    if early_exit == "build_failure":
        ready.set()
        session["agent_error"] = "No provider configured"

    def build(_sid, record):
        if early_exit == "build_failure":
            record["agent_error"] = "No provider configured"
            record["agent_ready"].set()

    server._sessions["sid"] = session
    try:
        monkeypatch.setattr(server.threading, "Thread", DeferredThread)
        monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
        monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
        monkeypatch.setattr(server, "_start_agent_build", build)
        monkeypatch.setattr(server, "_wait_agent", lambda *_args: None)
        result = server.handle_request({"id": "submit", "method": "prompt.submit",
                                        "params": {"session_id": "sid", "text": "Continue"}})
        assert result["result"]["status"] == "streaming"
        if early_exit == "stop":
            stopped = server.handle_request({"id": "stop", "method": "session.interrupt",
                                             "params": {"session_id": "sid"}})
            assert stopped["result"]
        assert session.get("_run_thread") in threads
        session["_run_thread"].target()
        assert store.retire_for_handoff("session-key", 1, "handoff")["status"] == "retired"
    finally:
        server._sessions.pop("sid", None)
