"""prompt.submit depth gate (#133716): a cut that archives more than one user turn needs
``confirm_deep_truncate`` — a valid anchor can still name the wrong (stale) row."""

import threading
import types

from tui_gateway import server


def test_prompt_submit_refuses_deep_truncation_without_confirm(monkeypatch):
    replaced = []

    class _FakeDB:
        def replace_messages(self, key, messages, **_kwargs):
            replaced.append(list(messages))

    history = []
    for turn in range(3):
        history += [
            {"_row_id": 100 + 2 * turn, "role": "user", "content": f"u{turn}"},
            {"_row_id": 101 + 2 * turn, "role": "assistant", "content": f"a{turn}"},
        ]
    sess = {
        "agent": types.SimpleNamespace(), "session_key": "session-key", "history": list(history),
        "history_lock": threading.Lock(), "history_version": 0, "running": False,
        "attached_images": [], "image_counter": 0, "cols": 80, "slash_worker": None,
        "show_reasoning": False, "tool_progress_mode": "all",
    }
    server._sessions["deep-trunc-sid"] = sess
    monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    # The confirmed submit starts a turn; never run it on a real thread (it would leak).
    monkeypatch.setattr(server.threading, "Thread", lambda *a, **k: types.SimpleNamespace(start=lambda: None))

    def _submit(**extra):
        return server.handle_request({"id": "1", "method": "prompt.submit", "params": {
            "session_id": "deep-trunc-sid", "text": "u1", "truncate_before_row_id": 102,
            "confirm_truncate": True, **extra}})

    try:
        refused = _submit()
        assert refused["error"]["code"] == 4033
        assert refused["error"]["data"] == {"archived_messages": 4, "archived_user_turns": 2}
        assert sess["history"] == history and replaced == []

        assert _submit(confirm_deep_truncate=True).get("error") is None
        assert replaced == [history[:2]]
    finally:
        server._sessions.pop("deep-trunc-sid", None)
