"""prompt.submit history-cut opt-ins: ``confirm_empty_truncate`` (wipe) and
``confirm_deep_truncate`` (#133716: a cut that archives more than one user turn — a valid
anchor can still name the wrong, stale row)."""

import threading
import types

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0,
        "cols": 80, "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
        **extra,
    }


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
    sess = _session(history=list(history))
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


def test_prompt_submit_empty_truncation_allowed_with_confirm(monkeypatch):
    """Intentional restore/regenerate of the first user turn may wipe history."""

    seen = {}
    replaced = []

    class _Agent:
        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None, **_kwargs
        ):
            seen["prompt"] = prompt
            seen["history"] = conversation_history
            return {
                "final_response": "regenerated",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "regenerated"},
                ],
            }

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_thread_options):
            self._target = target

        def start(self):
            self._target()

    class _FakeDB:
        def replace_messages(
            self,
            key,
            messages,
            active_only=False,
            archive_dropped=False,
            reject_active_turn_lease=False,
        ):
            replaced.append((key, list(messages)))

    history = [
        {"_row_id": 101, "role": "user", "content": "first"},
        {"_row_id": 102, "role": "assistant", "content": "ok"},
        {"_row_id": 103, "role": "user", "content": "second"},
        {"_row_id": 104, "role": "assistant", "content": "done"},
    ]
    server._sessions["confirm-empty-sid"] = _session(
        agent=_Agent(), history=list(history)
    )

    try:
        monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(server, "_get_usage", lambda _a: {})
        monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
        monkeypatch.setattr(server, "_emit", lambda *a: None)
        monkeypatch.setattr(server, "_get_db", lambda: _FakeDB())

        resp = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {
                    "session_id": "confirm-empty-sid",
                    "text": "first",
                    "truncate_before_row_id": 101,
                    "truncate_before_user_ordinal": 0,
                    "confirm_truncate": True,
                    "confirm_empty_truncate": True,
                    "confirm_deep_truncate": True,  # drops both user turns (#133716)
                },
            }
        )
        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert seen["prompt"] == "first"
        assert seen["history"] == []
        assert replaced == [("session-key", [])]
        assert server._sessions["confirm-empty-sid"]["history"] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "regenerated"},
        ]
    finally:
        server._sessions.pop("confirm-empty-sid", None)
