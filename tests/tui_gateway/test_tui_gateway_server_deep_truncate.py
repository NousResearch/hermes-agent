"""prompt.submit history-cut opt-ins: ``confirm_empty_truncate`` (wipe) and
``confirm_deep_truncate`` (#133716: a cut that archives more than one user turn — a valid
anchor can still name the wrong, stale row)."""

import contextlib
import threading
import types

from persistence.conversation.messages import ConversationMessage, ReplaceResult
from tui_gateway import server

_NO_LEDGER_WORK = types.SimpleNamespace(pending_ledger_status=lambda _key: None)  # no ledger holds work


def _use_messages(monkeypatch, messages):
    """Route the gateway's store through a handle whose transcript verbs are ``messages``."""
    monkeypatch.setattr(server, "_get_db", lambda: object())
    persistence = types.SimpleNamespace(messages=messages, runtime=_NO_LEDGER_WORK)
    monkeypatch.setattr(server, "_persistence_for", lambda _db: contextlib.nullcontext(persistence))


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key", "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0,
        "cols": 80, "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
        **extra,
    }


def test_prompt_submit_refuses_deep_truncation_without_confirm(monkeypatch):
    # register() rebinds the handlers onto the server namespace, so the flag is read there.
    monkeypatch.setattr(server, "DEEP_TRUNCATE_ENFORCED", True)  # the next release's behaviour
    replaced = []

    class _FakeMessages:
        def replace_transcript(self, key, messages, mode, **_kwargs):
            replaced.append(list(messages))
            return ReplaceResult(rows=())

    history = []
    for turn in range(3):
        history += [
            {"_row_id": 100 + 2 * turn, "role": "user", "content": f"u{turn}"},
            {"_row_id": 101 + 2 * turn, "role": "assistant", "content": f"a{turn}"},
        ]
    sess = _session(history=list(history))
    server._sessions["deep-trunc-sid"] = sess
    _use_messages(monkeypatch, _FakeMessages())
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
        assert replaced == [[server._new_message(m) for m in history[:2]]]

        # Durable-carrier cut: the target (150) was absorbed into a repaired live carrier;
        # the refusal counts live rows from the carrier on, never live minus physical.
        physical = [{"_row_id": 100, "role": "user", "content": "u0"},
                    {"_row_id": 150, "role": "user", "content": "u0b"}, *history[1:]]
        _FakeMessages.read_conversation = lambda self, key, request: tuple(  # type: ignore[attr-defined]
            ConversationMessage.from_openai_message(m) for m in physical)
        sess["running"] = False  # the confirmed submit above left a (never-started) turn
        sess["history"] = [{"_row_id": 100, "_absorbed_row_ids": [150], "role": "user",
                            "content": "u0\n\nu0b"}, *history[1:]]
        carrier = _submit(truncate_before_row_id=150)
        assert carrier["error"]["data"] == {"archived_messages": 6, "archived_user_turns": 3}
    finally:
        server._sessions.pop("deep-trunc-sid", None)


def test_deep_truncate_counts_every_durable_row_a_merged_carrier_holds(monkeypatch):
    """A user;user run repaired into one live carrier is several durable user turns: cutting at
    its first row archives all of them, so it needs confirm_deep_truncate like any deep cut."""
    monkeypatch.setattr(server, "DEEP_TRUNCATE_ENFORCED", True)
    replaced = []

    physical = [{"_row_id": 101, "role": "user", "content": "a"}, {"_row_id": 102, "role": "user", "content": "b"},
                {"_row_id": 103, "role": "user", "content": "c"}, {"_row_id": 104, "role": "assistant", "content": "reply"}]

    class _FakeMessages:
        def read_conversation(self, key, request):
            return tuple(ConversationMessage.from_openai_message(m) for m in physical)

        def replace_transcript(self, key, messages, mode, **_kwargs):
            replaced.append(list(messages))
            return ReplaceResult(rows=())

    history = [{"_row_id": 101, "_absorbed_row_ids": [102, 103], "role": "user", "content": "a\n\nb\n\nc"},
               {"_row_id": 104, "role": "assistant", "content": "reply"}]
    sess = _session(history=list(history))
    server._sessions["deep-carrier-sid"] = sess
    _use_messages(monkeypatch, _FakeMessages())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server.threading, "Thread", lambda *a, **k: types.SimpleNamespace(start=lambda: None))
    try:
        refused = server.handle_request({"id": "1", "method": "prompt.submit", "params": {
            "session_id": "deep-carrier-sid", "text": "a", "truncate_before_row_id": 101,
            "confirm_truncate": True, "confirm_empty_truncate": True}})
        assert refused["error"]["code"] == 4033
        assert refused["error"]["data"]["archived_user_turns"] == 3
        assert sess["history"] == history and replaced == []

    finally:
        server._sessions.pop("deep-carrier-sid", None)


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

    class _FakeMessages:
        def replace_transcript(self, key, messages, mode, reject_active_turn_lease=False):
            replaced.append((key, list(messages)))
            return ReplaceResult(rows=())

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
        _use_messages(monkeypatch, _FakeMessages())

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


def test_archived_user_turns_counts_a_carrier_without_its_own_row_id():
    """An unstamped carrier's own turn has no id to look up; it still counts (fail closed)."""
    from tui_gateway import methods_prompt

    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "r"},
               {"_absorbed_row_ids": [102], "role": "user", "content": "b\n\nc"}]
    sess = _session(history=history)
    physical = {"rows": [{"_row_id": 102, "role": "user", "content": "c"}]}
    original = methods_prompt._load_durable_truncation_history
    methods_prompt._load_durable_truncation_history = lambda *a, **k: physical["rows"]
    try:
        assert methods_prompt._archived_user_turns(sess, "sid", history, 2, set()) == 2
    finally:
        methods_prompt._load_durable_truncation_history = original


def test_unconfirmed_deep_truncation_is_warn_only_this_release(monkeypatch, caplog):
    """Rollout (G15-B): an older client without confirm_deep_truncate keeps working for one release;
    the cut runs and the would-be refusal is logged."""
    import logging

    replaced = []

    class _FakeMessages:
        def replace_transcript(self, key, messages, mode, **_kwargs):
            replaced.append(list(messages))
            return ReplaceResult(rows=())

    history = []
    for turn in range(3):
        history += [{"_row_id": 100 + 2 * turn, "role": "user", "content": f"u{turn}"},
                    {"_row_id": 101 + 2 * turn, "role": "assistant", "content": f"a{turn}"}]
    server._sessions["warn-only-sid"] = _session(history=list(history))
    _use_messages(monkeypatch, _FakeMessages())
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server.threading, "Thread", lambda *a, **k: types.SimpleNamespace(start=lambda: None))
    try:
        with caplog.at_level(logging.WARNING):
            result = server.handle_request({"id": "1", "method": "prompt.submit", "params": {
                "session_id": "warn-only-sid", "text": "u1", "truncate_before_row_id": 102,
                "confirm_truncate": True}})
        assert result.get("error") is None
        assert replaced == [[server._new_message(m) for m in history[:2]]]
        assert "ALLOWED (warn-only release) unconfirmed deep truncation" in caplog.text
    finally:
        server._sessions.pop("warn-only-sid", None)

