"""``pre_prompt_submit``: the plugin gate for a prompt typed in Desktop / the dashboard chat / the Ink TUI.

A plugin can hold the prompt behind a choice card (``ask``), drop it, or have the host switch model / compact
before the turn — the tui_gateway twin of the messaging gateway's ``pre_gateway_dispatch``.
"""

import sys
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server


@pytest.fixture
def plugin(monkeypatch):
    """Route ``pre_prompt_submit`` to whatever callback the test installs (the real dispatcher is the
    plugin manager's; this seam is the one prompt_hooks calls)."""
    from hermes_cli import lifecycle

    slot = {}

    def invoke(name, **kwargs):
        if name != "pre_prompt_submit" or "cb" not in slot:
            return []
        slot.setdefault("calls", []).append(kwargs)
        result = slot["cb"](**kwargs)
        return [] if result is None else [result]

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_prompt_submit" and "cb" in slot)
    monkeypatch.setattr(lifecycle, "invoke_hook", invoke)
    return slot


@pytest.fixture
def emitted(monkeypatch):
    events = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, payload)))
    monkeypatch.setattr(server, "_session_info", lambda agent, session: {"running": bool(session.get("running"))})
    return events


@pytest.fixture
def turn(monkeypatch):
    """Stub the build wait and the turn itself: the gate sits between them."""
    record = {"ran": [], "drained": []}
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda session, rid, sid: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda rid, sid, session, text, **kw: record["ran"].append(text))
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda rid, sid, session: record["drained"].append(sid))
    return record


def _session(**extra):
    agent = SimpleNamespace(model="local-35b", provider="custom", base_url="http://127.0.0.1:11434/v1",
                            session_id="agent-1", context_compressor=SimpleNamespace(last_prompt_tokens=61_000))
    session = {"agent": agent, "history_lock": threading.Lock(), "history": [
        {"role": "user", "content": "hi", "timestamp": 1_700_000_000.0},
        {"role": "assistant", "content": "hello", "timestamp": 1_700_000_005.0}],
        "running": True, "session_key": "conv-1", "source": "desktop"}
    session.update(extra)
    return session


def _gate(session, text="how many kms since june", hook=True):
    server._run_after_agent_ready("rid", "sid-1", session, text, None, None, None, None, pre_submit_hook=hook)


def test_plain_values_reach_the_plugin_and_allow_runs_the_turn(plugin, emitted, turn):
    plugin["cb"] = lambda **kw: None
    _gate(_session())
    assert turn["ran"] == ["how many kms since june"]
    kw = plugin["calls"][0]
    assert (kw["session_key"], kw["session_id"], kw["platform"]) == ("conv-1", "agent-1", "desktop")
    assert (kw["model"], kw["provider"], kw["context_tokens"]) == ("local-35b", "custom", 61_000)
    assert kw["last_message_at"] == 1_700_000_005.0
    assert callable(kw["ask"])


def test_skip_ends_the_turn_without_running_the_agent(plugin, emitted, turn):
    plugin["cb"] = lambda **kw: {"action": "skip", "reply": "Start a new chat and send it there."}
    session = _session()
    _gate(session)
    assert turn["ran"] == []
    assert session["running"] is False
    assert ("notification.show", "Start a new chat and send it there.") in [
        (e, (p or {}).get("text")) for e, p in emitted]
    assert ("session.info", {"running": False}) in emitted
    assert turn["drained"] == ["sid-1"], "a prompt queued while the gate waited must still run"


def test_allow_with_a_reply_runs_the_turn_and_shows_the_note(plugin, emitted, turn):
    plugin["cb"] = lambda **kw: {"action": "allow", "reply": "This will take ~1 min on local."}
    _gate(_session())
    assert turn["ran"] == ["how many kms since june"]
    assert ("notification.show", "This will take ~1 min on local.") in [
        (e, (p or {}).get("text")) for e, p in emitted]


def test_only_a_typed_prompt_meets_the_gate(plugin, emitted, turn):
    plugin["cb"] = lambda **kw: {"action": "skip"}
    _gate(_session(), hook=False)
    assert turn["ran"] == ["how many kms since june"] and "calls" not in plugin


def test_model_directive_switches_through_the_model_path_before_the_turn(plugin, emitted, turn, monkeypatch):
    switches = []

    def switch(sid, session, raw):
        assert turn["ran"] == [], "the switch must land before the turn starts"
        switches.append(raw)
        return {}

    monkeypatch.setattr(server, "_apply_model_switch", switch)
    plugin["cb"] = lambda **kw: {"action": "allow", "model": {
        "model": "gpt-luna", "provider": "openai-codex", "scope": "once"}}
    _gate(_session())
    assert switches == ["gpt-luna --provider openai-codex --once"]
    assert turn["ran"] == ["how many kms since june"]


def test_a_failed_switch_does_not_run_the_prompt_on_the_old_model(plugin, emitted, turn, monkeypatch):
    def switch(sid, session, raw):
        raise ValueError("no credentials for openai-codex")

    monkeypatch.setattr(server, "_apply_model_switch", switch)
    plugin["cb"] = lambda **kw: {"action": "allow", "model": {"model": "gpt-luna", "scope": "session"}}
    session = _session()
    _gate(session)
    assert turn["ran"] == [] and session["running"] is False
    texts = [(p or {}).get("text", "") for e, p in emitted if e == "notification.show"]
    assert any("no credentials for openai-codex" in t for t in texts)


def test_a_raising_plugin_fails_open(plugin, emitted, turn):
    def boom(**kw):
        raise RuntimeError("plugin bug")

    plugin["cb"] = boom
    _gate(_session())
    assert turn["ran"] == ["how many kms since june"]


def test_an_interrupt_while_the_card_is_open_ends_the_turn(plugin, emitted, turn):
    session = _session()

    def interrupted(**kw):
        session["_turn_cancel_requested"] = True  # session.interrupt while the plugin waited

    plugin["cb"] = interrupted
    _gate(session)
    assert turn["ran"] == [] and session["running"] is False


def test_ask_shows_one_clarify_card_and_returns_the_pick(plugin, emitted, turn, monkeypatch):
    from hermes_cli import banner
    from tui_gateway import server_requests

    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    frames = []

    def answer(frame):
        frames.append(frame)
        assert server_requests.resolve_response(
            {"jsonrpc": "2.0", "id": frame["id"], "result": {"answers": {"q1": "Run anyway"}}})
        return True

    monkeypatch.setattr(server, "write_json", answer)
    picks = []
    plugin["cb"] = lambda ask, **kw: picks.append(ask("Slow on local. What now?", ["Compact first", "Run anyway"]))
    monkeypatch.setitem(server._sessions, "sid-1", {"session_key": "conv-1", "source": "desktop"})
    try:
        _gate(_session())
    finally:
        server_requests.reset_for_tests()
    assert picks == ["Run anyway"]
    (frame,) = frames
    assert frame["method"] == "clarify"
    assert frame["params"]["questions"] == [{"qid": "q1", "question": "Slow on local. What now?",
                                             "choices": ["Compact first", "Run anyway"], "multi_select": False}]
    assert turn["ran"] == ["how many kms since june"]


@pytest.mark.parametrize("params,expected", [
    ({}, True),
    ({"display_kind": "hidden"}, False),
    ({"truncate_before_user_ordinal": 0, "confirm_truncate": True}, False),
])
def test_prompt_submit_arms_the_gate_only_for_typed_prompts(monkeypatch, tmp_path, params, expected):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    created = server.handle_request({"id": "c", "method": "session.create", "params": {"source": "desktop"}})
    sid = created["result"]["session_id"]

    class InlineThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    armed = []
    monkeypatch.setattr(server.threading, "Thread", InlineThread)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *args: False)
    monkeypatch.setattr(server, "_truncate_history_for_submit", lambda *args: (None, {}))
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *args: armed.append(args[8]))
    try:
        reply = server.handle_request({"id": "p", "method": "prompt.submit", "params": {
            "session_id": sid, "text": "typed", **params}})
        assert "result" in reply, reply
        assert armed == [expected]
    finally:
        server._sessions.pop(sid, None)
        db.close()


# ── compress before the turn: the submit-time user row must end up in the continuation exactly once ──

def _real_agent(tmp_path, monkeypatch, in_place):
    from unittest.mock import MagicMock

    from agent.context_compressor import SUMMARY_PREFIX
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    parent = "gate-compress-parent"
    db.create_session(parent, source="desktop")
    for i in range(12):
        db.append_message(parent, "user", f"question {i} " + "x" * 80)
        db.append_message(parent, "assistant", f"answer {i} " + "y" * 80)
    before = db.get_messages_as_conversation(parent)
    agent = AIAgent(api_key="fixture-only", base_url="http://127.0.0.1:9/v1", model="test/model",
                    platform="desktop", quiet_mode=True, session_db=db, session_id=parent,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": SUMMARY_PREFIX + " earlier turns"}, before[-1].copy()]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    agent.compression_in_place = in_place
    agent._compression_feasibility_checked = True
    monkeypatch.setattr(agent, "_build_system_prompt", lambda *args, **kwargs: "fixture prompt")
    return db, agent, parent, before


@pytest.mark.parametrize("in_place", [True, False])
def test_compress_directive_keeps_the_typed_message_exactly_once(
        plugin, emitted, turn, monkeypatch, tmp_path, in_place):
    from agent.turn_context import _stage_turn_user_message

    db, agent, parent, before = _real_agent(tmp_path, monkeypatch, in_place)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_restart_slash_worker", lambda sid, session: None)
    session = {"agent": agent, "history_lock": threading.Lock(), "history": list(before), "history_version": 1,
               "running": True, "session_key": parent, "source": "desktop", "cwd": str(tmp_path)}
    text = "and now the summary please"
    server._start_inflight_turn(session, text)
    assert server._persist_session_row_for_submit("rid", session, text, None) is None
    assert db.get_messages_as_conversation(parent)[-1]["content"] == text  # durable at send, in the parent

    plugin["cb"] = lambda **kw: {"action": "allow", "compress": True}
    _gate(session, text)

    assert turn["ran"] == [text]
    child = session["session_key"]
    assert agent.session_id == child and (child == parent) is in_place
    assert len(session["history"]) < len(before), "the compaction committed"
    assert text not in [m.get("content") for m in session["history"]], "compaction folded in the new prompt"
    # The turn that follows: adopt whatever submit row is still staged, then flush like a real turn.
    server._adopt_submit_user_row(session, agent, text, text)
    user_msg, _ = _stage_turn_user_message(agent, text, text, None, None, None, None)
    history = list(session["history"])
    agent._persist_user_message_idx = len(history)
    agent._flush_messages_to_session_db(history + [user_msg, {"role": "assistant", "content": "done"}], history)

    live = [(m["role"], m["content"]) for m in db.get_messages_as_conversation(child)]
    assert live[-2:] == [("user", text), ("assistant", "done")]
    assert [c for _, c in live].count(text) == 1, "the model-fed transcript carries the typed message once"
    lineage = [m["content"] for m in db.get_messages_as_conversation(child, include_ancestors=True)]
    assert lineage.count(text) == 1, "the typed message shows once across parent + continuation"
    db.close()


def test_a_failed_compaction_keeps_the_typed_message_and_runs_nothing(
        plugin, emitted, turn, monkeypatch, tmp_path):
    db, agent, parent, before = _real_agent(tmp_path, monkeypatch, True)
    monkeypatch.setattr(server, "_get_db", lambda: db)

    def fail(*args, **kwargs):
        raise RuntimeError("summary model unreachable")

    monkeypatch.setattr(server, "_compress_session_history", fail)
    session = {"agent": agent, "history_lock": threading.Lock(), "history": list(before), "history_version": 1,
               "running": True, "session_key": parent, "source": "desktop", "cwd": str(tmp_path)}
    text = "and now the summary please"
    server._start_inflight_turn(session, text)
    assert server._persist_session_row_for_submit("rid", session, text, None) is None

    plugin["cb"] = lambda **kw: {"action": "allow", "compress": True}
    _gate(session, text)

    assert turn["ran"] == [] and session["running"] is False
    assert [m["content"] for m in db.get_messages_as_conversation(parent)].count(text) == 1
    texts = [(p or {}).get("text", "") for e, p in emitted if e == "notification.show"]
    assert any("summary model unreachable" in t for t in texts)
    db.close()
