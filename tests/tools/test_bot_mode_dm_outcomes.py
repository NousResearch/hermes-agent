"""Bot-to-bot request outcomes the sender can act on: the typed request envelope, a reply the
teammate forgot to give, one plain "never answered" sentence per dead request, and a turn its
own gateway's restart interrupted reading as a failure, not a cancel."""

import json
from types import SimpleNamespace

from hermes_agent.hermes_state import SessionDB
from hermes_agent.hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input, settle_session_input
from hermes_agent.gateway.session_bot import _result
from hermes_agent.gateway.session_results import retain_result
from tools import bot_mode_dm, bot_mode_dm_outcome

from tests.tools.test_bot_mode_dm import _FakeAgent, _canonical_target, _capture_spawn, _fresh_probe_cache, _managed_home

ALPHA = {"id": "bot:alpha", "name": "alpha", "is_bot": True}


def test_constraints_and_expecting_reach_the_teammate_as_labelled_lines(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    authority = _canonical_target(monkeypatch, home / "profiles" / "researcher")
    _capture_spawn(monkeypatch)

    bot_mode_dm.message_agent_tool("researcher", "Summarize the Q3 report.", agent=_FakeAgent(home),
                                   constraints="under 20 words, no lists", expecting="a one-line summary")

    assert authority.calls[0][1]["message"].endswith(
        "): Summarize the Q3 report.\n\nConstraints: under 20 words, no lists\nReply needed: a one-line summary")


def _settled(tmp_path, messages, final, *, message="Message from 🤖 alpha (@alpha): what is the codeword?",
             author: dict | None = ALPHA, interrupt=None):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("bot", source="gui")
    epoch = begin_runtime_epoch(db, instance_id="first")
    admission = admit_session_input(db, epoch=epoch, principal_id="owner", session_id="bot",
                                    request_id="bot:" + "a" * 32, payload={"text": message})
    row = claim_session_input(db, epoch=epoch, session_id="bot")
    value = {"final_response": final, "messages": messages, "_messages_are_turn_suffix": True}
    if interrupt:
        value.update(interrupted=True, interrupt_message=interrupt)
        settle_session_input(db, epoch=epoch, admission_id=row["admission_id"], generation=row["generation"],
                             outcome="interrupted", result={"result": value, "usage": {}})
    else:
        retain_result(db, epoch=epoch, row=row, result={"result": value, "usage": {}})
    record = dict(status="claimed", admission_id=admission["admission_id"], delivery_id="a" * 32,
                  profile_home=str(tmp_path), session_id="bot", message=message, author=author)
    return db, record


def _reply_call(target, text):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "message_agent", "arguments": json.dumps({"target": target, "message": text})}}]}


def test_a_teammate_that_ends_silent_after_writing_its_answer_still_answers_once(tmp_path):
    messages = [_reply_call("alpha", "PELICAN-7731"), {"role": "tool", "content": "{\"error\": \"...\"}"},
                {"role": "assistant", "content": "NO_REPLY"}]
    db, record = _settled(tmp_path, messages, "NO_REPLY")
    try:
        receipt = _result(SimpleNamespace(db=db), record)
        assert receipt["status"] == "settled"
        assert receipt["reply"] == "[no final reply; their last message this turn:] PELICAN-7731"
    finally:
        db.close()


def test_notify_only_and_non_bot_requests_keep_their_silence(tmp_path):
    messages = [{"role": "assistant", "content": "noted, thanks"}, {"role": "assistant", "content": "NO_REPLY"}]
    for n, kwargs in enumerate(({"message": "Message from 🤖 alpha (@alpha): [no reply needed] FYI build is green"},
                                {"author": None})):
        (tmp_path / str(n)).mkdir()
        db, record = _settled(tmp_path / str(n), messages, "NO_REPLY", **kwargs)
        try:
            assert _result(SimpleNamespace(db=db), record)["reply"] == ""
        finally:
            db.close()


def test_a_turn_its_gateway_restart_interrupted_is_a_failure_that_says_so(tmp_path):
    db, record = _settled(tmp_path, [], "", interrupt="Gateway shutting down")
    try:
        receipt = _result(SimpleNamespace(db=db), record)
        assert receipt["status"] == "failed" and receipt["reason"] == "unknown"
        assert "gateway stopped or restarted mid-turn" in receipt["error"]
    finally:
        db.close()


def test_every_dead_request_reads_as_never_answered_with_its_reason():
    line = bot_mode_dm_outcome.outcome_line
    assert line("@beta", {"status": "settled", "reply": "x"}) == "@beta answered."
    assert line("@beta", {"status": "claimed", "target_state": "unreachable"}).startswith(
        "@beta never answered within the wait: its gateway is not reachable.")
    assert "do not resend" in line("@gamma", {"status": "ambiguous", "error": "outcome unknown"})
    assert line("@gamma", {"status": "failed", "error": "x — the message was NOT sent"}).endswith("you may resend.")
    summary = bot_mode_dm_outcome.combined_summary(
        [{"to": "@gamma", "status": "pending", "detail": "No outcome"}, {"to": "@beta", "status": "settled", "reply": "y"}])
    assert summary.startswith("@beta answered. @gamma never answered within the wait")
