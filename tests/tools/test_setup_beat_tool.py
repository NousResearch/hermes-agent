"""``setup_beat`` through the agent's inline executor, with a real session store and a fake card
answer standing in for the desktop."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import onboarding_record
from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from agent.onboarding_flow import End, Facts
from hermes_state import SessionDB
from tools.registry import registry

SID = "onb-session"
FACTS = Facts(os="macos", arch="arm64", machine_kind="Mac", plugins=("blender",))


@pytest.fixture
def db(tmp_path, monkeypatch):
    # The run's start and end write the install flag in the root config; that is not under test here.
    monkeypatch.setattr("hermes_cli.onboarding_run.set_run", lambda *a, **k: None)
    store = SessionDB(db_path=tmp_path / "state.db")
    store.create_session(SID, "desktop")
    yield store
    store.close()


@pytest.fixture(autouse=True)
def machine(monkeypatch):
    """What this machine has connected and installed right now, read when a card closes without its answer."""
    present = {"connected": ["slack"], "installed": ["blender"]}
    monkeypatch.setattr("tools.setup_beat_tool._PRESENT", {
        "connectors": lambda: {"connected": present["connected"]},
        "plugins": lambda: {"installed": present["installed"]}})
    # No connector list unless a test gives one: suggestions are not checked.
    monkeypatch.setattr("tools.setup_beat_tool._connector_slugs", lambda: None)
    return present


@pytest.fixture
def desktop():
    """A card answerer: replies come off a queue; every payload it was shown is kept, and every time the
    tool ended the agent's turn."""
    shown, replies, ended = [], [], []

    def answer(payload):
        shown.append(payload)
        return replies.pop(0) if replies else None
    return SimpleNamespace(shown=shown, replies=replies, answer=answer, ended=ended)


def call(db, desktop, call_id="call_1", **args):
    agent = SimpleNamespace(_session_db=db, session_id=SID, setup_beat_callback=desktop.answer,
                            interrupt=lambda **kw: desktop.ended.append(kw))
    return json.loads(INLINE_TOOL_EXECUTORS["setup_beat"](agent, args, InlineToolContext("task", call_id)))


def test_a_card_answer_settles_the_beat_and_names_the_next(db, desktop):
    onboarding_record.start(db, SID, FACTS)
    desktop.replies.append({"answer": {"accent": "Ultraviolet"}})

    result = call(db, desktop, beat="theme", question="Pick one.")

    assert desktop.shown == [{"beat": "theme", "question": "Pick one.", "card": {"picker": "accent"},
                              "tool_call_id": "call_1"}]
    assert result["answer"] == {"accent": "Ultraviolet"} and result["next"] == "name"
    assert "Ultraviolet" in result["reminder"]
    assert onboarding_record.load(db, SID)["beats"]["theme"] == "done"


def test_next_shows_the_first_open_beat_and_the_name_card_offers_the_account_name(db, desktop, monkeypatch):
    monkeypatch.setattr("agent.onboarding_facts.suggested_name", lambda: "Ada Lovelace")
    onboarding_record.start(db, SID, FACTS)
    desktop.replies.append({"said": "Ada"})

    result = call(db, desktop, beat="next")

    assert desktop.shown[0]["beat"] == "name" and desktop.shown[0]["card"] == {"suggested": "Ada Lovelace"}
    # Words typed while the name card waits are the name; the account name reaches the model only if picked.
    assert result["answer"] == {"name": "Ada"} and "Ada Lovelace" not in json.dumps(result)
    assert result["said"] == "Ada"  # drawn as the user's bubble, not a receipt
    assert desktop.ended == []  # the name answered the card: the turn goes on


def test_a_message_sent_while_a_card_waits_closes_it_unanswered_and_ends_the_turn(db, desktop):
    onboarding_record.start(db, SID, FACTS)
    timed_out = call(db, desktop, beat="layout")
    desktop.replies.append({"wrote_instead": True})
    wrote = call(db, desktop, beat="layout", call_id="call_2")

    assert timed_out["timed_out"] is True and "closed without an answer" in timed_out["reminder"]
    # The message is the user's own next turn: the result carries none of it, and the turn ends on it.
    assert wrote["beat"] == "layout" and wrote["wrote_instead"] is True and "said" not in wrote
    assert "wrote a message instead" in wrote["reminder"]
    assert desktop.ended == [{"tool_reason": "user sent a new message"}]
    record = onboarding_record.load(db, SID)
    assert record["waiting"] is None and record["beats"]["layout"] == "shown"


def test_a_card_closed_by_words_a_value_or_the_deadline_reports_what_is_connected_and_installed(db, desktop):
    onboarding_record.start(db, SID, FACTS)
    timed_out = call(db, desktop, beat="plugins")
    call(db, desktop, beat="connectors", call_id="call_2")  # refused once: the look-around offer
    desktop.replies.append({"wrote_instead": True})
    wrote = call(db, desktop, beat="connectors", call_id="call_3")
    continued = call(db, desktop, beat="connectors", value="continue")

    assert timed_out["installed"] == ["blender"] and wrote["connected"] == ["slack"]
    assert continued["connected"] == ["slack"] and "Connected: slack" in continued["reminder"]
    assert onboarding_record.load(db, SID)["picks"]["installed"] == ["blender"]


def test_suggest_pins_connectors_and_value_or_skip_records_without_a_card(db, desktop):
    onboarding_record.start(db, SID, FACTS)
    desktop.replies.append({"answer": {"connected": ["slack"], "newly_connected": ["slack"]}})
    call(db, desktop, beat="connectors", suggest=["slack", "linear"])
    recorded = call(db, desktop, beat="tour", value="skip")
    skipped = call(db, desktop, beat="plugins", skip=True)

    assert desktop.shown[0]["card"] == {"suggested": ["slack", "linear"]} and len(desktop.shown) == 1
    assert recorded["recorded"] == "tour" and "No tour" in recorded["reminder"]
    assert skipped["skipped"] is True


def test_the_offer_lists_the_real_connectors_and_an_unlisted_suggestion_shows_no_card(db, desktop, monkeypatch):
    # The model guessed GitHub from the gh CLI; this account cannot connect it, and the card showed nothing.
    monkeypatch.setattr("tools.setup_beat_tool._connector_slugs", lambda: {"slack", "linear"})
    onboarding_record.start(db, SID, FACTS)
    for beat in ("name", "theme", "layout"):
        call(db, desktop, beat=beat, skip=True)

    offer = call(db, desktop, beat="next")
    refused = call(db, desktop, beat="connectors", suggest=["github", "slack"], call_id="call_2")
    desktop.replies.append({"answer": {"connected": [], "newly_connected": []}})
    call(db, desktop, beat="connectors", suggest=["slack"], call_id="call_3")

    assert "by these slugs: linear, slack." in offer["reminder"]
    assert refused["error"] == "not connectors here" and "No card shown: github cannot" in refused["reminder"]
    assert [card["card"] for card in desktop.shown] == [{"suggested": ["slack"]}]


def test_next_to_the_connectors_card_before_the_offer_is_refused_without_a_card(db, desktop):
    onboarding_record.start(db, SID, FACTS)
    for beat in ("name", "theme", "layout"):
        call(db, desktop, beat=beat, skip=True)

    refused = call(db, desktop, beat="next")
    desktop.replies.append({"answer": {"connected": [], "newly_connected": []}})
    shown = call(db, desktop, beat="next", call_id="call_2")

    assert refused["error"] == "offer first" and "beat" not in refused
    assert "offer once to look around" in refused["reminder"]
    assert [card["beat"] for card in desktop.shown] == ["connectors"]
    assert shown["answer"] == {"connected": [], "newly_connected": []} and shown["next"] == "plugins"


def test_skip_setup_during_a_card_returns_skipped_all(db, desktop):
    onboarding_record.start(db, SID, FACTS)

    def skip_pressed(payload):
        onboarding_record.advance(db, SID, End())

    desktop.answer = skip_pressed
    result = call(db, desktop, beat="theme")

    assert result["skipped_all"] is True and "stop guiding" in result["reminder"]
    assert "error" in call(db, desktop, beat="next")


def test_done_ends_the_run_only_after_the_build_card(db, desktop):
    onboarding_record.start(db, SID, FACTS)
    refused = call(db, desktop, beat="done")
    # The model cannot end it early: a user who wants to stop is pointed at the Skip setup button.
    quit_in_words = call(db, desktop, beat="done", skip=True)
    assert "Skip setup" in quit_in_words["error"] and onboarding_record.load(db, SID)["phase"] == "onboarding"
    # No picks: refused with the eligible ids, and no card goes out.
    no_picks = call(db, desktop, beat="build")
    assert "notes" in no_picks["error"] and desktop.shown == []
    desktop.replies.append({"answer": {"build": "notes"}})
    build = call(db, desktop, beat="build", options=[{"id": "notes", "label": "Notes to wiki"}])
    assert desktop.shown[-1]["card"]["options"][-1] == {"id": "hermes", "label": "Explain how Hermes works"}
    done = call(db, desktop, beat="done")

    assert "error" in refused
    assert "How does this look?" in build["reminder"] and build["completed"] is True
    assert done == {"completed": True, "reminder": done["reminder"]}
    assert onboarding_record.load(db, SID)["phase"] == "complete"


def test_outside_an_onboarding_chat_the_tool_refuses(db, desktop):
    assert "not running" in call(db, desktop, beat="next")["error"]
    # A bare registry dispatch has no session store or card callback.
    assert "error" in json.loads(registry.dispatch("setup_beat", {"beat": "next"}))
