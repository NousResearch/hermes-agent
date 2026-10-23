"""First-run onboarding end to end through the gateway's real card path: setup_beat runs in the agent's
inline executor, its card goes out as the ``setup_beat`` server request (params checked against the
contract), a fake desktop answers each frame, and general tool results settle beats through the
executor's hook. The model is scripted: each step is the call it would make after reading the
previous result."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import onboarding_record
from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from agent.onboarding import observe_tool_result
from agent.onboarding_flow import Facts
from hermes_cli.config import read_user_config_raw
from tui_gateway import server, server_requests

SID, STORED = "ui-onboarding", "onboarding-chat"
FACTS = Facts(os="windows", arch="amd64", machine_kind="PC", nvidia_platform="discrete",
              plugins=("nvidia-app", "nvidia-broadcast", "blender"))


@pytest.fixture
def chat(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    db = server._get_db()
    db.create_session(STORED, "desktop")
    onboarding_record.start(db, STORED, FACTS)
    monkeypatch.setitem(server._sessions, SID, {"session_key": STORED, "source": "desktop"})
    ended = []
    agent = SimpleNamespace(_session_db=db, session_id=STORED, valid_tool_names={"setup_beat", "set_theme"},
                            setup_beat_callback=server._agent_cbs(SID)["setup_beat_callback"],
                            interrupt=lambda **kw: ended.append(kw), _gateway_turn_context_notes="")
    desktop = SimpleNamespace(cards=[], replies=[])

    def answer(frame):
        desktop.cards.append(frame["params"])
        result = desktop.replies.pop(0)
        assert server_requests.resolve_response({"jsonrpc": "2.0", "id": frame["id"], "result": result})
        return True

    monkeypatch.setattr(server_requests, "_answerable", lambda _sid: True)
    monkeypatch.setattr(server_requests, "_write", answer)
    monkeypatch.setattr(server_requests, "_emit", lambda *a: None)
    return SimpleNamespace(db=db, agent=agent, desktop=desktop, home=home, ended=ended)


def beat(chat, reply=None, **args):
    if reply is not None:
        chat.desktop.replies.append(reply)
    out = INLINE_TOOL_EXECUTORS["setup_beat"](chat.agent, args, InlineToolContext("task", f"call_{len(chat.desktop.cards)}"))
    return json.loads(out)


def test_every_beat_from_the_name_card_to_done(chat):
    named = beat(chat, {"answer": {"name": "Sid"}}, beat="name")
    assert named["next"] == "theme" and "Sid" in named["reminder"]

    assert beat(chat, {"answer": {"accent": "Ultraviolet"}}, beat="next")["next"] == "layout"

    # Typed instead of picking: the card closes, the words are the next turn, the model applies the pick
    # itself through set_theme.
    typed = beat(chat, {"wrote_instead": True}, beat="next")
    assert typed["wrote_instead"] is True
    observe_tool_result(chat.agent, "set_theme", {"layout": "terminal-deck"},
                        json.dumps({"success": True, "applied": {"layout": "terminal-deck"}}))

    connectors = beat(chat, {"answer": {"connected": ["slack"], "newly_connected": ["slack"]}},
                      beat="connectors", suggest=["slack", "outlook"])
    assert connectors["next"] == "plugins"
    plugins = beat(chat, {"answer": {"installed": ["nvidia-app"]}}, beat="next")
    assert plugins["next"] == "tour"
    tour = beat(chat, {"answer": {"tour": "quick"}}, beat="next")
    assert '"preset":"quick"' in tour["reminder"]
    observe_tool_result(chat.agent, "gui_tour", {"action": "start", "preset": "quick"}, json.dumps({"success": True}))

    build = beat(chat, {"answer": {"build": "games"}}, beat="next", options=[{"id": "games", "label": "Tune my games"}])
    assert "--accept-package-agreements" in build["reminder"] and "How does this look?" in build["reminder"]
    assert beat(chat, beat="done")["completed"] is True

    assert [card["beat"] for card in chat.desktop.cards] == [
        "name", "theme", "layout", "connectors", "plugins", "tour", "build"]
    assert chat.desktop.cards[3]["card"] == {"suggested": ["slack", "outlook"]}
    assert [row["id"] for row in chat.desktop.cards[6]["card"]["options"]][:1] == ["games"]
    record = onboarding_record.load(chat.db, STORED)
    assert record["phase"] == "complete" and record["picks"]["layout"] == "terminal-deck"
    onboarding = read_user_config_raw(chat.home / "config.yaml")["onboarding"]
    assert onboarding["run"] is False and onboarding["seen"]["profile_build_offered"] is True
    assert "error" in beat(chat, beat="next")


def test_a_message_written_while_a_card_waits_ends_the_turn_and_the_next_turn_brings_the_card_back(chat):
    beat(chat, {"answer": {"name": "Sid"}}, beat="name")
    beat(chat, {"answer": {"accent": "Ultraviolet"}}, beat="next")

    closed = beat(chat, {"wrote_instead": True}, beat="next")

    assert closed["beat"] == "layout" and closed["wrote_instead"] is True and "said" not in closed
    assert chat.ended == [{"tool_reason": "user sent a new message"}]
    # Their message starts the next turn; its note names the closed card, how a typed answer applies it,
    # and how to bring it back.
    server._stage_onboarding_turn_note(SID, chat.agent, history_empty=False)
    note = chat.agent._gateway_turn_context_notes
    assert "the layout card is closed" in note and 'setup_beat(beat="layout")' in note
    assert '"layout":"sidebar-left" or "terminal-deck"' in note


def test_skip_setup_on_the_card_ends_onboarding(chat, monkeypatch):
    def skip_on_show(frame):
        chat.desktop.cards.append(frame["params"])
        assert server.handle_request({"jsonrpc": "2.0", "id": 9, "method": "onboarding.skip",
                                      "params": {"session_id": SID}})["result"] == {"skipped": True}
        return True

    monkeypatch.setattr(server_requests, "_write", skip_on_show)
    skipped = beat(chat, beat="next")

    assert skipped["skipped_all"] is True and "stop guiding" in skipped["reminder"]
    assert onboarding_record.load(chat.db, STORED)["phase"] == "skipped"
