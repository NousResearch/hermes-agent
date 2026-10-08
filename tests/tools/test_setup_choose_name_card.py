"""The account's full name reaches the model only when the user picks it on the setup name card."""

import json

from agent import initiate_setup_facts
from agent.initiate_setup_prompt import NAME_QUESTION, _opening
from tools.setup_choose_tool import setup_choose_tool

NAME = "Ada Lovelace"


def _ask(monkeypatch, reply, session_id=None):
    monkeypatch.setattr(initiate_setup_facts, "suggested_name", lambda: NAME)
    shown = []

    def callback(payload):
        shown.append(payload)
        return reply

    result = setup_choose_tool(kind="question", question=NAME_QUESTION, options=[], callback=callback,
                               session_id=session_id)
    return shown[0], json.loads(result)


def test_name_card_shows_the_account_name_and_returns_it_only_when_picked(monkeypatch):
    shown, picked = _ask(monkeypatch, {"picked": "suggested"})
    assert shown["options"] == [{"id": "suggested", "label": NAME}]
    assert picked["label"] == NAME

    _, typed = _ask(monkeypatch, {"picked": "Ada"})
    assert NAME not in json.dumps(typed)


def test_facts_and_the_scripted_card_leave_the_name_out(monkeypatch):
    monkeypatch.setattr(initiate_setup_facts, "_account", lambda: ("ada", NAME))
    assert NAME not in json.dumps(initiate_setup_facts.facts())

    opening = _opening(set(), {})
    _text, _tool, card = next(opening)
    assert NAME not in json.dumps(card)
    line, _tool, _card = opening.send(json.dumps({"outcome": "submitted", "picked": "suggested", "label": NAME}))
    assert line == f"Good to meet you, {NAME}."


def test_a_name_typed_in_the_composer_is_the_name_answer(monkeypatch):
    from hermes_cli.setup_profile import read_cards

    _, typed = _ask(monkeypatch, {"said": "Ada"}, session_id="setup-typed-name")
    assert (typed["outcome"], typed["picked"]) == ("submitted", "Ada")
    assert read_cards("setup-typed-name")["picks"]["name"] == "Ada"

    opening = _opening(set(), {})
    next(opening)
    line, _tool, card = opening.send(json.dumps(typed))
    assert (line, card["kind"]) == ("Good to meet you, Ada.", "accent")
