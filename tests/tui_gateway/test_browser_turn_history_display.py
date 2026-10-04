"""Browser turn envelopes display as the typed message in history reads."""

import json

from tui_gateway import server

PROTOCOL = "hermes.browser.turn.v2"


def _envelope(text, *, context=None):
    return json.dumps({
        "protocol": PROTOCOL,
        "human_input": {"source": "composer", "text": text},
        "browser_context": context or {"delivery": "reference", "context_hash": "abc"},
        "attachment_context": {"items": []},
        "source_receipt": {"protocol": PROTOCOL, "version": 2},
    })


def _shown(content, role="user"):
    return [m["text"] for m in server._history_to_messages([{"role": role, "content": content}])]


def test_envelope_shows_only_the_typed_message():
    assert _shown(_envelope("fire, glad it works")) == ["fire, glad it works"]


def test_full_context_envelope_hides_page_payload():
    context = {"delivery": "full", "payload": {"pageContext": {"selectedText": "page text " * 200}}}
    assert _shown(_envelope("summarise this", context=context)) == ["summarise this"]


def test_envelope_wrapped_by_an_older_replay_unwraps_to_the_typed_text():
    assert _shown(_envelope(_envelope(_envelope("typed once")))) == ["typed once"]


def test_storage_and_model_history_are_not_mutated():
    row = {"role": "user", "content": _envelope("hello")}
    history = [row]
    server._history_to_messages(history)
    assert history == [{"role": "user", "content": _envelope("hello")}]


def test_non_envelopes_are_left_byte_identical():
    for text in (
        "plain message",
        '{"protocol": "something.else", "human_input": {"text": "x"}}',
        '{"protocol": "%s"}' % PROTOCOL,
        '{"protocol": "%s", "human_input": {"text": 5}}' % PROTOCOL,
        '{"protocol": "%s", "human_input": "oops"}' % PROTOCOL,
        "look at this " + _envelope("quoted inside prose"),
        _envelope("cut off")[:60],
    ):
        assert _shown(text) == [text]


def test_assistant_rows_are_never_rewritten():
    envelope = _envelope("model echoed this")
    assert _shown(envelope, role="assistant") == [envelope]


def test_inflight_turn_shows_the_typed_message():
    session = {}
    server._start_inflight_turn(session, _envelope("typing live"))
    assert session["inflight_turn"]["user"] == "typing live"
