"""Tests for the description-aware slash fuzzy scorer (grok-cli port).

Covers ``tui_gateway.slash_fuzzy`` (scoring tiers, catalog merge, stable
ordering) and how ``_rank_slash_completions`` consumes the ``score_of``
lookup: skill rows sort by fuzzy score first, then usage, then name.
"""

import math

from tui_gateway.server import _rank_slash_completions
from tui_gateway.slash_fuzzy import (
    fuzzy_rank_slash_items,
    normalize_slash_search_query,
    score_slash_completion_item,
    tokenize_search_text,
)


def _item(text, meta="", kind="command"):
    return {"text": text, "display": text, "meta": meta, "kind": kind}


def test_normalize_slash_search_query():
    assert normalize_slash_search_query(" /Model ") == "model"
    assert normalize_slash_search_query("//help") == "help"
    assert normalize_slash_search_query("plain") == "plain"


def test_tokenize_search_text_includes_full_value_and_words():
    assert tokenize_search_text("Commit & Push") == ["commit & push", "commit", "push"]


def test_score_tiers_name_before_description():
    item = _item("/recaps ", "Turn session recaps on/off")
    assert score_slash_completion_item(item, "recaps") == 0
    assert score_slash_completion_item(item, "rec") == 1
    assert score_slash_completion_item(item, "caps") == 2
    assert score_slash_completion_item(item, "session") == 3
    assert score_slash_completion_item(item, "sess") == 4
    assert score_slash_completion_item(item, "essio") == 5
    assert math.isinf(score_slash_completion_item(item, "zzz"))


def test_name_match_beats_description_match():
    # "recap" hits the description too, but the name tier must win.
    item = _item("/recap", "Turn session recaps on/off")
    assert score_slash_completion_item(item, "recap") == 0


def test_fuzzy_rank_merges_description_matches_from_catalog():
    prefix_hits = [_item("/summon")]
    catalog = [
        _item("/summon"),
        _item("/recaps", "Show a summary of the session"),
        _item("/help", "Show available commands"),
    ]
    ranked, score_of = fuzzy_rank_slash_items(prefix_hits, catalog, "summ")

    texts = [item["text"] for item in ranked]
    assert texts == ["/summon", "/recaps"]  # name prefix (1) before description (4)
    assert score_of(ranked[0]) == 1
    assert score_of(ranked[1]) == 4
    assert math.isinf(score_of(_item("/help", "Show available commands")))


def test_fuzzy_rank_is_stable_within_a_tier():
    items = [_item("/mod-b"), _item("/mod-a")]
    ranked, _ = fuzzy_rank_slash_items(items, [], "mod")
    assert [item["text"] for item in ranked] == ["/mod-b", "/mod-a"]


def test_fuzzy_rank_drops_non_matching_prefix_rows():
    ranked, _ = fuzzy_rank_slash_items([_item("/other")], [], "model")
    assert ranked == []


def test_rank_slash_completions_uses_score_before_usage():
    # Without a scorer, usage sorts skills; with one, score leads and usage
    # only breaks ties within a tier.
    name_hit = _item("/summarize", "Condense text", kind="skill")
    desc_hit = _item("/notes", "Write a summary of a meeting", kind="skill")
    items = [desc_hit, name_hit]

    usage = {"notes": 50, "summarize": 1}.get

    def usage_of(name):
        return usage(name, 0)

    def origin_of(_name):
        return "user"

    scores = {id(name_hit): 1.0, id(desc_hit): 4.0}

    ranked = _rank_slash_completions(
        items,
        usage_of,
        origin_of,
        browsing=False,
        score_of=lambda item: scores.get(id(item), math.inf),
    )
    assert [item["text"] for item in ranked] == ["/summarize", "/notes"]

    # Sanity: without score_of the heavier-used skill leads.
    ranked_plain = _rank_slash_completions(items, usage_of, origin_of, browsing=False)
    assert [item["text"] for item in ranked_plain] == ["/notes", "/summarize"]


def test_rank_slash_completions_does_not_truncate_commands_when_browsing():
    # A bare `/` must surface every registry command (issue: /goal, /loop,
    # /queue sat past position 30 and never reached the client) while the
    # user-growable skill block keeps its cap.
    commands = [_item(f"/cmd{i}") for i in range(40)]
    registry_names = frozenset(f"cmd{i}" for i in range(40))

    ranked = _rank_slash_completions(
        commands,
        lambda _name: 0,
        lambda _name: "user",
        browsing=True,
        registry_command_names=registry_names,
    )
    assert len(ranked) == 40
    assert ranked[-1]["text"] == "/cmd39"


def test_rank_slash_completions_still_caps_plugin_commands_when_browsing():
    # Plugin-registered commands (kind: "command", same as registry
    # commands) are an unbounded, user-installed source, unlike
    # COMMAND_REGISTRY — a bare `/` must still bound them, or a large
    # plugin install floods the browse list the way skills used to.
    registry = [_item(f"/reg{i}") for i in range(40)]
    plugins = [_item(f"/plugin{i}") for i in range(40)]
    registry_names = frozenset(f"reg{i}" for i in range(40))

    ranked = _rank_slash_completions(
        registry + plugins,
        lambda _name: 0,
        lambda _name: "user",
        browsing=True,
        registry_command_names=registry_names,
    )

    texts = [item["text"] for item in ranked]
    assert texts[:40] == [f"/reg{i}" for i in range(40)]
    plugin_texts = texts[40:]
    assert len(plugin_texts) == 30
    assert plugin_texts == [f"/plugin{i}" for i in range(30)]


def test_rank_slash_completions_defaults_to_the_real_command_registry():
    # No registry_command_names override: this must wire up to the actual
    # hermes_cli.commands.GATEWAY_KNOWN_COMMANDS, not just a test double.
    # /help is a real registry command; /plugin* stands in for names a
    # plugin registered (kind "command", but not in the registry) and
    # must still be capped.
    items = [_item("/help")] + [_item(f"/plugin{i}") for i in range(40)]

    ranked = _rank_slash_completions(
        items,
        lambda _name: 0,
        lambda _name: "user",
        browsing=True,
    )

    texts = [item["text"] for item in ranked]
    assert texts[0] == "/help"
    assert len([t for t in texts if t.startswith("/plugin")]) == 30


def test_rank_slash_completions_still_caps_commands_when_searching():
    commands = [_item(f"/cmd{i}") for i in range(40)]

    ranked = _rank_slash_completions(
        commands,
        lambda _name: 0,
        lambda _name: "user",
        browsing=False,
    )
    assert len(ranked) == 30


def test_rank_slash_completions_ties_break_on_usage_then_name():
    a = _item("/beta", kind="skill")
    b = _item("/alpha", kind="skill")
    items = [a, b]

    ranked = _rank_slash_completions(
        items,
        lambda name: {"alpha": 3, "beta": 3}.get(name, 0),
        lambda _name: "user",
        browsing=False,
        score_of=lambda item: 1.0,
    )
    assert [item["text"] for item in ranked] == ["/alpha", "/beta"]
