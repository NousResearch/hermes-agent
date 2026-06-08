from __future__ import annotations

from wisdom.classify import classify_capture, detect_explicit_trigger


def test_explicit_prefixes_capture_and_clean_text():
    match = detect_explicit_trigger("  Business idea: report language for prospects")
    assert match is not None
    assert match.category_hint == "business"
    assert match.cleaned_text == "report language for prospects"


def test_contains_only_phrase_does_not_capture():
    assert detect_explicit_trigger("I should remember this later") is None


def test_non_wisdom_slash_command_is_ignored():
    assert detect_explicit_trigger("/todo now call Yash") is None


def test_category_and_source_rules():
    assert classify_capture("client report sales").category == "business"
    assert classify_capture("portfolio risk sizing").category == "investing"
    assert classify_capture("sleep changes decision quality").category == "health"
    assert classify_capture("family meaning habit").category == "life"
    assert classify_capture("ambiguous unrelated text").category == "inbox"
    podcast = detect_explicit_trigger("Podcast idea: market structure")
    assert podcast is not None
    assert classify_capture("Podcast idea: market structure", podcast.cleaned_text, podcast).source_type == "podcast"
