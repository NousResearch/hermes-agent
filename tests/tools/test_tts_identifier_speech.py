"""Identifier-to-speech normalization (#119207).

Both speech normalizers — tools/tts_text_normalize.py (gateway/CLI/TTS tool)
and apps/desktop/src/lib/speech-text.ts (Desktop read-aloud / auto-speak) —
must apply the same policy to identifier-dense tokens (filenames with
extensions, hashes, UUIDs, dense model/version IDs, paths): silence, never
hardcoded English placeholder words (#86602 — the reply may be in any
language). This suite pins the Python half against the shared corpus
(tests/fixtures/identifier_speech_corpus.json); the vitest suite in
apps/desktop runs the identical corpus, so the two implementations stay in
lockstep.
"""

import json
from pathlib import Path

from tools.tts_text_normalize import prepare_spoken_text

_CORPUS = Path(__file__).resolve().parent.parent / "fixtures" / "identifier_speech_corpus.json"


def _cases() -> dict:
    return json.loads(_CORPUS.read_text())


def test_identifier_tokens_become_silence_prose_survives():
    for text, must_not_contain, must_contain in _cases()["identifier_tokens"]:
        spoken = prepare_spoken_text(text)
        for needle in must_not_contain:
            assert needle not in spoken, f"{needle!r} leaked from {text!r} into {spoken!r}"
        for needle in must_contain:
            assert needle in spoken, f"{needle!r} lost from {text!r} -> {spoken!r}"


def test_ordinary_speech_passes_through():
    for text, _unused, must_contain in _cases()["pass_through_tokens"]:
        spoken = prepare_spoken_text(text)
        for needle in must_contain:
            assert needle in spoken, f"{needle!r} lost from {text!r} -> {spoken!r}"


def test_issue_repro_filenames_are_not_spoken():
    spoken = prepare_spoken_text(
        "Saved peyton-sample-20260922.wav and peyton-sample-20260922.ogg.")
    assert ".wav" not in spoken
    assert ".ogg" not in spoken
    assert "peyton" not in spoken
    assert "Saved" in spoken


def test_hash_and_uuid_summaries_do_not_spell_out():
    spoken = prepare_spoken_text(
        "Commit 73688014f78 with trace 550e8400-e29b-41d4-a716-446655440000.")
    assert "73688014f78" not in spoken
    assert "550e8400" not in spoken
    assert "Commit" in spoken and "trace" in spoken


def test_plain_words_with_hyphens_and_digits_stay_spoken():
    spoken = prepare_spoken_text("COVID-19 numbers and the well-known 7 habit stay.")
    assert "COVID-19" in spoken
    assert "well-known" in spoken
    assert "7" in spoken


def test_dates_and_ratios_are_not_identifiers():
    spoken = prepare_spoken_text("Due 2026/06/02; the score was 3:2.")
    assert "2026/06/02" in spoken
    assert "3:2" in spoken
