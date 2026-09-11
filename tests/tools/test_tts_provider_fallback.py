"""Behaviour contract for an unresolvable TTS provider.

A provider name that is neither a built-in engine nor claimed by a plugin used to
be answered with Edge audio while the configured name was still reported, so the
log read ``provider: gcloud-tts`` over an Edge render and the only way to notice
was to recognise the wrong voice.

Two things must hold when that substitution happens: it is announced, and the
reported provider names the engine that actually spoke.
"""

import logging

from tools.tts_tool import _select_builtin_engine


def test_unresolvable_provider_falls_back_to_edge_and_says_so(caplog):
    """The substitution is logged at WARNING and names both providers."""
    with caplog.at_level(logging.WARNING, logger="tools.tts_tool"):
        engine, error = _select_builtin_engine("gcloud-tts")

    assert error is None, "Edge is available, so this is a fallback and not a failure"
    assert engine == "edge", "the reported engine must be the one that will speak"

    warning = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "gcloud-tts" in warning and "Edge" in warning, (
        "a silent substitution is the bug; the warning must name what was asked "
        f"for and what was used. Got: {warning!r}")


def test_a_real_builtin_is_returned_unchanged():
    """Resolution still passes known engines straight through."""
    engine, error = _select_builtin_engine("gemini")
    assert engine == "gemini" and error is None
