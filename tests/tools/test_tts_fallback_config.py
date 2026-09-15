"""Invariant tests for tts.fallback model-override routing (PR #103928).

Regression: model overrides on a cross-provider fallback entry were written to
the hardcoded ``gemini`` config section, so ``{provider: openai, model: ...}``
was silently dropped (the openai provider never reads the gemini section). The
override now lands on the entry's OWN provider section.
"""
from __future__ import annotations

from tools.tts_tool import _fallback_config_with_model


def test_gemini_override_lands_on_gemini_section():
    """Same-provider override (gemini -> older gemini) still works."""
    cfg = {"gemini": {"model": "current"}, "edge": {"voice": "en"}}
    out = _fallback_config_with_model(cfg, {"provider": "gemini", "model": "older"})
    assert out["gemini"]["model"] == "older"
    assert out["edge"] == {"voice": "en"}  # sibling section untouched


def test_cross_provider_override_lands_on_its_own_section():
    """A cross-provider override is honored, not dropped into the gemini block."""
    cfg = {"gemini": {"model": "current"}, "openai": {"voice": "a"}}
    out = _fallback_config_with_model(cfg, {"provider": "openai", "model": "gpt-4o-mini-tts"})
    assert out["openai"]["model"] == "gpt-4o-mini-tts"
    assert out["gemini"] == {"model": "current"}  # not leaked into gemini


def test_no_model_returns_config_unchanged():
    """An entry without a model override returns the config untouched."""
    cfg = {"gemini": {"model": "current"}}
    assert _fallback_config_with_model(cfg, {"provider": "gemini"}) == cfg
