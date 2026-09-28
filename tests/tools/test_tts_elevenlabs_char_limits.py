"""ElevenLabs per-request char limits resolve from the model-id map (#126676).

``eleven_v4`` / ``eleven_v4_turbo`` (launched 2026-09-28) carry a 10,000-char limit.
They happened to fall through to the provider fallback of 10000 by accident; the map
entry is the explicit contract, next to ``eleven_v3: 5000``.
"""
from tools.tts_tool_delivery import ELEVENLABS_MODEL_MAX_TEXT_LENGTH


def test_eleven_v4_limits_are_explicit():
    assert ELEVENLABS_MODEL_MAX_TEXT_LENGTH.get("eleven_v4") == 10000
    assert ELEVENLABS_MODEL_MAX_TEXT_LENGTH.get("eleven_v4_turbo") == 10000


def test_v3_limit_still_explicit():
    assert ELEVENLABS_MODEL_MAX_TEXT_LENGTH.get("eleven_v3") == 5000
