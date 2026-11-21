"""GPT-Live session voice: named voices are strings, custom voices are ``{"id": ...}`` objects.

OpenAI refuses a custom voice id sent as a bare string with 403 "Voice session access denied";
the same id sent as ``{"id": "voice_..."}`` opens the session.
"""

import pytest

from tools import voice_live


@pytest.fixture
def live_section(monkeypatch):
    section: dict = {}
    monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: section)
    return section


def _voice(config):
    return config["audio"]["output"]["voice"]


@pytest.mark.parametrize("name", ["marin", "gleam", "cedar"])
def test_named_voice_is_sent_as_a_string(live_section, name):
    live_section["voice"] = name
    assert _voice(voice_live.build_session_config()) == name


def test_default_voice_is_a_string(live_section):
    assert _voice(voice_live.build_session_config()) == voice_live.DEFAULT_LIVE_VOICE


def test_custom_voice_id_is_sent_as_an_object(live_section):
    live_section["voice"] = "voice_6b01382b0c808191a887e9cefd5f488d"
    assert _voice(voice_live.build_session_config()) == {"id": "voice_6b01382b0c808191a887e9cefd5f488d"}


def test_custom_voice_id_whitespace_is_trimmed(live_section):
    live_section["voice"] = "  voice_abc123 \n"
    assert _voice(voice_live.build_session_config()) == {"id": "voice_abc123"}
