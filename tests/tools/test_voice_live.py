"""``voice.voice_chat_mode`` resolution: built-in engines and extension engine ids.

Desktop mounts the conversation engine from the mode the status route reports. An extension
engine (``<plugin>:<engine>``, registered by a Desktop plugin on ``VOICE_ENGINE_AREA``) must reach
the renderer as written: collapsing it to ``chained`` would make the selection impossible to keep.
"""

import os
from pathlib import Path

import pytest

import hermes_yaml as yaml
from tools import voice_live


def _write_voice_config(voice: dict) -> None:
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"voice": voice}))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "chained"),
        ("chained", "chained"),
        ("gpt-live", "gpt-live"),
        ("GPT_LIVE", "gpt-live"),
        ("hermes-xai:grok", "hermes-xai:grok"),
        (" Hermes-XAI:Grok ", "hermes-xai:grok"),
        ("my_plugin:live_1", "my_plugin:live_1"),  # plugin ids keep their underscores
        ("realtime", "chained"),
        ("hermes-xai:", "chained"),
        (":grok", "chained"),
        ("a:b:c", "chained"),
        ("bad id:grok", "chained"),
        ("x" * 65 + ":grok", "chained"),
    ],
)
def test_voice_chat_mode_keeps_extension_ids_and_collapses_the_rest(raw, expected):
    assert voice_live.voice_chat_mode({"voice_chat_mode": raw}) == expected


@pytest.mark.parametrize("value", ["chained", "gpt-live", "hermes-xai:grok", "a:b", "p_1:e-2", "x" * 64 + ":y"])
def test_writers_accept_builtin_and_extension_modes(value):
    assert voice_live.is_voice_chat_mode(value)


@pytest.mark.parametrize(
    "value", ["", "realtime", "gpt_live", "Hermes-xai:grok", "-x:grok", "x:-grok", "x:", ":y", "x:y:z", "x/y:z", "x" * 65 + ":y"]
)
def test_writers_reject_everything_else(value):
    assert not voice_live.is_voice_chat_mode(value)


def test_status_reports_the_extension_mode_and_keeps_gpt_live_readiness(monkeypatch):
    """The route's payload for an extension engine: the namespaced mode, with ``available`` still
    describing GPT-Live (its composer row needs it) and no credential in the answer."""
    _write_voice_config({"voice_chat_mode": "hermes-xai:grok", "gpt_live": {"voice": "cedar"}})
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("sk-secret", voice_live.DEFAULT_LIVE_BASE_URL))

    status = voice_live.resolve_gpt_live_status()

    assert status["mode"] == "hermes-xai:grok"
    assert status["available"] is True and status["reason"] is None
    assert status["voice"] == "cedar"
    assert "sk-secret" not in repr(status)


def test_status_without_gpt_live_key_still_reports_the_extension_mode(monkeypatch):
    _write_voice_config({"voice_chat_mode": "hermes-xai:grok"})
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("", voice_live.DEFAULT_LIVE_BASE_URL))

    status = voice_live.resolve_gpt_live_status()

    assert status["mode"] == "hermes-xai:grok"
    assert status["available"] is False and "OpenAI" in status["reason"]
