"""`config.set voice.voice_chat_mode` is how the composer's voice menu swaps engines.

The renderer's radio row writes through this key and then re-reads the resolved status; if
the key were unlisted the handler would answer 4002 and the menu would show a switch that
never lands on disk.
"""

import pytest
import hermes_yaml as yaml

from tui_gateway import server


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    yield tmp_path / "config.yaml"
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


def _set(value):
    return server._methods["config.set"](1, {"key": "voice.voice_chat_mode", "value": value})


def test_engine_choice_reaches_the_config_file_and_round_trips(config_home):
    assert _set("gpt-live")["result"] == {"key": "voice.voice_chat_mode", "value": "gpt-live"}
    assert yaml.safe_load(config_home.read_text())["voice"]["voice_chat_mode"] == "gpt-live"

    assert _set("Chained ")["result"]["value"] == "chained"
    assert yaml.safe_load(config_home.read_text())["voice"]["voice_chat_mode"] == "chained"


def test_unknown_engine_is_refused_rather_than_written(config_home):
    answer = _set("realtime")

    assert answer["error"]["code"] == 4002
    assert not config_home.exists()


def test_extension_engine_id_is_written_as_the_mode(config_home):
    """A Desktop extension engine (`<plugin>:<engine>`) is a mode the menu can pick; the plugin's
    presence is the renderer's check, so the writer validates the shape only."""
    assert _set("Hermes-XAI:grok")["result"] == {"key": "voice.voice_chat_mode", "value": "hermes-xai:grok"}
    assert yaml.safe_load(config_home.read_text())["voice"]["voice_chat_mode"] == "hermes-xai:grok"


@pytest.mark.parametrize("value", ["hermes-xai:", "hermes xai:grok", "a:b:c", "../x:y"])
def test_malformed_extension_ids_are_refused(config_home, value):
    answer = _set(value)

    assert answer["error"]["code"] == 4002
    assert "<plugin>:<engine>" in answer["error"]["message"]
    assert not config_home.exists()
