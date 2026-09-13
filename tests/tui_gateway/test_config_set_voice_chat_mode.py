"""`config.set voice.voice_chat_mode` is how the composer's voice menu swaps engines.

The renderer's radio row writes through this key and then re-reads the resolved status; if
the key were unlisted the handler would answer 4002 and the menu would show a switch that
never lands on disk.
"""

import pytest
import yaml

from tui_gateway import server


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None
    yield tmp_path / "config.yaml"
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None


def _set(value, key="voice.voice_chat_mode"):
    return server._methods["config.set"](1, {"key": key, "value": value})


def test_engine_choice_reaches_the_config_file_and_round_trips(config_home):
    assert _set("gpt-live")["result"] == {"key": "voice.voice_chat_mode", "value": "gpt-live"}
    assert yaml.safe_load(config_home.read_text())["voice"]["voice_chat_mode"] == "gpt-live"

    assert _set("Chained ")["result"]["value"] == "chained"
    assert yaml.safe_load(config_home.read_text())["voice"]["voice_chat_mode"] == "chained"


def test_unknown_engine_is_refused_rather_than_written(config_home):
    answer = _set("realtime")

    assert answer["error"]["code"] == 4002
    assert not config_home.exists()


def test_busy_delegation_choice_is_profile_scoped_config(config_home):
    key = "voice.gpt_live.busy_delegation_mode"
    assert _set("queue", key)["result"] == {"key": key, "value": "queue"}
    assert yaml.safe_load(config_home.read_text())["voice"]["gpt_live"]["busy_delegation_mode"] == "queue"

    assert _set("Interrupt ", key)["result"]["value"] == "interrupt"
    assert yaml.safe_load(config_home.read_text())["voice"]["gpt_live"]["busy_delegation_mode"] == "interrupt"


def test_unknown_busy_delegation_choice_is_refused(config_home):
    answer = _set("steer", "voice.gpt_live.busy_delegation_mode")

    assert answer["error"]["code"] == 4002
    assert not config_home.exists()
