"""Authored transcript colors survive the supported writer and gateway contract."""

import argparse

import pytest
import yaml

from hermes_cli.skin_cmd import skin_command
from tui_gateway import server
from tui_gateway.contracts.events import SkinPayload


def test_transcript_colors_round_trip_to_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", str(tmp_path))
    monkeypatch.setattr(server, "_cfg_cache", None)
    (tmp_path / "skins").mkdir()
    (tmp_path / "config.yaml").write_text("display:\n  skin: transcript\n")
    original = {"background": "#121212", "ui_accent": "#e68e0d"}
    path = tmp_path / "skins" / "transcript.yaml"
    path.write_text(yaml.safe_dump({"name": "transcript", "colors": original}))
    tokens = {"ui_user": "#e68e0d", "ui_heading": "#4dd0e1", "user_message_bg": "#282828"}
    for key, value in tokens.items():
        with pytest.raises(SystemExit) as result:
            skin_command(argparse.Namespace(skin_command="set", key=key, value=value, skin=None))
        assert result.value.code == 0

    saved = yaml.safe_load(path.read_text())["colors"]
    assert saved == {**original, **tokens}
    payload = SkinPayload.model_validate(server.resolve_skin())
    for key, value in saved.items():
        assert payload.colors[key] == value
