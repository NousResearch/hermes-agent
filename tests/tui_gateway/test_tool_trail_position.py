"""Tool-trail position uses the real config RPC and preserves section visibility."""

import hermes_yaml as yaml
from tui_gateway import server


def test_config_set_tool_trail_position_writes_display_key(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"display": {"sections": {"tools": "expanded"}}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "tool_trail_position", "value": "below"},
        }
    )

    assert resp["result"] == {"key": "tool_trail_position", "value": "below"}
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["display"]["tool_trail_position"] == "below"
    assert saved["display"]["sections"] == {"tools": "expanded"}


def test_config_set_tool_trail_position_rejects_unknown_value(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "tool_trail_position", "value": "underneath"},
        }
    )

    assert resp["error"]["code"] == 4002
