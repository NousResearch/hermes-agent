from __future__ import annotations

from hermes_cli import plugins_cmd
import tui_gateway.server as server


_ENTRIES = [
    ("xai", "1.0", "images", "bundled", "/plugins/image", "image_gen/xai", "backend"),
    ("xai", "1.0", "video", "bundled", "/plugins/video", "video_gen/xai", "backend"),
]


def _call(params):
    response = server._methods["plugins.manage"]("rid", params)
    assert "error" not in response, response.get("error")
    return response["result"]


def test_plugins_manage_rows_add_key_without_replacing_name(monkeypatch):
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: list(_ENTRIES))
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())

    rows = _call({"action": "list"})["plugins"]

    assert [row["name"] for row in rows] == ["xai", "xai"]
    assert {row["key"] for row in rows} == {"image_gen/xai", "video_gen/xai"}


def test_plugins_manage_toggle_targets_key_and_returns_matching_row(monkeypatch):
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: list(_ENTRIES))
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {"video_gen/xai"})
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())
    calls = []

    def _toggle(identifier, *, enabled):
        calls.append((identifier, enabled))
        return {
            "ok": True,
            "name": identifier,
            "key": "video_gen/xai",
            "unchanged": False,
        }

    monkeypatch.setattr(plugins_cmd, "dashboard_set_agent_plugin_enabled", _toggle)

    result = _call({
        "action": "toggle",
        "enable": False,
        "key": "video_gen/xai",
        "name": "xai",
    })

    assert calls == [("video_gen/xai", False)]
    assert result["name"] == "xai"
    assert result["key"] == "video_gen/xai"
    assert result["plugin"]["key"] == "video_gen/xai"
