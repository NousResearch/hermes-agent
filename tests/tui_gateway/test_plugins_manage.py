from __future__ import annotations

from hermes_cli import plugins_cmd
from hermes_cli.plugin_activation import PluginActivationState
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


def test_plugins_manage_reports_and_toggles_active_bundled_fallback(monkeypatch):
    candidates = [
        (
            "shared",
            "1.0",
            "bundled fallback",
            "bundled",
            "/plugins/bundled/shared",
            "shared",
            "backend",
        ),
        (
            "shared",
            "9.0",
            "inactive override",
            "user",
            "/plugins/user/shared",
            "shared",
            "backend",
        ),
    ]
    monkeypatch.setattr(
        plugins_cmd,
        "_discover_plugin_candidates",
        lambda **_kwargs: candidates,
    )
    enabled_keys = set()
    disabled_keys = set()
    monkeypatch.setattr(
        "hermes_cli.config.load_plugin_activation_state",
        lambda: PluginActivationState(
            enabled=frozenset(enabled_keys),
            disabled=frozenset(disabled_keys),
        ),
    )
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled_keys))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled_keys))

    def _save_enabled(value):
        enabled_keys.clear()
        enabled_keys.update(value)

    def _save_disabled(value):
        disabled_keys.clear()
        disabled_keys.update(value)

    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", _save_enabled)
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", _save_disabled)
    monkeypatch.setattr(plugins_cmd, "_toggle_plugin_toolset", lambda *a, **k: None)

    rows = _call({"action": "list"})["plugins"]

    assert len(rows) == 1
    assert rows[0]["source"] == "bundled"
    assert rows[0]["status"] == "enabled"

    result = _call(
        {
            "action": "toggle",
            "enable": False,
            "key": rows[0]["key"],
            "name": rows[0]["name"],
        }
    )

    assert disabled_keys == {"shared"}
    assert result["key"] == "shared"
    assert result["plugin"]["source"] == "user"
    assert result["plugin"]["status"] == "disabled"
