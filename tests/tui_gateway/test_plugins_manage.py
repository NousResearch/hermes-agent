from __future__ import annotations

import pytest

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


def _shared_candidate(version, description, source):
    return ("shared", version, description, source, f"/plugins/{source}/shared", "shared", "backend")


def _replace_set(target, values):
    target.clear()
    target.update(values)


def _patch_shared_state(monkeypatch, candidates):
    enabled_keys, disabled_keys = set(), set()
    monkeypatch.setattr(plugins_cmd, "_discover_plugin_candidates", lambda **_: candidates)
    monkeypatch.setattr(
        "hermes_cli.config.load_plugin_activation_state",
        lambda: PluginActivationState(
            enabled=frozenset(enabled_keys), disabled=frozenset(disabled_keys)
        ),
    )
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", enabled_keys.copy)
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", disabled_keys.copy)
    monkeypatch.setattr(plugins_cmd, "_save_enabled_set", lambda values: _replace_set(enabled_keys, values))
    monkeypatch.setattr(plugins_cmd, "_save_disabled_set", lambda values: _replace_set(disabled_keys, values))
    monkeypatch.setattr(plugins_cmd, "_toggle_plugin_toolset", lambda *args, **kwargs: None)
    return disabled_keys


@pytest.mark.parametrize(
    ("toggle", "enabled"),
    ((False, frozenset()), (True, frozenset({"video_gen/xai"}))),
    ids=("list", "toggle"),
)
def test_plugins_manage_rows_and_toggle_use_canonical_key(monkeypatch, toggle, enabled):
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: list(_ENTRIES))
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set())

    rows = _call({"action": "list"})["plugins"]

    assert [row["name"] for row in rows] == ["xai", "xai"]
    assert {row["key"] for row in rows} == {"image_gen/xai", "video_gen/xai"}
    if not toggle:
        return
    calls = []

    def _toggle(identifier, *, enabled):
        calls.append((identifier, enabled))
        return {"ok": True, "name": identifier, "key": "video_gen/xai", "unchanged": False}

    monkeypatch.setattr(plugins_cmd, "dashboard_set_agent_plugin_enabled", _toggle)

    result = _call({"action": "toggle", "enable": False, "key": "video_gen/xai", "name": "xai"})

    assert calls == [("video_gen/xai", False)]
    assert result["name"] == "xai"
    assert result["key"] == result["plugin"]["key"] == "video_gen/xai"


def test_plugins_manage_reports_and_toggles_active_bundled_fallback(monkeypatch):
    candidates = [
        _shared_candidate("1.0", "bundled fallback", "bundled"),
        _shared_candidate("9.0", "inactive override", "user"),
    ]
    disabled_keys = _patch_shared_state(monkeypatch, candidates)

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
    assert (result["plugin"]["source"], result["plugin"]["status"]) == ("user", "disabled")
