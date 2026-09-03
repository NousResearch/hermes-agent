from __future__ import annotations

import yaml

import tui_gateway.server as server


def test_skin_options_include_every_discovered_skin_and_active_branding(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "skins").mkdir()
    (tmp_path / "skins" / "custom.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "custom",
                "description": "Custom picker fixture",
                "branding": {"agent_name": "Custom Agent"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text("display:\n  skin: custom\n", encoding="utf-8")

    response = server._methods["skin.options"]("skin-options", {})

    result = response["result"]
    names = {skin["name"] for skin in result["skins"]}
    assert {"default", "custom"}.issubset(names)
    assert result["active"] == "custom"
    assert result["active_skin"]["name"] == "custom"
    assert result["active_skin"]["branding"]["agent_name"] == "Custom Agent"


def test_skin_preview_resolves_branding_without_persisting_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("display:\n  skin: default\n", encoding="utf-8")

    response = server._methods["skin.preview"]("skin-preview", {"name": "charizard"})

    assert response["result"]["name"] == "charizard"
    assert response["result"]["branding"]["agent_name"] == "Charizard Agent"
    assert "skin: default" in (tmp_path / "config.yaml").read_text(encoding="utf-8")


def test_skin_preview_rejects_unknown_names(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    response = server._methods["skin.preview"]("skin-preview", {"name": "missing"})

    assert response["error"]["code"] == 4002


def test_skin_preview_rejects_traversal_names_from_user_skin_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    skins_dir = tmp_path / "skins"
    skins_dir.mkdir()
    (skins_dir / "decoy.yaml").write_text(
        yaml.safe_dump({"name": "../outside", "description": "unsafe metadata"}),
        encoding="utf-8",
    )
    (tmp_path / "outside.yaml").write_text(
        yaml.safe_dump({"name": "outside", "branding": {"agent_name": "Leaked Agent"}}),
        encoding="utf-8",
    )

    response = server._methods["skin.preview"]("skin-preview", {"name": "../outside"})

    assert response["error"]["code"] == 4002
    assert "result" not in response


def test_config_set_rejects_skin_names_outside_discovered_options(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("display:\n  skin: default\n", encoding="utf-8")

    response = server._methods["config.set"](
        "skin-set",
        {"key": "skin", "value": "../outside"},
    )

    assert response["error"]["code"] == 4002
    assert "skin: default" in (tmp_path / "config.yaml").read_text(encoding="utf-8")


def test_skin_options_rejects_an_empty_active_skin_payload(monkeypatch):
    handler = server._methods["skin.options"]
    monkeypatch.setitem(handler.__globals__, "resolve_skin", lambda _name=None: {})

    response = handler("skin-options", {})

    assert response["error"]["code"] == 5020
    assert "could not resolve active skin" in response["error"]["message"]
