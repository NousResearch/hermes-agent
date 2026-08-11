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
