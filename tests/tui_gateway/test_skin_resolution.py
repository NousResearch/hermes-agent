"""Skin resolution across the TUI gateway config boundary."""


def test_resolve_skin_applies_auto_from_config(tmp_path, monkeypatch):
    import tui_gateway.server as server
    from hermes_cli import skin_engine

    (tmp_path / "config.yaml").write_text(
        "display:\n  skin: auto\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_cfg_mtime", None)
    monkeypatch.setattr(server, "_cfg_path", None)
    monkeypatch.setattr(server, "get_hermes_home_override", lambda: None)
    monkeypatch.setattr(skin_engine, "_resolve_auto_skin_name", lambda: "daylight")

    resolved = server.resolve_skin()

    assert resolved["name"] == "daylight"
    assert resolved["colors"]["banner_title"] == "#0F172A"
    assert resolved["tool_prefix"] == "│"
