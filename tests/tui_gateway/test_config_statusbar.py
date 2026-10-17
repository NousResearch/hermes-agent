"""Statusbar RPC ownership and classic CLI vocabulary parity (#120585)."""
from tui_gateway import server


def test_config_get_statusbar_survives_non_dict_display(monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": "broken"})

    resp = server.handle_request(
        {"id": "1", "method": "config.get", "params": {"key": "statusbar"}}
    )

    assert resp["result"]["value"] == "top"


def test_config_set_statusbar_survives_non_dict_display(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"display": "broken"}), encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "config.set",
            "params": {"key": "statusbar", "value": "bottom"},
        }
    )

    assert resp["result"]["value"] == "bottom"
    saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
    assert saved["display"]["statusbar"] == "bottom"


def test_config_set_statusbar_normalizes_classic_hidden_aliases_and_owns_canonical_key(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"display": {"statusbar": "hidden", "tui_statusbar": "bottom"}}))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_load_cfg", lambda: yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig")))

    for display, expected in (
        ({"statusbar": "hidden", "tui_statusbar": "bottom"}, "off"),
        ({"tui_statusbar": "bottom"}, "bottom"),
        ({"statusbar": None, "tui_statusbar": "bottom"}, "top"),
    ):
        cfg_path.write_text(yaml.safe_dump({"display": display}), encoding="utf-8")
        resp = server.handle_request(
            {"id": "read", "method": "config.get", "params": {"key": "statusbar"}}
        )
        assert resp["result"]["value"] == expected
    cfg_path.write_text(yaml.safe_dump({"display": {"statusbar": "hidden", "tui_statusbar": "bottom"}}))

    for word in ("hidden", "no", "0", "false"):
        resp = server.handle_request(
            {"id": "1", "method": "config.set", "params": {"key": "statusbar", "value": word}}
        )
        assert resp["result"]["value"] == "off"
        saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
        assert saved["display"]["statusbar"] == "off"
        assert saved["display"]["tui_statusbar"] == "bottom"

    resp = server.handle_request(
        {"id": "2", "method": "config.set", "params": {"key": "statusbar", "value": "top"}}
    )
    assert resp["result"]["value"] == "top"
    saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
    assert saved["display"]["statusbar"] == "top"

    monkeypatch.setattr(server, "_load_cfg", lambda: yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig")))
    get_resp = server.handle_request(
        {"id": "3", "method": "config.get", "params": {"key": "statusbar"}}
    )
    assert get_resp["result"]["value"] == "top"

    for word, expected in (("on", "top"), ("off", "off"), ("bottom", "bottom"), ("toggle", "off")):
        resp = server.handle_request(
            {"id": "4", "method": "config.set", "params": {"key": "statusbar", "value": word}}
        )
        assert resp["result"]["value"] == expected
        assert yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))["display"]["statusbar"] == expected
