"""The TUI receives the profile's copy-on-select setting through its real config loader."""

import pytest

from tui_gateway import server


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False), ("auto", "auto")])
def test_copy_on_select_reaches_full_config(tmp_path, monkeypatch, value, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    for name in ("_cfg_cache", "_cfg_sig", "_cfg_path"):
        monkeypatch.setattr(server, name, None)
    config = tmp_path / "config.yaml"
    config.write_text(f"display:\n  copy_on_select: {value}\n", encoding="utf-8")
    before = config.read_bytes()

    result = server.handle_request({"id": "copy-config", "method": "config.get", "params": {"key": "full"}})

    assert "error" not in result, result
    assert result["result"]["config"]["display"]["copy_on_select"] == expected
    assert config.read_bytes() == before
