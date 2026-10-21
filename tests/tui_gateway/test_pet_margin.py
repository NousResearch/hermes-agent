"""Right-margin validation and live profile configuration contracts."""

import pytest

from agent.pet.constants import resolve_margin_right


@pytest.mark.parametrize("value, expected", [(None, 1), (0, 0), (6, 6), (-2, 0), (20, 10),
                                             (True, 1), (False, 1), (2.5, 1), ("4", 1), ([], 1)])
def test_margin_validation(value, expected):
    assert resolve_margin_right(value) == expected


def test_margin_reads_live_config(tmp_path, monkeypatch):
    from tui_gateway import server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text("display:\n  pet:\n    margin_right: 0\n")
    assert server._pet_config_margin_right() == 0
    config.write_text("display:\n  pet:\n    margin_right: 7\n")
    assert server._pet_config_margin_right() == 7
    config.write_text("display:\n  pet: {}\n")
    assert server._pet_config_margin_right() == 1


def test_metadata_and_watcher_refresh_margin(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tui_gateway import server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_active_config_path", lambda: tmp_path / "config.yaml")
    sheet = tmp_path / "pet.png"
    sheet.write_bytes(b"sheet")
    pet = SimpleNamespace(slug="pet", display_name="Pet", spritesheet=sheet, exists=True)
    monkeypatch.setattr(server, "_active_pet", lambda: (pet, 0.33))
    monkeypatch.setattr(server, "_pet_active_selection", lambda: (True, pet, 0.33))
    config = tmp_path / "config.yaml"
    config.write_text("display:\n  pet:\n    enabled: true\n    margin_right: 0\n")
    before = server._pet_sig()
    assert server._methods["pet.info.meta"](1, {})["result"]["marginRight"] == 0
    assert server._pet_changed_payload()["marginRight"] == 0
    config.write_text("display:\n  pet:\n    enabled: true\n    margin_right: 6\n")
    assert server._pet_sig() != before
    assert server._methods["pet.info.meta"](1, {})["result"]["marginRight"] == 6
    assert server._pet_changed_payload()["marginRight"] == 6
