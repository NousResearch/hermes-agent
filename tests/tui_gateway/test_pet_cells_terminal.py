"""``pet.cells`` follows the CLIENT's terminal: one gateway serves many TUIs, so its own env names none."""

import pytest

from hermes_agent.tui_gateway import server

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture
def boba(tmp_path, monkeypatch):
    from hermes_agent.agent.pet import store
    from hermes_agent.agent.pet.constants import FRAME_H, FRAME_W

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    pet_dir = store.pets_dir() / "boba"
    pet_dir.mkdir(parents=True)
    PIL.new("RGBA", (FRAME_W * 8, FRAME_H * 9), (40, 60, 120, 255)).save(pet_dir / "spritesheet.webp")
    (pet_dir / "pet.json").write_text('{"id":"boba","displayName":"Boba","spritesheetPath":"spritesheet.webp"}')
    monkeypatch.setattr(server, "_pet_display_cfg", lambda: {"enabled": True, "slug": "boba", "render_mode": "auto"})
    for key in ("TMUX", "STY", "TERM_PROGRAM", "WEZTERM_PANE", "KITTY_WINDOW_ID", "ITERM_SESSION_ID"):
        monkeypatch.delenv(key, raising=False)


def _cells(params):
    response = server.handle_request({"id": "pet", "method": "pet.cells", "params": params})
    assert "error" not in response, response
    return response["result"]


def test_pet_cells_kitty_for_a_kitty_client_under_a_plain_daemon(monkeypatch, boba):
    monkeypatch.setenv("TERM", "xterm-256color")
    assert _cells({"graphics": True, "terminal": "kitty"}).get("graphics") == "kitty"


def test_pet_cells_half_blocks_for_a_plain_client_under_a_kitty_daemon(monkeypatch, boba):
    monkeypatch.setenv("TERM", "xterm-kitty")
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    result = _cells({"graphics": True, "terminal": "unicode"})
    assert result["enabled"] and "graphics" not in result and isinstance(result["frames"][0], list)
