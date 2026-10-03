"""``display.pet.terminal_enabled`` gates ONLY the terminal-rendered pet.

The desktop app renders the pet from ``pet.info`` (native-resolution spritesheet);
the TUI + legacy CLI pane render half-block pixels via ``pet.cells``. Hiding the
pixelated terminal pet must keep the desktop mascot alive, so the gate lives on
the terminal path only:

* ``pet.cells`` answers ``enabled: false`` when the key is false (or any non-truthy
  value), and behaves as before when the key is absent (defaults to on).
* ``pet.info.meta`` reports the gate as ``terminalEnabled`` so the running TUI's
  steady poll hides the pet live (no restart); absent → ``True``.
* ``pet.info`` is UNTOUCHED by the gate — the desktop keeps its mascot.
"""

from __future__ import annotations

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session  # noqa: F401  (registers the RPC methods)


def _call(method: str, params: dict) -> dict:
    return srv._methods[method](1, params)


@pytest.fixture
def pet_home(tmp_path, monkeypatch):
    """Synthetic boba pet + config in a temp HERMES_HOME (mirrors test_cli_pet_pane)."""
    from PIL import Image

    from agent.pet import store
    from agent.pet.constants import FRAME_H, FRAME_W
    from hermes_cli.config import load_config, save_config

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    cols, rows = 8, 9
    sheet = Image.new("RGBA", (FRAME_W * cols, FRAME_H * rows), (0, 0, 0, 0))
    for r in range(rows):
        block = Image.new("RGBA", (FRAME_W, FRAME_H), (20 + r * 25, 60, 120, 255))
        for c in range(cols):
            sheet.paste(block, (c * FRAME_W, r * FRAME_H))

    pet_dir = store.pets_dir() / "boba"
    pet_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(pet_dir / "spritesheet.webp")
    (pet_dir / "pet.json").write_text(
        '{"id":"boba","displayName":"Boba","description":"d","spritesheetPath":"spritesheet.webp"}'
    )

    cfg = load_config()
    cfg.setdefault("display", {}).setdefault("pet", {}).update({"enabled": True, "slug": "boba"})
    save_config(cfg)
    return cfg


def test_cells_default_terminal_on(pet_home):
    """No terminal_enabled key (legacy config) → terminal pet behaves exactly as before."""
    result = _call("pet.cells", {"state": "idle"})["result"]
    assert result["enabled"] is True
    assert result["slug"] == "boba"


def test_cells_gated_off_by_terminal_enabled(pet_home):
    cfg = pet_home
    cfg["display"]["pet"]["terminal_enabled"] = False
    from hermes_cli.config import save_config

    save_config(cfg)

    result = _call("pet.cells", {"state": "idle"})["result"]
    assert result["enabled"] is False


def test_cells_honors_quoted_string_values(pet_home):
    """Quoted yaml `\"false\"` must read as off (bool('false') is True — the classic trap)."""
    cfg = pet_home
    cfg["display"]["pet"]["terminal_enabled"] = "false"
    from hermes_cli.config import save_config

    save_config(cfg)

    assert _call("pet.cells", {"state": "idle"})["result"]["enabled"] is False
    cfg["display"]["pet"]["terminal_enabled"] = "true"
    save_config(cfg)
    assert _call("pet.cells", {"state": "idle"})["result"]["enabled"] is True


def test_info_meta_reports_terminal_enabled(pet_home):
    """The running TUI's steady poll reads this to hide live, without a restart."""
    cfg = pet_home
    from hermes_cli.config import save_config

    # Absent → True (default on, backward compatible).
    assert _call("pet.info.meta", {})["result"]["terminalEnabled"] is True

    cfg["display"]["pet"]["terminal_enabled"] = False
    save_config(cfg)
    assert _call("pet.info.meta", {})["result"]["terminalEnabled"] is False


def test_pet_info_unaffected_by_gate(pet_home):
    """Desktop path: pet.info keeps serving the spritesheet no matter the terminal gate."""
    cfg = pet_home
    cfg["display"]["pet"]["terminal_enabled"] = False
    from hermes_cli.config import save_config

    save_config(cfg)

    result = _call("pet.info", {})["result"]
    assert result["enabled"] is True
    assert result.get("spritesheetBase64")
    assert _call("pet.info.meta", {})["result"]["enabled"] is True
