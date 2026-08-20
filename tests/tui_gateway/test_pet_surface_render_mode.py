"""Cross-surface pet rendering contracts for the TUI gateway."""

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from tui_gateway import server  # noqa: E402


def test_render_mode_off_hides_terminal_pet_but_keeps_desktop_enabled(monkeypatch):
    import hermes_cli.config as cli_config
    from agent.pet import constants, store

    sheet = Image.new(
        "RGBA",
        (constants.FRAME_W * 8, constants.FRAME_H * 9),
        (80, 120, 220, 255),
    )
    pet = store.register_local_pet(
        sheet,
        slug="desktop-only",
        display_name="Desktop Only",
    )
    monkeypatch.setattr(
        cli_config,
        "load_config",
        lambda: {
            "display": {
                "pet": {
                    "enabled": True,
                    "slug": pet.slug,
                    "render_mode": "off",
                    "scale": 0.33,
                }
            }
        },
    )

    for graphics in (False, True):
        cells = server._methods["pet.cells"](
            f"cells-{graphics}",
            {"graphics": graphics, "state": "idle"},
        )["result"]
        assert cells == {"enabled": False}

    info = server._methods["pet.info"]("info", {})["result"]
    meta = server._methods["pet.info.meta"]("meta", {})["result"]
    assert info["enabled"] is True
    assert info["slug"] == pet.slug
    assert meta["enabled"] is True
    assert meta["slug"] == pet.slug
