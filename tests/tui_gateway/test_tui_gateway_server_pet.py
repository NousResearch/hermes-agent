"""pet.cells honors ``display.pet.render_mode: off`` (#127805)."""

import pytest

from tui_gateway import server


@pytest.mark.parametrize(("render_mode", "expected"), [
    ("off", {"enabled": False}),
    ("OFF", {"enabled": False}),
    ("auto", {"enabled": True, "slug": "codex", "displayName": "Codex", "state": "idle", "cols": 2,
              "frameMs": 1100.0, "frames": [[[[1, 2, 3, 255, 4, 5, 6, 255]]]], "scale": 1.0}),
])
def test_pet_cells_honors_render_mode_off(monkeypatch, render_mode, expected):
    """display.pet.render_mode: off disables the terminal pet (#127805).

    The TUI draws from pet.cells while the desktop draws from pet.info, so
    gating pet.cells turns the terminal pet off and keeps the desktop pet.
    """
    from agent.pet import constants, render, store

    class _FakePet:
        slug = "codex"
        display_name = "Codex"
        exists = True
        spritesheet = "codex.png"

    class _FakeRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def frame_count(self, state):
            return 1

        def cells(self, state, index, *, cols=None):
            return [[((1, 2, 3, 255), (4, 5, 6, 255))]]

    pet_cfg = {"enabled": True, "slug": "codex", "scale": 1.0, "render_mode": render_mode}
    monkeypatch.setattr(server, "_pet_display_cfg", lambda: dict(pet_cfg))
    monkeypatch.setattr(server, "_pet_cfg", lambda: dict(pet_cfg))
    monkeypatch.setattr(server, "_pet_sheet_revision", lambda spritesheet: "1:1")
    monkeypatch.setattr(store, "resolve_active_pet", lambda slug: _FakePet())
    monkeypatch.setattr(render, "PetRenderer", _FakeRenderer)
    monkeypatch.setattr(constants, "LOOP_MS", 1100.0)

    resp = server.handle_request(
        {"id": "1", "method": "pet.cells", "params": {"state": "idle", "cols": 2}}
    )
    assert resp["result"] == expected

    info = server.handle_request({"id": "2", "method": "pet.info.meta", "params": {}})
    assert info["result"]["enabled"] is True
