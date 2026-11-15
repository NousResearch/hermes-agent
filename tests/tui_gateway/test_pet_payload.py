"""``pet.info`` sprite payload: rebuilt when pet.json changes, and callers never share the cached copy."""

import json
from types import SimpleNamespace

import pytest

from agent.pet.constants import CODEX_STATE_ROWS, FRAME_H, FRAME_W
from tui_gateway import server

Image = pytest.importorskip("PIL.Image")


def test_pet_payload_follows_pet_json_edits_and_is_isolated(tmp_path):
    rows = [*CODEX_STATE_ROWS, "crying"]
    Image.new("RGBA", (FRAME_W * 8, FRAME_H * 10), (9, 9, 9, 255)).save(tmp_path / "spritesheet.png")
    (tmp_path / "pet.json").write_text(json.dumps({"stateRows": rows}))
    pet = SimpleNamespace(slug="demo", display_name="Demo", spritesheet=tmp_path / "spritesheet.png")
    assert server._pet_sprite_payload(pet, scale=0.5)["behaviors"] == {}
    (tmp_path / "pet.json").write_text(json.dumps({"stateRows": rows, "behaviors": {"failed": {"row": "crying"}}}))
    payload = server._pet_sprite_payload(pet, scale=0.5)  # no cache_clear: pet.json is part of the revision
    assert payload["behaviors"] == {"rows": {"failed": ["crying"]}} and payload["stateRows"] == rows
    payload["behaviors"]["rows"]["failed"].append("mutated")  # callers must not reach the cached copy
    assert server._pet_sprite_payload(pet, scale=0.5)["behaviors"] == {"rows": {"failed": ["crying"]}}
