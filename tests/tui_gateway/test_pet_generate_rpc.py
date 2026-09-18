"""Gateway RPC tests for pet generation (pet.generate / pet.hatch).

Image generation is mocked, so these assert the RPC contract + staging behavior
(draft tokens, data-URI previews, expiry, activation) without any API calls.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from tui_gateway import server  # noqa: E402


def _png(path):
    Image.new("RGBA", (64, 64), (200, 80, 80, 255)).save(path)


def test_pet_generate_requires_prompt():
    resp = server._methods["pet.generate"]("r1", {"prompt": "  "})
    assert "error" in resp


def _fake_drafts_factory(tmp_path):
    def fake_drafts(prompt, *, n=4, style="auto", reference_images=None, provider=None, on_draft=None, is_cancelled=None):
        paths = []
        for i in range(n):
            p = tmp_path / f"d{i}.png"
            _png(p)
            paths.append(p)
            if on_draft is not None:
                on_draft(i, p)
        return paths

    return fake_drafts


def _fake_hatch_factory(captured):
    """A hatch that registers a real local pet (so the preview payload populates)."""
    import agent.pet.generate as gen
    from agent.pet import store

    def fake_hatch(*, base_image, slug, display_name="", description="", concept="", style="auto", on_progress=None, provider=None, is_cancelled=None):
        captured["base_image"] = str(base_image)
        captured["slug"] = slug
        pet = store.register_local_pet(
            Image.new("RGBA", (192, 208), (10, 20, 30, 255)),
            slug=slug,
            display_name=display_name,
            description=description,
        )
        return gen.HatchResult(
            slug=pet.slug,
            display_name=display_name or pet.display_name,
            spritesheet=pet.spritesheet,
            states=["idle", "wave"],
            validation={"ok": True, "warnings": ["state 'jump' has no frames"]},
        )

    return fake_hatch


def test_hatch_progress_crosses_the_typed_event_boundary(tmp_path, monkeypatch):
    """The hatch callback hands the typed emit a ``PetHatchProgressPayload`` for both shapes it produces.

    Drives the real ``pet.hatch`` handler through the real ``_emit``/``_event_frame``. ``_pet_emit``
    swallows a payload-type slip at debug level, so a dict here drops every hatch progress row with no
    error and no notification — the regression fails on the pre-typed callback (dict rejected by the
    frame builder) and passes once the callback builds the registered model.
    """
    import agent.pet.generate as gen
    from tui_gateway.contracts.events import PetHatchProgressPayload

    frames = []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame))
    monkeypatch.setattr(server, "_pet_pick_provider", lambda params, *, require_references: object())

    draft = server._pet_gen_root() / "tok" / "draft-0.png"
    draft.parent.mkdir(parents=True, exist_ok=True)
    _png(draft)

    def fake_hatch(*, base_image, slug, display_name="", description="", concept="", style="auto",
                   on_progress=None, provider=None, is_cancelled=None):
        on_progress("stage", "drawing the sheet")
        on_progress("row", "idle:1:3")
        from agent.pet import store

        pet = store.register_local_pet(Image.new("RGBA", (192, 208), (10, 20, 30, 255)),
                                       slug=slug, display_name=display_name)
        return gen.HatchResult(slug=pet.slug, display_name=display_name or pet.display_name,
                               spritesheet=pet.spritesheet, states=["idle"],
                               validation={"ok": True, "warnings": []})

    monkeypatch.setattr(gen, "hatch_pet", fake_hatch)

    resp = server._methods["pet.hatch"]("r1", {"token": "tok", "name": "Buddy", "index": 0})

    assert resp.get("error") is None, resp
    progress = [f for f in frames if f.get("params", {}).get("type") == "pet.hatch.progress"]
    assert len(progress) == 2, frames
    detail_row = PetHatchProgressPayload.model_validate(progress[0]["params"]["payload"])
    assert (detail_row.event, detail_row.detail) == ("stage", "drawing the sheet")
    grid_row = PetHatchProgressPayload.model_validate(progress[1]["params"]["payload"])
    assert (grid_row.event, grid_row.state, grid_row.done, grid_row.total) == ("row", "idle", "1", "3")
