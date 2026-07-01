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


def test_pet_generate_rejects_invalid_reference_image():
    resp = server._methods["pet.generate"](
        "r_invalid_ref",
        {"referenceImage": "data:image/svg+xml;base64,PHN2Zy8+"},
    )
    assert "error" in resp
    assert "unsupported reference image type" in resp["error"]["message"]


def test_pet_generate_rejects_oversized_reference_image(monkeypatch):
    import base64

    monkeypatch.setattr(server, "_PET_REFERENCE_MAX_BYTES", 8)
    payload = base64.b64encode(b"0123456789").decode("ascii")
    resp = server._methods["pet.generate"](
        "r_big_ref",
        {"referenceImage": f"data:image/png;base64,{payload}"},
    )
    assert "error" in resp
    assert "too large" in resp["error"]["message"].lower()


def test_pet_generate_returns_token_and_previews(monkeypatch, tmp_path):
    import agent.pet.generate as gen

    def fake_drafts(prompt, *, n=4, style="auto", reference_images=None, provider=None, on_draft=None, is_cancelled=None):
        paths = []
        for i in range(n):
            p = tmp_path / f"d{i}.png"
            _png(p)
            paths.append(p)
            if on_draft is not None:
                on_draft(i, p)
        return paths

    monkeypatch.setattr(gen, "generate_base_drafts", fake_drafts)

    resp = server._methods["pet.generate"]("r2", {"prompt": "a robot fox", "count": 4})
    result = resp["result"]
    assert result["ok"]
    assert len(result["drafts"]) == 4
    assert all(d["dataUri"].startswith("data:image/png;base64,") for d in result["drafts"])

    # Drafts are staged on disk under the returned token.
    staged = server._pet_gen_root() / result["token"] / "draft-0.png"
    assert staged.is_file()


def test_pet_cancel_unknown_token_is_noop():
    resp = server._methods["pet.cancel"]("c0", {"token": "missing"})
    assert resp["result"]["ok"] is True


def test_pet_generate_cancel_stops_run(monkeypatch, tmp_path):
    import agent.pet.generate as gen

    seen: dict = {}

    def cap_emit(event, sid, payload=None):
        # Capture the token from the up-front init event so we can cancel it.
        if event == "pet.generate.progress" and payload and payload.get("token") and not payload.get("dataUri"):
            seen["token"] = payload["token"]

    monkeypatch.setattr(server, "_emit", cap_emit)

    def fake_drafts(prompt, *, n=4, style="auto", reference_images=None, provider=None, on_draft=None, is_cancelled=None):
        # Simulate a Stop landing mid-run: the cooperative flag must read True.
        server._pet_cancel_request(seen["token"])
        assert is_cancelled() is True
        return []  # bailed before producing anything

    monkeypatch.setattr(gen, "generate_base_drafts", fake_drafts)

    resp = server._methods["pet.generate"]("rc", {"prompt": "x", "count": 4})
    assert "error" in resp
    assert "cancel" in resp["error"]["message"].lower()
    # The flag is released after the run so reusing the token isn't pre-cancelled.
    assert server._pet_is_cancelled(seen["token"]) is False


def test_pet_hatch_validates_params():
    assert "error" in server._methods["pet.hatch"]("r1", {"name": "x"})  # missing token
    assert "error" in server._methods["pet.hatch"]("r2", {"token": "abc"})  # missing name


def test_pet_hatch_expired_draft():
    resp = server._methods["pet.hatch"]("r3", {"token": "nope", "index": 0, "name": "Ghost"})
    assert "error" in resp
    assert "expired" in resp["error"]["message"]


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


def test_pet_generate_then_hatch_previews_without_activating(monkeypatch, tmp_path):
    import agent.pet.generate as gen
    from agent.pet import store

    captured = {}
    monkeypatch.setattr(gen, "generate_base_drafts", _fake_drafts_factory(tmp_path))
    monkeypatch.setattr(gen, "hatch_pet", _fake_hatch_factory(captured))

    token = server._methods["pet.generate"]("r1", {"prompt": "a fox"})["result"]["token"]

    resp = server._methods["pet.hatch"](
        "r2",
        {"token": token, "index": 1, "name": "My Fox", "description": "vulpine"},
    )
    result = resp["result"]
    assert result["ok"]
    assert result["slug"] == "my-fox"
    assert result["displayName"] == "My Fox"
    assert result["warnings"] == ["state 'jump' has no frames"]
    # Hatched from the chosen draft index.
    assert captured["base_image"].endswith("draft-1.png")

    # The pet is installed on disk and the preview payload carries the sheet,
    # but hatch must NOT activate it — adoption is a separate step.
    assert store.load_pet("my-fox") is not None
    assert result["pet"]["slug"] == "my-fox"
    assert result["pet"]["spritesheetBase64"]
    assert server._methods["pet.info"]("r3", {}).get("result", {}).get("enabled") in (False, None)


def test_pet_hatch_then_adopt_activates(monkeypatch, tmp_path):
    import agent.pet.generate as gen

    captured = {}
    monkeypatch.setattr(gen, "generate_base_drafts", _fake_drafts_factory(tmp_path))
    monkeypatch.setattr(gen, "hatch_pet", _fake_hatch_factory(captured))

    activated = {}
    monkeypatch.setattr("hermes_cli.pets._set_active", lambda slug: activated.setdefault("slug", slug))

    token = server._methods["pet.generate"]("r1", {"prompt": "a fox"})["result"]["token"]
    hatched = server._methods["pet.hatch"]("r2", {"token": token, "index": 0, "name": "My Fox"})["result"]

    # Adoption is the existing pet.select path, against the now-installed slug.
    adopt = server._methods["pet.select"]("r3", {"slug": hatched["slug"]})["result"]
    assert adopt["ok"]
    assert activated["slug"] == "my-fox"


def test_pet_sprite_payload_includes_concrete_row_counts():
    from agent.pet import constants, store

    cols, rows = 8, 9
    sheet = Image.new("RGBA", (constants.FRAME_W * cols, constants.FRAME_H * rows), (0, 0, 0, 0))
    # Current Codex rows can have more/fewer frames than Hermes' generic
    # FRAMES_PER_STATE. The desktop preview needs the concrete row count.
    real = {0: 6, 1: 8, 3: 4, 4: 5, 7: 6}
    for row, count in real.items():
        for col in range(count):
            block = Image.new("RGBA", (constants.FRAME_W, constants.FRAME_H), (80, 120, 220, 255))
            sheet.paste(block, (col * constants.FRAME_W, row * constants.FRAME_H))

    pet = store.register_local_pet(sheet, slug="row-counts", display_name="Row Counts")
    payload = server._pet_sprite_payload(pet, scale=0.7)

    assert payload["framesByRow"]["running-right"] == 8
    assert payload["framesByRow"]["waving"] == 4
    assert payload["framesByRow"]["jumping"] == 5
    assert payload["framesByState"]["run"] == 6


def test_pet_info_meta_avoids_full_payload(monkeypatch):
    import hermes_cli.config as cli_config
    from agent.pet import constants, store

    sheet = Image.new("RGBA", (constants.FRAME_W * 8, constants.FRAME_H * 9), (80, 120, 220, 255))
    pet = store.register_local_pet(sheet, slug="meta-pet", display_name="Meta Pet")
    monkeypatch.setattr(
        cli_config,
        "load_config",
        lambda: {"display": {"pet": {"enabled": True, "slug": pet.slug, "scale": 0.7}}},
    )

    resp = server._methods["pet.info.meta"]("r_meta", {})
    result = resp["result"]
    assert result["enabled"] is True
    assert result["slug"] == pet.slug
    assert result["displayName"] == "Meta Pet"
    assert result["scale"] == 0.7
    assert ":" in result["spritesheetRevision"]


def test_pet_info_omits_sheet_when_revision_unchanged(monkeypatch):
    """The ~3.2MB spritesheet is sent on the first/changed fetch but withheld on
    an unchanged poll that carries the matching ``knownRevision`` (issue #54730).
    """
    import hermes_cli.config as cli_config
    from agent.pet import constants, store

    sheet = Image.new("RGBA", (constants.FRAME_W * 8, constants.FRAME_H * 9), (80, 120, 220, 255))
    pet = store.register_local_pet(sheet, slug="revgate-pet", display_name="Rev Gate")
    monkeypatch.setattr(
        cli_config,
        "load_config",
        lambda: {"display": {"pet": {"enabled": True, "slug": pet.slug, "scale": 0.7}}},
    )

    # First fetch (no knownRevision) carries the full sheet.
    first = server._methods["pet.info"]("r_first", {})["result"]
    assert first["enabled"] is True
    assert first["spritesheetBase64"]
    assert not first.get("spritesheetOmitted")
    revision = first["spritesheetRevision"]

    # Unchanged poll echoing that revision omits the heavy bytes but keeps the
    # cheap geometry + revision so the client can confirm without re-decoding.
    same = server._methods["pet.info"]("r_same", {"knownRevision": revision})["result"]
    assert same["enabled"] is True
    assert same["spritesheetOmitted"] is True
    assert "spritesheetBase64" not in same
    assert "mime" not in same
    assert same["spritesheetRevision"] == revision
    assert same["frameW"] == constants.FRAME_W
    assert same["stateRows"] == first["stateRows"]

    # A stale/mismatched revision still gets the full sheet re-sent.
    stale = server._methods["pet.info"]("r_stale", {"knownRevision": "0:0"})["result"]
    assert stale["spritesheetBase64"] == first["spritesheetBase64"]
    assert not stale.get("spritesheetOmitted")
