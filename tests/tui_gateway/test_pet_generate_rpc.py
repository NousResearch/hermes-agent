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


def test_pet_generate_returns_token_and_previews(monkeypatch, tmp_path):
    import agent.pet.generate as gen

    def fake_drafts(prompt, *, n=4, style="auto", on_draft=None, is_cancelled=None):
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

    def fake_drafts(prompt, *, n=4, style="auto", on_draft=None, is_cancelled=None):
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
    def fake_drafts(prompt, *, n=4, style="auto", on_draft=None, is_cancelled=None):
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
