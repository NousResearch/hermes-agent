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


def test_pet_generate_status_scoped_to_active_profile(monkeypatch, tmp_path):
    """pet.generate.status must resolve credentials under the selected profile.

    Every other pet.* RPC (pet.select, pet.info, ...) carries ``@_profile_scoped``
    so a desktop session "switched" to profile X reads/writes X's HERMES_HOME.
    pet.generate.status resolved the image-gen provider from the launch
    profile's config instead, so the generate overlay could report "available"
    (or the wrong provider list) for a profile that never configured one.
    """
    from hermes_constants import get_hermes_home

    profile_home = tmp_path / "profiles" / "beta"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    monkeypatch.setattr(server, "_profile_home", lambda p: profile_home if p == "beta" else None)

    seen = []

    def fake_resolve_provider(*, require_references=True, prefer=None):
        from agent.pet.generate.imagegen import GenerationError

        seen.append(str(get_hermes_home()))
        raise GenerationError("no provider")

    monkeypatch.setattr("agent.pet.generate.imagegen.resolve_provider", fake_resolve_provider)
    monkeypatch.setattr("agent.pet.generate.imagegen.list_sprite_providers", lambda: (seen.append(str(get_hermes_home())) or []))

    server._methods["pet.generate.status"]("r1", {"profile": "beta"})

    assert seen == [str(profile_home), str(profile_home)]


def test_pet_generate_stages_drafts_under_active_profile_home(monkeypatch, tmp_path):
    """pet.generate must stage drafts under the selected profile's cache dir.

    ``_pet_gen_root()``'s own docstring says its staging dir is profile-scoped,
    but the handler lacked ``@_profile_scoped`` so it always resolved to the
    launch profile's HERMES_HOME regardless of the caller's active profile.
    """
    import agent.pet.generate as gen

    profile_home = tmp_path / "profiles" / "beta"
    profile_home.mkdir(parents=True)
    launch_home = tmp_path / "default"
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_profile_home", lambda p: profile_home if p == "beta" else None)
    monkeypatch.setattr(gen, "generate_base_drafts", _fake_drafts_factory(tmp_path))

    resp = server._methods["pet.generate"]("r1", {"prompt": "a fox", "profile": "beta"})
    result = resp["result"]
    assert result["ok"]

    staged = profile_home / "cache" / "pet-gen" / result["token"] / "draft-0.png"
    assert staged.is_file()
    assert not (launch_home / "cache" / "pet-gen" / result["token"]).exists()


def test_pet_hatch_installs_pet_under_active_profile_home(monkeypatch, tmp_path):
    """pet.hatch must install the new pet under the selected profile's pets dir.

    Without ``@_profile_scoped``, a pet hatched while a secondary profile is
    active landed in the LAUNCH profile's ``pets/`` directory. The subsequent
    (correctly profile-scoped) ``pet.select`` then can't find the slug in the
    active profile's store — the "just created" pet is unadoptable.
    """
    import agent.pet.generate as gen

    profile_home = tmp_path / "profiles" / "beta"
    profile_home.mkdir(parents=True)
    launch_home = tmp_path / "default"
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_profile_home", lambda p: profile_home if p == "beta" else None)

    captured = {}
    monkeypatch.setattr(gen, "generate_base_drafts", _fake_drafts_factory(tmp_path))
    monkeypatch.setattr(gen, "hatch_pet", _fake_hatch_factory(captured))

    token = server._methods["pet.generate"](
        "r1", {"prompt": "a fox", "profile": "beta"}
    )["result"]["token"]

    resp = server._methods["pet.hatch"](
        "r2",
        {"token": token, "index": 0, "name": "Beta Fox", "profile": "beta"},
    )
    result = resp["result"]
    assert result["ok"]

    assert (profile_home / "pets" / "beta-fox").is_dir()
    assert not (launch_home / "pets" / "beta-fox").exists()
