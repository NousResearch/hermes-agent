"""The desktop pet-generation RPCs must bind the profile secret scope (#130450).

``pet.generate.status`` / ``pet.generate`` / ``pet.hatch`` were registered
``scoped=False``, so once multi-profile hosting activates at boot every ``get_secret``
inside them raised ``UnscopedSecretError``. ``_pet_method``'s fail-open envelope turned
the status probe into ``{"available": False, "providers": []}`` — the desktop picker hid
itself and generation silently fell through to Nous Portal regardless of the configured
backend. The other eleven pet RPCs bind ``@_profile_scoped`` via ``_pet_method``'s
default; these three must too. The generation pool additionally carries the caller's
context into its workers, because provider clients resolve credentials per call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tui_gateway.server as server

PROBE_KEY = "HERMES_PET_PROBE_KEY"


@pytest.fixture
def hermes_root(tmp_path, monkeypatch):
    """A temp HERMES_HOME root with a named 'work' profile beside the launch home."""
    root = tmp_path / "hermes_home"
    (root / "profiles" / "work").mkdir(parents=True)
    (root / ".env").write_text(f"{PROBE_KEY}=launch-secret\n", encoding="utf-8")
    (root / "profiles" / "work" / ".env").write_text(f"{PROBE_KEY}=work-secret\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_constants import get_hermes_home_override

    assert get_hermes_home_override() is None
    return root


def _call(method, params=None):
    return server._methods[method](1, params or {})


@pytest.fixture
def multiplex_on():
    """Flip multiplexing on (what the desktop backend boots into with >=2 profile homes)."""
    from agent.secret_scope import is_multiplex_active, set_multiplex_active

    previous = is_multiplex_active()
    set_multiplex_active(True)
    yield
    set_multiplex_active(previous)


def _probe_list_sprite_providers():
    from agent.secret_scope import get_secret

    return [{"name": get_secret(PROBE_KEY), "label": "probe", "default": True}]


def test_status_probe_reads_the_launch_profile_secret(hermes_root, monkeypatch, multiplex_on):
    import agent.pet.generate.imagegen as imagegen

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **kwargs: None)
    monkeypatch.setattr(imagegen, "list_sprite_providers", _probe_list_sprite_providers)

    result = _call("pet.generate.status")["result"]
    # Before the fix the probe's UnscopedSecretError was swallowed (fail-open envelope /
    # best-effort picker catch) and the desktop saw providers == [].
    assert result["providers"] == [{"name": "launch-secret", "label": "probe", "default": True}]


def test_status_probe_reads_a_named_profiles_secret(hermes_root, monkeypatch, multiplex_on):
    import agent.pet.generate.imagegen as imagegen

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **kwargs: None)
    monkeypatch.setattr(imagegen, "list_sprite_providers", _probe_list_sprite_providers)

    result = _call("pet.generate.status", {"profile": "work"})["result"]
    assert result["providers"] == [{"name": "work-secret", "label": "probe", "default": True}]


def test_generate_resolves_credentials_inside_the_profile_scope(hermes_root, monkeypatch, multiplex_on):
    import agent.pet.generate as pet_generate

    seen = {}

    def fake_drafts(*args, **kwargs):
        from agent.secret_scope import get_secret

        seen["value"] = get_secret(PROBE_KEY)
        return []

    monkeypatch.setattr(pet_generate, "generate_base_drafts", fake_drafts)

    resp = _call("pet.generate", {"prompt": "a red cube"})
    assert "error" in resp  # no drafts produced — expected, the probe returns none
    assert seen["value"] == "launch-secret"


def test_hatch_resolves_credentials_inside_the_profile_scope(hermes_root, monkeypatch, multiplex_on):
    import agent.pet.generate as pet_generate

    draft_dir = server._pet_gen_root() / "probetok"
    draft_dir.mkdir(parents=True, exist_ok=True)
    (draft_dir / "draft-0.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    seen = {}

    def fake_hatch(*args, **kwargs):
        from agent.secret_scope import get_secret

        seen["value"] = get_secret(PROBE_KEY)
        return SimpleNamespace(slug="probe-pet", display_name="Probe", validation={"warnings": []})

    monkeypatch.setattr(pet_generate, "hatch_pet", fake_hatch)
    monkeypatch.setattr("agent.pet.store.unique_slug", lambda name: "probe-pet")
    monkeypatch.setattr("agent.pet.store.load_pet", lambda slug: None)

    resp = _call("pet.hatch", {"token": "probetok", "name": "Probe"})
    assert "error" not in resp, resp.get("error")
    assert seen["value"] == "launch-secret"


def test_generation_pool_carries_the_secret_scope_into_workers(hermes_root):
    from agent.pet.generate.orchestrate import _run_parallel
    from agent.secret_scope import (
        build_profile_secret_scope, get_secret, reset_secret_scope, set_secret_scope,
    )

    def read_probe(_item):
        return get_secret(PROBE_KEY)

    from agent.secret_scope import is_multiplex_active, set_multiplex_active

    previous = is_multiplex_active()
    token = set_secret_scope(build_profile_secret_scope(hermes_root))
    set_multiplex_active(True)
    try:
        got = list(_run_parallel(read_probe, range(2), cancelled=lambda: False, on_cancel_log=""))
    finally:
        set_multiplex_active(previous)
        reset_secret_scope(token)

    assert got == ["launch-secret", "launch-secret"]
