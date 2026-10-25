"""A client's literal "default" means "the profile THIS backend runs as", never the shared home.

When a ``serve`` backend is launched under a named profile (the per-profile dashboard/desktop
topology: ``dashboard --open-profile <p>`` with ``HERMES_HOME=<root>/profiles/<p>``), a client
that asks for the literal profile name ``"default"`` must be scoped to that backend's OWN launch
home. Resolving it to the shared install home sends every such session into one store no
profile-scoped sidebar can list and every served port can read (#112692).
"""
from pathlib import Path

import pytest

import hermes_constants
from tui_gateway import server


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """A named-profile launch home under a distinct shared root (the repo's A→B rule)."""
    shared_root = tmp_path / ".hermes"
    launch_home = shared_root / "profiles" / "ember"
    launch_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # The root memo caches the pre-switch value; drop it so the HERMES_HOME redirect is honoured.
    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    yield shared_root, launch_home
    hermes_constants._default_hermes_root_memo = None


def test_default_resolves_to_launch_home_when_it_differs_from_the_shared_root(two_homes):
    """The launch profile's name resolves to the launch profile, not the shared root.

    The backend's launch home is ``<root>/profiles/ember`` while the name "default" would
    naively map to ``<root>``. ``_profile_home("default")`` must return ``None`` -- "no
    override needed: this IS the launch profile" -- so the caller keeps the launch home
    instead of switching the session into the shared store.
    """
    shared_root, launch_home = two_homes

    # Establish the two homes genuinely differ before interpreting the result below: without
    # this, ``None`` could not be told apart from "the name resolved to itself anyway".
    assert launch_home.resolve() != shared_root.resolve()
    assert hermes_constants.get_default_hermes_root().resolve() == shared_root.resolve()

    assert server._profile_home("default") is None
