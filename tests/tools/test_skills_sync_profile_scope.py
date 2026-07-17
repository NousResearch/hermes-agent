"""Regression: tools/skills_sync must resolve the skills dir per call, not at import.

``tools/skills_sync`` binds ``HERMES_HOME`` / ``SKILLS_DIR`` / ``MANIFEST_FILE`` at
import time. Long-lived runtimes import it once under the launch profile and later
bind a different profile per request via ``set_hermes_home_override`` — the dashboard
console does exactly this (``hermes_cli/web_server.py`` ``_profile_scope``, which
retargets ``skills_tool`` and ``skill_manager_tool`` but not ``skills_sync``), and
console ``skills opt-in|opt-out|reset|diff|list-modified`` dispatch in-process.

Frozen constants therefore made these functions read and WRITE the launch profile
while a different profile was active.

Same fix as f8723c478 (skills_tool) and c6a3d412d (skill_manager_tool): resolve from
the live profile-scoped home on every call, while an explicit monkeypatch of the
module attribute still wins.
"""

from __future__ import annotations

import importlib

import pytest

import hermes_constants
import tools.skills_sync as skills_sync


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Import the module under profile A, then hand back both profiles.

    The reload is the point: it reproduces "a long-lived process imported this
    module while profile A was the active home", which is what freezes the
    module-level constants.
    """
    a = tmp_path / "profileA"
    b = tmp_path / "profileB"
    a.mkdir()
    b.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(a))
    importlib.reload(skills_sync)
    try:
        yield a, b
    finally:
        monkeypatch.undo()
        importlib.reload(skills_sync)


def test_opt_out_marker_is_read_from_the_active_profile(two_profiles):
    """is_bundled_skills_opt_out() must report the ACTIVE profile, not the launch one."""
    a, b = two_profiles
    # Only B has opted out.
    (b / skills_sync.NO_BUNDLED_SKILLS_MARKER).write_text("opted out\n")
    assert not (a / skills_sync.NO_BUNDLED_SKILLS_MARKER).exists()

    token = hermes_constants.set_hermes_home_override(str(b))
    try:
        assert skills_sync.is_bundled_skills_opt_out() is True, (
            "read the launch profile's home instead of the active profile's — "
            "is_bundled_skills_opt_out() promises the active profile"
        )
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_opt_out_marker_is_written_to_the_active_profile(two_profiles):
    """set_bundled_skills_opt_out() must not write into the launch profile."""
    a, b = two_profiles

    token = hermes_constants.set_hermes_home_override(str(b))
    try:
        skills_sync.set_bundled_skills_opt_out(True)
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert (b / skills_sync.NO_BUNDLED_SKILLS_MARKER).exists(), (
        "opt-out marker did not land in the active profile"
    )
    assert not (a / skills_sync.NO_BUNDLED_SKILLS_MARKER).exists(), (
        "opt-out marker was written into the LAUNCH profile while another "
        "profile was active — cross-profile write"
    )


def test_manifest_is_read_from_the_active_profile(two_profiles):
    """_read_manifest() must consult the active profile's manifest."""
    a, b = two_profiles
    for home, name in ((a, "skill-from-A"), (b, "skill-from-B")):
        manifest = home / "skills" / ".bundled_manifest"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(f"{name}:deadbeef\n")  # v2 manifest format: name:hash

    token = hermes_constants.set_hermes_home_override(str(b))
    try:
        entries = skills_sync._read_manifest()
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert "skill-from-B" in entries, (
        f"read the launch profile's manifest while B was active: {sorted(entries)}"
    )


def test_resolvers_follow_the_active_profile(two_profiles):
    """The call-time resolvers track the live override and restore afterwards."""
    a, b = two_profiles
    assert skills_sync._skills_dir() == a / "skills"

    token = hermes_constants.set_hermes_home_override(str(b))
    try:
        assert skills_sync._hermes_home() == b
        assert skills_sync._skills_dir() == b / "skills"
        assert skills_sync._manifest_file() == b / "skills" / ".bundled_manifest"
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert skills_sync._skills_dir() == a / "skills"


def test_explicit_monkeypatch_still_wins(two_profiles, monkeypatch):
    """An external patcher of SKILLS_DIR must override live resolution (tests rely on this)."""
    a, b = two_profiles
    forced = a / "forced-skills"
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", forced)

    token = hermes_constants.set_hermes_home_override(str(b))
    try:
        assert skills_sync._skills_dir() == forced, (
            "an explicitly patched SKILLS_DIR must take precedence over the "
            "live profile home"
        )
    finally:
        hermes_constants.reset_hermes_home_override(token)
