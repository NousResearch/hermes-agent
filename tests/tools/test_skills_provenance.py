"""Unit tests for tools/skills_provenance.py — registry read/write + classify."""

from pathlib import Path

import pytest

from tools.skills_provenance import (
    PROVENANCE_BUILTIN,
    PROVENANCE_HUB,
    PROVENANCE_LOCAL,
    PROVENANCE_LOCAL_EDIT,
    VALID_PROVENANCE,
    _dirs_match,
    _read_provenance_file,
    _write_provenance_file,
    classify,
    classify_with_hub_lock,
    record,
    record_many,
    trust_for,
)


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------


def test_round_trip_basic(tmp_path):
    path = tmp_path / ".provenance"
    entries = {
        "alpha": {"provenance": PROVENANCE_BUILTIN, "origin_path": "/path/a", "synced_at": "2026-06-23T18:00:00+00:00"},
        "beta": {"provenance": PROVENANCE_HUB, "origin_path": "/path/b", "synced_at": "2026-06-23T18:00:01+00:00"},
    }
    _write_provenance_file(entries, path)
    read_back = _read_provenance_file(path)
    assert read_back == entries


def test_write_creates_parent_dir(tmp_path):
    path = tmp_path / "deep" / "nested" / ".provenance"
    _write_provenance_file({"x": {"provenance": PROVENANCE_LOCAL, "origin_path": "", "synced_at": ""}}, path)
    assert path.exists()
    assert _read_provenance_file(path) == {"x": {"provenance": PROVENANCE_LOCAL, "origin_path": "", "synced_at": ""}}


def test_read_missing_file_returns_empty(tmp_path):
    assert _read_provenance_file(tmp_path / "missing") == {}


def test_read_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / ".provenance"
    path.write_text(
        "# header\n"
        "\n"
        "alpha|builtin|/p|2026-06-23T18:00:00+00:00\n"
        "  \n"
        "beta|hub|/q|2026-06-23T18:00:01+00:00\n"
    )
    result = _read_provenance_file(path)
    assert set(result) == {"alpha", "beta"}


def test_read_skips_malformed_lines(tmp_path):
    path = tmp_path / ".provenance"
    path.write_text(
        "alpha|builtin|/p|2026-06-23T18:00:00+00:00\n"
        "garbage_line\n"
        "beta|hub|/q|2026-06-23T18:00:01+00:00\n"
    )
    result = _read_provenance_file(path)
    assert set(result) == {"alpha", "beta"}


def test_read_skips_unknown_provenance_values(tmp_path):
    path = tmp_path / ".provenance"
    path.write_text(
        "alpha|builtin|/p|2026-06-23T18:00:00+00:00\n"
        "beta|mystery|/q|2026-06-23T18:00:01+00:00\n"
    )
    result = _read_provenance_file(path)
    assert "alpha" in result
    assert "beta" not in result


def test_record_writes_one_entry(tmp_path):
    path = tmp_path / ".provenance"
    record("alpha", PROVENANCE_BUILTIN, "/some/origin", path=path)
    result = _read_provenance_file(path)
    assert "alpha" in result
    assert result["alpha"]["provenance"] == PROVENANCE_BUILTIN
    assert result["alpha"]["origin_path"] == "/some/origin"
    assert result["alpha"]["synced_at"]


def test_record_preserves_existing_entries(tmp_path):
    path = tmp_path / ".provenance"
    record("alpha", PROVENANCE_BUILTIN, "/a", path=path)
    record("beta", PROVENANCE_HUB, "/b", path=path)
    record("alpha", PROVENANCE_LOCAL_EDIT, "/a2", path=path)
    result = _read_provenance_file(path)
    assert result["alpha"]["provenance"] == PROVENANCE_LOCAL_EDIT
    assert result["alpha"]["origin_path"] == "/a2"
    assert result["beta"]["provenance"] == PROVENANCE_HUB


def test_record_rejects_invalid_provenance(tmp_path):
    with pytest.raises(ValueError):
        record("alpha", "mystery", "/a", path=tmp_path / ".provenance")


def test_record_many_bulk_update(tmp_path):
    path = tmp_path / ".provenance"
    record_many([
        ("alpha", PROVENANCE_BUILTIN, "/a"),
        ("beta", PROVENANCE_HUB, "/b"),
        ("gamma", PROVENANCE_LOCAL_EDIT, "/g"),
        ("delta", "mystery", "/d"),
    ], path=path)
    result = _read_provenance_file(path)
    assert set(result) == {"alpha", "beta", "gamma"}


def test_write_is_sorted_by_name(tmp_path):
    path = tmp_path / ".provenance"
    _write_provenance_file({
        "zebra": {"provenance": PROVENANCE_LOCAL, "origin_path": "", "synced_at": ""},
        "alpha": {"provenance": PROVENANCE_LOCAL, "origin_path": "", "synced_at": ""},
        "middle": {"provenance": PROVENANCE_LOCAL, "origin_path": "", "synced_at": ""},
    }, path=path)
    lines = path.read_text().strip().splitlines()
    names = [line.split("|")[0] for line in lines]
    assert names == ["alpha", "middle", "zebra"]


# ---------------------------------------------------------------------------
# Trust derivation
# ---------------------------------------------------------------------------


def test_trust_for_builtin():
    assert trust_for(PROVENANCE_BUILTIN) == "builtin"


def test_trust_for_hub():
    assert trust_for(PROVENANCE_HUB) == "community"


def test_trust_for_local_edit():
    assert trust_for(PROVENANCE_LOCAL_EDIT) == "local"


def test_trust_for_local():
    assert trust_for(PROVENANCE_LOCAL) == "local"


def test_valid_provenance_constant():
    assert VALID_PROVENANCE == frozenset({
        PROVENANCE_BUILTIN, PROVENANCE_HUB, PROVENANCE_LOCAL_EDIT, PROVENANCE_LOCAL,
    })


# ---------------------------------------------------------------------------
# Path-based classify()
# ---------------------------------------------------------------------------


def test_classify_under_platform_skills_dir(tmp_path, monkeypatch):
    platform = tmp_path / "platform" / "skills" / "apple" / "macos-computer-use"
    platform.mkdir(parents=True)
    (platform / "SKILL.md").write_text("name: macos-computer-use\nbody\n")

    monkeypatch.setattr("tools.skills_provenance._platform_skills_dir", lambda: tmp_path / "platform" / "skills")
    monkeypatch.setattr("tools.skills_provenance.PROFILE_SKILLS_DIR", tmp_path / "profile" / "skills")

    provenance, origin = classify("macos-computer-use", platform)
    assert provenance == PROVENANCE_BUILTIN
    assert origin == str(platform)


def test_classify_profile_copy_matching_platform_is_builtin(tmp_path, monkeypatch):
    platform_dir = tmp_path / "platform" / "skills" / "apple" / "macos-computer-use"
    platform_dir.mkdir(parents=True)
    (platform_dir / "SKILL.md").write_text("name: macos-computer-use\nbody\n")

    profile_dir = tmp_path / "profile" / "skills" / "apple" / "macos-computer-use"
    profile_dir.mkdir(parents=True)
    (profile_dir / "SKILL.md").write_text("name: macos-computer-use\nbody\n")

    monkeypatch.setattr("tools.skills_provenance._platform_skills_dir", lambda: tmp_path / "platform" / "skills")
    monkeypatch.setattr("tools.skills_provenance.PROFILE_SKILLS_DIR", tmp_path / "profile" / "skills")

    provenance, origin = classify("macos-computer-use", profile_dir)
    assert provenance == PROVENANCE_BUILTIN
    assert origin == str(platform_dir)


def test_classify_unknown_local_skill(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profile" / "skills" / "lone-local"
    profile_dir.mkdir(parents=True)
    (profile_dir / "SKILL.md").write_text("name: lone-local\nbody\n")

    platform = tmp_path / "platform" / "skills"
    platform.mkdir(parents=True)
    bundled = tmp_path / "bundled"
    bundled.mkdir()

    monkeypatch.setattr("tools.skills_provenance._platform_skills_dir", lambda: platform)
    monkeypatch.setattr("tools.skills_provenance.PROFILE_SKILLS_DIR", tmp_path / "profile" / "skills")
    monkeypatch.setattr("tools.skills_sync._get_bundled_dir", lambda: bundled)

    provenance, origin = classify("lone-local", profile_dir)
    assert provenance == PROVENANCE_LOCAL
    assert origin == ""


def test_classify_modified_builtin_is_local_edit(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profile" / "skills" / "edited-builtin"
    profile_dir.mkdir(parents=True)
    (profile_dir / "SKILL.md").write_text("name: edited-builtin\nUSER EDIT\n")

    bundled = tmp_path / "bundled" / "edited-builtin"
    bundled.mkdir(parents=True)
    (bundled / "SKILL.md").write_text("name: edited-builtin\nORIGINAL\n")

    platform = tmp_path / "platform" / "skills"
    platform.mkdir(parents=True)

    monkeypatch.setattr("tools.skills_provenance._platform_skills_dir", lambda: platform)
    monkeypatch.setattr("tools.skills_provenance.PROFILE_SKILLS_DIR", tmp_path / "profile" / "skills")
    monkeypatch.setattr("tools.skills_sync._get_bundled_dir", lambda: bundled)

    provenance, origin = classify("edited-builtin", profile_dir)
    assert provenance == PROVENANCE_LOCAL_EDIT
    assert origin == str(bundled)


# ---------------------------------------------------------------------------
# classify_with_hub_lock
# ---------------------------------------------------------------------------


def test_classify_with_hub_lock_hub_wins(tmp_path):
    profile_dir = tmp_path / "profile" / "skills" / "hub-skill"
    profile_dir.mkdir(parents=True)
    (profile_dir / "SKILL.md").write_text("body\n")

    hub_entry = {"install_path": "hub-skill", "source": "github"}
    provenance, origin = classify_with_hub_lock("hub-skill", profile_dir, hub_entry)
    assert provenance == PROVENANCE_HUB
    assert "hub-skill" in origin


def test_classify_with_hub_lock_no_hub_falls_through(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profile" / "skills" / "macos-computer-use"
    profile_dir.mkdir(parents=True)
    (profile_dir / "SKILL.md").write_text("name: macos-computer-use\nbody\n")

    platform = tmp_path / "platform" / "skills" / "apple" / "macos-computer-use"
    platform.mkdir(parents=True)
    (platform / "SKILL.md").write_text("name: macos-computer-use\nbody\n")

    monkeypatch.setattr("tools.skills_provenance._platform_skills_dir", lambda: tmp_path / "platform" / "skills")
    monkeypatch.setattr("tools.skills_provenance.PROFILE_SKILLS_DIR", tmp_path / "profile" / "skills")

    provenance, origin = classify_with_hub_lock("macos-computer-use", profile_dir, None)
    assert provenance == PROVENANCE_BUILTIN


# ---------------------------------------------------------------------------
# _dirs_match
# ---------------------------------------------------------------------------


def test_dirs_match_identical_skill_md(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "SKILL.md").write_text("hello\n")
    (b / "SKILL.md").write_text("hello\n")
    assert _dirs_match(a, b)


def test_dirs_match_different_skill_md(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "SKILL.md").write_text("hello\n")
    (b / "SKILL.md").write_text("world\n")
    assert not _dirs_match(a, b)


def test_dirs_match_only_one_has_skill_md(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "SKILL.md").write_text("hello\n")
    assert not _dirs_match(a, b)


# ---------------------------------------------------------------------------
# Regression: builtin skill symlinked into per-profile dir must classify as builtin
# (regression test for t_ddf08a93 — the macos-computer-use scenario)
# ---------------------------------------------------------------------------


def test_builtin_symlinked_into_profile_dir_classifies_as_builtin(tmp_path, monkeypatch):
    """A builtin skill that has been symlinked (not copied) into a per-profile
    skills dir must still classify as PROVENANCE_BUILTIN with trust='builtin',
    not PROVENANCE_LOCAL. This is the macos-computer-use scenario: the platform
    skills dir has the canonical skill, a profile dir has a symlink to it.
    """
    # Platform builtin skill (real file)
    builtin_dir = tmp_path / "platform" / "skills" / "apple" / "macos-computer-use"
    builtin_dir.mkdir(parents=True)
    (builtin_dir / "SKILL.md").write_text("name: macos-computer-use\nbody\n")

    # Profile skills dir with a SYMLINK to the builtin (not a copy)
    profile_link = tmp_path / "profile" / "skills" / "apple" / "macos-computer-use"
    profile_link.parent.mkdir(parents=True)
    try:
        profile_link.symlink_to(builtin_dir)
    except OSError as exc:
        pytest.skip(f"symlinks not supported in this environment: {exc}")

    # Monkeypatch so the function sees our tmp dirs as the canonical
    # platform + profile skills roots. Without this the test would leak
    # into the real ~/.hermes/skills tree.
    monkeypatch.setattr(
        "tools.skills_provenance._platform_skills_dir",
        lambda: tmp_path / "platform" / "skills",
    )
    monkeypatch.setattr(
        "tools.skills_provenance.PROFILE_SKILLS_DIR",
        tmp_path / "profile" / "skills",
    )
    # Isolate bundled dir too — the real bundled dir may contain a SKILL.md
    # whose name happens to match and skew the Case 3 local-edit branch.
    monkeypatch.setattr(
        "tools.skills_sync._get_bundled_dir",
        lambda: None,
    )

    # classify() should resolve through the symlink and match the builtin content
    provenance, origin = classify("macos-computer-use", profile_link)
    assert provenance == PROVENANCE_BUILTIN, (
        f"expected PROVENANCE_BUILTIN for symlinked builtin, got {provenance}"
    )
    assert trust_for(provenance) == "builtin", (
        f"expected trust='builtin' for symlinked builtin, got {trust_for(provenance)}"
    )
    # Origin should be a path under the platform skills dir (the contract is
    # the skill is recognised as a builtin copy, not a local). The exact
    # form (symlink path vs resolved real dir) is implementation-detail.
    origin_path = Path(origin)
    platform_root = (tmp_path / "platform" / "skills").resolve()
    try:
        origin_path.resolve().relative_to(platform_root)
    except ValueError:
        pytest.fail(
            f"expected origin {origin} to be under platform skills dir "
            f"{platform_root}, got {origin}"
        )


def test_builtin_symlink_trust_label_via_trust_for():
    """trust_for() must return 'builtin' for PROVENANCE_BUILTIN, regardless of
    whether the file was reached via a real path or a symlink. (Smoke check
    for the trust label that `hermes skills list` would display.)"""
    # The actual classification is path-independent; this asserts the label
    # function is what the skills list relies on.
    assert trust_for(PROVENANCE_BUILTIN) == "builtin"
    assert trust_for(PROVENANCE_HUB) == "community"
    assert trust_for(PROVENANCE_LOCAL) == "local"
    assert trust_for(PROVENANCE_LOCAL_EDIT) == "local"