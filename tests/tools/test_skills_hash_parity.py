"""Regression tests for #71237: disk vs bundle hash parity across platforms.

``tools.skills_guard.content_hash`` (disk) and
``tools.skills_hub.bundle_content_hash`` (in-memory bundle) must produce the
same digest for the same skill content, regardless of platform path-separator
conventions or filesystem iteration order. When they diverge, every
``hermes skills check`` reports ``update_available`` for an up-to-date skill
and ``hermes skills update`` never converges (the perpetual-update loop on
Windows for skills with subdirectories).
"""

from tools.skills_guard import content_hash
from tools.skills_hub import SkillBundle, bundle_content_hash


def _make_skill_dir(tmp_path):
    """Skill layout with subdirectories, mirroring a real hub skill."""
    (tmp_path / "SKILL.md").write_bytes(b"---\nname: parity\n---\n\n# Parity\n")
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "cli.md").write_bytes(b"cli reference\n")
    (tmp_path / "references" / "composition.md").write_bytes(b"composition\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "setup.sh").write_bytes(b"#!/bin/sh\n")
    return tmp_path


def _bundle_from(files):
    return SkillBundle(
        name="parity",
        files=files,
        source="official",
        identifier="official/parity",
        trust_level="builtin",
    )


def test_disk_hash_matches_bundle_hash_with_subdirectories(tmp_path):
    """The disk digest and the bundle digest must agree on every platform.

    Regression for the sort-order half of #71237: ``_content_digest`` used to
    iterate ``sorted(Path.rglob(...))`` (component-wise Path ordering) while
    ``bundle_content_hash`` sorts the posix strings alphabetically, so skills
    with subdirectories hashed their files in a different order on disk than
    in the bundle.
    """
    skill_dir = _make_skill_dir(tmp_path)
    files = {
        f.relative_to(skill_dir).as_posix(): f.read_bytes()
        for f in skill_dir.rglob("*")
        if f.is_file()
    }
    assert content_hash(skill_dir) == bundle_content_hash(_bundle_from(files))


def test_bundle_hash_normalizes_backslash_keys(tmp_path):
    """Backslash-keyed bundles must hash identically to posix-keyed ones.

    Regression for the separator half of #71237: bundles built on Windows can
    carry ``references\\cli.md``-style keys; the digest must be computed on
    the posix form so it matches the disk-side ``content_hash``.
    """
    skill_dir = _make_skill_dir(tmp_path)
    posix_files = {
        f.relative_to(skill_dir).as_posix(): f.read_bytes()
        for f in skill_dir.rglob("*")
        if f.is_file()
    }
    windows_files = {k.replace("/", "\\"): v for k, v in posix_files.items()}

    posix_hash = bundle_content_hash(_bundle_from(posix_files))
    windows_hash = bundle_content_hash(_bundle_from(windows_files))

    assert posix_hash == windows_hash
    assert posix_hash == content_hash(skill_dir)


def test_flat_skill_still_matches(tmp_path):
    """Flat skills (a single SKILL.md) never regressed; keep it that way."""
    (tmp_path / "SKILL.md").write_bytes(b"---\nname: flat\n---\n")
    files = {"SKILL.md": (tmp_path / "SKILL.md").read_bytes()}
    assert content_hash(tmp_path) == bundle_content_hash(_bundle_from(files))
