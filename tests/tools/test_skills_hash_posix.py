"""Regression for #71237: skill hash must use posix paths on all platforms."""

import hashlib
from pathlib import Path, PurePosixPath


def test_bundle_content_hash_uses_posix_paths(monkeypatch, tmp_path):
    """bundle_content_hash must produce the same hash regardless of OS
    path separator. The disk-side _content_digest uses .as_posix(), so
    the bundle-side hash must too."""
    from tools.skills_hub import bundle_content_hash, SkillBundle

    # Simulate a skill with subdirectory (the case that breaks on Windows)
    files = {
        "references/cli.md": b"# CLI ref",
        "SKILL.md": b"# Skill",
    }
    bundle = SkillBundle(
        name="test",
        source="hub",
        identifier="test-skill",
        files=files,
        metadata={},
        trust_level="trusted",
    )
    h1 = bundle_content_hash(bundle)

    # Now simulate what _content_digest produces (posix paths, sorted)
    h2 = hashlib.sha256()
    for rel_path in sorted(files.keys()):
        h2.update(rel_path.encode("utf-8") + b"\x00")
        h2.update(files[rel_path])
    expected = f"sha256:{h2.hexdigest()[:16]}"

    assert h1 == expected, f"Hash mismatch: {h1} != {expected}"
