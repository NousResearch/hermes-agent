"""Local reference additions must not freeze bundled updates (related to #63482)."""
import shutil
from pathlib import Path

import pytest

from tools import skills_sync as ss
from tools.skills_sync_bundled_ops import list_user_modified_bundled_skills, remove_pristine_bundled_skills


@pytest.fixture
def installed(tmp_path, monkeypatch):
    home = tmp_path / "home"
    bundle = tmp_path / "bundle"
    src = bundle / "category/example"
    (src / "references").mkdir(parents=True)
    (src / "SKILL.md").write_text("---\nname: example\n---\nOriginal instructions.\n")
    (src / "references/shipped.md").write_text("Shipped reference.\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundle))
    assert ss.sync_skills(quiet=True)["copied"] == ["example"]
    return home, src, home / "skills/category/example"


@pytest.mark.parametrize("legacy", [
    "recorded", "missing", "malformed", "stale", "pristine-legacy",
    pytest.param("readonly", marks=pytest.mark.linux_only),
])
def test_local_references_survive_repeated_updates(installed, monkeypatch, tmp_path, legacy):
    home, src, dest = installed
    inventory = home / "skills/.bundled_references.json"
    if legacy == "pristine-legacy":
        inventory.unlink(missing_ok=True)
        (src / "references/shipped.md").unlink()
        assert ss.sync_skills(quiet=True)["updated"] == ["example"]
        (src / "references/shipped.md").write_text("Shipped reference.\n")
        assert ss.sync_skills(quiet=True)["updated"] == ["example"]
    if legacy == "missing":
        inventory.unlink(missing_ok=True)
    if legacy == "malformed":
        inventory.write_text('{"example": {"hash": [], "paths": false}}')
    if legacy == "stale":
        inventory.write_text('{"example": {"hash": "wrong-baseline", "paths": []}}')
    local = dest / "references/notes/local.md"
    local.parent.mkdir()
    local.write_bytes(b"My local notes.\r\n")
    (src / "SKILL.md").write_text("---\nname: example\n---\nUpstream v2.\n")
    (src / "references/new.md").write_text("New upstream reference.\n")
    assert not list_user_modified_bundled_skills()
    assert ss.sync_skills(quiet=True)["updated"] == ["example"]
    assert (dest / "SKILL.md").read_bytes() == (src / "SKILL.md").read_bytes()
    assert (dest / "references/new.md").read_bytes() == (src / "references/new.md").read_bytes()
    assert local.read_bytes() == b"My local notes.\r\n"
    assert ss._read_manifest()["example"] == ss._dir_hash(src)
    assert ss._dir_hash(dest) != ss._dir_hash(src)  # full-content hashing still sees local files
    # Ownership survives an upstream deletion; removed shipped files are NOT local additions.
    (src / "references/shipped.md").unlink()
    local.write_bytes(b"Updated local notes.\n")
    assert ss.sync_skills(quiet=True)["updated"] == ["example"]
    assert not (dest / "references/shipped.md").exists()
    assert local.read_bytes() == b"Updated local notes.\n"
    assert not ss.sync_skills(quiet=True)["updated"]
    assert not list_user_modified_bundled_skills()
    # Update-clean is not disposable: opt-out must not delete the user's additions.
    assert not remove_pristine_bundled_skills()["removed"]
    assert local.exists()
    # Same names in another home must not consume A's reference ownership metadata.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-home"))
    assert ss.sync_skills(quiet=True)["copied"] == ["example"]
    other = ss._skills_dir() / "category/example"
    (other / "references/shipped.md").write_text("B's new local reference.\n")
    (src / "SKILL.md").write_text("---\nname: example\n---\nUpstream v3.\n")
    assert ss.sync_skills(quiet=True)["updated"] == ["example"]
    assert (other / "references/shipped.md").read_text() == "B's new local reference.\n"
    monkeypatch.setenv("HERMES_HOME", str(home))
    readonly = [src, *src.rglob("*")] if legacy == "readonly" else []
    for path in readonly:
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        assert ss.sync_skills(quiet=True)["updated"] == ["example"]
    finally:
        for path in readonly:
            path.chmod(0o755 if path.is_dir() else 0o644)
    assert local.read_bytes() == b"Updated local notes.\n"
    # Recategorization carries local additions instead of stranding the old copy.
    moved = src.with_name("renamed-example")
    src.rename(moved)
    result = ss.sync_skills(quiet=True)
    assert result["relocated"] == ["example"]
    assert not dest.exists()
    assert (dest.with_name("renamed-example") / "references/notes/local.md").read_bytes() == b"Updated local notes.\n"


@pytest.mark.parametrize("case", [
    "edit-skill", "edit-reference", "delete-reference", "extra-script", "removed-owned-reference",
    "file-collision", "directory-collision", "ancestor-collision", "copy-failure", "unreadable-entry", "unreadable-reference",
    "ambiguous-legacy", pytest.param("symlink", marks=pytest.mark.linux_only),
])
def test_unsafe_updates_preserve_local_bytes_and_origin(installed, monkeypatch, case, tmp_path):
    home, src, dest = installed
    local = dest / "references/local.md"
    local.write_bytes(b"Local bytes.\r\n")
    if case == "edit-skill":
        (dest / "SKILL.md").write_text("User's instructions.")
    if case == "edit-reference":
        (dest / "references/shipped.md").write_text("User's edited reference.")
    if case == "delete-reference":
        (dest / "references/shipped.md").unlink()
    if case == "extra-script":
        (dest / "local.py").write_text("# Outside the reference-only exception.")
    if case == "removed-owned-reference":
        (dest / "references/shipped.md").write_text("Keep my edit even after upstream deletes it.")
        (src / "references/shipped.md").unlink()
    if case == "file-collision":
        (src / "references/local.md").write_text("New upstream owner of the same path.")
    if case == "directory-collision":
        (src / "references/local.md").mkdir()
        (src / "references/local.md/nested.md").write_text("Upstream nested content.")
    if case == "ancestor-collision":
        (src / "references/shipped.md").unlink()
        (src / "references").rmdir()
        (src / "references").write_text("Upstream file replaces directory.")
    if case == "ambiguous-legacy":
        (home / "skills/.bundled_references.json").unlink()
        (src / "references/shipped.md").unlink()
    if case == "symlink":
        outside = tmp_path / "outside.md"
        outside.write_text("Do not touch.")
        (dest / "references/link.md").symlink_to(outside)
    before = {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    origin = ss._read_manifest()
    inventory = (home / "skills/.bundled_references.json").read_bytes() if case != "ambiguous-legacy" else None
    (src / "SKILL.md").write_text("---\nname: example\n---\nUpdated upstream.\n")
    if case in {"unreadable-entry", "unreadable-reference"}:
        (dest / "z-local.txt").write_bytes(b"Cannot inspect this entry safely.")
        before[Path("z-local.txt")] = b"Cannot inspect this entry safely."
        real_is_file = Path.is_file
        def fail_stat(path):
            unreadable = local if case == "unreadable-reference" else dest / "z-local.txt"
            if path == unreadable:
                raise PermissionError("Injected unreadable entry")
            return real_is_file(path)
        monkeypatch.setattr(Path, "is_file", fail_stat)
    if case == "copy-failure":
        real_copy = shutil.copy2
        def fail_addition(source, target, *args, **kwargs):
            if Path(source).name == "local.md":
                raise OSError("Injected local-reference copy failure")
            return real_copy(source, target, *args, **kwargs)
        monkeypatch.setattr(shutil, "copy2", fail_addition)
    result = ss.sync_skills(quiet=True)
    if case in {"unreadable-entry", "unreadable-reference"}:
        assert [item["name"] for item in list_user_modified_bundled_skills()] == ["example"]
        monkeypatch.setattr(Path, "is_file", real_is_file)
    assert not result["updated"]
    if case != "copy-failure":
        assert result["user_modified"] == ["example"]
        assert [item["name"] for item in list_user_modified_bundled_skills()] == ["example"]
    assert {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file()} == before
    assert ss._read_manifest() == origin
    if inventory is not None:
        assert (home / "skills/.bundled_references.json").read_bytes() == inventory
    if case == "symlink":
        assert (dest / "references/link.md").is_symlink()
        assert outside.read_text() == "Do not touch."
