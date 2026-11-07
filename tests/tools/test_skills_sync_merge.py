"""Bundled updates reconcile a real temporary profile, never the live home."""
from pathlib import Path

import pytest

from tools import skills_sync as ss
from tools import skills_sync_merge as merge


def test_conflict_is_pending_outside_active_content_and_resolution_is_guarded(profile):
    home, source, local = profile
    ours = BASE.replace(b"first", b"my first")
    (local / "SKILL.md").write_bytes(ours)
    upstream = BASE.replace(b"first", b"upstream first")
    (source / "SKILL.md").write_bytes(upstream)
    origin = ss._read_manifest()["example"]
    result = ss.sync_skills(quiet=True)
    assert result["user_modified"] == ["example"]
    pending = merge.state_dir(home, "example") / "pending"
    assert (pending / "local" / "SKILL.md").read_bytes() == ours
    assert (pending / "upstream" / "SKILL.md").read_bytes() == upstream
    assert (local / "SKILL.md").read_bytes() == ours
    assert ss._read_manifest()["example"] == origin
    (source / "SKILL.md").write_bytes(upstream + b"later\n")
    ss.sync_skills(quiet=True)
    assert (pending / "upstream" / "SKILL.md").read_bytes() == upstream
    (local / "SKILL.md").write_bytes(ours + b"new local edit\n")
    with pytest.raises(ValueError, match="changed"):
        merge.resolve_skill("example", "merged")
    (local / "SKILL.md").write_bytes(ours)
    accepted = upstream.replace(b"upstream first", b"combined first")
    (pending / "merged" / "SKILL.md").write_bytes(accepted)
    merge.resolve_skill("example", "merged")
    assert (local / "SKILL.md").read_bytes() == accepted
    assert not pending.exists()
    assert any((p / "local" / "SKILL.md").read_bytes() == ours
               for p in pending.parent.glob("accepted-*"))
    assert merge.resolve_skill("example", "merged") is False
    assert ss.sync_skills(quiet=True)["updated"] == ["example"]
    assert (local / "SKILL.md").read_bytes() == accepted + b"later\n"


BASE = b"---\nname: example\ndescription: Example\n---\nfirst\nsecond\nthird\nfourth\nfifth\n"


def test_pending_never_overwrites_even_if_local_reverts_to_old_base(profile):
    _home, source, local = profile
    (local / "SKILL.md").write_bytes(BASE.replace(b"first", b"mine"))
    (source / "SKILL.md").write_bytes(BASE.replace(b"first", b"theirs"))
    ss.sync_skills(quiet=True)
    (local / "SKILL.md").write_bytes(BASE)
    ss.sync_skills(quiet=True)
    assert (local / "SKILL.md").read_bytes() == BASE


@pytest.mark.parametrize("origin", ["", "unrecoverable-legacy-hash"])
def test_unknown_legacy_base_requires_explicit_reconciliation(profile, origin):
    home, source, local = profile
    merge_base = merge.state_dir(home, "example") / "base.json"
    merge_base.unlink()
    ss._write_manifest({"example": origin})
    ours = BASE.replace(b"first", b"mine")
    (local / "SKILL.md").write_bytes(ours)
    (source / "SKILL.md").write_bytes(BASE.replace(b"fifth", b"new fifth"))
    ss.sync_skills(quiet=True)
    assert ss._read_manifest()["example"] == origin
    pending = merge.state_dir(home, "example") / "pending"
    assert (pending / "merged" / "SKILL.md").read_bytes() == ours
    assert merge.resolve_skill("example", "local")
    assert (local / "SKILL.md").read_bytes() == ours


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    source = tmp_path / "bundle" / "category" / "example"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_bytes(BASE)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ss, "_get_bundled_dir", lambda: source.parents[1])
    monkeypatch.setattr(ss, "_get_optional_dir", lambda: tmp_path / "no-optional")
    ss.sync_skills(quiet=True)
    return home, source, home / "skills" / "category" / "example"


def test_nonoverlapping_edits_merge_and_next_update_uses_upstream_base(profile):
    _home, source, local = profile
    (local / "SKILL.md").write_bytes(BASE.replace(b"first", b"my first"))
    upstream = BASE.replace(b"fifth", b"new fifth")
    (source / "SKILL.md").write_bytes(upstream)
    result = ss.sync_skills(quiet=True)
    assert result["updated"] == ["example"]
    assert (local / "SKILL.md").read_bytes() == upstream.replace(b"first", b"my first")
    assert ss._read_manifest()["example"] == ss.dir_hash(source)
    assert not ss.sync_skills(quiet=True)["updated"]
    (source / "SKILL.md").write_bytes(upstream.replace(b"third", b"new third"))
    assert ss.sync_skills(quiet=True)["updated"] == ["example"]
    assert (local / "SKILL.md").read_bytes() == upstream.replace(b"first", b"my first").replace(b"third", b"new third")


def test_file_additions_deletions_and_one_sided_binary_updates(profile):
    _home, source, local = profile
    (source / "removed.txt").write_bytes(b"remove me\n")
    (source / "asset.bin").write_bytes(b"\0old")
    ss.sync_skills(quiet=True)
    (local / "SKILL.md").write_bytes(BASE.replace(b"first", b"mine"))
    (local / "mine.txt").write_bytes(b"local only\n")
    (source / "removed.txt").unlink()
    (source / "asset.bin").write_bytes(b"\0new")
    (source / "added.txt").write_bytes(b"upstream only\n")
    assert ss.sync_skills(quiet=True)["updated"] == ["example"]
    assert not (local / "removed.txt").exists()
    assert (local / "asset.bin").read_bytes() == b"\0new"
    assert (local / "mine.txt").read_bytes() == b"local only\n"
    assert (local / "added.txt").read_bytes() == b"upstream only\n"


@pytest.mark.parametrize("base,ours,theirs", [
    (b"\0base", b"\0mine", b"\0theirs"),
    (b"base\n", None, b"changed\n"),
    (b"base\n", b"changed\n", None),
    (None, b"mine\n", b"theirs\n"),
])
def test_conflicting_file_operations_never_modify_active_tree(profile, base, ours, theirs):
    home, source, local = profile
    if base is not None:
        (source / "asset").write_bytes(base)
        ss.sync_skills(quiet=True)
    if ours is None:
        (local / "asset").unlink()
    else:
        (local / "asset").write_bytes(ours)
    if theirs is None:
        (source / "asset").unlink()
    else:
        (source / "asset").write_bytes(theirs)
    before = merge.read_tree(local)
    origin = ss._read_manifest()["example"]
    ss.sync_skills(quiet=True)
    assert merge.read_tree(local) == before
    assert ss._read_manifest()["example"] == origin
    assert (merge.state_dir(home, "example") / "pending").exists()


def test_file_directory_collision_is_pending_not_partial_update(profile):
    _home, source, local = profile
    (local / "extra").write_bytes(b"local file")
    (source / "extra").mkdir()
    (source / "extra" / "child").write_bytes(b"upstream child")
    assert ss.sync_skills(quiet=True)["user_modified"] == ["example"]
    assert (local / "extra").read_bytes() == b"local file"


def test_resolution_rejects_manifest_path_escape(profile, tmp_path):
    import json

    home, source, local = profile
    (local / "SKILL.md").write_bytes(BASE.replace(b"first", b"mine"))
    (source / "SKILL.md").write_bytes(BASE.replace(b"first", b"theirs"))
    ss.sync_skills(quiet=True)
    path = merge.state_dir(home, "example") / "pending" / "pending.json"
    data = json.loads(path.read_text())
    data["destination"] = "../outside"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsafe"):
        merge.resolve_skill("example", "upstream")
    assert not (home / "outside").exists()


def test_profiles_have_separate_snapshot_bases(profile, tmp_path, monkeypatch):
    home, source, local = profile
    other = tmp_path / "other"
    monkeypatch.setenv("HERMES_HOME", str(other))
    ss.sync_skills(quiet=True)
    (local / "SKILL.md").write_bytes(BASE.replace(b"first", b"mine"))
    (source / "SKILL.md").write_bytes(BASE.replace(b"fifth", b"new fifth"))
    ss.sync_skills(quiet=True)
    assert (local / "SKILL.md").read_bytes() == BASE.replace(b"first", b"mine")
    assert ss._read_manifest()["example"] == ss.dir_hash(source)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert ss.sync_skills(quiet=True)["updated"] == ["example"]


def test_explicit_refresh_keeps_old_evidence_and_reconciles_new_local_edits(profile):
    home, source, local = profile
    (local / "SKILL.md").write_bytes(BASE.replace(b"first", b"mine"))
    (source / "SKILL.md").write_bytes(BASE.replace(b"first", b"theirs"))
    ss.sync_skills(quiet=True)
    newer = BASE.replace(b"first", b"mine now")
    (local / "SKILL.md").write_bytes(newer)
    merge.refresh_pending("example")
    pending = merge.state_dir(home, "example") / "pending"
    assert (pending / "local" / "SKILL.md").read_bytes() == newer
    assert list(pending.parent.glob("superseded-*"))
    assert merge.resolve_skill("example", "local")
    assert (local / "SKILL.md").read_bytes() == newer


@pytest.mark.platforms("windows")
def test_sync_does_not_write_through_category_junction(profile, tmp_path):
    import subprocess

    _home, source, local = profile
    category = local.parent
    outside = tmp_path / "outside"
    category.rename(outside)
    subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(category), str(outside)],
                   check=True, capture_output=True)
    upstream = BASE.replace(b"fifth", b"new fifth")
    (source / "SKILL.md").write_bytes(upstream)
    ss.sync_skills(quiet=True)
    assert (outside / "example" / "SKILL.md").read_bytes() == BASE


@pytest.mark.platforms("windows")
def test_resolution_rejects_category_junction_into_another_skill_tree(profile):
    import subprocess

    home, source, local = profile
    ours = BASE.replace(b"first", b"mine")
    (local / "SKILL.md").write_bytes(ours)
    (source / "SKILL.md").write_bytes(BASE.replace(b"first", b"theirs"))
    ss.sync_skills(quiet=True)
    category = local.parent
    other = home / "skills" / "other"
    category.rename(other)
    subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(category), str(other)],
                   check=True, capture_output=True)
    try:
        with pytest.raises(ValueError, match="linked"):
            merge.resolve_skill("example", "upstream")
        assert (other / "example" / "SKILL.md").read_bytes() == ours
        assert (merge.state_dir(home, "example") / "pending").exists()
    finally:
        category.rmdir()


def test_accept_local_ignores_editable_review_copy_and_preserves_active_bytes(profile):
    home, source, local = profile
    ours = b"\xef\xbb\xbf" + BASE.replace(b"first", b"mine").replace(b"\n", b"\r\n")
    (local / "SKILL.md").write_bytes(ours)
    (local / "asset.bin").write_bytes(b"\x00\xffcustom")
    (source / "SKILL.md").write_bytes(BASE.replace(b"first", b"theirs"))
    ss.sync_skills(quiet=True)
    before = merge.read_tree(local)
    pending = merge.state_dir(home, "example") / "pending"
    (pending / "local" / "SKILL.md").write_bytes(b"tampered review copy\n")
    (pending / "local" / "asset.bin").unlink()
    assert merge.resolve_skill("example", "local")
    assert merge.read_tree(local) == before
    assert ss._read_manifest()["example"] == ss.dir_hash(source)
    assert not pending.exists()


def test_unchanged_sync_does_not_rewrite_valid_snapshot(profile, monkeypatch):
    def unexpected_write(*args, **kwargs):
        pytest.fail("An unchanged upstream snapshot must not be rewritten")

    monkeypatch.setattr(merge, "atomic_write_text", unexpected_write)
    assert not ss.sync_skills(quiet=True)["updated"]


def test_failed_manifest_write_explains_refresh_recovery_and_preserves_evidence(profile, monkeypatch):
    home, source, local = profile
    ours = BASE.replace(b"first", b"mine")
    theirs = BASE.replace(b"first", b"theirs")
    (local / "SKILL.md").write_bytes(ours)
    (source / "SKILL.md").write_bytes(theirs)
    ss.sync_skills(quiet=True)
    origin = ss._read_manifest()["example"]
    pending = merge.state_dir(home, "example") / "pending"
    with monkeypatch.context() as failure:
        failure.setattr(ss, "_write_manifest", lambda entries: None)
        with pytest.raises(OSError, match=r"hermes skills resolve example --refresh"):
            merge.resolve_skill("example", "upstream")
    assert pending.exists()
    assert (pending / "local" / "SKILL.md").read_bytes() == ours
    assert (local / "SKILL.md").read_bytes() == theirs
    assert ss._read_manifest()["example"] == origin
    merge.refresh_pending("example")
    assert any((p / "local" / "SKILL.md").read_bytes() == ours
               for p in pending.parent.glob("superseded-*"))
    assert merge.resolve_skill("example", "upstream")
    assert (local / "SKILL.md").read_bytes() == theirs
    assert ss._read_manifest()["example"] == ss.dir_hash(source)
    assert not pending.exists()


def test_exact_legacy_upstream_can_recover_base_for_next_merge(profile):
    home, source, local = profile
    (merge.state_dir(home, "example") / "base.json").unlink()
    (local / "SKILL.md").write_bytes(BASE.replace(b"first", b"mine"))
    ss.sync_skills(quiet=True)
    (source / "SKILL.md").write_bytes(BASE.replace(b"fifth", b"new fifth"))
    assert ss.sync_skills(quiet=True)["updated"] == ["example"]
    assert (local / "SKILL.md").read_bytes() == BASE.replace(b"first", b"mine").replace(b"fifth", b"new fifth")
