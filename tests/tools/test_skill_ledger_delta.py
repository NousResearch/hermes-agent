"""Ledger entries carry deltas, not whole-package manifests.

``rollback_entry`` writes every *before* path and removes *after*-only paths, so an unchanged
file in both lists is dead weight; a 4,000-file skill cost 1.5 MB of ledger per patch.
"""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def ledger_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _manifest(skills: Path, name: str, **files: str):
    from tools import skill_ledger
    return [{"path": str(skills / name / rel), "sha256": skill_ledger._store_blob(body.encode())}
            for rel, body in sorted(files.items())]


def test_entry_stores_only_changed_paths_and_rollback_still_restores(ledger_home):
    from tools import skill_ledger
    skills = ledger_home / "skills"
    before = _manifest(skills, "big", **{"SKILL.md": "v1", "references/a.md": "same", "references/gone.md": "old"})
    after = _manifest(skills, "big", **{"SKILL.md": "v2", "references/a.md": "same", "references/new.md": "added"})
    entry_id = skill_ledger.append_entry("patch", "big", before=before, after=after, actor="agent")
    entry = skill_ledger.get_entry(entry_id)
    names = lambda items: {Path(i["path"]).name for i in items}  # noqa: E731
    assert names(entry["before"]) == {"SKILL.md", "gone.md"}
    assert names(entry["after"]) == {"SKILL.md", "new.md"}

    # Lay the after-state on disk, roll back, and the before-state is exactly reproduced.
    for i in after:
        p = Path(i["path"]); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(skill_ledger.read_blob(i["sha256"]))
    ok, msg = skill_ledger.rollback_entry(entry_id)
    assert ok, msg
    assert (skills / "big" / "SKILL.md").read_text() == "v1"
    assert (skills / "big" / "references" / "a.md").read_text() == "same"
    assert (skills / "big" / "references" / "gone.md").read_text() == "old"
    assert not (skills / "big" / "references" / "new.md").exists()


def test_compact_rewrites_legacy_full_manifests_in_place(ledger_home):
    from tools import skill_ledger
    skills = ledger_home / "skills"
    full = _manifest(skills, "big", **{f"references/{n}.md": "same" for n in range(50)})
    changed_before = full + _manifest(skills, "big", **{"SKILL.md": "v1"})
    changed_after = full + _manifest(skills, "big", **{"SKILL.md": "v2"})
    legacy = {"id": "abc123", "ts": "2026-01-01T00:00:00+00:00", "actor": "agent", "action": "patch",
              "skill": "big", "evidence": {}, "before": changed_before, "after": changed_after}
    safety = {**legacy, "id": "def456", "action": "pre-rollback", "before": full, "after": full}
    path = skill_ledger.ledger_path()
    path.write_text(json.dumps(legacy) + "\n" + json.dumps(safety) + "\nnot json\n", encoding="utf-8")
    size_before = os.path.getsize(path)

    entries, raw_before, raw_after = skill_ledger.compact_ledger()
    assert (entries, raw_before) == (2, size_before)
    assert raw_after < raw_before * 0.6, "the legacy patch entry shrank to its delta (the safety entry keeps its full capture)"
    rows = skill_ledger.list_entries()
    by_id = {r["id"]: r for r in rows}
    assert {Path(i["path"]).name for i in by_id["abc123"]["before"]} == {"SKILL.md"}
    assert len(by_id["def456"]["before"]) == 50, "pre-rollback safety entries keep their full capture"
    assert "not json" in path.read_text(encoding="utf-8")


def test_compact_leaves_an_undecodable_ledger_untouched(ledger_home, caplog):
    """Compaction must not replace a ledger it could not decode (gate mutation M2)."""
    from tools import skill_ledger
    ledger = skill_ledger.ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(b"\xff")
    assert skill_ledger.compact_ledger() == (0, 0, 0)
    assert ledger.read_bytes() == b"\xff"
    assert "compaction skipped" in caplog.text


def test_list_entries_treats_an_undecodable_ledger_as_empty_and_warns(ledger_home, caplog):
    """A ledger that exists but is not UTF-8 lists as empty (rollback then fails closed on
    ``get_entry`` -> None) and, unlike a merely missing ledger, is warned about (re-gate M8/S1)."""
    from tools import skill_ledger
    ledger = skill_ledger.ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(b"\xff")
    assert skill_ledger.list_entries() == []
    assert skill_ledger.get_entry("deadbeef") is None
    assert "listing empty" in caplog.text


def test_list_entries_is_silent_when_the_ledger_is_merely_missing(ledger_home, caplog):
    from tools import skill_ledger
    assert not skill_ledger.ledger_path().exists()
    assert skill_ledger.list_entries() == []
    assert "listing empty" not in caplog.text


def test_gc_blobs_removes_only_unreferenced(ledger_home):
    """The blob store was write-only (#107539): after compaction, blobs no entry references are
    deleted; every referenced blob survives so any entry can still roll back."""
    from tools import skill_ledger
    skills = ledger_home / "skills"
    kept = _manifest(skills, "s", **{"SKILL.md": "v1"})
    skill_ledger.append_entry("create", "s", before=[], after=kept, actor="agent")
    orphan = skill_ledger._store_blob(b"never referenced by any entry")
    assert (skill_ledger.blobs_dir() / orphan).exists()

    deleted, freed = skill_ledger.gc_blobs()
    assert (deleted, freed) == (1, len(b"never referenced by any entry"))
    assert not (skill_ledger.blobs_dir() / orphan).exists()
    assert skill_ledger.read_blob(kept[0]["sha256"]) == b"v1"

    # A malformed line might hold references we cannot read: the sweep refuses rather than guesses.
    with open(skill_ledger.ledger_path(), "a", encoding="utf-8") as fh:
        fh.write("{broken\n")
    skill_ledger._store_blob(b"orphan two")
    assert skill_ledger.gc_blobs() == (0, 0)


@pytest.mark.parametrize("failure", ["unreadable", "missing", "invalid-encoding", "malformed-json", "non-dict-row"])
def test_gc_keeps_rollback_blobs_when_the_ledger_cannot_be_read(ledger_home, caplog, failure):
    from tools import skill_ledger

    skill = ledger_home / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text("original", encoding="utf-8")
    before = skill_ledger.snapshot_paths(skill)
    skill.write_text("edited", encoding="utf-8")
    entry_id = skill_ledger.record_mutation("patch", "demo", before=before, after_root=skill)
    ledger = skill_ledger.ledger_path()
    saved = ledger.read_bytes()
    blobs_before = {p.name: p.read_bytes() for p in skill_ledger.blobs_dir().iterdir()}
    if failure == "unreadable":
        # A directory at the ledger path fails read_text() with an OSError on every platform
        # (IsADirectoryError on POSIX, PermissionError on Windows) without patching Path.
        ledger.unlink()
        ledger.mkdir()
    elif failure == "missing":
        ledger.unlink()
    elif failure == "malformed-json":
        ledger.write_bytes(saved + b"{broken\n")
    elif failure == "non-dict-row":
        ledger.write_bytes(saved + b"[]\n")
    else:
        ledger.write_bytes(b"\xff")
    assert skill_ledger.gc_blobs() == (0, 0)
    assert "blob GC skipped" in caplog.text
    assert {p.name: p.read_bytes() for p in skill_ledger.blobs_dir().iterdir()} == blobs_before
    if ledger.is_dir():
        ledger.rmdir()
    ledger.write_bytes(saved)
    ok, message = skill_ledger.rollback_entry(entry_id)
    assert ok, message
    assert skill.read_text(encoding="utf-8") == "original"
