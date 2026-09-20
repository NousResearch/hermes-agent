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
