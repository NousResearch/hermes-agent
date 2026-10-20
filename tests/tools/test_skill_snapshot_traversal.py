"""Skill snapshots must prune transient trees before walking their contents."""

import os
from pathlib import Path

from tools import skill_ledger


def test_snapshot_never_descends_into_transient_directories(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    skill = home / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Demo", encoding="utf-8")
    # A plain file with an excluded directory name is still authored content.
    (skill / "venv").write_text("bootstrap", encoding="utf-8")
    kept = skill / "scripts" / "run.py"
    kept.parent.mkdir()
    kept.write_text("print('demo')", encoding="utf-8")
    excluded = {skill / ".venv", skill / "scripts" / "node_modules"}
    for directory in excluded:
        directory.mkdir()
        (directory / "irrelevant.txt").write_text("generated", encoding="utf-8")
    real_scandir = os.scandir

    def scandir(path):
        assert Path(path) not in excluded, "snapshot traversed an excluded dependency tree"
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    snapshot = skill_ledger.snapshot_paths(skill)
    contents = {Path(item["path"]).relative_to(skill).as_posix(): skill_ledger.read_blob(item["sha256"])
                for item in snapshot}
    assert contents == {"SKILL.md": b"# Demo", "venv": b"bootstrap", "scripts/run.py": b"print('demo')"}
