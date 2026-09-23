"""A blocked security scan must not delete a pre-existing directory (#119534)."""

from pathlib import Path

import tools.skill_manager_tool as smt
import hermes_constants

CONTENT = "---\nname: demo\ndescription: demo skill\n---\n\nbody\n"


def _stub(monkeypatch, tmp_path, blocked="blocked: test"):
    monkeypatch.setattr(smt, "_resolve_skill_dir", lambda name, category=None: tmp_path / name)
    monkeypatch.setattr(smt, "_find_skill", lambda name: None)
    monkeypatch.setattr(smt, "_security_scan_skill", lambda skill_dir: blocked)
    monkeypatch.setattr(
        hermes_constants, "mkdir_under_hermes_home",
        lambda p: Path(p).mkdir(parents=True, exist_ok=True),
    )


def test_blocked_scan_keeps_preexisting_dir_contents(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)
    (tmp_path / "demo").mkdir()
    keep = tmp_path / "demo" / "user-data.txt"
    keep.write_text("precious", encoding="utf-8")

    result = smt._create_skill("demo", CONTENT)

    assert result["success"] is False
    assert keep.read_text(encoding="utf-8") == "precious"
    assert not (tmp_path / "demo" / "SKILL.md").exists()


def test_blocked_scan_restores_preexisting_skill_md(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)
    (tmp_path / "demo").mkdir()
    skill_md = tmp_path / "demo" / "SKILL.md"
    skill_md.write_text("---\nname: old\ndescription: old\n---\n\nold body\n", encoding="utf-8")

    result = smt._create_skill("demo", CONTENT)

    assert result["success"] is False
    assert skill_md.read_text(encoding="utf-8") == "---\nname: old\ndescription: old\n---\n\nold body\n"


def test_blocked_scan_removes_a_directory_this_call_created(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path)

    result = smt._create_skill("demo", CONTENT)

    assert result["success"] is False
    assert not (tmp_path / "demo").exists()
