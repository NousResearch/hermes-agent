"""Proposed target identity must match native lookup without guessing traversal order."""

from pathlib import Path

import pytest

from tools.skill_proposed_mutations import ProposedSkillFS
from tools.skill_target_resolution import AmbiguousSkillTarget
from tools import skill_manager_tool as smt
from agent import skill_utils as su


def text(name):
    return f"---\nname: {name}\ndescription: Native lookup parity.\n---\nBody.\n"


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def create_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    write(tmp_path, "config.yaml", "skills:\n  write_approval: true\n  write_approval_mode: create\n")


@pytest.mark.parametrize("additions", [[], ["a/SKILL.md"], ["c/dup/SKILL.md"],
                                           ["c/dup/SKILL.md", "a/SKILL.md", "d/deep/dup/SKILL.md"]])
def test_virtual_lookup_rejects_ambiguity_before_and_after_real_publishes(tmp_path, additions, monkeypatch):
    root = tmp_path / "skills"
    for rel in ["a/dup/SKILL.md", "b/deep/dup/SKILL.md", "a/references/sample/SKILL.md"]:
        write(root, rel, text(Path(rel).parent.name))
    monkeypatch.setattr(su, "get_skill_search_roots", lambda **kwargs: [("global", root)])
    fs = ProposedSkillFS()
    assert set(fs.iter_skill_dirs(root)) == set(smt._iter_skill_dirs(root))
    for rel in additions:
        fs.write(root / rel, text(Path(rel).parent.name))
    predicted = set(fs.iter_skill_dirs(root))
    with pytest.raises(AmbiguousSkillTarget) as proposed:
        fs.find("dup")
    winner = fs.find("a/dup")
    for rel in additions:
        write(root, rel, text(Path(rel).parent.name))
    assert predicted == set(smt._iter_skill_dirs(root))
    with pytest.raises(AmbiguousSkillTarget) as actual:
        smt._find_skill("dup")
    assert proposed.value.candidates == actual.value.candidates
    assert winner == smt._find_skill("a/dup")


def test_frontmatter_alias_reads_proposed_bom_and_single_bom_semantics(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    target = write(root, "existing/SKILL.md", text("before"))
    monkeypatch.setattr(su, "get_skill_search_roots", lambda **kwargs: [("global", root)])
    fs = ProposedSkillFS()
    for value in ["\ufeff" + text("after"), "\ufeff\ufeff" + text("after")]:
        fs.write(target, value)
        expected = fs.find("after")
        target.write_text(value, encoding="utf-8")
        assert expected == smt._find_skill("after")


@pytest.mark.platforms("posix")
def test_lookup_does_not_follow_directory_aliases_but_follows_leaf_for_read(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    outside = tmp_path / "outside"
    write(outside, "child/SKILL.md", text("child"))
    leaf = write(outside, "SKILL.md", text("display"))
    root.mkdir()
    (root / "alias-dir").symlink_to(outside, target_is_directory=True)
    own = root / "existing"
    own.mkdir()
    (own / "SKILL.md").symlink_to(leaf)
    monkeypatch.setattr(su, "get_skill_search_roots", lambda **kwargs: [("global", root)])
    fs = ProposedSkillFS()
    assert set(fs.iter_skill_dirs(root)) == set(smt._iter_skill_dirs(root))
    assert fs.find("child") is None
    fs.write(own / "SKILL.md", text("renamed"))
    predicted = fs.find("renamed")
    leaf.write_text(text("renamed"), encoding="utf-8")
    assert predicted == smt._find_skill("renamed")
