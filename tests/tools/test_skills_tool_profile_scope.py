"""Regression tests for profile-scoped skills_tool path resolution.

Scope: the diagnostic half of #110524 (the reported resolution/containment "mismatch" was settled
in-thread as by-design symlink blocking) — a linked file that resolves outside the located skill
directory must say *that*, and the outside-trusted-dir warning must name the active skills dir
instead of the hardcoded ``~/.hermes/skills/``.
"""

import importlib
import json
import logging
import sys
from pathlib import Path

import pytest


def _write_skill(root: Path, category: str, name: str, description: str) -> Path:
    skill_dir = root / "skills" / category / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"---\n\n"
        f"# {name}\n\n"
        f"Loaded from {description}.\n",
        encoding="utf-8",
    )
    return skill_dir


def _reload_skills_tool(import_home: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(import_home))
    import tools.skills_tool as skills_tool

    return importlib.reload(skills_tool)


def test_skill_view_uses_live_profile_home_after_module_import(tmp_path, monkeypatch):
    """skill_view should not stay pinned to HERMES_HOME from import time."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    _write_skill(default_home, "autonomous-ai-agents", "default-only", "default home")
    profile_skill_dir = _write_skill(
        profile_home,
        "software-development",
        "kanban-orchestrator-operations",
        "orchestrator profile",
    )

    skills_tool = _reload_skills_tool(default_home, monkeypatch)
    assert skills_tool.SKILLS_DIR == default_home / "skills"

    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    result = json.loads(
        skills_tool.skill_view("kanban-orchestrator-operations", preprocess=False)
    )

    assert result["success"] is True
    assert result["name"] == "kanban-orchestrator-operations"
    assert Path(result["skill_dir"]) == profile_skill_dir
    assert "orchestrator profile" in result["content"]


def test_explicit_skills_dir_monkeypatch_still_wins(tmp_path, monkeypatch):
    """Existing tests can still override tools.skills_tool.SKILLS_DIR directly."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "orchestrator"
    patched_root = tmp_path / "patched"
    patched_skill_dir = _write_skill(
        patched_root,
        "software-development",
        "patched-skill",
        "patched skills dir",
    )
    _write_skill(
        profile_home,
        "software-development",
        "profile-skill",
        "orchestrator profile",
    )

    skills_tool = _reload_skills_tool(default_home, monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", patched_root / "skills")

    result = json.loads(skills_tool.skill_view("patched-skill", preprocess=False))

    assert result["success"] is True
    assert Path(result["skill_dir"]) == patched_skill_dir


# ---- linked-file diagnostics in a profile (#110524) ----


def _link(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:  # Windows without developer mode
        pytest.skip(f"symlinks unavailable: {exc}")


def _canonical_skill(root: Path, category: str, name: str) -> Path:
    """A real skill in the canonical ``<root>/skills`` tree with a linked file."""
    skill_dir = _write_skill(root, category, name, "canonical tree")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "api.md").write_text("canonical api docs", encoding="utf-8")
    return skill_dir


def _linked_profile_skill(profile_home: Path, canonical: Path, name: str, *, dir_link: bool):
    """The ``link_skill_tree.py`` layout: a profile skill whose entries point into the canonical
    tree, either per file or via one linked support directory."""
    linked = profile_home / "skills" / "cat" / name
    linked.mkdir(parents=True)
    _link(linked / "SKILL.md", canonical / "SKILL.md")
    if dir_link:
        _link(linked / "references", canonical / "references", directory=True)
    else:
        (linked / "references").mkdir()
        _link(linked / "references" / "api.md", canonical / "references" / "api.md")
    return linked


@pytest.mark.parametrize("dir_link", [False, True], ids=["file-link", "dir-link"])
def test_profile_linked_file_reports_that_it_is_a_link(tmp_path, monkeypatch, dir_link):
    """#110524: a file-level-linked profile skill still refuses the file (by design), but the error
    now names the cause — the entry is a link resolving outside the located skill directory — and
    still leaks nothing from the link target."""
    root = tmp_path / "root"
    profile_home = root / "profiles" / "p"
    canonical = _canonical_skill(root, "cat", "linked-skill")
    linked = _linked_profile_skill(profile_home, canonical, "linked-skill", dir_link=dir_link)

    skills_tool = _reload_skills_tool(profile_home, monkeypatch)

    main = json.loads(skills_tool.skill_view("linked-skill", preprocess=False))
    assert main["success"] is True
    assert Path(main["skill_dir"]) == linked

    result = json.loads(
        skills_tool.skill_view("linked-skill", "references/api.md", preprocess=False)
    )
    assert result["success"] is False, result
    error = result["error"]
    assert "escapes" in error.lower()
    assert "link" in error.lower()
    assert str(canonical / "references" / "api.md") in error  # names where it resolved to
    assert "link_skill_tree" in error  # names the layout that trips it
    assert "canonical api docs" not in json.dumps(result)


def test_profile_local_copy_linked_files_still_serve(tmp_path, monkeypatch):
    """A plain profile-local copy keeps serving its linked files (unchanged by this change)."""
    profile_home = tmp_path / "root" / "profiles" / "p"
    skill_dir = _write_skill(profile_home, "cat", "copy-skill", "profile copy")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "api.md").write_text("local api docs", encoding="utf-8")

    skills_tool = _reload_skills_tool(profile_home, monkeypatch)

    result = json.loads(skills_tool.skill_view("copy-skill", "references/api.md", preprocess=False))
    assert result["success"] is True, result
    assert result["content"] == "local api docs"


def test_profile_links_outside_every_skills_tree_still_blocked(tmp_path, monkeypatch):
    """Links resolving outside every trusted skills tree stay refused — the guard is not widened."""
    root = tmp_path / "root"
    profile_home = root / "profiles" / "p"
    outside = tmp_path / "outside" / "elsewhere"
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_text("---\nname: elsewhere\n---\nbody\n", encoding="utf-8")
    (outside / "api.md").write_text("TOP SECRET", encoding="utf-8")

    linked = profile_home / "skills" / "cat" / "escaped-skill"
    linked.mkdir(parents=True)
    _link(linked / "SKILL.md", outside / "SKILL.md")
    _link(linked / "api.md", outside / "api.md")

    skills_tool = _reload_skills_tool(profile_home, monkeypatch)

    result = json.loads(skills_tool.skill_view("escaped-skill", "api.md", preprocess=False))
    assert result["success"] is False
    assert "escapes" in result["error"].lower()
    assert "TOP SECRET" not in json.dumps(result)


def test_profile_linked_file_traversal_and_missing_paths(tmp_path, monkeypatch):
    """Traversal and not-found handling are unchanged in profile mode."""
    root = tmp_path / "root"
    profile_home = root / "profiles" / "p"
    canonical = _canonical_skill(root, "cat", "linked-skill")
    _linked_profile_skill(profile_home, canonical, "linked-skill", dir_link=False)

    skills_tool = _reload_skills_tool(profile_home, monkeypatch)

    escaped = json.loads(
        skills_tool.skill_view("linked-skill", "../../../../etc/passwd", preprocess=False)
    )
    assert escaped["success"] is False
    assert "traversal" in escaped["error"].lower()

    absent = json.loads(
        skills_tool.skill_view("linked-skill", "references/missing.md", preprocess=False)
    )
    assert absent["success"] is False
    assert "not found" in absent["error"].lower()


def test_security_warning_names_the_active_skills_dir(tmp_path, monkeypatch, caplog):
    """The outside-trusted-dir warning names the dir actually in force (the profile's own skills
    dir), not the hardcoded ``~/.hermes/skills/`` that misreported it."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "p"
    stray = tmp_path / "untracked" / "cat" / "stray-skill"
    stray.mkdir(parents=True)
    (stray / "SKILL.md").write_text("---\nname: stray-skill\n---\nbody\n", encoding="utf-8")

    skills_tool = _reload_skills_tool(default_home, monkeypatch)
    active = profile_home / "skills"
    active.mkdir(parents=True)

    with caplog.at_level(logging.WARNING, logger="tools.skills_tool"):
        skills_tool._log_security_warnings("stray-skill", stray / "SKILL.md", "body", [], active)

    assert "outside the trusted skills directory" in caplog.text
    assert str(active) in caplog.text
    assert "~/.hermes/skills/" not in caplog.text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path separator/case semantics")
def test_profile_linked_file_windows_separators_and_case(tmp_path, monkeypatch):
    """A local copy serves through backslash separators and mixed-case components on Windows."""
    profile_home = tmp_path / "root" / "profiles" / "p"
    skill_dir = _write_skill(profile_home, "cat", "copy-skill", "profile copy")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "api.md").write_text("local api docs", encoding="utf-8")

    skills_tool = _reload_skills_tool(profile_home, monkeypatch)

    for file_path in ("references\\api.md", "REFERENCES\\API.MD"):
        result = json.loads(skills_tool.skill_view("copy-skill", file_path, preprocess=False))
        assert result["success"] is True, (file_path, result)
        assert result["content"] == "local api docs"
