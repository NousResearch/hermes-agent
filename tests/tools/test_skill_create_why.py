"""Behavior contract for ``skill_manage(action='create', why=...)``.

A skill's creation DATE is already in the usage ledger; its REASON is not. The ``why`` argument
records why the skill exists as a frontmatter line, and a line the author already wrote is never
overwritten.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools import skill_manager_tool as smt  # noqa: E402


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with an empty local skills dir; nothing touches the real profile."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_hermes_home_cache", None, raising=False)

    from agent import skill_utils as su
    su._external_dirs_cache_clear()

    from tools import skills_tool
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(smt, "SKILLS_DIR", home / "skills")
    yield home
    su._external_dirs_cache_clear()


def _skill_md(name: str, extra_frontmatter: str = "") -> str:
    return (
        f"---\nname: {name}\n"
        "description: Use when testing why recording. One-line behavior.\n"
        f"{extra_frontmatter}"
        "---\n\n"
        f"# {name}\n\nBody.\n"
    )


def _read(home, name: str) -> str:
    return (home / "skills" / name / "SKILL.md").read_text(encoding="utf-8")


def test_why_argument_is_written_to_frontmatter(isolated_home):
    """The reason survives the session that created the skill."""
    result = smt._create_skill("why-skill", _skill_md("why-skill"),
                               why="no way to answer why a skill exists after the session ends")
    assert result["success"] is True
    text = _read(isolated_home, "why-skill")
    assert "why: no way to answer why a skill exists after the session ends" in text
    assert text.index("why:") < text.index("\n---\n")  # inside the frontmatter, not the body


def test_no_why_argument_leaves_frontmatter_unchanged(isolated_home):
    """An author who says nothing gets no invented line."""
    content = _skill_md("plain-skill")
    assert smt._create_skill("plain-skill", content)["success"] is True
    assert _read(isolated_home, "plain-skill") == content


def test_author_written_why_is_not_overwritten(isolated_home):
    """An explicit why: line — date, reason, or both — wins over the argument."""
    content = _skill_md("authored-skill", "why: 2026-11-05 · written by hand\n")
    result = smt._create_skill("authored-skill", content, why="from an argument")
    assert result["success"] is True
    text = _read(isolated_home, "authored-skill")
    assert "why: 2026-11-05 · written by hand" in text
    assert "from an argument" not in text
    assert text.count("why:") == 1
