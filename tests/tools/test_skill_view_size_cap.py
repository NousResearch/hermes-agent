"""Bare skill_view of a novel SKILL.md returns an outline, not the full body."""

from __future__ import annotations

import json
from unittest.mock import patch

from tools.skills_tool import SKILL_VIEW_MAX_CHARS, skill_view

_UNIQUE = "UNIQUE_SKILL_BODY_TOKEN_SHOULD_NOT_APPEAR_IN_OUTLINE"


def _fat_skill_md() -> str:
    filler = ("paragraph about incident lore. " * 80) + "\n"
    body = (
        "---\n"
        "name: fat-skill\n"
        "description: oversized recipe\n"
        "---\n\n"
        "# Fat skill\n\n"
        "## Dispatch\n\n"
        + filler
        + f"{_UNIQUE}\n\n"
        "## Harvest\n\n"
        + filler
    )
    while len(body) <= SKILL_VIEW_MAX_CHARS:
        body += filler
    return body


def test_bare_view_of_oversized_skill_is_outline(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "fat-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_fat_skill_md(), encoding="utf-8")
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "lore.md").write_text("LORE_FILE_BODY\n", encoding="utf-8")

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        data = json.loads(skill_view("fat-skill"))
    assert data["success"] is True
    assert data.get("truncated") is True
    assert "usage_hint" in data
    assert "file_path" in data["usage_hint"]
    assert _UNIQUE not in data["content"]
    assert "## Dispatch" in data["content"]
    assert "## Harvest" in data["content"]


def test_file_path_still_loads_named_file(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "fat-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_fat_skill_md(), encoding="utf-8")
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "lore.md").write_text("LORE_FILE_BODY\n", encoding="utf-8")

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        data = json.loads(skill_view("fat-skill", file_path="references/lore.md"))
    assert data["success"] is True
    assert "LORE_FILE_BODY" in json.dumps(data)
    assert _UNIQUE not in json.dumps(data)
