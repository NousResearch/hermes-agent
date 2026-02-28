"""Tests for tools/skills_tool.py — skill discovery and viewing."""

import json
from pathlib import Path
from unittest.mock import patch

from tools.skills_tool import (
    _parse_frontmatter,
    _parse_tags,
    _get_category_from_path,
    _estimate_tokens,
    _find_all_skills,
    _load_category_description,
    skills_list,
    skills_categories,
    skill_view,
    SKILLS_DIR,
    MAX_NAME_LENGTH,
    MAX_DESCRIPTION_LENGTH,
)


def _make_skill(skills_dir, name, frontmatter_extra="", body="Step 1: Do the thing.", category=None):
    """Helper to create a minimal skill directory."""
    if category:
        skill_dir = skills_dir / category / name
    else:
        skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"""\
---
name: {name}
description: Description for {name}.
{frontmatter_extra}---

# {name}

{body}
"""
    (skill_dir / "SKILL.md").write_text(content)
    return skill_dir


# ---------------------------------------------------------------------------
# _parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        content = "---\nname: test\ndescription: A test.\n---\n\n# Body\n"
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "test"
        assert fm["description"] == "A test."
        assert "# Body" in body

    def test_no_frontmatter(self):
        content = "# Just a heading\nSome content.\n"
        fm, body = _parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_empty_frontmatter(self):
        content = "---\n---\n\n# Body\n"
        fm, body = _parse_frontmatter(content)
        assert fm == {}

    def test_nested_yaml(self):
        content = "---\nname: test\nmetadata:\n  hermes:\n    tags: [a, b]\n---\n\nBody.\n"
        fm, body = _parse_frontmatter(content)
        assert fm["metadata"]["hermes"]["tags"] == ["a", "b"]

    def test_malformed_yaml_fallback(self):
        """Malformed YAML falls back to simple key:value parsing."""
        content = "---\nname: test\ndescription: desc\n: invalid\n---\n\nBody.\n"
        fm, body = _parse_frontmatter(content)
        # Should still parse what it can via fallback
        assert "name" in fm


# ---------------------------------------------------------------------------
# _parse_tags
# ---------------------------------------------------------------------------


class TestParseTags:
    def test_list_input(self):
        assert _parse_tags(["a", "b", "c"]) == ["a", "b", "c"]

    def test_comma_separated_string(self):
        assert _parse_tags("a, b, c") == ["a", "b", "c"]

    def test_bracket_wrapped_string(self):
        assert _parse_tags("[a, b, c]") == ["a", "b", "c"]

    def test_empty_input(self):
        assert _parse_tags("") == []
        assert _parse_tags(None) == []
        assert _parse_tags([]) == []

    def test_strips_quotes(self):
        result = _parse_tags('"tag1", \'tag2\'')
        assert "tag1" in result
        assert "tag2" in result

    def test_filters_empty_items(self):
        assert _parse_tags([None, "", "valid"]) == ["valid"]


# ---------------------------------------------------------------------------
# _get_category_from_path
# ---------------------------------------------------------------------------


class TestGetCategoryFromPath:
    def test_categorized_skill(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_md = tmp_path / "mlops" / "axolotl" / "SKILL.md"
            skill_md.parent.mkdir(parents=True)
            skill_md.touch()
            assert _get_category_from_path(skill_md) == "mlops"

    def test_uncategorized_skill(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_md = tmp_path / "my-skill" / "SKILL.md"
            skill_md.parent.mkdir(parents=True)
            skill_md.touch()
            assert _get_category_from_path(skill_md) is None

    def test_outside_skills_dir(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"):
            skill_md = tmp_path / "other" / "SKILL.md"
            assert _get_category_from_path(skill_md) is None


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_estimate(self):
        assert _estimate_tokens("1234") == 1
        assert _estimate_tokens("12345678") == 2
        assert _estimate_tokens("") == 0


# ---------------------------------------------------------------------------
# _find_all_skills
# ---------------------------------------------------------------------------


class TestFindAllSkills:
    def test_finds_skills(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "skill-a")
            _make_skill(tmp_path, "skill-b")
            skills = _find_all_skills()
        assert len(skills) == 2
        names = {s["name"] for s in skills}
        assert "skill-a" in names
        assert "skill-b" in names

    def test_empty_directory(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skills = _find_all_skills()
        assert skills == []

    def test_nonexistent_directory(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "nope"):
            skills = _find_all_skills()
        assert skills == []

    def test_categorized_skills(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "axolotl", category="mlops")
            skills = _find_all_skills()
        assert len(skills) == 1
        assert skills[0]["category"] == "mlops"

    def test_description_from_body_when_missing(self, tmp_path):
        """If no description in frontmatter, first non-header line is used."""
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\n\n# Heading\n\nFirst paragraph.\n")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skills = _find_all_skills()
        assert skills[0]["description"] == "First paragraph."

    def test_long_description_truncated(self, tmp_path):
        long_desc = "x" * (MAX_DESCRIPTION_LENGTH + 100)
        skill_dir = tmp_path / "long-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(f"---\nname: long\ndescription: {long_desc}\n---\n\nBody.\n")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skills = _find_all_skills()
        assert len(skills[0]["description"]) <= MAX_DESCRIPTION_LENGTH

    def test_skips_git_directories(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "real-skill")
            git_dir = tmp_path / ".git" / "fake-skill"
            git_dir.mkdir(parents=True)
            (git_dir / "SKILL.md").write_text("---\nname: fake\ndescription: x\n---\n\nBody.\n")
            skills = _find_all_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "real-skill"


# ---------------------------------------------------------------------------
# skills_list
# ---------------------------------------------------------------------------


class TestSkillsList:
    def test_empty_creates_directory(self, tmp_path):
        skills_dir = tmp_path / "skills"
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            raw = skills_list()
        result = json.loads(raw)
        assert result["success"] is True
        assert result["skills"] == []
        assert skills_dir.exists()

    def test_lists_skills(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "alpha")
            _make_skill(tmp_path, "beta")
            raw = skills_list()
        result = json.loads(raw)
        assert result["count"] == 2

    def test_category_filter(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "skill-a", category="devops")
            _make_skill(tmp_path, "skill-b", category="mlops")
            raw = skills_list(category="devops")
        result = json.loads(raw)
        assert result["count"] == 1
        assert result["skills"][0]["name"] == "skill-a"


# ---------------------------------------------------------------------------
# skill_view
# ---------------------------------------------------------------------------


class TestSkillView:
    def test_view_existing_skill(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "my-skill")
            raw = skill_view("my-skill")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["name"] == "my-skill"
        assert "Step 1" in result["content"]

    def test_view_nonexistent_skill(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "other-skill")
            raw = skill_view("nonexistent")
        result = json.loads(raw)
        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert "available_skills" in result

    def test_view_reference_file(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "my-skill")
            refs_dir = skill_dir / "references"
            refs_dir.mkdir()
            (refs_dir / "api.md").write_text("# API Docs\nEndpoint info.")
            raw = skill_view("my-skill", file_path="references/api.md")
        result = json.loads(raw)
        assert result["success"] is True
        assert "Endpoint info" in result["content"]

    def test_view_nonexistent_file(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "my-skill")
            raw = skill_view("my-skill", file_path="references/nope.md")
        result = json.loads(raw)
        assert result["success"] is False

    def test_view_shows_linked_files(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skill_dir = _make_skill(tmp_path, "my-skill")
            refs_dir = skill_dir / "references"
            refs_dir.mkdir()
            (refs_dir / "guide.md").write_text("guide content")
            raw = skill_view("my-skill")
        result = json.loads(raw)
        assert result["linked_files"] is not None
        assert "references" in result["linked_files"]

    def test_view_tags_from_metadata(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "tagged", frontmatter_extra="metadata:\n  hermes:\n    tags: [fine-tuning, llm]\n")
            raw = skill_view("tagged")
        result = json.loads(raw)
        assert "fine-tuning" in result["tags"]
        assert "llm" in result["tags"]

    def test_view_nonexistent_skills_dir(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "nope"):
            raw = skill_view("anything")
        result = json.loads(raw)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# skills_categories
# ---------------------------------------------------------------------------


class TestSkillsCategories:
    def test_lists_categories(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "s1", category="devops")
            _make_skill(tmp_path, "s2", category="mlops")
            raw = skills_categories()
        result = json.loads(raw)
        assert result["success"] is True
        names = {c["name"] for c in result["categories"]}
        assert "devops" in names
        assert "mlops" in names

    def test_empty_skills_dir(self, tmp_path):
        skills_dir = tmp_path / "skills"
        with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
            raw = skills_categories()
        result = json.loads(raw)
        assert result["success"] is True
        assert result["categories"] == []
