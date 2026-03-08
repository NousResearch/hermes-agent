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
    check_skill_prerequisites,
    skill_matches_platform,
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


# ---------------------------------------------------------------------------
# skill_matches_platform
# ---------------------------------------------------------------------------


class TestSkillMatchesPlatform:
    """Tests for the platforms frontmatter field filtering."""

    def test_no_platforms_field_matches_everything(self):
        """Skills without a platforms field should load on any OS."""
        assert skill_matches_platform({}) is True
        assert skill_matches_platform({"name": "foo"}) is True

    def test_empty_platforms_matches_everything(self):
        """Empty platforms list should load on any OS."""
        assert skill_matches_platform({"platforms": []}) is True
        assert skill_matches_platform({"platforms": None}) is True

    def test_macos_on_darwin(self):
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": ["macos"]}) is True

    def test_macos_on_linux(self):
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["macos"]}) is False

    def test_linux_on_linux(self):
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["linux"]}) is True

    def test_linux_on_darwin(self):
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": ["linux"]}) is False

    def test_windows_on_win32(self):
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert skill_matches_platform({"platforms": ["windows"]}) is True

    def test_windows_on_linux(self):
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["windows"]}) is False

    def test_multi_platform_match(self):
        """Skills listing multiple platforms should match any of them."""
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": ["macos", "linux"]}) is True
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["macos", "linux"]}) is True
            mock_sys.platform = "win32"
            assert skill_matches_platform({"platforms": ["macos", "linux"]}) is False

    def test_string_instead_of_list(self):
        """A single string value should be treated as a one-element list."""
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": "macos"}) is True
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": "macos"}) is False

    def test_case_insensitive(self):
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": ["MacOS"]}) is True
            assert skill_matches_platform({"platforms": ["MACOS"]}) is True

    def test_unknown_platform_no_match(self):
        with patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["freebsd"]}) is False


# ---------------------------------------------------------------------------
# _find_all_skills — platform filtering integration
# ---------------------------------------------------------------------------


class TestFindAllSkillsPlatformFiltering:
    """Test that _find_all_skills respects the platforms field."""

    def test_excludes_incompatible_platform(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
             patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "linux"
            _make_skill(tmp_path, "universal-skill")
            _make_skill(tmp_path, "mac-only", frontmatter_extra="platforms: [macos]\n")
            skills = _find_all_skills()
        names = {s["name"] for s in skills}
        assert "universal-skill" in names
        assert "mac-only" not in names

    def test_includes_matching_platform(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
             patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "darwin"
            _make_skill(tmp_path, "mac-only", frontmatter_extra="platforms: [macos]\n")
            skills = _find_all_skills()
        names = {s["name"] for s in skills}
        assert "mac-only" in names

    def test_no_platforms_always_included(self, tmp_path):
        """Skills without platforms field should appear on any platform."""
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
             patch("tools.skills_tool.sys") as mock_sys:
            mock_sys.platform = "win32"
            _make_skill(tmp_path, "generic-skill")
            skills = _find_all_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "generic-skill"

    def test_multi_platform_skill(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path), \
             patch("tools.skills_tool.sys") as mock_sys:
            _make_skill(tmp_path, "cross-plat", frontmatter_extra="platforms: [macos, linux]\n")
            mock_sys.platform = "darwin"
            skills_darwin = _find_all_skills()
            mock_sys.platform = "linux"
            skills_linux = _find_all_skills()
            mock_sys.platform = "win32"
            skills_win = _find_all_skills()
        assert len(skills_darwin) == 1
        assert len(skills_linux) == 1
        assert len(skills_win) == 0


# ---------------------------------------------------------------------------
# check_skill_prerequisites
# ---------------------------------------------------------------------------


class TestCheckSkillPrerequisites:
    def test_no_or_empty_prerequisites(self):
        """No field, empty dict, or non-dict all pass."""
        assert check_skill_prerequisites({})[0] is True
        assert check_skill_prerequisites({"prerequisites": {}})[0] is True
        assert check_skill_prerequisites({"prerequisites": "curl"})[0] is True

    def test_env_var_present_and_missing(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_KEY", "val")
        monkeypatch.delenv("NONEXISTENT_TEST_VAR_XYZ", raising=False)
        assert check_skill_prerequisites({"prerequisites": {"env_vars": ["MY_TEST_KEY"]}})[0] is True
        met, missing = check_skill_prerequisites({"prerequisites": {"env_vars": ["NONEXISTENT_TEST_VAR_XYZ"]}})
        assert met is False
        assert "env $NONEXISTENT_TEST_VAR_XYZ" in missing

    def test_command_present_and_missing(self):
        assert check_skill_prerequisites({"prerequisites": {"commands": ["python3"]}})[0] is True
        met, missing = check_skill_prerequisites({"prerequisites": {"commands": ["nonexistent_binary_xyz_123"]}})
        assert met is False
        assert "command `nonexistent_binary_xyz_123`" in missing

    def test_mixed_env_and_commands(self, monkeypatch):
        monkeypatch.delenv("MISSING_A", raising=False)
        met, missing = check_skill_prerequisites({
            "prerequisites": {
                "env_vars": ["MISSING_A"],
                "commands": ["python3", "nonexistent_cmd_xyz"],
            }
        })
        assert met is False
        assert len(missing) == 2

    def test_string_instead_of_list(self, monkeypatch):
        """YAML scalar (string) should be coerced to a single-element list."""
        monkeypatch.delenv("SOLO_VAR", raising=False)
        assert check_skill_prerequisites({"prerequisites": {"env_vars": "SOLO_VAR"}})[0] is False
        assert check_skill_prerequisites({"prerequisites": {"commands": "nonexistent_cmd_xyz_solo"}})[0] is False


# ---------------------------------------------------------------------------
# _find_all_skills — prerequisites integration
# ---------------------------------------------------------------------------


class TestFindAllSkillsPrerequisites:
    def test_skills_with_unmet_prereqs_flagged(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_API_KEY_XYZ", raising=False)
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path, "needs-key",
                frontmatter_extra="prerequisites:\n  env_vars: [NONEXISTENT_API_KEY_XYZ]\n",
            )
            skills = _find_all_skills()
        assert len(skills) == 1
        assert skills[0]["prerequisites_met"] is False
        assert any("NONEXISTENT_API_KEY_XYZ" in m for m in skills[0]["prerequisites_missing"])

    def test_skills_with_met_prereqs_no_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_PRESENT_KEY", "val")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path, "has-key",
                frontmatter_extra="prerequisites:\n  env_vars: [MY_PRESENT_KEY]\n",
            )
            skills = _find_all_skills()
        assert len(skills) == 1
        assert "prerequisites_met" not in skills[0]

    def test_skills_without_prereqs_no_flag(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "simple-skill")
            skills = _find_all_skills()
        assert len(skills) == 1
        assert "prerequisites_met" not in skills[0]


# ---------------------------------------------------------------------------
# skill_view — prerequisites warnings
# ---------------------------------------------------------------------------


class TestSkillViewPrerequisites:
    def test_warns_on_unmet_prerequisites(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MISSING_KEY_XYZ", raising=False)
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path, "gated-skill",
                frontmatter_extra="prerequisites:\n  env_vars: [MISSING_KEY_XYZ]\n",
            )
            raw = skill_view("gated-skill")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["prerequisites_met"] is False
        assert "MISSING_KEY_XYZ" in result["prerequisites_warning"]

    def test_no_warning_when_prereqs_met(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRESENT_KEY", "value")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path, "ready-skill",
                frontmatter_extra="prerequisites:\n  env_vars: [PRESENT_KEY]\n",
            )
            raw = skill_view("ready-skill")
        result = json.loads(raw)
        assert result["success"] is True
        assert "prerequisites_warning" not in result

    def test_no_warning_when_no_prereqs(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "plain-skill")
            raw = skill_view("plain-skill")
        result = json.loads(raw)
        assert result["success"] is True
        assert "prerequisites_warning" not in result
