"""Tests for tools/skills_tool.py — skill discovery and viewing."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.skills_tool as skills_tool_module
from tools.skills_tool import (
    _get_required_environment_variables,
    _parse_frontmatter,
    _parse_tags,
    _get_category_from_path,
    _find_all_skills,
    skill_matches_platform,
    skills_list,
    skill_view,
    MAX_DESCRIPTION_LENGTH,
)


def _make_skill(
    skills_dir, name, frontmatter_extra="", body="Step 1: Do the thing.", category=None
):
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
        content = (
            "---\nname: test\nmetadata:\n  hermes:\n    tags: [a, b]\n---\n\nBody.\n"
        )
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
        result = _parse_tags("\"tag1\", 'tag2'")
        assert "tag1" in result
        assert "tag2" in result

    def test_filters_empty_items(self):
        assert _parse_tags([None, "", "valid"]) == ["valid"]


class TestRequiredEnvironmentVariablesNormalization:
    def test_parses_new_required_environment_variables_metadata(self):
        frontmatter = {
            "required_environment_variables": [
                {
                    "name": "TENOR_API_KEY",
                    "prompt": "Tenor API key",
                    "help": "Get a key from https://developers.google.com/tenor",
                    "required_for": "full functionality",
                }
            ]
        }

        result = _get_required_environment_variables(frontmatter)

        assert result == [
            {
                "name": "TENOR_API_KEY",
                "prompt": "Tenor API key",
                "help": "Get a key from https://developers.google.com/tenor",
                "required_for": "full functionality",
            }
        ]

    def test_normalizes_legacy_prerequisites_env_vars(self):
        frontmatter = {"prerequisites": {"env_vars": ["TENOR_API_KEY"]}}

        result = _get_required_environment_variables(frontmatter)

        assert result == [
            {
                "name": "TENOR_API_KEY",
                "prompt": "Enter value for TENOR_API_KEY",
            }
        ]

    def test_empty_env_file_value_is_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("FILLED_KEY", "value")
        monkeypatch.setenv("EMPTY_HOST_KEY", "")

        from tools.skills_tool import _is_env_var_persisted

        assert _is_env_var_persisted("EMPTY_FILE_KEY", {"EMPTY_FILE_KEY": ""}) is False
        assert (
            _is_env_var_persisted("FILLED_FILE_KEY", {"FILLED_FILE_KEY": "x"}) is True
        )
        assert _is_env_var_persisted("EMPTY_HOST_KEY", {}) is False
        assert _is_env_var_persisted("FILLED_KEY", {}) is True


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
        (skill_dir / "SKILL.md").write_text(
            "---\nname: no-desc\n---\n\n# Heading\n\nFirst paragraph.\n"
        )
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skills = _find_all_skills()
        assert skills[0]["description"] == "First paragraph."

    def test_long_description_truncated(self, tmp_path):
        long_desc = "x" * (MAX_DESCRIPTION_LENGTH + 100)
        skill_dir = tmp_path / "long-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: long\ndescription: {long_desc}\n---\n\nBody.\n"
        )
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            skills = _find_all_skills()
        assert len(skills[0]["description"]) <= MAX_DESCRIPTION_LENGTH

    def test_skips_git_directories(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "real-skill")
            git_dir = tmp_path / ".git" / "fake-skill"
            git_dir.mkdir(parents=True)
            (git_dir / "SKILL.md").write_text(
                "---\nname: fake\ndescription: x\n---\n\nBody.\n"
            )
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
            _make_skill(
                tmp_path,
                "tagged",
                frontmatter_extra="metadata:\n  hermes:\n    tags: [fine-tuning, llm]\n",
            )
            raw = skill_view("tagged")
        result = json.loads(raw)
        assert "fine-tuning" in result["tags"]
        assert "llm" in result["tags"]

    def test_view_nonexistent_skills_dir(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path / "nope"):
            raw = skill_view("anything")
        result = json.loads(raw)
        assert result["success"] is False

    def test_view_disabled_skill_blocked(self, tmp_path):
        """Disabled skills should not be viewable via skill_view."""
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch(
                "tools.skills_tool._is_skill_disabled",
                return_value=True,
            ),
        ):
            _make_skill(tmp_path, "hidden-skill")
            raw = skill_view("hidden-skill")
        result = json.loads(raw)
        assert result["success"] is False
        assert "disabled" in result["error"].lower()

    def test_view_enabled_skill_allowed(self, tmp_path):
        """Non-disabled skills should be viewable normally."""
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch(
                "tools.skills_tool._is_skill_disabled",
                return_value=False,
            ),
        ):
            _make_skill(tmp_path, "active-skill")
            raw = skill_view("active-skill")
        result = json.loads(raw)
        assert result["success"] is True


class TestSkillViewSecureSetupOnLoad:
    def test_requests_missing_required_env_and_continues(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TENOR_API_KEY", raising=False)
        calls = []

        def fake_secret_callback(var_name, prompt, metadata=None):
            calls.append(
                {
                    "var_name": var_name,
                    "prompt": prompt,
                    "metadata": metadata,
                }
            )
            os.environ[var_name] = "stored-in-test"
            return {
                "success": True,
                "stored_as": var_name,
                "validated": False,
                "skipped": False,
            }

        monkeypatch.setattr(
            skills_tool_module,
            "_secret_capture_callback",
            fake_secret_callback,
            raising=False,
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gif-search",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                    "    help: Get a key from https://developers.google.com/tenor\n"
                    "    required_for: full functionality\n"
                ),
            )
            raw = skill_view("gif-search")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["name"] == "gif-search"
        assert calls == [
            {
                "var_name": "TENOR_API_KEY",
                "prompt": "Tenor API key",
                "metadata": {
                    "skill_name": "gif-search",
                    "help": "Get a key from https://developers.google.com/tenor",
                    "required_for": "full functionality",
                },
            }
        ]
        assert result["required_environment_variables"][0]["name"] == "TENOR_API_KEY"
        assert result["setup_skipped"] is False

    def test_allows_skipping_secure_setup_and_still_loads(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TENOR_API_KEY", raising=False)

        def fake_secret_callback(var_name, prompt, metadata=None):
            return {
                "success": True,
                "stored_as": var_name,
                "validated": False,
                "skipped": True,
            }

        monkeypatch.setattr(
            skills_tool_module,
            "_secret_capture_callback",
            fake_secret_callback,
            raising=False,
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gif-search",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                ),
            )
            raw = skill_view("gif-search")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_skipped"] is True
        assert result["content"].startswith("---")

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
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": ["macos"]}) is True

    def test_macos_on_linux(self):
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["macos"]}) is False

    def test_linux_on_linux(self):
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["linux"]}) is True

    def test_linux_on_darwin(self):
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": ["linux"]}) is False

    def test_windows_on_win32(self):
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert skill_matches_platform({"platforms": ["windows"]}) is True

    def test_windows_on_linux(self):
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["windows"]}) is False

    def test_multi_platform_match(self):
        """Skills listing multiple platforms should match any of them."""
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": ["macos", "linux"]}) is True
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["macos", "linux"]}) is True
            mock_sys.platform = "win32"
            assert skill_matches_platform({"platforms": ["macos", "linux"]}) is False

    def test_string_instead_of_list(self):
        """A single string value should be treated as a one-element list."""
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": "macos"}) is True
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": "macos"}) is False

    def test_case_insensitive(self):
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert skill_matches_platform({"platforms": ["MacOS"]}) is True
            assert skill_matches_platform({"platforms": ["MACOS"]}) is True

    def test_unknown_platform_no_match(self):
        with patch("agent.skill_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert skill_matches_platform({"platforms": ["freebsd"]}) is False


# ---------------------------------------------------------------------------
# _find_all_skills — platform filtering integration
# ---------------------------------------------------------------------------


class TestFindAllSkillsPlatformFiltering:
    """Test that _find_all_skills respects the platforms field."""

    def test_excludes_incompatible_platform(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("agent.skill_utils.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            _make_skill(tmp_path, "universal-skill")
            _make_skill(tmp_path, "mac-only", frontmatter_extra="platforms: [macos]\n")
            skills = _find_all_skills()
        names = {s["name"] for s in skills}
        assert "universal-skill" in names
        assert "mac-only" not in names

    def test_includes_matching_platform(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("agent.skill_utils.sys") as mock_sys,
        ):
            mock_sys.platform = "darwin"
            _make_skill(tmp_path, "mac-only", frontmatter_extra="platforms: [macos]\n")
            skills = _find_all_skills()
        names = {s["name"] for s in skills}
        assert "mac-only" in names

    def test_no_platforms_always_included(self, tmp_path):
        """Skills without platforms field should appear on any platform."""
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("agent.skill_utils.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            _make_skill(tmp_path, "generic-skill")
            skills = _find_all_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "generic-skill"

    def test_multi_platform_skill(self, tmp_path):
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch("agent.skill_utils.sys") as mock_sys,
        ):
            _make_skill(
                tmp_path, "cross-plat", frontmatter_extra="platforms: [macos, linux]\n"
            )
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
# _find_all_skills
# ---------------------------------------------------------------------------


class TestFindAllSkillsSecureSetup:
    def test_skills_with_missing_env_vars_remain_listed(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_API_KEY_XYZ", raising=False)
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "needs-key",
                frontmatter_extra="prerequisites:\n  env_vars: [NONEXISTENT_API_KEY_XYZ]\n",
            )
            skills = _find_all_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "needs-key"
        assert "readiness_status" not in skills[0]
        assert "missing_prerequisites" not in skills[0]

    def test_skills_with_met_prereqs_have_same_listing_shape(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("MY_PRESENT_KEY", "val")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "has-key",
                frontmatter_extra="prerequisites:\n  env_vars: [MY_PRESENT_KEY]\n",
            )
            skills = _find_all_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "has-key"
        assert "readiness_status" not in skills[0]

    def test_skills_without_prereqs_have_same_listing_shape(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "simple-skill")
            skills = _find_all_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "simple-skill"
        assert "readiness_status" not in skills[0]

    def test_skill_listing_does_not_probe_backend_for_env_vars(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "skill-a",
                frontmatter_extra="prerequisites:\n  env_vars: [A_KEY]\n",
            )
            _make_skill(
                tmp_path,
                "skill-b",
                frontmatter_extra="prerequisites:\n  env_vars: [B_KEY]\n",
            )
            skills = _find_all_skills()

        assert len(skills) == 2
        assert {skill["name"] for skill in skills} == {"skill-a", "skill-b"}


class TestSkillViewPrerequisites:
    def test_legacy_prerequisites_expose_required_env_setup_metadata(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("MISSING_KEY_XYZ", raising=False)
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gated-skill",
                frontmatter_extra="prerequisites:\n  env_vars: [MISSING_KEY_XYZ]\n",
            )
            raw = skill_view("gated-skill")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is True
        assert result["missing_required_environment_variables"] == ["MISSING_KEY_XYZ"]
        assert result["required_environment_variables"] == [
            {
                "name": "MISSING_KEY_XYZ",
                "prompt": "Enter value for MISSING_KEY_XYZ",
            }
        ]

    def test_no_setup_needed_when_legacy_prereqs_are_met(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRESENT_KEY", "value")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "ready-skill",
                frontmatter_extra="prerequisites:\n  env_vars: [PRESENT_KEY]\n",
            )
            raw = skill_view("ready-skill")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["missing_required_environment_variables"] == []

    def test_remote_backend_treats_persisted_env_as_available(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "remote-ready",
                frontmatter_extra="prerequisites:\n  env_vars: [PERSISTED_REMOTE_KEY]\n",
            )
            from hermes_cli.config import save_env_value

            save_env_value("PERSISTED_REMOTE_KEY", "persisted-value")
            monkeypatch.delenv("PERSISTED_REMOTE_KEY", raising=False)
            raw = skill_view("remote-ready")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["missing_required_environment_variables"] == []
        assert result["readiness_status"] == "available"

    def test_no_setup_metadata_when_no_required_envs(self, tmp_path):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "plain-skill")
            raw = skill_view("plain-skill")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["required_environment_variables"] == []

    def test_skill_view_treats_backend_only_env_as_setup_needed(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "backend-ready",
                frontmatter_extra="prerequisites:\n  env_vars: [BACKEND_ONLY_KEY]\n",
            )
            raw = skill_view("backend-ready")
        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is True
        assert result["missing_required_environment_variables"] == ["BACKEND_ONLY_KEY"]

    def test_local_env_missing_keeps_setup_needed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.delenv("SHELL_ONLY_KEY", raising=False)

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "shell-ready",
                frontmatter_extra="prerequisites:\n  env_vars: [SHELL_ONLY_KEY]\n",
            )
            raw = skill_view("shell-ready")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is True
        assert result["missing_required_environment_variables"] == ["SHELL_ONLY_KEY"]
        assert result["readiness_status"] == "setup_needed"

    @pytest.mark.parametrize(
        "backend",
        ["ssh", "daytona", "docker", "singularity", "modal"],
    )
    def test_remote_backend_becomes_available_after_local_secret_capture(
        self, tmp_path, monkeypatch, backend
    ):
        monkeypatch.setenv("TERMINAL_ENV", backend)
        monkeypatch.delenv("TENOR_API_KEY", raising=False)
        calls = []

        def fake_secret_callback(var_name, prompt, metadata=None):
            calls.append((var_name, prompt, metadata))
            os.environ[var_name] = "captured-locally"
            return {
                "success": True,
                "stored_as": var_name,
                "validated": False,
                "skipped": False,
            }

        monkeypatch.setattr(
            skills_tool_module,
            "_secret_capture_callback",
            fake_secret_callback,
            raising=False,
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gif-search",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                ),
            )
            raw = skill_view("gif-search")

        result = json.loads(raw)
        assert result["success"] is True
        assert len(calls) == 1
        assert result["setup_needed"] is False
        assert result["readiness_status"] == "available"
        assert result["missing_required_environment_variables"] == []
        assert "setup_note" not in result

    def test_skill_view_surfaces_skill_read_errors(self, tmp_path, monkeypatch):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(tmp_path, "broken-skill")
            skill_md = tmp_path / "broken-skill" / "SKILL.md"
            original_read_text = Path.read_text

            def fake_read_text(path_obj, *args, **kwargs):
                if path_obj == skill_md:
                    raise UnicodeDecodeError(
                        "utf-8", b"\xff", 0, 1, "invalid start byte"
                    )
                return original_read_text(path_obj, *args, **kwargs)

            monkeypatch.setattr(Path, "read_text", fake_read_text)
            raw = skill_view("broken-skill")

        result = json.loads(raw)
        assert result["success"] is False
        assert "Failed to read skill 'broken-skill'" in result["error"]

    def test_legacy_flat_md_skill_preserves_frontmatter_metadata(self, tmp_path):
        flat_skill = tmp_path / "legacy-skill.md"
        flat_skill.write_text(
            """\
---
name: legacy-flat
description: Legacy flat skill.
metadata:
  hermes:
    tags: [legacy, flat]
required_environment_variables:
  - name: LEGACY_KEY
    prompt: Legacy key
---

# Legacy Flat

Do the legacy thing.
""",
            encoding="utf-8",
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            raw = skill_view("legacy-skill")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["name"] == "legacy-flat"
        assert result["description"] == "Legacy flat skill."
        assert result["tags"] == ["legacy", "flat"]
        assert result["required_environment_variables"] == [
            {"name": "LEGACY_KEY", "prompt": "Legacy key"}
        ]

    def test_successful_secret_capture_reloads_empty_env_placeholder(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.delenv("TENOR_API_KEY", raising=False)

        def fake_secret_callback(var_name, prompt, metadata=None):
            from hermes_cli.config import save_env_value

            save_env_value(var_name, "captured-value")
            return {
                "success": True,
                "stored_as": var_name,
                "validated": False,
                "skipped": False,
            }

        monkeypatch.setattr(
            skills_tool_module,
            "_secret_capture_callback",
            fake_secret_callback,
            raising=False,
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_skill(
                tmp_path,
                "gif-search",
                frontmatter_extra=(
                    "required_environment_variables:\n"
                    "  - name: TENOR_API_KEY\n"
                    "    prompt: Tenor API key\n"
                ),
            )
            from hermes_cli.config import save_env_value

            save_env_value("TENOR_API_KEY", "")
            raw = skill_view("gif-search")

        result = json.loads(raw)
        assert result["success"] is True
        assert result["setup_needed"] is False
        assert result["missing_required_environment_variables"] == []
        assert result["readiness_status"] == "available"
