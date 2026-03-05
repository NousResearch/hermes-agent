"""Tests for tools/skill_manager_tool.py — skill creation, editing, and deletion."""

import json
from pathlib import Path
from unittest.mock import patch

from tools.skill_manager_tool import (
    _validate_name,
    _validate_frontmatter,
    _validate_file_path,
    _find_skill,
    _resolve_skill_dir,
    _create_skill,
    _edit_skill,
    _patch_skill,
    _delete_skill,
    _write_file,
    _remove_file,
    skill_manage,
    VALID_NAME_RE,
    ALLOWED_SUBDIRS,
    MAX_NAME_LENGTH,
)


VALID_SKILL_CONTENT = """\
---
name: test-skill
description: A test skill for unit testing.
---

# Test Skill

Step 1: Do the thing.
"""

VALID_SKILL_CONTENT_2 = """\
---
name: test-skill
description: Updated description.
---

# Test Skill v2

Step 1: Do the new thing.
"""


# ---------------------------------------------------------------------------
# _validate_name
# ---------------------------------------------------------------------------


class TestValidateName:
    def test_valid_names(self):
        assert _validate_name("my-skill") is None
        assert _validate_name("skill123") is None
        assert _validate_name("my_skill.v2") is None
        assert _validate_name("a") is None

    def test_empty_name(self):
        assert _validate_name("") is not None

    def test_too_long(self):
        assert _validate_name("a" * (MAX_NAME_LENGTH + 1)) is not None

    def test_uppercase_rejected(self):
        assert _validate_name("MySkill") is not None

    def test_starts_with_hyphen_rejected(self):
        assert _validate_name("-invalid") is not None

    def test_special_chars_rejected(self):
        assert _validate_name("skill/name") is not None
        assert _validate_name("skill name") is not None
        assert _validate_name("skill@name") is not None


# ---------------------------------------------------------------------------
# _validate_frontmatter
# ---------------------------------------------------------------------------


class TestValidateFrontmatter:
    def test_valid_content(self):
        assert _validate_frontmatter(VALID_SKILL_CONTENT) is None

    def test_empty_content(self):
        assert _validate_frontmatter("") is not None
        assert _validate_frontmatter("   ") is not None

    def test_no_frontmatter(self):
        err = _validate_frontmatter("# Just a heading\nSome content.\n")
        assert err is not None
        assert "frontmatter" in err.lower()

    def test_unclosed_frontmatter(self):
        content = "---\nname: test\ndescription: desc\nBody content.\n"
        assert _validate_frontmatter(content) is not None

    def test_missing_name_field(self):
        content = "---\ndescription: desc\n---\n\nBody.\n"
        assert _validate_frontmatter(content) is not None

    def test_missing_description_field(self):
        content = "---\nname: test\n---\n\nBody.\n"
        assert _validate_frontmatter(content) is not None

    def test_no_body_after_frontmatter(self):
        content = "---\nname: test\ndescription: desc\n---\n"
        assert _validate_frontmatter(content) is not None

    def test_invalid_yaml(self):
        content = "---\n: invalid: yaml: {{{\n---\n\nBody.\n"
        assert _validate_frontmatter(content) is not None


# ---------------------------------------------------------------------------
# _validate_file_path — path traversal prevention
# ---------------------------------------------------------------------------


class TestValidateFilePath:
    def test_valid_paths(self):
        assert _validate_file_path("references/api.md") is None
        assert _validate_file_path("templates/config.yaml") is None
        assert _validate_file_path("scripts/train.py") is None
        assert _validate_file_path("assets/image.png") is None

    def test_empty_path(self):
        assert _validate_file_path("") is not None

    def test_path_traversal_blocked(self):
        err = _validate_file_path("references/../../../etc/passwd")
        assert err is not None
        assert "traversal" in err.lower()

    def test_disallowed_subdirectory(self):
        err = _validate_file_path("secret/hidden.txt")
        assert err is not None

    def test_directory_only_rejected(self):
        err = _validate_file_path("references")
        assert err is not None

    def test_root_level_file_rejected(self):
        err = _validate_file_path("malicious.py")
        assert err is not None


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


class TestCreateSkill:
    def test_create_skill(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            result = _create_skill("my-skill", VALID_SKILL_CONTENT)
        assert result["success"] is True
        assert (tmp_path / "my-skill" / "SKILL.md").exists()

    def test_create_with_category(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            result = _create_skill("my-skill", VALID_SKILL_CONTENT, category="devops")
        assert result["success"] is True
        assert (tmp_path / "devops" / "my-skill" / "SKILL.md").exists()
        assert result["category"] == "devops"

    def test_create_duplicate_blocked(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _create_skill("my-skill", VALID_SKILL_CONTENT)
        assert result["success"] is False
        assert "already exists" in result["error"]

    def test_create_invalid_name(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            result = _create_skill("Invalid Name!", VALID_SKILL_CONTENT)
        assert result["success"] is False

    def test_create_invalid_content(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            result = _create_skill("my-skill", "no frontmatter here")
        assert result["success"] is False


class TestEditSkill:
    def test_edit_existing_skill(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _edit_skill("my-skill", VALID_SKILL_CONTENT_2)
        assert result["success"] is True
        content = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "Updated description" in content

    def test_edit_nonexistent_skill(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            result = _edit_skill("nonexistent", VALID_SKILL_CONTENT)
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_edit_invalid_content_rejected(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _edit_skill("my-skill", "no frontmatter")
        assert result["success"] is False
        # Original content should be preserved
        content = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "A test skill" in content


class TestPatchSkill:
    def test_patch_unique_match(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _patch_skill("my-skill", "Do the thing.", "Do the new thing.")
        assert result["success"] is True
        content = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "Do the new thing." in content

    def test_patch_nonexistent_string(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _patch_skill("my-skill", "this text does not exist", "replacement")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_patch_ambiguous_match_rejected(self, tmp_path):
        content = """\
---
name: test-skill
description: A test skill.
---

# Test

word word
"""
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", content)
            result = _patch_skill("my-skill", "word", "replaced")
        assert result["success"] is False
        assert "matched" in result["error"]

    def test_patch_replace_all(self, tmp_path):
        content = """\
---
name: test-skill
description: A test skill.
---

# Test

word word
"""
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", content)
            result = _patch_skill("my-skill", "word", "replaced", replace_all=True)
        assert result["success"] is True

    def test_patch_supporting_file(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _write_file("my-skill", "references/api.md", "old text here")
            result = _patch_skill("my-skill", "old text", "new text", file_path="references/api.md")
        assert result["success"] is True

    def test_patch_skill_not_found(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            result = _patch_skill("nonexistent", "old", "new")
        assert result["success"] is False


class TestDeleteSkill:
    def test_delete_existing(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("my-skill")
        assert result["success"] is True
        assert not (tmp_path / "my-skill").exists()

    def test_delete_nonexistent(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            result = _delete_skill("nonexistent")
        assert result["success"] is False

    def test_delete_cleans_empty_category_dir(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT, category="devops")
            _delete_skill("my-skill")
        assert not (tmp_path / "devops").exists()


# ---------------------------------------------------------------------------
# write_file / remove_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_write_reference_file(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _write_file("my-skill", "references/api.md", "# API\nEndpoint docs.")
        assert result["success"] is True
        assert (tmp_path / "my-skill" / "references" / "api.md").exists()

    def test_write_to_nonexistent_skill(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            result = _write_file("nonexistent", "references/doc.md", "content")
        assert result["success"] is False

    def test_write_to_disallowed_path(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _write_file("my-skill", "secret/evil.py", "malicious")
        assert result["success"] is False


class TestRemoveFile:
    def test_remove_existing_file(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _write_file("my-skill", "references/api.md", "content")
            result = _remove_file("my-skill", "references/api.md")
        assert result["success"] is True
        assert not (tmp_path / "my-skill" / "references" / "api.md").exists()

    def test_remove_nonexistent_file(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _remove_file("my-skill", "references/nope.md")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# skill_manage dispatcher
# ---------------------------------------------------------------------------


class TestSkillManageDispatcher:
    def test_unknown_action(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            raw = skill_manage(action="explode", name="test")
        result = json.loads(raw)
        assert result["success"] is False
        assert "Unknown action" in result["error"]

    def test_create_without_content(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            raw = skill_manage(action="create", name="test")
        result = json.loads(raw)
        assert result["success"] is False
        assert "content" in result["error"].lower()

    def test_patch_without_old_string(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            raw = skill_manage(action="patch", name="test")
        result = json.loads(raw)
        assert result["success"] is False

    def test_full_create_via_dispatcher(self, tmp_path):
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path):
            raw = skill_manage(action="create", name="test-skill", content=VALID_SKILL_CONTENT)
        result = json.loads(raw)
        assert result["success"] is True
