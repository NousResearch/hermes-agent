"""Tests for tools/skill_manager_tool.py — skill creation, editing, and deletion."""

import json
import os
from pathlib import Path
from unittest.mock import patch

from tools.skill_manager_tool import (
    _validate_name,
    _validate_category,
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
    _is_external_skill,
    _copy_on_write_to_local,
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
        assert _validate_name("") == "Skill name is required."

    def test_too_long(self):
        err = _validate_name("a" * (MAX_NAME_LENGTH + 1))
        assert err == f"Skill name exceeds {MAX_NAME_LENGTH} characters."

    def test_uppercase_rejected(self):
        err = _validate_name("MySkill")
        assert "Invalid skill name 'MySkill'" in err

    def test_starts_with_hyphen_rejected(self):
        err = _validate_name("-invalid")
        assert "Invalid skill name '-invalid'" in err

    def test_special_chars_rejected(self):
        err = _validate_name("skill/name")
        assert "Invalid skill name 'skill/name'" in err
        err = _validate_name("skill name")
        assert "Invalid skill name 'skill name'" in err
        err = _validate_name("skill@name")
        assert "Invalid skill name 'skill@name'" in err


class TestValidateCategory:
    def test_valid_categories(self):
        assert _validate_category(None) is None
        assert _validate_category("") is None
        assert _validate_category("devops") is None
        assert _validate_category("mlops-v2") is None

    def test_path_traversal_rejected(self):
        err = _validate_category("../escape")
        assert "Invalid category '../escape'" in err

    def test_absolute_path_rejected(self):
        err = _validate_category("/tmp/escape")
        assert "Invalid category '/tmp/escape'" in err


# ---------------------------------------------------------------------------
# _validate_frontmatter
# ---------------------------------------------------------------------------


class TestValidateFrontmatter:
    def test_valid_content(self):
        assert _validate_frontmatter(VALID_SKILL_CONTENT) is None

    def test_empty_content(self):
        assert _validate_frontmatter("") == "Content cannot be empty."
        assert _validate_frontmatter("   ") == "Content cannot be empty."

    def test_no_frontmatter(self):
        err = _validate_frontmatter("# Just a heading\nSome content.\n")
        assert err == "SKILL.md must start with YAML frontmatter (---). See existing skills for format."

    def test_unclosed_frontmatter(self):
        content = "---\nname: test\ndescription: desc\nBody content.\n"
        assert _validate_frontmatter(content) == "SKILL.md frontmatter is not closed. Ensure you have a closing '---' line."

    def test_missing_name_field(self):
        content = "---\ndescription: desc\n---\n\nBody.\n"
        assert _validate_frontmatter(content) == "Frontmatter must include 'name' field."

    def test_missing_description_field(self):
        content = "---\nname: test\n---\n\nBody.\n"
        assert _validate_frontmatter(content) == "Frontmatter must include 'description' field."

    def test_no_body_after_frontmatter(self):
        content = "---\nname: test\ndescription: desc\n---\n"
        assert _validate_frontmatter(content) == "SKILL.md must have content after the frontmatter (instructions, procedures, etc.)."

    def test_invalid_yaml(self):
        content = "---\n: invalid: yaml: {{{\n---\n\nBody.\n"
        assert "YAML frontmatter parse error" in _validate_frontmatter(content)


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
        assert _validate_file_path("") == "file_path is required."

    def test_path_traversal_blocked(self):
        err = _validate_file_path("references/../../../etc/passwd")
        assert err == "Path traversal ('..') is not allowed."

    def test_disallowed_subdirectory(self):
        err = _validate_file_path("secret/hidden.txt")
        assert "File must be under one of:" in err
        assert "'secret/hidden.txt'" in err

    def test_directory_only_rejected(self):
        err = _validate_file_path("references")
        assert "Provide a file path, not just a directory" in err
        assert "'references/myfile.md'" in err

    def test_root_level_file_rejected(self):
        err = _validate_file_path("malicious.py")
        assert "File must be under one of:" in err
        assert "'malicious.py'" in err


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

    def test_create_rejects_category_traversal(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        with patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir):
            result = _create_skill("my-skill", VALID_SKILL_CONTENT, category="../escape")

        assert result["success"] is False
        assert "Invalid category '../escape'" in result["error"]
        assert not (tmp_path / "escape").exists()

    def test_create_rejects_absolute_category(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        outside = tmp_path / "outside"

        with patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir):
            result = _create_skill("my-skill", VALID_SKILL_CONTENT, category=str(outside))

        assert result["success"] is False
        assert f"Invalid category '{outside}'" in result["error"]
        assert not (outside / "my-skill" / "SKILL.md").exists()


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
        assert "not found" in result["error"].lower() or "could not find" in result["error"].lower()

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
        assert "match" in result["error"].lower()

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


# ---------------------------------------------------------------------------
# Copy-on-write for external (shared / external_dirs) skills
# ---------------------------------------------------------------------------


def _write_external_skill(root: Path, name: str, description: str = "Ext skill") -> Path:
    """Helper: create a skill at root/name/SKILL.md and return the skill dir."""
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nStep 1: original step.\n"
    )
    return d


class TestIsExternalSkill:
    def test_local_skill_is_not_external(self, tmp_path):
        local = tmp_path / "local-skills"
        local.mkdir()
        skill = local / "a"
        skill.mkdir()
        with patch("tools.skill_manager_tool.SKILLS_DIR", local):
            assert _is_external_skill(skill) is False

    def test_sibling_skill_is_external(self, tmp_path):
        local = tmp_path / "local-skills"
        ext = tmp_path / "external"
        local.mkdir()
        ext.mkdir()
        skill = ext / "a"
        skill.mkdir()
        with patch("tools.skill_manager_tool.SKILLS_DIR", local):
            assert _is_external_skill(skill) is True


class TestCopyOnWriteHelper:
    def test_copies_directory_contents(self, tmp_path):
        local = tmp_path / "local"
        ext = tmp_path / "external"
        local.mkdir()
        source = _write_external_skill(ext, "shared-skill")
        (source / "references").mkdir()
        (source / "references" / "notes.md").write_text("notes")
        with patch("tools.skill_manager_tool.SKILLS_DIR", local):
            result = _copy_on_write_to_local(source)
        assert result == local / "shared-skill"
        assert (result / "SKILL.md").read_text() == (source / "SKILL.md").read_text()
        assert (result / "references" / "notes.md").read_text() == "notes"

    def test_raises_when_local_already_exists(self, tmp_path):
        local = tmp_path / "local"
        ext = tmp_path / "external"
        (local / "clash").mkdir(parents=True)
        source = _write_external_skill(ext, "clash")
        with patch("tools.skill_manager_tool.SKILLS_DIR", local):
            try:
                _copy_on_write_to_local(source)
            except RuntimeError as exc:
                assert "already exists" in str(exc)
            else:
                raise AssertionError("expected RuntimeError")


class TestEditExternalSkillCopyOnWrite:
    """Verify _edit_skill copy-on-writes external skills instead of mutating them in place."""

    def _setup(self, tmp_path):
        local = tmp_path / "hermes" / "skills"
        local.mkdir(parents=True)
        ext = tmp_path / "shared-skills"
        ext.mkdir()
        _write_external_skill(ext, "shared-editable", "original desc")
        hermes_home = tmp_path / "hermes"
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {ext}\n"
        )
        return local, ext, hermes_home

    def test_edit_external_skill_creates_local_copy(self, tmp_path):
        local, ext, hermes_home = self._setup(tmp_path)
        new_content = (
            "---\nname: shared-editable\ndescription: edited from profile A\n---\n\n"
            "# shared-editable\n\nStep 1: edited step.\n"
        )
        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", local),
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
        ):
            result = _edit_skill("shared-editable", new_content)
        # Edit succeeded
        assert result["success"] is True
        # A note was attached explaining the copy-on-write
        assert "note" in result
        assert "override" in result["note"].lower() or "profile-local" in result["note"].lower()
        # Local copy was created with the new content
        local_copy = local / "shared-editable" / "SKILL.md"
        assert local_copy.exists()
        assert "edited from profile A" in local_copy.read_text()
        # External skill is untouched
        external_original = ext / "shared-editable" / "SKILL.md"
        assert "original desc" in external_original.read_text()
        # Returned path points at the local copy, not the external
        assert str(local / "shared-editable") == result["path"]

    def test_edit_local_skill_still_writes_in_place(self, tmp_path):
        local, ext, hermes_home = self._setup(tmp_path)
        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", local),
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
        ):
            _create_skill("local-only", VALID_SKILL_CONTENT)
            result = _edit_skill("local-only", VALID_SKILL_CONTENT_2)
        assert result["success"] is True
        # No CoW note for local skills
        assert "note" not in result
        assert "Updated description" in (local / "local-only" / "SKILL.md").read_text()


class TestPatchExternalSkillCopyOnWrite:
    def _setup(self, tmp_path):
        local = tmp_path / "hermes" / "skills"
        local.mkdir(parents=True)
        ext = tmp_path / "shared-skills"
        ext.mkdir()
        _write_external_skill(ext, "shared-patchable")
        hermes_home = tmp_path / "hermes"
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {ext}\n"
        )
        return local, ext, hermes_home

    def test_patch_external_skill_creates_local_copy(self, tmp_path):
        local, ext, hermes_home = self._setup(tmp_path)
        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", local),
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
        ):
            result = _patch_skill(
                "shared-patchable", "original step", "patched step",
            )
        assert result["success"] is True
        assert "note" in result
        # Local copy has the patched content
        assert "patched step" in (local / "shared-patchable" / "SKILL.md").read_text()
        # External source is unchanged
        assert "original step" in (ext / "shared-patchable" / "SKILL.md").read_text()

    def test_patch_external_skill_rolls_back_copy_on_invalid_match(self, tmp_path):
        local, ext, hermes_home = self._setup(tmp_path)
        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", local),
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
        ):
            result = _patch_skill(
                "shared-patchable", "this does not exist", "whatever",
            )
        assert result["success"] is False
        # The CoW copy should have been rolled back; local dir is empty.
        assert not (local / "shared-patchable").exists()
        # External still untouched
        assert "original step" in (ext / "shared-patchable" / "SKILL.md").read_text()


class TestWriteFileExternalSkillCopyOnWrite:
    def test_write_file_on_external_copies_skill_first(self, tmp_path):
        local = tmp_path / "hermes" / "skills"
        local.mkdir(parents=True)
        ext = tmp_path / "shared-skills"
        ext.mkdir()
        _write_external_skill(ext, "shared-writefile")
        hermes_home = tmp_path / "hermes"
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {ext}\n"
        )
        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", local),
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
        ):
            result = _write_file(
                "shared-writefile", "references/notes.md", "local override notes",
            )
        assert result["success"] is True
        assert "note" in result
        # New file is in the local copy
        assert (local / "shared-writefile" / "references" / "notes.md").read_text() == "local override notes"
        # Original SKILL.md was also copied over as part of the CoW
        assert (local / "shared-writefile" / "SKILL.md").exists()
        # External skill has no new file
        assert not (ext / "shared-writefile" / "references" / "notes.md").exists()


class TestRemoveFileExternalSkillCopyOnWrite:
    def test_remove_file_on_external_copies_then_removes(self, tmp_path):
        local = tmp_path / "hermes" / "skills"
        local.mkdir(parents=True)
        ext = tmp_path / "shared-skills"
        ext.mkdir()
        source = _write_external_skill(ext, "shared-removefile")
        (source / "references").mkdir()
        (source / "references" / "notes.md").write_text("to be removed")
        hermes_home = tmp_path / "hermes"
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {ext}\n"
        )
        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", local),
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
        ):
            result = _remove_file("shared-removefile", "references/notes.md")
        assert result["success"] is True
        assert "note" in result
        # Local copy exists, but the file is gone from the copy
        assert (local / "shared-removefile" / "SKILL.md").exists()
        assert not (local / "shared-removefile" / "references" / "notes.md").exists()
        # External source still has the file
        assert (ext / "shared-removefile" / "references" / "notes.md").exists()


class TestDeleteExternalSkillRejected:
    def test_delete_external_skill_returns_error(self, tmp_path):
        local = tmp_path / "hermes" / "skills"
        local.mkdir(parents=True)
        ext = tmp_path / "shared-skills"
        ext.mkdir()
        _write_external_skill(ext, "shared-undeletable")
        hermes_home = tmp_path / "hermes"
        (hermes_home / "config.yaml").write_text(
            f"skills:\n  external_dirs:\n    - {ext}\n"
        )
        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", local),
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
        ):
            result = _delete_skill("shared-undeletable")
        assert result["success"] is False
        assert "skills.disabled" in result["error"]
        # External skill still exists
        assert (ext / "shared-undeletable" / "SKILL.md").exists()

    def test_delete_local_skill_still_works(self, tmp_path):
        local = tmp_path / "hermes" / "skills"
        local.mkdir(parents=True)
        hermes_home = tmp_path / "hermes"
        with (
            patch("tools.skill_manager_tool.SKILLS_DIR", local),
            patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}),
        ):
            _create_skill("local-goner", VALID_SKILL_CONTENT)
            result = _delete_skill("local-goner")
        assert result["success"] is True
        assert not (local / "local-goner").exists()
