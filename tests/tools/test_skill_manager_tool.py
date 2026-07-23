"""Tests for tools/skill_manager_tool.py — skill creation, editing, and deletion."""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.skill_manager_tool import (
    _validate_name,
    _validate_category,
    _validate_frontmatter,
    _validate_file_path,
    _create_skill,
    _edit_skill,
    _patch_skill,
    _delete_skill,
    _write_file,
    _remove_file,
    skill_manage,
    MAX_NAME_LENGTH,
)


@contextmanager
def _skill_dir(tmp_path):
    """Patch both SKILLS_DIR and get_all_skills_dirs so _find_skill searches
    only the temp directory — not the real ~/.hermes/skills/."""
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield


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

    def test_skill_md_accepted_at_root(self):
        # SKILL.md is the canonical skill file and must be accepted even
        # though it does not live under an allowed subdirectory.
        assert _validate_file_path("SKILL.md") is None

    def test_skill_md_accepted_name_prefixed(self):
        assert _validate_file_path("my-skill/SKILL.md") is None

    def test_skill_md_traversal_still_rejected(self):
        # The SKILL.md exception must not weaken the traversal guard.
        err = _validate_file_path("../SKILL.md")
        assert err == "Path traversal ('..') is not allowed."

    def test_other_root_md_still_rejected(self):
        # Only SKILL.md gets the root-level exception, not arbitrary files.
        err = _validate_file_path("README.md")
        assert "File must be under one of:" in err


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


class TestCreateSkill:
    def test_create_skill(self, tmp_path):
        with _skill_dir(tmp_path):
            result = _create_skill("my-skill", VALID_SKILL_CONTENT)
        assert result["success"] is True
        assert (tmp_path / "my-skill" / "SKILL.md").exists()

    def test_create_with_category(self, tmp_path):
        with _skill_dir(tmp_path):
            result = _create_skill("my-skill", VALID_SKILL_CONTENT, category="devops")
        assert result["success"] is True
        assert (tmp_path / "devops" / "my-skill" / "SKILL.md").exists()
        assert result["category"] == "devops"

    def test_create_duplicate_blocked(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _create_skill("my-skill", VALID_SKILL_CONTENT)
        assert result["success"] is False
        assert "already exists" in result["error"]

    def test_create_invalid_name(self, tmp_path):
        with _skill_dir(tmp_path):
            result = _create_skill("Invalid Name!", VALID_SKILL_CONTENT)
        assert result["success"] is False

    def test_create_invalid_content(self, tmp_path):
        with _skill_dir(tmp_path):
            result = _create_skill("my-skill", "no frontmatter here")
        assert result["success"] is False

    def test_create_rejects_category_traversal(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        with patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]):
            result = _create_skill("my-skill", VALID_SKILL_CONTENT, category="../escape")

        assert result["success"] is False
        assert "Invalid category '../escape'" in result["error"]
        assert not (tmp_path / "escape").exists()

    def test_create_rejects_absolute_category(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        outside = tmp_path / "outside"

        with patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]):
            result = _create_skill("my-skill", VALID_SKILL_CONTENT, category=str(outside))

        assert result["success"] is False
        assert f"Invalid category '{outside}'" in result["error"]
        assert not (outside / "my-skill" / "SKILL.md").exists()


class TestEditSkill:
    def test_edit_existing_skill(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _edit_skill("my-skill", VALID_SKILL_CONTENT_2)
        assert result["success"] is True
        content = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "Updated description" in content

    def test_edit_nonexistent_skill(self, tmp_path):
        with _skill_dir(tmp_path):
            result = _edit_skill("nonexistent", VALID_SKILL_CONTENT)
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_edit_invalid_content_rejected(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _edit_skill("my-skill", "no frontmatter")
        assert result["success"] is False
        # Original content should be preserved
        content = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "A test skill" in content


class TestPatchSkill:
    def test_patch_unique_match(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _patch_skill("my-skill", "Do the thing.", "Do the new thing.")
        assert result["success"] is True
        content = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "Do the new thing." in content

    def test_patch_nonexistent_string(self, tmp_path):
        with _skill_dir(tmp_path):
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
        with _skill_dir(tmp_path):
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
        with _skill_dir(tmp_path):
            _create_skill("my-skill", content)
            result = _patch_skill("my-skill", "word", "replaced", replace_all=True)
        assert result["success"] is True

    def test_patch_supporting_file(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _write_file("my-skill", "references/api.md", "old text here")
            result = _patch_skill("my-skill", "old text", "new text", file_path="references/api.md")
        assert result["success"] is True

    def test_patch_skill_not_found(self, tmp_path):
        with _skill_dir(tmp_path):
            result = _patch_skill("nonexistent", "old", "new")
        assert result["success"] is False

    def test_patch_supporting_file_symlink_escape_blocked(self, tmp_path):
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("old text here")

        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            link = tmp_path / "my-skill" / "references" / "evil.md"
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                link.symlink_to(outside_file)
            except OSError:
                pytest.skip("Symlinks not supported")

            result = _patch_skill("my-skill", "old text", "new text", file_path="references/evil.md")

        assert result["success"] is False
        assert "escapes" in result["error"].lower()
        assert outside_file.read_text() == "old text here"


class TestDeleteSkill:
    def test_delete_existing(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("my-skill")
        assert result["success"] is True
        assert not (tmp_path / "my-skill").exists()

    def test_delete_nonexistent(self, tmp_path):
        with _skill_dir(tmp_path):
            result = _delete_skill("nonexistent")
        assert result["success"] is False

    def test_delete_cleans_empty_category_dir(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT, category="devops")
            _delete_skill("my-skill")
        assert not (tmp_path / "devops").exists()

    def test_delete_with_absorbed_into_valid_target(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("umbrella", VALID_SKILL_CONTENT)
            _create_skill("narrow", VALID_SKILL_CONTENT)
            result = _delete_skill("narrow", absorbed_into="umbrella")
        assert result["success"] is True
        assert "absorbed into 'umbrella'" in result["message"]
        assert not (tmp_path / "narrow").exists()
        assert (tmp_path / "umbrella").exists()

    def test_delete_with_absorbed_into_empty_string_means_pruned(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("stale-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("stale-skill", absorbed_into="")
        assert result["success"] is True
        # Empty absorbed_into is explicit prune — no "absorbed into" suffix in message
        assert "absorbed into" not in result["message"]

    def test_delete_with_absorbed_into_nonexistent_target_rejected(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("narrow", VALID_SKILL_CONTENT)
            result = _delete_skill("narrow", absorbed_into="ghost-umbrella")
        assert result["success"] is False
        assert "does not exist" in result["error"]
        # Skill must NOT have been deleted on validation failure
        assert (tmp_path / "narrow").exists()

    def test_delete_with_absorbed_into_equals_self_rejected(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("narrow", VALID_SKILL_CONTENT)
            result = _delete_skill("narrow", absorbed_into="narrow")
        assert result["success"] is False
        assert "cannot equal" in result["error"]
        assert (tmp_path / "narrow").exists()

    def test_delete_with_absorbed_into_whitespace_only_treated_as_prune(self, tmp_path):
        # Leading/trailing whitespace only: .strip() → "" → pruned path
        with _skill_dir(tmp_path):
            _create_skill("narrow", VALID_SKILL_CONTENT)
            result = _delete_skill("narrow", absorbed_into="   ")
        assert result["success"] is True
        assert "absorbed into" not in result["message"]

    def test_delete_without_absorbed_into_backward_compat(self, tmp_path):
        # Legacy callers that don't pass the arg still work — the curator
        # reconciler falls back to its heuristic+YAML logic for such deletes.
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("my-skill")
        assert result["success"] is True


# ---------------------------------------------------------------------------
# write_file / remove_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_write_reference_file(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _write_file("my-skill", "references/api.md", "# API\nEndpoint docs.")
        assert result["success"] is True
        assert (tmp_path / "my-skill" / "references" / "api.md").exists()

    def test_write_to_nonexistent_skill(self, tmp_path):
        with _skill_dir(tmp_path):
            result = _write_file("nonexistent", "references/doc.md", "content")
        assert result["success"] is False

    def test_write_to_disallowed_path(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _write_file("my-skill", "secret/evil.py", "malicious")
        assert result["success"] is False

    def test_write_symlink_escape_blocked(self, tmp_path):
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            link = tmp_path / "my-skill" / "references" / "escape"
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                link.symlink_to(outside_dir, target_is_directory=True)
            except OSError:
                pytest.skip("Symlinks not supported")

            result = _write_file("my-skill", "references/escape/owned.md", "malicious")

        assert result["success"] is False
        assert "escapes" in result["error"].lower()
        assert not (outside_dir / "owned.md").exists()


class TestRemoveFile:
    def test_remove_existing_file(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _write_file("my-skill", "references/api.md", "content")
            result = _remove_file("my-skill", "references/api.md")
        assert result["success"] is True
        assert not (tmp_path / "my-skill" / "references" / "api.md").exists()

    def test_remove_nonexistent_file(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _remove_file("my-skill", "references/nope.md")
        assert result["success"] is False

    def test_remove_symlink_escape_blocked(self, tmp_path):
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "keep.txt"
        outside_file.write_text("content")

        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            link = tmp_path / "my-skill" / "references" / "escape"
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                link.symlink_to(outside_dir, target_is_directory=True)
            except OSError:
                pytest.skip("Symlinks not supported")

            result = _remove_file("my-skill", "references/escape/keep.txt")

        assert result["success"] is False
        assert "escapes" in result["error"].lower()
        assert outside_file.exists()


# ---------------------------------------------------------------------------
# skill_manage dispatcher
# ---------------------------------------------------------------------------


class TestSkillManageDispatcher:
    def test_unknown_action(self, tmp_path):
        with _skill_dir(tmp_path):
            raw = skill_manage(action="explode", name="test")
        result = json.loads(raw)
        assert result["success"] is False
        assert "Unknown action" in result["error"]

    def test_create_without_content(self, tmp_path):
        with _skill_dir(tmp_path):
            raw = skill_manage(action="create", name="test")
        result = json.loads(raw)
        assert result["success"] is False
        assert "content" in result["error"].lower()

    def test_patch_without_old_string(self, tmp_path):
        with _skill_dir(tmp_path):
            raw = skill_manage(action="patch", name="test")
        result = json.loads(raw)
        assert result["success"] is False

    def test_full_create_via_dispatcher(self, tmp_path):
        """Foreground create does NOT mark the skill as agent-created.

        Skills created by user-directed foreground turns belong to the user;
        only the background self-improvement review fork should mark its
        own sediment as agent-created (so the curator can later consolidate
        or prune it).
        """
        with _skill_dir(tmp_path):
            raw = skill_manage(action="create", name="test-skill", content=VALID_SKILL_CONTENT)
            from tools.skill_usage import load_usage
            usage = load_usage()
        result = json.loads(raw)
        assert result["success"] is True
        # No provenance marker on a foreground create — record either missing
        # entirely (telemetry best-effort) or present with created_by unset.
        rec = usage.get("test-skill") or {}
        assert rec.get("created_by") in {None, "", False}

    def test_create_from_background_review_marks_agent_created(self, tmp_path):
        """Background-review fork creates ARE marked as agent-created."""
        from tools.skill_provenance import set_current_write_origin, BACKGROUND_REVIEW
        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            with _skill_dir(tmp_path):
                raw = skill_manage(
                    action="create", name="review-sediment", content=VALID_SKILL_CONTENT
                )
                from tools.skill_usage import load_usage
                usage = load_usage()
        finally:
            from tools.skill_provenance import reset_current_write_origin
            reset_current_write_origin(token)
        result = json.loads(raw)
        assert result["success"] is True
        assert usage["review-sediment"]["created_by"] == "agent"

    def test_delete_via_dispatcher_threads_absorbed_into(self, tmp_path):
        # Dispatcher must plumb absorbed_into through to _delete_skill so the
        # validation + message suffix paths are exercised end-to-end.
        with _skill_dir(tmp_path):
            skill_manage(action="create", name="umbrella", content=VALID_SKILL_CONTENT)
            skill_manage(action="create", name="narrow", content=VALID_SKILL_CONTENT)
            raw = skill_manage(action="delete", name="narrow", absorbed_into="umbrella")
        result = json.loads(raw)
        assert result["success"] is True
        assert "absorbed into 'umbrella'" in result["message"]

    def test_delete_via_dispatcher_rejects_missing_absorbed_target(self, tmp_path):
        with _skill_dir(tmp_path):
            skill_manage(action="create", name="narrow", content=VALID_SKILL_CONTENT)
            raw = skill_manage(action="delete", name="narrow", absorbed_into="ghost")
        result = json.loads(raw)
        assert result["success"] is False
        assert "does not exist" in result["error"]

    def test_background_review_delete_refuses_bundled_even_with_absorbed_into(self, tmp_path):
        from tools.skill_provenance import (
            BACKGROUND_REVIEW,
            reset_current_write_origin,
            set_current_write_origin,
        )

        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            with _skill_dir(tmp_path), \
                 patch("tools.skill_usage.is_protected_builtin", return_value=False), \
                 patch("tools.skill_usage.is_hub_installed", return_value=False), \
                 patch("tools.skill_usage.is_bundled",
                       side_effect=lambda skill_name: skill_name == "bundled"):
                skill_manage(action="create", name="umbrella", content=VALID_SKILL_CONTENT)
                skill_manage(action="create", name="bundled", content=VALID_SKILL_CONTENT)
                raw = skill_manage(
                    action="delete",
                    name="bundled",
                    absorbed_into="umbrella",
                )
        finally:
            reset_current_write_origin(token)

        result = json.loads(raw)
        assert result["success"] is False
        assert "bundled" in result["error"].lower()
        assert (tmp_path / "bundled" / "SKILL.md").exists()


class TestSecurityScanGate:
    """_security_scan_skill is gated by skills.guard_agent_created config flag."""

    def test_scan_noop_when_flag_off(self, tmp_path):
        """Default config (flag off) short-circuits before running scan_skill."""
        from tools.skill_manager_tool import _security_scan_skill

        with patch("tools.skill_manager_tool._guard_agent_created_enabled", return_value=False), \
             patch("tools.skill_manager_tool.scan_skill") as mock_scan:
            result = _security_scan_skill(tmp_path)

        assert result is None
        mock_scan.assert_not_called()  # scan never ran

    def test_scan_runs_when_flag_on(self, tmp_path):
        """When flag is on, scan_skill is invoked and its verdict is honored."""
        from tools.skill_manager_tool import _security_scan_skill
        from tools.skills_guard import ScanResult

        # Fake a safe scan result — caller should return None (allow)
        fake_result = ScanResult(
            skill_name="test",
            source="agent-created",
            trust_level="agent-created",
            verdict="safe",
            findings=[],
            summary="ok",
        )
        with patch("tools.skill_manager_tool._guard_agent_created_enabled", return_value=True), \
             patch("tools.skill_manager_tool.scan_skill", return_value=fake_result) as mock_scan:
            result = _security_scan_skill(tmp_path)

        assert result is None
        mock_scan.assert_called_once()

    def test_scan_blocks_dangerous_when_flag_on(self, tmp_path):
        """Dangerous verdict + flag on → returns an error string for the agent."""
        from tools.skill_manager_tool import _security_scan_skill
        from tools.skills_guard import ScanResult, Finding

        finding = Finding(
            pattern_id="test", severity="critical", category="exfiltration",
            file="SKILL.md", line=1, match="curl $TOKEN", description="test",
        )
        fake_result = ScanResult(
            skill_name="test",
            source="agent-created",
            trust_level="agent-created",
            verdict="dangerous",
            findings=[finding],
            summary="dangerous",
        )
        with patch("tools.skill_manager_tool._guard_agent_created_enabled", return_value=True), \
             patch("tools.skill_manager_tool.scan_skill", return_value=fake_result):
            result = _security_scan_skill(tmp_path)

        assert result is not None
        assert "Security scan blocked" in result

    def test_guard_flag_reads_config_default_false(self):
        """_guard_agent_created_enabled returns False when config doesn't set it."""
        from tools.skill_manager_tool import _guard_agent_created_enabled

        with patch("hermes_cli.config.load_config", return_value={"skills": {}}):
            assert _guard_agent_created_enabled() is False

    def test_guard_flag_reads_config_when_set(self):
        """_guard_agent_created_enabled returns True when user explicitly enables."""
        from tools.skill_manager_tool import _guard_agent_created_enabled

        with patch("hermes_cli.config.load_config",
                   return_value={"skills": {"guard_agent_created": True}}):
            assert _guard_agent_created_enabled() is True

    def test_guard_flag_handles_config_error(self):
        """If load_config raises, _guard_agent_created_enabled defaults to False (fail-safe off)."""
        from tools.skill_manager_tool import _guard_agent_created_enabled

        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            assert _guard_agent_created_enabled() is False

    def test_guard_flag_quoted_false_stays_disabled(self):
        """Quoted 'false' from YAML edits must not enable the guard."""
        from tools.skill_manager_tool import _guard_agent_created_enabled

        for quoted in ("false", "False", "0", "no", "off"):
            with patch("hermes_cli.config.load_config",
                       return_value={"skills": {"guard_agent_created": quoted}}):
                assert _guard_agent_created_enabled() is False, \
                    f"guard_agent_created={quoted!r} must coerce to False"

    def test_guard_flag_quoted_true_enables(self):
        """Quoted truthy strings must enable the guard."""
        from tools.skill_manager_tool import _guard_agent_created_enabled

        for quoted in ("true", "True", "1", "yes", "on"):
            with patch("hermes_cli.config.load_config",
                       return_value={"skills": {"guard_agent_created": quoted}}):
                assert _guard_agent_created_enabled() is True, \
                    f"guard_agent_created={quoted!r} must coerce to True"


# ---------------------------------------------------------------------------
# External skills directories (skills.external_dirs) — mutations in place
# ---------------------------------------------------------------------------


@contextmanager
def _two_roots(local_dir: Path, external_dir: Path):
    """Patch the skill manager so local SKILLS_DIR = local_dir and
    get_all_skills_dirs() returns [local_dir, external_dir] in order."""
    with patch("tools.skill_manager_tool.SKILLS_DIR", local_dir), \
         patch("agent.skill_utils.get_all_skills_dirs",
               return_value=[local_dir, external_dir]):
        yield


def _write_external_skill(external_dir: Path, name: str = "ext-skill") -> Path:
    skill_dir = external_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: An external skill.\n---\n\n"
        "# External\n\nBody with OLD_MARKER here.\n"
    )
    return skill_dir


class TestExternalSkillMutations:
    """Verify skill_manage can patch/edit/write/remove/delete skills that live
    under skills.external_dirs — in place, without duplicating to local.

    Regression for issues #4759 and #4381: the read-only gate used to refuse
    with 'Skill X is in an external directory and cannot be modified', which
    caused agents to create duplicate copies in ~/.hermes/skills/ as a
    workaround.
    """

    def test_patch_external_skill_writes_in_place(self, tmp_path):
        local = tmp_path / "local"
        external = tmp_path / "vault"
        local.mkdir(); external.mkdir()
        skill_dir = _write_external_skill(external)

        with _two_roots(local, external):
            result = _patch_skill("ext-skill", "OLD_MARKER", "NEW_MARKER")

        assert result["success"] is True, result
        assert "NEW_MARKER" in (skill_dir / "SKILL.md").read_text()
        # No duplicate in local
        assert not (local / "ext-skill").exists()

    def test_edit_external_skill_writes_in_place(self, tmp_path):
        local = tmp_path / "local"
        external = tmp_path / "vault"
        local.mkdir(); external.mkdir()
        skill_dir = _write_external_skill(external)

        new_content = (
            "---\nname: ext-skill\ndescription: Rewritten.\n---\n\n"
            "# Rewritten\n\nBrand new body.\n"
        )
        with _two_roots(local, external):
            result = _edit_skill("ext-skill", new_content)

        assert result["success"] is True, result
        assert "Brand new body" in (skill_dir / "SKILL.md").read_text()
        assert not (local / "ext-skill").exists()

    def test_write_file_on_external_skill(self, tmp_path):
        local = tmp_path / "local"
        external = tmp_path / "vault"
        local.mkdir(); external.mkdir()
        skill_dir = _write_external_skill(external)

        with _two_roots(local, external):
            result = _write_file("ext-skill", "references/notes.md", "# Notes\n")

        assert result["success"] is True, result
        assert (skill_dir / "references" / "notes.md").read_text() == "# Notes\n"
        assert not (local / "ext-skill").exists()

    def test_remove_file_on_external_skill(self, tmp_path):
        local = tmp_path / "local"
        external = tmp_path / "vault"
        local.mkdir(); external.mkdir()
        skill_dir = _write_external_skill(external)
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "notes.md").write_text("# Notes\n")

        with _two_roots(local, external):
            result = _remove_file("ext-skill", "references/notes.md")

        assert result["success"] is True, result
        assert not (skill_dir / "references" / "notes.md").exists()

    def test_delete_external_skill_removes_skill_not_root(self, tmp_path):
        local = tmp_path / "local"
        external = tmp_path / "vault"
        local.mkdir(); external.mkdir()
        skill_dir = _write_external_skill(external)

        with _two_roots(local, external):
            result = _delete_skill("ext-skill")

        assert result["success"] is True, result
        assert not skill_dir.exists()
        # The external root must NOT be rmdir'd, even when empty after deletion
        assert external.exists() and external.is_dir()

    def test_delete_external_skill_cleans_empty_category(self, tmp_path):
        """When a skill lives under external/<category>/<name>, deleting the
        last skill in the category should rmdir the empty category dir but
        stop at the external root."""
        local = tmp_path / "local"
        external = tmp_path / "vault"
        local.mkdir(); external.mkdir()
        cat_dir = external / "team"
        cat_dir.mkdir()
        skill_dir = cat_dir / "ext-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: ext-skill\ndescription: An external skill.\n---\n\n"
            "# External\n\nBody.\n"
        )

        with _two_roots(local, external):
            result = _delete_skill("ext-skill")

        assert result["success"] is True, result
        assert not skill_dir.exists()
        assert not cat_dir.exists()  # empty category cleaned up
        assert external.exists()     # but never the external root

    def test_create_still_writes_to_local_root(self, tmp_path):
        """Creating a new skill always lands in local SKILLS_DIR, never
        external_dirs — create is unchanged by this PR."""
        local = tmp_path / "local"
        external = tmp_path / "vault"
        local.mkdir(); external.mkdir()

        with _two_roots(local, external):
            result = _create_skill("fresh-skill", VALID_SKILL_CONTENT.replace(
                "name: test-skill", "name: fresh-skill"))

        assert result["success"] is True, result
        assert (local / "fresh-skill" / "SKILL.md").exists()
        assert not (external / "fresh-skill").exists()

    def test_background_review_refuses_to_patch_external_skill(self, tmp_path):
        """Autonomous curator runs treat skills.external_dirs as read-only."""
        from tools.skill_provenance import (
            BACKGROUND_REVIEW,
            reset_current_write_origin,
            set_current_write_origin,
        )

        local = tmp_path / "local"
        external = tmp_path / "vault"
        local.mkdir(); external.mkdir()
        skill_dir = _write_external_skill(external)

        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            with _two_roots(local, external), patch(
                "agent.skill_utils.get_external_skills_dirs",
                return_value=[external.resolve()],
            ):
                raw = skill_manage(
                    action="patch",
                    name="ext-skill",
                    old_string="OLD_MARKER",
                    new_string="NEW_MARKER",
                )
        finally:
            reset_current_write_origin(token)

        result = json.loads(raw)
        assert result["success"] is False
        assert "external" in result["error"].lower()
        assert "OLD_MARKER" in (skill_dir / "SKILL.md").read_text()
        assert "NEW_MARKER" not in (skill_dir / "SKILL.md").read_text()

    def test_background_review_refuses_to_patch_pinned_skill(self, tmp_path):
        """#25839: the autonomous review fork respects pin like the curator
        does — a pinned skill is off-limits to background maintenance, even
        for patch/edit (which a foreground user-directed call is allowed to
        perform). Without a user in the loop there is no one to consent."""
        from tools.skill_provenance import (
            BACKGROUND_REVIEW,
            reset_current_write_origin,
            set_current_write_origin,
        )

        def _fake_get_record(skill_name):
            return {"pinned": True} if skill_name == "my-skill" else {"pinned": False}

        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            token = set_current_write_origin(BACKGROUND_REVIEW)
            try:
                with patch("tools.skill_usage.get_record", side_effect=_fake_get_record):
                    raw = skill_manage(
                        action="patch",
                        name="my-skill",
                        old_string="Do the thing.",
                        new_string="Do the new thing.",
                    )
            finally:
                reset_current_write_origin(token)

        result = json.loads(raw)
        assert result["success"] is False
        assert "pinned" in result["error"].lower()

    def test_background_review_unpinned_skill_not_blocked_by_pin_guard(self, tmp_path):
        """The pin guard must not over-block: an unpinned agent-owned skill is
        still writable by the review fork."""
        from tools.skill_provenance import (
            BACKGROUND_REVIEW,
            reset_current_write_origin,
            set_current_write_origin,
        )

        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            token = set_current_write_origin(BACKGROUND_REVIEW)
            try:
                from tools.skill_manager_tool import mark_background_review_skill_read

                mark_background_review_skill_read(tmp_path / "my-skill" / "SKILL.md")
                with patch(
                    "tools.skill_usage.get_record",
                    side_effect=lambda n: {"pinned": False},
                ):
                    raw = skill_manage(
                        action="patch",
                        name="my-skill",
                        old_string="Do the thing.",
                        new_string="Do the new thing.",
                    )
            finally:
                reset_current_write_origin(token)

        result = json.loads(raw)
        assert result["success"] is True

    def test_background_review_refuses_manually_authored_skill(self, tmp_path):
        """The curator must not archive/edit skills the user placed manually
        (created_by=None). Only agent-created skills are eligible for
        autonomous curation."""
        from tools.skill_provenance import (
            BACKGROUND_REVIEW,
            reset_current_write_origin,
            set_current_write_origin,
        )

        with _skill_dir(tmp_path):
            _create_skill("manual-skill", VALID_SKILL_CONTENT)
            token = set_current_write_origin(BACKGROUND_REVIEW)
            try:
                from tools.skill_manager_tool import mark_background_review_skill_read

                mark_background_review_skill_read(tmp_path / "manual-skill" / "SKILL.md")
                with patch(
                    "tools.skill_usage.load_usage",
                    return_value={"manual-skill": {"created_by": None, "use_count": 50}},
                ), patch(
                    "tools.skill_usage.get_record",
                    side_effect=lambda n: {"created_by": None, "use_count": 50} if n == "manual-skill" else {},
                ):
                    raw = skill_manage(
                        action="delete",
                        name="manual-skill",
                    )
            finally:
                reset_current_write_origin(token)

        result = json.loads(raw)
        assert result["success"] is False
        assert "manually authored" in result["error"].lower()

    def test_background_review_allows_agent_created_skill(self, tmp_path):
        """Agent-created skills (created_by='agent') are NOT blocked by the
        manual-skill guard — they remain eligible for autonomous curation."""
        from tools.skill_provenance import (
            BACKGROUND_REVIEW,
            reset_current_write_origin,
            set_current_write_origin,
        )

        with _skill_dir(tmp_path):
            _create_skill("agent-skill", VALID_SKILL_CONTENT)
            token = set_current_write_origin(BACKGROUND_REVIEW)
            try:
                from tools.skill_manager_tool import mark_background_review_skill_read

                mark_background_review_skill_read(tmp_path / "agent-skill" / "SKILL.md")
                with patch(
                    "tools.skill_usage.load_usage",
                    return_value={"agent-skill": {"created_by": "agent", "use_count": 5}},
                ), patch(
                    "tools.skill_usage.get_record",
                    side_effect=lambda n: {"created_by": "agent", "use_count": 5} if n == "agent-skill" else {},
                ), patch(
                    "tools.skill_usage.is_curation_eligible", return_value=True,
                ), patch(
                    "tools.skill_usage.archive_skill", return_value=(True, "archived"),
                ):
                    raw = skill_manage(
                        action="delete",
                        name="agent-skill",
                        absorbed_into="umbrella",
                    )
            finally:
                reset_current_write_origin(token)

        result = json.loads(raw)
        # Should not be blocked by the manual-skill guard (may be blocked by
        # the consolidation-delete guard if absorbed_into is empty, but the
        # manual-skill guard must not fire).
        assert "manually authored" not in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# Pinned-skill guard — skill_manage refuses only `delete` on pinned skills.
# Patches and edits go through so pinned skills can still evolve as pitfalls
# come up. The user unpins via `hermes curator unpin <name>` to delete.
# ---------------------------------------------------------------------------

class TestPinnedGuard:
    """Delete is refused on pinned skills; patch/edit/write_file/remove_file are allowed."""

    @staticmethod
    def _pin(name: str):
        """Return a patch context that marks *name* as pinned in skill_usage."""
        def _fake_get_record(skill_name, _name=name):
            return {"pinned": True} if skill_name == _name else {"pinned": False}
        return patch("tools.skill_usage.get_record", side_effect=_fake_get_record)

    def test_edit_allowed_when_pinned(self, tmp_path):
        """Pin does NOT block edit — agent can still improve pinned skills."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            with self._pin("my-skill"):
                result = _edit_skill("my-skill", VALID_SKILL_CONTENT_2)
        assert result["success"] is True, result
        # Content updated
        content = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "A test skill" not in content

    def test_patch_allowed_when_pinned(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            with self._pin("my-skill"):
                result = _patch_skill("my-skill", "Do the thing.", "Do the new thing.")
        assert result["success"] is True, result
        content = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "Do the new thing." in content

    def test_patch_supporting_file_allowed_when_pinned(self, tmp_path):
        """Supporting-file patches also go through on pinned skills."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _write_file("my-skill", "references/api.md", "original")
            with self._pin("my-skill"):
                result = _patch_skill(
                    "my-skill", "original", "modified",
                    file_path="references/api.md",
                )
        assert result["success"] is True, result
        assert (tmp_path / "my-skill" / "references" / "api.md").read_text() == "modified"

    def test_delete_refuses_pinned(self, tmp_path):
        """Delete is the one action pin still blocks — it's the irrecoverable one."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            with self._pin("my-skill"):
                result = _delete_skill("my-skill")
        assert result["success"] is False
        assert "pinned" in result["error"].lower()
        assert "cannot be deleted" in result["error"]
        assert "hermes curator unpin my-skill" in result["error"]
        # Skill still exists
        assert (tmp_path / "my-skill" / "SKILL.md").exists()

    def test_write_file_allowed_when_pinned(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            with self._pin("my-skill"):
                result = _write_file("my-skill", "references/api.md", "content")
        assert result["success"] is True, result
        assert (tmp_path / "my-skill" / "references" / "api.md").read_text() == "content"

    def test_remove_file_allowed_when_pinned(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _write_file("my-skill", "references/api.md", "content")
            with self._pin("my-skill"):
                result = _remove_file("my-skill", "references/api.md")
        assert result["success"] is True, result
        assert not (tmp_path / "my-skill" / "references" / "api.md").exists()

    def test_unpinned_skills_still_editable(self, tmp_path):
        """Sanity check: the guard doesn't fire for unpinned skills on delete.

        Only the specifically-pinned skill is refused from delete; a sibling
        skill must still be freely deletable.
        """
        with _skill_dir(tmp_path):
            _create_skill("pinned-one", VALID_SKILL_CONTENT)
            _create_skill("free-one", VALID_SKILL_CONTENT)
            with self._pin("pinned-one"):
                blocked = _delete_skill("pinned-one")
                allowed = _delete_skill("free-one")
        assert blocked["success"] is False
        assert allowed["success"] is True

    def test_broken_sidecar_fails_open(self, tmp_path):
        """If skill_usage.get_record raises, we allow delete through.

        Rationale: a corrupted telemetry file shouldn't lock the agent out
        of skills it would otherwise be allowed to touch.
        """
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            with patch("tools.skill_usage.get_record",
                       side_effect=RuntimeError("sidecar broken")):
                result = _delete_skill("my-skill")
        assert result["success"] is True


# ---------------------------------------------------------------------------
# _delete_skill — recursive-delete safety (port of Kilo Code #11240)
# ---------------------------------------------------------------------------


class TestDeleteSkillRmtreeGuard:
    """Defense-in-depth before ``shutil.rmtree`` in ``_delete_skill``.

    Mirrors the Kilo Code #11227 fix: never let a recursive skill delete
    escape the skills tree, target a skills root, or follow a symlink.
    """

    def test_normal_delete_still_works(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("good-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("good-skill", absorbed_into="")
        assert result["success"] is True, result
        assert not (tmp_path / "good-skill").exists()

    def test_symlinked_skill_dir_refused(self, tmp_path):
        """A skill dir that is a symlink must not be rmtree'd — rmtree would
        otherwise follow it and delete the link target's contents."""
        victim = tmp_path.parent / "precious_victim"
        victim.mkdir()
        (victim / "important.txt").write_text("DO NOT DELETE")
        skills = tmp_path / "skills"
        skills.mkdir()
        evil = skills / "evil-skill"
        evil.symlink_to(victim, target_is_directory=True)
        try:
            with patch("tools.skill_manager_tool.SKILLS_DIR", skills), \
                 patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills]), \
                 patch("tools.skill_manager_tool._find_skill",
                       return_value={"path": evil}):
                result = _delete_skill("evil-skill", absorbed_into="")
            assert result["success"] is False
            assert "symlink" in result["error"].lower()
            assert (victim / "important.txt").exists()
        finally:
            import shutil as _sh
            _sh.rmtree(victim, ignore_errors=True)

    def test_skills_root_itself_refused(self, tmp_path):
        """If discovery ever hands back the skills root, refuse — rmtree would
        wipe every installed skill."""
        with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]), \
             patch("tools.skill_manager_tool._find_skill",
                   return_value={"path": tmp_path}):
            result = _delete_skill("root-attack", absorbed_into="")
        assert result["success"] is False
        assert "skills root" in result["error"].lower()
        assert tmp_path.exists()

    def test_out_of_tree_path_refused(self, tmp_path):
        """A path that resolves outside every known skills root is refused."""
        skills = tmp_path / "skills"
        skills.mkdir()
        outside = tmp_path / "outside_skill"
        outside.mkdir()
        (outside / "SKILL.md").write_text("x")
        with patch("tools.skill_manager_tool.SKILLS_DIR", skills), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills]), \
             patch("tools.skill_manager_tool._find_skill",
                   return_value={"path": outside}):
            result = _delete_skill("outside", absorbed_into="")
        assert result["success"] is False
        assert "skills root" in result["error"].lower()
        assert outside.exists()


# ---------------------------------------------------------------------------
# Curator consolidation-pass fail-closed delete guard (#29912)
# ---------------------------------------------------------------------------


@contextmanager
def _curator_pass(tmp_path, *, monkeypatch):
    """Run the body as the curator/background-review fork.

    Points HERMES_HOME at ``tmp_path/.hermes`` so skill_usage's archive path
    (``get_hermes_home()``) resolves into the same tree the skill manager
    searches, and flips ``is_background_review()`` → True so the consolidation
    guard fires.
    """
    hermes_home = tmp_path / ".hermes"
    skills_root = hermes_home / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    with patch("tools.skill_manager_tool.SKILLS_DIR", skills_root), \
         patch("tools.skills_tool.SKILLS_DIR", skills_root), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_root]), \
         patch("tools.skill_provenance.is_background_review", return_value=True):
        yield skills_root


def _skill_content(name: str) -> str:
    """SKILL.md whose frontmatter ``name:`` matches the directory name.

    ``skill_usage._find_skill_dir`` (used by ``archive_skill``) resolves a
    skill by its frontmatter ``name:`` field, so archive-path tests must keep
    the two in sync.
    """
    return (
        "---\n"
        f"name: {name}\n"
        "description: A test skill for unit testing.\n"
        "---\n\n"
        f"# {name}\n\n"
        "Step 1: Do the thing.\n"
    )


class TestCuratorConsolidationDeleteGuard:
    """The curator's LLM consolidation pass must fail CLOSED on unverified
    deletes — it may only archive a skill it absorbed into an umbrella.

    Reproduces #29912: the pass archived clusters of active skills with zero
    verified consolidations (``consolidated_this_run == 0``) because a bare
    prune from the LLM pass was accepted. With the guard, a delete without a
    valid ``absorbed_into`` is refused and the skill stays active; a verified
    consolidation is archived RECOVERABLY (not rmtree'd).
    """

    def test_bare_prune_during_curator_pass_refused(self, tmp_path, monkeypatch):
        with _curator_pass(tmp_path, monkeypatch=monkeypatch) as skills_root:
            _create_skill("active-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("active-skill", absorbed_into="")
        assert result["success"] is False
        assert result.get("_fail_closed") is True
        # Skill must remain active on disk — fail closed, no archive.
        assert (skills_root / "active-skill").exists()

    def test_omitted_absorbed_into_during_curator_pass_refused(self, tmp_path, monkeypatch):
        with _curator_pass(tmp_path, monkeypatch=monkeypatch) as skills_root:
            _create_skill("active-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("active-skill")  # absorbed_into omitted
        assert result["success"] is False
        assert result.get("_fail_closed") is True
        assert (skills_root / "active-skill").exists()

    def test_whitespace_absorbed_into_during_curator_pass_refused(self, tmp_path, monkeypatch):
        with _curator_pass(tmp_path, monkeypatch=monkeypatch) as skills_root:
            _create_skill("active-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("active-skill", absorbed_into="   ")
        assert result["success"] is False
        assert result.get("_fail_closed") is True
        assert (skills_root / "active-skill").exists()

    def test_verified_consolidation_archives_recoverably(self, tmp_path, monkeypatch):
        with _curator_pass(tmp_path, monkeypatch=monkeypatch) as skills_root:
            _create_skill("umbrella", _skill_content("umbrella"))
            _create_skill("narrow", _skill_content("narrow"))
            result = _delete_skill("narrow", absorbed_into="umbrella")
        assert result["success"] is True, result
        assert result.get("_archived") is True
        assert "absorbed into 'umbrella'" in result["message"]
        # Recoverable: moved to .archive/, NOT permanently rmtree'd.
        assert not (skills_root / "narrow").exists()
        assert (skills_root / ".archive" / "narrow").exists()
        # Umbrella untouched.
        assert (skills_root / "umbrella").exists()

    def test_consolidation_into_missing_umbrella_still_rejected(self, tmp_path, monkeypatch):
        # The pre-existing target-existence check fires before the recoverable
        # archive — a hallucinated umbrella is refused and the skill stays put.
        with _curator_pass(tmp_path, monkeypatch=monkeypatch) as skills_root:
            _create_skill("narrow", VALID_SKILL_CONTENT)
            result = _delete_skill("narrow", absorbed_into="ghost-umbrella")
        assert result["success"] is False
        assert "does not exist" in result["error"]
        assert (skills_root / "narrow").exists()

    def test_foreground_bare_prune_unaffected(self, tmp_path):
        # Outside the curator pass (default foreground origin), a bare prune
        # still hard-deletes — the guard is curator-scoped only.
        with _skill_dir(tmp_path):
            _create_skill("user-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("user-skill", absorbed_into="")
        assert result["success"] is True
        assert result.get("_fail_closed") is None
        assert result.get("_archived") is None
        assert not (tmp_path / "user-skill").exists()

    def test_dispatcher_preserves_usage_record_on_curator_archive(self, tmp_path, monkeypatch):
        # skill_manage(delete) post-action telemetry must NOT forget a
        # recoverable curator archive — the record persists as archived so
        # `hermes curator restore` can bring it back.
        from tools import skill_usage
        with _curator_pass(tmp_path, monkeypatch=monkeypatch):
            _create_skill("umbrella", _skill_content("umbrella"))
            _create_skill("narrow", _skill_content("narrow"))
            skill_usage.mark_agent_created("narrow")
            raw = skill_manage("delete", "narrow", absorbed_into="umbrella")
            result = json.loads(raw)
            assert result["success"] is True, result
            rec = skill_usage.get_record("narrow")
        # Record kept (not forgotten) and marked archived.
        assert rec.get("state") == skill_usage.STATE_ARCHIVED

    def test_background_review_patch_requires_skill_view_first(self, tmp_path, monkeypatch):
        from tools.skills_tool import skill_view
        from tools.skill_manager_tool import _reset_background_review_read_marks

        _reset_background_review_read_marks()
        with _curator_pass(tmp_path, monkeypatch=monkeypatch):
            _create_skill("reviewed", _skill_content("reviewed"))

            blocked = json.loads(skill_manage(
                action="patch",
                name="reviewed",
                old_string="Step 1: Do the thing.",
                new_string="Step 1: Do the thing safely.",
            ))
            assert blocked["success"] is False
            assert blocked.get("_read_before_write_required") is True

            viewed = json.loads(skill_view("reviewed"))
            assert viewed["success"] is True

            allowed = json.loads(skill_manage(
                action="patch",
                name="reviewed",
                old_string="Step 1: Do the thing.",
                new_string="Step 1: Do the thing safely.",
            ))
            assert allowed["success"] is True, allowed

        _reset_background_review_read_marks()

    def test_background_review_support_file_overwrite_requires_that_file_read(self, tmp_path, monkeypatch):
        from tools.skills_tool import skill_view
        from tools.skill_manager_tool import _reset_background_review_read_marks

        _reset_background_review_read_marks()
        with _curator_pass(tmp_path, monkeypatch=monkeypatch):
            _create_skill("reviewed", _skill_content("reviewed"))
            ref = tmp_path / ".hermes" / "skills" / "reviewed" / "references"
            ref.mkdir()
            (ref / "workflow.md").write_text("old workflow\n", encoding="utf-8")

            # Reading SKILL.md does not authorize overwriting a linked file.
            assert json.loads(skill_view("reviewed"))["success"] is True
            blocked = json.loads(skill_manage(
                action="write_file",
                name="reviewed",
                file_path="references/workflow.md",
                file_content="new workflow\n",
            ))
            assert blocked["success"] is False
            assert blocked.get("_read_before_write_required") is True

            assert json.loads(skill_view("reviewed", "references/workflow.md"))["success"] is True
            allowed = json.loads(skill_manage(
                action="write_file",
                name="reviewed",
                file_path="references/workflow.md",
                file_content="new workflow\n",
            ))
            assert allowed["success"] is True, allowed

        _reset_background_review_read_marks()
