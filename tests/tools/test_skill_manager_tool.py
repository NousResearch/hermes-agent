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
# Operator-locked policy regions
#
# The self-patcher must never rewrite, drop, reorder, or forge operator-authored
# policy wrapped in <!-- operator-locked -->...<!-- /operator-locked -->. It may
# still edit freely OUTSIDE the markers. Modeled on a real incident: a live
# self-patch silently reversed the positions-optimize --wait-seconds rule.
# ---------------------------------------------------------------------------

from tools.skill_manager_tool import (  # noqa: E402
    _locked_region_violation,
    _extract_locked_regions,
    _count_lock_markers,
    _skill_dir_locked_files,
    _has_lock_marker,
    apply_skill_pending,
    OPERATOR_LOCK_OPEN,
    OPERATOR_LOCK_CLOSE,
)


# A SKILL.md whose execution policy is operator-locked (the real wait-seconds
# rule) with a free-to-edit calibration section below it.
LOCKED_SKILL_CONTENT = """\
---
name: positions-optimize
description: Optimize open positions safely.
---

# Positions Optimize

## Execution policy

<!-- operator-locked -->
- Keep the executor's `--wait-seconds` at 30.
- Treat `--wait-seconds 120` as cron-unsafe.
<!-- /operator-locked -->

## Calibration notes

(none yet)
"""

# The exact reversal a live agent performed in the incident this guards against.
REVERSED_LOCK = (
    "- `--wait-seconds 120` has been observed to work fine.\n"
    "- Only use `--wait-seconds 30` if you hit actual terminal timeouts."
)


def _make_locked_skill(skills_root: Path, content: str = LOCKED_SKILL_CONTENT) -> Path:
    """Author the positions-optimize skill with locked content straight to disk
    (operator out-of-band). create() now refuses to MINT lock markers, so the
    self-patch tool itself can't set this up.
    """
    return _author_locked_file(skills_root / "positions-optimize", "SKILL.md", content)


def _author_locked_file(skill_dir: Path, rel_path: str, body: str) -> Path:
    """Write a locked supporting file straight to disk, simulating an operator
    authoring it out-of-band (git / dashboard). The self-patch tool itself
    refuses to MINT lock markers, so tests can't use _write_file to set this up.
    """
    target = skill_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


class TestLockedRegionViolation:
    """Unit tests for the _locked_region_violation invariant checker."""

    def test_no_locks_anywhere_allowed(self):
        assert _locked_region_violation("plain body", "plain body edited") is None

    def test_edit_outside_lock_allowed(self):
        updated = LOCKED_SKILL_CONTENT.replace("(none yet)", "- Run 489: 120s took 95s.")
        assert _locked_region_violation(LOCKED_SKILL_CONTENT, updated) is None

    def test_modifying_locked_text_refused(self):
        updated = LOCKED_SKILL_CONTENT.replace(
            "- Treat `--wait-seconds 120` as cron-unsafe.", REVERSED_LOCK
        )
        err = _locked_region_violation(LOCKED_SKILL_CONTENT, updated)
        assert err is not None
        assert "operator-locked" in err
        assert "operator-locked" in err

    def test_dropping_locked_region_refused(self):
        # Remove the whole locked block (markers + content).
        region = _extract_locked_regions(LOCKED_SKILL_CONTENT)[0]
        updated = LOCKED_SKILL_CONTENT.replace(region, "- 120 is fine now.")
        assert _locked_region_violation(LOCKED_SKILL_CONTENT, updated) is not None

    def test_stripping_open_marker_refused(self):
        updated = LOCKED_SKILL_CONTENT.replace(OPERATOR_LOCK_OPEN + "\n", "")
        assert _locked_region_violation(LOCKED_SKILL_CONTENT, updated) is not None

    def test_stripping_close_marker_refused(self):
        updated = LOCKED_SKILL_CONTENT.replace("\n" + OPERATOR_LOCK_CLOSE, "")
        assert _locked_region_violation(LOCKED_SKILL_CONTENT, updated) is not None

    def test_whitespace_change_inside_lock_refused(self):
        # Even a trivial reflow inside the region is refused — policy is byte-exact.
        updated = LOCKED_SKILL_CONTENT.replace(
            "- Keep the executor's `--wait-seconds` at 30.",
            "-  Keep the executor's `--wait-seconds` at 30.",
        )
        assert _locked_region_violation(LOCKED_SKILL_CONTENT, updated) is not None

    def test_forging_new_lock_on_unlocked_original_refused(self):
        original = "---\nname: x\ndescription: y\n---\n\nBody.\n"
        updated = original + f"\n{OPERATOR_LOCK_OPEN}\n- my own rule\n{OPERATOR_LOCK_CLOSE}\n"
        err = _locked_region_violation(original, updated)
        assert err is not None
        assert "may only be" in err

    def test_adding_second_lock_while_preserving_first_refused(self):
        updated = LOCKED_SKILL_CONTENT + (
            f"\n{OPERATOR_LOCK_OPEN}\n- forged extra rule\n{OPERATOR_LOCK_CLOSE}\n"
        )
        assert _locked_region_violation(LOCKED_SKILL_CONTENT, updated) is not None

    def test_reordering_two_locks_refused(self):
        two = (
            f"{OPERATOR_LOCK_OPEN}\nA\n{OPERATOR_LOCK_CLOSE}\n"
            f"{OPERATOR_LOCK_OPEN}\nB\n{OPERATOR_LOCK_CLOSE}\n"
        )
        swapped = (
            f"{OPERATOR_LOCK_OPEN}\nB\n{OPERATOR_LOCK_CLOSE}\n"
            f"{OPERATOR_LOCK_OPEN}\nA\n{OPERATOR_LOCK_CLOSE}\n"
        )
        assert _locked_region_violation(two, swapped) is not None

    def test_label_is_included_in_error(self):
        updated = LOCKED_SKILL_CONTENT.replace("at 30.", "at 120.")
        err = _locked_region_violation(LOCKED_SKILL_CONTENT, updated, label="references/policy.md")
        assert "references/policy.md" in err

    def test_labeled_marker_variant_is_protected(self):
        # Operators may annotate the open marker with a reason/ticket.
        labeled = (
            "<!-- operator-locked: keep wait-seconds 30 -->\n"
            "- Keep --wait-seconds at 30.\n"
            "<!-- /operator-locked -->\n"
        )
        assert _extract_locked_regions(labeled), "labeled marker should be recognized"
        reversed_ = labeled.replace("- Keep --wait-seconds at 30.", "- 120 is fine.")
        assert _locked_region_violation(labeled, reversed_) is not None

    def test_count_lock_markers(self):
        assert _count_lock_markers(LOCKED_SKILL_CONTENT) == (1, 1)
        assert _count_lock_markers("no markers") == (0, 0)


class TestOperatorLockEdit:
    """_edit_skill refuses full rewrites that touch a locked region."""

    def test_edit_reversing_locked_rule_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            reversed_content = LOCKED_SKILL_CONTENT.replace(
                "- Treat `--wait-seconds 120` as cron-unsafe.", REVERSED_LOCK
            )
            result = _edit_skill("positions-optimize", reversed_content)
            on_disk = (tmp_path / "positions-optimize" / "SKILL.md").read_text()
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        # File must be untouched — the original rule still stands, reversal absent.
        assert "Treat `--wait-seconds 120` as cron-unsafe." in on_disk
        assert "has been observed to work fine" not in on_disk

    def test_edit_outside_lock_allowed(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            updated = LOCKED_SKILL_CONTENT.replace(
                "(none yet)", "- Run 491: 120s completed, but cron killed it once."
            )
            result = _edit_skill("positions-optimize", updated)
            on_disk = (tmp_path / "positions-optimize" / "SKILL.md").read_text()
        assert result["success"] is True, result
        assert "Run 491" in on_disk
        assert "cron-unsafe" in on_disk  # lock intact

    def test_edit_forging_lock_on_unlocked_skill_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            forged = VALID_SKILL_CONTENT + (
                f"\n{OPERATOR_LOCK_OPEN}\n- my own rule\n{OPERATOR_LOCK_CLOSE}\n"
            )
            result = _edit_skill("my-skill", forged)
        assert result["success"] is False
        assert "may only be" in result["error"]


class TestOperatorLockPatch:
    """_patch_skill refuses find/replace whose result touches a locked region."""

    def test_patch_reversing_locked_rule_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            result = _patch_skill(
                "positions-optimize",
                "- Treat `--wait-seconds 120` as cron-unsafe.",
                "- `--wait-seconds 120` works fine; only use 30 on terminal timeouts.",
            )
            on_disk = (tmp_path / "positions-optimize" / "SKILL.md").read_text()
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "cron-unsafe" in on_disk  # unchanged

    def test_patch_appending_calibration_note_allowed(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            result = _patch_skill(
                "positions-optimize", "(none yet)", "- Run 495: 120s ok in manual run."
            )
            on_disk = (tmp_path / "positions-optimize" / "SKILL.md").read_text()
        assert result["success"] is True, result
        assert "Run 495" in on_disk
        assert "cron-unsafe" in on_disk  # lock intact

    def test_patch_replace_all_touching_lock_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            # `--wait-seconds` occurs inside the locked region; replace_all would
            # rewrite locked text.
            result = _patch_skill(
                "positions-optimize", "`--wait-seconds`", "`--delay`", replace_all=True
            )
            on_disk = (tmp_path / "positions-optimize" / "SKILL.md").read_text()
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "`--delay`" not in on_disk

    def test_patch_supporting_file_lock_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _author_locked_file(
                tmp_path / "my-skill",
                "references/policy.md",
                f"# Policy\n\n{OPERATOR_LOCK_OPEN}\n- never trim more than 5%\n{OPERATOR_LOCK_CLOSE}\n",
            )
            result = _patch_skill(
                "my-skill",
                "- never trim more than 5%",
                "- trimming 50% is fine",
                file_path="references/policy.md",
            )
            on_disk = (tmp_path / "my-skill" / "references" / "policy.md").read_text()
        assert result["success"] is False
        assert "references/policy.md" in result["error"]
        assert "never trim more than 5%" in on_disk


class TestOperatorLockWriteRemoveFile:
    """write_file/remove_file cannot be used to bypass the lock."""

    def test_write_file_overwriting_skill_md_lock_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            reversed_content = LOCKED_SKILL_CONTENT.replace(
                "- Treat `--wait-seconds 120` as cron-unsafe.", REVERSED_LOCK
            )
            result = _write_file("positions-optimize", "SKILL.md", reversed_content)
            on_disk = (tmp_path / "positions-optimize" / "SKILL.md").read_text()
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "cron-unsafe" in on_disk

    def test_write_file_forging_lock_in_new_file_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _write_file(
                "my-skill",
                "references/policy.md",
                f"{OPERATOR_LOCK_OPEN}\n- forged\n{OPERATOR_LOCK_CLOSE}\n",
            )
        assert result["success"] is False
        assert "may only be" in result["error"]
        assert not (tmp_path / "my-skill" / "references" / "policy.md").exists()

    def test_write_file_editing_outside_supporting_lock_allowed(self, tmp_path):
        locked_ref = f"# Policy\n\n{OPERATOR_LOCK_OPEN}\n- cap trim at 5%\n{OPERATOR_LOCK_CLOSE}\n\nNotes: old\n"
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _author_locked_file(tmp_path / "my-skill", "references/policy.md", locked_ref)
            updated = locked_ref.replace("Notes: old", "Notes: updated 2026-06-21")
            result = _write_file("my-skill", "references/policy.md", updated)
            on_disk = (tmp_path / "my-skill" / "references" / "policy.md").read_text()
        assert result["success"] is True, result
        assert "Notes: updated" in on_disk
        assert "cap trim at 5%" in on_disk

    def test_remove_file_with_lock_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _author_locked_file(
                tmp_path / "my-skill",
                "references/policy.md",
                f"{OPERATOR_LOCK_OPEN}\n- cap trim at 5%\n{OPERATOR_LOCK_CLOSE}\n",
            )
            result = _remove_file("my-skill", "references/policy.md")
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert (tmp_path / "my-skill" / "references" / "policy.md").exists()

    def test_remove_unlocked_file_still_works(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _write_file("my-skill", "references/notes.md", "just notes")
            result = _remove_file("my-skill", "references/notes.md")
        assert result["success"] is True, result
        assert not (tmp_path / "my-skill" / "references" / "notes.md").exists()


class TestOperatorLockDelete:
    """_delete_skill refuses to drop a skill that carries locked policy."""

    def test_delete_skill_with_locked_skill_md_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            result = _delete_skill("positions-optimize", absorbed_into="")
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "SKILL.md" in result["error"]
        assert (tmp_path / "positions-optimize" / "SKILL.md").exists()

    def test_delete_skill_with_lock_only_in_supporting_file_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)  # clean SKILL.md
            _author_locked_file(
                tmp_path / "my-skill",
                "references/policy.md",
                f"{OPERATOR_LOCK_OPEN}\n- cap trim at 5%\n{OPERATOR_LOCK_CLOSE}\n",
            )
            result = _delete_skill("my-skill", absorbed_into="")
        assert result["success"] is False
        assert "references/policy.md" in result["error"]
        assert (tmp_path / "my-skill").exists()

    def test_delete_unlocked_skill_still_works(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("my-skill", absorbed_into="")
        assert result["success"] is True, result
        assert not (tmp_path / "my-skill").exists()

    def test_skill_dir_locked_files_lists_all_locked(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            _author_locked_file(
                tmp_path / "positions-optimize",
                "references/policy.md",
                f"{OPERATOR_LOCK_OPEN}\n- x\n{OPERATOR_LOCK_CLOSE}\n",
            )
            skill_dir = tmp_path / "positions-optimize"
            locked = _skill_dir_locked_files(skill_dir)
        assert "SKILL.md" in locked
        assert "references/policy.md" in locked


class TestOperatorLockDispatcher:
    """End-to-end through skill_manage() — the agent-facing entry point."""

    def test_patch_reversal_blocked_via_dispatcher(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            raw = skill_manage(
                action="patch",
                name="positions-optimize",
                old_string="- Treat `--wait-seconds 120` as cron-unsafe.",
                new_string="- 120 is fine.",
            )
            on_disk = (tmp_path / "positions-optimize" / "SKILL.md").read_text()
        result = json.loads(raw)
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "cron-unsafe" in on_disk

    def test_calibration_append_allowed_via_dispatcher(self, tmp_path):
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            raw = skill_manage(
                action="patch",
                name="positions-optimize",
                old_string="(none yet)",
                new_string="- Run 489 logged.",
            )
        result = json.loads(raw)
        assert result["success"] is True, result


# ---------------------------------------------------------------------------
# Hardening from a self-review of the guard: count-check coverage, lone markers,
# create() forging, replace_all discrimination, audit log, approval replay.
# ---------------------------------------------------------------------------


class TestLockedRegionViolationHardening:
    def test_lone_open_marker_smuggle_refused(self):
        # A lone trailing open marker forms NO balanced region, so the region
        # list is unchanged — only the marker-count invariant catches it. Pins
        # _count_lock_markers as load-bearing (not subsumed by region equality).
        updated = LOCKED_SKILL_CONTENT + f"\n{OPERATOR_LOCK_OPEN}\n- smuggled active rule\n"
        assert _extract_locked_regions(updated) == _extract_locked_regions(LOCKED_SKILL_CONTENT)
        assert _count_lock_markers(updated) != _count_lock_markers(LOCKED_SKILL_CONTENT)
        assert _locked_region_violation(LOCKED_SKILL_CONTENT, updated) is not None

    def test_lone_close_marker_appended_refused(self):
        updated = LOCKED_SKILL_CONTENT + f"\n{OPERATOR_LOCK_CLOSE}\n"
        assert _locked_region_violation(LOCKED_SKILL_CONTENT, updated) is not None

    def test_has_lock_marker(self):
        assert _has_lock_marker(LOCKED_SKILL_CONTENT) is True
        assert _has_lock_marker(f"only a close {OPERATOR_LOCK_CLOSE}") is True
        assert _has_lock_marker(f"only an open {OPERATOR_LOCK_OPEN}") is True
        assert _has_lock_marker("no markers at all") is False


class TestOperatorLockCreate:
    """create() must not be a forging hole — no action may mint operator locks."""

    def test_create_forging_lock_refused(self, tmp_path):
        forged = VALID_SKILL_CONTENT + (
            f"\n{OPERATOR_LOCK_OPEN}\n- auto-liquidation approved\n{OPERATOR_LOCK_CLOSE}\n"
        )
        with _skill_dir(tmp_path):
            result = _create_skill("executor-rules", forged)
        assert result["success"] is False
        assert "may only be" in result["error"]
        assert not (tmp_path / "executor-rules").exists()

    def test_create_unlocked_skill_still_works(self, tmp_path):
        with _skill_dir(tmp_path):
            result = _create_skill("plain", VALID_SKILL_CONTENT)
        assert result["success"] is True, result


class TestOperatorLockReplaceAllDiscriminates:
    """replace_all must block when ANY occurrence falls inside a locked region,
    even when other occurrences are legitimately outside it."""

    SKILL = (
        "---\nname: s\ndescription: d\n---\n\n# S\n\n"
        f"{OPERATOR_LOCK_OPEN}\n- cap is FIVE percent\n{OPERATOR_LOCK_CLOSE}\n\n"
        "Notes: FIVE here is just prose.\n"
    )

    def test_replace_all_hitting_inside_and_outside_refused(self, tmp_path):
        with _skill_dir(tmp_path):
            _author_locked_file(tmp_path / "s", "SKILL.md", self.SKILL)
            result = _patch_skill("s", "FIVE", "FIFTY", replace_all=True)
            on_disk = (tmp_path / "s" / "SKILL.md").read_text()
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "FIFTY" not in on_disk  # nothing written

    def test_targeted_replace_of_outside_occurrence_allowed(self, tmp_path):
        with _skill_dir(tmp_path):
            _author_locked_file(tmp_path / "s", "SKILL.md", self.SKILL)
            result = _patch_skill("s", "Notes: FIVE here", "Notes: FIFTY here")
            on_disk = (tmp_path / "s" / "SKILL.md").read_text()
        assert result["success"] is True, result
        assert "cap is FIVE percent" in on_disk  # lock intact
        assert "Notes: FIFTY here" in on_disk


class TestOperatorLockLoneCloseRemoval:
    def test_remove_file_with_lone_close_marker_refused(self, tmp_path):
        # A file carrying even a malformed/half lock (lone close) must not be
        # dropped silently — remove_file gates on _has_lock_marker, not balance.
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _author_locked_file(
                tmp_path / "my-skill", "references/policy.md",
                f"some operator note\n{OPERATOR_LOCK_CLOSE}\n",
            )
            result = _remove_file("my-skill", "references/policy.md")
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert (tmp_path / "my-skill" / "references" / "policy.md").exists()


class TestOperatorLockAuditAndReplay:
    def test_allowed_touch_of_locked_file_is_logged(self, tmp_path, caplog):
        import logging
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            with caplog.at_level(logging.INFO, logger="tools.skill_manager_tool"):
                result = _patch_skill("positions-optimize", "(none yet)", "- Run 489 logged.")
        assert result["success"] is True, result
        msgs = [r.getMessage() for r in caplog.records]
        assert any("operator-locked" in m and "positions-optimize" in m for m in msgs), msgs

    def test_approval_replay_still_enforces_lock(self, tmp_path):
        # apply_skill_pending replays an approved staged write with the approval
        # gate bypassed — the per-action lock guard must STILL fire.
        with _skill_dir(tmp_path):
            _make_locked_skill(tmp_path)
            raw = apply_skill_pending({
                "action": "patch",
                "name": "positions-optimize",
                "old_string": "- Treat `--wait-seconds 120` as cron-unsafe.",
                "new_string": "- 120 is fine.",
            })
            on_disk = (tmp_path / "positions-optimize" / "SKILL.md").read_text()
        result = json.loads(raw)
        assert result["success"] is False
        assert "operator-locked" in result["error"]
        assert "cron-unsafe" in on_disk
