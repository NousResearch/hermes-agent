"""Tests for tools/skill_manager_tool.py — skill creation, editing, and deletion."""

import json
import os
import shutil
import tempfile
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
)
from agent.skill_utils import (
    extract_skill_description,
    parse_frontmatter,
    SKILL_PROMPT_DESC_LIMIT,
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

LONG_DESC_CONTENT = """\
---
name: long-desc
description: Use when deploying multi-region Kubernetes clusters with custom CNI plugins and service mesh.
---

# Long Desc Skill

Step 1.
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

    def test_special_chars_rejected(self):
        err = _validate_name("skill/name")
        assert "Invalid skill name 'skill/name'" in err
        err = _validate_name("skill name")
        assert "Invalid skill name 'skill name'" in err
        err = _validate_name("skill@name")
        assert "Invalid skill name 'skill@name'" in err


class TestValidateCategory:
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
    def test_no_frontmatter(self):
        err = _validate_frontmatter("# Just a heading\nSome content.\n")
        assert err == "SKILL.md must start with YAML frontmatter (---). See existing skills for format."

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

    def test_path_traversal_blocked(self):
        err = _validate_file_path("references/../../../etc/passwd")
        assert err == "Path traversal ('..') is not allowed."


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

    def test_create_duplicate_blocked(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _create_skill("my-skill", VALID_SKILL_CONTENT)
        assert result["success"] is False
        assert "already exists" in result["error"]

    def test_create_rejects_category_traversal(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        with patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]):
            result = _create_skill("my-skill", VALID_SKILL_CONTENT, category="../escape")

        assert result["success"] is False
        assert "Invalid category '../escape'" in result["error"]
        assert not (tmp_path / "escape").exists()


    def test_edit_long_desc_still_allowed_with_preview(self, tmp_path):
        """Edit/patch paths stay permissive so existing over-limit skills
        remain maintainable — they warn via system_prompt_preview instead."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _edit_skill("my-skill", LONG_DESC_CONTENT)
        assert result["success"] is True
        assert "system_prompt_preview" in result
        assert "System prompt will show" in result["system_prompt_preview"]
        fm, _ = parse_frontmatter(LONG_DESC_CONTENT)
        assert extract_skill_description(fm) in result["system_prompt_preview"]


class TestEditSkill:
    def test_edit_existing_skill(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _edit_skill("my-skill", VALID_SKILL_CONTENT_2)
        assert result["success"] is True
        content = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "Updated description" in content


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
    def test_delete_cleans_empty_category_dir(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT, category="devops")
            _delete_skill("my-skill")
        assert not (tmp_path / "devops").exists()


    def test_delete_with_absorbed_into_equals_self_rejected(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("narrow", VALID_SKILL_CONTENT)
            result = _delete_skill("narrow", absorbed_into="narrow")
        assert result["success"] is False
        assert "cannot equal" in result["error"]
        assert (tmp_path / "narrow").exists()

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

    def test_successful_mutations_emit_lifecycle_with_correlation(self, tmp_path):
        with (
            _skill_dir(tmp_path),
            patch("tools.skill_provenance.is_background_review", return_value=False),
            patch("tools.skill_usage.record_created") as record_created,
            patch("tools.skill_usage.bump_patch") as bump_patch,
        ):
            created = json.loads(skill_manage(
                action="create",
                name="test-skill",
                content=VALID_SKILL_CONTENT,
                task_id="task-mutation",
                session_id="session-mutation",
            ))
            patched = json.loads(skill_manage(
                action="patch",
                name="test-skill",
                old_string="Step 1: Do the thing.",
                new_string="Step 1: Do the thing safely.",
                task_id="task-mutation",
                session_id="session-mutation",
            ))
            edited = json.loads(skill_manage(
                action="edit",
                name="test-skill",
                content=VALID_SKILL_CONTENT_2,
                task_id="task-mutation",
                session_id="session-mutation",
            ))

        assert created["success"] is True
        assert patched["success"] is True
        assert edited["success"] is True
        record_created.assert_called_once_with(
            "test-skill",
            agent_created=False,
            task_id="task-mutation",
            session_id="session-mutation",
        )
        assert [call.kwargs for call in bump_patch.call_args_list] == [
            {
                "action": "patch",
                "task_id": "task-mutation",
                "session_id": "session-mutation",
            },
            {
                "action": "edit",
                "task_id": "task-mutation",
                "session_id": "session-mutation",
            },
        ]
        assert all(call.args == ("test-skill",) for call in bump_patch.call_args_list)

    def test_failed_mutations_do_not_emit_lifecycle(self, tmp_path):
        with (
            _skill_dir(tmp_path),
            patch("tools.skill_usage.record_created") as record_created,
            patch("tools.skill_usage.bump_patch") as bump_patch,
        ):
            create_result = json.loads(skill_manage(
                action="create",
                name="test-skill",
            ))
            patch_result = json.loads(skill_manage(
                action="patch",
                name="test-skill",
            ))

        assert create_result["success"] is False
        assert patch_result["success"] is False
        record_created.assert_not_called()
        bump_patch.assert_not_called()


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


    def test_background_review_fails_closed_when_ownership_lookup_errors(self, tmp_path):
        from tools.skill_provenance import (
            BACKGROUND_REVIEW,
            reset_current_write_origin,
            set_current_write_origin,
        )

        with _skill_dir(tmp_path):
            _create_skill("manual-skill", VALID_SKILL_CONTENT)
            token = set_current_write_origin(BACKGROUND_REVIEW)
            try:
                with patch(
                    "tools.skill_usage.load_usage",
                    side_effect=ValueError("corrupt usage data"),
                ):
                    raw = skill_manage(
                        action="patch",
                        name="manual-skill",
                        old_string="Do the thing.",
                        new_string="Changed.",
                    )
            finally:
                reset_current_write_origin(token)

        result = json.loads(raw)
        assert result["success"] is False
        assert "ownership" in result["error"].lower()
        assert "Do the thing." in (
            tmp_path / "manual-skill" / "SKILL.md"
        ).read_text(encoding="utf-8")

class TestBackgroundOwnershipPolicyConsistency:
    """The autonomous write policy must not depend on its own side effects.

    Issue #67140: the ownership guard keyed on ``isinstance(usage_rec, dict)``,
    so a local skill with NO usage record passed. The successful write then
    called ``bump_patch()``, creating a ``created_by: null`` record — and the
    identical write was refused from then on. "Allowed exactly once" is a race
    with our own bookkeeping, not a policy.
    """

    @staticmethod
    def _bg_patch(tmp_path, name, old, new):
        from tools.skill_manager_tool import mark_background_review_skill_read
        from tools.skill_provenance import (
            BACKGROUND_REVIEW,
            reset_current_write_origin,
            set_current_write_origin,
        )

        token = set_current_write_origin(BACKGROUND_REVIEW)
        try:
            mark_background_review_skill_read(tmp_path / name / "SKILL.md")
            return json.loads(skill_manage(
                action="patch", name=name, old_string=old, new_string=new,
            ))
        finally:
            reset_current_write_origin(token)

    def test_repeated_identical_write_gets_the_same_answer(self, tmp_path, monkeypatch):
        """The real #67140 shape: no stubbing of load_usage, so the first write's
        telemetry side effect is live. Both attempts must agree."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes" / "skills").mkdir(parents=True, exist_ok=True)
        with _skill_dir(tmp_path):
            _create_skill("flip-skill", VALID_SKILL_CONTENT)
            first = self._bg_patch(
                tmp_path, "flip-skill", "Do the thing.", "Do the new thing.",
            )
            second = self._bg_patch(
                tmp_path, "flip-skill", "Do the thing.", "Do the new thing.",
            )

        assert first["success"] == second["success"], (
            "autonomous write policy flipped between two identical attempts: "
            f"first={first.get('success')} second={second.get('success')}"
        )
        assert first["success"] is False

    def test_foreground_write_to_unmanaged_skill_still_allowed(self, tmp_path, monkeypatch):
        """Fail-closed applies to AUTONOMOUS writes only. A user-directed
        foreground edit to their own skill must keep working."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        with _skill_dir(tmp_path):
            _create_skill("no-record", VALID_SKILL_CONTENT)
            with patch("tools.skill_usage.load_usage", return_value={}):
                res = json.loads(skill_manage(
                    action="patch", name="no-record",
                    old_string="Do the thing.", new_string="Do the new thing.",
                ))
        assert res["success"] is True

    def test_adopted_skill_becomes_writable_by_autonomous_curation(self, tmp_path, monkeypatch):
        """Adoption is the documented path from refused to allowed."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        with _skill_dir(tmp_path):
            _create_skill("adopt-me", VALID_SKILL_CONTENT)
            with patch("tools.skill_usage.load_usage", return_value={}):
                before = self._bg_patch(
                    tmp_path, "adopt-me", "Do the thing.", "Do the new thing.",
                )
            with patch(
                "tools.skill_usage.load_usage",
                return_value={"adopt-me": {"created_by": "agent"}},
            ), patch(
                "tools.skill_usage.get_record",
                side_effect=lambda n: {"created_by": "agent", "pinned": False},
            ):
                after = self._bg_patch(
                    tmp_path, "adopt-me", "Do the thing.", "Do the new thing.",
                )

        assert before["success"] is False
        assert after["success"] is True, after


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

    Also stubs the ownership check to report every skill as curator-managed.
    The ownership guard runs BEFORE the consolidation / read-before-write
    guards these tests target, and since #67140 a skill with no usage record
    fails closed — so without this, every test in this class would be refused
    by ownership and never reach the guard under test. The real curator only
    ever operates on managed sediment, so "managed" is the correct premise
    here; tests that specifically exercise the ownership guard set their own
    records instead.
    """
    hermes_home = tmp_path / ".hermes"
    skills_root = hermes_home / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    with patch("tools.skill_manager_tool.SKILLS_DIR", skills_root), \
         patch("tools.skills_tool.SKILLS_DIR", skills_root), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_root]), \
         patch("tools.skill_usage._is_curator_managed_record", return_value=True), \
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

def _create_curator_skill(name: str, content: str):
    """Create a skill and record the agent ownership a real curator create has."""
    from tools.skill_usage import mark_agent_created

    result = _create_skill(name, content)
    assert result["success"] is True, result
    mark_agent_created(name)
    return result


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
            _create_curator_skill("active-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("active-skill", absorbed_into="")
        assert result["success"] is False
        assert result.get("_fail_closed") is True
        # Skill must remain active on disk — fail closed, no archive.
        assert (skills_root / "active-skill").exists()


    def test_background_review_support_file_overwrite_requires_that_file_read(self, tmp_path, monkeypatch):
        from tools.skills_tool import skill_view
        from tools.skill_manager_tool import _reset_background_review_read_marks

        _reset_background_review_read_marks()
        with _curator_pass(tmp_path, monkeypatch=monkeypatch):
            _create_curator_skill("reviewed", _skill_content("reviewed"))
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


# ---------------------------------------------------------------------------
# Staging preflight — validate create/edit payloads before they become pending
# ---------------------------------------------------------------------------


@contextmanager
def _approval_home(monkeypatch, enabled):
    """HERMES_HOME on a temp dir with skills.write_approval set to ``enabled``.

    Always writes the flag explicitly: reading the operator's real config would
    make gate-on/gate-off assertions depend on the machine.
    """
    d = tempfile.mkdtemp(prefix="hermes_skill_gate_test_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    import hermes_cli.config as cfg
    config = cfg.load_config()
    config.setdefault("skills", {})["write_approval"] = enabled
    cfg.save_config(config)
    try:
        yield home
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _desc_of_length(n):
    """A one-sentence description padded to exactly ``n`` characters."""
    head = "Use when staging a skill "
    return head + "a" * (n - len(head) - 1) + "."


def _desc_with_quoted_tail(budgeted_len):
    """A plain-YAML description ending in a quote character.

    YAML leaves the quote in the parsed value, but the budget check measures
    ``desc.strip().strip("'\\"")`` — so the parsed value is one char longer
    than the length the limit is actually applied to. Used to pin the reported
    count to the measured one.
    """
    head = "Use when the user says "
    return head + '"' + "a" * (budgeted_len - len(head) - 1) + '"'


def _skill_md(desc, name="budget-skill"):
    return f"---\nname: {name}\ndescription: {desc}\n---\n\n# Budget Skill\n\nStep 1.\n"


def _skill_on_disk(tmp_path, name="budget-skill"):
    """Materialize a valid skill so disk-state checks pass and payload
    validation is what decides the outcome."""
    skill = tmp_path / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        _skill_md(_desc_of_length(SKILL_PROMPT_DESC_LIMIT), name=name), encoding="utf-8"
    )
    return skill


class TestStagedWritePreflight:
    """An invalid write payload must be rejected before it is staged.

    Staging defers the real write — and the validation inside the action
    handlers — to approval replay, so validation that only runs there surfaces
    the failure after the user already reviewed and approved the pending
    record.
    """

    def test_over_budget_description_is_not_staged(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        over = _desc_of_length(SKILL_PROMPT_DESC_LIMIT + 1)
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="create", name="budget-skill", content=_skill_md(over),
            ))
            assert result.get("success") is False, result
            assert result.get("staged") is not True
            assert wa.pending_count(wa.SKILLS) == 0
            # The rejection must name the real count, the limit, and the fix.
            error = result["error"]
            assert f"Description is {SKILL_PROMPT_DESC_LIMIT + 1} chars" in error
            assert f"{SKILL_PROMPT_DESC_LIMIT}-char system-prompt budget" in error
            assert "Move detail into the skill body." in error

    def test_at_budget_description_stages_normally(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        at_limit = _desc_of_length(SKILL_PROMPT_DESC_LIMIT)
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="create", name="budget-skill", content=_skill_md(at_limit),
            ))
            assert result["staged"] is True, result
            assert result["pending_id"]
            assert wa.pending_count(wa.SKILLS) == 1

    def test_edit_over_budget_still_stages(self, tmp_path, monkeypatch):
        """The 60-char rule is create-only, so edit must keep staging —
        otherwise existing over-limit skills become uneditable."""
        from tools import write_approval as wa
        over = _desc_of_length(SKILL_PROMPT_DESC_LIMIT + 40)
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="edit", name="budget-skill", content=_skill_md(over),
            ))
            assert result["staged"] is True, result
            assert wa.pending_count(wa.SKILLS) == 1

    def test_patch_merged_result_is_not_preflighted(self, tmp_path, monkeypatch):
        """A patch's merged result does not exist until replay, so approval-time
        validation stays its only checkpoint — even for a patch that will break
        the frontmatter once applied."""
        from tools import write_approval as wa
        _skill_on_disk(tmp_path)
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="patch", name="budget-skill",
                old_string="name: budget-skill", new_string="",
            ))
            assert result["staged"] is True, result
            assert wa.pending_count(wa.SKILLS) == 1

    def test_patch_on_missing_skill_still_stages(self, tmp_path, monkeypatch):
        """Whether the skill exists is disk state, which can change between
        staging and approval — so it stays a replay-time check."""
        from tools import write_approval as wa
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="patch", name="budget-skill",
                old_string="Step 1.", new_string="Step 2.",
            ))
            assert result["staged"] is True, result
            assert wa.pending_count(wa.SKILLS) == 1

    def test_missing_content_is_not_staged(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(action="create", name="budget-skill"))
            assert result.get("success") is False, result
            assert "content is required for 'create'" in result["error"]
            assert wa.pending_count(wa.SKILLS) == 0

    def test_invalid_category_is_not_staged(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        ok = _desc_of_length(SKILL_PROMPT_DESC_LIMIT)
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="create", name="budget-skill", content=_skill_md(ok),
                category="../escape",
            ))
            assert result.get("success") is False, result
            assert "Invalid category '../escape'" in result["error"]
            assert wa.pending_count(wa.SKILLS) == 0

    def test_malformed_frontmatter_is_not_staged(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="create", name="budget-skill", content="# No frontmatter\n",
            ))
            assert result.get("success") is False, result
            assert "must start with YAML frontmatter" in result["error"]
            assert wa.pending_count(wa.SKILLS) == 0

    def test_gate_off_writes_through_unchanged(self, tmp_path, monkeypatch):
        """With the gate off nothing stages and the create still lands."""
        from tools import write_approval as wa
        ok = _desc_of_length(SKILL_PROMPT_DESC_LIMIT)
        with _approval_home(monkeypatch, False), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="create", name="budget-skill", content=_skill_md(ok),
            ))
            assert result["success"] is True, result
            assert result.get("staged") is not True
            assert wa.pending_count(wa.SKILLS) == 0
        assert (tmp_path / "budget-skill" / "SKILL.md").exists()

    def test_reported_count_is_the_measured_count(self, tmp_path, monkeypatch):
        """The budget is applied to the quote-stripped description, so the
        rejection must report that length — not the raw parsed one."""
        over = _desc_with_quoted_tail(SKILL_PROMPT_DESC_LIMIT + 1)
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="create", name="budget-skill", content=_skill_md(over),
            ))
            assert result.get("success") is False, result
            error = result["error"]
            assert f"Description is {SKILL_PROMPT_DESC_LIMIT + 1} chars" in error
            assert f"Description is {SKILL_PROMPT_DESC_LIMIT + 2} chars" not in error

    def test_patch_params_are_preflighted(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        _skill_on_disk(tmp_path)
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="patch", name="budget-skill", new_string="Step 2.",
            ))
            assert result.get("success") is False, result
            assert "old_string is required for 'patch'" in result["error"]
            assert wa.pending_count(wa.SKILLS) == 0

    def test_patch_file_path_traversal_is_not_staged(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        _skill_on_disk(tmp_path)
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="patch", name="budget-skill", file_path="../escape.md",
                old_string="Step 1.", new_string="Step 2.",
            ))
            assert result.get("success") is False, result
            assert "Path traversal" in result["error"]
            assert wa.pending_count(wa.SKILLS) == 0

    def test_write_file_traversal_is_not_staged(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="write_file", name="budget-skill",
                file_path="../escape.md", file_content="pwned\n",
            ))
            assert result.get("success") is False, result
            assert "Path traversal" in result["error"]
            assert wa.pending_count(wa.SKILLS) == 0

    def test_write_file_over_byte_limit_is_not_staged(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        from tools.skill_manager_tool import MAX_SKILL_FILE_BYTES
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="write_file", name="budget-skill",
                file_path="references/big.md", file_content="a" * (MAX_SKILL_FILE_BYTES + 1),
            ))
            assert result.get("success") is False, result
            assert "1 MiB" in result["error"]
            assert wa.pending_count(wa.SKILLS) == 0

    def test_write_file_missing_content_is_not_staged(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="write_file", name="budget-skill", file_path="references/a.md",
            ))
            assert result.get("success") is False, result
            assert "file_content is required for 'write_file'" in result["error"]
            assert wa.pending_count(wa.SKILLS) == 0

    def test_write_file_empty_content_still_stages(self, tmp_path, monkeypatch):
        """An empty supporting file is a legal write, so the preflight must not
        be stricter than the handler it stands in for."""
        from tools import write_approval as wa
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="write_file", name="budget-skill",
                file_path="references/a.md", file_content="",
            ))
            assert result["staged"] is True, result
            assert wa.pending_count(wa.SKILLS) == 1

    def test_remove_file_traversal_is_not_staged(self, tmp_path, monkeypatch):
        from tools import write_approval as wa
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(
                action="remove_file", name="budget-skill", file_path="../escape.md",
            ))
            assert result.get("success") is False, result
            assert "Path traversal" in result["error"]
            assert wa.pending_count(wa.SKILLS) == 0

    def test_delete_stages_without_preflight(self, tmp_path, monkeypatch):
        """Delete carries no payload beyond the name, whose existence is disk
        state — nothing to preflight."""
        from tools import write_approval as wa
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            result = json.loads(skill_manage(action="delete", name="budget-skill"))
            assert result["staged"] is True, result
            assert wa.pending_count(wa.SKILLS) == 1

    @pytest.mark.parametrize("kwargs", [
        pytest.param(
            {"action": "create", "content": _skill_md(_desc_of_length(SKILL_PROMPT_DESC_LIMIT + 1))},
            id="create-over-budget",
        ),
        pytest.param({"action": "create"}, id="create-missing-content"),
        pytest.param(
            {"action": "create", "content": _skill_md(_desc_of_length(SKILL_PROMPT_DESC_LIMIT)),
             "category": "../escape"},
            id="create-bad-category",
        ),
        pytest.param(
            {"action": "patch", "new_string": "Step 2."}, id="patch-missing-old-string",
        ),
        pytest.param(
            {"action": "patch", "file_path": "../escape.md",
             "old_string": "Step 1.", "new_string": "Step 2."},
            id="patch-traversal",
        ),
        pytest.param(
            {"action": "write_file", "file_path": "../escape.md", "file_content": "x"},
            id="write-file-traversal",
        ),
        pytest.param(
            {"action": "write_file", "file_path": "notes/a.md", "file_content": "x"},
            id="write-file-bad-subdir",
        ),
        pytest.param({"action": "write_file", "file_path": "references/a.md"},
                     id="write-file-missing-content"),
        pytest.param({"action": "remove_file", "file_path": "../escape.md"},
                     id="remove-file-traversal"),
    ])
    def test_gated_and_ungated_rejections_are_identical(self, tmp_path, monkeypatch, kwargs):
        """The preflight must be the gate-off path's validation, not a parallel
        copy of it: the same payload has to produce the same error either way.

        Divergence is the real hazard here — a preflight that is stricter than
        replay rejects writes approval would have accepted, and one that is
        laxer re-introduces the round trip this exists to remove.
        """
        _skill_on_disk(tmp_path)
        with _approval_home(monkeypatch, True), _skill_dir(tmp_path):
            gated = json.loads(skill_manage(name="budget-skill", **kwargs))
        with _approval_home(monkeypatch, False), _skill_dir(tmp_path):
            ungated = json.loads(skill_manage(name="budget-skill", **kwargs))

        assert gated.get("success") is False, gated
        assert gated.get("staged") is not True, gated
        assert ungated.get("success") is False, ungated
        assert gated["error"] == ungated["error"]
