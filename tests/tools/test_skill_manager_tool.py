"""Tests for tools/skill_manager_tool.py — skill creation, editing, and deletion."""

import hashlib
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
    _final_skill_mutation_denial,
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
    from agent.self_improvement_decision_context import (
        bind_self_improvement_decision,
        reset_self_improvement_decision,
    )
    from agent.self_improvement_policy import Decision as _Decision

    token = bind_self_improvement_decision(
        _Decision(result="ALLOW", reason="test skill mutation opt-in")
    )
    try:
        with patch.dict("os.environ", {
                "HERMES_DISABLE_SELF_IMPROVEMENT": "0",
                "HERMES_READ_ONLY_SESSION": "0",
             }), \
             patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
            yield
    finally:
        reset_self_improvement_decision(token)


@pytest.fixture
def allow_decision():
    """Install an ALLOW Decision in the typed ContextVar for the test.

    Canonical recipe mirrored from
    ``tests/tools/test_self_improvement_write_boundaries.py::allow_decision``.
    Phase 2 contract: the L2 guards read their authorization state from the
    typed ContextVar, not from env vars at the call site. Tests that exercise
    the ``normal session, ALLOW`` path bind a Decision via
    ``bind_self_improvement_decision`` before invoking the guard, and reset
    it via the returned Token in ``finally`` so no stale ALLOW leaks into
    the next test. Cleanup uses explicit try/finally inside the fixture
    so a failure in the test body still resets the ContextVar.
    """
    from agent.self_improvement_decision_context import (
        bind_self_improvement_decision,
        reset_self_improvement_decision,
    )
    from agent.self_improvement_policy import Decision as _Dec

    token = bind_self_improvement_decision(
        _Dec(result="ALLOW", reason="explicit_test_opt_in_allow_decision_fixture")
    )
    try:
        yield token
    finally:
        try:
            reset_self_improvement_decision(token)
        except Exception:
            # Token reset failures must not mask the original exception;
            # subsequent bind calls overwrite the ContextVar anyway.
            pass


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

    def test_create_skill_surfaces_advisory_lint_findings(self, tmp_path):
        from tools.skill_linter import LintFinding, WARNING

        finding = LintFinding(
            WARNING,
            "test-advisory-rule",
            "deterministic advisory finding from the linter",
        )

        with _skill_dir(tmp_path), \
             patch("tools.skill_linter.lint_skill", return_value=[finding]) as lint_skill:
            result = _create_skill("my-skill", VALID_SKILL_CONTENT)

        skill_md = tmp_path / "my-skill" / "SKILL.md"
        assert result["success"] is True, result
        assert skill_md.exists()
        lint_skill.assert_called_once_with(skill_md)
        assert result["lint_warnings"] == [
            {
                "severity": WARNING,
                "rule": "test-advisory-rule",
                "message": "deterministic advisory finding from the linter",
            }
        ]
        assert result["lint_hint"] == (
            "The skill was created. These are advisory authoring-convention "
            "findings (not blockers) — fix them with skill_manage(action='patch') "
            "to match Hermes skill standards."
        )

    def test_create_skill_linter_failure_remains_advisory(self, tmp_path):
        with _skill_dir(tmp_path), \
             patch(
                 "tools.skill_linter.lint_skill",
                 side_effect=RuntimeError("deterministic linter failure"),
             ) as lint_skill:
            result = _create_skill("my-skill", VALID_SKILL_CONTENT)

        skill_md = tmp_path / "my-skill" / "SKILL.md"
        assert result["success"] is True, result
        assert (tmp_path / "my-skill" / "SKILL.md").exists()
        assert result["skill_md"] == str(skill_md)
        lint_skill.assert_called_once_with(skill_md)
        assert "lint_warnings" not in result
        assert "lint_hint" not in result

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
        """Category-dir resolution + Phase2 security-first refusal.

        RECONCILED (R2E, policy PRESERVE_STRONG_WRONG_OBJECT_NONINTERFERENCE):
        the original assertion required the category dir to be gone, which
        presupposes a SUCCESSFUL foreground recursive delete.  Phase2 has
        no identity-bound kernel-anchored recursive-delete primitive, so
        ``_delete_skill`` refuses at the last-mile boundary and the
        category-cleanup branch (``parent.rmdir()``) is unreachable by
        design.  See S3/S4 in the R2D audit.

        Preserved business behavior (category B):
          * a categorised skill is created under ``<root>/<category>/<name>``;
          * ``_delete_skill`` resolves that categorised target correctly
            (the refusal payload's ``target`` points at the category path,
            proving resolution reached the right object);
          * the category dir is NOT removed — and neither is the skill,
            because nothing was destroyed.
        """
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT, category="devops")
            assert (tmp_path / "devops" / "my-skill" / "SKILL.md").exists()
            result = _delete_skill("my-skill")
        # Security-first contract: refusal, not deletion.
        assert result["success"] is False, result
        assert result["policy_reason"] == "atomic_recursive_delete_unavailable", result
        assert (
            result["rollback_failure_kind"]
            == "identity_bound_recursive_delete_unavailable"
        ), result
        assert result["live_mutation_committed"] is False, result
        assert result["safe_to_retry"] is False, result
        # Correct categorised target resolution (preserved coverage).
        assert result["target"] == str(tmp_path / "devops" / "my-skill"), result
        # Nothing destroyed: skill and its category dir both survive.
        assert (tmp_path / "devops").exists()
        assert (tmp_path / "devops" / "my-skill" / "SKILL.md").exists()


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
        """Target resolution for an existing supporting file + Phase2 refusal.

        RECONCILED (R2E, policy PRESERVE_STRONG_WRONG_OBJECT_NONINTERFERENCE):
        the original assertions required the file to be gone, which
        presupposes a SUCCESSFUL foreground unlink.  ``os.unlink(name,
        dir_fd=parent_fd)`` re-resolves the basename at unlink time and
        cannot be made conditional on the validated ``st_dev/st_ino``, so
        ``_remove_file`` refuses at the last-mile boundary (Camino B).

        Preserved business behavior (category B):
          * request parsing accepts a nested ``references/api.md`` path;
          * the target resolves to the correct file under the correct
            skill (proved by the refusal payload's ``target``/``parent``);
          * the file's bytes are untouched — no wrong-object mutation.
        """
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            _write_file("my-skill", "references/api.md", "content")
            target = tmp_path / "my-skill" / "references" / "api.md"
            assert target.exists()
            result = _remove_file("my-skill", "references/api.md")
        # Security-first contract: refusal, not removal.
        assert result["success"] is False, result
        assert result["policy_reason"] == "atomic_identity_delete_unavailable", result
        assert (
            result["rollback_failure_kind"] == "identity_bound_unlink_unavailable"
        ), result
        assert result["live_mutation_committed"] is False, result
        assert result["safe_to_retry"] is False, result
        # Correct nested-target resolution (preserved coverage).
        assert result["target"] == str(target), result
        assert result["parent"] == str(target.parent), result
        # Target preserved, bytes unchanged.
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "content"


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


class TestFinalSkillMutationRevalidation:
    @staticmethod
    def _deny_decision(reason="final deny"):
        from agent.self_improvement_policy import Decision as _Decision

        return _Decision(result="DENY_UNKNOWN_OPERATION", reason=reason)

    def test_initial_allow_then_final_deny_blocks_patch_and_preserves_bytes(self, tmp_path, monkeypatch):
        with _skill_dir(tmp_path):
            assert _create_skill("my-skill", VALID_SKILL_CONTENT)["success"] is True
            target = tmp_path / "my-skill" / "SKILL.md"
            before = target.read_bytes()
            monkeypatch.setattr(
                "agent.self_improvement_decision_context.get_self_improvement_decision",
                lambda: self._deny_decision("stale ALLOW invalidated before sink"),
            )
            result = _patch_skill("my-skill", "Do the thing.", "Do the unsafe thing.")

        assert result["success"] is False, result
        assert result["policy_reason"] == "DENY_UNKNOWN_OPERATION"
        assert target.read_bytes() == before

    def test_missing_final_decision_blocks_create_without_partial_tree(self, tmp_path, monkeypatch):
        with _skill_dir(tmp_path):
            monkeypatch.setattr(
                "agent.self_improvement_decision_context.get_self_improvement_decision",
                lambda: self._deny_decision("missing typed context; fail-closed DENY"),
            )
            result = _create_skill("my-skill", VALID_SKILL_CONTENT)

        assert result["success"] is False, result
        assert not (tmp_path / "my-skill").exists()

    def test_final_session_policy_exception_blocks_edit_and_preserves_bytes(self, tmp_path, monkeypatch):
        import agent.session_write_policy as swp

        with _skill_dir(tmp_path):
            assert _create_skill("my-skill", VALID_SKILL_CONTENT)["success"] is True
            target = tmp_path / "my-skill" / "SKILL.md"
            before = target.read_bytes()
            monkeypatch.setattr(
                swp,
                "evaluate_session_write_policy",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            result = _edit_skill("my-skill", VALID_SKILL_CONTENT_2)

        assert result["success"] is False, result
        assert result["policy_reason"] == "policy_evaluation_failed"
        assert target.read_bytes() == before

    def test_readonly_session_blocks_write_file_at_sink(self, tmp_path):
        from agent.session_write_policy import SessionWritePolicy, session_write_policy_scope

        with _skill_dir(tmp_path):
            assert _create_skill("my-skill", VALID_SKILL_CONTENT)["success"] is True
            policy = SessionWritePolicy.deny_all("readonly-test")
            with session_write_policy_scope(policy):
                result = _write_file("my-skill", "references/new.md", "new")

        assert result["success"] is False, result
        assert result["policy_reason"] == "deny_all"
        assert not (tmp_path / "my-skill" / "references" / "new.md").exists()

    def test_env_disabled_at_sink_blocks_stale_allow_and_preserves_bytes(self, tmp_path, monkeypatch):
        with _skill_dir(tmp_path):
            assert _create_skill("my-skill", VALID_SKILL_CONTENT)["success"] is True
            target = tmp_path / "my-skill" / "SKILL.md"
            before = target.read_bytes()
            monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
            result = _patch_skill("my-skill", "Do the thing.", "Do the unsafe thing.")

        assert result["success"] is False, result
        assert result["policy_reason"] == "DENY_ENV_DISABLED"
        assert target.read_bytes() == before

    def test_background_review_stale_allow_final_deny_preserves_target(self, tmp_path, monkeypatch):
        from tools.skill_manager_tool import mark_background_review_skill_read
        from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin

        with _skill_dir(tmp_path):
            assert _create_skill("my-skill", VALID_SKILL_CONTENT)["success"] is True
            target = tmp_path / "my-skill" / "SKILL.md"
            before = target.read_bytes()
            token = set_current_write_origin(BACKGROUND_REVIEW)
            try:
                mark_background_review_skill_read(target)
                monkeypatch.setattr("tools.skill_manager_tool._background_review_write_guard", lambda *_a, **_kw: None)
                monkeypatch.setattr(
                    "agent.self_improvement_decision_context.get_self_improvement_decision",
                    lambda: self._deny_decision("background_review stale ALLOW invalidated"),
                )
                result = _patch_skill("my-skill", "Do the thing.", "Do the unsafe thing.")
            finally:
                reset_current_write_origin(token)

        assert result["success"] is False, result
        assert target.read_bytes() == before

    def test_delete_and_remove_file_final_deny_preserve_targets(self, tmp_path, monkeypatch):
        with _skill_dir(tmp_path):
            assert _create_skill("my-skill", VALID_SKILL_CONTENT)["success"] is True
            assert _write_file("my-skill", "references/old.md", "old")["success"] is True
            skill_md = tmp_path / "my-skill" / "SKILL.md"
            support = tmp_path / "my-skill" / "references" / "old.md"
            before_skill = skill_md.read_bytes()
            before_support = support.read_bytes()
            monkeypatch.setattr(
                "agent.self_improvement_decision_context.get_self_improvement_decision",
                lambda: self._deny_decision("final destructive op deny"),
            )
            delete_result = _delete_skill("my-skill")
            remove_result = _remove_file("my-skill", "references/old.md")

        assert delete_result["success"] is False, delete_result
        assert remove_result["success"] is False, remove_result
        assert skill_md.read_bytes() == before_skill
        assert support.read_bytes() == before_support

    def test_direct_final_helper_without_context_denies(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent.self_improvement_decision_context.get_self_improvement_decision",
            lambda: self._deny_decision("direct helper untrusted caller"),
        )
        result = _final_skill_mutation_denial(
            "write_file", tmp_path / "skill" / "references" / "x.md", origin="unit"
        )

        assert result is not None
        assert result["success"] is False
        assert result["operation_kind"] == "skill_write_file"

    @staticmethod
    def _tree_artifacts(root: Path) -> set[str]:
        names = set()
        try:
            paths = list(root.rglob("*"))
        except FileNotFoundError:
            paths = []
        for path in paths:
            try:
                name = path.name
                if (
                    name.startswith(".hermes-staging-")
                    or name.startswith(".hermes-skill-staging-")
                    or ".publish." in name
                    or ".tmp." in name
                ):
                    names.add(str(path.relative_to(root)))
            except FileNotFoundError:
                pass
        return names

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _revoke_during_scan(self, monkeypatch, calls, staging_paths):
        real_final = _final_skill_mutation_denial
        import tools.skill_manager_tool as smt
        real_create_staging = smt._create_private_staging

        def recording_final(action, target_path, *, origin):
            result = real_final(action, target_path, origin=origin)
            calls.append((origin, result is None, None if result is None else result.get("policy_reason")))
            return result

        def revoke_scan(_staged_skill_dir):
            monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
            return None

        def recording_create_staging(skills_root):
            handle = real_create_staging(skills_root)
            staging_paths.append(Path(handle.path))
            return handle

        monkeypatch.setattr(
            "tools.skill_manager_tool._final_skill_mutation_denial",
            recording_final,
        )
        monkeypatch.setattr(
            "tools.skill_manager_tool._security_scan_skill_fail_closed",
            revoke_scan,
        )
        monkeypatch.setattr(
            "tools.skill_manager_tool._create_private_staging",
            recording_create_staging,
        )

    def _assert_early_allow_late_deny(self, calls):
        assert any(allowed for _origin, allowed, _reason in calls), calls
        assert any(reason == "DENY_ENV_DISABLED" for _origin, _allowed, reason in calls), calls

    def _assert_staging_cleaned(self, staging_paths):
        assert staging_paths
        for staging_path in staging_paths:
            assert not staging_path.exists()
            assert not staging_path.parent.exists()

    def test_create_late_revocation_after_staging_blocks_publish_and_cleans_artifacts(self, tmp_path, monkeypatch):
        calls = []
        staging_paths = []
        with _skill_dir(tmp_path):
            self._revoke_during_scan(monkeypatch, calls, staging_paths)
            target = tmp_path / "my-skill"
            result = _create_skill("my-skill", VALID_SKILL_CONTENT)

        assert result["success"] is False, result
        assert result["policy_reason"] == "DENY_ENV_DISABLED"
        assert not target.exists()
        self._assert_staging_cleaned(staging_paths)
        self._assert_early_allow_late_deny(calls)

    def test_edit_late_revocation_after_scan_preserves_target_and_avoids_publish(self, tmp_path, monkeypatch):
        calls = []
        staging_paths = []
        with _skill_dir(tmp_path):
            assert _create_skill("my-skill", VALID_SKILL_CONTENT)["success"] is True
            target = tmp_path / "my-skill" / "SKILL.md"
            before_sha = self._sha(target)
            self._revoke_during_scan(monkeypatch, calls, staging_paths)
            with patch("tools.skill_manager_tool.atomic_replace") as atomic_replace:
                result = _edit_skill("my-skill", VALID_SKILL_CONTENT_2)

        assert result["success"] is False, result
        assert result["policy_reason"] == "DENY_ENV_DISABLED"
        assert self._sha(target) == before_sha
        self._assert_staging_cleaned(staging_paths)
        atomic_replace.assert_not_called()
        self._assert_early_allow_late_deny(calls)

    def test_patch_late_revocation_after_scan_preserves_target_and_avoids_publish(self, tmp_path, monkeypatch):
        calls = []
        staging_paths = []
        with _skill_dir(tmp_path):
            assert _create_skill("my-skill", VALID_SKILL_CONTENT)["success"] is True
            target = tmp_path / "my-skill" / "SKILL.md"
            before_sha = self._sha(target)
            self._revoke_during_scan(monkeypatch, calls, staging_paths)
            with patch("tools.skill_manager_tool.atomic_replace") as atomic_replace:
                result = _patch_skill("my-skill", "Do the thing.", "Do the safer thing.")

        assert result["success"] is False, result
        assert result["policy_reason"] == "DENY_ENV_DISABLED"
        assert self._sha(target) == before_sha
        self._assert_staging_cleaned(staging_paths)
        atomic_replace.assert_not_called()
        self._assert_early_allow_late_deny(calls)

    def test_write_file_late_revocation_after_scan_preserves_target_and_avoids_publish(self, tmp_path, monkeypatch):
        calls = []
        staging_paths = []
        with _skill_dir(tmp_path):
            assert _create_skill("my-skill", VALID_SKILL_CONTENT)["success"] is True
            assert _write_file("my-skill", "references/api.md", "before")["success"] is True
            target = tmp_path / "my-skill" / "references" / "api.md"
            before_sha = self._sha(target)
            self._revoke_during_scan(monkeypatch, calls, staging_paths)
            with patch("tools.skill_manager_tool.atomic_replace") as atomic_replace:
                result = _write_file("my-skill", "references/api.md", "after")

        assert result["success"] is False, result
        assert result["policy_reason"] == "DENY_ENV_DISABLED"
        assert self._sha(target) == before_sha
        self._assert_staging_cleaned(staging_paths)
        atomic_replace.assert_not_called()
        self._assert_early_allow_late_deny(calls)


    def test_background_review_delete_refuses_bundled_even_with_absorbed_into(self, tmp_path, allow_decision):
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
    from agent.self_improvement_decision_context import (
        bind_self_improvement_decision,
        reset_self_improvement_decision,
    )
    from agent.self_improvement_policy import Decision as _Decision

    token = bind_self_improvement_decision(
        _Decision(result="ALLOW", reason="test skill mutation opt-in")
    )
    try:
        with patch.dict("os.environ", {
                "HERMES_DISABLE_SELF_IMPROVEMENT": "0",
                "HERMES_READ_ONLY_SESSION": "0",
             }), \
             patch("tools.skill_manager_tool.SKILLS_DIR", local_dir), \
             patch("agent.skill_utils.get_all_skills_dirs",
                   return_value=[local_dir, external_dir]):
            yield
    finally:
        reset_self_improvement_decision(token)


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


    def test_background_review_refuses_to_patch_pinned_skill(self, tmp_path, allow_decision):
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


    def test_background_review_fails_closed_when_ownership_lookup_errors(self, tmp_path, allow_decision):
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

    def test_adopted_skill_becomes_writable_by_autonomous_curation(self, tmp_path, monkeypatch, allow_decision):
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
        """If skill_usage.get_record raises, the PIN GUARD must fail open.

        RECONCILED (R2E, policy PRESERVE_STRONG_WRONG_OBJECT_NONINTERFERENCE):
        the original assertion used ``success is True`` as the proxy for
        "the pin guard let this through".  Under Phase2 that proxy is no
        longer available, because every foreground delete now stops at the
        last-mile atomic refusal regardless of pinning.

        The business behavior under test is UNCHANGED and still fully
        asserted, using a sharper probe than the old success proxy: a
        corrupted telemetry file must NOT produce a pin denial.  We
        distinguish the two refusal kinds explicitly —

          * pinned          → error mentions "pinned" / "cannot be deleted"
          * broken sidecar  → falls open past the guard and reaches the
                              canonical ``atomic_recursive_delete_unavailable``

        so a regression that made a broken sidecar behave like a pin would
        still fail this test.
        """
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            with patch("tools.skill_usage.get_record",
                       side_effect=RuntimeError("sidecar broken")):
                result = _delete_skill("my-skill")
        # Fail-open: the pin guard did NOT deny (preserved business behavior).
        assert "pinned" not in result.get("error", "").lower(), result
        assert "cannot be deleted" not in result.get("error", ""), result
        assert result.get("policy_reason") != "pinned_skill", result
        # Execution proceeded past the guard to the security boundary.
        assert result["success"] is False, result
        assert result["policy_reason"] == "atomic_recursive_delete_unavailable", result
        assert (
            result["rollback_failure_kind"]
            == "identity_bound_recursive_delete_unavailable"
        ), result
        assert result["live_mutation_committed"] is False, result
        # Skill preserved — the refusal destroyed nothing.
        assert (tmp_path / "my-skill" / "SKILL.md").exists()



# ---------------------------------------------------------------------------
# _delete_skill — recursive-delete safety (port of Kilo Code #11240)
# ---------------------------------------------------------------------------


class TestDeleteSkillRmtreeGuard:
    """Defense-in-depth before ``shutil.rmtree`` in ``_delete_skill``.

    Mirrors the Kilo Code #11227 fix: never let a recursive skill delete
    escape the skills tree, target a skills root, or follow a symlink.
    """

    def test_normal_delete_still_works(self, tmp_path):
        """Control case: the defense-in-depth guards must NOT over-block.

        RECONCILED via SPLIT (R2E, policy
        PRESERVE_STRONG_WRONG_OBJECT_NONINTERFERENCE).  This test's role in
        the class is the CONTROL for the sibling guard tests
        (``test_symlinked_skill_dir_refused``, ``test_out_of_tree_path_refused``):
        it proves a well-formed, in-tree, non-symlink skill is not caught by
        any of those guards.  It used ``success is True`` as the proxy for
        "no guard fired".

        Under Phase2 that proxy is gone — every foreground delete stops at
        the last-mile atomic refusal.  The control semantics are preserved
        by asserting the refusal is the ATOMIC one and specifically NOT a
        guard rejection, so a regression where the symlink/out-of-tree
        guard started firing on a normal skill would still fail here.

        The preservation half of the old contract lives in the sibling
        ``test_normal_delete_refuses_without_destroying`` below.
        """
        with _skill_dir(tmp_path):
            _create_skill("good-skill", VALID_SKILL_CONTENT)
            result = _delete_skill("good-skill", absorbed_into="")
        # Reached the security boundary — no guard fired.
        assert result["success"] is False, result
        assert result["policy_reason"] == "atomic_recursive_delete_unavailable", result
        assert (
            result["rollback_failure_kind"]
            == "identity_bound_recursive_delete_unavailable"
        ), result
        # Explicitly NOT a defense-in-depth guard rejection.
        err = result.get("error", "").lower()
        assert "symlink" not in err, result
        assert "junction" not in err, result
        assert "outside" not in err, result
        assert "skills root" not in err, result
        assert result["rollback_failure_kind"] != "symlink_detected", result
        # Correct target resolution for a well-formed in-tree skill.
        assert result["target"] == str(tmp_path / "good-skill"), result

    def test_normal_delete_refuses_without_destroying(self, tmp_path):
        """Security half of the split: the refusal destroys nothing.

        Companion to ``test_normal_delete_still_works``.  Asserts the
        Phase2 preservation contract for the ordinary (non-adversarial)
        foreground delete: no recursive-delete primitive runs, the skill
        tree is intact byte-for-byte, and the payload reports no committed
        live mutation.
        """
        with _skill_dir(tmp_path):
            _create_skill("good-skill", VALID_SKILL_CONTENT)
            skill_md = tmp_path / "good-skill" / "SKILL.md"
            original_bytes = skill_md.read_bytes()
            with patch("tools.skill_manager_tool.shutil.rmtree") as spy_rmtree:
                result = _delete_skill("good-skill", absorbed_into="")
            assert spy_rmtree.call_count == 0, (
                f"no recursive-delete primitive may run; called "
                f"{spy_rmtree.call_count} time(s)"
            )
        assert result["success"] is False, result
        assert result["live_mutation_committed"] is False, result
        assert result["safe_to_retry"] is False, result
        # Skill tree preserved, bytes unchanged.
        assert (tmp_path / "good-skill").exists()
        assert skill_md.exists()
        assert skill_md.read_bytes() == original_bytes


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
    from agent.self_improvement_decision_context import (
        bind_self_improvement_decision,
        reset_self_improvement_decision,
    )
    from agent.self_improvement_policy import Decision as _Decision

    token = bind_self_improvement_decision(
        _Decision(result="ALLOW", reason="test curator mutation opt-in")
    )
    try:
        with patch.dict("os.environ", {
                "HERMES_DISABLE_SELF_IMPROVEMENT": "0",
                "HERMES_READ_ONLY_SESSION": "0",
             }), \
             patch("tools.skill_manager_tool.SKILLS_DIR", skills_root), \
             patch("tools.skills_tool.SKILLS_DIR", skills_root), \
             patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_root]), \
             patch("tools.skill_usage._is_curator_managed_record", return_value=True), \
             patch("tools.skill_provenance.is_background_review", return_value=True):
            yield skills_root
    finally:
        reset_self_improvement_decision(token)


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
# restore_skill shared publication guard glue (#79723 absorption)
# ---------------------------------------------------------------------------


def _make_archived_skill(tmp_path: Path, name: str = "restore-probe"):
    hermes_home = tmp_path / ".hermes"
    skills_root = hermes_home / "skills"
    archive = skills_root / ".archive"
    src = archive / name
    src.mkdir(parents=True, exist_ok=True)
    (src / "SKILL.md").write_text(_skill_content(name), encoding="utf-8")
    return hermes_home, skills_root, src, skills_root / name


class TestRestoreSkillPublicationGuard:
    def test_restore_skill_uses_shared_publication_guard_new_only(self, tmp_path, monkeypatch):
        from contextlib import contextmanager
        from pathlib import Path as _Path
        from tools import skill_publish_guard as _spg
        from tools import skill_usage as su

        hermes_home, _skills_root, src, dest = _make_archived_skill(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(su, "is_hub_installed", lambda _name: False)
        monkeypatch.setattr(su, "is_bundled", lambda _name: False)

        events = []
        inside = {"guard": False}
        original_rename = _Path.rename

        @contextmanager
        def guard(name, *, target, replacement_policy):
            events.append(("guard_called", name, target, replacement_policy))
            inside["guard"] = True
            events.append(("guard_enter", src.exists(), dest.exists()))
            try:
                yield
            finally:
                events.append(("guard_exit", src.exists(), dest.exists()))
                inside["guard"] = False

        def recording_rename(self, target):
            if self == src and target == dest:
                events.append(("rename", inside["guard"]))
            return original_rename(self, target)

        def recording_remove(name):
            events.append(("remove_suppressed_name", name, inside["guard"]))

        def recording_set_state(name, state):
            events.append(("set_state", name, state, inside["guard"]))

        monkeypatch.setattr(_spg, "live_skill_publish_guard", guard)
        monkeypatch.setattr(_Path, "rename", recording_rename)
        monkeypatch.setattr(su, "remove_suppressed_name", recording_remove)
        monkeypatch.setattr(su, "set_state", recording_set_state)

        ok, message = su.restore_skill("restore-probe")

        assert ok is True
        assert message == f"restored to {dest}"
        assert dest.exists()
        assert ("guard_called", "restore-probe", dest, "new_only") in events
        assert ("rename", True) in events
        assert ("remove_suppressed_name", "restore-probe", True) in events
        assert ("set_state", "restore-probe", su.STATE_ACTIVE, True) in events
        assert events.index(("guard_enter", True, False)) < events.index(("rename", True))
        assert events.index(("rename", True)) < events.index(("remove_suppressed_name", "restore-probe", True))
        assert events.index(("remove_suppressed_name", "restore-probe", True)) < events.index(("set_state", "restore-probe", su.STATE_ACTIVE, True))
        assert events.index(("set_state", "restore-probe", su.STATE_ACTIVE, True)) < events.index(("guard_exit", False, True))

    def test_restore_skill_lock_acquire_failure_preserves_archive_and_returns_false(self, tmp_path, monkeypatch):
        from contextlib import contextmanager
        from tools import skill_publish_guard as _spg
        from tools import skill_usage as su

        hermes_home, _skills_root, src, dest = _make_archived_skill(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(su, "is_hub_installed", lambda _name: False)
        monkeypatch.setattr(su, "is_bundled", lambda _name: False)

        @contextmanager
        def acquire_failure(*args, **kwargs):
            raise _spg.SkillMutationLockAcquireFailure(
                canonical_skill_path=dest,
                lock_path=tmp_path / "lock",
                platform="posix",
                lock_failure_stage=_spg.LOCK_FAILURE_STAGE_CONTENTION,
                safe_to_retry=True,
            )
            yield

        monkeypatch.setattr(_spg, "live_skill_publish_guard", acquire_failure)

        ok, message = su.restore_skill("restore-probe")

        assert ok is False
        assert "restore could not acquire the publication lock" in message
        assert src.exists()
        assert not dest.exists()

    def test_restore_skill_lock_release_failure_propagates(self, tmp_path, monkeypatch):
        from contextlib import contextmanager
        from tools import skill_publish_guard as _spg
        from tools import skill_usage as su

        hermes_home, _skills_root, _src, dest = _make_archived_skill(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(su, "is_hub_installed", lambda _name: False)
        monkeypatch.setattr(su, "is_bundled", lambda _name: False)
        monkeypatch.setattr(su, "remove_suppressed_name", lambda _name: None)
        monkeypatch.setattr(su, "set_state", lambda _name, _state: None)

        @contextmanager
        def release_failure(*args, **kwargs):
            yield
            raise _spg.SkillMutationLockReleaseFailure(
                canonical_skill_path=dest,
                lock_path=tmp_path / "lock",
                platform="posix",
                release_error=RuntimeError("release failed"),
                live_mutation_committed=True,
            )

        monkeypatch.setattr(_spg, "live_skill_publish_guard", release_failure)

        with pytest.raises(_spg.SkillMutationLockReleaseFailure):
            su.restore_skill("restore-probe")
