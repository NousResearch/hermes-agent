"""Tests for tools/skill_manager_tool.py — skill creation, editing, and deletion."""

import json
import threading
from contextlib import contextmanager
from contextvars import copy_context
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
    _discover_skill_subdirs,
    _write_file,
    _remove_file,
    _find_skill,
    _skill_lock_path,
    skill_manage,
)
from agent.skill_utils import (
    extract_skill_description,
    parse_frontmatter,
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
        for bad in ("skill/name", "skill name", "skill@name"):
            assert _validate_name(bad) is not None


class TestValidateCategory:
    def test_path_traversal_rejected(self):
        assert _validate_category("../escape") is not None

    def test_absolute_path_rejected(self):
        assert _validate_category("/tmp/escape") is not None


# ---------------------------------------------------------------------------
# _validate_frontmatter
# ---------------------------------------------------------------------------


class TestValidateFrontmatter:
    def test_no_frontmatter(self):
        err = _validate_frontmatter("# Just a heading\nSome content.\n")
        assert err is not None

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

    def test_path_traversal_blocked(self):
        assert _validate_file_path("references/../../../etc/passwd") is not None


    def test_skill_md_traversal_still_rejected(self):
        # The SKILL.md exception must not weaken the traversal guard.
        assert _validate_file_path("../SKILL.md") is not None

    def test_other_root_md_still_rejected(self):
        # Only SKILL.md gets the root-level exception, not arbitrary files.
        assert _validate_file_path("README.md") is not None


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
        assert not (tmp_path / "escape").exists()


    def test_edit_long_desc_still_allowed_with_preview(self, tmp_path):
        """Edit/patch paths stay permissive so existing over-limit skills
        remain maintainable — they warn via system_prompt_preview instead."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = _edit_skill("my-skill", LONG_DESC_CONTENT)
        assert result["success"] is True
        assert "system_prompt_preview" in result
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


    def test_edit_existing_skill_by_categorized_path(self, tmp_path):
        """Categorized names (``category/skill``) must resolve in skill_manage.

        skill_view's ambiguity hint explicitly tells the caller to use the
        full categorized path (``category/skill-name``), and skills_list
        reports skills with their category. But ``_find_skill`` only matched
        the bare directory name, so every skill_manage call that followed
        that hint failed with "not found in active profile" — the agent then
        retried repeatedly, burning LLM round trips (top recurring audit
        finding). Resolution parity with skill_view is the fix.
        """
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT, category="software-development")
            result = _edit_skill("software-development/my-skill", VALID_SKILL_CONTENT_2)
        assert result["success"] is True, result.get("error")
        content = (tmp_path / "software-development" / "my-skill" / "SKILL.md").read_text()
        assert "Updated description" in content

    def test_find_skill_accepts_categorized_path(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT, category="mlops")
            found = _find_skill("mlops/my-skill")
        assert found is not None
        assert found["path"] == tmp_path / "mlops" / "my-skill"

    def test_find_skill_bare_name_still_resolves_nested_skill(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT, category="mlops")
            found = _find_skill("my-skill")
        assert found is not None
        assert found["path"] == tmp_path / "mlops" / "my-skill"

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

    def test_patch_surfaces_oversized_body_finding_and_stays_quiet_when_clean(self, tmp_path):
        # SKILL.md grows by patches; the write that crosses the body budget carries the advisory
        # finding, a small clean patch attaches no lint keys at all.
        from tools.skill_linter import _BODY_SOFT_BUDGET_CHARS
        filler = "- Prefer the native tool; the shell path loses the structured result.\n"
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            quiet = _patch_skill("my-skill", "Do the thing.", "Do the new thing.")
            grown = _patch_skill("my-skill", "Do the new thing.",
                                 filler * (_BODY_SOFT_BUDGET_CHARS // len(filler) + 1))
        assert quiet["success"] is True and "lint_warnings" not in quiet
        assert grown["success"] is True
        assert "oversized-body" in {w["rule"] for w in grown["lint_warnings"]}


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


class TestSkillMutationLock:
    def test_concurrent_patches_keep_both_updates(self, tmp_path):
        """Two writers patching the same SKILL.md serialize on the per-skill lock (#111578):
        the second cannot read stale content while the first is between read and write."""
        first_entered = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        call_lock = threading.Lock()
        calls = 0
        results = []

        from tools.fuzzy_match import fuzzy_find_and_replace as real_replace

        def delayed_replace(*args, **kwargs):
            nonlocal calls
            with call_lock:
                calls += 1
                call = calls
            if call == 1:
                first_entered.set()
                assert release_first.wait(timeout=2)
            else:
                second_entered.set()
            return real_replace(*args, **kwargs)

        def run_patch(old, new):
            results.append(json.loads(skill_manage(
                action="patch", name="my-skill", old_string=old, new_string=new)))

        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            with patch("tools.fuzzy_match.fuzzy_find_and_replace", side_effect=delayed_replace):
                first = threading.Thread(target=run_patch, args=("# Test Skill", "# Test Skill Updated"))
                second = threading.Thread(target=run_patch, args=("Step 1:", "Step 1 Updated:"))
                first.start()
                assert first_entered.wait(timeout=2)
                second.start()
                assert not second_entered.wait(timeout=0.15), "second writer entered before first committed"
                release_first.set()
                first.join(timeout=2)
                second.join(timeout=2)
            content = (tmp_path / "my-skill" / "SKILL.md").read_text()

        assert all(result["success"] for result in results)
        assert "# Test Skill Updated" in content
        assert "Step 1 Updated:" in content

    @pytest.mark.parametrize("name", ["a" * 300, "bad\x00name", "../../etc", ""])
    def test_rejected_name_returns_json_and_leaves_no_lock_file(self, tmp_path, name):
        """Name validation runs before the lock is opened: an over-long / NUL / traversal / empty
        name yields the create handler's JSON error (never OSError/ValueError from the lock
        path) and leaves nothing behind in ``<skills>/.locks/``."""
        with _skill_dir(tmp_path):
            result = json.loads(skill_manage(action="create", name=name, content=VALID_SKILL_CONTENT))
        assert result["success"] is False
        assert result["error"] == _validate_name(name)
        assert not (tmp_path / ".locks").exists()

    def test_lock_path_is_digest_keyed_and_shared_across_name_forms(self, tmp_path):
        """``foo`` and ``category/foo`` share one lock, keyed on a fixed-width digest of the
        basename so the filename never depends on the skill name's length or characters."""
        with _skill_dir(tmp_path):
            lock = _skill_lock_path("mlops/foo")
            assert lock == _skill_lock_path("foo")
            assert lock.parent == tmp_path / ".locks"


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

    @pytest.mark.parametrize("op, stray_key, destination", [
        ({"action": "create", "file_content": VALID_SKILL_CONTENT}, "file_content", "'content'"),
        ({"action": "create", "new_string": VALID_SKILL_CONTENT}, "new_string", "'content'"),
        ({"action": "patch", "file_content": "body"}, "file_content", "old_string/new_string"),
    ])
    def test_misplaced_text_slot_error_names_the_key_it_arrived_in(self, tmp_path, op, stray_key,
                                                                    destination):
        """#112677 — a batch op whose SKILL.md text sits in another action's key must be told
        WHICH key it used and where to move it; the bare "X is required" error made a local
        model replay the identical payload until the tool-loop guardrail tripped. The misfiled
        op is rejected before any sibling is applied (no rollback needed), and a plain
        missing-content op gets no note."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = json.loads(skill_manage(action="", name="", operations=[
                {"name": "sibling", "action": "create", "content": VALID_SKILL_CONTENT},
                {"name": "my-skill", **op}]))
            bare = json.loads(skill_manage(action="", name="",
                                           operations=[{"name": "other", "action": "create"}]))
            sibling_created = _find_skill("sibling") is not None

        assert result["success"] is False
        assert "operations[1]" in result["error"]
        assert f"'{stray_key}'" in result["error"] and destination in result["error"]
        assert "rolled back" not in result["error"] and not sibling_created
        assert bare["success"] is False and "file_content" not in bare["error"]

    @pytest.mark.parametrize("op", [
        {"action": "patch", "old_string": "body"},
        {"action": "patch", "old_string": "body", "new_string": "x", "content": "# whole"},
    ])
    def test_patch_shape_misses_are_rejected_before_any_sibling_applies(self, tmp_path, op):
        """A patch missing new_string, or mixing content with old_string/new_string, is a shape
        miss like the misfiled text slot: decided in _validate_batch_ops, so op[0] is never
        created and rolled back."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            result = json.loads(skill_manage(action="", name="", operations=[
                {"name": "sibling", "action": "create", "content": VALID_SKILL_CONTENT},
                {"name": "my-skill", **op}]))
            sibling_created = _find_skill("sibling") is not None

        assert result["success"] is False
        assert "operations[1]" in result["error"]
        assert "rolled back" not in result["error"] and not sibling_created



    def test_write_file_given_content_names_the_reverse_misplacement(self, tmp_path):
        """#112677 — the reverse direction: create's `content` sent to write_file. Checked on the
        legacy flat call shape so both entry points share the one hint chokepoint."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", VALID_SKILL_CONTENT)
            crossed = json.loads(skill_manage(action="write_file", name="my-skill",
                                              file_path="references/a.md", content="hello"))
            ok = json.loads(skill_manage(action="write_file", name="my-skill",
                                         file_path="references/a.md", file_content="hello"))

        assert crossed["success"] is False
        assert "'content'" in crossed["error"] and "'file_content'" in crossed["error"]
        assert ok["success"] is True

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
        # Foreground create carries the "learn" learning-signal marker — never the
        # curator-management opt-in ("agent"), and the record may be missing
        # entirely (telemetry best-effort).
        rec = usage.get("test-skill") or {}
        assert rec.get("created_by") in {"learn", None, "", False}




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


class TestPatchRecoveryLoop:
    """#33064 — the recovery guidance must be FOLLOWABLE, not just well-worded.

    Asserting that an error string contains 'read' proves nothing about whether
    a model obeying it reaches a working patch. These tests walk the loop end to
    end through the public tool: bad call -> read the error -> do what it says ->
    patch succeeds, without ever touching action='write_file'.
    """

    CONTENT = """\
---
name: test-skill
description: A test skill.
---

# Test Skill

Step 1: Do the thing.
Step 2: Do the other thing.
"""

    def test_recovery_loop_reaches_a_working_patch(self, tmp_path):
        with _skill_dir(tmp_path):
            _create_skill("my-skill", self.CONTENT)

            # The half-formed call that used to dead-end.
            bad = json.loads(skill_manage(action="patch", name="my-skill",
                                          new_string="Step 1: Do it better."))
            assert bad["success"] is False
            assert "read" in bad["error"].lower()

            # Do exactly what the error says: read the file, copy verbatim, retry.
            on_disk = (tmp_path / "my-skill" / "SKILL.md").read_text()
            snippet = "Step 1: Do the thing."
            assert snippet in on_disk
            good = json.loads(skill_manage(action="patch", name="my-skill",
                                           old_string=snippet,
                                           new_string="Step 1: Do it better."))

        assert good["success"] is True, f"guided retry must succeed, got {good}"
        after = (tmp_path / "my-skill" / "SKILL.md").read_text()
        assert "Step 1: Do it better." in after
        # The recovery must be surgical — that is the whole point of not
        # falling back to a whole-file rewrite.
        assert "Step 2: Do the other thing." in after, "unrelated content lost"
        assert after.startswith("---"), "frontmatter destroyed"


    @pytest.mark.parametrize("kwargs", [
        {"old_string": None, "new_string": "x"},
        {"old_string": "", "new_string": "x"},
        {"old_string": "Step 1: Do the thing.", "new_string": "Step 1: Do the thing."},
    ])
    def test_rejected_patch_leaves_file_byte_identical(self, tmp_path, kwargs):
        """A rejected patch must not partially mutate the skill."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", self.CONTENT)
            before = (tmp_path / "my-skill" / "SKILL.md").read_text()
            result = json.loads(skill_manage(action="patch", name="my-skill", **kwargs))
            after = (tmp_path / "my-skill" / "SKILL.md").read_text()

        assert result["success"] is False
        assert before == after, "rejected patch mutated the file"


    def test_empty_new_string_still_deletes(self, tmp_path):
        """new_string='' is legal (delete matched text) and must NOT be rejected."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", self.CONTENT)
            result = json.loads(skill_manage(action="patch", name="my-skill",
                                             old_string="Step 2: Do the other thing.\n",
                                             new_string=""))
            after = (tmp_path / "my-skill" / "SKILL.md").read_text()

        assert result["success"] is True, f"empty new_string wrongly rejected: {result}"
        assert "Step 2:" not in after

    def test_missing_new_string_still_rejected(self, tmp_path):
        """The dispatcher also guarded new_string; _patch_skill must still catch it."""
        with _skill_dir(tmp_path):
            _create_skill("my-skill", self.CONTENT)
            result = json.loads(skill_manage(action="patch", name="my-skill",
                                             old_string="Step 1: Do the thing."))
        assert result["success"] is False
        assert "new_string" in result["error"]


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
        from tools.skill_manager_guards import mark_background_review_skill_read
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

    def test_background_review_read_survives_copied_tool_contexts(
        self, tmp_path, monkeypatch
    ):
        """A view in one tool worker authorizes a patch in the next worker."""
        from tools.skills_tool import skill_view
        from tools.skill_manager_guards import _reset_background_review_read_marks

        _reset_background_review_read_marks()
        with _curator_pass(tmp_path, monkeypatch=monkeypatch):
            _create_curator_skill("reviewed", _skill_content("reviewed"))

            viewed = copy_context().run(skill_view, "reviewed")
            assert json.loads(viewed)["success"] is True

            patched = copy_context().run(
                skill_manage,
                action="patch",
                name="reviewed",
                old_string="Step 1: Do the thing.",
                new_string="Step 1: Do the thing safely.",
            )
            assert json.loads(patched)["success"] is True

        _reset_background_review_read_marks()

    def test_background_review_read_marks_stay_isolated_between_reviews(
        self, tmp_path, monkeypatch
    ):
        """Copied tool contexts share only their own review's read marks."""
        from tools.skills_tool import skill_view
        from tools.skill_manager_guards import _reset_background_review_read_marks

        _reset_background_review_read_marks()
        with _curator_pass(tmp_path, monkeypatch=monkeypatch):
            _create_curator_skill("reviewed", _skill_content("reviewed"))

            first_review = copy_context()
            _reset_background_review_read_marks()
            second_review = copy_context()

            viewed = first_review.run(skill_view, "reviewed")
            assert json.loads(viewed)["success"] is True

            blocked = second_review.run(
                skill_manage,
                action="patch",
                name="reviewed",
                old_string="Step 1: Do the thing.",
                new_string="Step 1: Do the thing safely.",
            )
            result = json.loads(blocked)
            assert result["success"] is False
            assert result.get("_read_before_write_required") is True

        _reset_background_review_read_marks()

    def test_background_review_support_file_overwrite_requires_that_file_read(self, tmp_path, monkeypatch):
        from tools.skills_tool import skill_view
        from tools.skill_manager_guards import _reset_background_review_read_marks

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


# _discover_skill_subdirs — custom subdirectory auto-discovery
# ---------------------------------------------------------------------------


class TestDiscoverSkillSubdirs:
    def test_finds_custom_subdirs_with_content(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "steps").mkdir()
        (skill_dir / "steps" / "phase1.md").write_text("# Phase 1")
        (skill_dir / "steps" / "phase2.md").write_text("# Phase 2")
        (skill_dir / "checks").mkdir()
        (skill_dir / "checks" / "preflight.py").write_text("print('ok')")

        result = _discover_skill_subdirs(skill_dir)

        assert "steps" in result
        assert "checks" in result

    def test_skips_empty_subdirectory(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "empty").mkdir()

        result = _discover_skill_subdirs(skill_dir)

        assert "empty" not in result

    def test_skips_dot_directories(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / ".hidden").mkdir()
        (skill_dir / ".hidden" / "secret.md").write_text("shh")

        result = _discover_skill_subdirs(skill_dir)

        assert ".hidden" not in result

    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        result = _discover_skill_subdirs(tmp_path / "nope")
        assert result == set()

    def test_includes_known_standard_subdirs(self, tmp_path):
        """Standard dirs like references/ should also be discoverable."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "guide.md").write_text("guide")

        result = _discover_skill_subdirs(skill_dir)

        assert "references" in result
>>>>> (feat(skills): auto-discover custom subdirectories in skill linked_files (#30999))
