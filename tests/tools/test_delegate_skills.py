#!/usr/bin/env python3
"""Tests for the delegate_task skills parameter.

Covers: _merge_skills, _load_delegate_skills (the cron resolution pipeline:
bundle -> normalize -> skill_view -> ambiguous/config -> security scan), prompt
injection into _build_child_system_prompt, _build_child_agent AIAgent
constructor capture, schema, end-to-end integration through delegate_task(),
and survival of `skills` across the async-delegation ledger's task_json
round-trip (dispatch -> state.db -> restart-recovery event).

Current-architecture notes (ported from the original #54701 suite):
- The goal is the child's FIRST USER TURN, not part of the system prompt, so
  prompt assertions check skill content (never the goal text).
- The injection scan lives in tools.cronjob_prompt_scan (moved out of
  tools.cronjob_tools) and cron's loader also runs ambiguous_skill_label +
  _inject_skill_config; the delegate loader mirrors that full pipeline.
- Children are built on the calling thread BEFORE dispatch, so skills content
  is baked into the child's ephemeral_system_prompt and never needs to flow
  through the ledger; only the skill NAMES ride the dispatch record so an
  abandoned delegation can be re-dispatched from its recovery event alone.

Run:  scripts/run_tests.sh tests/tools/test_delegate_skills.py -q
"""

import json
import threading
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _merge_skills,
    _load_delegate_skills,
    _build_child_system_prompt,
    _build_child_agent,
    delegate_task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_load_pipeline(stack, *, skill_view_side_effect=None,
                         skill_view_return=None, scan_passthrough=True,
                         bundle_key_return=None):
    """Patch the cron resolution pipeline used by _load_delegate_skills.

    Returns the mock objects for further assertions.
    """
    mocks = {}
    mocks["view"] = stack.enter_context(patch("tools.skills_tool.skill_view"))
    if skill_view_side_effect is not None:
        mocks["view"].side_effect = skill_view_side_effect
    elif skill_view_return is not None:
        mocks["view"].return_value = skill_view_return
    mocks["bump"] = stack.enter_context(patch("tools.skill_usage.bump_use"))
    mocks["bundle"] = stack.enter_context(
        patch("agent.skill_bundles.resolve_bundle_command_key",
              return_value=bundle_key_return)
    )
    # None by default: without a resolved bundle key the message builder is
    # never reached, and a truthy MagicMock default would poison the join.
    mocks["bundle_msg"] = stack.enter_context(
        patch("agent.skill_bundles.build_bundle_invocation_message",
              return_value=None)
    )
    mocks["norm"] = stack.enter_context(
        patch("agent.skill_utils.normalize_skill_lookup_name", side_effect=lambda x: x)
    )
    mocks["ambiguous"] = stack.enter_context(
        patch("agent.skill_commands.ambiguous_skill_label", return_value=None)
    )
    mocks["config"] = stack.enter_context(patch("agent.skill_commands._inject_skill_config"))
    mocks["scan"] = stack.enter_context(
        patch("tools.cronjob_prompt_scan._scan_cron_skill_assembled")
    )
    if scan_passthrough:
        mocks["scan"].side_effect = lambda assembled: (assembled, "")
    return mocks


def _patch_child_agent_env(stack):
    """Patch _build_child_agent's external dependencies.

    The current architecture resolves delegation config/credentials for real
    against the conftest-sandboxed home (the canonical pattern in
    tests/tools/test_delegate.py), so only the AIAgent constructor is patched.
    """
    return stack.enter_context(patch("run_agent.AIAgent"))


def _make_mock_parent(depth=0):
    """Mock parent with the fields delegate_task/_build_child_agent expect
    (same shape as tests/tools/test_delegate.py's canonical parent)."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.request_overrides = {}
    parent.enabled_toolsets = None
    parent.valid_tool_names = []
    parent._session_db = None
    parent._delegate_depth = depth
    parent._subagent_id = None
    parent._delegate_spinner = None
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    return parent


def _capture_init(captured, completed_result=None):
    """Return a side_effect for AIAgent that captures kwargs into *captured*."""
    def _init(*args, **kwargs):
        captured.update(kwargs)
        captured.setdefault("calls", []).append(kwargs)
        inst = MagicMock()
        inst.tool_progress_callback = kwargs.get("tool_progress_callback")
        inst._delegate_depth = kwargs.get("_delegate_depth", 0)
        if completed_result is not None:
            inst.run_conversation.return_value = completed_result
        return inst
    return _init


_COMPLETED_CHILD_RESULT = {
    "final_response": "mock result",
    "completed": True,
    "api_calls": 1,
}


def _patch_delegate_env(stack):
    """Patch delegate_task's runtime environment for integration tests
    (live transcripts are a best-effort side channel; keep the run hermetic)."""
    stack.enter_context(
        patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], []))
    )
    stack.enter_context(patch("tools.delegation_live_log.update_manifest_statuses"))


# ---------------------------------------------------------------------------
# _merge_skills
# ---------------------------------------------------------------------------

class TestMergeSkills(unittest.TestCase):
    """_merge_skills: top-level + per-task merge with dedup."""

    def test_none_and_empty(self):
        self.assertIsNone(_merge_skills(None, None))
        self.assertIsNone(_merge_skills([], []))
        self.assertIsNone(_merge_skills(None, []))
        self.assertIsNone(_merge_skills([], None))

    def test_top_level_only(self):
        self.assertEqual(_merge_skills(["a", "b"], None), ["a", "b"])
        self.assertEqual(_merge_skills(["x"], []), ["x"])

    def test_per_task_only(self):
        self.assertEqual(_merge_skills(None, ["x", "y"]), ["x", "y"])
        self.assertEqual(_merge_skills([], ["z"]), ["z"])

    def test_merge_per_task_first(self):
        self.assertEqual(_merge_skills(["a", "b"], ["x", "y"]), ["x", "y", "a", "b"])

    def test_dedup_preserves_order(self):
        result = _merge_skills(["a", "b", "c"], ["b", "d"])
        self.assertEqual(result, ["b", "d", "a", "c"])
        # Per-task instance wins: "b" from per-task is at index 0, not index 2
        self.assertEqual(result[0], "b")


# ---------------------------------------------------------------------------
# _load_delegate_skills
# ---------------------------------------------------------------------------

class TestLoadDelegateSkills(unittest.TestCase):
    """_load_delegate_skills: cron-compatible resolution pipeline."""

    def test_empty(self):
        self.assertEqual(_load_delegate_skills(None), "")
        self.assertEqual(_load_delegate_skills([]), "")

    def test_successful_load(self):
        with ExitStack() as s:
            m = _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "# My Skill\nDo things efficiently.",
            }))
            result = _load_delegate_skills(["my-skill"])
        self.assertIn("my-skill", result)
        self.assertIn("# My Skill", result)
        self.assertIn("Do things efficiently.", result)
        m["bump"].assert_called_once_with("my-skill")

    def test_skill_not_found(self):
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": False, "error": "Skill not found",
            }))
            result = _load_delegate_skills(["missing-skill"])
        self.assertIn("missing-skill", result)
        self.assertIn("skipped", result)

    def test_invalid_json(self):
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_return="not json")
            result = _load_delegate_skills(["bad-skill"])
        self.assertIn("bad-skill", result)
        self.assertIn("skipped", result)

    def test_security_scan_returns_blocked_notice(self):
        """When scan blocks content, a notice is returned (not empty string)
        so the child agent can surface the block to the user."""
        with ExitStack() as s:
            m = _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "malicious content",
            }), scan_passthrough=False)
            m["scan"].return_value = ("cleaned", "Blocked: injection.")
            result = _load_delegate_skills(["malicious-skill"])
        self.assertNotEqual(result, "")
        self.assertIn("malicious-skill", result)
        self.assertIn("blocked", result.lower())
        self.assertIn("security scanner", result.lower())

    def test_multiple_skills(self):
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_side_effect=[
                json.dumps({"success": True, "content": "# Skill A"}),
                json.dumps({"success": True, "content": "# Skill B"}),
            ])
            result = _load_delegate_skills(["skill-a", "skill-b"])
        self.assertIn("skill-a", result)
        self.assertIn("skill-b", result)
        self.assertIn("# Skill A", result)
        self.assertIn("# Skill B", result)

    def test_bundle_resolution(self):
        """When a skill name matches a bundle, the bundle is expanded."""
        with ExitStack() as s:
            m = _patch_load_pipeline(s, bundle_key_return="/my-bundle")
            m["bundle_msg"].return_value = ("Bundle content", [], [])
            result = _load_delegate_skills(["my-bundle"])
        self.assertIn("Bundle content", result)
        m["view"].assert_not_called()

    def test_bundle_without_members_is_skipped(self):
        """A resolved bundle that loads nothing is skipped with a notice."""
        with ExitStack() as s:
            m = _patch_load_pipeline(s, bundle_key_return="/empty-bundle")
            m["bundle_msg"].return_value = None
            result = _load_delegate_skills(["empty-bundle"])
        self.assertIn("empty-bundle", result)
        self.assertIn("skipped", result)

    # --- Edge cases (P2 gaps from review) ---

    def test_scan_returns_sanitized_content(self):
        """When scan strips invisible unicode, the SANITIZED string is
        returned — not the raw assembled input."""
        with ExitStack() as s:
            m = _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "# Clean Skill",
            }), scan_passthrough=False)
            # Simulate sanitization: scan returns a modified string
            m["scan"].return_value = ("SANITIZED_OUTPUT", "")
            result = _load_delegate_skills(["clean-skill"])
        self.assertEqual(result, "SANITIZED_OUTPUT")
        self.assertNotIn("# Clean Skill", result)

    def test_bump_use_failure_does_not_block_load(self):
        """bump_use is best-effort — failure must not prevent skill loading."""
        with ExitStack() as s:
            m = _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "# Resilient Skill",
            }))
            m["bump"].side_effect = RuntimeError("DB locked")
            result = _load_delegate_skills(["resilient-skill"])
        self.assertIn("# Resilient Skill", result)
        self.assertIn("resilient-skill", result)

    def test_empty_skill_content_still_injects_frame(self):
        """When skill_view returns success but empty content, the activation
        header is still injected so the child knows the skill was requested."""
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "",
            }))
            result = _load_delegate_skills(["empty-skill"])
        self.assertIn("empty-skill", result)
        self.assertIn("IMPORTANT", result)

    def test_mixed_success_and_failure(self):
        """A list with one successful + one missing skill produces both
        the successful content AND the skipped notice."""
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_side_effect=[
                json.dumps({"success": True, "content": "# Good Skill"}),
                json.dumps({"success": False, "error": "Not found"}),
            ])
            result = _load_delegate_skills(["good-skill", "bad-skill"])
        self.assertIn("# Good Skill", result)
        self.assertIn("bad-skill", result)
        self.assertIn("skipped", result)

    def test_normalize_skill_lookup_name_is_called(self):
        """normalize_skill_lookup_name must be invoked for each skill."""
        with ExitStack() as s:
            m = _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "# Skill",
            }))
            _load_delegate_skills(["my-skill"])
        m["norm"].assert_called_once_with("my-skill")

    def test_resolve_bundle_command_key_is_called(self):
        """resolve_bundle_command_key must be invoked for each skill name."""
        with ExitStack() as s:
            m = _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "# Skill",
            }))
            _load_delegate_skills(["my-skill"])
        m["bundle"].assert_called_once_with("my-skill")

    def test_ambiguous_skill_label_skips_with_notice(self):
        """A same-tier name collision is skipped with the ambiguous label
        (mirrors cron's _load_cron_skill_parts)."""
        with ExitStack() as s:
            m = _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": False, "load_names": ["a/my-skill", "b/my-skill"],
            }))
            m["ambiguous"].return_value = "Ambiguous skill name my-skill: use one of a/my-skill, b/my-skill"
            result = _load_delegate_skills(["my-skill"])
        self.assertIn("Ambiguous", result)
        self.assertIn("skipped", result)

    def test_inject_skill_config_is_called(self):
        """The loaded payload goes through _inject_skill_config like cron's
        loader (skill config vars ride the same pipeline)."""
        with ExitStack() as s:
            m = _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "# Skill",
            }))
            _load_delegate_skills(["my-skill"])
        m["config"].assert_called_once()


# ---------------------------------------------------------------------------
# _build_child_system_prompt
# ---------------------------------------------------------------------------

class TestBuildChildSystemPromptWithSkills(unittest.TestCase):
    """_build_child_system_prompt with skills_content injection."""

    def test_no_skills_content(self):
        prompt = _build_child_system_prompt("Do something")
        self.assertIn("Complete this task", prompt)
        # The goal is the child's first USER turn, not part of the prompt.
        self.assertNotIn("Do something", prompt)

    def test_with_skills_content(self):
        prompt = _build_child_system_prompt(
            "Do something",
            skills_content='[IMPORTANT: The "test-skill" skill.]\n# Test Skill\nFollow this.',
        )
        self.assertIn("test-skill", prompt)
        self.assertIn("# Test Skill", prompt)
        self.assertIn("Follow this.", prompt)

    def test_empty_or_whitespace_skills_ignored(self):
        for empty in ["", "   ", "\n\n  "]:
            prompt = _build_child_system_prompt("Do something", skills_content=empty)
            self.assertNotIn("IMPORTANT", prompt, f"failed for skills_content={empty!r}")

    def test_skills_before_completion(self):
        prompt = _build_child_system_prompt(
            "Do something", skills_content="[IMPORTANT: SKILL CONTENT HERE]",
        )
        self.assertGreater(prompt.find("SKILL CONTENT HERE"), 0)
        self.assertGreater(
            prompt.find("Complete this task"),
            prompt.find("SKILL CONTENT HERE"),
        )

    def test_skills_after_workspace_path(self):
        """skills_content should come after the WORKSPACE PATH block."""
        prompt = _build_child_system_prompt(
            "Do something",
            workspace_path="/tmp/does-not-exist-delegate-skills",
            skills_content="[IMPORTANT: SKILL HERE]",
        )
        ws_pos = prompt.find("WORKSPACE PATH")
        skill_pos = prompt.find("SKILL HERE")
        self.assertGreater(ws_pos, 0)
        self.assertGreater(skill_pos, ws_pos,
                           "skills should come after workspace path")

    def test_skills_with_orchestrator_role(self):
        """skills_content composes correctly with the orchestrator role block."""
        prompt = _build_child_system_prompt(
            "Orchestrate work", role="orchestrator", child_depth=1, max_spawn_depth=3,
            skills_content="[IMPORTANT: SKILL FOR ORCHESTRATOR]",
        )
        self.assertIn("SKILL FOR ORCHESTRATOR", prompt)
        self.assertIn("Subagent Spawning", prompt)


# ---------------------------------------------------------------------------
# _build_child_agent
# ---------------------------------------------------------------------------

class TestBuildChildAgentWithSkills(unittest.TestCase):
    """_build_child_agent with skills: capture the AIAgent constructor and
    assert ephemeral_system_prompt contains the loaded skill content
    (teknium1 review comment #3 — no blanket except: pass)."""

    def test_skills_content_in_child_prompt(self):
        """Child's ephemeral_system_prompt MUST contain skill content."""
        captured = {}
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "# Test Skill\nThis is the skill body.",
            }))
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init(captured)

            _build_child_agent(
                task_index=0, goal="Test the feature", context=None,
                toolsets=None, model=None, max_iterations=50,
                task_count=1, parent_agent=_make_mock_parent(),
                skills=["test-skill"],
            )

        prompt = captured["ephemeral_system_prompt"]
        self.assertIn("# Test Skill", prompt)
        self.assertIn("This is the skill body.", prompt)
        self.assertIn("test-skill", prompt)
        # The goal rides as the first user turn — it must NOT be in the prompt.
        self.assertNotIn("Test the feature", prompt)

    def test_no_skills_no_skill_block(self):
        """Without skills, no skill invocation block in the prompt."""
        captured = {}
        with ExitStack() as s:
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init(captured)

            _build_child_agent(
                task_index=0, goal="Test the feature", context=None,
                toolsets=None, model=None, max_iterations=50,
                task_count=1, parent_agent=_make_mock_parent(),
                skills=None,
            )

        prompt = captured.get("ephemeral_system_prompt", "")
        self.assertIn("Complete this task", prompt)
        self.assertNotIn("IMPORTANT: The user has invoked", prompt)

    def test_blocked_skill_notice_in_child_prompt(self):
        """When scan blocks skill content, the blocked notice reaches the
        child prompt so the child can surface it to the user."""
        captured = {}
        with ExitStack() as s:
            m = _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "malicious",
            }), scan_passthrough=False)
            m["scan"].return_value = ("cleaned", "Blocked: injection.")
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init(captured)

            _build_child_agent(
                task_index=0, goal="Do work", context=None,
                toolsets=None, model=None, max_iterations=50,
                task_count=1, parent_agent=_make_mock_parent(),
                skills=["bad-skill"],
            )

        prompt = captured.get("ephemeral_system_prompt", "")
        self.assertIn("bad-skill", prompt)
        self.assertIn("blocked", prompt.lower())


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchemaValidation(unittest.TestCase):
    """Schema includes the skills property with correct semantics."""

    def test_top_level_has_skills(self):
        p = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["skills"]
        self.assertEqual(p["type"], "array")
        self.assertEqual(p["items"]["type"], "string")

    def test_per_task_has_skills(self):
        p = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]["skills"]
        self.assertEqual(p["type"], "array")
        self.assertEqual(p["items"]["type"], "string")

    def test_descriptions_say_merge_not_replace(self):
        top = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["skills"]["description"].lower()
        per = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]["skills"]["description"].lower()
        for desc, label in [(top, "top-level"), (per, "per-task")]:
            self.assertIn("merge", desc, f"{label} must say 'merge'")
            self.assertNotIn("replace", desc, f"{label} must NOT say 'replace'")

    def test_top_level_mentions_security_pipeline(self):
        desc = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["skills"]["description"].lower()
        self.assertIn("resolution pipeline", desc)
        self.assertIn("security scan", desc)


# ---------------------------------------------------------------------------
# Integration: delegate_task end-to-end
# ---------------------------------------------------------------------------

class TestDelegateTaskIntegrationWithSkills(unittest.TestCase):
    """End-to-end: delegate_task(...) threads skills through the full path:
    schema -> registry/run_agent dispatch -> delegate_task -> _build_children
    -> _build_child_agent -> _load_delegate_skills -> _build_child_system_prompt.
    """

    def test_single_task_with_skills(self):
        """delegate_task with skills= produces a child whose prompt contains
        the skill content."""
        captured = {}
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "# Integration Skill",
            }))
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init(captured, _COMPLETED_CHILD_RESULT)
            _patch_delegate_env(s)

            result = delegate_task(
                goal="Do the work",
                skills=["integration-skill"],
                parent_agent=_make_mock_parent(),
            )

        result_data = json.loads(result)
        self.assertTrue(result_data.get("results"))
        self.assertEqual(result_data["results"][0]["status"], "completed")
        prompt = captured.get("ephemeral_system_prompt", "")
        self.assertIn("# Integration Skill", prompt)
        self.assertIn("integration-skill", prompt)

    def test_single_task_without_skills_backward_compat(self):
        """delegate_task without skills= still works (backward compat)."""
        captured = {}
        with ExitStack() as s:
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init(captured, _COMPLETED_CHILD_RESULT)
            _patch_delegate_env(s)

            result = delegate_task(
                goal="Simple task",
                parent_agent=_make_mock_parent(),
            )

        result_data = json.loads(result)
        self.assertTrue(result_data.get("results"))
        prompt = captured.get("ephemeral_system_prompt", "")
        self.assertNotIn("IMPORTANT: The user has invoked", prompt)

    def test_batch_merges_top_level_and_per_task_skills(self):
        """Batch mode: per-task skills are merged with top-level skills
        (per-task first), and EACH child's prompt gets its own merged set."""
        captured = {}
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_side_effect=lambda name: json.dumps({
                "success": True, "content": f"# Skill {name}",
            }))
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init(captured, _COMPLETED_CHILD_RESULT)
            _patch_delegate_env(s)

            result = delegate_task(
                tasks=[
                    {"goal": "First batch task with skills", "skills": ["skill-a"]},
                    {"goal": "Second batch task with skills", "skills": ["skill-b"]},
                ],
                skills=["shared-skill"],
                parent_agent=_make_mock_parent(),
            )

        result_data = json.loads(result)
        self.assertEqual(len(result_data["results"]), 2)
        calls = captured["calls"]
        self.assertEqual(len(calls), 2)
        first_prompt, second_prompt = calls[0]["ephemeral_system_prompt"], calls[1]["ephemeral_system_prompt"]
        # Per-task skill first, then the top-level one — in each child's prompt.
        self.assertIn("# Skill skill-a", first_prompt)
        self.assertIn("# Skill shared-skill", first_prompt)
        self.assertNotIn("# Skill skill-b", first_prompt)
        self.assertIn("# Skill skill-b", second_prompt)
        self.assertIn("# Skill shared-skill", second_prompt)
        self.assertNotIn("# Skill skill-a", second_prompt)

    def test_bad_skills_fail_before_spawn(self):
        """A malformed skills value fails the whole call before any child
        spawns (same fail-fast contract as output_schema/images). A bare
        STRING is not malformed — it is wrapped, matching cron's
        _canonical_skills and the images precedent (see the wrapping test)."""
        with ExitStack() as s:
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init({}, _COMPLETED_CHILD_RESULT)
            _patch_delegate_env(s)

            result = delegate_task(
                goal="Never spawned at all",
                skills=42,
                parent_agent=_make_mock_parent(),
            )

        result_data = json.loads(result)
        self.assertIn("error", result_data)
        self.assertIn("array of skill names", result_data["error"])
        mock_agent.assert_not_called()

    def test_bad_per_task_skills_fail_before_spawn(self):
        """A malformed per-task skills list fails the whole call before any
        child spawns."""
        with ExitStack() as s:
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init({}, _COMPLETED_CHILD_RESULT)
            _patch_delegate_env(s)

            result = delegate_task(
                tasks=[{"goal": "Never spawned at all", "skills": {"not": "a list"}}],
                parent_agent=_make_mock_parent(),
            )

        result_data = json.loads(result)
        self.assertIn("error", result_data)
        mock_agent.assert_not_called()

    def test_bare_string_skills_is_wrapped(self):
        """A bare string skills value is wrapped into a one-entry list,
        matching cron's _canonical_skills and the images precedent."""
        captured = {}
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_side_effect=lambda name: json.dumps({
                "success": True, "content": f"# Skill {name}",
            }))
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init(captured, _COMPLETED_CHILD_RESULT)
            _patch_delegate_env(s)

            result = delegate_task(
                goal="Wrap the bare string skill",
                skills="wrapped-skill",
                parent_agent=_make_mock_parent(),
            )

        result_data = json.loads(result)
        self.assertTrue(result_data.get("results"))
        prompt = captured.get("ephemeral_system_prompt", "")
        self.assertIn("# Skill wrapped-skill", prompt)


# ---------------------------------------------------------------------------
# Async ledger: skills must survive the task_json round-trip
# ---------------------------------------------------------------------------

class TestAsyncLedgerSkillsRoundTrip(unittest.TestCase):
    """`skills` rides the dispatch record into the durable ledger
    (async_delegations.task_json) so a restart-recovered event can report
    which skills an abandoned delegation was running with — the parent can
    re-dispatch from the event alone. Skill CONTENT never goes to the ledger:
    it is baked into the child's prompt before dispatch."""

    def test_dispatch_persists_skills_in_task_json(self):
        from tools import async_delegation as ad
        ad._reset_for_tests()
        try:
            handle = ad.dispatch_async_delegation_batch(
                goals=["Ledger round-trip goal one"],
                context=None, toolsets=None, role="leaf", model=None,
                session_key="", runner=lambda: {"results": [], "total_duration_seconds": 0.0},
                skills=[["ledger-skill"], None],
            )
            self.assertEqual(handle["status"], "dispatched")
            with ad._DB_LOCK, ad._transaction() as conn:
                row = conn.execute(
                    "SELECT task_json FROM async_delegations WHERE delegation_id=?",
                    (handle["delegation_id"],),
                ).fetchone()
            task = json.loads(row[0])
            self.assertEqual(task["skills"], [["ledger-skill"], None])
        finally:
            ad._reset_for_tests()

    def test_recovery_event_reconstructs_skills(self):
        from tools import async_delegation as ad
        ad._reset_for_tests()
        try:
            ad._persist_dispatch({
                "delegation_id": "skills-recovery", "goal": "Recovery goal",
                "goals": ["Recovery goal"], "context": None, "role": "leaf",
                "model": None, "skills": [["recovery-skill"]],
                "dispatched_at": 1.0, "session_key": "",
                "parent_session_id": None, "origin_session_id": "",
            })
            # The dispatching process is alive by construction — mark the row's
            # owner dead so recover_abandoned_delegations() picks it up.
            with ad._DB_LOCK, ad._transaction() as conn:
                conn.execute(
                    "UPDATE async_delegations SET owner_pid=? WHERE delegation_id=?",
                    (999999, "skills-recovery"),
                )
            recovered = ad.recover_abandoned_delegations()
            self.assertEqual(recovered, 1)
            with ad._DB_LOCK, ad._transaction() as conn:
                row = conn.execute(
                    "SELECT event_json FROM async_delegations WHERE delegation_id=?",
                    ("skills-recovery",),
                ).fetchone()
            event = json.loads(row[0])
            self.assertEqual(event["skills"], [["recovery-skill"]])
            self.assertEqual(event["status"], "unknown")
        finally:
            ad._reset_for_tests()


if __name__ == "__main__":
    unittest.main()
