#!/usr/bin/env python3
"""Tests for the delegate_task skills parameter.

Covers: _merge_skills, _load_delegate_skills (cron pipeline), prompt
injection, _build_child_agent AIAgent constructor capture, schema,
and end-to-end integration through delegate_task().

Run:  python -m pytest tests/tools/test_delegate_skills.py -v
"""

import json
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
    mocks["norm"] = stack.enter_context(
        patch("agent.skill_utils.normalize_skill_lookup_name", side_effect=lambda x: x)
    )
    mocks["scan"] = stack.enter_context(
        patch("tools.cronjob_tools._scan_cron_skill_assembled")
    )
    if scan_passthrough:
        mocks["scan"].side_effect = lambda assembled: (assembled, "")
    return mocks


def _patch_child_agent_env(stack):
    """Patch _build_child_agent's external dependencies."""
    stack.enter_context(patch("tools.delegate_tool._load_config", return_value={}))
    stack.enter_context(patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
        "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "model": None, "request_overrides": None,
        "max_output_tokens": None, "command": None, "args": None,
    }))
    stack.enter_context(patch("tools.delegate_tool._get_max_spawn_depth", return_value=2))
    stack.enter_context(patch("tools.delegate_tool._get_orchestrator_enabled", return_value=True))
    stack.enter_context(patch("tools.delegate_tool._resolve_workspace_hint", return_value=None))
    stack.enter_context(patch("tools.delegate_tool._strip_blocked_tools", return_value=set()))
    mock_agent = stack.enter_context(patch("run_agent.AIAgent"))
    return mock_agent


def _make_mock_parent():
    parent = MagicMock()
    parent.api_key = "test-key"
    parent.model = "test-model"
    parent.enabled_toolsets = None
    parent.valid_tool_names = []
    parent._delegate_depth = 0
    parent._subagent_id = None
    parent._delegate_spinner = None
    parent.tool_progress_callback = None
    parent._client_kwargs = {}
    return parent


def _capture_init(captured):
    """Return a side_effect for AIAgent that captures kwargs into *captured*."""
    def _init(*args, **kwargs):
        captured.update(kwargs)
        inst = MagicMock()
        inst.tool_progress_callback = kwargs.get("tool_progress_callback")
        inst._delegate_depth = kwargs.get("_delegate_depth", 0)
        return inst
    return _init


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
            _patch_load_pipeline(s, bundle_key_return="/my-bundle")
            s.enter_context(patch(
                "agent.skill_bundles.build_bundle_invocation_message",
                return_value=("Bundle content", [], []),
            ))
            with patch("tools.skills_tool.skill_view") as mock_view:
                result = _load_delegate_skills(["my-bundle"])
        self.assertIn("Bundle content", result)
        mock_view.assert_not_called()

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


# ---------------------------------------------------------------------------
# _build_child_system_prompt
# ---------------------------------------------------------------------------

class TestBuildChildSystemPromptWithSkills(unittest.TestCase):
    """_build_child_system_prompt with skills_content injection."""

    def test_no_skills_content(self):
        prompt = _build_child_system_prompt("Do something")
        self.assertIn("YOUR TASK:\nDo something", prompt)
        self.assertIn("Complete this task", prompt)

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
        """skills_content should come after WORKSPACE PATH block."""
        prompt = _build_child_system_prompt(
            "Do something",
            workspace_path="/tmp/project",
            skills_content="[IMPORTANT: SKILL HERE]",
        )
        ws_pos = prompt.find("WORKSPACE PATH")
        skill_pos = prompt.find("SKILL HERE")
        self.assertGreater(ws_pos, 0)
        self.assertGreater(skill_pos, ws_pos,
                           "skills should come after workspace path")

    def test_skills_with_orchestrator_role(self):
        """skills_content composes correctly with orchestrator role block."""
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
    """_build_child_agent with skills: capture AIAgent constructor, assert
    ephemeral_system_prompt contains loaded skill content (teknium1 #3)."""

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
        self.assertIn("Test the feature", prompt)
        self.assertIn("# Test Skill", prompt)
        self.assertIn("This is the skill body.", prompt)

    def test_no_skills_no_skill_block(self):
        """Without skills, no skill invocation block in prompt."""
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
        self.assertIn("Test the feature", prompt)
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
    """Schema includes skills property with correct semantics."""

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
    """End-to-end test: delegate_task(goal=..., skills=[...]) threads skills
    through the full path: schema → registry handler → delegate_task →
    _build_child_agent → _load_delegate_skills → _build_child_system_prompt.
    """

    def _patch_delegate_env(self, stack):
        """Patch delegate_task's runtime environment for integration tests."""
        stack.enter_context(patch("tools.delegate_tool.is_spawn_paused", return_value=False))
        stack.enter_context(patch("tools.delegate_tool._get_max_concurrent_children", return_value=3))
        stack.enter_context(patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])))
        stack.enter_context(patch("tools.delegation_live_log.update_manifest_statuses"))
        stack.enter_context(patch("tools.delegation_live_log.wrap_progress_callback", side_effect=lambda cb, w: cb))
        import model_tools
        stack.enter_context(patch.object(model_tools, "_last_resolved_tool_names", []))
        # _run_single_child returns a result dict — patch it to avoid running
        # the child agent's full conversation loop.
        stack.enter_context(patch("tools.delegate_tool._run_single_child", return_value={
            "status": "completed", "summary": "mock result", "output": "mock output",
        }))

    def test_single_task_with_skills(self):
        """delegate_task with skills= produces a child whose prompt contains
        the skill content."""
        captured = {}
        with ExitStack() as s:
            _patch_load_pipeline(s, skill_view_return=json.dumps({
                "success": True, "content": "# Integration Skill",
            }))
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init(captured)
            self._patch_delegate_env(s)

            result = delegate_task(
                goal="Do the work",
                skills=["integration-skill"],
                parent_agent=_make_mock_parent(),
            )

        result_data = json.loads(result)
        self.assertTrue(result_data.get("results"))
        prompt = captured.get("ephemeral_system_prompt", "")
        self.assertIn("Do the work", prompt)
        self.assertIn("# Integration Skill", prompt)
        self.assertIn("integration-skill", prompt)

    def test_single_task_without_skills_backward_compat(self):
        """delegate_task without skills= still works (backward compat)."""
        captured = {}
        with ExitStack() as s:
            mock_agent = _patch_child_agent_env(s)
            mock_agent.side_effect = _capture_init(captured)
            self._patch_delegate_env(s)

            result = delegate_task(
                goal="Simple task",
                parent_agent=_make_mock_parent(),
            )

        result_data = json.loads(result)
        self.assertTrue(result_data.get("results"))
        prompt = captured.get("ephemeral_system_prompt", "")
        self.assertIn("Simple task", prompt)
        self.assertNotIn("IMPORTANT: The user has invoked", prompt)


if __name__ == "__main__":
    unittest.main()
