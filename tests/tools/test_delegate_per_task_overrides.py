#!/usr/bin/env python3
"""
Tests for per-task model selection and profile identity in delegate_task.

Two opt-in features, both gated behind config flags (default off):

1. **Per-task model selection** (``delegation.allow_model_selection``):
   The agent names a model per task ("opus", "gpt-5", "glm") when fanning
   work out; resolution reuses the shared ``model_switch`` pipeline so names
   are matched leniently and the provider is resolved (not dictated).

2. **Per-task profile identity** (``delegation.allow_profile_identity``):
   The agent names a Hermes profile per task; the child loads that profile's
   SOUL.md, IDENTITY.md, and AGENTS.md as its system prompt identity, and
   reads model/provider from its config.yaml when not explicitly overridden.
   The child "becomes" the named bot rather than a generic subagent.

Both flags are off by default to preserve the "subagents inherit the parent
model and use a generic identity" contract. The schema fields only appear
when the corresponding flag is enabled (keeping the tool surface minimal).

Run with:  python3 -m pytest tests/tools/test_delegate_per_task_overrides.py -v
"""

import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_dynamic_schema_overrides,
    _get_allow_model_selection,
    _get_allow_profile_identity,
    _resolve_task_model_creds,
    _load_profile_identity,
    _build_child_system_prompt,
    delegate_task,
    set_spawn_paused,
)


class _FakeParent:
    """Minimal parent agent for credential anchoring."""

    provider = "openrouter"
    model = "anthropic/claude-opus-4.8"
    base_url = "https://openrouter.ai/api/v1"
    api_key = "sk-test"


_BASE_CREDS = {
    "model": None,
    "provider": None,
    "base_url": None,
    "api_key": None,
    "api_mode": None,
    "command": None,
    "args": None,
}


# ---------------------------------------------------------------------------
# Schema gating: fields only appear when the flag is on
# ---------------------------------------------------------------------------

class TestSchemaGating(unittest.TestCase):
    """Per-task fields only appear when the corresponding flag is enabled."""

    def test_both_flags_off_no_extra_fields(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertNotIn("model", props)
        self.assertNotIn("profile", props)
        task_props = props["tasks"]["items"]["properties"]
        self.assertNotIn("model", task_props)
        self.assertNotIn("profile", task_props)

    def test_model_flag_on_adds_model_field(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_model_selection": True},
        ):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertIn("model", props)
        self.assertEqual(props["model"]["type"], "string")
        self.assertIn("model", props["tasks"]["items"]["properties"])

    def test_profile_flag_on_adds_profile_field(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_profile_identity": True},
        ):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertIn("profile", props)
        self.assertEqual(props["profile"]["type"], "string")
        self.assertIn("profile", props["tasks"]["items"]["properties"])

    def test_both_flags_on_adds_both_fields(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={
                "allow_model_selection": True,
                "allow_profile_identity": True,
            },
        ):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertIn("model", props)
        self.assertIn("profile", props)
        task_props = props["tasks"]["items"]["properties"]
        self.assertIn("model", task_props)
        self.assertIn("profile", task_props)

    def test_static_schema_never_mutated(self):
        """Dynamic overrides must not leak into the static schema."""
        with patch(
            "tools.delegate_tool._load_config",
            return_value={
                "allow_model_selection": True,
                "allow_profile_identity": True,
            },
        ):
            _build_dynamic_schema_overrides()
        static_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertNotIn("model", static_props)
        self.assertNotIn("profile", static_props)
        self.assertNotIn(
            "model", static_props["tasks"]["items"]["properties"]
        )
        self.assertNotIn(
            "profile", static_props["tasks"]["items"]["properties"]
        )


# ---------------------------------------------------------------------------
# Flag getters
# ---------------------------------------------------------------------------

class TestFlagGetters(unittest.TestCase):
    def test_model_selection_default_off(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            self.assertFalse(_get_allow_model_selection())

    def test_model_selection_truthy_on(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_model_selection": True},
        ):
            self.assertTrue(_get_allow_model_selection())

    def test_profile_identity_default_off(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            self.assertFalse(_get_allow_profile_identity())

    def test_profile_identity_truthy_on(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_profile_identity": True},
        ):
            self.assertTrue(_get_allow_profile_identity())


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestModelResolution(unittest.TestCase):
    """`_resolve_task_model_creds` reuses the model_switch pipeline."""

    def test_empty_name_returns_base_unchanged(self):
        out = _resolve_task_model_creds("", _FakeParent(), _BASE_CREDS)
        self.assertIs(out, _BASE_CREDS)

    def test_base_creds_not_mutated(self):
        before = dict(_BASE_CREDS)
        _resolve_task_model_creds("", _FakeParent(), _BASE_CREDS)
        self.assertEqual(_BASE_CREDS, before)


# ---------------------------------------------------------------------------
# Profile identity loading
# ---------------------------------------------------------------------------

class TestProfileIdentityLoading(unittest.TestCase):
    """`_load_profile_identity` reads identity files from a profile directory."""

    def test_nonexistent_profile_returns_none(self):
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the profiles root but not the named profile
            pathlib.Path(tmpdir, "profiles").mkdir(parents=True)
            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ):
                result = _load_profile_identity("nonexistent-profile")
        self.assertIsNone(result)

    def test_loads_identity_files_and_config(self):
        """A well-formed profile returns soul, identity, agents, model, provider."""
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = pathlib.Path(tmpdir) / "profiles" / "test-bot"
            profile_dir.mkdir(parents=True)
            (profile_dir / "SOUL.md").write_text("You are a test bot.")
            (profile_dir / "IDENTITY.md").write_text("Name: TestBot")
            (profile_dir / "AGENTS.md").write_text("# Test Agent Rules")
            (profile_dir / "config.yaml").write_text(
                "model:\n  default: glm-5.3\n  provider: ollama-cloud\n"
            )

            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ):
                result = _load_profile_identity("test-bot")

        self.assertIsNotNone(result)
        self.assertEqual(result["soul"], "You are a test bot.")
        self.assertEqual(result["identity"], "Name: TestBot")
        self.assertEqual(result["agents"], "# Test Agent Rules")
        self.assertEqual(result["model"], "glm-5.3")
        self.assertEqual(result["provider"], "ollama-cloud")

    def test_partial_profile_still_loads(self):
        """A profile with only SOUL.md (no IDENTITY) still returns a valid dict."""
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = pathlib.Path(tmpdir) / "profiles" / "partial-bot"
            profile_dir.mkdir(parents=True)
            (profile_dir / "SOUL.md").write_text("You are partial.")

            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ):
                result = _load_profile_identity("partial-bot")

        self.assertIsNotNone(result)
        self.assertEqual(result["soul"], "You are partial.")
        self.assertIsNone(result["identity"])
        self.assertIsNone(result["agents"])
        self.assertIsNone(result["model"])
        self.assertIsNone(result["provider"])

    def test_path_traversal_rejected(self):
        """Profile names with slashes or dots are rejected to prevent traversal."""
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory outside profiles that a traversal would hit
            secret_dir = pathlib.Path(tmpdir) / "secret"
            secret_dir.mkdir()
            (secret_dir / "SOUL.md").write_text("SECRET DATA")

            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ):
                # All of these should return None, never reading the secret
                for bad_name in [
                    "../../secret",
                    "../secret",
                    "foo/../../secret",
                    ".../.../secret",
                    "foo/../bar",
                    "a/b",
                    "a.b",
                    ".hidden",
                    "trailing/",
                    "/absolute",
                ]:
                    with self.subTest(name=bad_name):
                        result = _load_profile_identity(bad_name)
                        self.assertIsNone(
                            result,
                            f"Path traversal with '{bad_name}' should return None",
                        )


# ---------------------------------------------------------------------------
# System prompt with profile identity
# ---------------------------------------------------------------------------

class TestChildSystemPromptWithProfile(unittest.TestCase):
    """`_build_child_system_prompt` uses profile identity when provided."""

    def test_profile_identity_replaces_generic_preamble(self):
        identity = {
            "soul": "You are a code reviewer.",
            "identity": "Name: ReviewerBot",
            "agents": "# Review Rules",
            "model": None,
            "provider": None,
        }
        prompt = _build_child_system_prompt(
            "Review the PR", profile_identity=identity
        )
        self.assertIn("You are a code reviewer.", prompt)
        self.assertIn("Name: ReviewerBot", prompt)
        self.assertIn("# Review Rules", prompt)
        self.assertIn("YOUR TASK:\nReview the PR", prompt)
        self.assertNotIn("You are a focused subagent", prompt)

    def test_no_profile_identity_uses_generic_preamble(self):
        prompt = _build_child_system_prompt("Fix the tests")
        self.assertIn("You are a focused subagent", prompt)
        self.assertIn("YOUR TASK:\nFix the tests", prompt)

    def test_partial_profile_warns_and_uses_agents(self):
        """Profile with only AGENTS.md (no SOUL/IDENTITY) falls back gracefully."""
        identity = {
            "soul": None,
            "identity": None,
            "agents": "# Agent Rules Only",
            "model": None,
            "provider": None,
        }
        with patch("tools.delegate_tool.logger") as mock_logger:
            prompt = _build_child_system_prompt(
                "Do the work", profile_identity=identity
            )
        mock_logger.warning.assert_called_once()
        self.assertIn("You are a focused subagent", prompt)
        self.assertIn("# Agent Rules Only", prompt)


# ---------------------------------------------------------------------------
# P1 regressions (PR #98031 review)
# ---------------------------------------------------------------------------


def _make_mock_parent():
    """Mock parent with every field delegate_task / _build_child_agent touch.

    Mirrors the established pattern from tests/tools/test_delegate_test_gap.py.
    """
    parent = MagicMock()
    parent.session_id = "parent-session-overrides"
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-opus-4.8"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.provider_require_parameters = False
    parent.provider_data_collection = None
    parent.request_overrides = {}
    parent.max_tokens = None
    parent.enabled_toolsets = None
    parent.valid_tool_names = []
    parent.disabled_toolsets = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._subagent_finalization_lock = threading.RLock()
    parent._current_task_id = None
    parent._current_turn_id = ""
    parent._memory_manager = None
    parent._print_fn = None
    parent.tool_progress_callback = None
    return parent


# Long enough to clear the batch goal-quality gate (_MIN_BATCH_GOAL_LEN=10).
_GOAL_A = "Do the first delegated task end to end"
_GOAL_B = "Do the second delegated task end to end"


class TestAtomicBatchPreflight(unittest.TestCase):
    """P1 #1: a mid-batch resolution failure must construct ZERO children.

    The dispatcher preflights every task (model resolution, profile
    loading, credential bundling) before constructing any child, so a
    failure on task 1 can never orphan task 0's already-constructed child
    (open SessionDB, registration in _active_children) with no cleanup.
    """

    def setUp(self):
        set_spawn_paused(False)

    def test_invalid_profile_in_later_task_constructs_no_children(self):
        """Task 0 valid + task 1 invalid profile → error, zero children,
        no SessionDB construction side effect."""
        parent = _make_mock_parent()
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_profile_identity": True},
        ), patch(
            "tools.delegate_tool._load_profile_identity",
            side_effect=lambda name: (
                {
                    "soul": "You are Devbot.",
                    "identity": None,
                    "agents": None,
                    "model": None,
                    "provider": None,
                    "_profile_name": name,
                }
                if name == "devbot"
                else None
            ),
        ), patch("run_agent.AIAgent") as MockAgent, patch(
            "hermes_state.SessionDB"
        ) as MockSessionDB:
            out = delegate_task(
                tasks=[
                    {"goal": _GOAL_A, "profile": "devbot"},
                    {"goal": _GOAL_B, "profile": "no-such-profile-xyz"},
                ],
                parent_agent=parent,
            )
        payload = json.loads(out)
        # The whole batch is refused with a per-task error...
        self.assertIn("error", payload)
        self.assertIn("Task 1", payload["error"])
        self.assertIn("no-such-profile-xyz", payload["error"])
        # ...with ZERO children constructed.
        MockAgent.assert_not_called()
        self.assertEqual(len(parent._active_children), 0)
        # And no child SessionDB handle was opened (construction never ran).
        MockSessionDB.assert_not_called()


class TestProfileProviderRouting(unittest.TestCase):
    """P1 #2: a profile's model+provider resolve as one routing unit —
    against the PROFILE's provider, never the parent's."""

    def setUp(self):
        set_spawn_paused(False)

    def test_profile_provider_used_not_parent_provider(self):
        """Parent on openrouter + profile configured for openai/gpt-5 →
        the child receives openai credentials, not openrouter's."""
        parent = _make_mock_parent()

        fake_runtime = {
            "provider": "openai",
            "model": "gpt-5",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-openai",
            "api_mode": "chat_completions",
            "request_overrides": {},
            "max_output_tokens": None,
            "command": None,
            "args": [],
        }
        with patch(
            "tools.delegate_tool._load_config",
            return_value={
                "allow_profile_identity": True,
                "allow_model_selection": True,
            },
        ), patch(
            "tools.delegate_tool._load_profile_identity",
            return_value={
                "soul": "You are Devbot.",
                "identity": None,
                "agents": None,
                "model": "gpt-5",
                "provider": "openai",
                "_profile_name": "devbot",
            },
        ), patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ) as mock_rrp, patch("run_agent.AIAgent") as MockAgent:
            child = MagicMock()
            child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "tokens": {"input": 1, "output": 1},
            }
            MockAgent.return_value = child
            out = delegate_task(
                tasks=[{"goal": _GOAL_A, "profile": "devbot"}],
                parent_agent=parent,
            )
        payload = json.loads(out)
        self.assertNotIn("error", payload)
        # Resolution anchored on the PROFILE's provider...
        mock_rrp.assert_called_once()
        self.assertEqual(
            mock_rrp.call_args.kwargs.get("requested"), "openai"
        )
        # ...and the child was constructed with the openai routing unit.
        kwargs = MockAgent.call_args.kwargs
        self.assertEqual(kwargs["provider"], "openai")
        self.assertEqual(kwargs["model"], "gpt-5")
        self.assertEqual(kwargs["base_url"], "https://api.openai.com/v1")
        self.assertEqual(kwargs["api_key"], "sk-openai")
        self.assertNotEqual(kwargs["provider"], "openrouter")
        # Identity still applied alongside the provider routing.
        self.assertIn("You are Devbot", kwargs["ephemeral_system_prompt"])


class TestOverlongProfileName(unittest.TestCase):
    """P1 #3: a regex-valid name longer than the filesystem component
    limit returns None (profile not found), never an OSError."""

    def test_overlong_valid_name_returns_none_without_raising(self):
        long_name = "a" * 256  # matches ^[A-Za-z0-9_-]+$ but > 255 bytes
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ) as mock_root:
                # Must return None, not raise OSError(ENAMETOOLONG).
                result = _load_profile_identity(long_name)
            self.assertIsNone(result)
            # Rejected at the guard — no filesystem access needed.
            mock_root.assert_not_called()

    def test_name_at_255_limit_still_accepted_by_guard(self):
        """A 255-char name is legal: it passes the guard and fails the
        directory lookup gracefully (None, no crash)."""
        name_255 = "a" * 255
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ) as mock_root:
                result = _load_profile_identity(name_255)
            self.assertIsNone(result)  # no such profile on disk
            mock_root.assert_called_once()  # guard let it through


if __name__ == "__main__":
    unittest.main()