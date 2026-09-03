#!/usr/bin/env python3
"""
Test-gap closure for per-task model selection + profile identity (Card 4).

Four categories, all test-only (no production changes):

  1. Property-based (Hypothesis) fuzzing of the profile-name path traversal
     guard ``^[A-Za-z0-9_-]+$`` in ``_load_profile_identity`` — arbitrary
     strings, adversarial unicode / null bytes / control chars / extremely
     long names / path separators on both POSIX and Windows.
  2. Regression: ``allow_model_selection=False`` + ``allow_profile_identity=True``
     must gate the schema (profile advertised, model NOT) and must NOT let a
     profile's config.yaml model trigger a model switch.
  3. Fallback chain semantics: unresolvable models raise ValueError (fail
     loudly, never silently inherit); None/"" model names are no-ops that
     return the base creds untouched.
  4. Integration with mock child spawns: profile identity reaches the child
     system prompt, no profile uses the generic preamble, and explicit model
     beats the profile model.

Run with:  python3 -m pytest tests/tools/test_delegate_test_gap.py -v
"""

import json
import pathlib
import re
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, strategies as st

from tools.delegate_tool import (
    _build_dynamic_schema_overrides,
    _get_allow_model_selection,
    _get_allow_profile_identity,
    _load_profile_identity,
    _resolve_task_model_creds,
    delegate_task,
    set_spawn_paused,
)
# The exact guard regex _load_profile_identity enforces (delegate_tool.py).
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Shared goal for integration dispatches (long enough to clear the batch
# goal-quality gate even if the shape ever changes to multi-task).
GOAL = "Do the delegated integration work end to end"


# ── Shared fixtures ───────────────────────────────────────────────────────


class _FakeParent:
    """Minimal parent agent for credential anchoring (model_switch path)."""

    provider = "openrouter"
    model = "anthropic/claude-opus-4.8"
    base_url = "https://openrouter.ai/api/v1"
    api_key = "sk-test"


def _base_creds():
    """Fresh base credential bundle shaped like _resolve_delegation_credentials."""
    return {
        "model": None,
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }


def _make_switch_result(
    new_model="gpt-5x",
    target_provider="openai",
    success=True,
    error_message=None,
):
    """Fake ModelSwitchResult like hermes_cli.model_switch returns."""
    r = MagicMock()
    r.success = success
    r.error_message = error_message
    r.new_model = new_model
    r.target_provider = target_provider
    r.base_url = "https://api.openai.com/v1"
    r.api_key = "sk-openai"
    r.api_mode = "chat_completions"
    return r


def _identity_stub(
    soul="You are Devbot — a build agent.",
    identity="Name: Devbot",
    agents=None,
    model=None,
    provider=None,
    profile_name="devbot",
):
    """Profile identity dict shaped like _load_profile_identity output."""
    return {
        "soul": soul,
        "identity": identity,
        "agents": agents,
        "model": model,
        "provider": provider,
        "_profile_name": profile_name,
    }


def _make_mock_parent():
    """Mock parent with every field delegate_task / _build_child_agent touch.

    Mirrors the established pattern from tests/tools/test_delegate.py's
    _make_mock_parent.
    """
    parent = MagicMock()
    parent.session_id = "parent-session-gap"
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
    parent.thinking_callback = None
    # Usage-ledger fields (Card 2 contract).
    parent.session_delegation_usage = []
    parent.session_estimated_cost_usd = 0.0
    parent.session_cost_source = "none"
    parent.session_cost_status = "unknown"
    return parent


def _make_mock_child():
    """Mock child AIAgent whose run_conversation completes immediately."""
    child = MagicMock()
    child.run_conversation.return_value = {
        "final_response": "done",
        "completed": True,
        "interrupted": False,
        "api_calls": 2,
        "tokens": {"input": 100, "output": 50},
    }
    child._credential_pool = None
    return child


# ── Category 1: Property-based path traversal tests (Hypothesis) ─────────


class TestPathTraversalPropertyBased(unittest.TestCase):
    """Property-based fuzzing of the ^[A-Za-z0-9_-]+$ profile-name guard."""

    @settings(max_examples=100, deadline=None)
    @given(st.text(max_size=100))
    def test_invalid_names_rejected_before_filesystem_access(self, name):
        """Any string with characters outside [A-Za-z0-9_-] must return None,
        and must be rejected BEFORE the profiles root is ever consulted (the
        guard is the security control; the directory check is not)."""
        if _SAFE_NAME_RE.match(name or ""):
            return  # valid names are covered by the next property
        with patch(
            "hermes_constants.get_default_hermes_root"
        ) as mock_root:
            result = _load_profile_identity(name)
            self.assertIsNone(
                result,
                f"profile name {name!r} contains characters outside "
                "[A-Za-z0-9_-] and must be rejected",
            )
            mock_root.assert_not_called()

    @settings(max_examples=50, deadline=None)
    @given(st.from_regex(r"[A-Za-z0-9_-]{1,64}", fullmatch=True))
    def test_valid_names_pass_guard_and_load(self, name):
        """Names matching the safe regex must never be rejected by the guard:
        a real on-disk profile with that exact name loads successfully."""
        tmpdir = tempfile.mkdtemp()
        try:
            profile_dir = pathlib.Path(tmpdir) / "profiles" / name
            profile_dir.mkdir(parents=True)
            (profile_dir / "SOUL.md").write_text(
                "gap-test soul", encoding="utf-8"
            )
            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ):
                result = _load_profile_identity(name)
            self.assertIsNotNone(
                result,
                f"valid profile name {name!r} must not be rejected "
                "by the guard (false positive)",
            )
            self.assertEqual(result["soul"], "gap-test soul")
            self.assertEqual(result["_profile_name"], name)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_adversarial_traversal_names_rejected(self):
        """Hand-picked adversarial inputs (spec list) must all return None,
        with a real secret directory sitting outside the profiles root that a
        broken guard would read."""
        adversarial = [
            "../", "..\\", ".../", "./", ".\\", "/etc", "\\windows",
            "foo/../../bar", "a/b/c", "a.b.c", " ", "  ", "\t", "\n",
            "null\x00byte",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            secret = pathlib.Path(tmpdir) / "secret"
            secret.mkdir()
            (secret / "SOUL.md").write_text("SECRET DATA", encoding="utf-8")
            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ) as mock_root:
                for bad in adversarial:
                    with self.subTest(name=bad):
                        self.assertIsNone(
                            _load_profile_identity(bad),
                            f"adversarial name {bad!r} must be rejected",
                        )
                mock_root.assert_not_called()

    def test_unicode_names_rejected(self):
        """Non-ASCII codepoints must never satisfy the guard."""
        for bad in ["böt", "日本語", "🤖-bot", "pro\u00e9file", "профиль", "‮rtl"]:
            with self.subTest(name=bad):
                with patch(
                    "hermes_constants.get_default_hermes_root"
                ) as mock_root:
                    self.assertIsNone(_load_profile_identity(bad))
                    mock_root.assert_not_called()

    def test_control_characters_rejected(self):
        """Null bytes and control characters must be rejected at the guard."""
        for bad in ["\x00", "a\x00b", "null\x00byte", "a\x01b", "a\x1fb",
                    "a\x7fb"]:
            with self.subTest(name=bad):
                with patch(
                    "hermes_constants.get_default_hermes_root"
                ) as mock_root:
                    self.assertIsNone(_load_profile_identity(bad))
                    mock_root.assert_not_called()

    def test_extremely_long_names(self):
        """Extremely long names: an INVALID 5000-char name is rejected at the
        guard before any filesystem access; a long-but-filename-legal VALID
        name (240 chars) passes the guard and fails the directory lookup
        gracefully (None, no crash).

        KNOWN FINDING (not fixed here — production code is out of scope for
        this card): a VALID name longer than the OS filename limit (~255
        bytes) raises OSError ENAMETOOLONG from profile_path.is_dir() instead
        of returning None. Reported in the card summary.
        """
        # Invalid-and-huge: rejected at the guard, no filesystem access.
        with patch("hermes_constants.get_default_hermes_root") as mock_root:
            self.assertIsNone(_load_profile_identity("a/" * 5000))
            mock_root.assert_not_called()
        # Valid and huge but under the filename limit: guard passes, the
        # nonexistent-directory check returns None — no crash.
        long_valid = "a" * 240
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ) as mock_root:
                self.assertIsNone(_load_profile_identity(long_valid))
            mock_root.assert_called_once()  # guard let it through

    def test_path_separators_rejected_across_platforms(self):
        """POSIX and Windows separators (and dot/anchor edge cases) are all
        rejected by the single ASCII-only guard."""
        for bad in [
            "/", "\\", "//", "\\\\", "a/b", "a\\b", "..", "../..",
            "..\\..", "/etc/passwd", "C:\\Windows\\system32",
            "\\\\server\\share", "a//b", "a\\\\b", "./a", "a/.",
            "a/", "/a", ".profile", "profile.", "profile name",
        ]:
            with self.subTest(name=bad):
                with patch(
                    "hermes_constants.get_default_hermes_root"
                ) as mock_root:
                    self.assertIsNone(_load_profile_identity(bad))
                    mock_root.assert_not_called()

    def test_none_and_empty_rejected(self):
        """None and empty names are rejected at the guard (regex requires 1+)."""
        with patch("hermes_constants.get_default_hermes_root") as mock_root:
            self.assertIsNone(_load_profile_identity(None))
            self.assertIsNone(_load_profile_identity(""))
            mock_root.assert_not_called()


# ── Category 2: Regression — allow_model_selection=false gating ─────────


class TestProfileModelGatingRegression(unittest.TestCase):
    """Regression: allow_model_selection=False must gate model selection even
    when allow_profile_identity=True exposes profiles (whose config.yaml may
    name a model). The two flags must stay fully independent."""

    def test_flag_getters_when_model_selection_disabled(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={
                "allow_profile_identity": True,
                "allow_model_selection": False,
            },
        ):
            self.assertFalse(_get_allow_model_selection())
            self.assertTrue(_get_allow_profile_identity())

    def test_schema_has_profile_but_not_model(self):
        """The core gating regression: profile ON + model OFF must advertise
        the profile field but NOT the model field — at BOTH the top level and
        inside the tasks[] item properties."""
        with patch(
            "tools.delegate_tool._load_config",
            return_value={
                "allow_profile_identity": True,
                "allow_model_selection": False,
            },
        ):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertNotIn("model", props)
        self.assertIn("profile", props)
        self.assertEqual(props["profile"]["type"], "string")
        task_props = props["tasks"]["items"]["properties"]
        self.assertNotIn("model", task_props)
        self.assertIn("profile", task_props)

    def test_flag_isolation_profile_on_only(self):
        """Turning the profile flag on must not turn the model flag on."""
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_profile_identity": True},
        ):
            self.assertTrue(_get_allow_profile_identity())
            self.assertFalse(_get_allow_model_selection())
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertIn("profile", props)
        self.assertNotIn("model", props)

    def test_flag_isolation_model_on_only(self):
        """Turning the model flag on must not turn the profile flag on."""
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_model_selection": True},
        ):
            self.assertTrue(_get_allow_model_selection())
            self.assertFalse(_get_allow_profile_identity())
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertIn("model", props)
        self.assertNotIn("profile", props)

    def test_profile_model_ignored_when_model_selection_disabled(self):
        """Dispatch-level regression: with allow_model_selection=False, a
        profile's config.yaml model must NOT trigger switch_model. The child
        inherits the delegation default (parent) model instead, while still
        receiving the profile identity."""
        set_spawn_paused(False)
        parent = _make_mock_parent()
        with patch(
            "tools.delegate_tool._load_config",
            return_value={
                "allow_profile_identity": True,
                "allow_model_selection": False,
            },
        ), patch(
            "tools.delegate_tool._load_profile_identity",
            return_value=_identity_stub(model="glm-5.3"),
        ), patch(
            "hermes_cli.model_switch.switch_model"
        ) as mock_switch, patch("run_agent.AIAgent") as MockAgent:
            child = _make_mock_child()
            MockAgent.return_value = child
            out = delegate_task(
                tasks=[{"goal": GOAL, "profile": "devbot"}],
                parent_agent=parent,
            )
        payload = json.loads(out)
        self.assertNotIn("error", payload)
        # The profile's model was never resolved through the switch pipeline.
        mock_switch.assert_not_called()
        # The child inherits the parent's model, not the profile's model.
        kwargs = MockAgent.call_args.kwargs
        self.assertEqual(kwargs["model"], parent.model)
        # Identity still applied — gating the model does not gate the profile.
        self.assertIn("You are Devbot", kwargs["ephemeral_system_prompt"])
        self.assertEqual(child._delegate_profile, "devbot")


# ── Category 3: Fallback chain semantics ─────────────────────────────────


class TestFallbackChainSemantics(unittest.TestCase):
    """Fail loudly, never silently inherit."""

    def test_unresolvable_model_raises_value_error(self):
        """A name the shared model_switch pipeline cannot resolve raises
        ValueError — no silent fallback to the base model."""
        creds = _base_creds()
        with self.assertRaises(ValueError):
            _resolve_task_model_creds(
                "zzz-not-a-real-model-xyz", _FakeParent(), creds
            )

    def test_resolver_message_surfaced_in_value_error(self):
        """The resolver's error message must reach the caller (per-task error
        quality), not be swallowed into a generic message."""
        creds = _base_creds()
        with patch(
            "hermes_cli.model_switch.switch_model"
        ) as mock_switch:
            mock_switch.return_value = _make_switch_result(
                success=False, error_message="no such model in catalog"
            )
            with self.assertRaises(ValueError) as ctx:
                _resolve_task_model_creds("bogus", _FakeParent(), creds)
        self.assertIn("no such model in catalog", str(ctx.exception))

    def test_none_model_name_returns_base_unchanged(self):
        """None model name is treated as empty — a no-op returning the base
        creds by identity, with no resolution attempted."""
        creds = _base_creds()
        out = _resolve_task_model_creds(None, _FakeParent(), creds)
        self.assertIs(out, creds)

    def test_empty_model_name_returns_base_unchanged(self):
        """Empty model name is a no-op (covered elsewhere too; kept here for
        completeness as the fallback-chain contract)."""
        creds = _base_creds()
        out = _resolve_task_model_creds("", _FakeParent(), creds)
        self.assertIs(out, creds)

    def test_whitespace_model_name_returns_base_unchanged(self):
        """A whitespace-only name strips to empty and is a no-op."""
        creds = _base_creds()
        out = _resolve_task_model_creds("   ", _FakeParent(), creds)
        self.assertIs(out, creds)

    def test_base_creds_not_mutated_on_failure(self):
        """Even when resolution fails (ValueError), the base creds dict must
        be byte-identical to its pre-call state."""
        creds = _base_creds()
        before = dict(creds)
        with patch(
            "hermes_cli.model_switch.switch_model"
        ) as mock_switch:
            mock_switch.return_value = _make_switch_result(
                success=False, error_message="unknown model"
            )
            with self.assertRaises(ValueError):
                _resolve_task_model_creds("zzz-not-real", _FakeParent(), creds)
        self.assertEqual(creds, before)

    def test_dispatch_fails_loudly_on_unresolvable_model(self):
        """delegate_task must surface a per-task error for an unresolvable
        model instead of silently spawning the child on the default model."""
        set_spawn_paused(False)
        parent = _make_mock_parent()
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_model_selection": True},
        ), patch(
            "hermes_cli.model_switch.switch_model"
        ) as mock_switch, patch("run_agent.AIAgent") as MockAgent:
            mock_switch.return_value = _make_switch_result(
                success=False, error_message="unknown model"
            )
            out = delegate_task(
                tasks=[{"goal": GOAL, "model": "zzz-bogus-model-xyz"}],
                parent_agent=parent,
            )
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn("could not resolve model", payload["error"])
        self.assertIn("zzz-bogus-model-xyz", payload["error"])
        # No child was ever constructed — no silent fallback spawn.
        MockAgent.assert_not_called()


# ── Category 4: Integration with mock child spawns ───────────────────────


class TestProfileIdentityIntegration(unittest.TestCase):
    """delegate_task end-to-end with mocked AIAgent spawns (no real API
    calls): profile identity threads through to the child system prompt."""

    def setUp(self):
        set_spawn_paused(False)

    def test_profile_identity_appears_in_child_system_prompt(self):
        """When a profile is specified, the child's system prompt is built
        from the profile's SOUL/IDENTITY — not the generic subagent
        preamble."""
        parent = _make_mock_parent()
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_profile_identity": True},
        ), patch(
            "tools.delegate_tool._load_profile_identity",
            return_value=_identity_stub(),
        ), patch("run_agent.AIAgent") as MockAgent:
            child = _make_mock_child()
            MockAgent.return_value = child
            out = delegate_task(
                tasks=[{"goal": GOAL, "profile": "devbot"}],
                parent_agent=parent,
            )
        payload = json.loads(out)
        self.assertNotIn("error", payload)
        kwargs = MockAgent.call_args.kwargs
        prompt = kwargs["ephemeral_system_prompt"]
        self.assertIn("You are Devbot", prompt)
        self.assertIn("Name: Devbot", prompt)
        self.assertIn(f"YOUR TASK:\n{GOAL}", prompt)
        self.assertNotIn("You are a focused subagent", prompt)
        # Identity stashed on the child for usage-ledger attribution.
        self.assertEqual(child._delegate_profile, "devbot")

    def test_no_profile_uses_generic_preamble(self):
        """Without a profile, the child gets the generic subagent preamble."""
        parent = _make_mock_parent()
        with patch(
            "tools.delegate_tool._load_config", return_value={}
        ), patch("run_agent.AIAgent") as MockAgent:
            child = _make_mock_child()
            MockAgent.return_value = child
            out = delegate_task(
                tasks=[{"goal": GOAL}], parent_agent=parent
            )
        payload = json.loads(out)
        self.assertNotIn("error", payload)
        prompt = MockAgent.call_args.kwargs["ephemeral_system_prompt"]
        self.assertIn("You are a focused subagent", prompt)
        self.assertIn(f"YOUR TASK:\n{GOAL}", prompt)
        self.assertNotIn("You are Devbot", prompt)

    def test_explicit_model_takes_precedence_over_profile_model(self):
        """When both profile and model are specified, the explicit task model
        wins: switch_model runs exactly once, for the task's model — never
        for the profile's config.yaml model."""
        parent = _make_mock_parent()
        with patch(
            "tools.delegate_tool._load_config",
            return_value={
                "allow_profile_identity": True,
                "allow_model_selection": True,
            },
        ), patch(
            "tools.delegate_tool._load_profile_identity",
            return_value=_identity_stub(model="glm-5.3"),
        ), patch(
            "hermes_cli.model_switch.switch_model"
        ) as mock_switch, patch("run_agent.AIAgent") as MockAgent:
            mock_switch.return_value = _make_switch_result(
                new_model="gpt-5x", target_provider="openai"
            )
            child = _make_mock_child()
            MockAgent.return_value = child
            out = delegate_task(
                tasks=[
                    {"goal": GOAL, "profile": "devbot", "model": "gpt-5x"}
                ],
                parent_agent=parent,
            )
        payload = json.loads(out)
        self.assertNotIn("error", payload)
        # Resolver ran once, for the EXPLICIT model.
        self.assertEqual(mock_switch.call_count, 1)
        self.assertEqual(mock_switch.call_args.kwargs["raw_input"], "gpt-5x")
        # The child was constructed with the resolved model/provider.
        kwargs = MockAgent.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-5x")
        self.assertEqual(kwargs["provider"], "openai")
        # Profile identity still applied alongside the model switch.
        self.assertIn("You are Devbot", kwargs["ephemeral_system_prompt"])


if __name__ == "__main__":
    unittest.main()