"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_agent,
    _strip_blocked_tools,
    delegate_task,
)


class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_requested_toolsets_intersected_with_parent(self):
        """LLM requests toolsets parent doesn't have — extras are dropped."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file"])

        # Simulate the intersection logic from _build_child_agent
        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "file", "web", "browser", "rl"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["file", "terminal"]
        assert "web" not in scoped
        assert "browser" not in scoped
        assert "rl" not in scoped

    def test_all_requested_toolsets_available_on_parent(self):
        """LLM requests subset of parent tools — all pass through."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file", "web", "browser"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "web"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["terminal", "web"]

    def test_no_toolsets_requested_inherits_parent(self):
        """When toolsets is None/empty, child inherits parent's set."""
        parent_toolsets = ["terminal", "file", "web"]
        child = _strip_blocked_tools(parent_toolsets)
        assert "terminal" in child
        assert "file" in child
        assert "web" in child

    def test_strip_blocked_removes_delegation(self):
        """Blocked toolsets (delegation, clarify, etc.) are always removed."""
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "terminal" in child

    def test_empty_intersection_yields_empty_toolsets(self):
        """If parent has no overlap with requested, child gets nothing extra."""
        parent = SimpleNamespace(enabled_toolsets=["terminal"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["web", "browser"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert scoped == []


def _make_mcp_restricted_parent():
    """Parent agent whose MCP context is empty (e.g. no_mcp orchestrator).

    enabled_toolsets=[] means the parent loaded no toolsets at all — the
    intersection against it would normally drop every requested toolset.
    """
    parent = MagicMock()
    parent.enabled_toolsets = []
    parent._delegate_depth = 0
    parent._credential_pool = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._print_fn = None
    return parent


class TestProfileMcpToolsetBypass:
    """MCP toolsets declared by a named agent_profile bypass parent intersection.

    Regression coverage for NousResearch/hermes-agent#32668: an orchestrator
    that restricts its own MCP servers must still be able to hand domain MCP
    toolsets to a child via a named profile. Non-MCP toolsets keep going
    through the parent intersection (the security boundary).
    """

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_profile_mcp_toolsets_bypass_parent_intersection(self, _):
        """profile_name set → MCP toolsets pass through even when parent has none."""
        parent = _make_mcp_restricted_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="Check mail",
                context=None,
                toolsets=["mcp-fastmail", "mcp-knowledge"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name="mail",
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
        assert "mcp-fastmail" in child_toolsets
        assert "mcp-knowledge" in child_toolsets

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_no_profile_mcp_toolsets_still_intersected(self, _):
        """No profile → MCP toolset request is dropped (intersection enforced)."""
        parent = _make_mcp_restricted_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="Check mail",
                context=None,
                toolsets=["mcp-fastmail"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name=None,
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
        assert "mcp-fastmail" not in child_toolsets

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_profile_non_mcp_toolsets_still_intersected(self, _):
        """Even with a profile, non-MCP toolsets the parent lacks are dropped."""
        parent = _make_mcp_restricted_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="Browse the web",
                context=None,
                toolsets=["mcp-fastmail", "browser"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name="mail",
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
        # MCP toolset bypasses intersection ...
        assert "mcp-fastmail" in child_toolsets
        # ... but the non-MCP toolset the parent never had is still dropped.
        assert "browser" not in child_toolsets


# ---------------------------------------------------------------------------
# TestDelegateTaskProfileWiring
# Regression guard for NousResearch/hermes-agent#32668 / #32727.
# The full call chain from delegate_task() → _build_child_agent() must
# thread the profile name and resolved toolsets through so MCP toolsets
# bypass the no_mcp parent intersection at the _build_child_agent level.
# ---------------------------------------------------------------------------

_FAKE_AGENT_PROFILES = {
    "documents": {
        "toolsets": ["mcp-nextcloud-files", "mcp-knowledge"],
    },
    "mail": {
        "toolsets": ["mcp-fastmail"],
    },
}

# Minimal delegate_task config: no delegation-specific overrides.
_BARE_DELEGATION_CFG = {}

# A structurally valid result dict from _run_single_child (enough fields for
# the post-run aggregation logic in delegate_task to not raise AttributeError).
def _make_fake_child_result(task_index: int = 0) -> dict:
    """Return a minimal _run_single_child result dict.

    Why: delegate_task's post-run loop calls entry.pop('_child_role', None),
    entry.get('api_calls', 0), etc. — mocking with a plain string raises
    AttributeError.  This helper produces a dict with every field that the
    post-run aggregation touches.
    Test: Used internally by TestDelegateTaskProfileWiring patches.
    """
    return {
        "task_index": task_index,
        "status": "ok",
        "summary": "done",
        "error": None,
        "exit_reason": "complete",
        "api_calls": 0,
        "duration_seconds": 0.1,
        "_child_role": "leaf",
        "diagnostic_path": None,
    }


def _make_no_mcp_parent():
    """Parent agent that simulates a no_mcp orchestrator.

    enabled_toolsets contains only the non-MCP platform toolsets.
    No mcp-* names are present, so the intersection path in
    _build_child_agent would strip every MCP toolset when profile_name=None.
    """
    parent = MagicMock()
    parent.enabled_toolsets = ["delegation", "knowledge"]
    parent._delegate_depth = 0
    parent._credential_pool = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._print_fn = None
    parent.api_key = None
    parent._client_kwargs = {}
    parent.model = "test-model"
    parent._subagent_id = None
    parent.valid_tool_names = []
    return parent


class TestDelegateTaskProfileWiring:
    """delegate_task() wires profile → _build_child_agent(profile_name=...).

    Key regression guard: MUST fail against pre-fix code (profile_name=None
    hardcoded) and MUST pass after the fix (profile_name=resolved name).
    """

    @patch("tools.delegate_tool._load_profiles", return_value=_FAKE_AGENT_PROFILES)
    @patch("tools.delegate_tool._load_config", return_value=_BARE_DELEGATION_CFG)
    @patch("tools.delegate_tool._build_child_agent")
    def test_profile_forwarded_to_build_child_agent(
        self, mock_build, _mock_cfg, _mock_profiles
    ):
        """delegate_task with profile='documents' calls _build_child_agent with
        profile_name='documents' and the profile's toolsets.

        This test FAILS on pre-fix code (profile_name=None hardcoded) and
        PASSES after the fix (profile_name='documents' forwarded).

        Why: Without this wiring the bypass branch at lines 972-976 of
        _build_child_agent never activates, so mcp-nextcloud-files is stripped
        from a no_mcp parent's child agent.
        Test: Assert _build_child_agent is called with profile_name='documents'
        and toolsets=['mcp-nextcloud-files', 'mcp-knowledge'].
        """
        fake_child = MagicMock()
        fake_child._delegate_saved_tool_names = []
        mock_build.return_value = fake_child

        parent = _make_no_mcp_parent()

        # delegate_task needs _run_single_child; patch it out with a proper dict.
        with patch(
            "tools.delegate_tool._run_single_child",
            return_value=_make_fake_child_result(0),
        ):
            delegate_task(
                goal="Fetch the document",
                context="Use the Nextcloud MCP server",
                profile="documents",
                parent_agent=parent,
            )

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("profile_name") == "documents", (
            f"profile_name should be 'documents', got {kwargs.get('profile_name')!r}. "
            "Pre-fix code hardcodes profile_name=None — this is the regression guard."
        )
        assert "mcp-nextcloud-files" in (kwargs.get("toolsets") or []), (
            f"Expected mcp-nextcloud-files in toolsets, got {kwargs.get('toolsets')!r}. "
            "Profile toolsets must override the toolsets arg so the bypass activates."
        )

    @patch("tools.delegate_tool._load_profiles", return_value=_FAKE_AGENT_PROFILES)
    @patch("tools.delegate_tool._load_config", return_value=_BARE_DELEGATION_CFG)
    @patch("tools.delegate_tool._build_child_agent")
    def test_no_profile_leaves_profile_name_none(
        self, mock_build, _mock_cfg, _mock_profiles
    ):
        """Without profile= arg, _build_child_agent is called with profile_name=None.

        Preserves backward-compatible behavior: ad-hoc delegation without a
        named profile still goes through the security intersection path.

        Why: Must not accidentally enable the bypass for unprofiled calls.
        Test: Call delegate_task without profile= and assert profile_name=None.
        """
        fake_child = MagicMock()
        fake_child._delegate_saved_tool_names = []
        mock_build.return_value = fake_child

        parent = _make_no_mcp_parent()
        parent.enabled_toolsets = ["terminal", "file"]

        with patch(
            "tools.delegate_tool._run_single_child",
            return_value=_make_fake_child_result(0),
        ):
            delegate_task(
                goal="List files",
                toolsets=["terminal", "file"],
                parent_agent=parent,
            )

        _, kwargs = mock_build.call_args
        assert kwargs.get("profile_name") is None, (
            f"Expected profile_name=None for unprofiled call, got {kwargs.get('profile_name')!r}"
        )

    @patch("tools.delegate_tool._load_profiles", return_value=_FAKE_AGENT_PROFILES)
    @patch("tools.delegate_tool._load_config", return_value=_BARE_DELEGATION_CFG)
    @patch("tools.delegate_tool._build_child_agent")
    def test_unknown_profile_returns_error(
        self, mock_build, _mock_cfg, _mock_profiles
    ):
        """Unknown profile name HARD-FAILS: error JSON, delegation refused.

        Why: An unknown profile must fail closed. Falling back to the caller's
        raw toolsets would let a typo'd profile run with an unrestricted/unintended
        toolset and silently bypass the profile's declared scope — the canonical,
        security-correct behavior is to refuse the delegation entirely.
        Test: Pass profile='nonexistent', assert the result is an
        {"error": "Unknown agent profile ..."} JSON and _build_child_agent is
        never called (no child constructed).
        """
        import json

        fake_child = MagicMock()
        fake_child._delegate_saved_tool_names = []
        mock_build.return_value = fake_child

        parent = _make_no_mcp_parent()
        parent.enabled_toolsets = ["terminal"]

        with patch(
            "tools.delegate_tool._run_single_child",
            return_value=_make_fake_child_result(0),
        ):
            result = json.loads(
                delegate_task(
                    goal="Do something",
                    toolsets=["terminal"],
                    profile="nonexistent",
                    parent_agent=parent,
                )
            )

        assert "error" in result, (
            f"Unknown profile must return an error JSON, got {result!r}"
        )
        assert "Unknown agent profile" in result["error"], (
            f"Expected 'Unknown agent profile' in error, got {result['error']!r}"
        )
        mock_build.assert_not_called()


class TestDelegateTaskSchemaProfile:
    """DELEGATE_TASK_SCHEMA must formally declare the 'profile' property.

    The model reliably emits only schema-declared fields. Without a formal
    'profile' entry in the schema the model would silently omit it even when
    the SOUL/orchestrator prompt instructs it to pass profile=.
    """

    def test_profile_field_in_schema(self):
        """'profile' is a declared property in DELEGATE_TASK_SCHEMA.

        Why: Without a formal schema entry the LLM strips the field before
        calling the tool, making the orchestrator's profile= instruction
        invisible to the handler.
        Test: Assert 'profile' key exists in schema properties.
        """
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "profile" in props, (
            "'profile' missing from DELEGATE_TASK_SCHEMA parameters.properties. "
            "Add it so the model reliably emits the field."
        )

    def test_profile_field_is_string_type(self):
        """'profile' schema property must be type=string.

        Why: Any other type would cause schema validation failures at the
        provider API boundary.
        Test: Assert type == 'string' for the profile property.
        """
        prop = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["profile"]
        assert prop.get("type") == "string", (
            f"Expected profile schema type='string', got {prop.get('type')!r}"
        )

    def test_profile_field_not_required(self):
        """'profile' must not be in the 'required' array (it is optional).

        Why: Making it required would break all existing delegate_task calls
        that don't specify a profile.
        Test: Assert 'profile' absent from the required list.
        """
        required = DELEGATE_TASK_SCHEMA["parameters"].get("required", [])
        assert "profile" not in required, (
            "'profile' should not be in required — it is an optional field."
        )


class TestProfileMcpBypassEndToEnd:
    """End-to-end bypass: no_mcp parent + profile → child keeps MCP toolsets.

    This is the key regression guard from the PR description. It directly
    tests _build_child_agent with profile_name set vs. None to confirm the
    bypass branch activates/deactivates correctly.
    """

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_no_mcp_parent_with_profile_retains_mcp_toolsets(self, _):
        """CRITICAL: no_mcp parent + profile_name → child retains mcp-nextcloud-files.

        This test FAILS against pre-fix code (profile_name=None hardcoded in
        the _build_child_agent call site) because the bypass branch is never
        entered — mcp-nextcloud-files gets stripped by the parent intersection.

        After the fix (profile_name='documents' forwarded) the bypass branch
        activates and mcp-nextcloud-files survives.

        Why: The confirmed prod gap: fat sub-agents under no_mcp orchestrators
        received 0 MCP tools. This verifies the fix at the _build_child_agent
        level independent of the delegate_task wiring.
        Test: Pass profile_name='documents' and assert mcp-nextcloud-files in
        child enabled_toolsets. Then repeat with profile_name=None and assert
        it IS stripped.
        """
        parent = _make_mcp_restricted_parent()

        # --- With profile_name set (post-fix behavior) ---
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="File management",
                context=None,
                toolsets=["mcp-nextcloud-files", "knowledge"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name="documents",
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
        assert "mcp-nextcloud-files" in child_toolsets, (
            "mcp-nextcloud-files should bypass no_mcp parent intersection when "
            "profile_name is set. If this fails the fix is not working."
        )

    @patch("tools.delegate_tool._load_config", return_value={})
    def test_no_mcp_parent_without_profile_strips_mcp_toolsets(self, _):
        """Confirm the pre-fix behavior: no profile → MCP stripped (security).

        This must remain true to preserve the security boundary. Ad-hoc
        delegation without a profile cannot bypass the intersection.

        Why: The bypass is a deliberate whitelist; only named profiles whose
        config explicitly declares MCP servers should get them through.
        Test: profile_name=None → mcp-nextcloud-files absent from child.
        """
        parent = _make_mcp_restricted_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="File management",
                context=None,
                toolsets=["mcp-nextcloud-files"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name=None,
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]
        assert "mcp-nextcloud-files" not in child_toolsets, (
            "mcp-nextcloud-files must be stripped when profile_name=None (security boundary)."
        )


# ---------------------------------------------------------------------------
# TestBatchToolsetInjectionBlocked
# Security regression guard for the batch-mode privilege-escalation hole:
# a model that names a valid profile (activating the mcp-* bypass in
# _build_child_agent) AND supplies per-task toolsets must NOT gain toolsets
# beyond what the profile declares.
# ---------------------------------------------------------------------------

_PROFILES_WITH_NEXTCLOUD = {
    "documents": {
        "toolsets": ["mcp-nextcloud-files"],
    },
}

_PROFILES_EMPTY_TOOLSETS = {
    "bare-profile": {
        # no 'toolsets' key → empty profile
    },
}


class TestBatchToolsetInjectionBlocked:
    """Batch-mode model-supplied toolsets cannot override a resolved profile.

    The vulnerability: batch tasks=[{"goal":"x","toolsets":["mcp-evil"]}] with
    profile="documents" previously let the model inject arbitrary mcp-* toolsets
    because the batch loop used `t.get("toolsets") or toolsets`, overriding the
    profile-resolved toolsets while resolved_profile_name stayed set (activating
    the mcp-* bypass in _build_child_agent).

    These tests FAIL on pre-fix code (mcp-injected-evil present in child) and
    PASS after the fix (child toolsets match profile declaration only).
    """

    @patch("tools.delegate_tool._load_profiles", return_value=_PROFILES_WITH_NEXTCLOUD)
    @patch("tools.delegate_tool._load_config", return_value=_BARE_DELEGATION_CFG)
    @patch("tools.delegate_tool._build_child_agent")
    def test_batch_injection_blocked_model_cannot_inject_evil_mcp_toolset(
        self, mock_build, _mock_cfg, _mock_profiles
    ):
        """CRITICAL security: batch per-task toolsets are ignored when a profile is set.

        Why: Pre-fix, the batch loop used t.get("toolsets") or toolsets, so a
        model could name profile="documents" (activating the mcp-* bypass) and
        supply tasks=[{"goal":"x","toolsets":["mcp-injected-evil"]}] to gain
        mcp-injected-evil — which the 'documents' profile never declared.
        What: Assert _build_child_agent receives only the profile's toolsets
        (mcp-nextcloud-files), and NOT the model-injected mcp-injected-evil.
        Test: Compare the toolsets kwarg of the mock call against the profile
        declaration; mcp-injected-evil must be absent, mcp-nextcloud-files present.
        """
        fake_child = MagicMock()
        fake_child._delegate_saved_tool_names = []
        mock_build.return_value = fake_child

        parent = _make_no_mcp_parent()

        with patch(
            "tools.delegate_tool._run_single_child",
            return_value=_make_fake_child_result(0),
        ):
            delegate_task(
                # No top-level goal — use batch tasks path
                tasks=[{"goal": "do something", "toolsets": ["mcp-injected-evil"]}],
                profile="documents",
                parent_agent=parent,
            )

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        child_toolsets = kwargs.get("toolsets") or []

        assert "mcp-injected-evil" not in child_toolsets, (
            f"SECURITY: mcp-injected-evil must be blocked but found in {child_toolsets!r}. "
            "Pre-fix code lets batch per-task toolsets override the profile. "
            "This test must FAIL before the fix and PASS after."
        )
        assert "mcp-nextcloud-files" in child_toolsets, (
            f"mcp-nextcloud-files (from profile 'documents') must be present, "
            f"got {child_toolsets!r}"
        )

    @patch("tools.delegate_tool._load_profiles", return_value=_PROFILES_WITH_NEXTCLOUD)
    @patch("tools.delegate_tool._load_config", return_value=_BARE_DELEGATION_CFG)
    @patch("tools.delegate_tool._build_child_agent")
    def test_single_task_profile_toolsets_unchanged(
        self, mock_build, _mock_cfg, _mock_profiles
    ):
        """Single-task profile path: child receives profile's declared toolsets.

        Why: Regression guard — the batch fix must not break single-task profile
        delegation, which already worked correctly pre-fix.
        What: delegate_task(goal=..., profile="documents") → child gets
        mcp-nextcloud-files (the profile's only declared toolset).
        Test: Assert mcp-nextcloud-files in toolsets kwarg and profile_name set.
        """
        fake_child = MagicMock()
        fake_child._delegate_saved_tool_names = []
        mock_build.return_value = fake_child

        parent = _make_no_mcp_parent()

        with patch(
            "tools.delegate_tool._run_single_child",
            return_value=_make_fake_child_result(0),
        ):
            delegate_task(
                goal="Manage my documents",
                profile="documents",
                parent_agent=parent,
            )

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        child_toolsets = kwargs.get("toolsets") or []

        assert "mcp-nextcloud-files" in child_toolsets, (
            f"Single-task profile path must forward profile toolsets, got {child_toolsets!r}"
        )
        assert kwargs.get("profile_name") == "documents", (
            f"profile_name must be 'documents', got {kwargs.get('profile_name')!r}"
        )

    @patch("tools.delegate_tool._load_profiles", return_value=_PROFILES_EMPTY_TOOLSETS)
    @patch("tools.delegate_tool._load_config", return_value=_BARE_DELEGATION_CFG)
    @patch("tools.delegate_tool._build_child_agent")
    def test_empty_profile_toolsets_bypass_not_activated(
        self, mock_build, _mock_cfg, _mock_profiles
    ):
        """Profile with no toolsets does NOT activate the mcp-* bypass.

        Why: Pre-fix, a profile with empty/missing toolsets still set
        resolved_profile_name, activating the bypass with whatever caller-
        supplied toolsets were present.  Any mcp-* name the model supplied
        would bypass the parent intersection — a privilege-escalation vector.
        What: Assert profile_name=None in the _build_child_agent call so the
        intersection path (not bypass) is used, stripping mcp-x from a no_mcp
        parent.
        Test: Profile 'bare-profile' has no toolsets; caller passes
        toolsets=['mcp-x']; assert profile_name=None in build call.
        """
        fake_child = MagicMock()
        fake_child._delegate_saved_tool_names = []
        mock_build.return_value = fake_child

        parent = _make_no_mcp_parent()

        with patch(
            "tools.delegate_tool._run_single_child",
            return_value=_make_fake_child_result(0),
        ):
            delegate_task(
                goal="Do something restricted",
                toolsets=["mcp-x"],
                profile="bare-profile",
                parent_agent=parent,
            )

        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args

        assert kwargs.get("profile_name") is None, (
            f"Empty-toolsets profile must NOT set profile_name (bypass must be inactive), "
            f"got profile_name={kwargs.get('profile_name')!r}. "
            "Pre-fix: resolved_profile_name was set regardless of toolsets presence."
        )


# ---------------------------------------------------------------------------
# TestProfileToolsetsAliasing
# Regression guard for the config-aliasing vulnerability: toolsets must be
# copied (list()) from the config dict, not assigned by reference.
#
# Without the copy, any in-place mutation of the working `toolsets` variable
# (or of `profile_resolved_toolsets`, which is derived from it) silently
# corrupts the agent_profiles config dict for the entire process lifetime.
# On the next delegate_task call the now-mutated list would be used as the
# authoritative profile declaration, bypassing the intended toolset scope.
# ---------------------------------------------------------------------------

class TestProfileToolsetsAliasing:
    """Profile toolsets assignment must copy, not reference, the config list.

    Why: profile_cfg["toolsets"] lives inside the dict returned by
    _load_agent_profiles() and may be cached for the process lifetime.
    Assigning it directly to the working variable means any in-place
    mutation (e.g., .append(), .clear(), .pop()) of that variable would
    corrupt the config — all subsequent delegate_task calls in the same
    process would see the mutated toolsets as the 'official' profile
    declaration.  This is a privilege-escalation / DoS vector.

    The fix (list(profile_toolsets) at line ~2082 and
    list(toolsets) at line ~2110) ensures the working variable and the
    authoritative capture are independent copies.
    """

    @patch("tools.delegate_tool._load_profiles")
    @patch("tools.delegate_tool._load_config", return_value=_BARE_DELEGATION_CFG)
    @patch("tools.delegate_tool._build_child_agent")
    def test_profile_toolsets_copy_prevents_config_corruption(
        self, mock_build, _mock_cfg, mock_load_profiles
    ):
        """Mutating the toolsets list passed to _build_child_agent must NOT
        corrupt the original agent_profiles config entry.

        Why: Pre-fix code did ``toolsets = profile_toolsets`` (reference), so
        mutating the kwarg received by _build_child_agent also mutated
        profiles["documents"]["toolsets"] — corrupting it for every future call.
        What: After delegate_task resolves a profile, capture the toolsets list
        that was passed to _build_child_agent, mutate it in-place, then verify
        profiles["documents"]["toolsets"] is unchanged.
        Test: Assert that appending a sentinel to the captured child toolsets
        list does NOT appear in the original config dict, proving list() copy.
        """
        # Hold a direct reference to the config list so we can inspect it after
        # delegate_task has run and after we mutate the captured child toolsets.
        original_toolsets_list = ["mcp-nextcloud-files", "mcp-knowledge"]
        profiles_cfg = {
            "documents": {
                "toolsets": original_toolsets_list,
            },
        }
        mock_load_profiles.return_value = profiles_cfg

        fake_child = MagicMock()
        fake_child._delegate_saved_tool_names = []
        mock_build.return_value = fake_child

        parent = _make_no_mcp_parent()

        with patch(
            "tools.delegate_tool._run_single_child",
            return_value=_make_fake_child_result(0),
        ):
            delegate_task(
                goal="Fetch documents",
                profile="documents",
                parent_agent=parent,
            )

        # Retrieve the toolsets list that was handed to _build_child_agent.
        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        child_toolsets_arg: list = kwargs.get("toolsets") or []

        # Sanity: the profile toolsets made it through correctly.
        assert "mcp-nextcloud-files" in child_toolsets_arg, (
            f"Profile toolsets must be forwarded; got {child_toolsets_arg!r}"
        )

        # Now mutate the list that _build_child_agent received.
        sentinel = "mcp-INJECTED-SENTINEL"
        child_toolsets_arg.append(sentinel)
        child_toolsets_arg.clear()

        # CRITICAL ASSERTION: the original config list must be untouched.
        # Pre-fix (reference): original_toolsets_list would now be empty.
        # Post-fix (copy):     original_toolsets_list is still the original.
        assert original_toolsets_list == ["mcp-nextcloud-files", "mcp-knowledge"], (
            f"Config corruption detected: profiles['documents']['toolsets'] was mutated "
            f"to {original_toolsets_list!r}. "
            "The toolsets working variable must be a list() copy, not a reference to the "
            "config dict's list. Pre-fix code assigns `toolsets = profile_toolsets` "
            "(reference); the fix is `toolsets = list(profile_toolsets)`."
        )


# ---------------------------------------------------------------------------
# TestInheritMcpToolsetsProfileGuard
# Security hardening: _preserve_parent_mcp_toolsets must NOT run when a named
# profile is in use.  The profile's declared toolsets are authoritative — the
# parent's MCP toolsets must not bleed in via the inherit_mcp_toolsets path.
#
# Scenario: parent has mcp-nextcloud-files AND mcp-fastmail in its toolsets.
# Profile "documents" declares only ["mcp-nextcloud-files"].
# With inherit_mcp_toolsets=True (default), the pre-fix code calls
# _preserve_parent_mcp_toolsets unconditionally, which appends mcp-fastmail
# (a parent MCP toolset missing from the child list) to the resolved profile
# toolsets — violating the invariant that the profile is authoritative.
#
# These tests FAIL on pre-fix code (mcp-fastmail bleeds in) and PASS after
# the fix (guard `and not profile_name` prevents _preserve_parent_mcp_toolsets
# from running when a profile is set).
# ---------------------------------------------------------------------------

_PROFILES_DOCUMENTS_NEXTCLOUD_ONLY = {
    "documents": {
        "toolsets": ["mcp-nextcloud-files"],
    },
}


def _make_mcp_rich_parent():
    """Parent agent that carries two MCP toolsets: nextcloud-files AND fastmail.

    Why: Simulates an orchestrator that has loaded both domain MCP servers.
    The profile "documents" only declares mcp-nextcloud-files, so mcp-fastmail
    is the parent-only MCP toolset that must NOT bleed into the child when a
    profile is named.

    What: enabled_toolsets includes both mcp-nextcloud-files and mcp-fastmail
    so that _preserve_parent_mcp_toolsets would normally append mcp-fastmail.

    Test: Use as the parent arg in _build_child_agent calls below.
    """
    parent = MagicMock()
    parent.enabled_toolsets = ["mcp-nextcloud-files", "mcp-fastmail", "delegation"]
    parent._delegate_depth = 0
    parent._credential_pool = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._print_fn = None
    parent.api_key = None
    parent._client_kwargs = {}
    parent.model = "test-model"
    parent._subagent_id = None
    parent.valid_tool_names = []
    return parent


class TestInheritMcpToolsetsProfileGuard:
    """inherit_mcp_toolsets=True must NOT add parent MCP toolsets when a profile is named.

    The profile's declared toolsets are authoritative.  _preserve_parent_mcp_toolsets
    must be skipped entirely when profile_name is set so that parent-only MCP toolsets
    cannot bleed into the child.

    FAIL before fix: mcp-fastmail (parent-only) appears in child toolsets.
    PASS after fix:  child toolsets are EXACTLY the profile's (mcp-nextcloud-files only).
    """

    @patch("tools.delegate_tool._get_inherit_mcp_toolsets", return_value=True)
    @patch("tools.delegate_tool._load_config", return_value={})
    def test_parent_mcp_bleed_blocked_under_profile(self, _mock_cfg, _mock_inherit):
        """CRITICAL: profile toolsets are authoritative — parent MCP toolsets must not bleed.

        Why: Pre-fix, _preserve_parent_mcp_toolsets runs unconditionally when
        inherit_mcp_toolsets=True, appending every parent MCP toolset absent from
        the child list.  With a profile that declares only mcp-nextcloud-files,
        the parent's mcp-fastmail gets added — a silent scope expansion that
        violates the "profile declares exactly which MCP servers the child needs"
        invariant.

        What: _build_child_agent with profile_name="documents" (declares only
        mcp-nextcloud-files) and a parent that has BOTH mcp-nextcloud-files and
        mcp-fastmail.  Child toolsets must be exactly ["mcp-nextcloud-files"]
        (after stripping delegation); mcp-fastmail must NOT appear.

        Test: Assert "mcp-nextcloud-files" in child toolsets and
        "mcp-fastmail" NOT in child toolsets.  FAILS pre-fix (mcp-fastmail
        bleeds in via _preserve_parent_mcp_toolsets), PASSES post-fix (guard
        `and not profile_name` skips the bleed path).
        """
        parent = _make_mcp_rich_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="Manage documents",
                context=None,
                toolsets=["mcp-nextcloud-files"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name="documents",
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]

        assert "mcp-nextcloud-files" in child_toolsets, (
            f"Profile-declared mcp-nextcloud-files must be present, got {child_toolsets!r}"
        )
        assert "mcp-fastmail" not in child_toolsets, (
            f"SECURITY: mcp-fastmail (parent-only) must NOT bleed into the child when "
            f"profile_name is set, but got {child_toolsets!r}. "
            "Pre-fix: _preserve_parent_mcp_toolsets runs unconditionally, appending "
            "every parent MCP toolset not in the child list. "
            "Fix: add `and not profile_name` guard to the inherit_mcp_toolsets check "
            "at line ~980 of delegate_tool.py."
        )

    @patch("tools.delegate_tool._get_inherit_mcp_toolsets", return_value=True)
    @patch("tools.delegate_tool._load_config", return_value={})
    def test_non_profile_inheritance_unchanged(self, _mock_cfg, _mock_inherit):
        """Non-profile delegation with inherit_mcp_toolsets=True still inherits parent MCP.

        Why: Regression baseline — the profile guard must not break the existing
        behavior for ad-hoc (no profile) delegation.  When no profile is named,
        _preserve_parent_mcp_toolsets should still run and add parent MCP toolsets
        to a narrowed child that requests only a subset.

        What: parent has mcp-nextcloud-files + mcp-fastmail; child requests only
        mcp-nextcloud-files with NO profile.  After the fix, mcp-fastmail must
        still appear in child toolsets (non-profile path is unchanged).

        Test: Assert BOTH mcp-nextcloud-files and mcp-fastmail are present in the
        child toolsets, proving that the guard is scoped to the profile case only.
        """
        parent = _make_mcp_rich_parent()

        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()

            _build_child_agent(
                task_index=0,
                goal="Ad-hoc file management",
                context=None,
                toolsets=["mcp-nextcloud-files"],
                model=None,
                max_iterations=10,
                task_count=1,
                parent_agent=parent,
                profile_name=None,  # no profile — inheritance must still work
            )

        child_toolsets = MockAgent.call_args[1]["enabled_toolsets"]

        assert "mcp-nextcloud-files" in child_toolsets, (
            f"Requested mcp-nextcloud-files must be present, got {child_toolsets!r}"
        )
        assert "mcp-fastmail" in child_toolsets, (
            f"mcp-fastmail must be inherited from parent when no profile is named "
            f"(non-profile inheritance must be unchanged), got {child_toolsets!r}. "
            "If this fails the profile guard is too broad."
        )
