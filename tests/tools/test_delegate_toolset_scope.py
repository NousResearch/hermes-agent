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

    @patch("tools.delegate_tool._load_agent_profiles", return_value=_FAKE_AGENT_PROFILES)
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

    @patch("tools.delegate_tool._load_agent_profiles", return_value=_FAKE_AGENT_PROFILES)
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

    @patch("tools.delegate_tool._load_agent_profiles", return_value=_FAKE_AGENT_PROFILES)
    @patch("tools.delegate_tool._load_config", return_value=_BARE_DELEGATION_CFG)
    @patch("tools.delegate_tool._build_child_agent")
    def test_unknown_profile_falls_back_gracefully(
        self, mock_build, _mock_cfg, _mock_profiles
    ):
        """Unknown profile name logs a warning and falls back to explicit toolsets.

        Why: A typo in a profile name should not hard-fail the delegation;
        it should fall back to whatever toolsets the caller provided, if any.
        Test: Pass profile='nonexistent', assert profile_name=None in call.
        """
        fake_child = MagicMock()
        fake_child._delegate_saved_tool_names = []
        mock_build.return_value = fake_child

        parent = _make_no_mcp_parent()
        parent.enabled_toolsets = ["terminal"]

        with patch(
            "tools.delegate_tool._run_single_child",
            return_value=_make_fake_child_result(0),
        ):
            delegate_task(
                goal="Do something",
                toolsets=["terminal"],
                profile="nonexistent",
                parent_agent=parent,
            )

        _, kwargs = mock_build.call_args
        # Unknown profile → resolved_profile_name stays None → no bypass.
        assert kwargs.get("profile_name") is None, (
            f"Unknown profile should leave profile_name=None, got {kwargs.get('profile_name')!r}"
        )


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
