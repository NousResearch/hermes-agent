"""Tests for MCP tool trust-tier gating via readOnlyHint annotations.

Security boundary under test: write-capable MCP tools (anything whose
``readOnlyHint`` annotation is not exactly ``True``) on servers configured
``trust: untrusted`` must route through the existing dangerous-approval
path before the RPC fires. Read-only tools and tools on trusted servers
pass straight through.

Adversarial notes encoded in these tests:
- ``readOnlyHint`` is a HINT supplied by the (potentially hostile) server.
  It can only ever RELAX gating on a server the operator already marked
  untrusted; the trust tier itself is operator-side config, so a lying
  server can at worst skip approval for a tool it claims is read-only —
  which is why the trust key is per-server and gating is fail-closed for
  missing/unknown metadata.
- Missing annotations ⇒ write-capable (fail closed).
- Unknown/garbage ``trust`` values ⇒ treated as untrusted (fail closed).
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool
from tools import mcp_tool_handlers as _mcp_handlers
from tools import mcp_tool_registration as _mcp_registration


class _FakeContentBlock:
    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type


class _FakeCallToolResult:
    def __init__(self, content, is_error=False, structuredContent=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent


def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _install_lock_and_run():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_install_lock_and_run())
    finally:
        loop.close()


@pytest.fixture
def fake_session():
    """Patch a fake connected server + MCP loop; yield its session mock."""
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=_FakeCallToolResult(content=[_FakeContentBlock("ok")])
    )
    server = SimpleNamespace(session=session, _rpc_lock=None)
    with patch.dict(mcp_tool._servers, {"srv": server}), \
         patch("tools.mcp_tool_loop._run_on_mcp_loop",
               side_effect=_fake_run_on_mcp_loop), \
         patch.dict(mcp_tool._server_error_counts, {}, clear=True):
        yield session


@pytest.fixture(autouse=True)
def _clean_trust_state():
    """Isolate the module-level trust metadata between tests."""
    with patch.dict(mcp_tool._server_trust_levels, {}, clear=True), \
         patch.dict(mcp_tool._tool_read_only_hints, {}, clear=True), \
         patch.dict(mcp_tool._tool_read_only_overrides, {}, clear=True), \
         patch.dict(mcp_tool._tool_auto_approvals, {}, clear=True), \
         patch.dict(mcp_tool._tool_approval_pins, {}, clear=True):
        yield


def _set_trust(server: str, trust: str):
    mcp_tool._server_trust_levels[server] = trust


def _set_read_only(server: str, tool: str, value: bool):
    mcp_tool._tool_read_only_hints.setdefault(server, {})[tool] = value


class TestTrustGateAtCallTime:
    """The handler preamble consults the approval path when required."""

    def test_write_capable_on_untrusted_server_requires_approval(
        self, fake_session
    ):
        """Approval consulted; 'accept' lets the RPC through."""
        _set_trust("srv", "untrusted")
        # No readOnlyHint recorded for delete_repo → write-capable.
        handler = _mcp_handlers._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval_prompt.request_elicitation_consent",
            return_value="accept",
        ) as consent:
            raw = handler({"repo": "x"})
        consent.assert_called_once()
        assert json.loads(raw) == {"result": "ok"}
        fake_session.call_tool.assert_awaited_once()

    def test_denied_approval_blocks_rpc(self, fake_session):
        """'decline' blocks the call — the RPC must never fire."""
        _set_trust("srv", "untrusted")
        handler = _mcp_handlers._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval_prompt.request_elicitation_consent",
            return_value="decline",
        ):
            raw = handler({"repo": "x"})
        fake_session.call_tool.assert_not_awaited()
        assert "error" in json.loads(raw)

    def test_read_only_tool_on_untrusted_server_skips_approval(
        self, fake_session
    ):
        """readOnlyHint=True tools pass without consulting approval."""
        _set_trust("srv", "untrusted")
        _set_read_only("srv", "list_repos", True)
        handler = _mcp_handlers._make_tool_handler("srv", "list_repos", 30.0)
        with patch(
            "tools.approval_prompt.request_elicitation_consent"
        ) as consent:
            raw = handler({})
        consent.assert_not_called()
        assert json.loads(raw) == {"result": "ok"}

    def test_trusted_server_skips_approval_for_write_tools(
        self, fake_session
    ):
        """trust: full (and the default) never consults approval."""
        _set_trust("srv", "full")
        handler = _mcp_handlers._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval_prompt.request_elicitation_consent"
        ) as consent:
            raw = handler({"repo": "x"})
        consent.assert_not_called()
        assert json.loads(raw) == {"result": "ok"}

    def test_unconfigured_server_defaults_to_full_trust(self, fake_session):
        """Backward compat: servers with no trust key behave as before."""
        handler = _mcp_handlers._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval_prompt.request_elicitation_consent"
        ) as consent:
            raw = handler({"repo": "x"})
        consent.assert_not_called()
        assert json.loads(raw) == {"result": "ok"}

    def test_exact_auto_approved_write_skips_approval(self, fake_session):
        """``approval.auto_approve_tools`` exempts one exact write tool without trusting its server."""
        _set_trust("srv", "untrusted")
        mcp_tool._tool_auto_approvals["srv"] = frozenset({"create_draft"})
        handler = _mcp_handlers._make_tool_handler("srv", "create_draft", 30.0)
        with patch("tools.approval_prompt.request_elicitation_consent") as consent:
            raw = handler({})
        consent.assert_not_called()
        assert json.loads(raw) == {"result": "ok"}

    @pytest.mark.parametrize("tool", ["create_campaign", "send_draft", "create_draft2"])
    def test_auto_approval_is_exact_and_sibling_writes_stay_gated(self, fake_session, tool):
        """No server-wide leak and no pattern matching: neighbouring write tools still need consent."""
        _set_trust("srv", "untrusted")
        mcp_tool._tool_auto_approvals["srv"] = frozenset({"create_draft", "create_*"})
        handler = _mcp_handlers._make_tool_handler("srv", tool, 30.0)
        with patch("tools.approval_prompt.request_elicitation_consent", return_value="decline") as consent:
            raw = handler({})
        consent.assert_called_once()
        fake_session.call_tool.assert_not_awaited()
        assert "error" in json.loads(raw)

    def test_read_only_false_hint_is_gated(self, fake_session):
        """An explicit readOnlyHint=False is write-capable."""
        _set_trust("srv", "untrusted")
        _set_read_only("srv", "write_file", False)
        handler = _mcp_handlers._make_tool_handler("srv", "write_file", 30.0)
        with patch(
            "tools.approval_prompt.request_elicitation_consent",
            return_value="decline",
        ) as consent:
            handler({"path": "/etc/passwd"})
        consent.assert_called_once()
        fake_session.call_tool.assert_not_awaited()

    def test_approval_exception_fails_closed(self, fake_session):
        """Any exception in the consent path blocks the call."""
        _set_trust("srv", "untrusted")
        handler = _mcp_handlers._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval_prompt.request_elicitation_consent",
            side_effect=RuntimeError("approval backend down"),
        ):
            raw = handler({"repo": "x"})
        fake_session.call_tool.assert_not_awaited()
        assert "error" in json.loads(raw)


class TestTrustGateApprovalRouting:
    """MCP consent uses the API run's registered approval lifecycle only."""

    def test_api_run_callback_can_resolve_mcp_trust_gate(self, monkeypatch):
        """An API run emits and resolves the exact per-call consent request."""
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools import approval, approval_prompt
        from tools.approval_context import (
            reset_current_session_key,
            set_current_session_key,
        )

        session_key = "api-run-mcp-trust"
        seen = []
        # If the API callback is not selected, fail fast instead of waiting for
        # the CLI input timeout in this non-interactive test process.
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *_args, **_kwargs: "deny")

        def notify(approval_data):
            seen.append(dict(approval_data))
            assert approval.resolve_gateway_approval(
                session_key, "once", request_id=approval_data["request_id"]
            ) == 1

        session_tokens = set_session_vars(
            platform="api_server", session_key=session_key, async_delivery=False
        )
        key_token = set_current_session_key(session_key)
        approval.register_gateway_notify(session_key, notify)
        try:
            assert approval_prompt.request_elicitation_consent(
                "MCP tool 'write' on UNTRUSTED server 'srv' wants to run.",
                "Approve once or deny.", surface="mcp-trust/srv",
            ) == "accept"
        finally:
            approval.unregister_gateway_notify(session_key)
            reset_current_session_key(key_token)
            clear_session_vars(session_tokens)

        assert len(seen) == 1
        assert seen[0]["pattern_key"] == "mcp_elicitation"
        assert seen[0]["request_id"]

    @pytest.mark.parametrize("platform,cron,single_query", [
        ("webhook", "", False),
        ("telegram", "1", False),
        ("api_server", "", True),
        ("local", "", True),  # a `chat -q` turn admitted to the gateway: finite, nobody answers
    ])
    def test_unattended_cron_and_single_query_contexts_do_not_use_callback(
        self, monkeypatch, platform, cron, single_query
    ):
        """A registered callback must not widen consent for webhook, cron, or -q workers (#111526)."""
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools import approval, approval_prompt
        from tools.approval_context import (
            reset_current_session_key,
            set_current_session_key,
        )

        session_key = f"{platform}-mcp-trust"
        notified = []
        session_tokens = set_session_vars(
            platform=platform, session_key=session_key, cron_session=cron
        )
        key_token = set_current_session_key(session_key)
        approval.register_gateway_notify(session_key, lambda data: notified.append(data))
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *_args, **_kwargs: "deny")
        gateway_waits = []
        monkeypatch.setattr(
            approval_prompt._gw,
            "_await_gateway_decision",
            lambda *_args, **_kwargs: gateway_waits.append(_args) or {"resolved": True, "choice": "once"},
        )
        if single_query:
            monkeypatch.setattr(approval_prompt._ctx, "_is_single_query_approval_context", lambda: True)
        try:
            assert approval_prompt.request_elicitation_consent("write", "Approve once or deny.") == "decline"
        finally:
            approval.unregister_gateway_notify(session_key)
            reset_current_session_key(key_token)
            clear_session_vars(session_tokens)

        assert notified == []
        assert gateway_waits == []


    @pytest.mark.parametrize("context", ["single_query", "cron", "webhook"])
    def test_cli_panel_callback_is_never_asked_without_a_user(self, monkeypatch, context):
        """`hermes chat -q`, cron and unattended platforms can keep the CLI panel callback registered, but
        nobody answers it: consent declines at once instead of waiting the approval timeout (the e2e
        hang after ef1faa4cf8)."""
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools import approval_prompt
        from tools.terminal_tool import set_approval_callback

        if context == "single_query":  # cli_single_query exports the marker into the process env
            monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
            tokens = set_session_vars()
        elif context == "cron":  # the cron scheduler binds it per job
            tokens = set_session_vars(cron_session="1")
        else:  # an unattended platform: no adapter can carry an approval prompt
            tokens = set_session_vars(platform="webhook")
        asked = []
        set_approval_callback(lambda *args, **kwargs: asked.append(args) or "once")
        try:
            assert approval_prompt.request_elicitation_consent("write", "Approve once or deny.") == "decline"
        finally:
            set_approval_callback(None)
            clear_session_vars(tokens)
        assert asked == []


class TestTrustNormalization:
    def test_unknown_trust_value_treated_as_untrusted(self):
        """Garbage trust strings fail closed to untrusted."""
        assert _mcp_registration._normalize_server_trust("banana") == "untrusted"

    def test_known_values(self):
        assert _mcp_registration._normalize_server_trust("full") == "full"
        assert _mcp_registration._normalize_server_trust("UNTRUSTED") == "untrusted"
        assert _mcp_registration._normalize_server_trust("  Full ") == "full"
        # Missing key → default full (backward compatible; documented).
        assert _mcp_registration._normalize_server_trust(None) == "full"


class TestAnnotationCaptureAtDiscovery:
    """_register_server_tools records trust + readOnlyHint metadata."""

    def _make_tool(self, name, annotations=None):
        return SimpleNamespace(
            name=name, description="", inputSchema=None,
            annotations=annotations,
        )

    def test_registration_records_hints_and_trust(self):
        from tools.registry import ToolRegistry

        server = mcp_tool.MCPServerTask("srv")
        server.session = MagicMock()
        server._tools = [
            self._make_tool(
                "list_repos", SimpleNamespace(readOnlyHint=True)
            ),
            self._make_tool(
                "delete_repo", SimpleNamespace(readOnlyHint=False)
            ),
            self._make_tool("no_annotations", None),
        ]
        config = {
            "trust": "untrusted",
            "tools": {"resources": False, "prompts": False},
        }
        with patch("tools.registry.registry", ToolRegistry()), \
             patch("tools.mcp_tool_registration._track_mcp_tool_server"):
            _mcp_registration._register_server_tools("srv", server, config)

        assert mcp_tool._server_trust_levels["srv"] == "untrusted"
        hints = mcp_tool._tool_read_only_hints["srv"]
        assert hints.get("list_repos") is True
        # Anything not exactly True is write-capable.
        assert not hints.get("delete_repo")
        assert not hints.get("no_annotations")

    def test_exact_approval_policy_marks_read_only_and_auto_approves_named_write(self):
        """Per-tool policy is captured without broadening the untrusted server."""
        tools = [self._make_tool("list_items"), self._make_tool("create_draft"), self._make_tool("send_item")]
        config = {"trust": "untrusted",
                  "approval": {"read_only_tools": ["list_items"], "auto_approve_tools": ["create_draft"]}}

        _mcp_registration._record_tool_trust_metadata("srv", config, tools)

        assert mcp_tool._server_trust_levels["srv"] == "untrusted"
        # Operator policy never enters the connection's readOnlyHint data (shared by adopters).
        assert mcp_tool._tool_read_only_hints["srv"] == {
            "list_items": False, "create_draft": False, "send_item": False}
        assert mcp_tool._tool_read_only_overrides["srv"] == {"list_items"}
        assert mcp_tool._tool_auto_approvals["srv"] == {"create_draft"}
        assert _mcp_handlers._tool_is_read_only("srv", "list_items") is True
        assert _mcp_handlers._trust_gate_check("srv", "list_items") is None
        assert _mcp_handlers._trust_gate_check("srv", "create_draft") is None
        with patch("tools.approval_prompt.request_elicitation_consent", return_value="decline"):
            assert _mcp_handlers._trust_gate_check("srv", "send_item") is not None

    def test_approval_policy_fails_closed_on_malformed_or_unknown_names(self):
        """A scalar string is not split, a glob is not a pattern, an unknown name is dropped."""
        tools = [self._make_tool("create_draft")]
        config = {"trust": "untrusted",
                  "approval": {"read_only_tools": "create_draft",
                               "auto_approve_tools": ["create_*", "missing_tool", 7, ""]}}

        _mcp_registration._record_tool_trust_metadata("srv", config, tools)

        assert mcp_tool._tool_read_only_hints["srv"] == {"create_draft": False}
        assert mcp_tool._tool_read_only_overrides["srv"] == set()
        assert mcp_tool._tool_auto_approvals["srv"] == set()

    @pytest.mark.parametrize("approval,expected", [
        (["create_draft"], "approval must be a mapping"),
        ({"auto_approve": ["create_draft"]}, "approval has unknown key(s) 'auto_approve'"),
        ({"auto_approve_tools": ["create_draft", 7, ""]}, "accepts only non-empty tool-name strings; ignoring 7, ''"),
        ({"auto_approve_tools": ["create_*"]}, "names unknown tool(s) 'create_*'"),
    ])
    def test_malformed_approval_is_ignored_with_a_warning(self, caplog, approval, expected):
        """A misspelled or mis-shaped block grants nothing, but the operator is told why."""
        tools = [self._make_tool("create_draft"), self._make_tool("create_campaign")]
        with caplog.at_level("WARNING", logger=_mcp_registration.logger.name):
            _mcp_registration._record_tool_trust_metadata("srv", {"trust": "untrusted", "approval": approval}, tools)
        assert expected in caplog.text
        assert mcp_tool._tool_auto_approvals["srv"] <= {"create_draft"}
        assert "create_campaign" not in mcp_tool._tool_auto_approvals["srv"]

    def test_list_changed_cannot_add_authority_but_a_reload_revokes(self):
        """Names are accepted once per loaded config object: a tool that appears after discovery under a
        name the operator was warned was ignored gains nothing, while a reloaded config re-derives."""
        config = {"trust": "untrusted",
                  "approval": {"read_only_tools": ["list_items", "future_read"],
                               "auto_approve_tools": ["create_draft", "future_write"]}}
        tools = [self._make_tool("list_items"), self._make_tool("create_draft"), self._make_tool("send_item")]
        _mcp_registration._record_tool_trust_metadata("srv", config, tools)
        assert mcp_tool._tool_auto_approvals["srv"] == {"create_draft"}

        # tools/list_changed re-registers with the same config object.
        late = tools + [self._make_tool("future_read"), self._make_tool("future_write")]
        _mcp_registration._record_tool_trust_metadata("srv", config, late)
        assert mcp_tool._tool_read_only_overrides["srv"] == {"list_items"}
        assert mcp_tool._tool_auto_approvals["srv"] == {"create_draft"}

        # A config reload is a fresh object: dropping the lists revokes the exemption.
        _mcp_registration._record_tool_trust_metadata("srv", {"trust": "untrusted", "approval": {}}, tools)
        assert mcp_tool._tool_auto_approvals["srv"] == set()
        with patch("tools.approval_prompt.request_elicitation_consent", return_value="decline"):
            assert _mcp_handlers._trust_gate_check("srv", "create_draft") is not None

    def test_dict_annotations_supported(self):
        """Cached/JSON annotations arrive as plain dicts."""
        assert _mcp_registration._annotation_read_only_hint(
            SimpleNamespace(annotations={"readOnlyHint": True})
        ) is True
        assert _mcp_registration._annotation_read_only_hint(
            SimpleNamespace(annotations={"readOnlyHint": "yes"})
        ) is False  # non-bool truthy → NOT read-only (hint must be True)
        assert _mcp_registration._annotation_read_only_hint(
            SimpleNamespace(annotations=None)
        ) is False
        assert _mcp_registration._annotation_read_only_hint(
            SimpleNamespace()
        ) is False

    def test_sdk2_snake_case_annotations_supported(self):
        """mcp 2.x models expose ``read_only_hint``; camelCase is only a serialization alias."""
        from mcp.types import ToolAnnotations
        sdk2 = ToolAnnotations.model_validate({"readOnlyHint": True, "destructiveHint": False})
        assert _mcp_registration._annotation_read_only_hint(SimpleNamespace(annotations=sdk2)) is True
        assert _mcp_registration._annotation_read_only_hint(
            SimpleNamespace(annotations=SimpleNamespace(read_only_hint=True))
        ) is True
        assert _mcp_registration._annotation_read_only_hint(
            SimpleNamespace(annotations=SimpleNamespace(read_only_hint=False))
        ) is False
        # 1.x-shaped object (camelCase attribute) still works.
        assert _mcp_registration._annotation_read_only_hint(
            SimpleNamespace(annotations=SimpleNamespace(readOnlyHint=True))
        ) is True
        # Cache dict written by an older client in snake_case is honoured too.
        assert _mcp_registration._annotation_read_only_hint(
            SimpleNamespace(annotations={"read_only_hint": True})
        ) is True
