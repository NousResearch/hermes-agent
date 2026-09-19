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
         patch.dict(mcp_tool._tool_read_only_hints, {}, clear=True):
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
        assert "did not approve" in json.loads(raw)["error"]

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
            # "cancel": nobody was asked (no approval channel), which is NOT the same as a user refusal.
            assert approval_prompt.request_elicitation_consent("write", "Approve once or deny.") == "cancel"
        finally:
            approval.unregister_gateway_notify(session_key)
            reset_current_session_key(key_token)
            clear_session_vars(session_tokens)

        assert notified == []
        assert gateway_waits == []


class TestApprovalChannelHonesty:
    """A surface that cannot ask a human fails closed *and says so*: no refusal is attributed to a user who
    was never prompted, and no interactive prompt is rendered where nobody can read it (#111526)."""

    def _untrusted_write_handler(self, tool: str = "send_test_notice"):
        _set_trust("srv", "untrusted")
        return _mcp_handlers._make_tool_handler("srv", tool, 30.0)

    def test_single_query_worker_denies_without_prompting_and_explains_why(self, fake_session, monkeypatch):
        """A dispatcher-spawned ``-q`` worker (Kanban) is never prompted: fail closed, with why + what to do."""
        from tools import approval_context, approval_prompt

        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")  # cli.py sets this for -q runs
        assert approval_context._is_single_query_approval_context() is True
        prompted = []
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval",
                            lambda *args, **kwargs: prompted.append(args) or "deny")

        raw = json.loads(self._untrusted_write_handler()({"recipient": "x", "text": "y"}))

        assert prompted == []  # no DANGEROUS COMMAND banner into a worker log nobody watches
        assert "error" in raw
        fake_session.call_tool.assert_not_awaited()  # still fail-closed
        err = raw["error"]
        assert "no approval channel" in err
        assert "trust: full" in err  # actionable: how to allow it on purpose
        assert "did not approve" not in err  # never a refusal the user did not give

    def test_gateway_session_without_a_callback_is_unresolved_not_a_refusal(self, monkeypatch):
        """A gateway session whose notify callback is gone fails closed as unresolved (§ cancelled ≠ denied)."""
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools import approval, approval_prompt
        from tools.approval_context import reset_current_session_key, set_current_session_key

        session_key = "tui-mcp-trust-no-callback"
        assert approval._gateway_notify_cb(session_key) is None
        session_tokens = set_session_vars(platform="tui", session_key=session_key, async_delivery=False)
        key_token = set_current_session_key(session_key)
        try:
            assert approval_prompt.request_elicitation_consent(
                "MCP tool 'write' on UNTRUSTED server 'srv' wants to run.", "Approve once or deny."
            ) == "cancel"
        finally:
            reset_current_session_key(key_token)
            clear_session_vars(session_tokens)

    def test_api_run_callback_runs_the_gated_tool_end_to_end(self, fake_session, monkeypatch):
        """A live /v1/runs run answers the trust gate through its approval callback; the tool then runs."""
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools import approval, approval_prompt
        from tools.approval_context import reset_current_session_key, set_current_session_key

        session_key = "api-run-trust-gate-e2e"
        seen = []

        def notify(approval_data):
            seen.append(dict(approval_data))
            assert approval.resolve_gateway_approval(
                session_key, "once", request_id=approval_data["request_id"]
            ) == 1

        # The API run must route through its own callback, never through a CLI prompt (no TTY there).
        monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval",
                            lambda *args, **kwargs: pytest.fail("API run fell back to the CLI prompt"))
        session_tokens = set_session_vars(
            platform="api_server", session_key=session_key, async_delivery=False
        )
        key_token = set_current_session_key(session_key)
        approval.register_gateway_notify(session_key, notify)
        try:
            raw = json.loads(self._untrusted_write_handler()({"recipient": "x", "text": "y"}))
        finally:
            approval.unregister_gateway_notify(session_key)
            reset_current_session_key(key_token)
            clear_session_vars(session_tokens)

        assert raw == {"result": "ok"}
        fake_session.call_tool.assert_awaited_once()
        assert [event["pattern_key"] for event in seen] == ["mcp_elicitation"]


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
