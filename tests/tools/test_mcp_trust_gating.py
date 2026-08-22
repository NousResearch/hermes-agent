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
import concurrent.futures
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cli as cli_module
from cli import HermesCLI
from run_agent import AIAgent
from tools import approval as approval_module
from tools import mcp_tool
from tools.registry import registry
from gateway.session_context import clear_session_vars, set_session_vars
from tools.terminal_tool import set_approval_callback
from tools.thread_context import propagate_context_to_thread


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
         patch("tools.mcp_tool._run_on_mcp_loop",
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


@contextmanager
def _session_platform(platform: str, session_key: str = "mcp-cli-test"):
    tokens = set_session_vars(platform=platform, session_key=session_key)
    try:
        yield
    finally:
        clear_session_vars(tokens)


def _make_cli_approval_stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._sudo_state = None
    cli._sudo_deadline = 0
    cli._modal_input_snapshot = None
    cli._invalidate = MagicMock()
    cli._paint_now = MagicMock()
    cli._persist_prompt_summary = MagicMock()
    return cli


class TestCliApprovalSurface:
    """Untrusted MCP confirmation reaches the registered CLI callback."""

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [("once", "accept"), ("session", "accept"),
         ("deny", "decline"), ("timeout", "cancel")],
    )
    def test_registered_callback_choice_is_normalized_without_persistence(
        self, choice, expected
    ):
        calls = []
        session_before = {
            key: values.copy()
            for key, values in approval_module._session_approved.items()
        }
        permanent_before = approval_module._permanent_approved.copy()
        fake_secret = "ghp_" + "FAKEGITHUBTOKEN123456789012345678"

        def callback(command, description, **kwargs):
            calls.append((command, description, kwargs))
            return choice

        set_approval_callback(callback)
        try:
            with _session_platform("cli"), patch(
                "prompt_toolkit.application.current.get_app_or_none",
                return_value=object(),
            ):
                result = approval_module.request_elicitation_consent(
                    f"run write with token {fake_secret}",
                    f"approve external mutation using {fake_secret}",
                    timeout_seconds=1,
                )
        finally:
            set_approval_callback(None)

        assert result == expected
        assert len(calls) == 1
        command, description, kwargs = calls[0]
        assert fake_secret not in command
        assert fake_secret not in description
        assert kwargs == {"allow_permanent": False}
        assert approval_module._session_approved == session_before
        assert approval_module._permanent_approved == permanent_before

    def test_callback_absence_under_prompt_toolkit_fails_closed(self):
        set_approval_callback(None)
        with _session_platform("cli"), patch(
            "prompt_toolkit.application.current.get_app_or_none",
            return_value=object(),
        ), patch(
            "builtins.input",
            side_effect=AssertionError("must not read stdin under prompt_toolkit"),
        ):
            result = approval_module.request_elicitation_consent(
                "write external state", "untrusted MCP server", timeout_seconds=1
            )

        assert result == "decline"

    def test_gateway_routing_remains_on_pending_approval_queue(self):
        session_key = "mcp-gateway-routing-test"
        captured = []

        def notify(data):
            captured.append(dict(data))
            approval_module.resolve_gateway_approval(session_key, "once")

        token = approval_module.set_current_session_key(session_key)
        approval_module.register_gateway_notify(session_key, notify)
        try:
            with _session_platform("telegram", session_key):
                result = approval_module.request_elicitation_consent(
                    "gateway MCP write", "confirm one operation",
                    surface="mcp-trust/srv",
                )
        finally:
            approval_module.unregister_gateway_notify(session_key)
            approval_module.reset_current_session_key(token)

        assert result == "accept"
        assert len(captured) == 1
        assert captured[0]["command"] == "gateway MCP write"
        assert captured[0]["pattern_key"] == "mcp_elicitation"

    def test_real_cli_panel_once_invokes_untrusted_transport_once(
        self, fake_session
    ):
        _set_trust("srv", "untrusted")
        handler = mcp_tool._make_tool_handler("srv", "save_knowledge", 30.0)
        cli = _make_cli_approval_stub()
        result = {}

        set_approval_callback(cli._approval_callback)
        try:
            with _session_platform("cli"):
                thread = threading.Thread(
                    target=propagate_context_to_thread(
                        lambda: result.setdefault("raw", handler({"note": "x"}))
                    ),
                    daemon=True,
                )
                thread.start()

                deadline = time.time() + 2
                while cli._approval_state is None and time.time() < deadline:
                    time.sleep(0.01)

                assert cli._approval_state is not None
                assert "save_knowledge" in cli._approval_state["command"]
                assert cli._approval_state["choices"][:3] == [
                    "once", "session", "deny"
                ]
                assert "always" not in cli._approval_state["choices"]
                cli._approval_state["response_queue"].put("once")
                thread.join(timeout=3)
        finally:
            set_approval_callback(None)

        assert not thread.is_alive()
        assert json.loads(result["raw"]) == {"result": "ok"}
        assert fake_session.call_tool.await_count == 1

    @pytest.mark.parametrize(("choice", "rpc_count"), [("once", 1), ("deny", 0)])
    def test_foreground_cli_turn_routes_untrusted_mcp_to_real_modal(
        self, fake_session, tmp_path, choice, rpc_count
    ):
        """The production CLI turn owns approval without test-side callback wiring.

        This traverses ``HermesCLI.chat``'s real foreground worker closure, a
        deterministic two-response ``AIAgent.run_conversation`` turn, executor
        selection, registry dispatch, and the MCP trust gate.  The test never
        calls ``set_approval_callback`` or ``propagate_context_to_thread``:
        callback installation is exclusively the production CLI's job.
        """
        server_name = "foreground-cli"
        registered_name = mcp_tool.mcp_prefixed_tool_name(server_name, "save_knowledge")
        schema = {
            "name": registered_name,
            "description": "Save one knowledge note",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
        tool_def = {"type": "function", "function": schema}
        mcp_tool._servers[server_name] = mcp_tool._servers["srv"]
        registry.register(
            name=registered_name,
            toolset=f"mcp-{server_name}",
            schema=schema,
            handler=mcp_tool._make_tool_handler(
                server_name, "save_knowledge", 5.0
            ),
            is_async=False,
            description=schema["description"],
        )
        _set_trust(server_name, "untrusted")

        tool_call = SimpleNamespace(
            id="call-save",
            type="function",
            function=SimpleNamespace(
                name=registered_name,
                arguments=json.dumps({"text": "foreground approval proof"}),
            ),
        )
        tool_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )],
            model="test/model",
            usage=None,
        )
        final_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None),
                finish_reason="stop",
            )],
            model="test/model",
            usage=None,
        )

        try:
            with patch("run_agent.get_tool_definitions", return_value=[tool_def]), \
                 patch("run_agent.check_toolset_requirements", return_value={}), \
                 patch("run_agent.OpenAI"), \
                 patch("run_agent._hermes_home", tmp_path), \
                 patch("agent.model_metadata.fetch_model_metadata", return_value={}):
                agent = AIAgent(
                    api_key="test-key",
                    base_url="https://example.test/v1",
                    provider="test",
                    model="test/model",
                    quiet_mode=True,
                    skip_context_files=True,
                    skip_memory=True,
                    platform="cli",
                )
            agent.client = MagicMock()
            agent.client.chat.completions.create.side_effect = [
                tool_response,
                final_response,
            ]
            agent._cached_system_prompt = "You are helpful."
            agent._use_prompt_caching = False
            agent.compression_enabled = False
            agent.save_trajectories = False

            with patch.object(cli_module, "get_tool_definitions", return_value=[]):
                cli = HermesCLI(compact=True)
            cli.agent = agent
            agent.session_id = cli.session_id
            setattr(agent, "tools", [tool_def])
            setattr(agent, "valid_tool_names", {registered_name})
            # Keep the deterministic single-tool snapshot byte-stable; this
            # regression targets foreground dispatch rather than MCP discovery.
            setattr(agent, "_skip_mcp_refresh", True)

            modal = {}

            def answer_modal():
                deadline = time.time() + 5
                while cli._approval_state is None and time.time() < deadline:
                    time.sleep(0.01)
                state = cli._approval_state
                if state is not None:
                    modal["choices"] = list(state["choices"])
                    state["response_queue"].put(choice)

            responder = threading.Thread(target=answer_modal, daemon=True)
            responder.start()
            with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
                 patch.object(cli, "_resolve_turn_agent_config", return_value={
                     "signature": cli._active_agent_route_signature,
                     "model": agent.model,
                     "runtime": None,
                     "request_overrides": None,
                 }), \
                 patch.object(cli, "_init_agent", return_value=True), \
                 patch.object(cli_module, "_cprint"), \
                 patch.object(cli_module, "set_approval_callback"):
                # Disable the legacy duplicate setter inside ``run_agent`` so
                # this regression proves the production thread handoff carries
                # the owning CLI instance's callback explicitly.
                result = cli.chat("save one knowledge note")
            responder.join(timeout=5)
        finally:
            registry.deregister(registered_name)
            mcp_tool._servers.pop(server_name, None)

        assert not responder.is_alive()
        assert modal["choices"][:3] == ["once", "session", "deny"]
        assert "always" not in modal["choices"]
        assert fake_session.call_tool.await_count == rpc_count
        assert result == "done"

    @pytest.mark.parametrize(
        ("choice", "rpc_count"), [("once", 1), ("deny", 0)]
    )
    def test_untrusted_handler_uses_propagated_cli_callback(
        self, fake_session, choice, rpc_count
    ):
        _set_trust("srv", "untrusted")
        handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
        calls = []

        def callback(command, description, **kwargs):
            calls.append((command, description, kwargs))
            return choice

        set_approval_callback(callback)
        try:
            with _session_platform("cli"), patch(
                "prompt_toolkit.application.current.get_app_or_none",
                return_value=object(),
            ), concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                raw = executor.submit(
                    propagate_context_to_thread(handler), {"repo": "x"}
                ).result(timeout=5)
        finally:
            set_approval_callback(None)

        assert len(calls) == 1
        assert calls[0][2] == {"allow_permanent": False}
        assert fake_session.call_tool.await_count == rpc_count
        if choice == "once":
            assert json.loads(raw) == {"result": "ok"}
        else:
            assert "did not approve" in json.loads(raw)["error"]


class TestTrustGateAtCallTime:
    """The handler preamble consults the approval path when required."""

    def test_write_capable_on_untrusted_server_requires_approval(
        self, fake_session
    ):
        """Approval consulted; 'accept' lets the RPC through."""
        _set_trust("srv", "untrusted")
        # No readOnlyHint recorded for delete_repo → write-capable.
        handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent",
            return_value="accept",
        ) as consent:
            raw = handler({"repo": "x"})
        consent.assert_called_once()
        assert json.loads(raw) == {"result": "ok"}
        fake_session.call_tool.assert_awaited_once()

    def test_denied_approval_blocks_rpc(self, fake_session):
        """'decline' blocks the call — the RPC must never fire."""
        _set_trust("srv", "untrusted")
        handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent",
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
        handler = mcp_tool._make_tool_handler("srv", "list_repos", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent"
        ) as consent:
            raw = handler({})
        consent.assert_not_called()
        assert json.loads(raw) == {"result": "ok"}

    def test_trusted_server_skips_approval_for_write_tools(
        self, fake_session
    ):
        """trust: full (and the default) never consults approval."""
        _set_trust("srv", "full")
        handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent"
        ) as consent:
            raw = handler({"repo": "x"})
        consent.assert_not_called()
        assert json.loads(raw) == {"result": "ok"}

    def test_unconfigured_server_defaults_to_full_trust(self, fake_session):
        """Backward compat: servers with no trust key behave as before."""
        handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent"
        ) as consent:
            raw = handler({"repo": "x"})
        consent.assert_not_called()
        assert json.loads(raw) == {"result": "ok"}

    def test_read_only_false_hint_is_gated(self, fake_session):
        """An explicit readOnlyHint=False is write-capable."""
        _set_trust("srv", "untrusted")
        _set_read_only("srv", "write_file", False)
        handler = mcp_tool._make_tool_handler("srv", "write_file", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent",
            return_value="decline",
        ) as consent:
            handler({"path": "/etc/passwd"})
        consent.assert_called_once()
        fake_session.call_tool.assert_not_awaited()

    def test_approval_exception_fails_closed(self, fake_session):
        """Any exception in the consent path blocks the call."""
        _set_trust("srv", "untrusted")
        handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
        with patch(
            "tools.approval.request_elicitation_consent",
            side_effect=RuntimeError("approval backend down"),
        ):
            raw = handler({"repo": "x"})
        fake_session.call_tool.assert_not_awaited()
        assert "error" in json.loads(raw)


class TestTrustNormalization:
    def test_unknown_trust_value_treated_as_untrusted(self):
        """Garbage trust strings fail closed to untrusted."""
        assert mcp_tool._normalize_server_trust("banana") == "untrusted"

    def test_known_values(self):
        assert mcp_tool._normalize_server_trust("full") == "full"
        assert mcp_tool._normalize_server_trust("UNTRUSTED") == "untrusted"
        assert mcp_tool._normalize_server_trust("  Full ") == "full"
        # Missing key → default full (backward compatible; documented).
        assert mcp_tool._normalize_server_trust(None) == "full"


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
             patch("tools.mcp_tool._track_mcp_tool_server"):
            mcp_tool._register_server_tools("srv", server, config)

        assert mcp_tool._server_trust_levels["srv"] == "untrusted"
        hints = mcp_tool._tool_read_only_hints["srv"]
        assert hints.get("list_repos") is True
        # Anything not exactly True is write-capable.
        assert not hints.get("delete_repo")
        assert not hints.get("no_annotations")

    def test_dict_annotations_supported(self):
        """Cached/JSON annotations arrive as plain dicts."""
        assert mcp_tool._annotation_read_only_hint(
            SimpleNamespace(annotations={"readOnlyHint": True})
        ) is True
        assert mcp_tool._annotation_read_only_hint(
            SimpleNamespace(annotations={"readOnlyHint": "yes"})
        ) is False  # non-bool truthy → NOT read-only (hint must be True)
        assert mcp_tool._annotation_read_only_hint(
            SimpleNamespace(annotations=None)
        ) is False
        assert mcp_tool._annotation_read_only_hint(
            SimpleNamespace()
        ) is False
