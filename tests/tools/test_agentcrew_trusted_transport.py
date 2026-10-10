from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run_busy import GatewayBusySessionMixin
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, get_trusted_transport, set_session_vars
from gateway.trusted_transport import agentcrew_transport
from gateway.turn_context import TurnContext
from tools import mcp_tool
from tools.mcp_tool_handlers import _call_tool_racing_stdio_death, _make_tool_handler


def _transport(message_id: str = "77", update_id: str = "9901") -> dict[str, str]:
    return {
        "version": "hermes-agentcrew-trusted-transport/0.1",
        "platform": "telegram",
        "profile": "agentcrewm3",
        "session_id": "session-001",
        "chat_id": "123456789",
        "thread_id": "",
        "message_id": message_id,
        "update_id": update_id,
        "received_at": "2026-10-10T01:02:03.004Z",
    }


def _invoke(server_name: str, tool_name: str, arguments: dict) -> MagicMock:
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=SimpleNamespace(
        content=[], structuredContent=None, isError=False, meta=None,
    ))
    server = SimpleNamespace(session=session, _rpc_lock=None)

    def run_on_loop(coro_or_factory, timeout=30):
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory

        async def execute():
            server._rpc_lock = asyncio.Lock()
            return await coro

        return asyncio.run(execute())

    with patch.dict(mcp_tool._servers, {server_name: server}), patch(
        "tools.mcp_tool_loop._run_on_mcp_loop", side_effect=run_on_loop,
    ):
        _make_tool_handler(server_name, tool_name, 30.0)(arguments)
    return session


def test_telegram_event_builds_authoritative_transport_envelope() -> None:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123456789",
        message_id="77",
        profile="agentcrewm3",
    )
    event = MessageEvent(
        text="fixture",
        source=source,
        message_id="77",
        platform_update_id=9901,
    )
    event._transport_received_at = "2026-10-10T01:02:03.004Z"
    assert agentcrew_transport(event, source, "session-001") == _transport()


def test_trusted_transport_is_context_local_and_cleared() -> None:
    envelope = _transport()
    tokens = set_session_vars(trusted_transport=envelope)
    envelope["message_id"] = "forged-after-bind"
    assert get_trusted_transport()["message_id"] == "77"
    clear_session_vars(tokens)
    assert get_trusted_transport() is None


def test_agentcrew_mcp_call_gets_meta_outside_model_arguments() -> None:
    tokens = set_session_vars(trusted_transport=_transport())
    try:
        session = _invoke("agentcrew_m3", "agentcrew_submit_intent", {"message": "fixture"})
        session.call_tool.assert_awaited_once_with(
            "agentcrew_submit_intent",
            arguments={"message": "fixture"},
            meta={"com.agentcrew/trustedTransport": _transport()},
        )
    finally:
        clear_session_vars(tokens)


def test_non_agentcrew_mcp_call_never_receives_transport_identity() -> None:
    tokens = set_session_vars(trusted_transport=_transport())
    try:
        session = _invoke("other_server", "read", {})
        session.call_tool.assert_awaited_once_with("read", arguments={})
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
async def test_busy_agentcrew_turn_queues_new_inbound_for_fresh_transport(mode: str) -> None:
    turn = SimpleNamespace(
        agent=None,
        event=None,
        ctx=TurnContext(session_key="session-key"),
    )

    class BusyHarness(GatewayBusySessionMixin):
        _BusySteerOutcome = staticmethod(lambda **values: SimpleNamespace(**values))

        @staticmethod
        def _peek_session_state(session_key: str):
            return SimpleNamespace(turn=turn) if session_key == "session-key" else None

    runner = BusyHarness()
    agent = MagicMock()
    agent._supports_active_turn_redirect = True
    agent.redirect.return_value = True
    agent.steer.return_value = True
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123456789",
        message_id="77",
        profile="agentcrewm3",
    )
    opening = MessageEvent(text="message A", source=source, message_id="77", platform_update_id=9901)
    incoming = MessageEvent(text="message B", source=source, message_id="88", platform_update_id=9902)
    incoming._transport_received_at = "2026-10-10T01:02:04.004Z"
    turn.agent = agent
    turn.event = opening

    outcome = await runner._resolve_busy_steer_or_redirect(
        incoming, "session-key", mode, agent,
    )
    assert outcome.effective_mode == "queue"
    assert outcome.steered is False and outcome.redirected is False
    agent.steer.assert_not_called()
    agent.redirect.assert_not_called()

    transport_b = agentcrew_transport(incoming, source, "session-001")
    assert transport_b is not None
    tokens = set_session_vars(trusted_transport=transport_b)
    try:
        session = AsyncMock()
        session.call_tool.return_value = SimpleNamespace(
            content=[], structuredContent=None, isError=False, meta=None,
        )
        server = SimpleNamespace(session=session)
        await _call_tool_racing_stdio_death(
            server, "agentcrew_m3", "agentcrew_submit_intent", {"message": "fixture B"},
        )
        session.call_tool.assert_awaited_once_with(
            "agentcrew_submit_intent",
            arguments={"message": "fixture B"},
            meta={"com.agentcrew/trustedTransport": transport_b},
        )
        assert transport_b["message_id"] == "88"
        assert transport_b["update_id"] == "9902"
    finally:
        clear_session_vars(tokens)
