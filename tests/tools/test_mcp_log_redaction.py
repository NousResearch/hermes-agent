"""MCP diagnostics must use the resolving generation, before lossy rendering."""

import asyncio
import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools import mcp_tool, mcp_tool_config, mcp_tool_registration
from tools.mcp_tool_common import _sanitize_error


SECRETS = [
    "OpaqueAlpha1937ValueTail8274",
    r'OpaqueAlpha1937\nQuote"Slash\ValueTail8274',
    "OpaqueAlpha1937/token=inner;ValueTail8274",
    "7294158603174926",
    "  OpaqueAlpha1937ValueTail8274  ",
]
SECRET_B = "new-generation-value-5826"


@pytest.fixture(params=["rotate", "delete"])
def generation(tmp_path, monkeypatch, request):
    def resolve(secret, **extra):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("MCP_LOG_TEST_SECRET", raising=False)
        env_file = tmp_path / "server.env"
        # The line-based env parser preserves these quoted literal backslashes.
        env_file.write_text(f"MCP_LOG_TEST_SECRET='{secret}'\n", encoding="utf-8")
        config = mcp_tool_config._resolve_mcp_server_config({
            "command": sys.executable,
            "env_file": str(env_file),
            "sampling": {"enabled": False},
            "elicitation": {"enabled": False},
            **extra,
        })
        assert secret in mcp_tool_config._mcp_redaction_values(config)
        # The secret is deliberately overlay-only, not recoverable from config fields.
        assert secret not in json.dumps(config)
        if request.param == "rotate":
            env_file.write_text(f"MCP_LOG_TEST_SECRET={SECRET_B}\n", encoding="utf-8")
            newer = mcp_tool_config._resolve_mcp_server_config({"env_file": str(env_file)})
            assert SECRET_B in mcp_tool_config._mcp_redaction_values(newer)
        else:
            env_file.unlink()
        return config
    return resolve


def assert_clean_logs(caplog, secret, *, redacted=True):
    records = [r for r in caplog.records if r.name == "tools.mcp_tool"]
    assert records, "the diagnostic path must actually log"
    # Logging handlers can serialize raw args and exc_info, not just getMessage().
    for output in [caplog.text, *(repr((r.msg, r.args, r.exc_info)) for r in records)]:
        for fragment in (secret, secret[:8], secret[-8:]):
            assert fragment not in output
    if redacted:
        assert "[REDACTED]" in caplog.text
    assert all(r.exc_info is None for r in records)


async def notification(server, secret, monkeypatch, shape):
    data = {
        "text": secret,
        "cutoff": "." * 1990 + secret + "." * 100,
        "json": {secret: ["diagnostic", {"reflected": int(secret) if secret.isdecimal() else secret}]},
    }
    params = SimpleNamespace(level="warning", data=data[shape], logger=secret)
    await server._make_logging_callback()(params)


async def callback_failure(server, secret, monkeypatch):
    class BrokenParams:
        @property
        def data(self):
            raise ValueError(f"broken notification: {secret}")
    await server._make_logging_callback()(BrokenParams())


async def refresh_failure(server, secret, monkeypatch):
    # Real scheduler and refresh implementation; only remote tools/list fails.
    server.session = SimpleNamespace(list_tools=AsyncMock(side_effect=ValueError(f"tools/list: {secret}")))
    await server._schedule_tools_refresh()
    await asyncio.sleep(0)
    assert not server._pending_refresh_tasks


async def message_exception(server, secret, monkeypatch):
    await server._make_message_handler()(ValueError(f"message: {secret}"))


async def handler_failure(server, secret, monkeypatch):
    from mcp.types import ServerNotification, ToolListChangedNotification

    message = ToolListChangedNotification(method="notifications/tools/list_changed")
    if hasattr(ServerNotification, "model_validate"):
        message = ServerNotification(root=message)

    def fail(_self):
        raise ValueError(f"handler: {secret}")
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_schedule_tools_refresh", fail)
    await server._make_message_handler()(message)


async def tools_changed(server, secret, monkeypatch):
    # Keep registry implementation outside this health-log invariant.
    server.session = SimpleNamespace(list_tools=AsyncMock(return_value=SimpleNamespace(tools=[])))
    from tools.mcp_tool_schema import mcp_prefixed_tool_name
    normalized = mcp_prefixed_tool_name(server.name, secret)
    monkeypatch.setattr(mcp_tool_registration, "_register_server_tools", lambda *_a: [normalized])
    await server._refresh_tools()
    assert server._registered_tool_names == [normalized]


async def protocol_fallback(server, secret, monkeypatch, mode):
    server._config["protocol"] = mode
    error = ValueError(f"Unsupported protocol version: {secret}")
    success = SimpleNamespace(capabilities=object())
    session = SimpleNamespace(initialize=AsyncMock(), discover=AsyncMock())
    primary, fallback = (session.discover, session.initialize) if mode == "stateless" else (session.initialize, session.discover)
    primary.side_effect = error
    fallback.return_value = success
    assert await server._negotiate_session(session, 5) is success
    primary.assert_awaited_once()
    fallback.assert_awaited_once()


async def protocol_config(server, secret, monkeypatch):
    server._config["protocol"] = secret
    session = SimpleNamespace(initialize=AsyncMock(return_value="connected"))
    assert await server._negotiate_session(session, 5) == "connected"


async def transport_group(server, secret, monkeypatch):
    server._ready.set()
    group = ExceptionGroup(f"outer: {secret}", [
        ExceptionGroup("nested", [ValueError(f"leaf: {secret}")]),
        ConnectionError("stream dropped"),
    ])
    assert server._reconnect_or_reraise_group(group) == "reconnect"
    server._shutdown_event.set()
    with pytest.raises(ExceptionGroup) as caught:
        server._reconnect_or_reraise_group(group)
    assert caught.value is group


async def oauth_setup(server, secret, monkeypatch):
    error = ValueError(f"OAuth setup: {secret}")
    def fail():
        raise error
    monkeypatch.setattr("tools.mcp_oauth_manager.get_manager", fail)
    server._auth_type = "oauth"
    with pytest.raises(ValueError) as caught:
        server._build_oauth_auth("https://example.invalid/mcp", server._config)
    assert caught.value is error


SINKS = {
    "notification": lambda s, a, m: notification(s, a, m, "text"),
    "notification-cutoff": lambda s, a, m: notification(s, a, m, "cutoff"),
    "notification-json": lambda s, a, m: notification(s, a, m, "json"),
    "callback-failure": callback_failure,
    "refresh-failure": refresh_failure,
    "message-exception": message_exception,
    "handler-failure": handler_failure,
    "tools-changed": tools_changed,
    "protocol-auto": lambda s, a, m: protocol_fallback(s, a, m, "auto"),
    "protocol-stateless": lambda s, a, m: protocol_fallback(s, a, m, "stateless"),
    "protocol-config": protocol_config,
    "transport-group": transport_group,
    "oauth-setup": oauth_setup,
}


@pytest.mark.parametrize("secret", SECRETS, ids=["opaque", "escaped", "assignment", "numeric", "whitespace"])
@pytest.mark.parametrize("sink", SINKS)
def test_mcp_logs_keep_generation_before_rendering(generation, monkeypatch, caplog, secret, sink):
    config = generation(secret)
    caplog.set_level(logging.DEBUG, logger="tools.mcp_tool")

    async def exercise():
        server = mcp_tool.MCPServerTask("redaction")
        assert await server._prepare_run(config)
        await SINKS[sink](server, secret, monkeypatch)
        assert _sanitize_error(SECRET_B, server._redaction_values) == SECRET_B

    asyncio.run(exercise())
    assert_clean_logs(caplog, secret, redacted=sink != "tools-changed")
    if sink == "tools-changed":
        assert "tools changed dynamically — added: 1" in caplog.text


@pytest.mark.parametrize("secret", SECRETS, ids=["opaque", "escaped", "assignment", "numeric", "whitespace"])
@pytest.mark.parametrize("render", [str, repr, json.dumps], ids=["plain", "repr", "json"])
def test_exact_values_are_redacted_before_credential_patterns(generation, secret, render):
    config = generation(secret)
    values = mcp_tool_config._mcp_redaction_values(config)
    rendered = render(secret)
    assert _sanitize_error(f"diagnostic {rendered} end", values) == f"diagnostic {render('[REDACTED]')} end"
    assert _sanitize_error("diagnostic token=unknown end", values) == "diagnostic [REDACTED] end"
