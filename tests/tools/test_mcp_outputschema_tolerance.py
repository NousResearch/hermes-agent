"""Tolerant ``outputSchema`` revalidation (#101330, ``tools/mcp_tool_structured_output.py``).

``mcp`` 2.x ``ClientSession.call_tool`` raises ``RuntimeError`` after a completed RPC when a tool
with a declared ``outputSchema`` returns no ``structuredContent`` or non-conforming content. Hermes
then dropped the usable ``content`` and struck the breaker, so a reachable server went
"unreachable" after three calls. Real shapes: Semrush MCP v2 answers quota exhaustion as a bare
text block on tools that all declare a schema; Bonusly returns ``required`` fields as ``null``.
"""
import asyncio
import json
import logging
import threading
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp.client.auth.oauth2")
from mcp import types as mcp_types
from mcp.client.session import ClientSession

from tools import mcp_tool_structured_output as hook_mod
from tools.mcp_tool_structured_output import _HOOK_MARKER, install_lenient_output_validation

SCHEMA = {"type": "object", "properties": {"reports": {"type": "array"}}, "required": ["reports"]}
QUOTA_TEXT = ('{"code":"no_api_units","message":"The user has an active subscription, but does not have '
              'enough API units to complete this request.","retryable":false}')
DEGRADED = "serving its content unvalidated"


@pytest.fixture(autouse=True)
def _fresh_warn_once_state():
    hook_mod._warned.clear()
    yield
    hook_mod._warned.clear()


def _session(schemas=None) -> ClientSession:
    """Real SDK session on unconnected in-memory streams; never entered."""
    import anyio
    _read_send, read_recv = anyio.create_memory_object_stream(0)
    write_send, _write_recv = anyio.create_memory_object_stream(0)
    session = ClientSession(read_recv, write_send)
    session._tool_output_schemas = dict(schemas if schemas is not None else {"report": SCHEMA})
    return session


def _result(text=None, structured=None, *, present=False, is_error=False):
    """Wire-shaped CallToolResult; ``present=True`` sends ``structuredContent: null``."""
    payload = {"content": [{"type": "text", "text": text}] if text is not None else []}
    if structured is not None or present:
        payload["structuredContent"] = structured
    if is_error:
        payload["isError"] = True
    return mcp_types.CallToolResult.model_validate(payload)


def _validate(session, name, result):
    return asyncio.run(session.validate_tool_result(name, result))


def _with_canned_reply(session, result):
    async def send_request(*args, **kwargs):
        return result

    session.send_request = send_request
    return session


def _recording_stub(schemas, impl=None):
    """Session stand-in whose original validator records every delegation."""
    calls = []

    async def validate_tool_result(name, result):
        calls.append(name)
        if impl is not None:
            impl(name, result)

    return types.SimpleNamespace(_tool_output_schemas=schemas, validate_tool_result=validate_tool_result), calls


def _degraded_records(caplog):
    return [r for r in caplog.records if r.name == hook_mod.logger.name and DEGRADED in r.getMessage()]


# --------------------------------------------------------------------------- SDK behaviour this exists for

@pytest.mark.parametrize("result", [
    _result(text=QUOTA_TEXT),
    _result(text="{}", structured={"nope": 1}),
], ids=["missing-structured-content", "non-conforming-structured-content"])
def test_sdk_raises_after_a_completed_rpc_without_the_hook(result):
    """Pin the SDK behaviour; if this stops raising, the hook can be retired."""
    with pytest.raises(RuntimeError):
        _validate(_session(), "report", result)


# --------------------------------------------------------------------------- degraded, not raised

@pytest.mark.parametrize("result", [
    _result(text=QUOTA_TEXT),
    _result(text=QUOTA_TEXT, present=True),
    _result(text="{}", structured={"nope": 1}),
    _result(text="{}", structured={}),
], ids=["missing", "explicit-null", "non-conforming", "empty-object"])
def test_declared_schema_failures_are_served_not_raised(result):
    session = _session()
    assert install_lenient_output_validation(session, "srv") is True
    assert _validate(session, "report", result) is None


def test_uncompilable_schema_is_served_not_raised():
    session = _session({"report": {"type": 12}})
    install_lenient_output_validation(session, "srv")
    assert _validate(session, "report", _result(text="x", structured={"a": 1})) is None


def test_call_tool_returns_the_server_text_end_to_end():
    session = _with_canned_reply(_session(), _result(text=QUOTA_TEXT))
    with pytest.raises(RuntimeError, match="did not return structured content"):
        asyncio.run(session.call_tool("report", {}))
    install_lenient_output_validation(session, "srv")
    assert asyncio.run(session.call_tool("report", {})).content[0].text == QUOTA_TEXT


def test_is_error_results_never_reach_the_validator():
    session = _with_canned_reply(_session(), _result(text="boom", is_error=True))
    calls = []

    async def validate_tool_result(name, result):
        calls.append(name)

    session.validate_tool_result = validate_tool_result
    install_lenient_output_validation(session, "srv")
    assert asyncio.run(session.call_tool("report", {})).is_error is True
    assert calls == []


# --------------------------------------------------------------------------- everything else stays with the SDK

def test_conforming_structured_content_is_delegated_and_not_logged(caplog):
    stub, calls = _recording_stub({"report": SCHEMA})
    install_lenient_output_validation(stub, "srv")
    with caplog.at_level(logging.DEBUG, logger=hook_mod.logger.name):
        asyncio.run(stub.validate_tool_result("report", _result(text="[]", structured={"reports": []})))
    assert calls == ["report"]
    assert not _degraded_records(caplog)


def test_tool_without_schema_is_delegated_and_not_logged(caplog):
    stub, calls = _recording_stub({"plain": None})
    install_lenient_output_validation(stub, "srv")
    with caplog.at_level(logging.DEBUG, logger=hook_mod.logger.name):
        asyncio.run(stub.validate_tool_result("plain", _result(text="hello")))
    assert calls == ["plain"]
    assert not _degraded_records(caplog)


def test_unlisted_tool_is_delegated_and_its_errors_propagate():
    """No cached schema: the SDK refreshes via list_tools(); its failures are not ours to hide."""
    def boom(name, result):
        raise RuntimeError("list_tools failed")

    stub, calls = _recording_stub({}, impl=boom)
    install_lenient_output_validation(stub, "srv")
    with pytest.raises(RuntimeError, match="list_tools failed"):
        asyncio.run(stub.validate_tool_result("ghost", _result(text="hello")))
    assert calls == ["ghost"]


def test_non_runtime_errors_propagate():
    def boom(name, result):
        raise ValueError("unexpected")

    stub, _calls = _recording_stub({"report": SCHEMA}, impl=boom)
    install_lenient_output_validation(stub, "srv")
    with pytest.raises(ValueError, match="unexpected"):
        asyncio.run(stub.validate_tool_result("report", _result(text="x", structured={"reports": []})))


# --------------------------------------------------------------------------- logging

def test_warns_once_per_tool_with_the_server_text(caplog):
    session = _session()
    install_lenient_output_validation(session, "srv")
    with caplog.at_level(logging.DEBUG, logger=hook_mod.logger.name):
        _validate(session, "report", _result(text=QUOTA_TEXT))
        _validate(session, "report", _result(text=QUOTA_TEXT))
    records = _degraded_records(caplog)
    assert [r.levelno for r in records] == [logging.WARNING, logging.DEBUG]
    assert "no_api_units" in records[0].getMessage() and "'report'" in records[0].getMessage()


def test_validation_failure_reason_is_logged(caplog):
    session = _session()
    install_lenient_output_validation(session, "srv")
    with caplog.at_level(logging.DEBUG, logger=hook_mod.logger.name):
        _validate(session, "report", _result(text="{}", structured={"nope": 1}))
    (record,) = _degraded_records(caplog)
    assert "failed outputSchema validation" in record.getMessage()
    assert "reports" in record.getMessage()


# --------------------------------------------------------------------------- install mechanics

def test_install_is_idempotent():
    session = _session()
    assert install_lenient_output_validation(session, "srv") is True
    first = session.validate_tool_result
    assert getattr(first, _HOOK_MARKER, False)
    assert install_lenient_output_validation(session, "srv") is True
    assert session.validate_tool_result is first, "a second install must not wrap the wrapper"


def test_wraps_the_private_validator_name_of_mcp_1x():
    calls = []

    async def _validate_tool_result(name, result):
        calls.append(name)

    stub = types.SimpleNamespace(_tool_output_schemas={"t": SCHEMA}, _validate_tool_result=_validate_tool_result)
    assert install_lenient_output_validation(stub, "legacy") is True
    asyncio.run(stub._validate_tool_result("t", _result(text="x")))
    assert calls == []
    asyncio.run(stub._validate_tool_result("t", _result(text="x", structured={"reports": []})))
    assert calls == ["t"]


def test_no_validator_means_nothing_to_install():
    assert install_lenient_output_validation(object(), "none") is False


def test_serve_session_installs_the_hook_before_the_handshake():
    """HTTP, SSE and stdio sessions all route through ``_serve_session``."""
    from tools import mcp_tool_transport as transport

    session = _session()
    order = []

    class _Conn(transport.MCPServerTransportMixin):
        def __init__(self):
            self.name = "outputschema-wiring-test"
            self._config = {}
            self._ready = asyncio.Event()
            self._ever_connected = False
            self.session = None
            self.initialize_result = None

        async def _negotiate_session(self, sess, timeout):
            order.append(getattr(sess.validate_tool_result, _HOOK_MARKER, False))
            return "init"

        async def _discover_tools(self):
            pass

        async def _wait_for_lifecycle_event(self):
            return "shutdown"

    assert asyncio.run(_Conn()._serve_session(session, 1.0)) == "shutdown"
    assert order == [True]


# --------------------------------------------------------------------------- the breaker (#101330)

def _install_session_server(mcp_tool_module, name, session):
    server = MagicMock()
    server.name = name
    server.session = session
    ready = threading.Event()
    ready.set()
    server._ready = ready
    server._is_recycled_stdio.return_value = False
    mcp_tool_module._servers[name] = server
    mcp_tool_module._server_error_counts.pop(name, None)
    mcp_tool_module._server_breaker_opened_at.pop(name, None)
    return server


def _cleanup_server(mcp_tool_module, name):
    mcp_tool_module._servers.pop(name, None)
    mcp_tool_module._server_error_counts.pop(name, None)
    mcp_tool_module._server_breaker_opened_at.pop(name, None)


@pytest.mark.parametrize("hooked", [True, False], ids=["with-hook", "without-hook"])
def test_repeated_schema_mismatches_do_not_trip_the_breaker(monkeypatch, tmp_path, hooked):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool
    from tools import mcp_tool_loop
    from tools.mcp_tool_handlers import _make_tool_handler

    name = "outputschema-breaker-test"
    session = _with_canned_reply(_session(), _result(text=QUOTA_TEXT))
    if hooked:
        install_lenient_output_validation(session, name)
    _install_session_server(mcp_tool, name, session)
    mcp_tool_loop._ensure_mcp_loop()
    try:
        handler = _make_tool_handler(name, "report", 10.0)
        replies = [json.loads(handler({})) for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD + 1)]
        if hooked:
            assert all(r.get("result") == QUOTA_TEXT for r in replies), replies
            assert mcp_tool._server_error_counts.get(name, 0) == 0
        else:  # the bug: the usable text is dropped and the breaker opens
            assert all("error" in r for r in replies), replies
            assert "unreachable" in replies[-1]["error"].lower()
    finally:
        _cleanup_server(mcp_tool, name)
