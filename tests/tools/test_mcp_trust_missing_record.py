"""A call that finds no recorded trust tier must not run ungated (#153706).

The call-time gate read a missing ``_server_trust_levels`` entry as ``full``, so a server the profile
configures ``trust: untrusted`` ran its write-capable tools with no prompt whenever the record was
absent for the calling scope. The gate now reads the raw ``trust`` key from the profile's own config
(no ``${VAR}`` expansion) and fails closed when that read fails or no source configures the server.
"""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_constants import get_hermes_home
from tools import mcp_tool
from tools import mcp_tool_handlers as _mcp_handlers
from tools import mcp_tool_registration as _mcp_registration


def _run_on_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _with_lock():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_with_lock())
    finally:
        loop.close()


@pytest.fixture
def session():
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], isError=False,
                             structuredContent=None)
    sess = MagicMock()
    sess.call_tool = AsyncMock(return_value=result)
    with patch.dict(mcp_tool._servers, {"srv": SimpleNamespace(session=sess, _rpc_lock=None)}), \
         patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=_run_on_loop), \
         patch.dict(mcp_tool._server_error_counts, {}, clear=True), \
         patch.dict(mcp_tool._server_trust_levels, {}, clear=True), \
         patch.dict(mcp_tool._tool_read_only_hints, {}, clear=True):
        yield sess


def _configure(trust, **extra):
    entry = {"command": "fake-mcp-server", **extra}
    if trust is not None:
        entry["trust"] = trust
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"mcp_servers": {"srv": entry}}))  # JSON is YAML
    return entry


def _write_call_asks():
    """Run a write-capable call with no trust record; True when the gate asked (and the call was blocked)."""
    handler = _mcp_handlers._make_tool_handler("srv", "delete_repo", 30.0)
    with patch("tools.approval_prompt.request_elicitation_consent", return_value="decline") as consent:
        raw = handler({"repo": "x"})
    if consent.called:
        assert "error" in json.loads(raw)
    return consent.called


def test_untrusted_server_without_a_trust_record_is_gated(session):
    """Registered untrusted, then the record is gone for the caller: the write call must ask."""
    config = _configure("untrusted")
    _mcp_registration._record_tool_trust_metadata("srv", config, [SimpleNamespace(name="delete_repo")])
    mcp_tool._server_trust_levels.clear()
    handler = _mcp_handlers._make_tool_handler("srv", "delete_repo", 30.0)
    with patch("tools.approval_prompt.request_elicitation_consent", return_value="decline") as consent:
        raw = handler({"repo": "x"})
    consent.assert_called_once()
    session.call_tool.assert_not_awaited()
    assert "error" in json.loads(raw)


def test_missing_record_on_a_full_trust_server_still_runs(session):
    """The config fallback keeps a default (no ``trust`` key) server ungated, as registration would."""
    _configure(None)
    handler = _mcp_handlers._make_tool_handler("srv", "delete_repo", 30.0)
    with patch("tools.approval_prompt.request_elicitation_consent") as consent:
        raw = handler({"repo": "x"})
    consent.assert_not_called()
    assert json.loads(raw) == {"result": "ok"}


def test_an_approved_gated_call_is_logged(session, caplog):
    """Operators can tell 'gate asked and the user accepted' from 'gate skipped'."""
    mcp_tool._server_trust_levels["srv"] = "untrusted"
    handler = _mcp_handlers._make_tool_handler("srv", "delete_repo", 30.0)
    with caplog.at_level(logging.INFO, logger=_mcp_handlers.logger.name), \
         patch("tools.approval_prompt.request_elicitation_consent", return_value="accept"):
        handler({"repo": "x"})
    session.call_tool.assert_awaited_once()
    assert any("approved 'delete_repo'" in r.getMessage() and "'srv'" in r.getMessage() for r in caplog.records)


def test_templated_untrusted_server_is_gated_under_unscoped_multiplex(session, caplog):
    """A ``${VAR}`` in the entry must not void the tier: expanding it with no secret scope bound raises."""
    from agent.secret_scope import set_multiplex_active
    _configure("untrusted", url="https://mcp.example.invalid/", headers={"Authorization": "Bearer ${PROBE_TOKEN}"})
    set_multiplex_active(True)
    try:
        with caplog.at_level(logging.WARNING, logger=_mcp_handlers.logger.name):
            assert _write_call_asks()
    finally:
        set_multiplex_active(False)
    session.call_tool.assert_not_awaited()
    assert any("'untrusted' (configured value)" in r.getMessage() for r in caplog.records)


def test_unparsable_config_fails_closed(session, caplog):
    (get_hermes_home() / "config.yaml").write_text("mcp_servers:\n  srv: {command: x, trust: untrusted\n  [")
    with caplog.at_level(logging.WARNING, logger=_mcp_handlers.logger.name):
        assert _write_call_asks()
    session.call_tool.assert_not_awaited()
    assert any("config read failed" in r.getMessage() for r in caplog.records)


def test_server_in_no_config_source_fails_closed(session, caplog):
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"mcp_servers": {"other": {"command": "x"}}}))
    with caplog.at_level(logging.WARNING, logger=_mcp_handlers.logger.name):
        assert _write_call_asks()
    session.call_tool.assert_not_awaited()
    assert any("server not in config" in r.getMessage() for r in caplog.records)
