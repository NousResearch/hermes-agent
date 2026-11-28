"""Regression for MCP SDK Connection closed errors after a driver replacement."""
import threading
from typing import Any, cast
from unittest.mock import MagicMock
import pytest
from mcp.shared.exceptions import MCPError
from tools.computer_use import tool
from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend_session import _CuaDriverSession


def _session(effects):
    session = cast(Any, _CuaDriverSession.__new__(_CuaDriverSession))
    session._started = True
    session._lock = threading.Lock()
    session._declared_session_id = 'read-before-action'
    session._capabilities = {}
    session._tool_schemas = {}
    session._timeout_suspect = False
    session._reconnects = []
    session._stop_lifecycle_locked = lambda: session._reconnects.append('stop')
    session._start_lifecycle_locked = lambda: session._reconnects.append('start')
    session._call_tool_async = lambda name, args: (name, args)
    session._bridge = MagicMock()
    session._bridge.run.side_effect = effects
    return session


@pytest.mark.parametrize('exc, expected', [
    (MCPError(code=-32000, message='Connection closed'), True),
    (MCPError(code=-32000, message='Permission denied'), False),
    (MCPError(code=-32602, message='Connection closed'), False),
    (RuntimeError('Connection closed'), False),
    (EOFError(), True),
])
def test_classify_only_transport_connection_closed(exc, expected):
    assert _CuaDriverSession._is_closed_session_error(exc) is expected


@pytest.mark.parametrize('operation', ['list_windows', 'get_window_state', 'screenshot', 'get_desktop_state'])
def test_mcp_closed_reconnects_and_restores_label_before_safe_read(operation):
    session = _session([
        MCPError(code=-32000, message='Connection closed'),
        {'isError': False},
        {'isError': False, 'data': {'ok': True}},
    ])
    args = {'session': 'read-before-action'}
    assert session.call_tool(operation, args)['data'] == {'ok': True}
    assert session._reconnects == ['stop', 'start']
    assert [c.args[0][0] for c in session._bridge.run.call_args_list] == [
        operation, 'start_session', operation,
    ]


@pytest.mark.parametrize('operation', ['click', 'type_text', 'press_key'])
def test_mcp_closed_never_replays_mutations(operation):
    session = _session([
        MCPError(code=-32000, message='Connection closed'),
        {'isError': False},
    ])
    result = session.call_tool(operation, {'session': 'read-before-action'})
    assert result['isError'] is True
    assert result['structuredContent']['code'] == 'transport_outcome_unknown'
    assert result['structuredContent']['next_step'] == 'fresh_state'
    assert [c.args[0][0] for c in session._bridge.run.call_args_list] == [
        operation, 'start_session',
    ]


@pytest.mark.parametrize('failure_stage', ['restart', 'redeclare'])
def test_mutation_stays_outcome_unknown_when_reconnect_fails(failure_stage):
    session = _session([MCPError(code=-32000, message='Connection closed'),
                        RuntimeError('replacement transport unavailable')])
    if failure_stage == 'restart':
        session._start_lifecycle_locked = MagicMock(side_effect=RuntimeError('restart failed'))
    result = session.call_tool('click', {'session': 'read-before-action'})
    assert result['structuredContent']['code'] == 'transport_outcome_unknown'
    assert result['structuredContent']['next_step'] == 'fresh_state'
    assert [c.args[0][0] for c in session._bridge.run.call_args_list].count('click') == 1


@pytest.mark.parametrize('code', ['transport_outcome_unknown', 'timeout_outcome_unknown'])
def test_unknown_mutation_requires_observation_not_escalation(code):
    payload = tool._action_payload(ActionResult(ok=False, action='click', code=code))
    assert payload['verdict']['decision'] == 'verify_fresh_state'
    assert 'do not repeat' in payload['verdict']['hint'].lower()
