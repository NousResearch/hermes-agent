"""Tests for the dsh integration: tools/dsh_client.py + tools/dsh_tool.py.

Covers (mocked -- no real dsh process, no model spend):
- check_fn truth table (no config -> hidden; command/url present -> visible)
- headless outcome matrix via the pure interpret_run() classifier:
  success / non-zero exec failure / auth-ish failure / empty stdout / truncation
- JSON-RPC envelope parsing: ok value, server error envelope, malformed frames
- HTTP status mapping (401/403 auth, 404 protocol)
- handler dispatch matrix via mocked client:
  run success, run missing goal, run timeout, list unreachable, unknown action
- registry integration: toolset "dsh", async handler, schema shape
"""

import asyncio
import json

import pytest


def _run_handler(args):
    """Run the async dsh_task handler synchronously."""
    from tools.dsh_tool import _handle_dsh_task

    return asyncio.run(_handle_dsh_task(args))


def _load_json(text):
    return json.loads(text)


# ---------------------------------------------------------------------------
# Config loading & availability gate
# ---------------------------------------------------------------------------

def _config_with(block):
    """Return a loader that makes load_config_readonly() yield a config whose
    ``dsh`` block is *block* (or absent when block is None)."""
    def _loader():
        return {"dsh": block} if block is not None else {}
    return _loader


def test_load_config_absent_section(monkeypatch):
    from tools import dsh_client

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _config_with(None))
    assert dsh_client.load_config() == {}


def test_load_config_present_section(monkeypatch):
    from tools import dsh_client

    block = {"command": ["dsh"], "url": "http://127.0.0.1:3080"}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _config_with(block))
    assert dsh_client.load_config() == block


@pytest.mark.parametrize(
    "block,expected",
    [
        (None, False),                      # no dsh section -> hidden
        ({}, False),                        # empty section -> hidden
        ({"url": "http://127.0.0.1:3080"}, True),   # list-capable
        ({"command": ["dsh"]}, True),       # run-capable
        ({"command": [], "url": ""}, False),
    ],
)
def test_check_dsh_available_truth_table(monkeypatch, block, expected):
    from tools import dsh_tool

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _config_with(block))
    assert dsh_tool._check_dsh_available() is expected


def test_check_fn_config_cache_safe(monkeypatch):
    """check_fn must not probe the network -- config presence only."""
    from tools import dsh_tool

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _config_with({"command": ["dsh"]}))
    assert dsh_tool._check_dsh_available() is True


def test_cfg_command_shapes():
    from tools.dsh_client import _cfg_command

    assert _cfg_command({"command": ["node", "--import", "tsx/esm", "bin.ts"]}) == [
        "node", "--import", "tsx/esm", "bin.ts"
    ]
    assert _cfg_command({"command": "dsh --profile headless"}) == ["dsh", "--profile", "headless"]
    assert _cfg_command({"command": []}) == []
    assert _cfg_command({}) == []


# ---------------------------------------------------------------------------
# interpret_run(): headless outcome matrix
# ---------------------------------------------------------------------------

def test_interpret_run_success():
    from tools.dsh_client import interpret_run

    result = interpret_run(
        0, "done: created hello.py\n", "",
        elapsed_s=12.6, cwd="F:/work", sessions_dir="C:/Users/g/.dsh/sessions",
    )
    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["output"] == "done: created hello.py"
    assert result["cwd"] == "F:/work"
    assert result["elapsed_s"] == 13


def test_interpret_run_truncates_long_output():
    from tools.dsh_client import interpret_run, _OUTPUT_CAP_CHARS

    result = interpret_run(
        0, "x" * (_OUTPUT_CAP_CHARS + 5000), "",
        elapsed_s=1, cwd=None, sessions_dir="s",
    )
    assert "[truncated" in result["output"]
    assert len(result["output"]) <= _OUTPUT_CAP_CHARS + 60


def test_interpret_run_nonzero_exit_exec_failed():
    from tools.dsh_client import interpret_run, DshExecFailed

    with pytest.raises(DshExecFailed) as ei:
        interpret_run(1, "", "error: agent crashed", elapsed_s=3, cwd=None, sessions_dir="s")
    assert ei.value.code == "DSH_EXEC_FAILED"
    assert "error: agent crashed" in ei.value.message


def test_interpret_run_auth_failure_classified():
    from tools.dsh_client import interpret_run, DshAuthError

    with pytest.raises(DshAuthError) as ei:
        interpret_run(
            1, "",
            '{"code":"auth","message":"401 invalid api key"}',
            elapsed_s=3, cwd=None, sessions_dir="s",
        )
    assert ei.value.code == "DSH_AUTH_FAILED"
    assert "DEEPSEEK_API_KEY" in ei.value.message


def test_interpret_run_empty_stdout_empty_result():
    from tools.dsh_client import interpret_run, DshEmptyResult

    with pytest.raises(DshEmptyResult) as ei:
        interpret_run(0, "   \n", "", elapsed_s=3, cwd=None, sessions_dir="s")
    assert ei.value.code == "DSH_EMPTY_RESULT"


# ---------------------------------------------------------------------------
# JSON-RPC envelope parsing & HTTP status mapping
# ---------------------------------------------------------------------------

def test_parse_envelope_ok():
    from tools.dsh_client import _parse_envelope

    value = _parse_envelope(
        {"type": "server-response", "rpcId": "r1", "result": {"ok": True, "value": {"items": []}}}
    )
    assert value == {"items": []}


def test_parse_envelope_server_error():
    from tools.dsh_client import _parse_envelope, DshRpcError

    with pytest.raises(DshRpcError) as ei:
        _parse_envelope(
            {
                "type": "server-response",
                "rpcId": "r1",
                "result": {
                    "ok": False,
                    "error": {"code": "bad-request", "message": "invalid client-request message"},
                },
            }
        )
    assert ei.value.code == "DSH_RPC_ERROR"
    assert "bad-request" in ei.value.message


@pytest.mark.parametrize(
    "frame",
    [
        "not-a-dict",
        {"type": "client-request", "rpcId": "x", "method": "session.list"},
        {"type": "server-response", "rpcId": "x", "result": {"ok": True}},  # no value
    ],
)
def test_parse_envelope_protocol_errors(frame):
    from tools.dsh_client import _parse_envelope, DshProtocolError

    with pytest.raises(DshProtocolError):
        _parse_envelope(frame)


def test_map_http_status():
    from tools.dsh_client import _map_http_status, DshAuthError, DshProtocolError

    assert isinstance(_map_http_status(401, "nope"), DshAuthError)
    assert isinstance(_map_http_status(403, "nope"), DshAuthError)
    assert isinstance(_map_http_status(404, "nope"), DshProtocolError)
    assert isinstance(_map_http_status(500, "boom"), DshProtocolError)


# ---------------------------------------------------------------------------
# Handler dispatch (client fully mocked)
# ---------------------------------------------------------------------------

def test_handler_run_success(monkeypatch):
    from tools import dsh_client

    async def _fake_run(task, *, cwd=None, timeout=None):
        assert task == "write hello.py and run it"
        assert cwd == "F:/work"
        assert timeout == 60
        return {"ok": True, "status": "completed", "exit_code": 0,
                "output": "wrote and ran hello.py", "cwd": "F:/work",
                "elapsed_s": 5}

    monkeypatch.setattr(dsh_client, "run_headless", _fake_run)
    payload = _load_json(
        _run_handler({"action": "run", "goal": "write hello.py and run it",
                      "cwd": "F:/work", "timeout": 60})
    )
    assert payload["ok"] is True
    assert payload["status"] == "completed"


def test_handler_run_missing_goal():
    payload = _load_json(_run_handler({"action": "run"}))
    assert payload["error"]
    assert payload["code"] == "DSH_BAD_ARGUMENT"


def test_handler_run_timeout_maps_code(monkeypatch):
    from tools import dsh_client
    from tools.dsh_client import DshTimeoutError

    async def _fake_timeout(task, **kw):
        raise DshTimeoutError("dsh task exceeded the 10s timeout and was killed",
                              extra={"elapsed_s": 10})

    monkeypatch.setattr(dsh_client, "run_headless", _fake_timeout)
    payload = _load_json(_run_handler({"action": "run", "goal": "long job", "timeout": 10}))
    assert payload["code"] == "DSH_TIMEOUT"
    assert "killed" in payload["error"]


def test_handler_run_not_configured(monkeypatch):
    from tools import dsh_client
    from tools.dsh_client import DshNotConfigured

    async def _fake_not_configured(task, **kw):
        raise DshNotConfigured("dsh.command is empty in the dsh: config block")

    monkeypatch.setattr(dsh_client, "run_headless", _fake_not_configured)
    payload = _load_json(_run_handler({"action": "run", "goal": "x"}))
    assert payload["code"] == "DSH_NOT_CONFIGURED"
    assert "dsh.command" in payload["error"]


def test_handler_list_unreachable(monkeypatch):
    from tools import dsh_client
    from tools.dsh_client import DshUnreachable

    async def _fake_list(**kw):
        raise DshUnreachable("dsh web is not listening at http://127.0.0.1:3080")

    monkeypatch.setattr(dsh_client, "list_sessions", _fake_list)
    payload = _load_json(_run_handler({"action": "list"}))
    assert payload["code"] == "DSH_UNREACHABLE"
    assert "not listening" in payload["error"]


def test_handler_list_success(monkeypatch):
    from tools import dsh_client

    async def _fake_list(**kw):
        assert kw.get("limit") == 5
        return {"ok": True, "count": 1,
                "items": [{"sessionId": "s1", "cwd": "F:/work", "agentPreset": "standard"}]}

    monkeypatch.setattr(dsh_client, "list_sessions", _fake_list)
    payload = _load_json(_run_handler({"action": "list", "limit": 5}))
    assert payload["ok"] is True
    assert payload["items"][0]["sessionId"] == "s1"


def test_handler_unknown_action():
    payload = _load_json(_run_handler({"action": "steer"}))
    assert payload["code"] == "DSH_BAD_ARGUMENT"
    assert "Unknown action" in payload["error"]


def test_handler_run_empty_goal_with_action_run():
    payload = _load_json(_run_handler({"action": "run", "goal": "   "}))
    assert payload["code"] == "DSH_BAD_ARGUMENT"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_entry_registered():
    from tools.registry import registry
    import tools.dsh_tool  # noqa: F401 -- module-level register() call

    entry = registry.get_entry("dsh_task")
    assert entry is not None
    assert entry.toolset == "dsh"
    assert entry.is_async is True
    assert entry.check_fn is tools.dsh_tool._check_dsh_available


def test_schema_shape():
    import tools.dsh_tool as mod

    schema = mod.DSH_TASK_SCHEMA
    assert schema["name"] == "dsh_task"
    props = schema["parameters"]["properties"]
    assert set(props) == {"action", "goal", "cwd", "timeout", "limit"}
    assert props["action"]["enum"] == ["run", "list"]
    assert props["goal"]["type"] == "string"


def test_tool_error_code_surfaces_as_extra_field():
    """tool_error(message, code=...) must carry the DSH_* code to the model."""
    from tools.dsh_tool import _err

    out = _load_json(_err("boom", code="DSH_TIMEOUT", elapsed_s=9))
    assert out["code"] == "DSH_TIMEOUT"
    assert out["elapsed_s"] == 9
