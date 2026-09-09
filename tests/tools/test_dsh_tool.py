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
- run_headless subprocess lifecycle (create_subprocess_exec mocked): argv/env
  construction, launch failures (missing executable / bad cwd), timeout with
  kill-tree + bounded drain
- rpc()/list_sessions HTTP transport (aiohttp mocked): Typert frame shape,
  HTTP error mapping (401/403 auth, 500), non-JSON body, connection refused,
  compaction + limit, malformed session.list payload
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


# ---------------------------------------------------------------------------
# run_headless(): subprocess lifecycle (create_subprocess_exec fully mocked)
# ---------------------------------------------------------------------------

def _fake_proc(returncode, stdout=b"", stderr=b"", *, first_communicate=None):
    """Build a fake asyncio subprocess with the surface run_headless touches."""
    state = {"communicate_calls": 0}

    class _Proc:
        pid = 4242

        def __init__(self):
            # instance attr: a class-body ``returncode = returncode`` would
            # hit the unbound class-local (Python scoping gotcha)
            self.returncode = returncode

        async def communicate(self):
            state["communicate_calls"] += 1
            if first_communicate is not None and state["communicate_calls"] == 1:
                raise first_communicate
            return stdout, stderr

        def kill(self):
            state["killed"] = True

    return _Proc(), state


def _spawn_fake(monkeypatch, proc, *, fail=None):
    """Monkeypatch asyncio.create_subprocess_exec; capture argv/env/cwd."""
    from tools import dsh_client

    captured = {}

    async def _fake_exec(*argv, **kwargs):
        captured["argv"] = argv
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        captured["start_new_session"] = kwargs.get("start_new_session")
        if fail is not None:
            raise fail
        return proc

    monkeypatch.setattr(dsh_client.asyncio, "create_subprocess_exec", _fake_exec)
    return captured


def test_run_headless_missing_command_raises_not_configured():
    from tools.dsh_client import run_headless, DshNotConfigured

    with pytest.raises(DshNotConfigured) as ei:
        asyncio.run(run_headless("do a thing", config={}))
    assert ei.value.code == "DSH_NOT_CONFIGURED"
    assert "dsh.command" in ei.value.message


def test_run_headless_nonpositive_config_timeout_raises(monkeypatch):
    from tools.dsh_client import run_headless, DshNotConfigured

    captured = {}

    async def _never_called(*argv, **kwargs):
        captured["spawned"] = True

    monkeypatch.setattr("asyncio.create_subprocess_exec", _never_called)
    with pytest.raises(DshNotConfigured):
        asyncio.run(
            run_headless("x", config={"command": ["dsh"], "timeout": 0})
        )
    assert "spawned" not in captured  # fail fast, before any process starts


def test_run_headless_launch_file_not_found(monkeypatch):
    from tools.dsh_client import run_headless, DshLaunchFailed

    proc, _ = _fake_proc(0)
    _spawn_fake(monkeypatch, proc, fail=FileNotFoundError(2, "no such file"))
    with pytest.raises(DshLaunchFailed) as ei:
        asyncio.run(run_headless("x", config={"command": ["dsh"]}))
    assert ei.value.code == "DSH_LAUNCH_FAILED"
    assert "executable not found" in ei.value.message


def test_run_headless_launch_oserror_bad_cwd(monkeypatch):
    from tools.dsh_client import run_headless, DshLaunchFailed

    proc, _ = _fake_proc(0)
    _spawn_fake(monkeypatch, proc, fail=OSError(2, "bad cwd"))
    with pytest.raises(DshLaunchFailed) as ei:
        asyncio.run(
            run_headless("x", config={"command": ["dsh"]}, cwd="Z:/no/such/dir")
        )
    assert ei.value.code == "DSH_LAUNCH_FAILED"
    assert "cwd" in ei.value.message


def test_run_headless_success_builds_argv_env_and_summary(monkeypatch):
    from tools.dsh_client import run_headless, _DEFAULT_PROFILE

    proc, _ = _fake_proc(0, stdout=b"final assistant text here\n")
    captured = _spawn_fake(monkeypatch, proc)
    config = {
        "command": ["node", "--import", "tsx/esm", "D:/harness/bin.ts"],
        "profile": "headless",
        "bash_dir": "D:/Git/bin",
        "cwd": "F:/work",
    }
    result = asyncio.run(run_headless("fix the bug", config=config))
    assert result["ok"] is True
    assert result["status"] == "completed"
    assert result["output"] == "final assistant text here"
    assert result["cwd"] == "F:/work"

    argv = captured["argv"]
    assert argv == ("node", "--import", "tsx/esm", "D:/harness/bin.ts",
                    "--profile", "headless", "fix the bug")
    # bash_dir is prepended to PATH so the child never resolves the WSL shim.
    assert captured["env"]["PATH"].startswith("D:/Git/bin" + __import__("os").pathsep)
    assert captured["cwd"] == "F:/work"
    assert captured["start_new_session"] is (__import__("sys").platform != "win32")
    assert _DEFAULT_PROFILE == "headless"  # guard: test relies on this default


def test_run_headless_profile_default_and_goal_arg(monkeypatch):
    """profile/argv defaults when the config omits them."""
    from tools.dsh_client import run_headless

    proc, _ = _fake_proc(0, stdout=b"ok\n")
    captured = _spawn_fake(monkeypatch, proc)
    asyncio.run(run_headless("do it", config={"command": ["dsh"]}))
    assert captured["argv"] == ("dsh", "--profile", "headless", "do it")


def test_run_headless_timeout_kills_tree_and_raises(monkeypatch):
    from tools import dsh_client
    from tools.dsh_client import run_headless, DshTimeoutError

    proc, state = _fake_proc(
        1, stdout=b"partial output before the deadline\n",
        first_communicate=asyncio.TimeoutError(),
    )
    _spawn_fake(monkeypatch, proc)
    killed = []

    def _spy_kill(target):
        killed.append(target)

    monkeypatch.setattr(dsh_client, "_kill_process_tree", _spy_kill)
    with pytest.raises(DshTimeoutError) as ei:
        asyncio.run(
            asyncio.wait_for(
                run_headless("slow job", config={"command": ["dsh"], "timeout": 5}),
                timeout=15,
            )
        )
    assert ei.value.code == "DSH_TIMEOUT"
    assert "killed" in ei.value.message
    assert killed == [proc]                      # the tree killer ran on it
    assert state["communicate_calls"] == 2       # bounded drain after the kill
    assert "partial output" in ei.value.extra["stdout_tail"]


# ---------------------------------------------------------------------------
# rpc() / list_sessions(): HTTP transport (aiohttp fully mocked)
# ---------------------------------------------------------------------------

class _FakeResp:
    """Stand-in for an aiohttp response: status + text() + async CM."""

    def __init__(self, status=200, body=""):
        self.status = status
        self._body = body

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _fake_aiohttp(monkeypatch, resp_factory):
    """Inject a minimal fake aiohttp module; capture posted frames."""
    import sys
    import types

    captured = []

    class _ClientTimeout:
        def __init__(self, **kwargs):
            self.total = kwargs.get("total")

    class _ClientConnectionError(OSError):
        pass

    class _FakeSession:
        def __init__(self):
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.closed = True
            return False

        def post(self, url, *, json=None, timeout=None):
            captured.append({"url": url, "json": json, "timeout": timeout})
            return resp_factory()

    fake = types.ModuleType("aiohttp")
    fake.ClientSession = _FakeSession
    fake.ClientTimeout = _ClientTimeout
    fake.ClientConnectionError = _ClientConnectionError
    monkeypatch.setitem(sys.modules, "aiohttp", fake)
    return captured


def test_rpc_success_posts_typert_envelope(monkeypatch):
    from tools.dsh_client import rpc

    ok_body = json.dumps(
        {"type": "server-response", "rpcId": "x",
         "result": {"ok": True, "value": {"items": [1, 2]}}}
    )
    captured = _fake_aiohttp(monkeypatch, lambda: _FakeResp(200, ok_body))
    value = asyncio.run(
        rpc("session.list", {}, config={"url": "http://127.0.0.1:3080"})
    )
    assert value == {"items": [1, 2]}

    call = captured[0]
    assert call["url"] == "http://127.0.0.1:3080/api/session.list"
    frame = call["json"]
    assert frame["type"] == "client-request"
    assert frame["method"] == "session.list"
    assert frame["payload"] == {}
    assert frame["rpcId"].startswith("hermes-")
    assert call["timeout"].total == 15


def test_rpc_http_error_mapping(monkeypatch):
    from tools.dsh_client import rpc, DshAuthError, DshProtocolError

    for status, exc_type, code in [
        (401, DshAuthError, "DSH_AUTH_FAILED"),
        (403, DshAuthError, "DSH_AUTH_FAILED"),
        (500, DshProtocolError, "DSH_PROTOCOL_ERROR"),
    ]:
        _fake_aiohttp(monkeypatch, lambda s=status: _FakeResp(s, "boom"))
        with pytest.raises(exc_type) as ei:
            asyncio.run(rpc("session.list", {}, config={"url": "http://127.0.0.1:3080"}))
        assert ei.value.code == code


def test_rpc_non_json_body_protocol_error(monkeypatch):
    from tools.dsh_client import rpc, DshProtocolError

    _fake_aiohttp(monkeypatch, lambda: _FakeResp(200, "<html>not json</html>"))
    with pytest.raises(DshProtocolError) as ei:
        asyncio.run(rpc("session.list", {}, config={"url": "http://127.0.0.1:3080"}))
    assert "non-JSON" in ei.value.message


def test_rpc_connection_refused_unreachable(monkeypatch):
    from tools.dsh_client import rpc, DshUnreachable

    def _raising_resp():
        fake = __import__("sys").modules["aiohttp"]
        raise fake.ClientConnectionError("connection refused")

    _fake_aiohttp(monkeypatch, _raising_resp)
    with pytest.raises(DshUnreachable) as ei:
        asyncio.run(rpc("session.list", {}, config={"url": "http://127.0.0.1:3080"}))
    assert ei.value.code == "DSH_UNREACHABLE"
    assert "not listening" in ei.value.message


def test_rpc_missing_url_not_configured():
    from tools.dsh_client import rpc, DshNotConfigured

    with pytest.raises(DshNotConfigured):
        asyncio.run(rpc("session.list", {}, config={"url": ""}))


def test_list_sessions_compacts_and_limits(monkeypatch):
    from tools import dsh_client
    from tools.dsh_client import list_sessions

    items = []
    for i in range(10):
        items.append({
            "sessionId": f"s{i}",
            "cwd": f"F:/work{i}",
            "agentPreset": "standard",
            "running": i % 2 == 0,
            "updatedAt": 12345 + i,
            "projections": {"values": {"title": f"Task {i}"}},
        })

    async def _fake_rpc(method, payload, **kw):
        assert method == "session.list"
        return {"items": items}

    monkeypatch.setattr(dsh_client, "rpc", _fake_rpc)
    out = asyncio.run(list_sessions(limit=3))
    assert out["ok"] is True
    assert out["count"] == 3
    assert out["items"][0]["title"] == "Task 0"
    assert out["items"][0]["sessionId"] == "s0"


def test_list_sessions_missing_items_protocol_error(monkeypatch):
    from tools import dsh_client
    from tools.dsh_client import list_sessions, DshProtocolError

    async def _fake_rpc(method, payload, **kw):
        return {"unexpected": True}

    monkeypatch.setattr(dsh_client, "rpc", _fake_rpc)
    with pytest.raises(DshProtocolError):
        asyncio.run(list_sessions())
