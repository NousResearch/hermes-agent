"""Tool calls made from inside an execute_code cell fire pre_tool_call hooks under the calling
conversation's identity (#145870): without it a hook sees ``session_id=""`` and, in a
multiplexed gateway, can only fall back to the process-wide HERMES_SESSION_ID of whichever
conversation bound a turn last."""

import json
from unittest.mock import patch

import pytest

from model_tools import handle_function_call
from tools.code_execution_rpc import _default_dispatch
from tools.code_kernel import shutdown_all_kernels


@pytest.fixture(autouse=True)
def _local_kernel(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    shutdown_all_kernels()
    with patch("tools.code_execution_tool._load_config", return_value={"mode": "strict", "timeout": 30}):
        yield
    shutdown_all_kernels()


@pytest.fixture
def hook_calls():
    calls = []

    def _record(tool_name, args, **hook_kwargs):
        calls.append((tool_name, hook_kwargs))
        return None, None

    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=_record):
        yield calls


def _nested(calls, tool):
    return [kw for name, kw in calls if name == tool]


def test_nested_call_in_a_local_cell_carries_the_outer_session_and_turn(tmp_path, hook_calls):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    code = f"from hermes_tools import read_file\nprint(read_file({str(target)!r}))"

    result = json.loads(handle_function_call(
        "execute_code", {"code": code}, task_id="task-A", session_id="sess-A", turn_id="turn-A",
        conversation_id="conv-A", enabled_tools=["execute_code", "read_file"]))

    assert result["status"] == "success", result
    nested = _nested(hook_calls, "read_file")
    assert len(nested) == 1
    assert nested[0]["session_id"] == "sess-A"
    assert nested[0]["turn_id"] == "turn-A"
    assert nested[0]["conversation_id"] == "conv-A"


def test_nested_call_over_the_remote_rpc_dispatch_carries_the_outer_session_and_turn(tmp_path, hook_calls):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")

    dispatch = _default_dispatch("task-A", "sess-A", "turn-A", "conv-A")
    assert "hello" in json.loads(dispatch("read_file", {"path": str(target)}))["content"]

    nested = _nested(hook_calls, "read_file")
    assert len(nested) == 1
    assert nested[0]["session_id"] == "sess-A"
    assert nested[0]["turn_id"] == "turn-A"
    assert nested[0]["conversation_id"] == "conv-A"


@pytest.mark.parametrize("route", ["kernel", "per_call"])
def test_both_remote_routes_hand_the_outer_identity_to_the_rpc_dispatch(route, monkeypatch):
    import inspect
    from contextlib import ExitStack

    from tests.tools.test_code_kernel_remote import ScriptedEnv, _cell, _spawn_ok_handlers
    from tools import code_execution_rpc
    from tools.code_kernel_remote import shutdown_all_remote_kernels

    real, seen = code_execution_rpc._default_dispatch, []

    def _recording(*args, **kwargs):
        seen.append(inspect.signature(real).bind(*args, **kwargs).arguments)
        return real(*args, **kwargs)

    handlers = [("command -v python3", lambda c: {"output": "OK\n", "returncode": 0})]
    if route == "kernel":
        handlers += _spawn_ok_handlers([_cell(stdout="ran\n")])
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    with ExitStack() as stack:
        stack.callback(shutdown_all_remote_kernels)
        stack.enter_context(patch("tools.code_execution_tool._get_or_create_env",
                                  return_value=(ScriptedEnv(handlers), "ssh")))
        stack.enter_context(patch("tools.code_execution_tool._remote_write"))
        stack.enter_context(patch("tools.code_kernel_remote._remote_write"))
        stack.enter_context(patch.object(code_execution_rpc, "_default_dispatch", side_effect=_recording))
        if route == "per_call":
            stack.enter_context(patch("tools.code_kernel_remote.execute_in_remote_kernel", return_value=None))
        result = json.loads(handle_function_call(
            "execute_code", {"code": "print(1)"}, task_id="task-A", session_id="sess-A", turn_id="turn-A",
            conversation_id="conv-A", enabled_tools=["execute_code", "read_file"]))

    assert result["status"] == "success", result
    assert [(a["session_id"], a["turn_id"], a["conversation_id"]) for a in seen] == [("sess-A", "turn-A", "conv-A")]


# ---------------------------------------------------------------------------
# Inner-call identity: an inner call is dispatched with ``<owner>#<n>`` and is the
# owner of its own post_tool_call. Without both halves the observer that opens a
# span on pre_tool_call can never close it, so every nested call ends up marked
# "Turn ended before this call's response/result was recorded."
# ---------------------------------------------------------------------------


def test_a_nested_call_carries_its_own_id_and_emits_both_hooks(tmp_path):
    """The whole contract through the real registry and a real kernel: an inner call inside a
    suppressing executor (how the agent loop runs every tool) still emits pre AND post, and both
    carry ``<owner>#1``. Fails if ``tool_call_id`` is dropped by ``registry.dispatch``'s
    signature filter, and fails if the leaf does not re-enable the post hook."""
    from hermes_cli import lifecycle, plugins
    import model_tools

    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    code = f"from hermes_tools import read_file\nprint(read_file({str(target)!r}))"
    events = []

    def _pre(tool_name, args, **hook_kwargs):
        events.append(("pre", tool_name, hook_kwargs.get("tool_call_id", "")))
        return None, None

    def _post(hook_name, **hook_kwargs):
        if hook_name == "post_tool_call":
            events.append(("post", hook_kwargs.get("tool_name"), hook_kwargs.get("tool_call_id", "")))

    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=_pre), \
         patch.object(lifecycle, "has_hook", lambda name: name == "post_tool_call"), \
         patch.object(lifecycle, "invoke_hook", _post), \
         patch("tools.approval.check_execute_code_guard",
               return_value={"approved": True, "user_approved": False, "message": ""}):
        with model_tools.suppress_post_tool_call_hook():
            result = json.loads(handle_function_call(
                "execute_code", {"code": code}, task_id="task-A", tool_call_id="B1",
                session_id="sess-A", turn_id="turn-A", conversation_id="conv-A",
                enabled_tools=["execute_code", "read_file"]))

    assert result["status"] == "success", result
    assert [e for e in events if e[1] == "read_file"] == [
        ("pre", "read_file", "B1#1"),
        ("post", "read_file", "B1#1"),
    ]


def test_cell_authority_mints_one_id_per_call_and_clears_the_post_suppression():
    import model_tools
    from tools.code_kernel import CellAuthority

    seen = []

    def _fake(name, args, **kwargs):
        seen.append((name, kwargs.get("tool_call_id"), model_tools._post_tool_call_hook_suppressed.get()))
        return json.dumps({"ok": True})

    with patch.object(model_tools, "handle_function_call", _fake):
        with model_tools.suppress_post_tool_call_hook():
            authority = CellAuthority("task-A", "sess-A", "turn-A", "conv-A", "B1")
            authority.dispatch("web_search", {"query": "a"})
            authority.dispatch("web_search", {"query": "b"})

    assert seen == [("web_search", "B1#1", False), ("web_search", "B1#2", False)]


def test_cell_authority_without_an_owner_passes_no_id():
    import model_tools
    from tools.code_kernel import CellAuthority

    seen = []

    def _fake(name, args, **kwargs):
        seen.append(kwargs.get("tool_call_id"))
        return json.dumps({"ok": True})

    with patch.object(model_tools, "handle_function_call", _fake):
        CellAuthority("task-A").dispatch("web_search", {"query": "a"})

    assert seen == [None]


def test_the_rpc_dispatch_mints_only_when_an_owner_is_known():
    import model_tools

    seen = []

    def _fake(name, args, **kwargs):
        seen.append(kwargs.get("tool_call_id"))
        return json.dumps({"ok": True})

    with patch.object(model_tools, "handle_function_call", _fake):
        _default_dispatch("task-A")("web_search", {"query": "x"})
        owned = _default_dispatch("task-A", owner_tool_call_id="B2")
        owned("web_search", {"query": "y"})
        owned("web_search", {"query": "z"})

    assert seen == [None, "B2#1", "B2#2"]


def test_the_poll_loop_forwards_the_owner_and_a_refused_call_takes_no_id(tmp_path):
    """The remote/Docker transport end to end: the real file client allocates seq, the poll loop
    dispatches with ``<owner>#<n>``, and an over-budget refusal neither dispatches nor consumes a
    number."""
    import os
    import threading

    import model_tools
    from tests.tools.test_code_execution_rpc import _FakeFileRpcEnvironment
    from tools.code_execution_tool import _rpc_poll_loop, generate_hermes_tools_module

    rpc_dir = tmp_path / "rpc"
    rpc_dir.mkdir()
    token = "rpc-token-probe"
    env = _FakeFileRpcEnvironment(str(rpc_dir))
    tool_log, counter, stop = [], [0], threading.Event()
    seen = []

    def _fake(name, args, **kwargs):
        seen.append((name, kwargs.get("tool_call_id")))
        return json.dumps({"ok": 1})

    def _poll():
        with patch.object(model_tools, "handle_function_call", _fake):
            _rpc_poll_loop(
                env, str(rpc_dir), "task-A", tool_log, counter,
                max_tool_calls=2, allowed_tools=frozenset({"web_search"}),
                stop_event=stop, rpc_token=token, owner_tool_call_id="B7",
            )

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    namespace = {"__name__": "hermes_tools"}
    try:
        with patch.dict(os.environ, {"HERMES_RPC_DIR": str(rpc_dir), "HERMES_RPC_TOKEN": token}):
            exec(generate_hermes_tools_module(["web_search"], transport="file"), namespace)
            call = namespace["_call"]
            responses = [call("web_search", {"query": q}) for q in ("a", "b", "c")]
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert seen == [("web_search", "B7#1"), ("web_search", "B7#2")]
    assert responses[0] == {"ok": 1} and responses[1] == {"ok": 1}
    assert "Tool call limit reached (2)" in json.dumps(responses[2])
    assert counter[0] == 2, "a refused call must not consume budget or an id"


def test_the_identity_record_keeps_its_hook_contract():
    """PR 1 adds no field to the identity record: the hook payload stays exactly six keys."""
    import model_tools

    assert set(model_tools._CallIds("task-A").hook_kwargs()) == {
        "task_id", "session_id", "tool_call_id", "turn_id", "api_request_id", "conversation_id",
    }
    assert model_tools._CallIds("task-A").hook_kwargs()["tool_call_id"] == ""
