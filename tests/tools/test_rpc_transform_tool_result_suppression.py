"""#138520: execute_code RPC dispatch must skip ``transform_tool_result``.

A tool call made from inside a sandbox script never enters the conversation — its result
returns to the generated ``hermes_tools`` stub, which ``json.loads``es it. A plugin that
appends to or rewrites a tool result (a note, a preview, a warning block) therefore made
every such call raise ``JSONDecodeError`` inside the script even though the tool had
already run, so scripts often retried a write that had happened.

The fix mirrors ``suppress_post_tool_call_hook``: a ContextVar-scoped skip wrapped around
the two RPC dispatchers only. The outer ``execute_code`` result still transforms on the
normal agent-loop path — that is the boundary the hook documents ("before the result
enters context").
"""

import model_tools
from tools.registry import registry

_RAW_RESULT = '{"output": "original"}'
_REWRITTEN_RESULT = _RAW_RESULT + "\n\nplugin note"


def _install_rewrite_hook(monkeypatch):
    """Registry dispatch returns plain JSON; a transform_tool_result plugin appends to it."""
    monkeypatch.setattr(registry, "dispatch", lambda name, args, **kw: _RAW_RESULT)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kw: [_REWRITTEN_RESULT] if hook_name == "transform_tool_result" else [],
    )


def _call(tool_name="dummy_tool"):
    return model_tools.handle_function_call(
        tool_name, {}, task_id="t1", session_id="s1", tool_call_id="tc1",
        skip_pre_tool_call_hook=True,
    )


def test_agent_loop_result_still_transforms(monkeypatch):
    """Outside any suppress block the rewrite keeps working (the documented hook contract)."""
    _install_rewrite_hook(monkeypatch)
    assert _call() == _REWRITTEN_RESULT


def test_suppress_block_returns_raw_result_and_resets(monkeypatch):
    _install_rewrite_hook(monkeypatch)
    with model_tools.suppress_transform_tool_result_hook():
        assert _call() == _RAW_RESULT
    # The token-based reset restores the transform for the next call, exception or not.
    assert _call() == _REWRITTEN_RESULT


def test_suppress_false_reenables_inside_block(monkeypatch):
    _install_rewrite_hook(monkeypatch)
    with model_tools.suppress_transform_tool_result_hook():
        with model_tools.suppress_transform_tool_result_hook(False):
            assert _call() == _REWRITTEN_RESULT
        assert _call() == _RAW_RESULT


def test_default_rpc_dispatch_skips_transform(monkeypatch):
    """The per-call sandbox dispatch and the remote poll loop both come from here."""
    from tools.code_execution_rpc import _default_dispatch

    _install_rewrite_hook(monkeypatch)
    dispatch = _default_dispatch("task-1")
    assert dispatch("dummy_tool", {}) == _RAW_RESULT


def test_kernel_authority_dispatch_skips_transform(monkeypatch):
    """Session kernels route every call through the current cell's CellAuthority."""
    from tools.code_kernel import CellAuthority

    _install_rewrite_hook(monkeypatch)
    authority = CellAuthority("turn-1")
    assert authority.dispatch("dummy_tool", {}) == _RAW_RESULT
