"""A ``tool_call`` bridge unwrap must run the UNDERLYING tool's own policy hooks.

``handle_function_call`` unwraps the Tool Search ``tool_call`` wrapper so "every downstream
hook (pre/post, edit approval, guardrails) sees the real tool name, never the bridge". A
caller that fired the wrapper's policy itself (the agent loop's parse step, a run where the
parse-time peel could not resolve the catalog) therefore passes
``skip_pre_tool_call_hook=True`` — but the unwrapped re-dispatch forwarded that flag, so the
REAL tool skipped ``pre_tool_call`` entirely: a plugin guardrail matched on ``terminal`` or
``write_file`` never ran, and ``tool_request`` / ``tool_execution`` middleware never saw the
call either. The connector batch re-dispatch (``tools/connectors/dispatch.py``) already
passes ``False`` for all three flags; these tests pin the single-tool path to the same rule.

Each case drives the real ``handle_function_call`` bridge so the assertion fails on the
behaviour (hook never called / tool still executed) rather than on an import.
"""

import json

import pytest

TOOLSET = "mcp-probe-hookpin"
TOOL = "mcp_probe_hookpin_op"


@pytest.fixture
def deferred_tool():
    """Register a deferrable tool and hand back the list of argument dicts it ran with."""
    from tools.registry import invalidate_check_fn_cache, registry

    calls = []

    def _handler(args, task_id=None, **kw):
        calls.append(args)
        return json.dumps({"ok": True, "args": args})

    registry.register(
        name=TOOL,
        handler=_handler,
        schema={"name": TOOL, "description": f"desc {TOOL}",
                "parameters": {"type": "object", "properties": {}}},
        toolset=TOOLSET,
    )
    invalidate_check_fn_cache()
    return calls


def _bridge_skipping_wrapper_policy(**extra):
    """Invoke *TOOL* the way the agent loop invokes ``tool_call``: wrapper policy already fired."""
    import model_tools

    return model_tools.handle_function_call(
        "tool_call",
        {"name": TOOL, "arguments": {"document_id": "abc"}},
        enabled_toolsets=[TOOLSET],
        session_id="bridge-hook-scope",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
        **extra,
    )


def test_unwrapped_tool_runs_its_own_pre_tool_call_hook(monkeypatch, deferred_tool):
    from hermes_cli import plugins

    seen = []

    def _hook(name, args, **kw):
        seen.append((name, args))
        return None, None

    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", _hook)

    result = json.loads(_bridge_skipping_wrapper_policy())

    assert result["ok"] is True
    # The wrapper's flag describes the wrapper; the real tool still owns its hook.
    assert [name for name, _ in seen] == [TOOL]
    assert seen[0][1] == {"document_id": "abc"}
    assert deferred_tool == [{"document_id": "abc"}]


def test_a_guardrail_matched_on_the_underlying_tool_blocks_it(monkeypatch, deferred_tool):
    from hermes_cli import plugins

    def _hook(name, args, **kw):
        return ("blocked by policy", None) if name == TOOL else (None, None)

    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", _hook)

    result = json.loads(_bridge_skipping_wrapper_policy())

    assert "blocked by policy" in json.dumps(result)
    assert deferred_tool == []  # the vetoed tool never executed


def test_unwrapped_tool_runs_its_own_request_and_execution_middleware(monkeypatch, deferred_tool):
    from hermes_cli import plugins

    events = []

    def request(**kw):
        events.append(("request", kw["tool_name"]))
        return {"args": {"document_id": "request-rewrite"}, "source": "hookpin"}

    def execution(**kw):
        events.append(("execution", kw["tool_name"]))
        return kw["next_call"]({"document_id": "execution-rewrite"})

    monkeypatch.setattr(plugins.get_plugin_manager(), "_middleware",
                        {"tool_request": [request], "tool_execution": [execution]})

    result = json.loads(_bridge_skipping_wrapper_policy())

    assert result["ok"] is True
    assert events == [("request", TOOL), ("execution", TOOL)]
    assert deferred_tool == [{"document_id": "execution-rewrite"}]
