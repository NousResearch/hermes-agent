"""Dispatch refuses a call whose args omit a schema-required key, naming the key."""

import json

import model_tools
from tools.registry import registry

_SCHEMA = {
    "name": "zz_required_probe",
    "description": "probe",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
}


def _register(calls):
    registry.register(name="zz_required_probe", toolset="zz_probe", schema=_SCHEMA,
                      handler=lambda args, **kw: (calls.append(dict(args)), json.dumps({"ok": True}))[1])


def test_missing_required_arg_is_named_and_handler_not_run():
    calls: list = []
    _register(calls)
    try:
        result = json.loads(model_tools.handle_function_call("zz_required_probe", {"path": "a.txt"}))
        assert calls == []
        assert "missing required argument(s): content" in result["error"]
        assert "received: path" in result["error"]

        ok = json.loads(model_tools.handle_function_call("zz_required_probe", {"path": "a.txt", "content": ""}))
        assert ok == {"ok": True} and calls == [{"path": "a.txt", "content": ""}]
    finally:
        registry.deregister("zz_required_probe")


def test_undeclared_key_defers_to_handler_and_hooks_can_fill_required(monkeypatch):
    """Alias keys (``a2a_call`` takes ``agent_name``) reach the handler; a pre_tool_call
    hook that supplies the missing key is honoured because the check runs on final args."""
    calls: list = []
    _register(calls)
    try:
        model_tools.handle_function_call("zz_required_probe", {"path": "a.txt", "body": "x"})
        assert calls == [{"path": "a.txt", "body": "x"}]

        monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks",
                            lambda name, args, **kw: (None, {**args, "content": "filled"}))
        model_tools.handle_function_call("zz_required_probe", {"path": "b.txt"})
        assert calls[-1] == {"path": "b.txt", "content": "filled"}
    finally:
        registry.deregister("zz_required_probe")
