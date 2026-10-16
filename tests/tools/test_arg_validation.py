"""Silent tool-argument drops (INTA-269): wrong calls must say so, valid calls stay untouched."""

import json
from types import SimpleNamespace

import pytest

from model_tools import handle_function_call
from tools.registry import registry


def _call(name, args, task_id):
    return json.loads(handle_function_call(name, args, task_id=task_id))


def test_missing_required_argument_is_rejected_before_the_tool_runs(tmp_path):
    for name, args, missing in [("search_files", {"path": str(tmp_path)}, "pattern"),
                                ("web_search", {"queries": ["x"]}, "query"),
                                ("execute_code", {"command": "print(1)"}, "code")]:
        result = _call(name, args, f"t-{name}")
        assert missing in result["error"] and "NOT invoked" in result["error"]
        assert "total_count" not in result


def test_unknown_search_key_is_rejected_and_corrected_call_runs(tmp_path):
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "b.ini").write_text("needle\n")
    result = _call("search_files", {"pattern": "needle", "path": str(tmp_path), "include": "*.ini"}, "t-inc")
    assert "NOT invoked" in result["error"]
    assert "include" in result["error"] and "file_glob" in result["error"]
    assert "total_count" not in result
    valid = _call("search_files", {"pattern": "needle", "path": str(tmp_path), "file_glob": "*.ini"}, "t-ok")
    assert valid["total_count"] == 1 and "warning" not in valid


def test_write_file_rejects_edit_keys_without_touching_the_file(tmp_path):
    target = tmp_path / "big.txt"
    target.write_text("keep me\n" * 100)
    result = _call("write_file", {"path": str(target), "content": "x", "old_string": "keep", "new_string": "y"}, "t-wf")
    assert "old_string" in result["error"] and "NOT invoked" in result["error"]
    assert target.read_text() == "keep me\n" * 100


def test_arguments_a_handler_accepts_beyond_its_schema_are_not_flagged():
    from tools.arg_validation import validate_tool_args
    assert validate_tool_args("write_file", {"path": "p", "content": "c", "cross_profile": True}) is None



def test_execute_code_does_not_treat_command_as_an_alias(monkeypatch):
    seen = []

    def dispatch(name, args, **kwargs):
        seen.append(args)
        return json.dumps({"output": "ran code"})

    monkeypatch.setattr("model_tools.registry.dispatch", dispatch)
    rejected = _call("execute_code", {"command": "print(1)"}, "command-only")
    assert "missing required argument(s): code" in rejected["error"]
    assert "terminal(command=...)" in rejected["error"]
    assert "execute_code(code=...)" in rejected["error"]
    assert seen == []

    args = {"code": "print(1)", "command": "print(2)"}
    result = _call("execute_code", args, "code-and-command")
    assert "command" in result["error"] and "NOT invoked" in result["error"]
    assert "terminal(command=...)" in result["error"]
    assert "execute_code(code=...)" in result["error"]
    assert seen == []
    valid = _call("execute_code", {"code": "print(1)"}, "corrected-code")
    assert valid["output"] == "ran code"
    assert seen == [{"code": "print(1)"}]


def _install_argument_rewrite(monkeypatch, phase, rewrite):
    monkeypatch.setattr(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
        lambda name, args, **kwargs: (None, rewrite(args) if phase == "hook" else None),
    )
    if phase == "execution":
        def middleware(**kwargs):
            return kwargs["next_call"](rewrite(kwargs["args"]))

        manager = SimpleNamespace(_middleware={"tool_execution": [middleware]})
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)


@pytest.mark.parametrize("phase", ["hook", "execution"])
def test_rewritten_arguments_are_validated_and_errors_name_only_remaining_unknown_keys(tmp_path, monkeypatch, phase):
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "b.ini").write_text("needle\n")

    def rewrite(args):
        rewritten = dict(args)
        rewritten["pattern"] = rewritten.pop("query")
        rewritten["file_glob"] = rewritten.pop("include")
        rewritten["unused_option"] = True
        return rewritten

    _install_argument_rewrite(monkeypatch, phase, rewrite)
    result = json.loads(handle_function_call(
        "search_files", {"query": "needle", "path": str(tmp_path), "include": "*.ini"},
        task_id=f"rewrite-{phase}", skip_tool_request_middleware=True,
    ))
    assert "NOT invoked" in result["error"]
    assert "total_count" not in result
    rejected_keys = result["error"].split(". The tool")[0]
    assert "unused_option" in rejected_keys
    assert "query" not in rejected_keys and "include" not in rejected_keys


@pytest.mark.parametrize("phase", ["hook", "execution"])
@pytest.mark.parametrize("invalid", ["missing_required", "unknown_write_key"])
def test_rewrites_cannot_dispatch_invalid_arguments(tmp_path, monkeypatch, phase, invalid):
    target = tmp_path / "keep.txt"
    original = "keep me\n"
    target.write_text(original)
    if invalid == "missing_required":
        name, args, rejected_key = "search_files", {"pattern": "needle", "path": str(tmp_path)}, "pattern"

        def rewrite(args):
            return {key: value for key, value in args.items() if key != "pattern"}
    else:
        name, args, rejected_key = "write_file", {"path": str(target), "content": "replacement"}, "old_string"

        def rewrite(args):
            return {**args, "old_string": "keep me"}

    _install_argument_rewrite(monkeypatch, phase, rewrite)
    result = json.loads(handle_function_call(
        name, args, task_id=f"reject-{phase}-{invalid}", skip_tool_request_middleware=True,
    ))
    assert rejected_key in result["error"] and "NOT invoked" in result["error"]
    assert "total_count" not in result
    assert target.read_text() == original


@pytest.mark.parametrize("shape", ["missing", "unknown", "rewritten_unknown"])
def test_invalid_writes_report_validation_errors(tmp_path, monkeypatch, shape):
    from hermes_cli import lifecycle

    target = tmp_path / "keep.txt"
    target.write_text("original\n")
    args = {"path": str(target), "content": "replacement"}
    if shape == "missing":
        args.pop("content")
    elif shape == "unknown":
        args["old_string"] = "original"
    else:
        _install_argument_rewrite(monkeypatch, "execution", lambda args: {**args, "old_string": "original"})
    events = []
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name in {"post_tool_call", "transform_tool_result"})
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda name, **kwargs: (events.append(kwargs) or [])
                        if name in {"post_tool_call", "transform_tool_result"} else [])
    result = json.loads(handle_function_call(
        "write_file", args, task_id=f"invalid-observer-{shape}", skip_tool_request_middleware=True,
    ))
    assert "NOT invoked" in result["error"] and target.read_text() == "original\n"
    assert len(events) == 2
    assert all(event["status"] == "error" and event["error_type"] == "invalid_arguments" for event in events)
    assert all(event["error_message"] == result["error"] for event in events)


@pytest.mark.parametrize("name,args", [
    ("search_files", {"pattern": "needle", "invented_filter": "*.py"}),
    ("web_search", {"query": "needle", "invented_limit": 2}),
    ("write_file", {"path": "output.txt", "content": "changed", "invented_mode": "append"}),
    ("execute_code", {"code": "print(1)", "workdir": "/somewhere"}),
    ("execute_code", {"code": "print(1)", "timeout": 30}),
    ("execute_code", {"code": "print(1)", "invented_option": True}),
])
def test_unknown_arguments_never_dispatch_or_retry(monkeypatch, name, args):
    dispatched = []
    monkeypatch.setattr("model_tools.registry.dispatch", lambda *a, **kw: dispatched.append(a))
    result = _call(name, args, f"unsupported-{name}")
    assert "NOT invoked" in result["error"]
    assert "Valid arguments:" in result["error"]
    assert dispatched == []


@pytest.mark.parametrize("phase", ["hook", "execution"])
def test_middleware_can_repair_a_call_before_validation(tmp_path, monkeypatch, phase):
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "b.ini").write_text("needle\n")

    def rewrite(args):
        return {"pattern": args["query"], "file_glob": args["include"], "path": args["path"]}

    _install_argument_rewrite(monkeypatch, phase, rewrite)
    result = json.loads(handle_function_call(
        "search_files", {"query": "needle", "path": str(tmp_path), "include": "*.ini"},
        task_id=f"repaired-{phase}", skip_tool_request_middleware=True,
    ))
    assert result["total_count"] == 1
    assert "error" not in result


@pytest.mark.parametrize("parameters", [
    {"properties": {"code": {"type": "string"}, "extension_option": {}}, "required": ["code"]},
    {"properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": True},
    {"properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": {}},
    {"properties": {"code": {"type": "string"}}, "required": ["code"],
     "additionalProperties": False, "patternProperties": {"^extension_": {}}},
])
def test_current_schema_can_explicitly_allow_extension_arguments(monkeypatch, parameters):
    original = registry.get_schema
    monkeypatch.setattr("tools.arg_validation.registry.get_schema", lambda name:
                        {"name": name, "parameters": parameters} if name == "execute_code" else original(name))
    seen = []
    result_text = json.dumps({"output": "handled", "warning": "existing handler warning"})

    def dispatch(name, args, **kwargs):
        seen.append(args)
        return result_text

    monkeypatch.setattr("model_tools.registry.dispatch", dispatch)
    args = {"code": "print(1)", "extension_option": "value"}
    result = handle_function_call("execute_code", args, task_id="extension-arguments")
    assert seen == [args]
    assert result == result_text
    rejected = _call("execute_code", {"extension_option": "value"}, "extension-missing-code")
    assert "missing required argument(s): code" in rejected["error"]
    assert seen == [args]


def test_rejected_code_cannot_write_a_file_and_explicit_correction_can(tmp_path, monkeypatch):
    from tools.code_kernel import shutdown_all_kernels

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    marker = tmp_path / "executed.txt"
    code = f"from pathlib import Path\nPath({str(marker)!r}).write_text('done')"
    shutdown_all_kernels()
    try:
        rejected = _call("execute_code", {"code": code, "timeout": 30}, "code-side-effect")
        assert "NOT invoked" in rejected["error"]
        assert not marker.exists()
        corrected = _call("execute_code", {"code": code}, "code-side-effect")
        assert not corrected.get("error"), corrected
        assert marker.read_text() == "done"
    finally:
        shutdown_all_kernels()


@pytest.mark.parametrize("name", ["custom_plugin_tool", "mcp__example__lookup"])
def test_newly_registered_tools_are_validated_without_an_allowlist(monkeypatch, name):
    from tools.registry import ToolRegistry

    isolated = ToolRegistry()
    seen = []
    isolated.register(name=name, toolset="test", schema={
        "name": name, "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
        },
    }, handler=lambda args, **kw: seen.append(args) or json.dumps({"answer": args["query"]}))
    for module in ["model_tools", "tools.arg_coercion", "tools.arg_validation"]:
        monkeypatch.setattr(f"{module}.registry", isolated)
    assert "NOT invoked" in _call(name, {}, "new-tool-missing")["error"]
    assert "invented" in _call(name, {"query": "hello", "invented": True}, "new-tool-unknown")["error"]
    assert seen == []
    assert _call(name, {"query": "hello"}, "new-tool-valid")["answer"] == "hello"
    assert seen == [{"query": "hello"}]


@pytest.mark.parametrize("name,args", [
    ("write_file", {"path": "out.txt", "content": "data", "cross_profile": True}),
    ("patch", {"path": "out.txt", "old_string": "a", "new_string": "b", "cross_profile": True}),
    ("patch", {"mode": "patch", "patch": "*** Begin Patch\n*** Delete File: out.txt\n*** End Patch"}),
    ("terminal", {"command": "true", "background": True, "notify_on_complete": True, "watch_patterns": []}),
    ("delegate_task", {"goal": "Summarize", "context": "text", "output_schema": {}, "role": "leaf"}),
    ("clarify", {"question": "Which?", "choices": ["a", "b"], "multi_select": False}),
    ("cronjob_manage", {"action": "list", "include_disabled": True}),
    ("cronjob_manage", {"action": "create", "monitor_script": "probe.py", "monitor_url": None}),
    ("image_generate", {"prompt": "a tree", "image_url": "image.png", "reference_image_urls": [], "upscale": True}),
    ("video_generate", {"prompt": "a tree", "audio": True, "seed": 12, "negative_prompt": "noise"}),
    ("browser_exec", {"code": "await page.title()", "local": True}),
    ("kanban_create", {"title": "Investigate", "assignee": "worker", "project_id": "project", "session_id": "session"}),
    ("kanban_attach_url", {"task_id": "task", "url": "https://example.com/a", "title": "a.txt"}),
    ("a2a_call", {"agent_name": "helper", "text": "Summarize", "contextId": "context"}),
    ("a2a_history", {"contextId": "context"}),
    ("a2a_orchestrate", {"capability": "summarize", "task": "text"}),
])
def test_supported_compatibility_arguments_do_not_exempt_the_tool(monkeypatch, name, args):
    from tools.arg_validation import validate_tool_args

    active_registry = registry
    if name.startswith("a2a_"):
        from plugins.platforms.a2a import tools as a2a_tools
        from tools.registry import ToolRegistry
        active_registry = ToolRegistry()
        a2a_tools.register_tools(SimpleNamespace(register_tool=active_registry.register))
        monkeypatch.setattr("tools.arg_validation.registry", active_registry)
    assert active_registry.get_schema(name) is not None
    assert validate_tool_args(name, args) is None
    rejected = json.loads(validate_tool_args(name, {**args, "invented_parameter": True}))
    assert "invented_parameter" in rejected["error"]
    assert "NOT invoked" in rejected["error"]


def test_current_dynamic_parameters_are_accepted_without_mutating_static_schema(monkeypatch):
    from tools.arg_validation import validate_tool_args
    from tools.registry import ToolRegistry

    isolated = ToolRegistry()
    schema = {"name": "dynamic_plugin", "parameters": {
        "type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"],
    }}
    isolated.register(name="dynamic_plugin", toolset="test", schema=schema, handler=lambda args, **kw: "{}",
                      dynamic_schema_overrides=lambda: {"parameters": {
                          "type": "object", "properties": {"new_option": {}}, "required": ["prompt"],
                      }})
    monkeypatch.setattr("tools.arg_validation.registry", isolated)
    assert validate_tool_args("dynamic_plugin", {"prompt": "hello", "new_option": 1}) is None
    assert "new_option" not in schema["parameters"]["properties"]
    assert "NOT invoked" in json.loads(validate_tool_args("dynamic_plugin", {"new_option": 1}))["error"]
    assert "invented" in json.loads(validate_tool_args("dynamic_plugin", {"prompt": "hello", "invented": 1}))["error"]


def test_registered_zero_argument_tool_rejects_invented_arguments(monkeypatch):
    from tools.arg_validation import validate_tool_args
    from tools.registry import ToolRegistry

    isolated = ToolRegistry()
    isolated.register(name="no_args", toolset="test", schema={
        "parameters": {"type": "object", "properties": {}}}, handler=lambda args, **kw: "{}")
    monkeypatch.setattr("tools.arg_validation.registry", isolated)
    assert validate_tool_args("no_args", {}) is None
    assert "NOT invoked" in json.loads(validate_tool_args("no_args", {"invented": 1}))["error"]


@pytest.mark.parametrize("extra", [{"timeout": 60}, {"workdir": "/tmp"}, {"invented": True}])
def test_execute_code_unrelated_argument_errors_do_not_suggest_switching_tools(monkeypatch, extra):
    seen = []
    monkeypatch.setattr("model_tools.registry.dispatch", lambda *args, **kwargs: seen.append(args))
    result = _call("execute_code", {"code": "print(1)", **extra}, "code-unrelated-argument")
    assert "NOT invoked" in result["error"]
    assert next(iter(extra)) in result["error"]
    assert "terminal(command=...)" not in result["error"]
    assert seen == []


def test_tool_owned_hint_cannot_bypass_rejection_and_valid_calls_skip_it(monkeypatch):
    from tools.registry import ToolRegistry

    isolated = ToolRegistry()
    hints, executed = [], []

    def broken_hint(args):
        hints.append(args)
        args.clear()
        raise RuntimeError("hint unavailable")

    isolated.register(
        name="hinted_tool", toolset="test",
        schema={"name": "hinted_tool", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
        }},
        handler=lambda args, **kwargs: executed.append(dict(args)) or json.dumps({"answer": args["query"]}),
        argument_error_hint=broken_hint,
    )
    for module in ["model_tools", "tools.arg_coercion", "tools.arg_validation"]:
        monkeypatch.setattr(f"{module}.registry", isolated)
    args = {"wrong_key": "value"}
    result = _call("hinted_tool", args, "broken-hint")
    assert "missing required argument(s): query" in result["error"]
    assert "NOT invoked" in result["error"]
    assert args == {"wrong_key": "value"}
    assert executed == [] and len(hints) == 1
    assert _call("hinted_tool", {"query": "hello"}, "valid-hint") == {"answer": "hello"}
    assert executed == [{"query": "hello"}] and len(hints) == 1
