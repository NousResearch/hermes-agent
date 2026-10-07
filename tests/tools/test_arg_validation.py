"""Silent tool-argument drops (INTA-269): wrong calls must say so, valid calls stay untouched."""

import json
from types import SimpleNamespace

import pytest

from model_tools import handle_function_call


def _call(name, args, task_id):
    return json.loads(handle_function_call(name, args, task_id=task_id))


def test_missing_required_argument_is_rejected_before_the_tool_runs(tmp_path):
    for name, args, missing in [("search_files", {"path": str(tmp_path)}, "pattern"),
                                ("web_search", {"queries": ["x"]}, "query"),
                                ("execute_code", {"command": "print(1)"}, "code")]:
        result = _call(name, args, f"t-{name}")
        assert missing in result["error"] and "NOT invoked" in result["error"]
        assert "total_count" not in result


def test_unknown_key_runs_the_call_and_names_the_ignored_argument(tmp_path):
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "b.ini").write_text("needle\n")
    result = _call("search_files", {"pattern": "needle", "path": str(tmp_path), "include": "*.ini"}, "t-inc")
    assert result["total_count"] == 2                       # still ran
    assert "include" in result["warning"] and "file_glob" in result["warning"]
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
    assert validate_tool_args("write_file", {"path": "p", "content": "c", "cross_profile": True}) == (None, None)



def test_execute_code_does_not_treat_command_as_an_alias(monkeypatch):
    seen = []

    def dispatch(name, args, **kwargs):
        seen.append(args)
        return json.dumps({"output": "ran code"})

    monkeypatch.setattr("model_tools.registry.dispatch", dispatch)
    rejected = _call("execute_code", {"command": "print(1)"}, "command-only")
    assert "missing required argument(s): code" in rejected["error"]
    assert seen == []

    args = {"code": "print(1)", "command": "print(2)"}
    result = _call("execute_code", args, "code-and-command")
    assert seen == [args] and result["output"] == "ran code"
    assert "Ignored unknown argument(s): command." in result["warning"]


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
def test_rewritten_arguments_are_validated_and_warnings_name_only_dropped_keys(tmp_path, monkeypatch, phase):
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
    assert result["total_count"] == 1
    warning_keys = result["warning"].split(". Valid arguments:")[0]
    assert "unused_option" in warning_keys
    assert "query" not in warning_keys and "include" not in warning_keys


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


def test_warning_is_first_field_and_keeps_a_handler_warning():
    from tools.arg_validation import attach_arg_warning
    out = attach_arg_warning(json.dumps({"output": "x", "warning": "truncated"}), "Ignored unknown argument(s): a.")
    data = json.loads(out)
    assert next(iter(data)) == "warning"
    assert data["warning"].startswith("Ignored unknown argument(s): a.") and data["warning"].endswith("truncated")
