"""Unit tests for tool_eval/scorer.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import pytest
from tool_eval.scorer import (
    TestResult,
    _safe_first_choice,
    _extract_tool_calls,
    _has_text_content,
    _text_content,
    _is_infra_error,
    _check_arg_values,
    _check_type_compliance,
    _check_no_extra_params,
    _check_list_field,
    _score_single_args_ratio,
    score_test,
    score_debug_fixture,
)


# --- _safe_first_choice ---

def test_safe_first_choice_normal():
    raw = {"choices": [{"message": {"content": "hello"}}]}
    assert _safe_first_choice(raw) == {"message": {"content": "hello"}}


def test_safe_first_choice_null_choices():
    assert _safe_first_choice({"choices": None}) is None


def test_safe_first_choice_empty_list():
    assert _safe_first_choice({"choices": []}) is None


def test_safe_first_choice_missing_key():
    assert _safe_first_choice({}) is None


def test_safe_first_choice_not_dict():
    assert _safe_first_choice("not a dict") is None


# --- _extract_tool_calls ---

def _make_tool_response(name: str, args: dict) -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args)
                    }
                }]
            }
        }]
    }


def test_extract_tool_calls_standard_shape():
    raw = _make_tool_response("terminal", {"command": "ls"})
    calls = _extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "terminal"
    assert calls[0]["function"]["arguments"]["command"] == "ls"


def test_extract_tool_calls_empty_when_null_choices():
    raw = {"choices": None, "error": {"message": "rate limit"}}
    assert _extract_tool_calls(raw) == []


def test_extract_tool_calls_empty_when_no_tools():
    raw = {"choices": [{"message": {"content": "hello", "tool_calls": None}}]}
    assert _extract_tool_calls(raw) == []


def test_extract_tool_calls_multiple():
    raw = {
        "choices": [{
            "message": {
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "todo", "arguments": '{"todos": []}'}},
                    {"id": "c2", "type": "function", "function": {"name": "memory", "arguments": '{"action": "add", "target": "user"}'}},
                ]
            }
        }]
    }
    calls = _extract_tool_calls(raw)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "todo"
    assert calls[1]["function"]["name"] == "memory"


def test_extract_tool_calls_malformed_json_args():
    raw = {
        "choices": [{
            "message": {
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "NOT JSON"}}]
            }
        }]
    }
    calls = _extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["arguments"] == {}


# --- _has_text_content / _text_content ---

def test_text_content_extracts_string():
    raw = {"choices": [{"message": {"content": "Sure, I can help!"}}]}
    assert _text_content(raw) == "Sure, I can help!"


def test_text_content_empty_when_null():
    raw = {"choices": [{"message": {"content": None}}]}
    assert _text_content(raw) == ""


def test_has_text_content_true():
    raw = {"choices": [{"message": {"content": "hello"}}]}
    assert _has_text_content(raw) is True


def test_has_text_content_false_for_tool_call():
    raw = {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "todo"}}]}}]}
    assert _has_text_content(raw) is False


# --- _is_infra_error ---

def test_is_infra_error_rate_limit():
    raw = {"choices": None, "error": {"message": "rate limit exceeded", "code": 429}}
    assert _is_infra_error(raw) is True


def test_is_infra_error_502():
    raw = {"choices": None, "error": {"message": "Bad Gateway", "code": "502"}}
    assert _is_infra_error(raw) is True


def test_is_infra_error_all_null():
    raw = {"id": None, "choices": None, "model": None}
    assert _is_infra_error(raw) is True


def test_is_infra_error_false_for_normal():
    raw = {"choices": [{"message": {"content": "hello"}}], "id": "chatcmpl-123"}
    assert _is_infra_error(raw) is False


def test_is_infra_error_false_for_tool_call():
    raw = _make_tool_response("terminal", {"command": "ls"})
    raw["id"] = "chatcmpl-abc"
    assert _is_infra_error(raw) is False


# --- _check_arg_values ---

def test_check_arg_values_match():
    passed, details = _check_arg_values({"action": "create"}, {"action": "create"})
    assert passed
    assert details["action"]["passed"]


def test_check_arg_values_case_insensitive():
    passed, details = _check_arg_values({"action": "CREATE"}, {"action": "create"})
    assert passed


def test_check_arg_values_mismatch():
    passed, _ = _check_arg_values({"action": "delete"}, {"action": "create"})
    assert not passed


def test_check_arg_values_missing_key():
    passed, details = _check_arg_values({}, {"action": "create"})
    assert not passed
    assert details["action"]["actual"] is None


# --- _check_type_compliance ---

def test_check_type_compliance_pass():
    passed, _ = _check_type_compliance({"count": 3}, {"count": "int"})
    assert passed


def test_check_type_compliance_fail():
    passed, details = _check_type_compliance({"count": "3"}, {"count": "int"})
    assert not passed
    assert details["count"]["actual_type"] == "str"


def test_check_type_compliance_list():
    passed, _ = _check_type_compliance({"items": [1, 2]}, {"items": "list"})
    assert passed


# --- _check_no_extra_params ---

def test_no_extra_params_pass():
    schema = {"parameters": {"properties": {"command": {}, "timeout": {}}}}
    passed, msg = _check_no_extra_params({"command": "ls"}, schema)
    assert passed
    assert msg == ""


def test_no_extra_params_fail():
    schema = {"parameters": {"properties": {"command": {}}}}
    passed, msg = _check_no_extra_params({"command": "ls", "secret_flag": True}, schema)
    assert not passed
    assert "secret_flag" in msg


# --- _check_list_field ---

def test_check_list_field_pass():
    passed, msg = _check_list_field({"items": [1, 2, 3]}, "items", 3)
    assert passed
    assert "3" in msg


def test_check_list_field_too_short():
    passed, msg = _check_list_field({"items": [1]}, "items", 3)
    assert not passed


def test_check_list_field_missing():
    passed, msg = _check_list_field({}, "items", 1)
    assert not passed
    assert "missing" in msg


# --- _score_single_args_ratio ---

def test_score_single_args_all_criteria():
    spec = {
        "required_args": ["action", "target"],
        "arg_values": {"action": "add"},
        "list_field_check": {"field": "todos", "min_items": 2},
    }
    actual = {"action": "add", "target": "memory", "todos": [{"id": "1"}, {"id": "2"}]}
    ratio, details = _score_single_args_ratio(actual, spec)
    assert ratio == 1.0


def test_score_single_args_partial_credit():
    spec = {"required_args": ["action", "content"]}
    actual = {"action": "add"}  # missing content
    ratio, details = _score_single_args_ratio(actual, spec)
    assert ratio == 0.5


def test_score_single_args_no_criteria():
    ratio, details = _score_single_args_ratio({"anything": "goes"}, {})
    assert ratio == 1.0
    assert "No detailed arg scoring criteria" in details["_criteria_summary"]


# --- score_test integration ---

def test_score_test_no_tool_calls_pass():
    test_case = {
        "id": "t1", "category": "test", "description": "no tool",
        "expected": {"no_tool_calls": True},
    }
    raw = {"id": "x", "choices": [{"message": {"content": "Sure!", "tool_calls": None}}]}
    result = score_test(test_case, raw)
    assert result.score == 100
    assert result.passed


def test_score_test_no_tool_calls_fail():
    test_case = {
        "id": "t1", "category": "test", "description": "no tool",
        "expected": {"no_tool_calls": True},
    }
    raw = _make_tool_response("terminal", {"command": "ls"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    assert result.score == 0
    assert not result.passed


def test_score_test_tool_name_match():
    test_case = {
        "id": "t2", "category": "terminal", "description": "run ls",
        "expected": {"function_name": "terminal", "arguments": {}},
    }
    raw = _make_tool_response("terminal", {"command": "ls"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    assert result.score == 100
    assert result.passed


def test_score_test_wrong_tool_name():
    test_case = {
        "id": "t3", "category": "terminal", "description": "run ls",
        "expected": {"function_name": "terminal"},
    }
    raw = _make_tool_response("web_search", {"query": "ls command"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    # name_score=0 (wrong tool), args_spec={} so args_score=60 (full credit, no criteria)
    assert result.score == 60
    assert not result.passed


def test_score_test_40_pts_for_name_0_for_args():
    """Wrong tool name = 0 pts name, no args scored."""
    test_case = {
        "id": "t3b", "category": "test", "description": "wrong tool",
        "expected": {
            "function_name": "terminal",
            "arguments": {"required_args": ["command"]},
        },
    }
    raw = _make_tool_response("web_search", {"query": "ls"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    assert result.score == 0  # name mismatch = 0 name pts, 0 arg pts (tool not found)


def test_score_test_full_arg_scoring():
    test_case = {
        "id": "t4", "category": "memory", "description": "add memory",
        "expected": {
            "function_name": "memory",
            "arguments": {
                "required_args": ["action", "target"],
                "arg_values": {"action": "add", "target": "user"},
            },
        },
    }
    raw = _make_tool_response("memory", {"action": "add", "target": "user", "content": "Eric lives in Colorado"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    assert result.score == 100
    assert result.passed


def test_score_test_partial_arg_credit():
    """Correct tool name (40 pts) + partial args (30 pts) = 70."""
    test_case = {
        "id": "t5", "category": "memory", "description": "add memory partial",
        "expected": {
            "function_name": "memory",
            "arguments": {
                "required_args": ["action", "target", "content"],
            },
        },
    }
    # Only provide 2 of 3 required args
    raw = _make_tool_response("memory", {"action": "add", "target": "user"})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    # Name: 40 pts. Args: 2/3 = 0.667 * 60 = 40 pts. Total: 80.
    assert result.score == 80


def test_score_test_infra_error():
    test_case = {"id": "t6", "category": "test", "description": "x", "expected": {"function_name": "terminal"}}
    raw = {"choices": None, "error": {"message": "rate limit exceeded"}}
    result = score_test(test_case, raw)
    assert result.is_infra_error
    assert result.score == 0


def test_score_test_no_tools_called():
    test_case = {"id": "t7", "category": "terminal", "description": "must use terminal",
                 "expected": {"function_name": "terminal"}}
    raw = {"id": "x", "choices": [{"message": {"content": "I'll just tell you", "tool_calls": None}}]}
    result = score_test(test_case, raw)
    assert result.score == 0
    assert "Model did not call any tools" in result.error


def test_score_test_text_contains_pass():
    test_case = {"id": "t8", "category": "test", "description": "text check",
                 "expected": {"text_contains": "Paris"}}
    raw = {"id": "x", "choices": [{"message": {"content": "The capital of France is Paris."}}]}
    result = score_test(test_case, raw)
    assert result.score == 100
    assert result.passed


def test_score_test_text_contains_fail():
    test_case = {"id": "t9", "category": "test", "description": "text check",
                 "expected": {"text_contains": "Paris"}}
    raw = {"id": "x", "choices": [{"message": {"content": "I don't know."}}]}
    result = score_test(test_case, raw)
    assert result.score == 0
    assert not result.passed


def test_score_test_list_field_check():
    test_case = {
        "id": "t10", "category": "todo", "description": "create todos",
        "expected": {
            "function_name": "todo",
            "arguments": {"list_field_check": {"field": "todos", "min_items": 3}},
        },
    }
    raw = _make_tool_response("todo", {"todos": [{"id": "1"}, {"id": "2"}, {"id": "3"}]})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    assert result.score == 100
    assert result.passed


def test_score_test_list_field_too_short():
    test_case = {
        "id": "t11", "category": "todo", "description": "create todos",
        "expected": {
            "function_name": "todo",
            "arguments": {"list_field_check": {"field": "todos", "min_items": 3}},
        },
    }
    raw = _make_tool_response("todo", {"todos": [{"id": "1"}]})
    raw["id"] = "x"
    result = score_test(test_case, raw)
    # Name: 40 pts. Args: list too short = 0 * 60 = 0. Total: 40.
    assert result.score == 40


# --- score_debug_fixture ---

def test_debug_fixture_tool_call_scores_100():
    test_case = {
        "id": "d1", "category": "terminal", "description": "terminal test",
        "expected": {
            "function_name": "terminal",
            "arguments": {
                "required_args": ["command"],
                "arg_values": {"command": "ls /tmp"},
            },
        },
    }
    result = score_debug_fixture(test_case)
    assert result.score == 100, f"Expected 100, got {result.score}: {result.details}"


def test_debug_fixture_no_tool_calls_scores_100():
    test_case = {
        "id": "d2", "category": "test", "description": "no tool",
        "expected": {"no_tool_calls": True},
    }
    result = score_debug_fixture(test_case)
    assert result.score == 100, f"Expected 100, got {result.score}"


def test_debug_fixture_list_field_scores_100():
    test_case = {
        "id": "d3", "category": "todo", "description": "create todos",
        "expected": {
            "function_name": "todo",
            "arguments": {
                "list_field_check": {"field": "todos", "min_items": 3},
            },
        },
    }
    result = score_debug_fixture(test_case)
    assert result.score == 100, f"Expected 100, got {result.score}: {result.details}"


def test_debug_fixture_text_contains_scores_100():
    test_case = {
        "id": "d4", "category": "test", "description": "text check",
        "expected": {"text_contains": "Paris"},
    }
    result = score_debug_fixture(test_case)
    assert result.score == 100, f"Expected 100, got {result.score}"


def test_debug_fixture_arg_substring_scores_100():
    test_case = {
        "id": "d5", "category": "file", "description": "search files",
        "expected": {
            "function_name": "search_files",
            "arguments": {
                # No required_args — fixture will set pattern from arg_substring_checks
                "arg_substring_checks": {"pattern": "TODO"},
            },
        },
    }
    result = score_debug_fixture(test_case)
    assert result.score == 100, f"Expected 100, got {result.score}: {result.details}"
