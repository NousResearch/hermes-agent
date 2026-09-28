"""Lossless JSON compaction and real profile/config resolution."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.tool_result_compaction import _compact_json, compact_tool_result


@pytest.fixture
def payload():
    return json.dumps({"records": [
        {"id": i, "title": "会议 notes", "body": "Keep  two spaces\n  and indentation.",
         "available": True, "missing": None}
        for i in range(30)
    ]}, ensure_ascii=False, indent=2)


@pytest.fixture
def configure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def write(policy):
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"tool_output": {"json_compaction": policy}}), encoding="utf-8",
        )
    return write


def test_default_is_byte_identical(configure, payload):
    configure({})
    assert compact_tool_result(payload, "mcp_records") is payload


def test_compact_keeps_all_values_and_order(configure, payload):
    configure({"mode": "compact"})
    result = compact_tool_result(payload, "mcp_records")
    assert len(result) < len(payload) * 0.9
    assert json.loads(result) == json.loads(payload)
    assert result == json.dumps(json.loads(payload), ensure_ascii=False, separators=(",", ":"))
    assert compact_tool_result(result, "mcp_records") == result


def test_observe_reports_sizes_without_content(configure, payload, caplog):
    configure({"mode": "observe"})
    with caplog.at_level(logging.INFO, logger="tools.tool_result_compaction"):
        result = compact_tool_result(payload, "mcp_records")
    assert result is payload
    assert "original_chars=" in caplog.text and "compact_chars=" in caplog.text
    assert "会议" not in caplog.text and "Keep  two spaces" not in caplog.text


@pytest.mark.parametrize("policy", [
    None, [], "compact", {"mode": "unknown"}, {"mode": False},
    {"mode": "compact", "min_chars": 0},
    {"mode": "compact", "min_chars": True},
    {"mode": "compact", "min_chars": "1024"},
    {"mode": "compact", "min_savings_ratio": 0},
    {"mode": "compact", "min_savings_ratio": 1},
    {"mode": "compact", "min_savings_ratio": float("nan")},
    {"mode": "compact", "min_savings_ratio": float("inf")},
    {"mode": "compact", "min_savings_ratio": 10 ** 400},
    {"mode": "compact", "min_savings_ratio": True},
    {"mode": "compact", "exclude_tools": "mcp_*"},
    {"mode": "compact", "exclude_tools": [None]},
])
def test_invalid_policy_fails_open(configure, payload, policy):
    configure(policy)
    assert compact_tool_result(payload, "mcp_records") is payload


def test_minimum_and_savings_gates(configure, payload):
    configure({"mode": "compact", "min_chars": len(payload) + 1})
    assert compact_tool_result(payload, "mcp_records") is payload
    configure({"mode": "compact", "min_savings_ratio": 0.99})
    assert compact_tool_result(payload, "mcp_records") is payload


def test_exact_and_glob_exclusions(configure, payload):
    configure({"mode": "compact"})
    assert compact_tool_result(payload, "read_file") is payload
    configure({"mode": "compact", "exclude_tools": ["mcp_verbatim_*"]})
    assert compact_tool_result(payload, "mcp_verbatim_calendar") is payload
    assert len(compact_tool_result(payload, "mcp_records")) < len(payload)


@pytest.mark.parametrize("value", [
    "plain text " * 200, '{ "truncated": [ ' * 100,
    '{ "not_finite": NaN, "padding": "' + "x" * 1024 + '" }',
    '{ "inf": Infinity, "padding": "' + "x" * 1024 + '" }',
    '{ "a": 1 } trailing text' + " " * 1024,
    '{ "a": 1\u00a0}' + " " * 1024,
    "[" * 2000 + "0" + "]" * 2000,
    '{ "a": 1 }', '{ "a": 1 }' + " " * 1_000_000,
    {"_multimodal": True, "text": "image", "images": []}, None,
])
def test_ineligible_payload_is_unchanged(configure, value):
    configure({"mode": "compact"})
    assert compact_tool_result(value, "mcp_records") is value


def test_lexical_fidelity_not_just_parsed_equivalence():
    number = "9" * 5000
    literal = r'"line\n  indent \\ \"quote\" \u4f60\/\t"'
    original = '{\r\n "dup" : -0, "dup" : 1.23000000000000000001e+003, "s" : ' + literal + ', "big" : ' + number + ' }'
    expected = '{"dup":-0,"dup":1.23000000000000000001e+003,"s":' + literal + ',"big":' + number + '}'
    assert _compact_json(original) == expected


def test_profile_policies_do_not_leak_between_threads(tmp_path, payload):
    profiles = []
    for mode in ("off", "compact"):
        home = tmp_path / mode
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump({"tool_output": {"json_compaction": {"mode": mode}}}),
            encoding="utf-8",
        )
        profiles.append(home)

    def run(home):
        token = set_hermes_home_override(home)
        try:
            return compact_tool_result(payload, "mcp_records")
        finally:
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        disabled, enabled = list(pool.map(run, profiles))
    assert disabled is payload
    assert len(enabled) < len(payload)


def test_cli_config_set_reaches_runtime(configure, payload):
    from hermes_cli.config import set_config_value

    configure({})
    assert compact_tool_result(payload, "mcp_records") is payload
    set_config_value("tool_output.json_compaction.mode", "compact")
    assert len(compact_tool_result(payload, "mcp_records")) < len(payload)
