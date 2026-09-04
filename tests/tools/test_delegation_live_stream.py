"""Tests for tools/delegation_live_stream.py — transcript parser + filter."""

import pytest

from tools.delegation_live_stream import (
    filter_events,
    parse_event,
    parse_lines,
    summarise,
)


# Sample transcript lines (mirrors the format written by tools/delegation_live_log.py).
SAMPLE_LINES = [
    "14:22:08 start    | child goal: verify CLI flag",
    "14:22:08 user     | kickoff: verify --json flag",
    "14:22:09 assistant| planning to run a few probes",
    "14:22:10 think    | considering timeout defaults",
    "14:22:10 tool     | -> terminal({\"command\": \"cli --json\"})",
    "14:22:11 result   | terminal ok 0.42s: {\"ok\": true, \"returncode\": 0}",
    "14:22:12 tool     | -> web_extract({\"urls\": [\"https://example.com\"]})",
    "14:22:13 result   | web_extract ERROR 1.20s: 404 Not Found",
    "14:22:14 final    | child complete",
    "unparseable line that the parser must not crash on",
    "",
]


# ----------------------------- parse_event -----------------------------


def test_parse_event_tool_start_extracts_name_and_args():
    ev = parse_event(0, "14:22:10 tool     | -> terminal({\"command\": \"ls\"})")
    assert ev.kind == "tool_start"
    assert ev.ts == "14:22:10"
    assert ev.tool_name == "terminal"
    assert ev.tool_args is not None and "command" in ev.tool_args
    assert ev.is_error is False


def test_parse_event_tool_result_success_extracts_status_and_duration():
    ev = parse_event(0, "14:22:11 result   | terminal ok 0.42s: done")
    assert ev.kind == "tool_result"
    assert ev.tool_name == "terminal"
    assert ev.tool_status == "ok"
    assert ev.tool_duration_seconds == 0.42
    assert ev.is_error is False
    assert ev.tool_result_preview == "done"


def test_parse_event_tool_result_error_sets_is_error_true():
    ev = parse_event(0, "14:22:13 result   | web_extract ERROR 1.20s: 404")
    assert ev.kind == "tool_result"
    assert ev.tool_name == "web_extract"
    assert ev.tool_status == "ERROR"
    assert ev.is_error is True
    assert ev.tool_duration_seconds == 1.20


def test_parse_event_assistant_and_thinking_become_typed_kinds():
    a = parse_event(0, "14:22:09 assistant| hi there")
    t = parse_event(0, "14:22:10 think    | reasoning about it")
    assert a.kind == "assistant"
    assert t.kind == "thinking"


def test_parse_event_user_becomes_kickoff():
    u = parse_event(0, "14:22:08 user     | kickoff: do thing")
    assert u.kind == "kickoff"
    assert u.text == "kickoff: do thing"


def test_parse_event_unparseable_line_returns_raw_kind_no_crash():
    ev = parse_event(0, "## header line ##")
    assert ev.kind == "raw"
    assert ev.ts is None
    assert ev.role == ""


def test_parse_event_index_is_preserved():
    ev = parse_event(7, "14:22:08 start | x")
    assert ev.index == 7


# ----------------------------- parse_lines -----------------------------


def test_parse_lines_returns_one_event_per_input_line():
    events = parse_lines(SAMPLE_LINES)
    assert len(events) == len(SAMPLE_LINES)
    assert [e.kind for e in events] == [
        "marker",      # start
        "kickoff",     # user
        "assistant",
        "thinking",
        "tool_start",  # terminal
        "tool_result", # terminal ok
        "tool_start",  # web_extract
        "tool_result", # web_extract ERROR
        "marker",      # final
        "raw",         # unparseable
        "raw",         # blank
    ]


def test_parse_lines_preserves_indexes_in_order():
    events = parse_lines(SAMPLE_LINES)
    assert [e.index for e in events] == list(range(len(SAMPLE_LINES)))


# ----------------------------- filter_events ---------------------------


def test_filter_events_by_kinds_returns_only_matching():
    events = parse_lines(SAMPLE_LINES)
    tool_events = filter_events(events, kinds=["tool_start", "tool_result"])
    assert all(e.kind in {"tool_start", "tool_result"} for e in tool_events)
    assert len(tool_events) == 4


def test_filter_events_by_tool_name():
    events = parse_lines(SAMPLE_LINES)
    only_terminal = filter_events(events, tool_name="terminal")
    names = [e.tool_name for e in only_terminal]
    assert names == ["terminal", "terminal"]


def test_filter_events_errors_only():
    events = parse_lines(SAMPLE_LINES)
    errors = filter_events(events, errors_only=True)
    assert len(errors) == 1
    assert errors[0].is_error is True
    assert errors[0].tool_name == "web_extract"


def test_filter_events_combined_kinds_and_errors_only():
    events = parse_lines(SAMPLE_LINES)
    # Combine kinds + errors_only — both must apply.
    out = filter_events(events, kinds=["tool_result"], errors_only=True)
    assert len(out) == 1
    assert out[0].is_error is True


def test_filter_events_no_filters_returns_all():
    events = parse_lines(SAMPLE_LINES)
    assert len(filter_events(events)) == len(events)


# ----------------------------- summarise -------------------------------


def test_summarise_counts_tool_calls_and_errors():
    events = parse_lines(SAMPLE_LINES)
    s = summarise(events)
    assert s["event_count"] == len(events)
    assert s["tool_call_count"] == 2
    assert s["tool_error_count"] == 1
    assert s["tool_call_breakdown"] == {"terminal": 1, "web_extract": 1}
    assert s["tool_error_breakdown"] == {"web_extract": 1}
    assert s["total_tool_duration_seconds"] == pytest.approx(1.62, abs=1e-3)


def test_summarise_handles_empty_event_list():
    s = summarise([])
    assert s["event_count"] == 0
    assert s["tool_call_count"] == 0
    assert s["tool_error_count"] == 0
    assert s["tool_call_breakdown"] == {}
    assert s["tool_error_breakdown"] == {}
    assert s["total_tool_duration_seconds"] == 0


def test_summarise_ignores_events_without_tool_name():
    # No tool_start/tool_result events → zero tool activity.
    s = summarise(parse_lines(["14:22:09 assistant| hi"]))
    assert s["tool_call_count"] == 0
    assert s["total_tool_duration_seconds"] == 0
