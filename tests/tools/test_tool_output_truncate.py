"""Invariant: terminal, execute_code, MCP and the bounded output collector truncate through one
head/tail algorithm — 40% head / 60% tail, exactly one notice, kept text equal to the budget.
"""

import re

import pytest

from tools.tool_output_truncate import HEAD_RATIO, truncate_head_tail

_NOTICE = re.compile(r"\n\n\.\.\. \[(?P<label>[A-Z ]+) TRUNCATED - (?P<omitted>[\d,]+) (?P<unit>chars|bytes) "
                     r"omitted out of (?P<total>[\d,]+) total\] \.\.\.\n\n")

TEXT = "".join(f"{i:04d}|" for i in range(2000))  # 10,000 chars, position-addressable


def _assert_head_tail(result: str, *, budget: int, total: int, unit: str = "chars") -> dict:
    notices = list(_NOTICE.finditer(result))
    assert len(notices) == 1, result[:200]
    m = notices[0]
    head, tail = result[:m.start()], result[m.end():]
    assert head == TEXT[:int(budget * HEAD_RATIO)]
    assert tail == TEXT[-(budget - int(budget * HEAD_RATIO)):]
    assert int(m["omitted"].replace(",", "")) == total - budget
    assert int(m["total"].replace(",", "")) == total
    assert m["unit"] == unit
    return m.groupdict()


def test_terminal_result_truncation(monkeypatch):
    from tools import terminal_tool_result
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 1000)
    info = _assert_head_tail(terminal_tool_result._truncate_head_tail(TEXT), budget=1000, total=10_000)
    assert info["label"] == "OUTPUT"


def test_mcp_result_truncation():
    from tools.mcp_tool_content import _truncate_mcp_text_result
    info = _assert_head_tail(_truncate_mcp_text_result(TEXT, max_chars=1000), budget=1000, total=10_000)
    assert info["label"] == "MCP RESULT"


def test_execute_code_stdout_truncation(monkeypatch):
    from tools import code_execution_tool
    monkeypatch.setattr(code_execution_tool, "MAX_STDOUT_BYTES", 1000)
    monkeypatch.setattr(code_execution_tool, "_spill_full_stdout", lambda _text: None)
    text, meta = code_execution_tool._truncate_stdout_text(TEXT)
    info = _assert_head_tail(text, budget=1000, total=10_000, unit="bytes")
    assert info["label"] == "OUTPUT" and meta["stdout_bytes_omitted"] == 9000


def test_bounded_output_collector_truncation():
    from tools.environments.base_output import _BoundedOutputCollector
    collector = _BoundedOutputCollector(1000)
    for i in range(0, len(TEXT), 333):
        collector.append(TEXT[i:i + 333])
    rendered = collector.render()
    notices = list(_NOTICE.finditer(rendered))
    assert len(notices) == 1 and notices[0]["label"] == "OUTPUT"
    assert len(rendered) <= 1000
    head, tail = rendered[:notices[0].start()], rendered[notices[0].end():]
    assert TEXT.startswith(head) and TEXT.endswith(tail)
    assert abs(len(head) / (len(head) + len(tail)) - HEAD_RATIO) < 0.01


@pytest.mark.parametrize("size", [0, 999, 1000])
def test_text_within_budget_passes_through_untouched(size):
    assert truncate_head_tail(TEXT[:size], 1000) == TEXT[:size]
