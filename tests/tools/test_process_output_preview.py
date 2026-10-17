"""Plain-text background process previews, including clipped escapes (#135062)."""

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.mark.parametrize("query,limit", [("list", 200), ("poll", 1000)])
@pytest.mark.parametrize("text", ["", "plain output", "x" * 1100])
def test_process_preview_keeps_plain_text(query, limit, text):
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_preview", command="echo output", output_buffer=text)
    registry._running[session.id] = session

    result = registry.list_sessions()[0] if query == "list" else registry.poll(session.id)

    assert result["output_preview"] == text[-limit:]
    assert session.output_buffer == text


@pytest.mark.parametrize("query,limit", [("list", 200), ("poll", 1000)])
@pytest.mark.parametrize("escape", ["\x1b[32m\x1b[4m", "\x1b]0;window title\x07"])
def test_process_preview_strips_escapes_before_clipping(query, limit, escape):
    registry = ProcessRegistry()
    visible = "a" * limit + "done"
    # the raw tail starts inside the escape sequence
    raw = "a" * limit + escape + "done" + "\x1b[0m" + "b" * (limit - 10)
    clean = visible + "b" * (limit - 10)
    session = ProcessSession(id="proc_preview", command="echo output", output_buffer=raw)
    registry._running[session.id] = session

    result = registry.list_sessions()[0] if query == "list" else registry.poll(session.id)

    assert result["output_preview"] == clean[-limit:]
    assert session.output_buffer == raw
