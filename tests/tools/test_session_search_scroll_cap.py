"""Scroll output is bounded: tool rows and tool_calls JSON no longer swamp the answer text.

`_scroll` shaped every message uncapped and `_shape_message` never capped ``tool_calls``. Measured
on a real 440K-message state.db: window=10 scrolls returned p90 129K chars (window=20: 34% over 100K),
51% of it tool-role output and 21% tool_calls JSON. Above the tool-result budget the whole result is
replaced by a short preview that does not contain the anchor. The anchor message is never cut below 8000 chars;
tool rows are cut at 1500 and user/assistant text at 8000; tool_calls are reduced to name + a short argument preview.
"""

import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    d.create_session("s1", source="cli")
    ids = []
    for i in range(10):
        ids.append(d.append_message("s1", role="assistant", content="x" * 50,
                                    tool_calls=[{"id": f"c{i}", "type": "function",
                                                 "function": {"name": "terminal", "arguments": json.dumps({"command": "y" * 20000})}}]))
        ids.append(d.append_message("s1", role="tool", content="z" * 30000, tool_call_id=f"c{i}", tool_name="terminal"))
    anchor = d.append_message("s1", role="assistant", content="The answer is TEAL-HERON-4417. " + "a" * 3000)
    for i in range(10):
        d.append_message("s1", role="tool", content="w" * 30000, tool_name="terminal")
    yield d, anchor
    d.close()


def test_scroll_output_is_bounded_and_keeps_the_anchor(db):
    d, anchor = db
    out = session_search(session_id="s1", around_message_id=anchor, window=20, db=d)
    payload = json.loads(out)
    assert len(out) < 60000, len(out)                      # was ~1.2M chars for this window
    anchor_row = next(m for m in payload["messages"] if m.get("anchor"))
    assert "TEAL-HERON-4417" in anchor_row["content"] and not anchor_row.get("content_truncated")
    for m in payload["messages"]:
        for call in m.get("tool_calls") or []:
            assert len(json.dumps(call)) < 600
