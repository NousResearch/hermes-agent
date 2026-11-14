"""``hermes_tools.search_files`` must hand programmatic callers the structured ``matches`` array.

``SearchResult.to_dict(densify=True)`` swaps the ``matches`` array for a path-grouped
``matches_text`` block once a search has ``_DENSIFY_MIN_MATCHES`` (5) or more hits. That trade
is for the MODEL, which reads prose: a caller inside ``execute_code`` reads fields, and the
helper's own docstring promises ``matches`` — so ``payload.get("matches", [])`` answered an
empty list exactly when there were five or more hits.

The codebase already marks "this call came from code, not the model"
(``file_tools_read_tracking.programmatic_file_read``, set by both kernel lanes: the execute_code
RPC dispatch and the session-kernel dispatch). The search path was the one place that marker was
not consulted; these tests pin that it now is, on both sides of the boundary.
"""

import json
import time

import pytest

from tools.file_tools import search_tool
from tools.file_tools_read_tracking import programmatic_file_read


NEEDLE = "kernel_reads_fields_not_prose"


def _seed(root, count=6):
    """*count* separate files, one matching line each: enough hits to trip densification."""
    for index in range(count):
        (root / f"hit_{index}.py").write_text(f"# {NEEDLE} {index}\n", encoding="utf-8")
    return sorted(root.glob("hit_*.py"))


def test_programmatic_search_returns_the_matches_array(tmp_path):
    _seed(tmp_path)
    with programmatic_file_read():
        payload = json.loads(search_tool(NEEDLE, path=str(tmp_path), task_id="prog-array"))
    assert "matches" in payload, payload
    assert len(payload["matches"]) == 6, payload["matches"]
    assert "matches_text" not in payload, "code callers must never receive the model rendering"
    assert payload["matches"][0].keys() >= {"path", "line", "content"}


def test_model_search_keeps_the_densified_rendering(tmp_path):
    """The token saving still applies to the model's own tool call — no marker, no change."""
    _seed(tmp_path)
    payload = json.loads(search_tool(NEEDLE, path=str(tmp_path), task_id="model-text"))
    assert "matches_text" in payload, payload
    assert "matches" not in payload, "the model path must stay on the densified shape"


def test_execute_code_rpc_dispatch_returns_matches(tmp_path):
    """End to end over the RPC lane the issue reported: no marker is patched in by hand."""
    from tools.code_execution_rpc import _default_dispatch, _handle_rpc_request

    _seed(tmp_path)
    raw = _handle_rpc_request(
        {"tool": "search_files", "args": {"pattern": NEEDLE, "path": str(tmp_path)}, "token": ""},
        allowed_tools=frozenset({"search_files"}), tool_call_counter=[0], max_tool_calls=5,
        dispatch=_default_dispatch("prog-rpc"), tool_call_log=[], call_start=time.monotonic(),
        where="test",
    )
    payload = json.loads(raw)
    assert "matches" in payload, raw
    assert len(payload["matches"]) == 6, payload["matches"]
    assert payload["total_count"] == 6


def test_marker_is_reset_after_the_kernel_call(tmp_path):
    """The marker is a boundary, not a switch: the model's next search is densified again."""
    _seed(tmp_path)
    with programmatic_file_read():
        json.loads(search_tool(NEEDLE, path=str(tmp_path), task_id="prog-scoped"))
    payload = json.loads(search_tool(NEEDLE, path=str(tmp_path), task_id="after-scope"))
    assert "matches_text" in payload, payload


@pytest.mark.parametrize("count", [1, 4])
def test_below_the_threshold_both_callers_agree(tmp_path, count):
    """Under 5 hits nothing was ever densified, so both lanes already returned ``matches``."""
    _seed(tmp_path, count)
    with programmatic_file_read():
        programmed = json.loads(search_tool(NEEDLE, path=str(tmp_path), task_id=f"small-{count}"))
    payload = json.loads(search_tool(NEEDLE, path=str(tmp_path), task_id=f"model-{count}"))
    assert len(programmed["matches"]) == count == len(payload["matches"])
