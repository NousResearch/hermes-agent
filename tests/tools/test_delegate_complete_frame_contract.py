"""The ``subagent.complete`` frame the child runner emits must validate against the wire contract.

``_ChildProgressRelay._relay`` refuses (logs + drops) any frame ``SubagentEventPayload`` rejects, so a
producer/contract drift would silently strand a finished child as "running". Pin the real
``complete_kwargs`` shape from ``tools/delegate_tool_child_run.py`` plus the relay's identity kwargs here
so such drift fails in CI instead of in production.
"""

from tools.delegate_tool_progress import _ChildProgressRelay
from tui_gateway.contracts.events import SubagentEventPayload


def _complete_kwargs_as_emitted_by_child_run() -> dict:
    # Mirror of ``_ChildRun.emit_complete`` (delegate_tool_child_run.py) — every key, plus the optional
    # ``failure_reason``/``cost_usd`` branches, with values of the same types the runner produces.
    return {
        "preview": "summary text"[:160],
        "status": "failed",
        "duration_seconds": 12.5,
        "summary": "summary text"[:500],
        "input_tokens": 1200,
        "output_tokens": 340,
        "reasoning_tokens": 0,
        "api_calls": 4,
        "files_read": ["/repo/a.py"],
        "files_written": ["/repo/b.py"],
        # ``_extract_output_tail`` rows: {tool, preview, is_error}
        "output_tail": [{"tool": "terminal", "preview": "ok", "is_error": False}],
        "failure_reason": "rate_limit",
        "cost_usd": 0.0123,
    }


def test_complete_frame_from_child_run_validates_against_wire_contract():
    relay = _ChildProgressRelay(
        task_index=1, goal="scan the repo", spinner=None, parent_cb=None, task_count=2,
        subagent_id="sub-1", parent_id="parent-1", depth=1, model="test-model", toolsets=["terminal"],
        session_ref={"session_id": "child-sess", "delegation_id": "dlg-1"},
    )
    relay.tool_count = 3
    identity = {**relay._identity_kwargs(), **_complete_kwargs_as_emitted_by_child_run()}
    # Same construction as ``_relay`` (preview rides as ``text``); must not raise ValidationError.
    kwargs = dict(identity)
    preview = kwargs.pop("preview")
    payload = SubagentEventPayload(**kwargs, text=preview)
    assert payload.status == "failed"
    assert payload.child_session_id == "child-sess" and payload.delegation_id == "dlg-1"
    assert payload.output_tail and payload.output_tail[0].tool == "terminal"
    assert payload.failure_reason == "rate_limit" and payload.cost_usd == 0.0123


def test_relay_delivers_complete_frame_to_parent_callback():
    seen = []

    def parent_cb(event_type, tool_name, preview, args, **kw):
        seen.append((event_type, kw.get("subagent_payload")))

    relay = _ChildProgressRelay(
        task_index=0, goal="g", spinner=None, parent_cb=parent_cb, task_count=1,
        subagent_id="s", parent_id="p", depth=1, model="m", toolsets=None, session_ref={},
    )
    kwargs = _complete_kwargs_as_emitted_by_child_run()
    preview = kwargs.pop("preview")
    relay._relay("subagent.complete", preview=preview, **kwargs)
    assert len(seen) == 1 and seen[0][0] == "subagent.complete"
    assert isinstance(seen[0][1], SubagentEventPayload) and seen[0][1].status == "failed"
