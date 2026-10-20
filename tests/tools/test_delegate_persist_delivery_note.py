"""The background dispatch note must match the session's delivery contract (#135818).

On stateless api_server sessions that declared a server-history consumer
(``X-Hermes-Session-Id``), ``_resolve_async_wake_sid`` returns a non-empty id: the
detached completion is persisted as a transcript row WITHOUT a model wake and only
reaches the model inside the user's next turn (#85957). The dispatch note previously
promised "re-enters the conversation as a new message", so the model ended its turn
vowing an unprompted follow-up that never came.
"""

import json
from types import SimpleNamespace

from tools.delegate_tool import _build_top_level_description
from tools.delegate_tool_dispatch import _Batch, _dispatch_background, _dispatched_payload


def _batch(n_tasks=1, *, wake_sid="api-parent", history=True):
    tasks = [{"goal": f"g{i}"} for i in range(n_tasks)]
    children = [(i, t, SimpleNamespace()) for i, t in enumerate(tasks)]
    return _Batch(
        task_list=tasks, children=children, parent_agent=SimpleNamespace(session_id="parent-1"),
        creds={"model": "m"}, context="ctx", top_role=None, max_children=4,
        live_deleg_id="live-1", live_writers=[], live_paths=[],
        origin_wake_sid=wake_sid, origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, origin_session_history_delivery=history,
        overall_start=0.0,
    )


def _patch_dispatch(monkeypatch):
    monkeypatch.setattr(
        "tools.delegate_tool_dispatch._dispatch_unit",
        lambda *a, **k: {"status": "dispatched", "delegation_id": "d-1"},
    )
    monkeypatch.setattr("tools.delegate_tool_dispatch._detach_child", lambda *a, **k: None)


def test_persist_only_dispatch_note_states_next_message_delivery(monkeypatch):
    """A non-empty wake sid (persist-only surface) must not promise a self-driven new message."""
    _patch_dispatch(monkeypatch)
    # Kanban-style env makes async_delivery_supported() False, so a declared history
    # consumer routes _resolve_async_wake_sid to its persist-only branch.
    monkeypatch.setenv("HERMES_KANBAN_TASK", "1")

    result = json.loads(_dispatch_background(_batch()))

    note = result["note"]
    assert result["status"] == "dispatched"
    assert "saved to this session's transcript" in note
    assert "the user's NEXT message" in note
    assert "Do not promise an unprompted follow-up" in note
    assert "re-enters the conversation as a new message" not in note


def test_push_dispatch_note_keeps_new_message_promise(monkeypatch):
    """A wake-capable session (empty sid) keeps the original re-enter wording."""
    _patch_dispatch(monkeypatch)

    result = json.loads(_dispatch_background(_batch(wake_sid="")))

    note = result["note"]
    assert result["status"] == "dispatched"
    assert "re-enters the conversation as a new message" in note
    assert "NEXT message" not in note


def test_many_persist_note_formats_counts(monkeypatch):
    """The multi-task persist-only variant formats its counts like the push one."""
    _patch_dispatch(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "1")

    result = json.loads(_dispatch_background(_batch(n_tasks=3)))

    note = result["note"]
    assert "3 subagents" in note
    assert "1 completion unit(s)" in note
    assert "the user's NEXT message" in note
    assert "re-enter the conversation as their own new message" not in note


def test_description_states_the_stateless_api_exception():
    """The static description must not promise a re-enter unconditionally."""
    description = _build_top_level_description()
    assert "stateless API sessions" in description
    assert "only with the user's next message" in description
