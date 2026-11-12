"""Published children must be readable before their first turn, including queued children."""

import json
import sqlite3
import threading
from functools import partial
from types import SimpleNamespace

import pytest

from tests.gateway.test_stored_history_mount import _connect
from agent.prompt_refresh import stamp_prompt_epoch
from agent.turn_facade_lease import admit_durable_turn_lease
from hermes_agent.tools.delegate_tool import _build_child_agent


def _parent(db, session_id, on_progress):
    return SimpleNamespace(
        _session_db=db,
        session_id=session_id,
        provider="custom",
        base_url="http://127.0.0.1:9/v1",
        api_key="synthetic-unused-key",
        model="synthetic-model",
        api_mode="chat_completions",
        platform="cli",
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        thinking_callback=None,
        request_overrides={},
        reasoning_config=None,
        prefill_messages=None,
        enabled_toolsets=["file"],
        disabled_toolsets=[],
        tool_progress_callback=on_progress,
    )


@pytest.fixture
def offline_children(monkeypatch):
    # Only metadata discovery is stubbed. Construction, storage and session.resume are real.
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *a, **kw: 128000
    )
    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length", lambda *a, **kw: 128000
    )

    def no_model_call(*args, **kwargs):
        pytest.fail("A transcript-readiness test must not call a model")

    monkeypatch.setattr(
        "openai.resources.chat.completions.Completions.create", no_model_call
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("child_count", [1, 2])
async def test_published_children_open_before_any_child_starts(
    tmp_path, monkeypatch, offline_children, child_count
):
    connection, parent_ref = await _connect(tmp_path, monkeypatch, "sub-1")
    published = []

    def on_progress(event, *args, **payload):
        if event == "subagent.spawn_requested":
            sid = payload["child_session_id"]
            # Check at publication, not just after the constructor returns.
            published.append((sid, connection.authority.db.get_session(sid)))

    parent = _parent(connection.authority.db, parent_ref.session_id, on_progress)
    children = []
    try:
        # All batch children are built before the executor starts them. The second child
        # must be readable even when the first one occupies the only execution slot.
        for index in range(child_count):
            children.append(
                _build_child_agent(
                    task_index=index,
                    goal="Synthetic inspection",
                    context=None,
                    toolsets=["file"],
                    model=None,
                    max_iterations=2,
                    task_count=child_count,
                    parent_agent=parent,
                )
            )
        assert len(published) == child_count
        for child, (sid, row) in zip(children, published, strict=True):
            assert row is not None
            assert row["source"] == "subagent"
            assert row["parent_session_id"] == parent_ref.session_id
            # Only the lineage marker: the rest of the constructor config lands on the first turn.
            assert json.loads(row["model_config"]) == {"_delegate_from": parent_ref.session_id}
            assert not child._session_db_created
            reply = await connection.dispatch(
                {
                    "id": 10,
                    "method": "session.resume",
                    "params": {
                        "session_id": sid,
                        "source": "desktop",
                        "cols": 96,
                        "omit_messages": False,
                        "defer_history": False,
                    },
                }
            )
            assert "result" in reply, reply
            assert reply["result"]["messages"] == []

            # Real turn admission must not mistake a reserved identity for completed
            # initialization and suppress the first-turn metadata write.
            admission = admit_durable_turn_lease(
                child,
                session_id=sid,
                relay_turn_id="synthetic-turn",
                task_context={"session_id": sid, "platform": "subagent"},
                conversation_history=None,
            )
            assert admission.lease is not None
            try:
                assert not child._session_db_created
                child._session_init_model_config["synthetic_startup_setting"] = "ready"
                child._cached_system_prompt = "Synthetic completed prompt"
                # The real first turn stamps the prompt-refresh epoch (any profile that ran
                # refresh-prompts) BEFORE the constructor create; that early patch must not
                # freeze the reserved row's config and drop the lineage marker.
                stamp_prompt_epoch(child, 17)
                child._ensure_db_session()
                ready = connection.authority.db.get_session(sid)
                config = json.loads(ready["model_config"])
                assert config["_delegate_from"] == parent_ref.session_id
                assert config["synthetic_startup_setting"] == "ready"
                assert config["max_iterations"] == 2
                assert config["prompt_refresh_epoch"] == 17
                # Without the marker the parent's resume followed the child as a continuation,
                # so opening the parent showed the subagent transcript.
                db = connection.authority.db
                db.append_message(parent_ref.session_id, "user", "parent turn")
                db.append_message(sid, "assistant", "child report")
                assert db.resolve_resume_session_id(parent_ref.session_id) == parent_ref.session_id
                assert ready["system_prompt"] == "Synthetic completed prompt"
                assert ready["model"] == "synthetic-model"
                assert child._session_db_created
            finally:
                admission.lease.release()
    finally:
        for child in children:
            child.close()


@pytest.mark.asyncio
async def test_failed_reservation_closes_unpublished_child_and_a_later_build_can_retry(
    tmp_path,
    monkeypatch,
    offline_children,
):
    from hermes_agent.hermes_state import SessionDB
    from hermes_agent.hermes_state_registry import stats

    connection, parent_ref = await _connect(tmp_path, monkeypatch, "sub-1")
    published = []
    parent = _parent(
        connection.authority.db,
        parent_ref.session_id,
        lambda *a, **kw: published.append(kw),
    )
    create_session = SessionDB.create_session
    failed = []

    def fail_once(db, session_id, source, **kwargs):
        if source == "subagent" and not failed:
            failed.append(db)
            raise sqlite3.OperationalError("synthetic reservation failure")
        return create_session(db, session_id, source, **kwargs)

    monkeypatch.setattr(SessionDB, "create_session", fail_once)
    references_before = stats()["total_refcounts"]
    build = partial(
        _build_child_agent,
        task_index=0,
        goal="Synthetic inspection",
        context=None,
        toolsets=["file"],
        model=None,
        max_iterations=2,
        task_count=1,
        parent_agent=parent,
    )
    with pytest.raises(sqlite3.OperationalError, match="synthetic reservation failure"):
        build()
    assert not published
    assert not parent._active_children
    assert stats()["total_refcounts"] == references_before
    assert connection.authority.db.get_session(parent_ref.session_id) is not None

    child = build()
    try:
        assert connection.authority.db.get_session(child.session_id) is not None
        assert published
    finally:
        child.close()
