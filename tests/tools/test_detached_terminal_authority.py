"""Real background-terminal admission must use its detached owner's authority."""
from contextvars import copy_context

import pytest

from agent.chat_execution_context import (
    bind_chat_execution, bind_concurrent_tool_resource, concurrent_tool_resource_for_submit,
)
from gateway.chat_execution_recovery import ChatExecutionStore
from tools.process_registry_chat_execution import reserve_local_process, start_local_process


def _started(home):
    store = ChatExecutionStore(home)
    assert store.reserve("key", 0, 0) == 1
    assert store.bind_worker_started("key", 1, "session")
    return store


def _parent_state(store, state):
    if state != "pending":
        assert store.worker_finished("key", 1)
    if state == "superseded":
        assert store.reserve("key", 0, 0, predecessor=(1, "session")) == 2


@pytest.mark.parametrize("parent_state", ["pending", "completed", "superseded"])
def test_detached_child_can_start_local_terminal_without_settlement(tmp_path, parent_state):
    store = _started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        child = concurrent_tool_resource_for_submit("child", kind="async_delegation_thread")
        with bind_concurrent_tool_resource(child) as admitted:
            assert admitted
            _parent_state(store, parent_state)
            process = reserve_local_process()
            assert process is not None
            start_local_process(process)
            assert not store.finish_concurrent_tool_resource(
                "key", 1, process.resource_id, "session", "terminal_local_process")
    rows = {row["resource_id"]: row for row in store.inspect_resources("key")}
    assert rows[process.resource_id]["state"] == "started"
    assert store.inspect("key", 1)["effects"] == "unknown"


@pytest.mark.parametrize("parent_state", ["pending", "completed", "superseded"])
def test_finished_detached_context_cannot_start_reserved_local_process(tmp_path, parent_state):
    store = _started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        child = concurrent_tool_resource_for_submit("child", kind="async_delegation_thread")
        with bind_concurrent_tool_resource(child) as admitted:
            assert admitted
            process = reserve_local_process()
            copied = copy_context()
        _parent_state(store, parent_state)
        with pytest.raises(RuntimeError, match="refused"):
            copied.run(start_local_process, process)
    rows = {row["resource_id"]: row for row in store.inspect_resources("key")}
    assert rows[process.resource_id]["state"] == "reserved"


def test_fenced_detached_context_cannot_start_reserved_local_process(tmp_path):
    store = _started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        child = concurrent_tool_resource_for_submit("child", kind="async_delegation_thread")
        with bind_concurrent_tool_resource(child) as admitted:
            assert admitted
            process = reserve_local_process()
            assert store.begin_recovery("key", 1, "fence")
            with pytest.raises(RuntimeError, match="refused"):
                start_local_process(process)
