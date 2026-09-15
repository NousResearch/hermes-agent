from __future__ import annotations

import pytest

from workstation.durable_tasks import (
    AtomicPersistenceViolation,
    DurableTaskStore,
    WorkItemStatus,
)


@pytest.fixture
def store(tmp_path):
    board_name = f"test_board_{tmp_path.name}"
    return DurableTaskStore(board=board_name)


def test_durable_task_100_items_resume_at_38(store):
    """Scenario 1: Start 100 items, complete 37, simulate restart/resume, continues at 38 without duplicates."""
    from uuid import uuid4
    task_id = f"task_ig_100_{uuid4().hex[:6]}"
    items = [{"media_id": f"id_{i:04d}", "target": f"post_{i}"} for i in range(1, 101)]

    plan = store.create_plan(task_id, "Instagram 100 Posts", items)
    assert plan.total_items == 100

    work_items = store.get_work_items(plan.id)
    assert len(work_items) == 100
    assert work_items[0].status == WorkItemStatus.PENDING

    # Complete first 37 items with full atomic pipeline
    for i in range(37):
        item = work_items[i]
        store.update_item_checkpoint(item.id, "navigate", "ok")
        store.update_item_checkpoint(item.id, "ready", "ok")
        store.mark_item_captured(item.id, f"artifact://tasks/{task_id}/raw_{item.id}.json")
        store.mark_item_persisted(item.id, f"artifact://tasks/{task_id}/norm_{item.id}.json")
        store.mark_item_validated(item.id, {"valid": True})
        completed = store.complete_item(item.id)
        assert completed.status == WorkItemStatus.COMPLETED

    # Verify progress summary at crash point
    summary = store.get_progress_summary(task_id)
    assert summary["completed"] == 37
    assert summary["total_items"] == 100

    # SIMULATE CRASH & RESTART:
    # A new store instance resumes task_id from durable SQLite
    resumed_store = DurableTaskStore(board=store.board)
    uncompleted = resumed_store.resume_plan(task_id)

    # Exactly 63 items remain
    assert len(uncompleted) == 63
    # First uncompleted item is index 38
    assert uncompleted[0].item_index == 38
    assert uncompleted[0].id == f"{plan.id}_0038"

    # All uncompleted items are distinct and strictly > 37
    indices = [item.item_index for item in uncompleted]
    assert indices == list(range(38, 101))
    assert len(set(indices)) == 63


def test_atomic_persistence_violation(store):
    """Scenario 2: Captured or running item cannot complete if output is not persisted/validated."""
    items = [{"id": 1}]
    plan = store.create_plan("task_atomic", "Atomic Test", items)
    item = store.get_work_items(plan.id)[0]

    # Mark only captured (in memory / chat, but not persisted to disk)
    store.mark_item_captured(item.id, "artifact://raw_1.json")

    # Attempting to mark complete must fail closed!
    with pytest.raises(AtomicPersistenceViolation, match="has not been persisted"):
        store.complete_item(item.id)

    # Check status remains CAPTURED, never completed
    reloaded = store.get_item(item.id)
    assert reloaded.status == WorkItemStatus.CAPTURED
    assert reloaded.status != WorkItemStatus.COMPLETED

    # Now persist, but don't validate yet
    store.mark_item_persisted(item.id, "artifact://norm_1.json")
    with pytest.raises(AtomicPersistenceViolation, match="has not been validated"):
        store.complete_item(item.id)

    # Validate, now it can complete
    store.mark_item_validated(item.id, {"valid": True})
    completed = store.complete_item(item.id)
    assert completed.status == WorkItemStatus.COMPLETED


def test_context_compaction_structured_handle_and_delta(store):
    """Scenario 8 & Delta: Context compaction receives task:// handle, delta state is compact."""
    items = [{"id": i} for i in range(1, 87)]
    task_id = "task_compact_test"
    plan = store.create_plan(task_id, "Instagram September", items)

    # Complete 82 items
    work_items = store.get_work_items(plan.id)
    for i in range(82):
        it = work_items[i]
        store.mark_item_captured(it.id, f"raw_{i}")
        store.mark_item_persisted(it.id, f"norm_{i}")
        store.mark_item_validated(it.id, {"valid": True})
        store.complete_item(it.id)

    # Mark 2 as suspect
    store.mark_item_captured(work_items[82].id, "raw_83")
    store.mark_item_persisted(work_items[82].id, "norm_83")
    store.mark_item_validated(work_items[82].id, {"valid": True, "suspect": True, "reason": "duplicate labels"})

    store.mark_item_captured(work_items[83].id, "raw_84")
    store.mark_item_persisted(work_items[83].id, "norm_84")
    store.mark_item_validated(work_items[83].id, {"valid": True, "suspect": True, "reason": "text length 18"})

    # Check structured context handle
    compact_str = store.to_compact_context(task_id)
    assert "task://task_compact_test" in compact_str
    assert "82/86 completed" in compact_str
    assert "2 suspect" in compact_str

    # Check delta state
    delta = store.get_delta_state(task_id, last_completed_count=72)
    assert delta["delta"]["completed"] == "+10"
    assert delta["delta"]["current"] == "82/86"
    assert delta["delta"]["suspect"] == 2
