"""One suppression rule for every lane that can deliver a subagent's process notifications.

``delegation.surface_child_process_notifications`` (default false) withholds a child's process
notifications from the parent conversation — the child's delegation result is the deliverable
(see website/docs/user-guide/features/delegation.md#child-background-process-notifications).
The registry drain, the gateway process watcher and the gateway watch drain must agree, so they
share :func:`child_process_notification_suppressed`.
"""

import pytest

from tools.process_registry import ProcessRegistry, child_process_notification_suppressed


def test_parent_owned_event_is_never_suppressed():
    assert child_process_notification_suppressed(
        {"type": "completion", "owner_task_id": "parent-turn-1"}) is False
    assert child_process_notification_suppressed({"type": "completion"}) is False


def test_subagent_owned_event_is_suppressed_by_default(monkeypatch):
    monkeypatch.setattr(
        ProcessRegistry, "_surface_child_process_notifications", staticmethod(lambda: False))

    assert child_process_notification_suppressed(
        {"type": "completion", "owner_task_id": "sa-0-crawler"}) is True
    # task_id is the fallback for events that carry no raw owner id.
    assert child_process_notification_suppressed(
        {"type": "completion", "task_id": "sa-1-other"}) is True
    # Watch matches from the same child follow the same rule.
    assert child_process_notification_suppressed(
        {"type": "watch_match", "owner_task_id": "sa-1-other"}) is True


def test_delegation_result_is_never_suppressed(monkeypatch):
    """The delegation result itself is the deliverable, whatever the flag says."""
    monkeypatch.setattr(
        ProcessRegistry, "_surface_child_process_notifications", staticmethod(lambda: False))

    assert child_process_notification_suppressed(
        {"type": "async_delegation", "owner_task_id": "sa-0-crawler"}) is False


def test_surface_flag_restores_delivery(monkeypatch):
    monkeypatch.setattr(
        ProcessRegistry, "_surface_child_process_notifications", staticmethod(lambda: True))

    assert child_process_notification_suppressed(
        {"type": "completion", "owner_task_id": "sa-0-crawler"}) is False


def test_caller_resolved_flag_is_reused(monkeypatch):
    """A drain that already resolved the config must not re-read it per event."""
    reads = []

    def _read() -> bool:
        reads.append(1)
        return False

    monkeypatch.setattr(ProcessRegistry, "_surface_child_process_notifications", staticmethod(_read))

    assert child_process_notification_suppressed({"owner_task_id": "sa-0-x"}, surface_child=False) is True
    assert reads == []
