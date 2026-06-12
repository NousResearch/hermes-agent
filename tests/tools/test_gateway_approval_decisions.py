import tools.approval as approval


def setup_function():
    with approval._lock:
        approval._gateway_queues.clear()


def teardown_function():
    with approval._lock:
        approval._gateway_queues.clear()


def test_resolve_gateway_approval_by_id_resolves_only_matching_live_request():
    first = approval._ApprovalEntry({"approval_id": "approval-one", "command": "rm -rf /tmp/a"})
    second = approval._ApprovalEntry({"approval_id": "approval-two", "command": "rm -rf /tmp/b"})
    with approval._lock:
        approval._gateway_queues["session-a"] = [first]
        approval._gateway_queues["session-b"] = [second]

    count = approval.resolve_gateway_approval_by_id("approval-two", "once")

    assert count == 1
    assert first.result is None
    assert not first.event.is_set()
    assert second.result == "once"
    assert second.event.is_set()
    with approval._lock:
        assert approval._gateway_queues == {"session-a": [first]}


def test_resolve_gateway_approval_by_id_rejects_replay_after_resolution():
    entry = approval._ApprovalEntry({"approval_id": "approval-one", "command": "rm -rf /tmp/a"})
    with approval._lock:
        approval._gateway_queues["session-a"] = [entry]

    assert approval.resolve_gateway_approval_by_id("approval-one", "deny") == 1
    assert approval.resolve_gateway_approval_by_id("approval-one", "deny") == 0
    assert approval.resolve_gateway_approval_by_id("missing", "once") == 0
