"""Tests for the file-backed CLI/Gateway approval broker.

The existing gateway approval queue is process-local.  CLI/TUI approvals live in
another process, so Telegram /approve cannot see them unless the CLI publishes a
pending request through a shared broker under HERMES_HOME.
"""

import importlib
import os
import stat
import threading
import time


def test_cli_approval_request_persists_across_imports():
    """A CLI approval request is visible to a fresh importer/process."""
    from hermes_constants import get_hermes_home
    from tools import shared_approval_broker as broker

    request_id = broker.register_cli_approval(
        {
            "session_key": "cli-session-1",
            "session_id": "20260512_151213_8a0a60",
            "title": "업무 환경 상황 파악",
            "cwd": "/tmp/hermes-work",
            "pid": 12345,
            "command": "rm -rf /tmp/demo",
            "description": "recursive delete",
        },
        ttl_seconds=600,
    )

    approval_file = get_hermes_home() / "approvals" / "cli" / f"{request_id}.json"
    assert approval_file.exists()

    # Simulate the gateway importing the broker in another process after the CLI
    # has already registered the request.  The pending request must be read from
    # disk, not from module globals.
    fresh_broker = importlib.reload(broker)
    pending = fresh_broker.list_pending_cli_approvals()

    assert [item["request_id"] for item in pending] == [request_id]
    assert pending[0]["session_key"] == "cli-session-1"
    assert pending[0]["command"] == "rm -rf /tmp/demo"
    assert pending[0]["status"] == "pending"

    if os.name == "posix":
        assert stat.S_IMODE(approval_file.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(approval_file.stat().st_mode) == 0o600


def test_gateway_resolution_is_scoped_to_notified_target():
    """A /approve from a different platform/chat must not resolve CLI prompts."""
    from tools import shared_approval_broker as broker

    broker.register_cli_approval(
        {
            "session_key": "cli-session-scoped",
            "session_id": "sid-scoped",
            "title": "scoped approval",
            "cwd": "/tmp",
            "pid": os.getpid(),
            "command": "rm -rf /tmp/scoped",
            "description": "delete scoped dir",
            "notify_target": "telegram:chat-1",
        },
        ttl_seconds=600,
    )

    assert broker.resolve_oldest_cli_approval(
        "once",
        source={"platform": "discord", "chat_id": "chat-1"},
    ) == 0
    assert broker.resolve_oldest_cli_approval(
        "once",
        source={"platform": "telegram", "chat_id": "chat-2"},
    ) == 0
    assert broker.resolve_oldest_cli_approval(
        "once",
        source={"platform": "telegram", "chat_id": "chat-1"},
    ) == 1


def test_platform_only_target_is_not_remotely_resolvable():
    """Bare platform notify targets are not specific enough for /approve."""
    from tools import shared_approval_broker as broker

    broker.register_cli_approval(
        {
            "session_key": "cli-session-platform-wide",
            "session_id": "sid-platform-wide",
            "title": "platform-wide approval",
            "cwd": "/tmp",
            "pid": os.getpid(),
            "command": "rm -rf /tmp/platform-wide",
            "description": "delete platform-wide dir",
            "notify_target": "telegram",
        },
        ttl_seconds=600,
    )

    assert broker.resolve_oldest_cli_approval(
        "once",
        source={"platform": "telegram", "chat_id": "chat-1"},
    ) == 0


def test_gateway_resolve_oldest_unblocks_cli_waiter():
    """Gateway-side resolution is observed by the CLI polling waiter."""
    from tools import shared_approval_broker as broker

    request_id = broker.register_cli_approval(
        {
            "session_key": "cli-session-2",
            "session_id": "sid-2",
            "title": "approval test",
            "cwd": "/tmp",
            "pid": os.getpid(),
            "command": "git reset --hard",
            "description": "reset working tree",
            "notify_target": "telegram:chat-1",
        },
        ttl_seconds=600,
    )

    result_box = {}

    def waiter():
        result_box["choice"] = broker.wait_for_cli_approval(
            request_id,
            timeout_seconds=2,
            poll_interval=0.01,
        )

    thread = threading.Thread(target=waiter)
    thread.start()
    try:
        time.sleep(0.05)
        resolved = broker.resolve_oldest_cli_approval(
            "session",
            source={"platform": "telegram", "chat_id": "chat-1"},
        )
        assert resolved == 1
        thread.join(2)
    finally:
        if thread.is_alive():
            broker.resolve_cli_approval(request_id, "deny")
            thread.join(2)

    assert not thread.is_alive()
    assert result_box["choice"] == "session"
    assert broker.get_cli_approval(request_id)["status"] == "resolved"
    assert broker.get_cli_approval(request_id)["choice"] == "session"


def test_expired_cli_approval_is_not_resolved():
    """Gateway /approve must not approve stale CLI prompts."""
    from tools import shared_approval_broker as broker

    request_id = broker.register_cli_approval(
        {
            "session_key": "cli-session-3",
            "session_id": "sid-3",
            "title": "expired approval",
            "cwd": "/tmp",
            "pid": os.getpid(),
            "command": "rm -rf /tmp/old",
            "description": "delete old dir",
            "notify_target": "telegram:chat-1",
        },
        ttl_seconds=0.01,
    )

    time.sleep(0.05)

    assert broker.resolve_oldest_cli_approval(
        "once",
        source={"platform": "telegram", "chat_id": "chat-1"},
    ) == 0
    assert broker.wait_for_cli_approval(
        request_id,
        timeout_seconds=0,
        poll_interval=0.01,
    ) is None
    assert broker.list_pending_cli_approvals() == []
