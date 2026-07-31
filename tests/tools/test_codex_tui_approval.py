"""Focused tests for the Hermes-owned native Codex PTY approval bridge."""

import threading
import time

import pytest

from tools.codex_tui_approval import (
    APPROVE_ONCE_KEY,
    DENY_KEY,
    CodexTuiApprovalDetector,
    prepare_managed_codex_tui_command,
)
from tools.process_registry import ProcessRegistry, ProcessSession


def test_prepare_bridge_is_process_local_and_discord_only(monkeypatch):
    monkeypatch.setattr(
        "tools.codex_tui_approval._supported_codex_executable",
        lambda _executable: "/opt/codex/bin/codex",
    )
    session_key = "agent:main:discord:channel:123:user:456"
    rewritten, enabled = prepare_managed_codex_tui_command(
        "codex --no-alt-screen",
        session_key,
        approval_sink_available=True,
    )

    assert enabled is True
    assert rewritten.startswith("/opt/codex/bin/codex -c ")
    assert 'tui.keymap.approval.approve="ctrl-g"' in rewritten
    assert 'tui.keymap.approval.approve_for_session="ctrl-o"' in rewritten
    assert 'tui.keymap.approval.deny="ctrl-x"' in rewritten
    assert rewritten.endswith("--no-alt-screen")

    for command, key, sink in (
        ("codex", "agent:main:telegram:dm:123:user:456", True),
        ("codex exec pwd", session_key, True),
        ("codex remote-control", session_key, True),
        ("codex doctor", session_key, True),
        ("codex && echo unsafe", session_key, True),
        ("codex", session_key, False),
    ):
        untouched, enabled = prepare_managed_codex_tui_command(
            command, key, approval_sink_available=sink
        )
        assert (untouched, enabled) == (command, False)


def test_detector_waits_for_full_prompt_and_deduplicates_redraws():
    detector = CodexTuiApprovalDetector()
    assert detector.feed("\x1b[2JWould you like to run the following ") is None
    assert detector.feed("command?\r\n  $ rm build.tmp\r\n") is None

    prompt = detector.feed("\x1b[4;1H  Yes, just this once")
    assert prompt is not None
    assert prompt.kind == "command execution"
    assert "$ rm build.tmp" in prompt.command
    assert detector.feed("\x1b[4;1H  Yes, just this once") is None

    detector.mark_resolved()
    assert detector.feed("ordinary Codex output") is None


class _FakePty:
    def __init__(self):
        self.writes = []

    def write(self, value):
        self.writes.append(value)


def _wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.parametrize(
    ("choice", "expected_key"),
    (("once", APPROVE_ONCE_KEY), ("deny", DENY_KEY)),
)
def test_approve_and_deny_unblock_only_the_originating_session(choice, expected_key):
    from tools.approval import (
        has_blocking_approval,
        resolve_gateway_approval,
        unregister_gateway_notify,
    )

    registry = ProcessRegistry()
    notified = []
    registry.on_approval = lambda session, data: notified.append((session.id, data))
    session = ProcessSession(
        id="proc_codex_a",
        command="codex",
        session_key="agent:main:discord:channel:111:user:7",
    )
    session.managed_codex_tui = True
    session._codex_approval_detector = CodexTuiApprovalDetector()
    session._pty = _FakePty()

    output = (
        "Would you like to run the following command?\r\n"
        "  $ python deploy.py\r\n"
        "  Yes, just this once"
    )
    worker = threading.Thread(
        target=registry._check_codex_tui_approval,
        args=(session, output),
    )
    worker.start()
    try:
        assert _wait_for(lambda: has_blocking_approval(session.session_key))
        assert notified[0][0] == session.id
        assert session._codex_approval_pending is True

        # The launching agent turn can finish before Codex asks. Its normal
        # callback teardown must not expire this process-lifetime request.
        unregister_gateway_notify(session.session_key)
        assert has_blocking_approval(session.session_key)

        other_key = "agent:main:discord:channel:222:user:7"
        assert resolve_gateway_approval(other_key, "once") == 0
        worker.join(timeout=0.05)
        assert worker.is_alive(), "another Discord session crossed the approval boundary"
        assert session._pty.writes == []

        assert resolve_gateway_approval(session.session_key, choice) == 1
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert session._pty.writes == [expected_key.encode()]
        assert session._codex_approval_pending is False
    finally:
        resolve_gateway_approval(session.session_key, "deny")
        worker.join(timeout=2)


def test_public_stdin_cannot_race_a_pending_codex_approval():
    registry = ProcessRegistry()
    session = ProcessSession(id="proc_codex", command="codex")
    session.managed_codex_tui = True
    session._codex_approval_pending = True
    session._pty = _FakePty()
    registry._running[session.id] = session

    result = registry.write_stdin(session.id, "y")

    assert result["status"] == "approval_pending"
    assert session._pty.writes == []


def test_kill_cancels_only_that_process_approval(monkeypatch):
    from tools.approval import (
        _ApprovalEntry,
        _gateway_queues,
        has_blocking_approval,
    )

    registry = ProcessRegistry()
    pty = _FakePty()
    pty.terminate = lambda force: None
    session = ProcessSession(
        id="proc_codex",
        command="codex",
        session_key="discord-session",
    )
    session.managed_codex_tui = True
    session._pty = pty
    registry._running[session.id] = session
    own = _ApprovalEntry(
        {
            "approval_source": "managed_codex_tui",
            "approval_source_id": session.id,
        }
    )
    unrelated = _ApprovalEntry({"approval_source": "terminal_guard"})
    _gateway_queues[session.session_key] = [own, unrelated]
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)

    try:
        result = registry.kill_process(session.id)
        assert result["status"] == "killed"
        assert own.event.is_set() and own.result == "deny"
        assert not unrelated.event.is_set()
        assert has_blocking_approval(session.session_key)
    finally:
        _gateway_queues.pop(session.session_key, None)
