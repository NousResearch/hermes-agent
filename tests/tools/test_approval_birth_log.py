"""Birth-record logging for gateway approval prompts (#91980).

The BLOCKED outcome only lands in logs after the full approval timeout
window, and a notify write onto a dead client transport drops silently —
so a delivery-time failure used to leave no trace at all and look
identical to user silence. ``_await_gateway_decision`` now logs every
prompt raise at WARNING level (session / pattern / surface) before the
notify fires.
"""

import logging

from tools import approval as mod


def _resolve_now(_data):
    """Simulate the user answering from inside the notify callback."""
    with mod._lock:
        queue = mod._gateway_queues.get(_resolve_now.session_key, [])
        for entry in queue:
            entry.result = "once"
            entry.event.set()


def _await_with_immediate_reply(session_key, command):
    _resolve_now.session_key = session_key
    return mod._await_gateway_decision(
        session_key,
        _resolve_now,
        {
            "command": command,
            "description": "needs user consent",
            "pattern_key": "shell.dangerous",
            "pattern_keys": ["shell.dangerous"],
        },
        surface="tui_gateway",
    )


def test_prompt_raise_logs_birth_record(caplog):
    with caplog.at_level(logging.WARNING, logger="tools.approval"):
        decision = _await_with_immediate_reply("birth-log-sess", "rm -rf /tmp/x")

    assert decision["resolved"] is True
    records = [
        r for r in caplog.records if "Approval prompt raised" in r.getMessage()
    ]
    assert records, "expected a birth-record WARNING at raise time"
    line = records[0].getMessage()
    assert "session=birth-log-sess" in line
    assert "pattern=shell.dangerous" in line
    assert "surface=tui_gateway" in line


def test_birth_record_omits_raw_command(caplog):
    # The approval payload carries the RAW command (redaction happens in the
    # transport); the birth record must not echo it into logs.
    with caplog.at_level(logging.WARNING, logger="tools.approval"):
        _await_with_immediate_reply("birth-log-nocmd", "secret-token-command")

    raise_lines = [
        r for r in caplog.records if "Approval prompt raised" in r.getMessage()
    ]
    assert raise_lines
    assert "secret-token-command" not in raise_lines[0].getMessage()
