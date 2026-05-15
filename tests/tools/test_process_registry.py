from tools.process_registry import format_process_notification


def test_format_completion_event():
    evt = {
        "type": "completion",
        "session_id": "proc_abc",
        "command": "sleep 5",
        "exit_code": 0,
        "output": "done",
    }
    result = format_process_notification(evt)
    assert "[IMPORTANT: Background process proc_abc completed" in result
    assert "exit code 0" in result
    assert "Command: sleep 5" in result
    assert "Output:\ndone]" in result


def test_format_watch_match_event():
    evt = {
        "type": "watch_match",
        "session_id": "proc_xyz",
        "command": "tail -f log",
        "pattern": "ERROR",
        "output": "ERROR: disk full",
        "suppressed": 0,
    }
    result = format_process_notification(evt)
    assert 'watch pattern "ERROR"' in result
    assert "Matched output:\nERROR: disk full" in result


def test_format_watch_match_with_suppressed():
    evt = {
        "type": "watch_match",
        "session_id": "proc_xyz",
        "command": "tail -f log",
        "pattern": "WARN",
        "output": "WARN: low mem",
        "suppressed": 3,
    }
    result = format_process_notification(evt)
    assert "3 earlier matches were suppressed" in result


def test_format_watch_disabled_event():
    evt = {
        "type": "watch_disabled",
        "message": "Watch disabled for proc_xyz: too many matches",
    }
    result = format_process_notification(evt)
    assert "[IMPORTANT: Watch disabled for proc_xyz" in result


def test_format_returns_none_for_empty_event():
    # Default type is "completion", session_id defaults to "unknown"
    evt = {}
    result = format_process_notification(evt)
    assert result is not None  # still formats with defaults
    assert "unknown" in result
