from tui_gateway.tool_events import prepare_tool_complete_payload, should_emit_tool_complete


def test_explicit_read_file_preview_emits_without_tool_progress_or_diff():
    assert should_emit_tool_complete(
        "read_file",
        {"path": "notes.txt", "preview": True},
        tool_progress_enabled=False,
        has_inline_diff=False,
    )


def test_ordinary_read_file_does_not_broaden_completion_emission():
    assert not should_emit_tool_complete(
        "read_file",
        {"path": "notes.txt"},
        tool_progress_enabled=False,
        has_inline_diff=False,
    )


def test_existing_progress_and_inline_diff_reasons_still_emit():
    assert should_emit_tool_complete("read_file", {}, tool_progress_enabled=True, has_inline_diff=False)
    assert should_emit_tool_complete("write_file", {}, tool_progress_enabled=False, has_inline_diff=True)


def test_quiet_successful_preview_completion_is_content_free():
    payload = {
        "args": {"path": "notes.txt", "preview": True},
        "name": "read_file",
        "result": {"content": "1|private body", "total_lines": 1},
        "tool_id": "tool-1",
    }

    assert prepare_tool_complete_payload(payload, tool_progress_enabled=False) == {
        "args": {"path": "notes.txt", "preview": True},
        "name": "read_file",
        "preview_success": True,
        "tool_id": "tool-1",
    }


def test_quiet_failed_or_denied_preview_completion_is_not_emitted():
    payload = {
        "args": {"path": "~/.hermes/auth.json", "preview": True},
        "name": "read_file",
        "result": {"error": "Access denied"},
        "tool_id": "tool-2",
    }

    assert prepare_tool_complete_payload(payload, tool_progress_enabled=False) is None


def test_progress_enabled_preview_completion_keeps_result_and_marks_failure():
    payload = {
        "args": {"path": "missing.txt", "preview": True},
        "name": "read_file",
        "result": {"error": "File not found"},
        "tool_id": "tool-3",
    }

    assert prepare_tool_complete_payload(payload, tool_progress_enabled=True) == {
        **payload,
        "preview_success": False,
    }
