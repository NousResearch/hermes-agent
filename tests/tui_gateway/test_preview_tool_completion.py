from tui_gateway.tool_events import should_emit_tool_complete


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
