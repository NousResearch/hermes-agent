"""Pure gateway tool-event emission rules."""


def should_emit_tool_complete(
    name: str,
    args: dict,
    *,
    tool_progress_enabled: bool,
    has_inline_diff: bool,
) -> bool:
    """Return whether a completion must be sent to gateway clients.

    Explicit Desktop/Webapp ``read_file`` preview requests need their arguments
    even when ordinary tool progress is disabled. No other quiet read is
    promoted to a completion event.
    """

    explicit_file_preview = name == "read_file" and args.get("preview") is True
    return tool_progress_enabled or has_inline_diff or explicit_file_preview
