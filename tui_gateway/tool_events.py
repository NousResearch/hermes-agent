"""Pure gateway tool-event emission and redaction rules."""

from typing import Any


def _is_explicit_file_preview(name: str, args: dict) -> bool:
    return name == "read_file" and args.get("preview") is True


def prepare_tool_complete_payload(
    payload: dict[str, Any],
    *,
    tool_progress_enabled: bool,
) -> dict[str, Any] | None:
    """Mark explicit preview-read success and minimize quiet routing events.

    When normal tool progress is enabled, the completion keeps its ordinary
    result so the tool row behaves as before. Quiet preview completions exist
    only to route a successfully read path into Desktop/Webapp, so they carry
    no file body. Failed or denied quiet reads are not emitted at all.
    """

    name = payload.get("name")
    args = payload.get("args")
    if not isinstance(name, str) or not isinstance(args, dict) or not _is_explicit_file_preview(name, args):
        return payload

    result = payload.get("result")
    succeeded = isinstance(result, dict) and isinstance(result.get("content"), str) and not result.get("error")

    if tool_progress_enabled:
        return {**payload, "preview_success": succeeded}

    if not succeeded:
        return None

    return {
        "tool_id": payload.get("tool_id"),
        "name": name,
        "args": args,
        "preview_success": True,
    }


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
    promoted to a completion event. The caller applies
    :func:`prepare_tool_complete_payload` so denied/failed quiet reads are
    suppressed and successful quiet events contain no file body.
    """

    return tool_progress_enabled or has_inline_diff or _is_explicit_file_preview(name, args)
