"""tools/handoff_tool.py — write-only session handoff document preparation.

Writes a handoff document to disk so a human can start a fresh session and
manually paste/reference it. This tool is deliberately write-only: it does
NOT reset, restart, or trigger `/new`, and it does NOT inject its content
into any future turn or session. Continuation is always a manual, human-
initiated step.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.file_tools_write_guards import (
    _check_approval_required_write,
    _check_binary_document_write,
    _check_cross_profile_path,
    _check_protected_instruction_write,
    _check_sensitive_path,
)
from tools.registry import registry, tool_error, tool_result

# Hard cap on handoff document size. Generous enough for any realistic
# session-state document while bounding disk-fill risk from a runaway write.
MAX_HANDOFF_CONTENT_LENGTH = 200_000

HANDOFF_SCHEMA = {
    "name": "handoff",
    "description": (
        "Write a handoff document to disk describing the current session's state "
        "so a human can manually continue the work in a fresh session. This tool "
        "only writes a file — it does not reset the session and does not inject "
        "anything into a future turn. After writing, tell the user to run /new "
        "and reference (or paste) the handoff file themselves."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write"],
                "description": "The only supported action: write the handoff document to disk.",
            },
            "content": {
                "type": "string",
                "maxLength": MAX_HANDOFF_CONTENT_LENGTH,
                "description": "The handoff document content (markdown) to write to disk.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Optional destination. An absolute path is used as-is. A bare "
                    "filename (or relative path) is written under the handoffs/ "
                    "directory inside the Hermes home and may not use '..' to "
                    "escape that directory. If omitted, a UTC-timestamped filename "
                    "is used."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "If true, allow overwriting a file that already exists at the "
                    "resolved path. Defaults to false, in which case a write to an "
                    "existing path is refused."
                ),
            },
        },
        "required": ["action", "content"],
    },
}


class HandoffPathError(ValueError):
    """Raised when a relative handoff path would escape the handoffs directory."""


def _resolve_handoff_path(path: str | None) -> Path:
    """Resolve the target path for a handoff document.

    - Absolute ``path`` is used as-is (explicit, intentional escape hatch) —
      the caller (``_handoff_write``) is responsible for running the same
      write-guard stack ``write_file`` applies before this path is written to,
      since an absolute path is not confined to the handoffs directory.
    - Relative/bare ``path`` is resolved under ``get_hermes_home()/handoffs/``
      and rejected via ``HandoffPathError`` if ``..`` (or a symlink) would let
      it resolve outside that directory.
    - No ``path`` defaults to a UTC-timestamped filename under the same directory.

    ``Path.resolve()`` can raise ``ValueError`` (embedded null byte) or
    ``RuntimeError`` (symlink loop / ELOOP) for a hostile or malformed input;
    both are caught and converted to ``HandoffPathError`` so callers get the
    module's own error contract instead of an unhandled exception (the outer
    ``registry.dispatch()`` catch-all already contained this, but converting
    it here keeps the error message and behavior consistent with the rest of
    this module).
    """
    handoffs_dir = get_hermes_home() / "handoffs"
    if path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        try:
            resolved_root = handoffs_dir.resolve()
            resolved = (handoffs_dir / candidate).resolve()
        except (ValueError, OSError, RuntimeError) as exc:
            raise HandoffPathError(f"path '{path}' could not be resolved: {exc}") from exc
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise HandoffPathError(
                f"path '{path}' resolves outside the handoffs directory; "
                "remove '..' segments or use an absolute path."
            )
        return resolved
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return handoffs_dir / f"{timestamp}-handoff.md"


def _handoff_write_guard_error(target: Path, task_id: str) -> str | None:
    """Run the same write-guard stack ``write_file`` applies, against the resolved
    handoff target. Relative paths are already confined under ``handoffs/`` inside
    the Hermes home (exempt from the protected-instruction gate by design — see
    ``_hermes_exempt_homes`` in ``tools/file_tools_write_guards.py``), so this is a
    no-op for them; it is the load-bearing check for the absolute-path escape hatch,
    which is otherwise unconstrained. Order matches ``write_file_tool``: sensitive-path
    hard-deny, binary-document corruption guard, protected-instruction ALWAYS-ask gate,
    approval-required gate, then the cross-profile sandbox-mirror warning.
    """
    target_str = str(target)
    return (
        _check_sensitive_path(target_str, task_id)
        or _check_binary_document_write(target_str, task_id)
        or _check_protected_instruction_write([target_str], task_id)
        or _check_approval_required_write([target_str], task_id)
        or _check_cross_profile_path(target_str, task_id)
    )


def _handoff_write(content: str, path: str | None = None, overwrite: bool = False,
                    task_id: str = "default") -> str:
    """Write a handoff document to disk. Returns a JSON tool_result/tool_error string."""
    if not content or not content.strip():
        return tool_error("content is required and cannot be empty.")
    if len(content) > MAX_HANDOFF_CONTENT_LENGTH:
        return tool_error(
            f"content is too large ({len(content)} chars); "
            f"limit is {MAX_HANDOFF_CONTENT_LENGTH} chars."
        )

    try:
        target = _resolve_handoff_path(path)
    except HandoffPathError as exc:
        return tool_error(str(exc))

    guard_err = _handoff_write_guard_error(target, task_id)
    if guard_err:
        return tool_error(guard_err)

    if target.exists() and not overwrite:
        return tool_error(
            f"refusing to overwrite existing file at {target}; "
            "pass overwrite=true to replace it, or choose a different path."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    resolved = str(target.resolve())
    instructions = (
        f"Handoff written to {resolved}. Run /new and reference this file "
        "(or paste its content) to continue in a fresh session."
    )
    return tool_result({
        "success": True,
        "path": resolved,
        "instructions": instructions,
    })


def handoff(args: dict, **kwargs) -> str:
    """Dispatch handler for the `handoff` tool. Only `action="write"` is supported —
    the schema enum permits no other value, but this defensive check keeps the
    handler correct even if something upstream constructs a call by hand."""
    action = args.get("action")
    if action != "write":
        return tool_error("Unsupported action for handoff tool; only 'write' is supported.")
    task_id = kwargs.get("task_id") or "default"
    return _handoff_write(
        content=args.get("content", ""),
        path=args.get("path"),
        overwrite=bool(args.get("overwrite", False)),
        task_id=task_id,
    )


registry.register(
    name="handoff",
    toolset="handoff",
    schema=HANDOFF_SCHEMA,
    handler=lambda args, **kw: handoff(args, **kw),
    emoji="📝",
)
