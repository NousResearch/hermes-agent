"""``verification_status``: what the workspace's last verification run actually proved.

A compaction summary can say "tests pass" with no age and no commit, and the claim stays on the page after
a branch switch or another edit. The evidence ledger (``agent/verification_evidence.py``) already knows the
answer and already marks evidence stale relative to later edits, but until now only the verify-on-stop
nudge and the TUI gateway read it. This exposes the same read, unchanged, to the model.

Read-only: it never runs a suite and never upgrades a targeted check into "repo green". Gated on the switch
that turns the ledger on, because with ``agent.verify_on_stop`` off nothing records evidence and the tool
would only ever answer "disabled" while costing prompt space.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from tools.registry import registry

_MAX_OUTPUT_SUMMARY_CHARS = 400

VERIFICATION_STATUS_SCHEMA = {
    "name": "verification_status",
    "description": (
        "Report what the current workspace's last verification run actually proved: whether it is "
        "verified, stale (an edit or commit came after it), unverified, or not a code project. Returns the "
        "command, its exit status, when it ran and which paths changed since. Read-only; it never runs "
        "tests. Use it before repeating a 'tests pass' / 'verified' claim carried over from a compaction "
        "summary or from an earlier branch."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "cwd": {
                "type": "string",
                "description": "Workspace directory to inspect. Defaults to the current working directory.",
            },
        },
        "required": [],
    },
}


def verification_status_available() -> bool:
    """Available only when the ledger is live; otherwise the tool can only answer 'disabled'."""
    try:
        from agent.verification_stop import verify_on_stop_enabled

        return verify_on_stop_enabled()
    except Exception:  # health: allow BLE001 -- an unreadable config must not crash discovery
        return False


def _resolve_target(args: dict[str, Any], cwd: Optional[str]) -> str:
    explicit = (args or {}).get("cwd") or cwd
    if explicit:
        return str(explicit)
    try:
        from agent.runtime_cwd import resolve_context_cwd

        resolved = resolve_context_cwd()
        if resolved:
            return str(resolved)
    except Exception:  # health: allow BLE001 -- fall through to the process cwd
        pass
    return os.getcwd()


def _evidence_line(evidence: dict[str, Any]) -> str:
    output = str(evidence.get("output_summary") or "").strip()
    if len(output) > _MAX_OUTPUT_SUMMARY_CHARS:
        output = output[:_MAX_OUTPUT_SUMMARY_CHARS].rstrip() + "… [truncated]"
    parts = [
        f"evidence #{evidence.get('id')}",
        f"command: {evidence.get('command')}",
        f"kind: {evidence.get('kind')}",
        f"scope: {evidence.get('scope')}",
        f"exit: {evidence.get('exit_code')}",
        f"result: {evidence.get('status')}",
        f"ran: {evidence.get('created_at')}",
    ]
    line = " | ".join(parts)
    return f"{line}\noutput: {output}" if output else line


def verification_status_tool(args: Optional[dict[str, Any]] = None, **kwargs: Any) -> str:
    """Render the ledger's verdict for this workspace, with the reason when it is not 'verified'."""
    from agent.verification_evidence import verification_status

    args = args or {}
    target = _resolve_target(args, kwargs.get("cwd"))
    session_id = kwargs.get("session_id") or ""
    state = verification_status(session_id=session_id, cwd=target) or {}
    status = str(state.get("status") or "unknown")
    lines = [f"status: {status}", f"workspace: {state.get('root') or target}"]
    evidence = state.get("evidence")
    if isinstance(evidence, dict) and evidence:
        lines.append(_evidence_line(evidence))
    changed = state.get("changed_paths") or []
    if changed:
        shown = ", ".join(str(p) for p in changed[:10])
        more = f" (+{len(changed) - 10} more)" if len(changed) > 10 else ""
        lines.append(f"changed since: {shown}{more}")
    if status == "stale":
        lines.append(
            "stale: the evidence predates a later edit in this workspace. Re-run the check before "
            "reporting this as passing."
        )
    elif status == "unverified":
        lines.append("unverified: no passing verification is recorded for this workspace in this session.")
    elif status == "not_applicable":
        lines.append("not_applicable: no code project detected here (no manifest or repository root).")
    elif status == "disabled":
        lines.append("disabled: the evidence ledger is off (agent.verify_on_stop), so nothing is recorded.")
    elif status == "failed":
        lines.append("failed: the last recorded verification run did not pass.")
    elif status == "passed":
        lines.append("passed: the last recorded verification run passed and no edit has landed since.")
    return "\n".join(lines)


def _handler(args: Optional[dict[str, Any]] = None, **kwargs: Any) -> str:
    try:
        return verification_status_tool(args, **kwargs)
    except Exception as exc:  # health: allow BLE001 -- a status query must never take the turn down
        return json.dumps({"error": f"verification_status failed: {type(exc).__name__}: {exc}"})


registry.register(
    name="verification_status",
    toolset="terminal",
    schema=VERIFICATION_STATUS_SCHEMA,
    check_fn=verification_status_available,
    handler=_handler,
    emoji="🧾",
)
