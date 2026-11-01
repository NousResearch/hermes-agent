"""Completion verdicts at the delegated-child result boundary."""

from __future__ import annotations

from typing import Any

from agent.turn_final_response_stalls import MANGLED_REASONING_EXIT

MANGLED_REASONING_ERROR = "Subagent returned overlong reasoning without a final answer."


def child_result_outcome(result: dict[str, Any], summary: str) -> tuple[str, str, str, str]:
    """Structured exits outrank summary presence; budget exhaustion retains its own reason."""
    interrupt_note = ""
    if result.get("interrupted", False):
        status, exit_reason = "interrupted", "interrupted"
        from agent.message_content import flatten_message_text

        placeholders = {"", summary.strip(), "Operation interrupted."}
        partial = next((text for row in reversed(result.get("messages") or []) if row.get("role") == "assistant"
                        and (text := flatten_message_text(row.get("content")).strip()) not in placeholders), "")
        if partial:
            interrupt_note, summary = summary.strip(), partial
    elif result.get("failed") or result.get("error"):
        status, exit_reason = "failed", "error"
    elif result.get("completed") and result.get("turn_exit_reason") == MANGLED_REASONING_EXIT:
        status, exit_reason = "failed", "mangled"
        summary = "[Unreliable reasoning-only output; no final answer was produced.]\n\n" + summary
    else:
        exit_reason = "completed" if result.get("completed", False) else "max_iterations"
        status = "completed" if summary and summary.strip() != "(empty)" else "failed"
    return status, exit_reason, summary, interrupt_note
