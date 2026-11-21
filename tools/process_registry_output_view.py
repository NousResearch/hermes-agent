"""Read-side output snapshots for ProcessRegistry: the ANSI-stripped tail helper and the
completion-sized output dict that wait/poll/kill snapshots and completion notifications
must render identically (moved off the facade per the facade + siblings layout)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession


def _output_tail(session: "ProcessSession", n: int) -> str:
    """Last *n* chars of the session output with ANSI sequences stripped."""
    from tools.ansi_strip import strip_ansi

    return strip_ansi(session.output_buffer[-n:])


def _completion_output(session: "ProcessSession") -> dict:
    """``output`` sized by the session's ``completion_output_chars`` plus ``output_cut`` when trimmed.

    Shared by the completion notification AND the wait/poll/kill snapshots: a bot in an api_server
    or one-shot session cannot receive notifications and polls instead, so the polled result must
    carry the same whole reply and the same cut marker (#115334)."""
    from tools.process_registry import COMPLETION_OUTPUT_CHARS

    limit = session.completion_output_chars or COMPLETION_OUTPUT_CHARS
    cut = len(session.output_buffer) - limit
    return {"output": _output_tail(session, limit), **({"output_cut": cut} if cut > 0 else {})}
