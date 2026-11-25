"""One plain sentence per ``message_agent`` request outcome, for the sender's wake-up.

A receipt says ``status: ambiguous`` or ``queued`` and a delivery id; models read that as "not
delivered" and offer to resend, or as "still coming" and wait forever. Every runner that ends a
request's wait (the live lane, the fan-out join) leads its output with ``outcome``: the teammate
answered, or "<who> never answered: <why>", plus whether resending is safe. Stdlib only: the fan-out
runner imports it before Hermes boots.
"""

from __future__ import annotations

_NOT_YET = ("queued", "claimed", "pending")


def outcome_line(to: str, entry: dict) -> str:
    """The sentence for one request's final record (``status`` plus ``reply``/``error``/``detail``)."""
    status = entry.get("status")
    if status == "settled":
        return f"{to} answered." if entry.get("reply") else f"{to} finished its turn without replying."
    why = str(entry.get("error") or entry.get("detail") or "").strip().rstrip(".")
    if status == "cancelled":
        return f"{to} never answered: the request was cancelled."
    if status in _NOT_YET:
        state = entry.get("target_state")
        where = {"unreachable": "its gateway is not reachable",
                 "waiting_approval": "it is waiting for its own user to approve a command",
                 "waiting_clarify": "it is waiting for its own user to answer a question",
                 "paused": "it is not starting new turns (restarting or updating)",
                 "working": "its turn is still running"}.get(str(state or ""), "it has not finished")
        return (f"{to} never answered within the wait: {where}. The request is still on their side and its "
                "answer will NOT wake you; tell your user, do not resend.")
    if status == "ambiguous":
        return (f"{to} never answered: whether it received the request is unknown ({why}). It may still run "
                "once their gateway is back; do not resend.")
    safe = "NOT sent" in why or "safe to resend" in why
    return (f"{to} never answered: {why or 'the delivery failed'}. "
            + ("Nothing reached them; you may resend." if safe else "Do not resend blindly."))


def combined_summary(entries: list[dict]) -> str:
    """The fan-out join's lead line: every teammate's sentence, answered ones first."""
    lines = [outcome_line(e.get("to") or "a teammate", e) for e in entries]
    return " ".join(sorted(lines, key=lambda line: " never answered" in line))
