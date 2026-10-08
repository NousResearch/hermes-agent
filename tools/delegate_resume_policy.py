"""Whether a detached child completion may start another parent turn.

A user stop, or a paused or cancelled session, stores the delivery and waits.
A later real user message is the resume. A natural completion still wakes the
parent so a free slot can take the next role.
"""

from __future__ import annotations


def session_blocks_synthetic_turn(session=None) -> bool:
    """A stopped, closing, or already-held session must not start from a notification."""
    session = session or {}
    return bool(
        session.get("_delegation_hold")
        or session.get("_turn_cancel_requested")
        or session.get("_closing")
        or session.get("_finalized")
    )


def completion_should_autoresume(event, session=None) -> bool:
    session = session or {}
    if session_blocks_synthetic_turn(session):
        return False
    if not isinstance(event, dict) or event.get("type") != "async_delegation":
        return True
    # Only an admitted, real user turn may release a held interrupted result.
    if event.get("_released_by_user_turn") is True:
        return True
    if str(event.get("status") or "") == "interrupted":
        return False
    results = event.get("results")
    if isinstance(results, list) and results and all(
        isinstance(item, dict) and str(item.get("status") or "") == "interrupted" for item in results
    ):
        return False
    return True


def hold_instead_of_turn(event, session=None) -> bool:
    return isinstance(event, dict) and event.get("type") == "async_delegation" and not completion_should_autoresume(event, session)


def release_delegation_hold(session) -> list:
    """Clear the hold. The caller requeues events that arrived while it was set."""
    if not isinstance(session, dict):
        return []
    session.pop("_delegation_hold", None)
    held = session.pop("_delegation_held_events", []) or []
    released = list(held)
    for event in released:
        if isinstance(event, dict) and event.get("type") == "async_delegation":
            event["_released_by_user_turn"] = True
    return released
