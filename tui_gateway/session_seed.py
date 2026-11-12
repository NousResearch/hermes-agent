"""Seed compensation and replacement commit evidence."""

import logging

logger = logging.getLogger("tui_gateway.server")


def finish_seed_replacement(session, session_key):
    """The caller holds history_lock and has committed the complete replacement."""
    pending = session.get("_branch_seed_pending")
    if pending is not None and pending[0] == session_key:
        session.pop("_branch_seed_pending")
        session["_branch_seed_persisted"] = True


def compensate_seed(db, session_key, record):
    """Only a positively acknowledged discard restores fresh-ID eligibility."""
    compensated = False
    try:
        compensated = db.discard_unadmitted_session(session_key) is True
    except Exception:
        logger.debug("branch seed compensation delete failed for %s", session_key, exc_info=True)
    if record is not None:
        record["_fresh_generated_key"] = compensated
        record["_seed_compensation_unverified"] = not compensated
        if compensated:
            record.pop("_branch_seed_pending", None)
