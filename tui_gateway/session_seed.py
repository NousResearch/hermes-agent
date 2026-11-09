"""Retire pending seed evidence only after a replacement transcript commits."""


def finish_seed_replacement(session, session_key):
    """The caller holds history_lock and has committed the complete replacement."""
    pending = session.get("_branch_seed_pending")
    if pending is not None and pending[0] == session_key:
        session.pop("_branch_seed_pending")
        session["_branch_seed_persisted"] = True
