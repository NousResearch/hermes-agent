"""Continue only the default-store retirements an earlier profile resume authorized."""

import sqlite3

from hermes_state_lineage import SessionLineageError


def _retirement_error(ctx, outcome, error):
    message = outcome.get("error") or "Donor retirement is incomplete; inspect the donor before retrying resume"
    return error(ctx.rid, 4001, message)


def resume_pending(ctx, get_default_db, error, log):
    from hermes_state_adoption_records import retirement_pending_source
    try:
        source = retirement_pending_source(ctx.db, ctx.target)
        if source is None:
            return None
        donor = get_default_db()
        if donor is None:
            return error(ctx.rid, 4001, "Pending adoption donor is unavailable; restore its connection before retrying resume")
        # This path initiates default-store healing. Other explicit transfers keep
        # their own donor selection; never open arbitrary paths from a saved hint.
        if str(donor.db_path.resolve(strict=True)) != source:
            return None
        outcome = ctx.db.adopt_session_lineage_from(donor, ctx.target, _pending_only=True)
        if outcome.get("retry_skipped"):
            return None  # intent was revoked or another request already completed it
        if not outcome.get("adopted") or not outcome.get("donor_retired"):
            return _retirement_error(ctx, outcome, error)
        ctx.found = ctx.db.get_session(ctx.target)
        return None
    except (SessionLineageError, sqlite3.Error) as exc:
        log.warning("pending profile adoption could not be verified for %s", ctx.target, exc_info=True)
        return error(ctx.rid, 4001, f"Could not verify pending adoption: {exc}")


def adopt_stranded(ctx, get_default_db, error, log):
    """Fresh healing uses exact IDs and an unretired default-store donor."""
    try:
        donor = get_default_db()
        row = donor.get_session(ctx.target) if donor is not None else None
        if not row or row.get("archived"):
            return None
        outcome = ctx.db.adopt_session_lineage_from(donor, row["id"])
        if outcome.get("adopted"):
            if not outcome.get("donor_retired"):
                return _retirement_error(ctx, outcome, error)
            log.info("adopted stranded session %s into profile %s", row["id"], ctx.profile or "?")
            ctx.found = ctx.db.get_session(row["id"])
            if ctx.found:
                ctx.target = ctx.found["id"]
        elif outcome.get("error"):
            return error(ctx.rid, 4001, outcome["error"])
        return None
    except Exception as exc:
        log.exception("stranded-session adoption failed for %s", ctx.target)
        return error(ctx.rid, 4001, f"Could not adopt the stranded session: {exc}")
