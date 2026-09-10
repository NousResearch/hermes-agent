"""Tests for tools/skill_provenance.py — write-origin ContextVar."""

import contextvars


def test_set_and_get_origin():
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )
    token = set_current_write_origin("background_review")
    try:
        assert get_current_write_origin() == "background_review"
    finally:
        reset_current_write_origin(token)


def test_empty_origin_falls_back_to_foreground():
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )
    token = set_current_write_origin("")
    try:
        # Empty is coerced to "foreground" at the set() boundary.
        assert get_current_write_origin() == "foreground"
    finally:
        reset_current_write_origin(token)


def test_context_isolation_between_copies():
    """ContextVar scoping: modifications in one copy do not leak out."""
    from tools.skill_provenance import (
        set_current_write_origin,
        get_current_write_origin,
        BACKGROUND_REVIEW,
    )

    # Start at the module default.
    original = get_current_write_origin()

    def _run_in_copy():
        set_current_write_origin(BACKGROUND_REVIEW)
        return get_current_write_origin()

    ctx = contextvars.copy_context()
    inside = ctx.run(_run_in_copy)
    assert inside == BACKGROUND_REVIEW
    # Parent context unaffected.
    assert get_current_write_origin() == original


def test_attended_review_is_still_a_background_review():
    """/refine keeps the background_review origin: every curator/skill-ledger/approval guard keyed on
    is_background_review() must still apply; only the unattended-only memory delete gate stands down."""
    from tools.skill_provenance import (
        BACKGROUND_REVIEW, is_background_review, is_unattended_review, reset_current_write_origin,
        reset_review_attended, set_current_write_origin, set_review_attended,
    )

    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        assert is_background_review() and is_unattended_review()
        att = set_review_attended(True)
        try:
            assert is_background_review() and not is_unattended_review()
        finally:
            reset_review_attended(att)
    finally:
        reset_current_write_origin(token)
    assert not is_background_review() and not is_unattended_review()
