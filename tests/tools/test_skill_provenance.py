"""Tests for tools/skill_provenance.py — write-origin ContextVar."""

import contextvars

import pytest




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


def test_reset_restores_prior_origin():
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        get_current_write_origin,
    )
    outer = set_current_write_origin("assistant_tool")
    try:
        inner = set_current_write_origin("background_review")
        try:
            assert get_current_write_origin() == "background_review"
        finally:
            reset_current_write_origin(inner)
        assert get_current_write_origin() == "assistant_tool"
    finally:
        reset_current_write_origin(outer)


def test_is_background_review_truthy_only_for_review():
    from tools.skill_provenance import (
        set_current_write_origin,
        reset_current_write_origin,
        is_background_review,
        BACKGROUND_REVIEW,
    )
    for origin, expected in (
        ("foreground", False),
        ("assistant_tool", False),
        ("random_other_value", False),
        (BACKGROUND_REVIEW, True),
    ):
        token = set_current_write_origin(origin)
        try:
            assert is_background_review() is expected, (
                f"is_background_review() wrong for origin={origin!r}"
            )
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
