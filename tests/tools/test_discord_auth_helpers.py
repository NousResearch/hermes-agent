# tests/tools/test_discord_auth_helpers.py
"""Unit tests for ``tools/discord_auth_helpers.py``.

These helpers are pure functions (no discord.py dependency), so the whole
module runs in CI without the optional ``messaging`` extra installed —
unlike ``test_discord_interactive_views.py``, which skips at module level
when discord.py is absent.  The ``session_owner_only`` auth decision is
security-critical (it gates who may answer a rich clarify prompt), so it
needs CI coverage that does not depend on discord.py being installed.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from tools.discord_auth_helpers import (
    component_check_auth,
    session_owner_check_auth,
)


# ---------------------------------------------------------------------------
# mock interaction helper
# ---------------------------------------------------------------------------

def _make_interaction(user_id: str = "123") -> MagicMock:
    """Build a lightweight mock that quacks like discord.Interaction."""
    user = MagicMock()
    user.id = int(user_id)
    user.roles = []
    interaction = MagicMock()
    interaction.user = user
    return interaction


# ===========================================================================
# session_owner_check_auth — session_owner_only policy (T2)
# ===========================================================================


class TestSessionOwnerCheckAuth:
    """``session_owner_only`` auth decision — union semantics.

    The policy admits the user who started the session (``origin_user_id``)
    even with no static allowlist, while an existing allowlist keeps gating
    independently (union).  Fail-closed invariant: no owner match + no
    allowlist → reject everyone.
    """

    # -- AC #2: owner fast-path with no allowlist ---------------------------

    def test_owner_admitted_no_allowlist(self):
        assert session_owner_check_auth(
            _make_interaction("42"), origin_user_id="42", allowed_user_ids=set(),
        ) is True

    def test_non_owner_rejected_no_allowlist(self):
        assert session_owner_check_auth(
            _make_interaction("99"), origin_user_id="42", allowed_user_ids=set(),
        ) is False

    # -- Fail-closed invariant (no owner + no allowlist) --------------------

    def test_no_owner_no_allowlist_rejects_everyone(self):
        assert session_owner_check_auth(
            _make_interaction("42"), origin_user_id=None, allowed_user_ids=set(),
        ) is False

    def test_no_owner_no_allowlist_rejects_any_id(self):
        assert session_owner_check_auth(
            _make_interaction("1"), origin_user_id=None, allowed_user_ids=None,
        ) is False

    # -- AC #3: allowlist independence (union semantics) --------------------

    def test_allowlist_user_admitted_when_not_owner(self):
        """User in allowlist admits even when they are not the owner."""
        assert session_owner_check_auth(
            _make_interaction("42"), origin_user_id="7", allowed_user_ids={"42"},
        ) is True

    def test_owner_admitted_when_not_in_allowlist(self):
        """Owner admits via the fast-path even when not in the allowlist."""
        assert session_owner_check_auth(
            _make_interaction("7"), origin_user_id="7", allowed_user_ids={"42"},
        ) is True

    def test_neither_owner_nor_allowlist_rejects(self):
        """User who is neither owner nor in allowlist → reject."""
        assert session_owner_check_auth(
            _make_interaction("99"), origin_user_id="7", allowed_user_ids={"42"},
        ) is False

    # -- Edge cases ---------------------------------------------------------

    def test_owner_id_compared_as_string(self):
        """Origin id compared as string regardless of input type."""
        assert session_owner_check_auth(
            _make_interaction("42"), origin_user_id=42, allowed_user_ids=set(),
        ) is True

    def test_interaction_without_user_rejects(self):
        """Interaction with no .user → reject (fail closed)."""
        interaction = MagicMock(spec=[])
        assert session_owner_check_auth(
            interaction, origin_user_id="42", allowed_user_ids=set(),
        ) is False

    def test_user_without_id_rejects(self):
        """User object without .id → reject gracefully."""
        user = MagicMock(spec=[])
        interaction = MagicMock()
        interaction.user = user
        assert session_owner_check_auth(
            interaction, origin_user_id="42", allowed_user_ids={"42"},
        ) is False

    def test_allowlist_only_no_owner(self):
        """origin_user_id=None but allowlist set → allowlist gates alone."""
        assert session_owner_check_auth(
            _make_interaction("42"), origin_user_id=None, allowed_user_ids={"42"},
        ) is True
        assert session_owner_check_auth(
            _make_interaction("99"), origin_user_id=None, allowed_user_ids={"42"},
        ) is False

    def test_does_not_consult_roles(self):
        """session_owner_only ignores roles entirely (user allowlist + owner only)."""
        # Even with a matching role, a non-owner non-allowlisted user rejects.
        interaction = _make_interaction("99")
        interaction.user.roles = [MagicMock(id=55)]
        assert session_owner_check_auth(
            interaction, origin_user_id="42", allowed_user_ids=set(),
        ) is False


# ===========================================================================
# component_check_auth — user-or-role OR semantics (regression guard)
# ===========================================================================


class TestComponentCheckAuth:
    """Mirror of the key invariants for the shared helper — runs in CI."""

    def test_both_empty_allows(self):
        assert component_check_auth(_make_interaction(), None, None) is True

    def test_user_allowed(self):
        assert component_check_auth(_make_interaction("42"), {"42"}, None) is True

    def test_user_not_allowed(self):
        assert component_check_auth(_make_interaction("99"), {"42"}, None) is False

    def test_no_user_attribute_rejects(self):
        interaction = MagicMock(spec=[])
        assert component_check_auth(interaction, {"42"}, None) is False
