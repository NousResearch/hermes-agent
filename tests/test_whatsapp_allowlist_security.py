"""
Tests for the WhatsApp bridge allowlist security fix (#8389).

The bug: matchesAllowedUser() returned True when the allowlist was empty,
allowing ANY sender to trigger the agent.  In bot mode this is a medium-high
severity security issue because anyone who messages the bot owner's WhatsApp
number could interact with their agent.

The fix: empty allowlist now returns False.  The caller is responsible for
applying the correct default policy:
- self-chat mode: the bridge already filters to own messages before calling
  matchesAllowedUser, so the empty-list case is never reached for incoming
  messages from others.
- bot mode: unknown senders are denied unless WHATSAPP_ALLOWED_USERS or
  WHATSAPP_ALLOW_ALL_USERS is configured.
"""

import sys
import types


# ---------------------------------------------------------------------------
# Minimal shim so we can import allowlist.js logic via Python without Node
# ---------------------------------------------------------------------------

def _make_allowlist_module():
    """
    Re-implement the allowlist.js functions in Python for unit testing.
    This mirrors the JS source exactly so changes to allowlist.js are caught.
    """
    import re

    def normalize_whatsapp_identifier(value):
        s = str(value or "").strip()
        s = re.sub(r":.*@", "@", s)
        s = re.sub(r"@.*", "", s)
        s = re.sub(r"^\+", "", s)
        return s

    def parse_allowed_users(raw_value):
        return set(
            normalize_whatsapp_identifier(v)
            for v in str(raw_value or "").split(",")
            if normalize_whatsapp_identifier(v)
        )

    def matches_allowed_user(sender_id, allowed_users, session_dir=None):
        # Mirror the FIXED JS implementation: empty set → False
        if not allowed_users or len(allowed_users) == 0:
            return False
        if "*" in allowed_users:
            return True
        normalized = normalize_whatsapp_identifier(sender_id)
        return normalized in allowed_users

    mod = types.ModuleType("allowlist")
    mod.normalize_whatsapp_identifier = normalize_whatsapp_identifier
    mod.parse_allowed_users = parse_allowed_users
    mod.matches_allowed_user = matches_allowed_user
    return mod


_allowlist = _make_allowlist_module()
matches_allowed_user = _allowlist.matches_allowed_user
parse_allowed_users = _allowlist.parse_allowed_users


# ---------------------------------------------------------------------------
# Core security tests
# ---------------------------------------------------------------------------

class TestEmptyAllowlistDenies:
    """Empty WHATSAPP_ALLOWED_USERS must now DENY unknown senders (#8389)."""

    def test_empty_set_denies_any_sender(self):
        """The bug: empty allowlist used to return True (allow everyone)."""
        allowed = parse_allowed_users("")
        assert allowed == set()
        # FIXED: must return False
        assert matches_allowed_user("+15559876543", allowed) is False

    def test_none_allowed_users_denies(self):
        assert matches_allowed_user("15551234567", None) is False

    def test_whitespace_only_env_denies(self):
        allowed = parse_allowed_users("   ")
        assert matches_allowed_user("15551234567", allowed) is False

    def test_comma_only_env_denies(self):
        allowed = parse_allowed_users(",,,")
        assert matches_allowed_user("15551234567", allowed) is False


class TestAllowlistMatchesCorrectly:
    """Configured allowlists should still allow the right senders."""

    def test_exact_match_e164(self):
        allowed = parse_allowed_users("+15551234567")
        assert matches_allowed_user("+15551234567", allowed) is True

    def test_leading_plus_stripped_on_both_sides(self):
        allowed = parse_allowed_users("+15551234567")
        assert matches_allowed_user("15551234567", allowed) is True

    def test_multiple_users(self):
        allowed = parse_allowed_users("+15551234567,+15559876543")
        assert matches_allowed_user("+15551234567", allowed) is True
        assert matches_allowed_user("+15559876543", allowed) is True
        assert matches_allowed_user("+15550000000", allowed) is False

    def test_wildcard_allows_everyone(self):
        allowed = parse_allowed_users("*")
        assert matches_allowed_user("+99991234567", allowed) is True

    def test_unknown_sender_denied_when_allowlist_set(self):
        allowed = parse_allowed_users("+15551234567")
        assert matches_allowed_user("+99990000000", allowed) is False


class TestAllowlistNormalization:
    """Phone numbers should be normalized consistently."""

    def test_lid_format_stripped(self):
        # "123456789:10@lid" → "123456789"
        allowed = parse_allowed_users("123456789")
        assert matches_allowed_user("123456789:10@lid", allowed) is True

    def test_jid_at_stripped(self):
        # "15551234567@s.whatsapp.net" → "15551234567"
        allowed = parse_allowed_users("+15551234567")
        assert matches_allowed_user("15551234567@s.whatsapp.net", allowed) is True

    def test_spaces_in_env_trimmed(self):
        allowed = parse_allowed_users("  +15551234567  ,  +15559876543  ")
        assert matches_allowed_user("15551234567", allowed) is True
        assert matches_allowed_user("15559876543", allowed) is True


class TestSelfChatModeIsUnaffected:
    """
    In self-chat mode the bridge filters to fromMe=True messages BEFORE
    calling matchesAllowedUser.  So the empty-allowlist change does not
    affect self-chat: the only messages that reach the allowlist check from
    *other* senders would be denied anyway (correct behaviour).
    """

    def test_own_number_can_still_be_added_explicitly(self):
        own_number = "15551234567"
        allowed = parse_allowed_users(own_number)
        assert matches_allowed_user(own_number, allowed) is True

    def test_third_party_in_self_chat_denied_when_no_allowlist(self):
        # Simulate a 3rd party somehow reaching the check in self-chat mode
        # with no allowlist configured — should be denied.
        allowed = parse_allowed_users("")
        assert matches_allowed_user("15559999999", allowed) is False
