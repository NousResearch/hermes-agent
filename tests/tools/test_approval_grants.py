"""Bounded approval grants (tools/approval_grants.py) and their wiring into the gate.

Invariants pinned here, in order of how much they matter:

1. Every grant expires. There is no way to create an unbounded one; the cap holds
   even when the caller passes a huge duration.
2. Grants are chat-scoped: a grant in chat A never approves anything in chat B.
3. A use-limited grant is consumed, and stops covering after the last use.
4. Grants persist across a gateway restart (fresh process, same HERMES_HOME),
   because a VPS deploy must not silently revoke what the user just approved.
5. ``is_approved`` (the read every gate does) honors grants, so terminal,
   execute_code, plugin escalations and MCP trust all pick them up with no
   per-gate wiring.
6. The strict conversational parser refuses a scope with a trailing instruction
   ("yes for an hour and also wipe the disk") so a grant can never smuggle in
   extra intent.
"""

import time

import pytest

from tools import approval_grants as grants


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    # HERMES_HOME is already redirected by the suite-wide fixture; drop in-memory state so
    # each test starts from an empty store on disk.
    grants.reset_for_tests()
    yield
    grants.reset_for_tests()


class TestParseGrantSpec:
    @pytest.mark.parametrize("text,seconds,uses", [
        ("for 30m", 1800, None),
        ("for 30 minutes", 1800, None),
        ("for 2 hours", 7200, None),
        ("for 2h", 7200, None),
        ("an hour", 3600, None),
        ("for an hour", 3600, None),
        ("for the next hour", 3600, None),
        ("3 times", None, 3),
        ("three times", None, 3),
        ("3 more times", None, 3),
        ("3x", None, 3),
        ("for 1 hour, 5 times", 3600, 5),
    ])
    def test_parses_duration_and_count(self, text, seconds, uses):
        spec = grants.parse_grant_spec(text)
        assert spec is not None
        assert spec.seconds == seconds
        assert spec.max_uses == uses

    def test_today_is_until_local_midnight(self):
        spec = grants.parse_grant_spec("for today")
        assert spec is not None
        assert grants.MIN_GRANT_SECONDS <= spec.seconds <= 86400

    @pytest.mark.parametrize("text", ["", "yes", "session", "always", "no", "do it", "1 second please maybe"])
    def test_no_scope_returns_none(self, text):
        # "1 second please maybe" is under MIN and has no clean scope: falls through.
        # (It does parse as 1s in non-strict mode and gets clamped; the point of this row is
        # that strict mode refuses it — see TestStrict.)
        if text == "1 second please maybe":
            assert grants.parse_grant_spec(text, strict=True) is None
        else:
            assert grants.parse_grant_spec(text) is None

    def test_duration_is_clamped_to_cap(self):
        spec = grants.parse_grant_spec("for 6 days")
        assert spec.seconds == grants.MAX_GRANT_SECONDS

    def test_duration_is_clamped_to_floor(self):
        spec = grants.parse_grant_spec("for 5 seconds")
        assert spec.seconds == grants.MIN_GRANT_SECONDS

    def test_zero_uses_is_not_a_grant(self):
        assert grants.parse_grant_spec("0 times") is None


class TestStrict:
    """The conversational router calls the parser in strict mode."""

    @pytest.mark.parametrize("text", ["for 30m", "for an hour", "3 times", "for 1 hour and 3 times", "for 2h, please"])
    def test_pure_scope_passes(self, text):
        assert grants.parse_grant_spec(text, strict=True) is not None

    @pytest.mark.parametrize("text", [
        "for an hour and also wipe the disk",
        "for 30m but skip the backups",
        "3 times then delete everything",
        "for today if you think it's safe",
    ])
    def test_trailing_instruction_is_refused(self, text):
        assert grants.parse_grant_spec(text, strict=True) is None


class TestBoundedByConstruction:
    def test_create_always_sets_expiry(self):
        g = grants.create("s", "rm -rf", "recursive delete", grants.GrantSpec(max_uses=3))
        assert g.expires_at > time.time()
        assert g.expires_at - time.time() <= grants.MAX_GRANT_SECONDS + 1

    def test_create_caps_absurd_duration(self):
        g = grants.create("s", "k", "d", grants.GrantSpec(seconds=10 ** 9))
        assert g.expires_at - time.time() <= grants.MAX_GRANT_SECONDS + 1

    def test_expired_grant_does_not_cover(self):
        g = grants.create("s", "k", "d", grants.GrantSpec(seconds=grants.MIN_GRANT_SECONDS))
        g.expires_at = time.time() - 1  # simulate the clock passing
        assert grants.consume("s", "k") is False
        assert grants.list_active("s") == []


class TestChatScoped:
    def test_grant_in_one_chat_does_not_cover_another(self):
        grants.create("photon:+1555:dm", "k", "d", grants.GrantSpec(seconds=3600))
        assert grants.consume("photon:+1666:dm", "k") is False
        assert grants.consume("photon:+1555:dm", "k") is True

    def test_empty_session_key_never_matches(self):
        grants.create("s", "k", "d", grants.GrantSpec(seconds=3600))
        assert grants.consume("", "k") is False


class TestConsumed:
    def test_use_limited_grant_runs_out(self):
        grants.create("s", "k", "d", grants.GrantSpec(max_uses=2))
        assert grants.consume("s", "k") is True
        assert grants.consume("s", "k") is True
        assert grants.consume("s", "k") is False
        assert grants.list_active("s") == []

    def test_list_active_is_read_only(self):
        grants.create("s", "k", "d", grants.GrantSpec(max_uses=1))
        grants.list_active("s")
        grants.list_active("s")
        assert grants.consume("s", "k") is True  # listing did not burn the single use

    def test_aliases_are_honored(self):
        grants.create("s", "canonical", "d", grants.GrantSpec(seconds=3600))
        assert grants.consume("s", "legacy-key", aliases={"canonical", "legacy-key"}) is True


class TestPersistsAcrossRestart:
    def test_grant_survives_process_state_reset(self):
        grants.create("s", "k", "d", grants.GrantSpec(seconds=3600, max_uses=3))
        assert grants.consume("s", "k") is True  # uses -> 1, saved
        grants.reset_for_tests()  # == new gateway process
        active = grants.list_active("s")
        assert len(active) == 1
        assert active[0].uses == 1
        assert grants.consume("s", "k") is True

    def test_store_is_owner_only(self):
        grants.create("s", "k", "d", grants.GrantSpec(seconds=3600))
        path = grants._store_path()
        assert path.exists()
        assert (path.stat().st_mode & 0o777) == 0o600


class TestRevoke:
    def test_revoke_one_by_prefix(self):
        a = grants.create("s", "k1", "d", grants.GrantSpec(seconds=3600))
        grants.create("s", "k2", "d", grants.GrantSpec(seconds=3600))
        assert grants.revoke("s", a.id[:4]) == 1
        assert [g.pattern_key for g in grants.list_active("s")] == ["k2"]

    def test_revoke_all_is_chat_scoped(self):
        grants.create("s", "k", "d", grants.GrantSpec(seconds=3600))
        grants.create("other", "k", "d", grants.GrantSpec(seconds=3600))
        assert grants.revoke("s") == 1
        assert grants.list_active("other")


class TestGateIntegration:
    """``is_approved`` is what every gate reads; grants must flow through it."""

    def test_is_approved_honors_grant(self):
        from tools.approval import is_approved

        grants.create("sess", "dangerous:rm", "recursive delete", grants.GrantSpec(max_uses=1))
        assert is_approved("sess", "dangerous:rm") is True
        assert is_approved("sess", "dangerous:rm") is False  # consumed

    def test_is_approved_does_not_leak_across_sessions(self):
        from tools.approval import is_approved

        grants.create("sess-a", "k", "d", grants.GrantSpec(seconds=3600))
        assert is_approved("sess-b", "k") is False

    def test_persist_choice_grant_creates_grant_for_each_warning(self):
        from tools.approval import _persist_choice

        warnings = [("k1", "first", False), ("tirith:x", "tirith finding", True)]
        _persist_choice("sess", "grant:for 30m", warnings)
        active = {g.pattern_key: g for g in grants.list_active("sess")}
        assert set(active) == {"k1", "tirith:x"}
        for g in active.values():
            assert 0 < g.remaining_seconds() <= 1800

    def test_persist_choice_unparseable_grant_persists_nothing(self):
        from tools.approval import _persist_choice

        _persist_choice("sess", "grant:whatever", [("k", "d", False)])
        assert grants.list_active("sess") == []
