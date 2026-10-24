"""Tests for cross-instance channel claim surface in tools/bot_relay.py.

These tests exercise acquire, renew, release, and steal against the
bot_relay/channel_claims/ file tree. They do not touch a real gateway."""

import time

import pytest

from tools import bot_relay


@pytest.fixture()
def root(tmp_path):
    return tmp_path


# ── profile normalization ──────────────────────────────────────────────────────


def test_normalize_profile_for_claim():
    assert bot_relay._normalize_profile_for_claim(None) == "main"
    assert bot_relay._normalize_profile_for_claim("") == "main"
    assert bot_relay._normalize_profile_for_claim("default") == "main"
    assert bot_relay._normalize_profile_for_claim("main") == "main~"
    assert bot_relay._normalize_profile_for_claim("custom") == "custom"


# ── claim path construction ──────────────────────────────────────────────────


def test_channel_claim_path_omits_empty_scope_id(root):
    path = bot_relay._channel_claim_path(root, "mattermost", "", "C123", "main")
    assert "bot_relay/channel_claims/mattermost/C123/main.json" in str(path).replace("\\", "/")


def test_channel_claim_path_includes_scope_id(root):
    path = bot_relay._channel_claim_path(root, "slack", "team1", "C123", "main")
    assert "bot_relay/channel_claims/slack/team1/C123/main.json" in str(path).replace("\\", "/")


# ── acquire ──────────────────────────────────────────────────────────────────


def test_acquire_empty_channel(root):
    ok, claim = bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1"
    )
    assert ok is True
    assert claim["instance_id"] == "inst-a"
    assert claim["session_id"] == "sid-1"
    assert claim["lease_expire"] > time.time()


def test_acquire_blocked_by_other_instance(root):
    bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1"
    )
    ok, claim = bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-b", "sid-2"
    )
    assert ok is False
    assert claim["instance_id"] == "inst-a"


def test_acquire_same_instance_overwrites(root):
    """The same instance may replace its own claim (session reset)."""
    bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1"
    )
    ok, claim = bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-2"
    )
    assert ok is True
    assert claim["session_id"] == "sid-2"


def test_acquire_steals_expired_claim(root):
    # Write a claim that expired one second ago.
    bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1",
        lease_seconds=-1,
    )
    ok, claim = bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-b", "sid-2"
    )
    assert ok is True
    assert claim["instance_id"] == "inst-b"


# ── read ─────────────────────────────────────────────────────────────────────


def test_read_relay_claim_roundtrip(root):
    bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1"
    )
    claim = bot_relay.read_relay_claim(
        root, "mattermost", "", "C123", "default"
    )
    assert claim is not None
    assert claim["session_id"] == "sid-1"


def test_read_missing_relay_claim(root):
    assert bot_relay.read_relay_claim(
        root, "mattermost", "", "C123", "default"
    ) is None


# ── renew ────────────────────────────────────────────────────────────────────


def test_renew_extends_lease(root):
    bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1",
        lease_seconds=10,
    )
    ok = bot_relay.renew_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1",
        lease_seconds=600,
    )
    assert ok is True
    claim = bot_relay.read_relay_claim(
        root, "mattermost", "", "C123", "default"
    )
    assert claim["lease_expire"] > time.time() + 580


def test_renew_refused_for_wrong_instance(root):
    bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1"
    )
    ok = bot_relay.renew_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-b", "sid-1"
    )
    assert ok is False


def test_renew_refused_for_wrong_session(root):
    bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1"
    )
    ok = bot_relay.renew_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-2"
    )
    assert ok is False


def test_renew_missing_claim(root):
    ok = bot_relay.renew_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1"
    )
    assert ok is False


# ── release ──────────────────────────────────────────────────────────────────


def test_release_deletes_claim(root):
    bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1"
    )
    bot_relay.release_relay_claim(
        root, "mattermost", "", "C123", "default", "sid-1"
    )
    assert bot_relay.read_relay_claim(
        root, "mattermost", "", "C123", "default"
    ) is None


def test_release_idempotent(root):
    """Releasing a missing or already-released claim is a no-op."""
    bot_relay.release_relay_claim(
        root, "mattermost", "", "C123", "default", "sid-1"
    )


def test_release_refused_for_wrong_session(root):
    bot_relay.acquire_relay_claim(
        root, "mattermost", "", "C123", "default", "inst-a", "sid-1"
    )
    # Another session's release does nothing.
    bot_relay.release_relay_claim(
        root, "mattermost", "", "C123", "default", "sid-2"
    )
    claim = bot_relay.read_relay_claim(
        root, "mattermost", "", "C123", "default"
    )
    assert claim is not None
    assert claim["session_id"] == "sid-1"


def test_acquire_concedes_a_lost_create_race(root, monkeypatch):
    """A gateway that read the tree before a peer's create landed must lose via
    O_EXCL, not overwrite the peer and admit both (pass-1 Major 3)."""
    won, _ = bot_relay.acquire_relay_claim(
        root, "mattermost", "", "chat-race", "main", "gw-a", "sid-a")
    assert won
    real_read = bot_relay._read_relay_claim
    reads = {"n": 0}

    def stale_read(path):
        reads["n"] += 1
        if reads["n"] == 1:
            return None  # gw-b's read ran before gw-a's file landed
        return real_read(path)

    monkeypatch.setattr(bot_relay, "_read_relay_claim", stale_read)
    lost, seen = bot_relay.acquire_relay_claim(
        root, "mattermost", "", "chat-race", "main", "gw-b", "sid-b")
    assert not lost
    assert seen is not None
    assert seen["instance_id"] == "gw-a"
    claim = bot_relay.read_relay_claim(root, "mattermost", "", "chat-race", "main")
    assert claim is not None
    assert claim["instance_id"] == "gw-a"


def test_claim_path_rejects_traversal_segments(root):
    """Hostile or corrupted relay data must not turn a segment into a path
    traversal for an arbitrary JSON write (pass-1 Minor)."""
    with pytest.raises(ValueError):
        bot_relay.acquire_relay_claim(
            root, "mattermost", "", "../../escape", "main", "gw-a", "sid-a")
    assert list(root.rglob("*.json")) == []


def test_unreadable_young_claim_fails_closed(root):
    """A crashed write leaves a zero-byte file; the name is claimed by an
    unknown writer, so the gate must not read it as 'no claim' (pass-2 Major)."""
    prof = bot_relay._normalize_profile_for_claim("main")
    p = bot_relay._channel_claim_path(root, "mattermost", "", "chat-corrupt", prof)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    acquired, claim = bot_relay.acquire_relay_claim(
        root, "mattermost", "", "chat-corrupt", "main", "gw-a", "sid-a")
    assert acquired is False
    assert claim is not None
    assert claim["instance_id"] == "<unreadable-claim-file>"
    assert claim["lease_expire"] >= time.time()


def test_unreadable_dead_claim_heals_then_acquires(root):
    """Past the lease window the crashed writer's leftover heals: unlink,
    fresh create, acquire True (pass-2 Major)."""
    import os

    prof = bot_relay._normalize_profile_for_claim("main")
    p = bot_relay._channel_claim_path(root, "mattermost", "", "chat-corrupt", prof)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    old = time.time() - 7200
    os.utime(p, (old, old))
    acquired, _ = bot_relay.acquire_relay_claim(
        root, "mattermost", "", "chat-corrupt", "main", "gw-a", "sid-a")
    assert acquired is True
    reread = bot_relay.read_relay_claim(root, "mattermost", "", "chat-corrupt", "main")
    assert reread is not None and reread["instance_id"] == "gw-a"


def test_release_removes_unreadable_claim(root):
    """Release heals an unreadable claim file instead of leaving it to stand
    the channel down for the rest of the lease (pass-2 Minor)."""
    prof = bot_relay._normalize_profile_for_claim("main")
    p = bot_relay._channel_claim_path(root, "mattermost", "", "chat-corrupt", prof)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\xff not json")
    bot_relay.release_relay_claim(root, "mattermost", "", "chat-corrupt", "main", "sid-any")
    assert not p.exists()
