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
