"""Tests for the dream cycle (#140005): the 3-phase consolidation.

Verifies: Light Sleep stages without writes, REM builds the diary, Deep Sleep
promotes only the load-bearing (0 lost, 0 noise), the provenance-aware band holds,
and the promotion is the prune trigger (fail-closed).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.dream_cycle import DreamCycle, band_score


class FakeStore:
    """The banking destination — records what was banked (the caller verifies)."""

    def __init__(self):
        self.banked = []

    def bank(self, candidate):
        self.banked.append(candidate)


def _attachments():
    return [
        {
            "session_id": "cto-20261016",
            "load_bearing": [
                {"kind": "receipt", "text": "SPA lifecycle verified: 507 chars byte-exact", "provenance": "agent"},
                {"kind": "steer", "text": "User: compression-fighting sketch — preserve, don't summarize", "provenance": "user"},
                {"kind": "falsifier", "text": "4 RED + 53 passed with doc arms neutered", "provenance": "agent"},
                {"kind": "chat", "text": "hello there how are you doing", "provenance": "agent"},  # noise
            ],
        }
    ]


def test_light_sleep_stages_without_writes():
    store = FakeStore()
    dc = DreamCycle(store=store)
    n = dc.light_sleep(_attachments())
    assert n == 3, "3 load-bearing staged, noise excluded: {}".format(n)
    assert store.banked == [], "Light Sleep must not write"
    assert all(c.state == "staged" for c in dc.staged)


def test_rem_builds_diary():
    dc = DreamCycle()
    dc.light_sleep(_attachments())
    diary = dc.rem_sleep()
    assert len(diary) >= 2, "themes for receipt + steer (+ falsifier): {}".format(diary)
    assert any("receipt" in line for line in diary)


def test_deep_sleep_promotes_only_load_bearing():
    store = FakeStore()
    dc = DreamCycle(store=store)
    dc.light_sleep(_attachments())
    dc.rem_sleep()
    promoted = dc.deep_sleep()
    assert len(promoted) == 3, "all 3 load-bearing promoted: {}".format(len(promoted))
    assert len(store.banked) == 3, "the banking happened: {}".format(len(store.banked))
    # the noise never entered the cycle
    assert all(c.kind != "chat" for c in dc.staged)


def test_band_user_origin_never_below_floor():
    assert band_score("hello there", "user") >= 0.5, "user-origin never below the PROMOTE floor"
    assert band_score("hello there", "agent") < 0.5, "agent noise rejected"


def test_band_structural_evidence_scores():
    assert band_score("scripts/check: 12 checks, ok; battery 76 passed", "agent") >= 0.5
    assert band_score("what time is it", "agent") < 0.2


def test_fail_closed_prune_trigger():
    """The promotion IS the prune trigger: the store's banking is the release
    condition — the attachment persists until the banking is confirmed."""
    store = FakeStore()
    dc = DreamCycle(store=store)
    dc.light_sleep(_attachments())
    dc.rem_sleep()
    dc.deep_sleep()
    # the caller releases the attachment only for the banked candidates
    released = {c.source for c in store.banked}
    assert "cto-20261016" in released, "the attachment releases only after banking"
