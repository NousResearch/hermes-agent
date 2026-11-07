"""Tests for the SPA preserve -> dream seam (#140005).

What ``preserve_session`` emits must be what ``DreamCycle.light_sleep``
stages: a kind outside ``LOAD_BEARING_KINDS`` is silently dropped at the seam,
and a missing provenance scores as "agent" (losing the user-origin PROMOTE
floor). These tests wire the two modules together so the contract is
code-enforced, not assumed.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.dream_cycle import LOAD_BEARING_KINDS, DreamCycle
from tools.spa_integration import preserve_session, should_preserve


class FakeChainStore:
    """The doc-chain store surface preserve_session writes against."""

    def __init__(self):
        self.adds = []

    def add(self, target, content):
        self.adds.append((target, content))
        return {"success": True}


def _preserved_payload(store, doc_name):
    target, payload = next(
        (t, c) for t, c in store.adds if t == f"doc:{doc_name}"
    )
    return json.loads(payload)


def test_preserved_items_stage_in_light_sleep():
    """The seam: every preserved item must survive Light Sleep's kind filter —
    nothing the SPA kept may be silently dropped by the dream cycle."""
    store = FakeChainStore()
    doc = preserve_session(store, "s1", [
        {"role": "user", "content": "Decision: we ship the bounded read first"},
        {"role": "assistant", "content": "Battery verified: 297 passed, 0 failed"},
    ])
    assert doc == "session-s1"
    payload = _preserved_payload(store, doc)
    assert len(payload["load_bearing"]) == 2

    dc = DreamCycle()
    staged = dc.light_sleep([payload])
    assert staged == 2, "preserved items must not be dropped at the seam"


def test_emitted_kinds_are_load_bearing():
    """Every emitted kind is in LOAD_BEARING_KINDS — never a transport shape."""
    store = FakeChainStore()
    doc = preserve_session(store, "s2", [
        {"role": "assistant", "content": "Falsifier: 4 RED + 53 passed"},
        {"role": "assistant", "content": "Measured +2.2pp at the headline ratio"},
        {"role": "user", "content": "Steer: preserve, don't summarize"},
        {"role": "assistant", "content": "Run verified green"},
    ])
    payload = _preserved_payload(store, doc)
    kinds = [item["kind"] for item in payload["load_bearing"]]
    assert all(k in LOAD_BEARING_KINDS for k in kinds)
    assert kinds == ["falsifier", "measured", "steer", "receipt"]


def test_provenance_follows_role():
    """User-origin content keeps the PROMOTE floor; agent content is scored."""
    store = FakeChainStore()
    doc = preserve_session(store, "s3", [
        {"role": "user", "content": "Decision: keep the anchor under budget"},
        {"role": "assistant", "content": "Decision recorded: 200/200 verified"},
    ])
    payload = _preserved_payload(store, doc)
    by_role = {item["role"]: item for item in payload["load_bearing"]}
    assert by_role["user"]["provenance"] == "user"
    assert by_role["assistant"]["provenance"] == "agent"


def test_user_item_promotes_end_to_end():
    """A user's preserved steer promotes through the full cycle (the floor)."""
    store = FakeChainStore()
    doc = preserve_session(store, "s4", [
        {"role": "user", "content": "Steer: preserve, don't summarize"},
    ])
    payload = _preserved_payload(store, doc)
    dc = DreamCycle()
    summary = dc.run([payload])
    assert summary["promoted"] == 1


def test_noise_is_not_preserved():
    """Messages without load-bearing markers never reach the attachment."""
    store = FakeChainStore()
    doc = preserve_session(store, "s5", [
        {"role": "user", "content": "hello there, how are you"},
    ])
    assert doc is None
    assert store.adds == []


def test_should_preserve_threshold():
    """The trigger fires at the marker threshold and not below it."""
    msgs = [{"role": "assistant", "content": "verified ok"} for _ in range(3)]
    assert should_preserve(msgs, threshold=3) is True
    assert should_preserve(msgs[:2], threshold=3) is False
