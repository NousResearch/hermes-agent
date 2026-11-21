"""``stage_write`` must not queue a second proposal for the same change.

It minted a UUID and wrote the record without ever reading the pending
directory, so two background-review passes that reached the same conclusion
always produced two entries. The damaging shape is two ``replace`` proposals
sharing an ``old_text``: they are mutually exclusive, not merely redundant,
because once either applies the other's anchor no longer exists in the file.
"""

from __future__ import annotations

import pytest

from tools import write_approval as wa


def _stage(payload, *, subsystem=None, summary="s"):
    return wa.stage_write(
        subsystem or wa.MEMORY, payload, summary=summary, origin="background_review"
    )


def _replace(old_text, new_text, target="user"):
    return {"action": "replace", "target": target,
            "old_text": old_text, "new_text": new_text}


def test_reworded_replace_of_the_same_anchor_is_not_queued_twice():
    """The reported shape: identical old_text, different wording."""
    first = _stage(_replace("likes tea", "prefers tea in the morning"))
    second = _stage(_replace("likes tea", "is a morning tea drinker"))

    assert second["id"] == first["id"], "a second proposal for the same anchor was queued"
    assert second.get("deduplicated") is True
    assert len(wa.list_pending(wa.MEMORY)) == 1


def test_the_first_proposal_is_the_one_kept():
    """Keeping the earlier record preserves review order and its id stays valid."""
    first = _stage(_replace("likes tea", "prefers tea"))
    _stage(_replace("likes tea", "drinks tea"))

    pending = wa.list_pending(wa.MEMORY)
    assert [r["id"] for r in pending] == [first["id"]]
    assert pending[0]["payload"]["new_text"] == "prefers tea"


def test_distinct_anchors_both_queue():
    """Guard: different old_text is a different change and must survive."""
    a = _stage(_replace("likes tea", "prefers tea"))
    b = _stage(_replace("likes coffee", "prefers coffee"))

    assert a["id"] != b["id"]
    assert len(wa.list_pending(wa.MEMORY)) == 2


def test_same_anchor_in_a_different_target_both_queue():
    """Guard: the target file is part of the identity."""
    _stage(_replace("likes tea", "prefers tea", target="user"))
    _stage(_replace("likes tea", "prefers tea", target="agent"))

    assert len(wa.list_pending(wa.MEMORY)) == 2


def test_identical_adds_are_deduplicated():
    add = {"action": "add", "target": "user", "content": "lives in Ankara"}
    first = _stage(dict(add))
    second = _stage(dict(add))

    assert second["id"] == first["id"]
    assert len(wa.list_pending(wa.MEMORY)) == 1


def test_different_add_content_both_queue():
    """Guard: re-worded *content* has no safe equivalence and must not collapse."""
    _stage({"action": "add", "target": "user", "content": "lives in Ankara"})
    _stage({"action": "add", "target": "user", "content": "is based in Ankara"})

    assert len(wa.list_pending(wa.MEMORY)) == 2


def test_identical_removes_are_deduplicated():
    """``remove`` is anchored on old_text, same as replace."""
    rm = {"action": "remove", "target": "user", "old_text": "stale entry"}
    first = _stage(dict(rm))
    second = _stage(dict(rm))

    assert second["id"] == first["id"]
    assert len(wa.list_pending(wa.MEMORY)) == 1


def test_removes_of_different_anchors_both_queue():
    """Guard: distinct anchors are distinct removals."""
    _stage({"action": "remove", "target": "user", "old_text": "one"})
    _stage({"action": "remove", "target": "user", "old_text": "two"})

    assert len(wa.list_pending(wa.MEMORY)) == 2


def test_a_remove_and_a_replace_of_the_same_anchor_stay_separate():
    """Deleting a line and rewriting it are different outcomes to review."""
    _stage({"action": "remove", "target": "user", "old_text": "likes tea"})
    _stage(_replace("likes tea", "prefers tea"))

    assert len(wa.list_pending(wa.MEMORY)) == 2


def test_actions_without_a_safe_key_are_never_deduplicated():
    """Guard: only replace/remove/add have a defined identity."""
    _stage({"action": "reorder", "target": "user", "content": "x"})
    _stage({"action": "reorder", "target": "user", "content": "x"})

    assert len(wa.list_pending(wa.MEMORY)) == 2


def test_subsystems_do_not_collide():
    """Guard: an identical payload in another subsystem is a separate queue."""
    payload = _replace("likes tea", "prefers tea")
    _stage(dict(payload), subsystem=wa.MEMORY)
    _stage(dict(payload), subsystem=wa.SKILLS)

    assert len(wa.list_pending(wa.MEMORY)) == 1
    assert len(wa.list_pending(wa.SKILLS)) == 1


def test_unreadable_queue_still_stages(monkeypatch):
    """Fail open: a broken queue scan must not swallow a pending write."""
    monkeypatch.setattr(
        wa, "list_pending", lambda _s: (_ for _ in ()).throw(OSError("boom"))
    )

    record = _stage(_replace("likes tea", "prefers tea"))

    assert record.get("deduplicated") is not True
    assert record["id"]


def test_blank_anchor_is_not_a_dedup_key():
    """An empty old_text identifies nothing; two such proposals stay separate."""
    _stage({"action": "replace", "target": "user", "old_text": "  ", "new_text": "a"})
    _stage({"action": "replace", "target": "user", "old_text": "  ", "new_text": "b"})

    assert len(wa.list_pending(wa.MEMORY)) == 2


def test_dedup_key_shape():
    key = wa._dedup_key(wa.MEMORY, _replace("anchor", "whatever"))

    assert key == (wa.MEMORY, "replace", "user", "anchor")
    assert wa._dedup_key(wa.MEMORY, {"action": "replace", "target": "user"}) is None
    assert wa._dedup_key(wa.MEMORY, {"action": "noop"}) is None
