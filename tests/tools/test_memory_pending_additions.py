"""Review UI distinguishes independent saves from the pending destructive proposal."""

import json

import pytest

from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import memory_tool as mt
from tools import write_approval as wa


@pytest.fixture
def review_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tools.skill_provenance.is_unattended_review", lambda: True)
    monkeypatch.setattr(wa, "current_origin", lambda: "background_review")
    monkeypatch.setattr(wa, "write_approval_enabled", lambda subsystem: False)
    monkeypatch.setattr(wa, "unattended_literal_preservation_enabled", lambda: False)
    return tmp_path


def _stage(store, target="memory", additions=None):
    result = json.loads(mt.memory_tool(target=target, store=store, operations=[
        {"action": "remove", "old_text": "rule one"},
        *(additions if additions is not None else [
            {"action": "add", "content": "fork consolidation"}]),
    ]))
    assert result.get("staged"), result
    return result


def _entries(target):
    return mt.MemoryStore._read_file(mt.MemoryStore._path_for(target))


@pytest.mark.parametrize("target", ["memory", "user"])
@pytest.mark.parametrize("selection", ["id", "all"])
def test_mixed_review_explains_saved_additions_before_and_after_reject(review_home, target, selection):
    store = mt.MemoryStore()
    assert store.add(target, "rule one")["success"]
    result = _stage(store, target)
    assert result["additions_saved"] == 1
    assert _entries(target) == ["rule one", "fork consolidation"]
    pending = handle_pending_subcommand("memory", ["pending"])
    assert "- add: fork consolidation (already on disk; kept on reject)" in pending
    assert "removes entry: rule one" in pending
    assert "Reject drops only pending proposals" in pending
    assert "rejecting the proposal will not undo these saved additions" in result["message"]
    assert "reject to drop only the pending proposal" in result["message"]
    rejected = handle_pending_subcommand(
        "memory", ["reject", result["pending_id"] if selection == "id" else "all"])
    assert "Rejected" in rejected
    assert "entries already saved on disk are kept" in rejected
    assert _entries(target) == ["rule one", "fork consolidation"]
    assert not wa.list_pending("memory")


def test_partial_capacity_alias_and_duplicate_get_current_disk_labels(review_home):
    store = mt.MemoryStore(memory_char_limit=35)
    assert store.add("memory", "rule one")["success"]
    # The final batch fits, but only the small addition fits before removing rule one.
    big = "x" * 27
    result = _stage(store, additions=[
        {"action": "add", "content": big},
        {"action": "add", "new_text": "  tiny  "},
        {"action": "add", "content": "tiny"},
    ])
    assert result["additions_saved"] == 1
    pending = handle_pending_subcommand("memory", ["pending"])
    assert f"- add: {big} (not on disk; pending approval)" in pending
    assert "- add: tiny (already on disk; kept on reject)" in pending
    assert _entries("memory") == ["rule one", "tiny"]


def test_approval_on_still_stages_everything(review_home, monkeypatch):
    monkeypatch.setattr(wa, "write_approval_enabled", lambda subsystem: True)
    store = mt.MemoryStore()
    assert store.add("memory", "rule one")["success"]
    result = _stage(store)
    assert result["additions_saved"] == 0
    pending = handle_pending_subcommand("memory", ["pending"])
    assert "- add: fork consolidation (not on disk; pending approval)" in pending
    assert "already on disk; kept on reject" not in pending
    assert "saved independently" not in result["message"]
    handle_pending_subcommand("memory", ["reject", "all"])
    assert _entries("memory") == ["rule one"]


def test_listing_refreshes_disk_not_the_live_store_or_proposal_metadata(review_home):
    store = mt.MemoryStore()
    assert store.add("memory", "rule one")["success"]
    _stage(store)
    # The listing must read another session's edit, not the cached entries.
    other_session = mt.MemoryStore()
    assert other_session.replace("memory", "fork consolidation", "edited elsewhere")["success"]
    assert "fork consolidation" in store.memory_entries  # stale session cache
    pending = handle_pending_subcommand("memory", ["pending"], memory_store=store)
    assert "- add: fork consolidation (not on disk; pending approval)" in pending
    handle_pending_subcommand("memory", ["reject", "all"])
    assert _entries("memory") == ["rule one", "edited elsewhere"]


def test_unreadable_memory_is_unknown_not_assumed_empty(review_home):
    store = mt.MemoryStore()
    assert store.add("memory", "rule one")["success"]
    _stage(store)
    path = store._path_for("memory")
    path.write_bytes(b"\xffinvalid utf8")
    pending = handle_pending_subcommand("memory", ["pending"])
    assert "- add: fork consolidation (disk status unavailable)" in pending
    assert "not on disk; pending approval" not in pending
    assert "already on disk; kept on reject" not in pending
    assert path.read_bytes() == b"\xffinvalid utf8"


def test_approve_replays_saved_addition_once(review_home):
    store = mt.MemoryStore()
    assert store.add("memory", "rule one")["success"]
    result = _stage(store)
    approved = handle_pending_subcommand("memory", ["approve", result["pending_id"]], memory_store=store)
    assert "Approved 1" in approved
    assert _entries("memory") == ["fork consolidation"]
    assert not wa.list_pending("memory")


def test_skills_and_missing_reject_keep_existing_behavior(review_home):
    record = wa.stage_write("skills", {"action": "create", "name": "demo"},
                            summary="create demo", origin="background_review")
    pending = handle_pending_subcommand("skills", ["pending"])
    assert "Review full diff: /skills diff" in pending
    assert "disk" not in pending
    assert handle_pending_subcommand("skills", ["reject", record["id"]]) == (
        f"Rejected pending skills write '{record['id']}'.")
    assert handle_pending_subcommand("memory", ["reject", "missing"]) == (
        "No pending memory write with id 'missing'.")


def test_profile_listing_does_not_reuse_disk_snapshot(review_home, monkeypatch):
    store = mt.MemoryStore()
    assert store.add("memory", "rule one")["success"]
    _stage(store)
    assert "already on disk; kept on reject" in handle_pending_subcommand("memory", ["pending"])

    other_home = review_home / "other-profile"
    other_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other_home))
    wa.stage_write("memory", {"action": "add", "target": "memory", "content": "fork consolidation"},
                   summary="legacy pending add", origin="background_review")
    assert "not on disk; pending approval" in handle_pending_subcommand("memory", ["pending"])
    assert not mt.MemoryStore._path_for("memory").exists()  # listing is read-only

    monkeypatch.setenv("HERMES_HOME", str(review_home))
    assert "already on disk; kept on reject" in handle_pending_subcommand("memory", ["pending"])
    assert _entries("memory") == ["rule one", "fork consolidation"]


def test_listing_reads_once_per_target_without_mixing_memory_and_user(review_home, monkeypatch):
    store = mt.MemoryStore()
    assert store.add("memory", "shared text")["success"]
    for target in ("memory", "memory", "user"):
        wa.stage_write("memory", {"action": "add", "target": target, "content": "shared text"},
                       summary=f"add to {target}", origin="background_review")
    real_read = mt.MemoryStore._read_raw_checked
    reads = []

    def counted_read(path):
        reads.append(path)
        return real_read(path)

    monkeypatch.setattr(mt.MemoryStore, "_read_raw_checked", staticmethod(counted_read))
    pending = handle_pending_subcommand("memory", ["pending"])
    assert pending.count("- add: shared text (already on disk; kept on reject)") == 2
    assert pending.count("- add: shared text (not on disk; pending approval)") == 1
    assert len(reads) == len(set(reads)) == 2
    assert not mt.MemoryStore._path_for("user").exists()


def test_invalid_legacy_target_has_unknown_disk_status(review_home):
    wa.stage_write("memory", {"action": "add", "target": "invalid", "content": "some fact"},
                   summary="legacy invalid target", origin="foreground")
    pending = handle_pending_subcommand("memory", ["pending"])
    assert "- add: some fact (disk status unavailable)" in pending
    assert not (review_home / "memories").exists()
