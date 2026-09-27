"""Regression for #124582: the memory tool had no find-and-splice edit, so a
partial edit of one clause of a packed multi-fact entry could only be expressed
as a whole-entry ``replace`` — silently discarding the sibling facts in the same
entry. The ``patch`` action splices ``old_text`` → ``new_text`` inside the
matched entry, preserving the rest of the entry verbatim."""

import json

import pytest

from tools.memory_tool import memory_tool
from tools.memory_tool_store import MemoryStore

PACKED = "todo: item A; item B; item C"


@pytest.fixture
def packed_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    s = MemoryStore(memory_char_limit=500, user_char_limit=300)
    s.add("memory", PACKED)
    return s


class TestPatchSingleOp:
    def test_patch_splices_only_the_matched_span(self, packed_store):
        result = json.loads(memory_tool(
            action="patch", old_text="item B", new_text="item B done", store=packed_store))
        assert result["success"] is True
        assert packed_store._entries_for("memory") == ["todo: item A; item B done; item C"]

    def test_patch_returns_before_and_after(self, packed_store):
        result = json.loads(memory_tool(
            action="patch", old_text="item B", new_text="item B done", store=packed_store))
        assert result["patched_entry_before"] == PACKED
        assert result["patched_entry_after"] == "todo: item A; item B done; item C"

    def test_patch_requires_old_text(self, packed_store):
        result = json.loads(memory_tool(action="patch", new_text="x", store=packed_store))
        assert result["success"] is False
        assert "old_text" in result["error"]

    def test_patch_requires_new_text(self, packed_store):
        result = json.loads(memory_tool(action="patch", old_text="item B", store=packed_store))
        assert result["success"] is False

    def test_patch_unmatched_old_text_fails_without_write(self, packed_store):
        result = json.loads(memory_tool(
            action="patch", old_text="item Z", new_text="whatever", store=packed_store))
        assert result["success"] is False
        assert packed_store._entries_for("memory") == [PACKED]

    def test_patch_multi_entry_match_is_rejected(self, packed_store):
        packed_store.add("memory", "also mentions item B elsewhere")
        result = json.loads(memory_tool(
            action="patch", old_text="item B", new_text="x", store=packed_store))
        assert result["success"] is False
        assert packed_store._entries_for("memory") == [PACKED, "also mentions item B elsewhere"]


class TestPatchStoreLevel:
    def test_store_patch_method(self, packed_store):
        result = packed_store.patch("memory", "item B", "item B done")
        assert result["success"] is True
        assert packed_store._entries_for("memory") == ["todo: item A; item B done; item C"]

    def test_patch_all_occurrences_within_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        s = MemoryStore(memory_char_limit=500)
        s.add("memory", "x: see B and again B")
        result = s.patch("memory", "B", "C")
        assert result["success"] is True
        assert s._entries_for("memory") == ["x: see C and again C"]


class TestPatchBatch:
    def test_patch_op_in_batch(self, packed_store):
        result = packed_store.apply_batch("memory", [
            {"action": "patch", "old_text": "item B", "new_text": "item B done"},
        ])
        assert result["success"] is True
        assert packed_store._entries_for("memory") == ["todo: item A; item B done; item C"]

    def test_patch_op_counts_as_destructive_for_staging(self, packed_store):
        # patch rewrites part of an entry: it must be staged (matched_entry pinned)
        # exactly like replace, never replayed against a newer entry.
        from tools.memory_tool import destructive_ops
        ops = destructive_ops({"action": "batch", "operations": [
            {"action": "patch", "old_text": "item B", "new_text": "x"}]})
        assert [op["action"] for op in ops] == ["patch"]
