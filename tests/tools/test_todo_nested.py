"""Tests for nested subtasks in the todo tool (parent field).

Covers: validation of the parent field, dangling/cyclic parent
sanitization, merge-mode parent updates, tree-aware post-compression
injection, and the flat-only reorder guard.
"""

import json

from tools.todo_tool import TodoStore, todo_tool


def _item(i, status="pending", parent=None, content=None):
    d = {"id": i, "content": content or f"task {i}", "status": status}
    if parent is not None:
        d["parent"] = parent
    return d


class TestParentValidation:
    def test_parent_preserved(self):
        store = TodoStore()
        items = store.write([_item("a"), _item("a1", parent="a")])
        assert items[1]["parent"] == "a"

    def test_self_parent_dropped(self):
        store = TodoStore()
        items = store.write([_item("a", parent="a")])
        assert "parent" not in items[0]

    def test_dangling_parent_dropped(self):
        store = TodoStore()
        items = store.write([_item("a", parent="ghost")])
        assert "parent" not in items[0]

    def test_cycle_broken(self):
        store = TodoStore()
        items = store.write([_item("a", parent="b"), _item("b", parent="a")])
        # At least one link removed; no item can walk a loop
        by_id = {i["id"]: i for i in items}
        for item in items:
            seen = set()
            node = item
            while node.get("parent"):
                assert node["parent"] not in seen
                seen.add(node["id"])
                node = by_id[node["parent"]]

    def test_empty_parent_omitted(self):
        store = TodoStore()
        items = store.write([_item("a", parent="")])
        assert "parent" not in items[0]


class TestMergeParent:
    def test_merge_sets_parent(self):
        store = TodoStore()
        store.write([_item("a"), _item("b")])
        items = store.write([{"id": "b", "parent": "a"}], merge=True)
        by_id = {i["id"]: i for i in items}
        assert by_id["b"]["parent"] == "a"

    def test_merge_clears_parent_with_empty_string(self):
        store = TodoStore()
        store.write([_item("a"), _item("b", parent="a")])
        items = store.write([{"id": "b", "parent": ""}], merge=True)
        by_id = {i["id"]: i for i in items}
        assert "parent" not in by_id["b"]

    def test_merge_new_child_appended(self):
        store = TodoStore()
        store.write([_item("a")])
        items = store.write([_item("a2", parent="a")], merge=True)
        assert items[-1] == {"id": "a2", "content": "task a2", "status": "pending", "parent": "a"}


class TestNestedInjection:
    def test_children_indented(self):
        store = TodoStore()
        store.write([
            _item("wp1", status="in_progress"),
            _item("t1", parent="wp1"),
            _item("t2", parent="wp1"),
        ])
        text = store.format_for_injection()
        assert text is not None
        lines = text.split("\n")
        assert lines[1].startswith("- [>] wp1.")
        assert lines[2].startswith("  - [ ] t1.")
        assert lines[3].startswith("  - [ ] t2.")

    def test_completed_parent_kept_when_child_active(self):
        store = TodoStore()
        store.write([
            _item("wp1", status="completed"),
            _item("t1", status="pending", parent="wp1"),
        ])
        text = store.format_for_injection()
        assert text is not None
        assert "wp1" in text  # parent context survives
        assert "[x]" in text
        assert "  - [ ] t1." in text

    def test_finished_subtree_omitted(self):
        store = TodoStore()
        store.write([
            _item("wp1", status="completed"),
            _item("t1", status="completed", parent="wp1"),
            _item("wp2", status="pending"),
        ])
        text = store.format_for_injection()
        assert text is not None
        assert "wp1" not in text
        assert "t1" not in text
        assert "wp2" in text

    def test_all_finished_returns_none(self):
        store = TodoStore()
        store.write([
            _item("a", status="completed"),
            _item("a1", status="cancelled", parent="a"),
        ])
        assert store.format_for_injection() is None

    def test_flat_injection_unchanged(self):
        store = TodoStore()
        store.write([_item("a", status="in_progress"), _item("b")])
        text = store.format_for_injection()
        assert text is not None
        assert text.split("\n")[1] == "- [>] a. task a (in_progress)"


class TestOrderGuard:
    def test_nested_list_keeps_authored_order(self):
        store = TodoStore()
        items = store.write([
            _item("a", status="pending"),
            _item("b", status="in_progress"),
            _item("b1", parent="b"),
        ])
        assert [i["id"] for i in items] == ["a", "b", "b1"]

    def test_flat_list_still_reordered(self):
        store = TodoStore()
        items = store.write([
            _item("a", status="pending"),
            _item("b", status="in_progress"),
        ])
        assert [i["id"] for i in items] == ["b", "a"]


class TestToolRoundTrip:
    def test_parent_survives_json(self):
        store = TodoStore()
        out = todo_tool(todos=[_item("a"), _item("a1", parent="a")], store=store)
        data = json.loads(out)
        assert data["todos"][1]["parent"] == "a"
        assert data["summary"]["total"] == 2

    def test_hydration_replay_preserves_parent(self):
        # Simulate _hydrate_todo_store: write the previous tool result's
        # todos array back into a fresh store in replace mode.
        store = TodoStore()
        out = json.loads(todo_tool(todos=[_item("a"), _item("a1", parent="a")], store=store))
        fresh = TodoStore()
        replayed = fresh.write(out["todos"], merge=False)
        assert replayed[1]["parent"] == "a"
