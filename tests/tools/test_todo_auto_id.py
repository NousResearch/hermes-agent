"""Tests for the dict-without-id repair in `_validate` + `_resolve_auto_ids`.

Dr-Apex-Metabolic's `MiniMax-M3` model was emitting `todos` as a bare dict (not a list).
`coerce_tool_args` wrapped it as `[bare_dict]`, then the bare dict lacked `id`, and the
deferred validator rejected the call. The wrapper, however, has always been permissive
about missing fields — these tests pin the new contract: dict items without an id get a
unique synthetic `auto_N`; non-dict items still get `"?"` (the existing contract).
"""

from __future__ import annotations

import json

from tools.todo_tool import TodoStore, todo_tool


class TestAutoIdRepair:
    def test_dict_without_id_gets_auto_id(self):
        store = TodoStore()
        result = store.write([{"content": "do thing", "status": "pending"}])
        assert len(result) == 1
        assert result[0]["id"].startswith("auto_"), f"got {result[0]['id']!r}"
        assert result[0]["content"] == "do thing"
        assert result[0]["status"] == "pending"

    def test_two_dicts_without_ids_get_distinct_auto_ids(self):
        store = TodoStore()
        result = store.write([
            {"content": "first", "status": "pending"},
            {"content": "second", "status": "pending"},
        ])
        ids = [item["id"] for item in result]
        assert ids[0].startswith("auto_") and ids[1].startswith("auto_")
        assert ids[0] != ids[1]

    def test_explicit_id_wins_over_auto_id(self):
        store = TodoStore()
        result = store.write([
            {"id": "real", "content": "real", "status": "pending"},
            {"content": "auto", "status": "pending"},
        ])
        ids = {item["id"]: item["content"] for item in result}
        assert ids.get("real") == "real"
        assert any(k.startswith("auto_") and v == "auto" for k, v in ids.items())

    def test_existing_auto_ids_get_incremented_on_next_write(self):
        # Replace mode is the default: the second write wipes the first item.
        # The contract we test is "second auto-id must not collide with the first",
        # because reusing auto_1 would silently overwrite the prior item's identity
        # in callers that key off the id.
        store = TodoStore()
        store.write([{"content": "x", "status": "pending"}])  # auto_1
        store.write([{"content": "y", "status": "pending"}])  # auto_2 (must not be auto_1)
        ids = [item["id"] for item in store.read()]
        assert ids == ["auto_2"]

    def test_merge_mode_preserves_existing_auto_id(self):
        # The same auto-id increment contract must hold under merge mode: a follow-up
        # merge that adds a no-id item picks an auto_N above every existing auto_N.
        store = TodoStore()
        store.write([{"content": "x", "status": "pending"}])  # auto_1
        store.write([{"content": "y", "status": "pending"}], merge=True)  # auto_2
        ids = sorted(item["id"] for item in store.read())
        assert ids == ["auto_1", "auto_2"]

    def test_dedupe_collision_keeps_user_id_when_explicit_collides_with_auto(self):
        """If the user writes 'auto_1' explicitly and then a no-id item, the user value wins."""
        store = TodoStore()
        store.write([
            {"id": "auto_1", "content": "user-supplied auto_1", "status": "pending"},
            {"content": "should be auto_2", "status": "pending"},
        ])
        items = {item["id"]: item["content"] for item in store.read()}
        assert items["auto_1"] == "user-supplied auto_1"
        assert items["auto_2"] == "should be auto_2"

    def test_via_todo_tool_end_to_end(self):
        store = TodoStore()
        out = json.loads(todo_tool(todos=[{"content": "z", "status": "pending"}], store=store))
        items = out["todos"]
        assert len(items) == 1
        assert items[0]["id"].startswith("auto_")
        assert items[0]["content"] == "z"
        assert out["summary"]["pending"] == 1


class TestNonDictStillQuestionMark:
    """Existing contract: a *non-dict* item (e.g. the string 'garbage') still maps to '?'.

    The auto-id repair is strictly for dict items missing `id`. Strings, ints, None
    inside the list are not `id`-repairable because they were never meant to be items.
    """

    def test_string_item_still_gets_question_mark(self):
        store = TodoStore()
        result = store.write(["not-a-dict"])
        assert result[0]["id"] == "?"
        assert result[0]["content"] == "(invalid item)"

    def test_int_item_still_gets_question_mark(self):
        store = TodoStore()
        result = store.write([42])
        assert result[0]["id"] == "?"