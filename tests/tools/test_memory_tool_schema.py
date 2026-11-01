"""Schema-shape tests for the built-in memory tool.

The memory tool previously used ``allOf: [{if: ..., then: {required: ...}}]``
at the top level of ``parameters`` to hint per-action required fields.  That
form was:

  1. Ignored by every provider (Chat Completions doesn't honour ``if/then``
     on function schemas), so it never actually enforced anything.
  2. **Rejected outright by strict backends** — OpenAI's Codex endpoint
     (``chatgpt.com/backend-api/codex``, gpt-5.x) returns
     ``Invalid schema for function 'memory': schema must have type 'object'
     and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'not' at the top level``.

We now rely on the runtime handler (``memory_tool()`` in ``tools/memory_tool.py``)
to validate required fields per action and return actionable error messages.
These tests guard the schema against regressing back to a shape strict
backends reject.
"""

import json

from tools.memory_tool import MEMORY_SCHEMA, MemoryStore
from tools.registry import registry

_FORBIDDEN_TOP_LEVEL_KEYS = ("allOf", "anyOf", "oneOf", "enum", "not")

def test_memory_schema_has_no_forbidden_top_level_combinators():
    """OpenAI's Codex backend rejects these at the top level of parameters."""
    params = MEMORY_SCHEMA["parameters"]
    for key in _FORBIDDEN_TOP_LEVEL_KEYS:
        assert key not in params, (
            f"top-level {key!r} in memory tool parameters will break the "
            "Codex backend (chatgpt.com/backend-api/codex). Per-action "
            "required-field checks belong in the runtime handler, not the schema."
        )


def test_each_memory_field_is_advertised_once_and_legacy_shapes_still_write(tmp_path, monkeypatch):
    """The schema advertises one shape (target + operations[]) with no alias; the handler still
    accepts the top-level single-op shape and the new_text alias models and transcripts send."""
    params = MEMORY_SCHEMA["parameters"]
    assert set(params["properties"]) == {"target", "operations"}
    assert "new_text" not in json.dumps(params)

    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    store = MemoryStore(memory_char_limit=500, user_char_limit=300)
    store.load_from_disk()
    handler = registry.get_entry("memory").handler
    assert json.loads(handler({"action": "add", "target": "memory", "new_text": "alpha fact"}, store=store))["success"]
    batch = {"target": "memory", "operations": [{"action": "patch", "old_text": "alpha", "new_text": "beta"}]}
    assert json.loads(handler(batch, store=store))["success"]
    assert store.memory_entries == ["beta fact"]


def test_advertised_one_item_remove_deletes_the_last_entry_but_a_multi_op_batch_cannot_empty(tmp_path, monkeypatch):
    """operations[] is the only advertised shape, so a one-item remove is how the agent forgets
    the final fact; a larger batch that would empty the store is still refused (#103419)."""
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    store = MemoryStore(memory_char_limit=500, user_char_limit=300)
    store.load_from_disk()
    handler = registry.get_entry("memory").handler
    for fact in ("alpha fact", "beta fact"):
        assert json.loads(handler({"target": "user", "operations": [{"action": "add", "content": fact}]}, store=store))["success"]
    both = [{"action": "remove", "old_text": "alpha"}, {"action": "remove", "old_text": "beta"}]
    refused = json.loads(handler({"target": "user", "operations": both}, store=store))
    assert not refused["success"] and "one-item" in refused["error"]
    assert store.user_entries == ["alpha fact", "beta fact"]
    for old in ("alpha", "beta"):
        assert json.loads(handler({"target": "user", "operations": [{"action": "remove", "old_text": old}]}, store=store))["success"]
    assert store.user_entries == []


def test_one_item_add_that_overflows_returns_the_live_entries_like_the_legacy_add(tmp_path, monkeypatch):
    """The one-item batch is the advertised single add, so its over-budget failure carries
    current_entries: without them the model consolidates against its load-time snapshot."""
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    store = MemoryStore(memory_char_limit=60, user_char_limit=300)
    store.load_from_disk()
    handler = registry.get_entry("memory").handler
    store.add("memory", "written by another session after load")
    result = json.loads(handler({"target": "memory", "operations": [{"action": "add", "content": "x" * 40}]}, store=store))
    assert result["success"] is False
    assert result["current_entries"] == ["written by another session after load"]
    assert store.memory_entries == ["written by another session after load"]


def test_one_item_remove_or_duplicate_add_works_on_a_store_already_over_a_lowered_cap(tmp_path, monkeypatch):
    """Like the legacy remove (no budget check) and duplicate add ("already exists"), a one-item
    batch that does not grow the store is not refused for the store's pre-existing excess."""
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    facts = ["a" * 50, "b" * 50, "c" * 50]
    (tmp_path / "MEMORY.md").write_text("\n§\n".join(facts), encoding="utf-8")
    store = MemoryStore(memory_char_limit=100, user_char_limit=300)  # cap lowered below the store
    store.load_from_disk()
    handler = registry.get_entry("memory").handler
    dup = json.loads(handler({"target": "memory", "operations": [{"action": "add", "content": facts[1]}]}, store=store))
    assert dup["success"] is True
    gone = json.loads(handler({"target": "memory", "operations": [{"action": "remove", "old_text": "c" * 50}]}, store=store))
    assert gone["success"] is True
    assert store.memory_entries == facts[:2]


def test_background_review_one_item_remove_of_the_last_entry_is_staged_not_applied(tmp_path, monkeypatch):
    """A one-item remove may empty a store, but an unattended background review never gets
    that far: the #105921 delete gate stages it, so #103419's USER.md wipe stays closed."""
    from tools.memory_tool import memory_tool
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    store = MemoryStore(memory_char_limit=500, user_char_limit=300)
    store.load_from_disk()
    store.add("user", "Name: Alice")
    token = set_current_write_origin("background_review")
    try:
        result = json.loads(memory_tool(target="user", operations=[{"action": "remove", "old_text": "Alice"}], store=store))
    finally:
        reset_current_write_origin(token)
    assert result["staged"] is True
    assert store.user_entries == ["Name: Alice"]
