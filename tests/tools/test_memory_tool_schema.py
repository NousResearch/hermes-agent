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
