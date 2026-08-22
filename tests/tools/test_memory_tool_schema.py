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

from tools.memory_tool import MEMORY_SCHEMA


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


def test_memory_schema_is_json_serializable():
    json.dumps(MEMORY_SCHEMA)


def test_memory_schema_routes_facts_skills_and_history():
    description = MEMORY_SCHEMA["description"]
    assert "MEMORY/USER" in description
    assert "skill_manage" in description
    assert "session_search" in description
    assert "create or patch" in description
    assert "autonomous routing" in description
    assert "explicit user-authored memory writes are still accepted" in description
    assert "user_requested=true" in description
    assert "Do not write those to memory" not in description
    assert MEMORY_SCHEMA["parameters"]["properties"]["user_requested"]["default"] is False


def test_memory_store_accepts_explicit_task_progress_writes(tmp_path, monkeypatch):
    """Guidance routes autonomous writes; MemoryStore stays permissive."""
    from tools.memory_tool import MemoryStore

    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    store = MemoryStore(memory_char_limit=2000, user_char_limit=1000)
    store.load_from_disk()
    result = store.add("memory", "Phase 3 done — submitted PR #4242 at abcdef123")
    assert result["success"] is True
    assert any("Phase 3 done" in entry for entry in store.memory_entries)
