import json

from agent.evolution_log import read_events
from tools.memory_tool import MemoryStore, memory_tool


def _enable_evolution(home):
    (home / "config.yaml").write_text(
        "evolution:\n"
        "  enabled: true\n"
        "  record_diff: true\n"
        "  redact: true\n"
        "  max_diff_chars: 20000\n",
        encoding="utf-8",
    )


def _store():
    store = MemoryStore(memory_char_limit=1000, user_char_limit=1000)
    store.load_from_disk()
    return store


def test_memory_add_records_evolution_event_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    store = _store()

    result = json.loads(
        memory_tool(
            "add",
            target="user",
            content="User prefers example output.",
            store=store,
            summary="Recorded example preference",
            reason="User stated a durable response preference.",
        )
    )

    assert result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "memory.add"
    assert event["source_tool"] == "memory"
    assert event["target"] == "memories/USER.md"
    assert event["target_kind"] == "memory"
    assert event["target_name"] == "user"
    assert event["summary"] == "Recorded example preference"
    assert event["reason"] == "User stated a durable response preference."
    assert "--- before" in event["diff"]
    assert "+++ after" in event["diff"]
    assert "+User prefers example output." in event["diff"]


def test_memory_replace_and_remove_record_evolution_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    store = _store()
    store.add("memory", "Project uses Make.")

    replace_result = json.loads(
        memory_tool(
            "replace",
            target="memory",
            old_text="Make",
            content="Project uses uv.",
            store=store,
            summary="Updated project tool preference",
        )
    )
    remove_result = json.loads(
        memory_tool(
            "remove",
            target="memory",
            old_text="uv",
            store=store,
            summary="Removed stale project tool preference",
        )
    )

    assert replace_result["success"] is True
    assert remove_result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert [event["type"] for event in events] == ["memory.replace", "memory.remove"]
    assert events[0]["target"] == "memories/MEMORY.md"
    assert events[0]["target_name"] == "memory"
    assert "-Project uses Make." in events[0]["diff"]
    assert "+Project uses uv." in events[0]["diff"]
    assert "-Project uses uv." in events[1]["diff"]


def test_memory_batch_records_one_evolution_event_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    store = _store()
    store.add("memory", "Project uses Make.")

    result = json.loads(
        memory_tool(
            target="memory",
            operations=[
                {
                    "action": "replace",
                    "old_text": "Make",
                    "content": "Project uses uv.",
                },
                {"action": "add", "content": "Run focused pytest after each change."},
            ],
            store=store,
            summary="Updated project memory in batch",
            reason="Batch memory consolidation should be observable.",
        )
    )

    assert result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "memory.batch"
    assert event["target"] == "memories/MEMORY.md"
    assert event["target_name"] == "memory"
    assert event["summary"] == "Updated project memory in batch"
    assert event["reason"] == "Batch memory consolidation should be observable."
    assert "-Project uses Make." in event["diff"]
    assert "+Project uses uv." in event["diff"]
    assert "+Run focused pytest after each change." in event["diff"]


def test_failed_memory_mutation_records_no_evolution_event(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    store = _store()

    result = json.loads(
        memory_tool(
            "replace",
            target="memory",
            old_text="missing",
            content="New content",
            store=store,
        )
    )

    assert result["success"] is False
    events, warnings = read_events()
    assert warnings == []
    assert events == []


def test_duplicate_memory_add_records_no_evolution_event(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    store = _store()
    store.add("user", "User prefers concise output.")

    result = json.loads(
        memory_tool(
            "add",
            target="user",
            content="User prefers concise output.",
            store=store,
        )
    )

    assert result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert events == []


def test_apply_memory_pending_records_evolution_event_when_enabled(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    store = _store()

    from tools.memory_tool import apply_memory_pending

    result = apply_memory_pending(
        {
            "action": "add",
            "target": "user",
            "content": "User prefers approved memory writes.",
            "summary": "Approved staged memory write",
            "reason": "Write approval replay should preserve observability.",
        },
        store,
    )

    assert result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "memory.add"
    assert event["target"] == "memories/USER.md"
    assert event["target_name"] == "user"
    assert event["summary"] == "Approved staged memory write"
    assert event["reason"] == "Write approval replay should preserve observability."
    assert "+User prefers approved memory writes." in event["diff"]


def test_staged_memory_replay_preserves_summary_and_reason_in_evolution_event(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "evolution:\n"
        "  enabled: true\n"
        "  record_diff: true\n"
        "  redact: true\n"
        "  max_diff_chars: 20000\n"
        "memory:\n"
        "  write_approval: true\n",
        encoding="utf-8",
    )
    store = _store()

    from tools import write_approval as wa
    from tools.memory_tool import apply_memory_pending

    staged = json.loads(
        memory_tool(
            "add",
            target="user",
            content="User prefers approved staged memory.",
            store=store,
            summary="Staged memory summary",
            reason="Staged memory reason",
        )
    )
    pending = wa.get_pending("memory", staged["pending_id"])
    result = apply_memory_pending(pending["payload"], store)

    assert result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert len(events) == 1
    assert events[0]["type"] == "memory.add"
    assert events[0]["summary"] == "Staged memory summary"
    assert events[0]["reason"] == "Staged memory reason"


def test_staged_memory_batch_replay_preserves_summary_and_reason_in_evolution_event(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "evolution:\n"
        "  enabled: true\n"
        "  record_diff: true\n"
        "  redact: true\n"
        "  max_diff_chars: 20000\n"
        "memory:\n"
        "  write_approval: true\n",
        encoding="utf-8",
    )
    store = _store()

    from tools import write_approval as wa
    from tools.memory_tool import apply_memory_pending

    staged = json.loads(
        memory_tool(
            target="memory",
            operations=[{"action": "add", "content": "Batch staged memory entry."}],
            store=store,
            summary="Staged memory batch summary",
            reason="Staged memory batch reason",
        )
    )
    pending = wa.get_pending("memory", staged["pending_id"])
    result = apply_memory_pending(pending["payload"], store)

    assert result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert len(events) == 1
    assert events[0]["type"] == "memory.batch"
    assert events[0]["summary"] == "Staged memory batch summary"
    assert events[0]["reason"] == "Staged memory batch reason"


def test_memory_registry_passes_summary_and_reason_to_evolution_event(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _enable_evolution(tmp_path)
    store = _store()

    from tools.registry import registry

    result = json.loads(
        registry.dispatch(
            "memory",
            {
                "action": "add",
                "target": "user",
                "content": "User prefers registry dispatch.",
                "summary": "Registry summary",
                "reason": "Registry reason",
            },
            store=store,
        )
    )

    assert result["success"] is True
    events, warnings = read_events()
    assert warnings == []
    assert events[0]["summary"] == "Registry summary"
    assert events[0]["reason"] == "Registry reason"
