import json
import sqlite3

from agent.memory_manager import MemoryManager, inject_memory_provider_tools
from plugins.memory.turso import TursoMemoryProvider


def test_turso_provider_crud_and_prefetch(tmp_path):
    provider = TursoMemoryProvider({"db_path": str(tmp_path / "memory.db"), "top_k": 3, "min_similarity": 0.0})
    provider.initialize("sess-1", hermes_home=str(tmp_path))

    added = json.loads(provider.handle_tool_call("turso_memory_add", {
        "content": "User prefers Rust for backend services",
        "kind": "preference",
    }))
    assert added["success"] is True
    memory_id = added["memory"]["id"]

    search = json.loads(provider.handle_tool_call("turso_memory_search", {"query": "Rust backend"}))
    assert search["success"] is True
    assert search["results"][0]["id"] == memory_id

    context = provider.prefetch("Which backend language should I use?")
    assert "Turso Memory" in context
    assert "Rust" in context

    updated = json.loads(provider.handle_tool_call("turso_memory_update", {
        "memory_id": memory_id,
        "content": "User prefers Go for backend services",
    }))
    assert updated["success"] is True

    deleted = json.loads(provider.handle_tool_call("turso_memory_delete", {"memory_id": memory_id}))
    assert deleted["success"] is True

    empty = json.loads(provider.handle_tool_call("turso_memory_search", {"query": "backend services"}))
    assert empty["results"] == []


def test_turso_provider_uses_profile_scoped_default_db(tmp_path):
    provider = TursoMemoryProvider()
    provider.initialize("sess-1", hermes_home=str(tmp_path))

    provider.handle_tool_call("turso_memory_add", {"content": "Profile-local memory"})

    db_path = tmp_path / "turso-memory.db"
    assert db_path.exists()
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_turso_provider_mirrors_builtin_memory_writes(tmp_path):
    provider = TursoMemoryProvider({"db_path": str(tmp_path / "memory.db"), "min_similarity": 0.0})
    provider.initialize("sess-1", hermes_home=str(tmp_path))

    provider.on_memory_write("add", "user", "User likes terse answers", {"session_id": "sess-1"})

    results = json.loads(provider.handle_tool_call("turso_memory_search", {"query": "terse answers"}))
    assert results["success"] is True
    assert results["results"][0]["kind"] == "preference"
    assert results["results"][0]["target"] == "user"


def test_turso_provider_auto_capture_is_opt_in(tmp_path):
    off = TursoMemoryProvider({"db_path": str(tmp_path / "off.db"), "auto_capture": False})
    off.initialize("sess-1", hermes_home=str(tmp_path))
    off.sync_turn("Remember this project uses uv", "Got it", session_id="sess-1")
    assert json.loads(off.handle_tool_call("turso_memory_search", {"query": "project uv"}))["results"] == []

    on = TursoMemoryProvider({"db_path": str(tmp_path / "on.db"), "auto_capture": True, "min_similarity": 0.0})
    on.initialize("sess-1", hermes_home=str(tmp_path))
    on.sync_turn("Remember this project uses uv", "Got it, I will use uv.", session_id="sess-1")
    assert json.loads(on.handle_tool_call("turso_memory_search", {"query": "project uv"}))["results"]


def test_turso_provider_save_config_keeps_secrets_out_of_json(tmp_path):
    provider = TursoMemoryProvider()
    provider.save_config({
        "db_path": "$HERMES_HOME/turso-memory.db",
        "sync_enabled": "true",
        "auth_token": "secret",
        "database_url": "libsql://example.turso.io",
    }, str(tmp_path))

    saved = json.loads((tmp_path / "turso.json").read_text(encoding="utf-8"))
    assert saved["sync_enabled"] == "true"
    assert "database_url" not in saved
    assert "auth_token" not in saved


def test_turso_provider_integrates_with_memory_manager_tools(tmp_path):
    provider = TursoMemoryProvider({"db_path": str(tmp_path / "memory.db")})
    mgr = MemoryManager()
    mgr.add_provider(provider)
    mgr.initialize_all("sess-1", hermes_home=str(tmp_path))

    class Agent:
        enabled_toolsets = ["memory"]
        tools = [{"type": "function", "function": {"name": "memory"}}]
        valid_tool_names = {"memory"}
        _memory_manager = mgr

    added = inject_memory_provider_tools(Agent)
    assert added == 5
    assert "turso_memory_search" in Agent.valid_tool_names

    result = json.loads(mgr.handle_tool_call("turso_memory_add", {"content": "Hermes uses profile-scoped memory"}))
    assert result["success"] is True
