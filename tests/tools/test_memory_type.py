"""B3 tests for memory type write/store/read/display/migration behavior."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tools.memory_tool import (
    DEFAULT_MEMORY_TYPE,
    MEMORY_SCHEMA,
    MemoryStore,
    memory_tool,
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_memory_types.py"
_SPEC = importlib.util.spec_from_file_location("migrate_memory_types", SCRIPT_PATH)
migrate_memory_types = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(migrate_memory_types)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=500, user_char_limit=300)
    s.load_from_disk()
    return s


class TestMemoryTypes:
    def test_schema_exposes_optional_type_parameter(self):
        params = MEMORY_SCHEMA["parameters"]

        assert "type" in params["properties"]
        assert "type" not in params["required"]

    def test_legacy_call_without_type_still_defaults_to_uncategorized(self, store):
        result = json.loads(
            memory_tool(action="add", target="memory", content="Legacy-safe note", store=store)
        )

        assert result["success"] is True
        assert result["typed_entries"][0]["type"] == DEFAULT_MEMORY_TYPE

    def test_typed_write_persists_across_reload(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.MEMORY_DIR", tmp_path)
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)

        store1 = MemoryStore(memory_char_limit=500, user_char_limit=300)
        store1.load_from_disk()
        store1.add("memory", "Project uses PostgreSQL", memory_type="project")

        store2 = MemoryStore(memory_char_limit=500, user_char_limit=300)
        store2.load_from_disk()
        result = store2.read("memory")

        assert result["typed_entries"][0]["type"] == "project"
        assert (tmp_path / "MEMORY.types.json").exists()

    def test_read_results_include_type_display(self, store):
        store.add("memory", "Do not append a summary footer", memory_type="feedback")

        result = json.loads(memory_tool(action="read", target="memory", store=store))

        assert result["success"] is True
        assert result["typed_entries"][0]["type"] == "feedback"
        assert result["typed_entries"][0]["display"].startswith("[feedback] ")

    def test_type_filter_is_effective(self, store):
        store.add("memory", "Project uses PostgreSQL", memory_type="project")
        store.add("memory", "Do not append a summary footer", memory_type="feedback")
        store.add("memory", "Legacy note with no type")

        result = json.loads(
            memory_tool(
                action="read",
                target="memory",
                memory_type="project",
                store=store,
            )
        )

        assert result["success"] is True
        assert result["type_filter"] == "project"
        assert result["entry_count"] == 1
        assert result["total_entry_count"] == 3
        assert result["typed_entries"][0]["content"] == "Project uses PostgreSQL"

    def test_read_without_type_includes_legacy_uncategorized_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.MEMORY_DIR", tmp_path)
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        (tmp_path / "MEMORY.md").write_text("Legacy entry from old data", encoding="utf-8")

        store = MemoryStore(memory_char_limit=500, user_char_limit=300)
        store.load_from_disk()
        result = store.read("memory")

        assert result["success"] is True
        assert result["entry_count"] == 1
        assert result["typed_entries"][0]["type"] == DEFAULT_MEMORY_TYPE
        assert result["typed_entries"][0]["content"] == "Legacy entry from old data"


class TestMemoryTypeMigration:
    def test_dry_run_outputs_reasonable_suggestions(self, store, capsys):
        store.add("memory", "Project uses PostgreSQL and pytest")
        store.add("memory", "Do not append a summary footer")

        exit_code = migrate_memory_types.main(["--dry-run", "--target", "memory"])
        stdout = capsys.readouterr().out

        assert exit_code == 0
        assert '"mode": "dry-run"' in stdout
        assert '"suggested_type": "project"' in stdout
        assert '"suggested_type": "feedback"' in stdout
