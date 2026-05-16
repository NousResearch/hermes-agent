"""Tests for MemoryStore.get_readout and parse_memory_command."""

import pytest

from tools.memory_tool import MemoryStore, parse_memory_command


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
    s = MemoryStore(memory_char_limit=500, user_char_limit=300)
    s.load_from_disk()
    return s


class TestGetReadout:
    def test_empty_stores(self, store):
        data = store.get_readout()
        assert data["memory"]["entries"] == []
        assert data["user"]["entries"] == []
        assert data["memory"]["char_count"] == 0
        assert data["user"]["char_count"] == 0
        assert data["memory"]["pct"] == 0
        assert data["user"]["pct"] == 0

    def test_limits_reflect_constructor(self, store):
        data = store.get_readout()
        assert data["memory"]["char_limit"] == 500
        assert data["user"]["char_limit"] == 300

    def test_populated_memory(self, store):
        store.add("memory", "fact one")
        store.add("memory", "fact two")
        data = store.get_readout()
        assert data["memory"]["entries"] == ["fact one", "fact two"]
        assert data["memory"]["char_count"] > 0
        assert data["user"]["entries"] == []

    def test_percentage_calculation(self, store):
        # ~40% of 500 chars
        store.add("memory", "x" * 200)
        data = store.get_readout()
        assert 35 <= data["memory"]["pct"] <= 45

    def test_percentage_capped_at_100(self, tmp_path, monkeypatch):
        # Write a file directly past the limit so load_from_disk reads it.
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        (tmp_path / "MEMORY.md").write_text("x" * 1000)
        store = MemoryStore(memory_char_limit=500, user_char_limit=300)
        data = store.get_readout()
        assert data["memory"]["pct"] == 100

    def test_reloads_from_disk(self, store, tmp_path):
        from tools.memory_tool import ENTRY_DELIMITER
        store.add("memory", "initial")
        # External writer appends a new entry using the real delimiter.
        mem_file = tmp_path / "MEMORY.md"
        mem_file.write_text(mem_file.read_text() + ENTRY_DELIMITER + "external")
        data = store.get_readout()
        assert "external" in data["memory"]["entries"]


class TestParseMemoryCommand:
    def test_no_args_reads_all(self):
        assert parse_memory_command("") == {"action": "read", "target": "all"}

    def test_whitespace_only_reads_all(self):
        assert parse_memory_command("   ") == {"action": "read", "target": "all"}

    def test_target_memory(self):
        assert parse_memory_command("memory") == {"action": "read", "target": "memory"}

    def test_target_user(self):
        assert parse_memory_command("user") == {"action": "read", "target": "user"}

    def test_case_insensitive(self):
        assert parse_memory_command("MEMORY") == {"action": "read", "target": "memory"}
        assert parse_memory_command("User") == {"action": "read", "target": "user"}

    def test_remove_valid(self):
        result = parse_memory_command("remove memory 3")
        assert result == {"action": "remove", "target": "memory", "index": 3}

    def test_remove_user(self):
        result = parse_memory_command("remove user 1")
        assert result == {"action": "remove", "target": "user", "index": 1}

    def test_remove_too_few_args(self):
        result = parse_memory_command("remove memory")
        assert "error" in result
        assert "Usage" in result["error"]

    def test_remove_invalid_target(self):
        result = parse_memory_command("remove bogus 1")
        assert "error" in result
        assert "bogus" in result["error"]

    def test_remove_non_integer_index(self):
        result = parse_memory_command("remove memory abc")
        assert "error" in result
        assert "abc" in result["error"]

    def test_remove_zero_index_rejected(self):
        result = parse_memory_command("remove memory 0")
        assert "error" in result
        assert "1 or greater" in result["error"]

    def test_remove_negative_index_rejected(self):
        result = parse_memory_command("remove memory -1")
        assert "error" in result

    def test_unknown_subcommand(self):
        result = parse_memory_command("explode")
        assert "error" in result
        assert "explode" in result["error"]
