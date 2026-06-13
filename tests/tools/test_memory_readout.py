"""Tests for MemoryStore.get_readout, parse_memory_show_args, and MemoryStore.from_config."""

import pytest

from tools.memory_tool import MemoryStore, parse_memory_show_args


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


class TestParseMemoryShowArgs:
    def test_no_args_reads_all(self):
        assert parse_memory_show_args("") == {"target": "all"}

    def test_whitespace_only_reads_all(self):
        assert parse_memory_show_args("   ") == {"target": "all"}

    def test_target_memory(self):
        assert parse_memory_show_args("memory") == {"target": "memory"}

    def test_target_user(self):
        assert parse_memory_show_args("user") == {"target": "user"}

    def test_case_insensitive(self):
        assert parse_memory_show_args("MEMORY") == {"target": "memory"}
        assert parse_memory_show_args("User") == {"target": "user"}

    def test_unknown_target_errors(self):
        result = parse_memory_show_args("bogus")
        assert "error" in result
        assert "bogus" in result["error"]

    def test_extra_tokens_ignored_after_target(self):
        # Only the first token is the target; trailing tokens are ignored.
        assert parse_memory_show_args("user extra junk") == {"target": "user"}


class TestFromConfig:
    def test_uses_configured_limits(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"memory": {"memory_char_limit": 999, "user_char_limit": 444}},
        )
        store = MemoryStore.from_config()
        assert store.memory_char_limit == 999
        assert store.user_char_limit == 444

    def test_falls_back_to_defaults_on_missing_config(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
        store = MemoryStore.from_config()
        assert store.memory_char_limit == 2200
        assert store.user_char_limit == 1375

    def test_falls_back_on_load_error(self, monkeypatch):
        def _boom():
            raise RuntimeError("no config")
        monkeypatch.setattr("hermes_cli.config.load_config", _boom)
        store = MemoryStore.from_config()
        assert store.memory_char_limit == 2200
        assert store.user_char_limit == 1375
