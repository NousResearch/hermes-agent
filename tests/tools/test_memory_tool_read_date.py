"""Tests for PR: memory_tool read action (2026-07-11)."""

import json
import pytest


class TestMemoryReadAction:
    @pytest.fixture()
    def store(self, tmp_path, monkeypatch):
        from tools.memory_tool import MemoryStore
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        s = MemoryStore()
        s.load_from_disk()
        s.add("memory", "Project uses pytest.")
        s.add("user", "Name: Alice")
        return s

    def test_read_memory(self, store):
        from tools.memory_tool import memory_tool
        result = json.loads(memory_tool(action="read", target="memory", store=store))
        assert result["success"] is True
        assert "Project uses pytest" in result["content"]

    def test_read_user(self, store):
        from tools.memory_tool import memory_tool
        result = json.loads(memory_tool(action="read", target="user", store=store))
        assert result["success"] is True
        assert "Name: Alice" in result["content"]

    def test_read_empty_store(self, tmp_path, monkeypatch):
        from tools.memory_tool import MemoryStore, memory_tool
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        s = MemoryStore()
        result = json.loads(memory_tool(action="read", target="memory", store=s))
        assert result.get("content", "") == ""
