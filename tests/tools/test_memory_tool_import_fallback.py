"""Regression tests for memory-tool import fallbacks."""

import builtins
import importlib
import sys

from hermes_state import SessionDB
from tools.registry import registry


def test_memory_tool_imports_without_fcntl(monkeypatch, tmp_path):
    """DB-backed MemoryStore doesn't need fcntl — confirm it works regardless."""
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fcntl":
            raise ImportError("simulated missing fcntl")
        return original_import(name, globals, locals, fromlist, level)

    registry.deregister("memory")
    monkeypatch.delitem(sys.modules, "tools.memory_tool", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    memory_tool = importlib.import_module("tools.memory_tool")

    db = SessionDB(db_path=tmp_path / "fallback_state.db")
    store = memory_tool.MemoryStore(session_db=db, memory_char_limit=200, user_char_limit=200)
    store.load_from_db()
    result = store.add("memory", "fact learned during import fallback test")

    assert registry.get_entry("memory") is not None
    assert result["success"] is True
