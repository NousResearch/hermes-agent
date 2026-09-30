"""Replacing every normalized match preserves each region's untouched Unicode."""

import json

from tools import file_tools, terminal_tool
from tools.registry import registry


def test_replace_all_preserves_each_matches_unicode(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = tmp_path / "notes.txt"
    original = "Price ‘old’—ready\nPrice 'old'—ready\n"
    path.write_text(original, encoding="utf-8")
    task = "unicode-replace-all"
    terminal_tool.register_task_env_overrides(task, {"env_type": "local", "cwd": str(tmp_path)})
    try:
        read = json.loads(registry.dispatch("read_file", {"path": str(path)}, task_id=task))
        assert "error" not in read
        result = json.loads(registry.dispatch("patch", {
            "mode": "replace", "path": str(path), "old_string": "Price 'old'--ready",
            "new_string": "Price 'new'--ready", "replace_all": True,
        }, task_id=task))
        assert "error" not in result, result
        assert path.read_text(encoding="utf-8") == original.replace("old", "new")
    finally:
        file_tools.clear_file_ops_cache(task)
        terminal_tool.clear_task_env_overrides(task)
