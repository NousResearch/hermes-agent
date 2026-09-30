"""JSON file writes reject non-JSON constants without changing existing bytes."""

import json

from tools import file_tools, terminal_tool
from tools.registry import registry


def test_json_write_refuses_nonstandard_constants_without_overwriting(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    task = "strict-json-write"
    path = tmp_path / "config.json"
    original = '{"value": 1}\n'
    path.write_text(original, encoding="utf-8")
    terminal_tool.register_task_env_overrides(task, {"env_type": "local", "cwd": str(tmp_path)})
    try:
        read = json.loads(registry.dispatch("read_file", {"path": str(path)}, task_id=task))
        assert "error" not in read, read
        for token in ("NaN", "Infinity", "-Infinity"):
            result = json.loads(registry.dispatch("write_file", {
                "path": str(path), "content": '{"value": ' + token + '}\n',
            }, task_id=task))
            assert "error" in result, result
            assert path.read_text(encoding="utf-8") == original
        valid = '{"value": ["NaN", "Infinity", "-Infinity", 1.5, null]}\n'
        result = json.loads(registry.dispatch("write_file", {
            "path": str(path), "content": valid,
        }, task_id=task))
        assert "error" not in result, result
        assert path.read_text(encoding="utf-8") == valid
    finally:
        file_tools.clear_file_ops_cache(task)
        terminal_tool.clear_task_env_overrides(task)
