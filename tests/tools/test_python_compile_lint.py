"""File write lint agrees with Python compilation without executing the file."""

import json

from tools import file_tools, terminal_tool
from tools.registry import registry


def test_python_write_reports_compilation_errors_without_executing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    task = "python-compile-lint"
    terminal_tool.register_task_env_overrides(task, {"env_type": "local", "cwd": str(tmp_path)})
    sources = {"return.py": "return 42\n", "continue.py": "continue\n", "valid.py": "raise RuntimeError('must not execute')\n"}
    try:
        for name, source in sources.items():
            path = tmp_path / name
            try:
                compile(source, str(path), "exec", dont_inherit=True)
                compilable = True
            except SyntaxError:
                compilable = False
            result = json.loads(registry.dispatch("write_file", {
                "path": str(path), "content": source,
            }, task_id=task))
            assert "error" not in result, result
            assert path.read_text(encoding="utf-8") == source
            assert (result["lint"]["status"] == "ok") is compilable, result
            assert not list(tmp_path.glob("__pycache__/*.pyc"))
    finally:
        file_tools.clear_file_ops_cache(task)
        terminal_tool.clear_task_env_overrides(task)
