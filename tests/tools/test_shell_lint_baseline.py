"""External lint reports the actual pre-edit file, not the rewritten file."""

import json
import os
import shutil
from pathlib import Path

import pytest

from tools import file_tools, terminal_tool
from tools.registry import registry


@pytest.mark.parametrize("already_broken", [False, True])
def test_javascript_patch_uses_real_prewrite_lint(tmp_path, monkeypatch, already_broken):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for real JavaScript syntax checks")
    # Supply the available runtime at PM's environment-composition boundary;
    # production lookup and the actual node --check subprocess still execute.
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", lambda env: {
        **env, "PATH": str(Path(node).parent) + os.pathsep + env.get("PATH", ""),
    })
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    task = "shell-lint-baseline"
    path = tmp_path / "sample.js"
    source = "const broken = ;\nconst ok = 1;\n" if already_broken else "const ok = 1;\n"
    path.write_text(source, encoding="utf-8")
    new = "const ok = 2;" if already_broken else "const ok = ;"
    terminal_tool.register_task_env_overrides(task, {"env_type": "local", "cwd": str(tmp_path)})
    try:
        read = json.loads(registry.dispatch("read_file", {"path": str(path)}, task_id=task))
        assert "error" not in read, read
        result = json.loads(registry.dispatch("patch", {
            "path": str(path), "old_string": "const ok = 1;", "new_string": new,
        }, task_id=task))
        assert "error" not in result, result
        assert result["lint"]["status"] == "error", result
        assert ("Pre-existing lint errors" in result["lint"].get("message", "")) is already_broken, result
        assert path.read_text(encoding="utf-8") == source.replace("const ok = 1;", new)
    finally:
        file_tools.clear_file_ops_cache(task)
        terminal_tool.clear_task_env_overrides(task)
