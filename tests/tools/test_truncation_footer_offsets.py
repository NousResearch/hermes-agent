"""Recovery hints must include the first omitted text, even when a cut is mid-line."""

import json
import re
from pathlib import Path

import pytest

from tools import delegate_tool_results, file_tools, web_tools_truncate
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.registry import registry


@pytest.mark.parametrize("source", ["web", "delegation"])
@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param("x" * 1600, id="single-line"),
        pytest.param("intro\n" + "x" * 1600, id="partial-line"),
        pytest.param("x" * 1000 + "\n" + "x" * 600, id="snapped-line"),
        pytest.param("x" * 1500 + "\n", id="newline-at-cut"),
    ],
)
def test_recovery_hint_reads_first_omitted_text(source, prefix, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    marker = "FIRST_OMITTED_TEXT"
    body = prefix + marker + "y" * 4000 + "\n"
    if source == "web":
        output, _ = web_tools_truncate._truncate_with_footer(
            body, "https://example.com/long-page", 2000
        )
    else:
        output, _ = delegate_tool_results._trim_summary_with_footer(body, 2000, 0)

    assert marker not in output
    hint = re.search(r'read_file path="([^"]+)" offset=(\d+) limit=(\d+)', output)
    assert hint is not None
    path, offset, limit = hint.groups()
    assert Path(path).is_relative_to(tmp_path)
    assert Path(path).read_text(encoding="utf-8") == body

    env = LocalEnvironment(cwd=str(tmp_path))
    monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id: ShellFileOperations(env))
    try:
        result = json.loads(registry.dispatch(
            "read_file", {"path": path, "offset": int(offset), "limit": int(limit)},
            task_id="truncation-recovery",
        ))
        assert marker in result.get("content", ""), result
    finally:
        env.cleanup()
