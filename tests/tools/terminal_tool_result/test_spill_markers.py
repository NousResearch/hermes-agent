"""The overflow spill file holds the command's output, not the session wrapper's bookkeeping.

The output collector tees the raw stream to the spill file, and the raw stream ends with the
wrapper's PIPESTATUS line and cwd marker line. ``_extract_cwd_from_output`` strips both from the
in-memory output only, so ``read_file`` on the spill showed ``__HERMES_PIPE_..`` and
``__HERMES_CWD_..`` lines as if the command had printed them, and a large JSON document saved
there no longer parsed.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.terminal_tool import terminal_tool
from tools.terminal_tool_result import finalize_foreground_result

CWD = "__HERMES_CWD_abc12345__"
PIPE = "__HERMES_PIPE_abc12345__"
ENV = SimpleNamespace(_cwd_marker=CWD, _pipe_marker=PIPE, cwd="/work")


@pytest.fixture
def small_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_constants import hermes_home_key
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {hermes_home_key(): {
        "max_bytes": 50_000, "max_lines": 2_000, "max_line_length": 2_000,
    }})
    return tmp_path


def _spill(result):
    assert result["exit_code"] == 0
    assert result["output_total_chars"] > 50_000
    return Path(result["full_output_path"]).read_text()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("pipe", ["", " | cat"], ids=["plain", "piped"])
def test_a_large_json_spill_parses(small_cap, pipe):
    command = "python3 -c \"import json; print(json.dumps({'id': 'item', 'data': 'x' * 60000}))\"" + pipe
    spill = _spill(json.loads(terminal_tool(command, task_id="t-spill-markers")))
    assert "__HERMES_CWD_" not in spill
    assert "__HERMES_PIPE_" not in spill
    assert json.loads(spill) == {"id": "item", "data": "x" * 60_000}


def _finalize(tmp_path, text, cwd_observed):
    """Run a raw result whose spill file holds *text* through the tool's result step."""
    path = tmp_path / "spill.log"
    path.write_text(text)
    result = {"output": "preview", "returncode": 0, "full_output_path": str(path),
              "output_total_chars": len(text)}
    if cwd_observed:
        result.update(cwd_observed=True, cwd="/work")
    finalize_foreground_result(
        command="printf data", result=result, env=ENV, env_type="local",
        effective_task_id="spill-markers", task_id=None, session_id=None,
        session_key="spill-markers", workdir="/work", command_cwd="/work", approval_note=None)
    return path.read_text()


def test_the_wrapper_suffix_is_removed(small_cap, tmp_path):
    text = f"command output\n\n{PIPE}0 0{PIPE}\n\n{CWD}/work{CWD}\n"
    assert _finalize(tmp_path, text, True) == "command output\n"


def test_marker_shaped_output_before_the_suffix_is_kept(small_cap, tmp_path):
    body = f"command output\n{CWD}/echoed{CWD}\nmore output\n"
    text = f"{body}\n{CWD}/work{CWD}\n"
    assert _finalize(tmp_path, text, True) == body


def test_a_spill_whose_command_reported_no_cwd_is_left_alone(small_cap, tmp_path):
    # An interrupted or capped stream has no real suffix; a marker-shaped tail is command output.
    text = f"command output\n{CWD}/data{CWD}\n"
    assert _finalize(tmp_path, text, False) == text
