"""A capped spill must be advertised as partial, not full output (#109757).

The terminal collector tees overflow to a spill file with a hard cap (``_SPILL_CAP_CHARS``);
execute_code spills stdout with its own cap (``MAX_SPILLED_STDOUT_BYTES``). When a cap is hit,
or a spill write/close fails, the saved file holds only a prefix of the stream, so the
structured result and the model-facing note must not claim complete recovery. The terminal
test drives the real collector, native finalizer and foreground result formatter; nothing is
patched except the cap/close itself.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.environments.base_output import (
    _BoundedOutputCollector,
    _finalize_wait_result,
)
from tools.terminal_tool_result import finalize_foreground_result


def _finalize_collector_output(collector, tmp_path):
    raw = _finalize_wait_result(collector, collector.render(), 0)
    return json.loads(
        finalize_foreground_result(
            command="printf example",
            result=raw,
            env=SimpleNamespace(cwd=str(tmp_path)),
            env_type="local",
            effective_task_id="fixture",
            task_id="fixture",
            session_id=None,
            session_key="fixture",
            workdir=None,
            command_cwd=str(tmp_path),
            approval_note=None,
        )
    )


def _hit_cap(collector):
    collector._SPILL_CAP_CHARS = 500
    collector.append("abc " * 1000)
    return "[spill capped"


def _fail_close(collector):
    collector.append("abc " * 100)
    collector._spill_fh.close = lambda: (_ for _ in ()).throw(OSError("disk gone"))
    return None


@pytest.mark.parametrize("make_partial", [_hit_cap, _fail_close], ids=["cap-hit", "close-failed"])
def test_partial_terminal_spill_is_not_advertised_as_full_output(tmp_path, make_partial):
    collector = _BoundedOutputCollector(100, tmp_path / "partial.log")
    marker = make_partial(collector)
    result = _finalize_collector_output(collector, tmp_path)

    assert result["full_output_capped"] is True
    note = result["truncation_note"].lower()
    assert "full output" not in note and "capped" in note
    if marker:
        assert marker in Path(result["full_output_path"]).read_text(encoding="utf-8")

    # Control: an uncapped spill holds the whole stream and keeps the full-output note.
    full = _BoundedOutputCollector(100, tmp_path / "full.log")
    source = "abc " * 100
    full.append(source)
    full_result = _finalize_collector_output(full, tmp_path)
    assert Path(full_result["full_output_path"]).read_text(encoding="utf-8") == source
    assert "full_output_capped" not in full_result
    assert "Full output" in full_result["truncation_note"]


def test_capped_execute_code_stdout_spill_is_flagged_incomplete(tmp_path, monkeypatch):
    from tools import code_execution_tool as cet

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cet, "MAX_STDOUT_BYTES", 100)
    monkeypatch.setattr(cet, "MAX_SPILLED_STDOUT_BYTES", 500)

    _, capped = cet._truncate_stdout_text("x" * 1000)
    assert capped["stdout_spill_capped"] is True
    assert "FULL output" not in capped["warning"] and "INCOMPLETE" in capped["warning"]
    assert "[... spill capped" in Path(capped["stdout_spill_path"]).read_text(encoding="utf-8")

    _, full = cet._truncate_stdout_text("y" * 300)
    assert "stdout_spill_capped" not in full
    assert "FULL output" in full["warning"]
    assert Path(full["stdout_spill_path"]).read_text(encoding="utf-8") == "y" * 300
