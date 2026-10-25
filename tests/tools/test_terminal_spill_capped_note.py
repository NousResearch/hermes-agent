"""A capped terminal spill must be advertised as partial, not full output (#109757).

The collector tees overflow to a spill file with a hard cap (``_SPILL_CAP_CHARS``).
When the cap is hit — or a spill write/close fails — the saved file holds only a
prefix of the stream, so the structured result and ``truncation_note`` must not
claim complete recovery. These tests drive the real collector, native finalizer,
and foreground result formatter; nothing is patched except the cap/close itself.
"""

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_capped_spill_is_not_advertised_as_full_output(tmp_path):
    collector = _BoundedOutputCollector(100, tmp_path / "capped.log")
    collector._SPILL_CAP_CHARS = 500
    source = "abc " * 1000
    collector.append(source)
    result = _finalize_collector_output(collector, tmp_path)

    saved = Path(result["full_output_path"]).read_text()
    assert saved != source and "[spill capped" in saved
    assert result["full_output_capped"] is True
    assert "full output" not in result["truncation_note"].lower()
    assert "capped" in result["truncation_note"].lower()


def test_uncapped_spill_still_advertises_full_output(tmp_path):
    collector = _BoundedOutputCollector(100, tmp_path / "full.log")
    source = "abc " * 100  # overflows the capture window, far below the spill cap
    collector.append(source)
    result = _finalize_collector_output(collector, tmp_path)

    saved = Path(result["full_output_path"]).read_text()
    assert saved == source  # the uncapped spill holds the full stream
    assert "full_output_capped" not in result
    assert "Full output" in result["truncation_note"]


def test_failed_spill_close_is_not_advertised_as_full_output(tmp_path):
    collector = _BoundedOutputCollector(100, tmp_path / "close-fail.log")
    collector.append("abc " * 100)
    collector._spill_fh.close = lambda: (_ for _ in ()).throw(OSError("disk gone"))
    result = _finalize_collector_output(collector, tmp_path)

    assert result["full_output_capped"] is True
    assert "full output" not in result["truncation_note"].lower()
