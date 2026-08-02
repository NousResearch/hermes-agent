"""Tests for subagent summary budgeting (PR #9126).

delegate_task caps subagent summaries against the parent's remaining context
headroom (split across the batch) before they enter the parent's context, and
spills the full text to disk so nothing is lost. This guards the
compression/429 death spiral that batch fan-out could trigger by returning N
full summaries verbatim into the parent.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest

import tools.delegate_tool as dt


class _FakeCompressor:
    def __init__(self, context_length, max_tokens):
        self.context_length = context_length
        self.max_tokens = max_tokens


class _FakeParent:
    def __init__(self, context_length, used_tokens, max_tokens):
        self.context_compressor = _FakeCompressor(context_length, max_tokens)
        self.session_prompt_tokens = used_tokens
        self._current_task_id: str | None = None


def test_small_summaries_pass_through_untouched():
    parent = _FakeParent(context_length=200_000, used_tokens=10_000, max_tokens=8_000)
    results = [
        {"task_index": 0, "summary": "short result A", "status": "completed"},
        {"task_index": 1, "summary": "short result B", "status": "completed"},
    ]
    dt._apply_summary_budget(results, parent)
    assert results[0]["summary"] == "short result A"
    assert "summary_truncated" not in results[0]
    assert "summary_truncated" not in results[1]


def test_batch_overflow_trimmed_and_spilled_losslessly(monkeypatch):
    # Isolate spill directory to a temp HERMES_HOME.
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("HERMES_HOME", os.path.join(td, ".hermes"))
        # Distinct head + tail markers so we can prove the tail survives.
        big = "HEAD_MARKER\n" + ("X" * 50_000) + "\nTAIL_MARKER"
        # Parent nearly full (120k/131k) → tiny headroom → aggressive trim.
        parent = _FakeParent(context_length=131_000, used_tokens=120_000, max_tokens=8_000)
        results = [
            {"task_index": i, "summary": big, "status": "completed"} for i in range(5)
        ]
        dt._apply_summary_budget(results, parent)
        for r in results:
            assert r["summary_truncated"] is True
            assert len(r["summary"]) < len(big)
            # Head+tail window: both ends survive in-context.
            assert "HEAD_MARKER" in r["summary"]
            assert "TAIL_MARKER" in r["summary"]
            path = r.get("summary_full_path")
            assert path and os.path.exists(path)
            # The spill file holds the FULL original text — nothing is lost.
            with open(path, encoding="utf-8") as fh:
                assert fh.read() == big
            # The footer points the parent at the full version with an offset.
            assert "read_file" in r["summary"]
            assert "offset=" in r["summary"]
            # Spilled into the delegation cache (mounted into remote backends).
            assert os.path.join("cache", "delegation") in path


def test_docker_spill_path_points_to_the_mounted_cache(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    big = "HEAD\n" + ("X" * 30_000) + "\nTAIL"
    results = [{"task_index": 0, "summary": big, "status": "completed"}]

    dt._apply_summary_budget(results, _FakeParent(131_000, 120_000, 8_000))

    agent_path = results[0]["summary_full_path"]
    assert agent_path.startswith("/root/.hermes/cache/delegation/")
    assert f'path="{agent_path}"' in results[0]["summary"]
    host_files = list((hermes_home / "cache" / "delegation").glob("subagent-summary-*.txt"))
    assert len(host_files) == 1
    assert host_files[0].read_text(encoding="utf-8") == big


def test_ssh_spill_path_is_synced_to_the_remote_cache(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    sync_calls = []
    env = SimpleNamespace(
        _remote_home="/home/remote",
        _sync_manager=SimpleNamespace(
            sync=lambda *, force=False: sync_calls.append(force)
        ),
    )
    monkeypatch.setattr("tools.terminal_tool.get_active_env", lambda task_id: env)
    parent = _FakeParent(131_000, 120_000, 8_000)
    parent._current_task_id = "parent-task"
    results = [
        {"task_index": 0, "summary": "X" * 30_000, "status": "completed"}
    ]

    dt._apply_summary_budget(results, parent)

    assert results[0]["summary_full_path"].startswith(
        "/home/remote/.hermes/cache/delegation/"
    )
    assert sync_calls == [True]


def test_empty_results_is_noop():
    # No summaries → nothing to do, must not raise.
    dt._apply_summary_budget([], _FakeParent(131_000, 1_000, 8_000))
    dt._apply_summary_budget(
        [{"task_index": 0, "status": "failed", "summary": None}],
        _FakeParent(131_000, 1_000, 8_000),
    )
