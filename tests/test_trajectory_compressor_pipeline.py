"""Tests for the trajectory_compressor.py *pipeline*: process_entry(_async),
_process_directory_async, _print_summary, and main().

These cover the end-to-end CLI/pipeline regions (lines ~1017-1594) that the
unit tests in test_trajectory_compressor.py (config / metrics / compression
math) deliberately skip.  Everything is hermetic: the HF tokenizer and the LLM
summarizer are stubbed so no network or model download happens.
"""

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trajectory_compressor import (
    AggregateMetrics,
    CompressionConfig,
    TrajectoryCompressor,
    TrajectoryMetrics,
    main,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_compressor(config=None):
    """Build a TrajectoryCompressor via __new__ with a stubbed tokenizer.

    Unlike ``_make_compressor`` in the unit-test module (which goes through
    ``__init__`` while patching the init hooks), we bypass ``__init__`` so the
    pipeline helpers can be monkeypatched on the instance without touching the
    real tokenizer/LLM plumbing.
    """
    if config is None:
        config = CompressionConfig()
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = config
    compressor.logger = logging.getLogger("test_trajectory_compressor_pipeline")
    compressor.tokenizer = MagicMock()
    # 1 token per 4 chars (matches the existing test-module convention).
    compressor.tokenizer.encode = lambda text: [0] * max(1, len(text) // 4)
    compressor.aggregate_metrics = AggregateMetrics()
    return compressor


def _pipeline_config(**overrides):
    """A CompressionConfig tuned to force compression through the pipeline."""
    config = CompressionConfig(
        target_max_tokens=30,
        summary_target_tokens=10,
        per_trajectory_timeout=60,
        max_retries=1,
        protect_last_n_turns=2,
    )
    config.metrics_enabled = True
    config.metrics_per_trajectory = True
    config.metrics_output_file = "compression_metrics.json"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _trajectory(middle_value):
    """An 8-turn trajectory whose protected region is [4, 6)."""
    return [
        {"from": "system", "value": "You are an agent."},
        {"from": "human", "value": "Please do the task."},
        {"from": "gpt", "value": "I will use a tool to search."},
        {"from": "tool", "value": "Search result returned."},
        {"from": "gpt", "value": middle_value},
        {"from": "tool", "value": "tool result here."},
        {"from": "gpt", "value": "final answer."},
        {"from": "human", "value": "thanks"},
    ]


def _big_middle_entry(marker):
    """A compressible entry; ``marker`` is embedded in the compressible middle."""
    assert len(marker) >= 1
    middle = (marker + " ") + ("blah " * 200)
    return {"id": marker, "conversations": _trajectory(middle)}


async def _fake_generate_summary_async(self, content, metrics):
    """Stands in for the LLM: counts a call and returns a summary."""
    metrics.summarization_api_calls += 1
    return "[CONTEXT SUMMARY]: fake summary"


@pytest.fixture
def stub_summarizer(monkeypatch):
    """Point the async summary generator at the deterministic fake."""

    async def _fake(self, content, metrics):
        metrics.summarization_api_calls += 1
        return "[CONTEXT SUMMARY]: fake summary"

    monkeypatch.setattr(TrajectoryCompressor, "_generate_summary_async", _fake)


def _write_jsonl(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# process_entry / process_entry_async  (lines ~1017-1063)
# ---------------------------------------------------------------------------


class TestProcessEntry:
    """process_entry: entry plumbing without the compression math."""

    def test_no_conversations_returns_entry_unchanged(self):
        """An entry lacking 'conversations' is returned as-is with default metrics."""
        compressor = _make_compressor()
        entry = {"id": "metadata-only"}

        result, metrics = compressor.process_entry(entry)

        assert result == entry
        assert metrics.original_tokens == 0
        assert metrics.was_compressed is False

    def test_conversations_replaced_by_compressor_output(self):
        """The compressor's output replaces entry['conversations']."""
        compressor = _make_compressor()
        original = {"id": "x", "conversations": _trajectory("old middle")}
        compressed = _trajectory("new middle")
        metrics = TrajectoryMetrics()
        metrics.was_compressed = True
        compressor.compress_trajectory = MagicMock(return_value=(compressed, metrics))

        result, _ = compressor.process_entry(original)

        assert result["conversations"] == compressed
        compressor.compress_trajectory.assert_called_once_with(original["conversations"])

    def test_compression_metrics_added_when_compressed_and_enabled(self):
        """compression_metrics appears only on a compressed, enabled entry."""
        compressor = _make_compressor(_pipeline_config(metrics_per_trajectory=True))
        original = {"id": "x", "conversations": _trajectory("old")}
        metrics = TrajectoryMetrics()
        metrics.original_tokens = 100
        metrics.was_compressed = True
        compressor.compress_trajectory = MagicMock(
            return_value=(_trajectory("new"), metrics)
        )

        result, _ = compressor.process_entry(original)

        assert "compression_metrics" in result
        assert result["compression_metrics"] == metrics.to_dict()

    def test_no_compression_metrics_when_metric_disabled(self):
        """metrics_per_trajectory=False suppresses the per-entry metrics key."""
        compressor = _make_compressor(_pipeline_config(metrics_per_trajectory=False))
        original = {"id": "x", "conversations": _trajectory("old")}
        metrics = TrajectoryMetrics()
        metrics.was_compressed = True
        compressor.compress_trajectory = MagicMock(
            return_value=(_trajectory("new"), metrics)
        )

        result, _ = compressor.process_entry(original)

        assert "compression_metrics" not in result

    def test_no_compression_metrics_when_skipped(self):
        """A skipped (uncompressed) entry never gains a compression_metrics key."""
        compressor = _make_compressor(_pipeline_config(metrics_per_trajectory=True))
        original = {"id": "x", "conversations": _trajectory("old")}
        metrics = TrajectoryMetrics()  # was_compressed is False
        compressor.compress_trajectory = MagicMock(
            return_value=(_trajectory("same"), metrics)
        )

        result, _ = compressor.process_entry(original)

        assert "compression_metrics" not in result


class TestProcessEntryAsync:
    """process_entry_async: the same invariants through the async path."""

    def test_no_conversations_returns_entry_unchanged(self):
        compressor = _make_compressor()
        entry = {"id": "metadata-only"}

        result, metrics = asyncio.run(compressor.process_entry_async(entry))

        assert result == entry
        assert metrics.was_compressed is False

    def test_conversations_replaced_by_async_compressor_output(self):
        compressor = _make_compressor()
        original = {"id": "x", "conversations": _trajectory("old")}
        compressed = _trajectory("new")
        metrics = TrajectoryMetrics()
        metrics.was_compressed = True
        compressor.compress_trajectory_async = AsyncMock(
            return_value=(compressed, metrics)
        )

        result, _ = asyncio.run(compressor.process_entry_async(original))

        assert result["conversations"] == compressed
        compressor.compress_trajectory_async.assert_awaited_once_with(
            original["conversations"]
        )

    def test_compression_metrics_added_when_compressed_and_enabled(self):
        compressor = _make_compressor(_pipeline_config(metrics_per_trajectory=True))
        original = {"id": "x", "conversations": _trajectory("old")}
        metrics = TrajectoryMetrics()
        metrics.original_tokens = 100
        metrics.was_compressed = True
        compressor.compress_trajectory_async = AsyncMock(
            return_value=(_trajectory("new"), metrics)
        )

        result, _ = asyncio.run(compressor.process_entry_async(original))

        assert "compression_metrics" in result
        assert result["compression_metrics"] == metrics.to_dict()

    def test_no_compression_metrics_when_skipped(self):
        compressor = _make_compressor(_pipeline_config(metrics_per_trajectory=True))
        original = {"id": "x", "conversations": _trajectory("old")}
        metrics = TrajectoryMetrics()
        compressor.compress_trajectory_async = AsyncMock(
            return_value=(_trajectory("same"), metrics)
        )

        result, _ = asyncio.run(compressor.process_entry_async(original))

        assert "compression_metrics" not in result


# ---------------------------------------------------------------------------
# _process_directory_async  (lines ~1076-1269)
# ---------------------------------------------------------------------------


class TestProcessDirectoryAsync:
    """End-to-end directory pipeline with the LLM + tokenizer stubbed."""

    def test_happy_path_compresses_and_writes_metrics(
        self, tmp_path, stub_summarizer
    ):
        """Valid entries compress, invalid JSON is skipped, metrics are written."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file1 = input_dir / "file1.jsonl"
        _write_jsonl(
            file1,
            [
                json.dumps(_big_middle_entry("first")),
                "{ this is not valid json",  # skipped
            ],
        )
        file2 = input_dir / "file2.jsonl"
        _write_jsonl(
            file2,
            [
                json.dumps({"id": "no-conversations"}),  # untouched entry
                json.dumps(_big_middle_entry("second")),
            ],
        )

        compressor = _make_compressor(_pipeline_config())
        compressor.process_directory(input_dir, output_dir)

        # Output files exist with the same names, in the same order.
        out1 = output_dir / "file1.jsonl"
        out2 = output_dir / "file2.jsonl"
        assert out1.exists()
        assert out2.exists()

        # file1: the invalid line was dropped, so exactly one entry remains.
        out1_entries = [json.loads(l) for l in out1.read_text().splitlines()]
        assert len(out1_entries) == 1
        assert out1_entries[0]["id"] == "first"
        assert "compression_metrics" in out1_entries[0]

        # file2: the no-conversations entry passes through untouched; the
        # compressible one is compressed.
        out2_entries = [json.loads(l) for l in out2.read_text().splitlines()]
        assert len(out2_entries) == 2
        assert out2_entries[0] == {"id": "no-conversations"}
        assert out2_entries[1]["id"] == "second"
        assert "compression_metrics" in out2_entries[1]

        # Aggregate metrics count every valid trajectory; two were compressed.
        assert compressor.aggregate_metrics.total_trajectories == 3
        assert compressor.aggregate_metrics.trajectories_compressed == 2

        # Metrics file is written with the expected top-level keys.
        metrics_path = output_dir / "compression_metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert set(metrics) == {
            "summary",
            "tokens",
            "turns",
            "averages",
            "summarization",
            "processing",
        }
        assert metrics["summary"]["total_trajectories"] == 3
        assert metrics["summary"]["trajectories_compressed"] == 2

    def test_error_path_keeps_original_and_counts_failed(
        self, tmp_path, monkeypatch
    ):
        """An entry whose summarization raises is preserved verbatim + counted."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        file1 = input_dir / "file1.jsonl"
        _write_jsonl(
            file1,
            [
                json.dumps(_big_middle_entry("FAILMARK")),  # raises
                json.dumps(_big_middle_entry("ok")),  # compresses
            ],
        )

        async def _raise_on_failmark(self, content, metrics):
            if "FAILMARK" in content:
                raise RuntimeError("summarization exploded")
            metrics.summarization_api_calls += 1
            return "[CONTEXT SUMMARY]: fake summary"

        monkeypatch.setattr(
            TrajectoryCompressor, "_generate_summary_async", _raise_on_failmark
        )

        compressor = _make_compressor(_pipeline_config())
        compressor.process_directory(input_dir, output_dir)

        out_entries = [
            json.loads(l)
            for l in (output_dir / "file1.jsonl").read_text().splitlines()
        ]
        assert len(out_entries) == 2

        # The failing entry is preserved in its original (uncompressed) form.
        failed = next(e for e in out_entries if e["id"] == "FAILMARK")
        assert failed["conversations"] == _trajectory(
            ("FAILMARK ") + ("blah " * 200)
        )
        # The surviving entry was compressed.
        ok = next(e for e in out_entries if e["id"] == "ok")
        assert "compression_metrics" in ok

        assert compressor.aggregate_metrics.trajectories_failed == 1
        assert compressor.aggregate_metrics.trajectories_compressed == 1

    def test_empty_input_dir_returns_early_no_output(self, tmp_path):
        """A directory without any .jsonl produces no output dir."""
        input_dir = tmp_path / "empty_in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        compressor = _make_compressor(_pipeline_config())
        compressor.process_directory(input_dir, output_dir)

        assert not output_dir.exists()
        # No trajectories were recorded.
        assert compressor.aggregate_metrics.total_trajectories == 0


# ---------------------------------------------------------------------------
# _print_summary  (lines ~1271-1377)
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """_print_summary: report content, not exact formatting."""

    def test_compressed_branch_prints_report(self, capsys):
        """A report with compressor work shows the headline + Space Savings."""
        compressor = _make_compressor()
        aggregate = AggregateMetrics()
        aggregate.processing_start_time = "2026-01-01T00:00:00"
        aggregate.processing_end_time = "2026-01-01T00:05:00"
        aggregate.processing_duration_seconds = 5.0
        m = TrajectoryMetrics()
        m.original_tokens = 1000
        m.compressed_tokens = 500
        m.tokens_saved = 500
        m.compression_ratio = 0.5
        m.original_turns = 10
        m.compressed_turns = 6
        m.turns_removed = 4
        m.was_compressed = True
        aggregate.add_trajectory_metrics(m)
        compressor.aggregate_metrics = aggregate

        compressor._print_summary()
        out = capsys.readouterr().out

        assert "TRAJECTORY COMPRESSION REPORT" in out
        assert "Total Processed" in out
        assert "Space Savings" in out
        # Short duration renders in seconds; throughput line is present.
        assert "seconds" in out

    def test_no_compressed_branch_prints_message(self, capsys):
        """With nothing compressed, the report says so instead of crashing."""
        compressor = _make_compressor()
        aggregate = AggregateMetrics()
        aggregate.processing_start_time = "2026-01-01T00:00:00"
        aggregate.processing_end_time = "2026-01-01T00:01:00"
        aggregate.processing_duration_seconds = 1.0
        compressor.aggregate_metrics = aggregate

        compressor._print_summary()
        out = capsys.readouterr().out

        assert "TRAJECTORY COMPRESSION REPORT" in out
        assert "No trajectories were compressed" in out
        # tokens_before == 0, so the Space Savings line must be omitted.
        assert "Space Savings" not in out

    def test_long_duration_branch_renders_minutes(self, capsys):
        """A duration over 60s is reported in minutes."""
        compressor = _make_compressor()
        aggregate = AggregateMetrics()
        aggregate.processing_start_time = "2026-01-01T00:00:00"
        aggregate.processing_end_time = "2026-01-01T01:01:00"
        aggregate.processing_duration_seconds = 61.0
        compressor.aggregate_metrics = aggregate

        compressor._print_summary()
        out = capsys.readouterr().out

        assert "1.0 minutes" in out
        assert "minutes" in out


# ---------------------------------------------------------------------------
# main()  (lines ~1380-1594)
# ---------------------------------------------------------------------------


class TestMain:
    """main(): the Fire-wrapped CLI entry point.

    TrajectoryCompressor.__init__ normally loads a real HF tokenizer
    (_init_tokenizer) and builds an LLM client (_init_summarizer).  Both are
    stubbed so constructing the compressor inside main() is cheap and
    network-free.  The async summarizer is also stubbed.
    """

    @pytest.fixture(autouse=True)
    def _stub_compressor_init(self, monkeypatch):
        def _init_tokenizer(self):
            self.tokenizer = MagicMock()
            self.tokenizer.encode = lambda text: [0] * max(1, len(text) // 4)

        def _init_summarizer(self):
            self._use_call_llm = False
            self.client = MagicMock()
            self.async_client = None
            self._async_client_api_key = "fake"

        async def _generate_summary_async(self, content, metrics):
            metrics.summarization_api_calls += 1
            return "[CONTEXT SUMMARY]: fake summary"

        monkeypatch.setattr(TrajectoryCompressor, "_init_tokenizer", _init_tokenizer)
        monkeypatch.setattr(
            TrajectoryCompressor, "_init_summarizer", _init_summarizer
        )
        monkeypatch.setattr(
            TrajectoryCompressor, "_generate_summary_async", _generate_summary_async
        )

    @staticmethod
    def _missing_config(tmp_path):
        return str(tmp_path / "does-not-exist.yaml")

    def test_input_not_found_returns_early(self, tmp_path, capsys):
        """A nonexistent input prints an error and returns without crashing."""
        ret = main(
            input=str(tmp_path / "nope.jsonl"),
            config=self._missing_config(tmp_path),
        )
        assert ret is None
        assert "Input not found" in capsys.readouterr().out

    @pytest.mark.parametrize("bad_pct", [0, 150])
    def test_invalid_sample_percent_returns_early(self, tmp_path, capsys, bad_pct):
        """sample_percent outside (1,100) returns with an error message."""
        input_file = tmp_path / "in.jsonl"
        _write_jsonl(input_file, [json.dumps({"id": "a"})])

        ret = main(
            input=str(input_file),
            sample_percent=bad_pct,
            config=self._missing_config(tmp_path),
        )
        assert ret is None
        out = capsys.readouterr().out
        assert "sample_percent must be between 1 and 100" in out

    def test_file_input_dry_run_creates_no_output(self, tmp_path, capsys):
        """dry_run on a file prints a preview and writes nothing."""
        input_file = tmp_path / "in.jsonl"
        _write_jsonl(input_file, [json.dumps({"id": "a"})])
        output_file = tmp_path / "out.jsonl"

        ret = main(
            input=str(input_file),
            output=str(output_file),
            config=self._missing_config(tmp_path),
            dry_run=True,
        )
        assert ret is None
        assert "DRY RUN MODE" in capsys.readouterr().out
        assert not output_file.exists()

    def test_directory_input_dry_run_creates_no_output(self, tmp_path, capsys):
        """dry_run on a directory prints a preview and writes nothing."""
        input_dir = tmp_path / "in_dir"
        input_dir.mkdir()
        _write_jsonl(input_dir / "a.jsonl", [json.dumps({"id": "a"})])
        output_dir = tmp_path / "out_dir"

        ret = main(
            input=str(input_dir),
            output=str(output_dir),
            config=self._missing_config(tmp_path),
            dry_run=True,
        )
        assert ret is None
        assert "DRY RUN MODE" in capsys.readouterr().out
        assert not output_dir.exists()

    def test_file_input_happy_path(self, tmp_path, capsys):
        """A file input is compressed and the metrics file is copied beside it."""
        input_file = tmp_path / "in.jsonl"
        _write_jsonl(
            input_file,
            [
                json.dumps({"id": "a", "conversations": _trajectory("aa " * 200)}),
                json.dumps({"id": "b"}),  # no conversations: passes through
            ],
        )
        output_file = tmp_path / "out.jsonl"

        main(
            input=str(input_file),
            output=str(output_file),
            config=self._missing_config(tmp_path),
        )

        out_entries = [
            json.loads(l) for l in output_file.read_text().splitlines()
        ]
        # The two input entries are both present in the output.
        assert len(out_entries) == 2
        assert {e["id"] for e in out_entries} == {"a", "b"}
        # The file-input path copies the metrics file as <stem>_metrics.json.
        assert (tmp_path / "out_metrics.json").exists()
        assert "Compression complete" in capsys.readouterr().out

    def test_directory_input_happy_path(self, tmp_path):
        """A directory input writes every same-named JSONL into the output dir."""
        input_dir = tmp_path / "in_dir"
        input_dir.mkdir()
        _write_jsonl(input_dir / "a.jsonl", [json.dumps({"id": "a"})])
        _write_jsonl(input_dir / "b.jsonl", [json.dumps({"id": "b"})])
        output_dir = tmp_path / "out_dir"

        main(
            input=str(input_dir),
            output=str(output_dir),
            config=self._missing_config(tmp_path),
        )

        assert (output_dir / "a.jsonl").exists()
        assert (output_dir / "b.jsonl").exists()
        assert (
            json.loads((output_dir / "a.jsonl").read_text().splitlines()[0])["id"]
            == "a"
        )

    def test_directory_input_sample_percent(self, tmp_path):
        """sample_percent in directory mode samples entries from each file."""
        input_dir = tmp_path / "in_dir"
        input_dir.mkdir()
        lots = "x " * 200
        # 4 entries per file; a 50% sample keeps 2 per file (seed=42).
        for name in ("a.jsonl", "b.jsonl"):
            _write_jsonl(
                input_dir / name,
                [json.dumps({"id": f"{name}-{i}", "conversations": _trajectory(lots)})
                 for i in range(4)]
                + ["{ not json"],  # invalid line sampled-out during load
            )
        output_dir = tmp_path / "out_dir"

        main(
            input=str(input_dir),
            output=str(output_dir),
            config=self._missing_config(tmp_path),
            sample_percent=50,
        )

        for name in ("a.jsonl", "b.jsonl"):
            entries = [
                json.loads(l)
                for l in (output_dir / name).read_text().splitlines()
            ]
            assert len(entries) == 2

    def test_file_input_default_output_and_cli_overrides(
        self, tmp_path, capsys
    ):
        """Config-found branch, CLI overrides, and output=None default path."""
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("compression:\n  target_max_tokens: 40\n", encoding="utf-8")
        input_file = tmp_path / "in.jsonl"
        _write_jsonl(input_file, [json.dumps({"id": "a", "conversations": _trajectory("aa " * 200)})])

        # output=None -> default <stem>_compressed.jsonl beside the input.
        main(
            input=str(input_file),
            config=str(config_path),
            target_max_tokens=40,
            tokenizer="fake-tokenizer",
        )

        default_out = tmp_path / "in_compressed.jsonl"
        assert default_out.exists()
        out = capsys.readouterr().out
        assert "Loading config from" in out
        # The entry was compressed (target ~40 < trajectory tokens), so the
        # output differs from the input and the metrics file was copied.
        assert (tmp_path / "in_compressed_metrics.json").exists()

    def test_file_input_skips_invalid_json(self, tmp_path, capsys):
        """A malformed JSON line is skipped and reported."""
        input_file = tmp_path / "in.jsonl"
        _write_jsonl(
            input_file,
            [json.dumps({"id": "a"}), "{ not json"],
        )
        output_file = tmp_path / "out.jsonl"

        main(
            input=str(input_file),
            output=str(output_file),
            config=self._missing_config(tmp_path),
        )

        out_entries = [json.loads(l) for l in output_file.read_text().splitlines()]
        assert len(out_entries) == 1
        assert out_entries[0]["id"] == "a"
        assert "Skipping invalid JSON" in capsys.readouterr().out

    def test_file_input_sample_percent(self, tmp_path, capsys):
        """sample_percent in file mode samples from the loaded entries."""
        input_file = tmp_path / "in.jsonl"
        _write_jsonl(
            input_file,
            [json.dumps({"id": f"e{i}", "conversations": _trajectory("x " * 200)}) for i in range(4)],
        )
        output_file = tmp_path / "out.jsonl"

        main(
            input=str(input_file),
            output=str(output_file),
            config=self._missing_config(tmp_path),
            sample_percent=50,
        )

        out_entries = [json.loads(l) for l in output_file.read_text().splitlines()]
        # 4 entries x 50% -> 2 sampled entries.
        assert len(out_entries) == 2
        assert "Sampled" in capsys.readouterr().out

    def test_directory_input_default_output(self, tmp_path):
        """output=None for a directory defaults to <name>_compressed."""
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        _write_jsonl(input_dir / "a.jsonl", [json.dumps({"id": "a"})])

        main(
            input=str(input_dir),
            config=self._missing_config(tmp_path),
        )

        default_out = tmp_path / "data_compressed"
        assert (default_out / "a.jsonl").exists()

    def test_directory_sample_percent_dry_run_no_output(self, tmp_path, capsys):
        """dry_run in sampled-directory mode returns without writing."""
        input_dir = tmp_path / "in_dir"
        input_dir.mkdir()
        _write_jsonl(
            input_dir / "a.jsonl",
            [json.dumps({"id": f"a-{i}", "conversations": _trajectory("x " * 200)}) for i in range(4)],
        )
        output_dir = tmp_path / "out_dir"

        ret = main(
            input=str(input_dir),
            output=str(output_dir),
            config=self._missing_config(tmp_path),
            sample_percent=50,
            dry_run=True,
        )
        assert ret is None
        assert "DRY RUN MODE" in capsys.readouterr().out
        assert not output_dir.exists()
