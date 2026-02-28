"""Tests for trajectory_compressor.py — config, metrics, and compression logic."""

import json
from unittest.mock import patch, MagicMock

from trajectory_compressor import (
    CompressionConfig,
    TrajectoryMetrics,
    AggregateMetrics,
    TrajectoryCompressor,
)


# ---------------------------------------------------------------------------
# CompressionConfig
# ---------------------------------------------------------------------------


class TestCompressionConfig:
    def test_defaults(self):
        config = CompressionConfig()
        assert config.target_max_tokens == 15250
        assert config.summary_target_tokens == 750
        assert config.protect_last_n_turns == 4
        assert config.skip_under_target is True

    def test_from_yaml(self, tmp_path):
        yaml_content = """\
tokenizer:
  name: custom-tokenizer
  trust_remote_code: false
compression:
  target_max_tokens: 10000
  summary_target_tokens: 500
protected_turns:
  first_system: true
  first_human: false
  last_n_turns: 6
summarization:
  model: gpt-4
  temperature: 0.5
  max_retries: 5
output:
  add_summary_notice: false
  output_suffix: _short
processing:
  num_workers: 8
  max_concurrent_requests: 100
  skip_under_target: false
  save_over_limit: false
metrics:
  enabled: false
  per_trajectory: false
  output_file: my_metrics.json
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        config = CompressionConfig.from_yaml(str(yaml_file))
        assert config.tokenizer_name == "custom-tokenizer"
        assert config.trust_remote_code is False
        assert config.target_max_tokens == 10000
        assert config.summary_target_tokens == 500
        assert config.protect_first_human is False
        assert config.protect_last_n_turns == 6
        assert config.summarization_model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_retries == 5
        assert config.add_summary_notice is False
        assert config.output_suffix == "_short"
        assert config.num_workers == 8
        assert config.max_concurrent_requests == 100
        assert config.skip_under_target is False
        assert config.save_over_limit is False
        assert config.metrics_enabled is False
        assert config.metrics_output_file == "my_metrics.json"

    def test_from_yaml_partial(self, tmp_path):
        """Only specified sections override defaults."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("compression:\n  target_max_tokens: 8000\n")
        config = CompressionConfig.from_yaml(str(yaml_file))
        assert config.target_max_tokens == 8000
        # Other sections keep defaults
        assert config.protect_last_n_turns == 4
        assert config.num_workers == 4

    def test_from_yaml_empty(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("{}\n")
        config = CompressionConfig.from_yaml(str(yaml_file))
        assert config.target_max_tokens == 15250  # all defaults


# ---------------------------------------------------------------------------
# TrajectoryMetrics
# ---------------------------------------------------------------------------


class TestTrajectoryMetrics:
    def test_to_dict(self):
        m = TrajectoryMetrics()
        m.original_tokens = 10000
        m.compressed_tokens = 5000
        m.tokens_saved = 5000
        m.compression_ratio = 0.5
        m.original_turns = 20
        m.compressed_turns = 10
        m.turns_removed = 10
        m.was_compressed = True
        d = m.to_dict()
        assert d["original_tokens"] == 10000
        assert d["compressed_tokens"] == 5000
        assert d["compression_ratio"] == 0.5
        assert d["was_compressed"] is True
        assert d["compression_region"]["start_idx"] == -1

    def test_default_values(self):
        m = TrajectoryMetrics()
        d = m.to_dict()
        assert d["original_tokens"] == 0
        assert d["was_compressed"] is False
        assert d["skipped_under_target"] is False


# ---------------------------------------------------------------------------
# AggregateMetrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_empty_to_dict(self):
        agg = AggregateMetrics()
        d = agg.to_dict()
        assert d["summary"]["total_trajectories"] == 0
        assert d["averages"]["avg_compression_ratio"] == 1.0
        assert d["averages"]["avg_tokens_saved_per_compressed"] == 0

    def test_add_compressed_trajectory(self):
        agg = AggregateMetrics()
        m = TrajectoryMetrics()
        m.original_tokens = 20000
        m.compressed_tokens = 10000
        m.tokens_saved = 10000
        m.compression_ratio = 0.5
        m.original_turns = 30
        m.compressed_turns = 15
        m.turns_removed = 15
        m.was_compressed = True
        agg.add_trajectory_metrics(m)
        assert agg.total_trajectories == 1
        assert agg.trajectories_compressed == 1
        assert agg.total_tokens_saved == 10000
        assert len(agg.compression_ratios) == 1

    def test_add_skipped_trajectory(self):
        agg = AggregateMetrics()
        m = TrajectoryMetrics()
        m.original_tokens = 5000
        m.compressed_tokens = 5000
        m.skipped_under_target = True
        agg.add_trajectory_metrics(m)
        assert agg.trajectories_skipped_under_target == 1
        assert agg.trajectories_compressed == 0

    def test_add_over_limit_trajectory(self):
        agg = AggregateMetrics()
        m = TrajectoryMetrics()
        m.original_tokens = 20000
        m.compressed_tokens = 16000
        m.still_over_limit = True
        m.was_compressed = True
        m.compression_ratio = 0.8
        agg.add_trajectory_metrics(m)
        assert agg.trajectories_still_over_limit == 1

    def test_multiple_trajectories_aggregation(self):
        agg = AggregateMetrics()
        for i in range(3):
            m = TrajectoryMetrics()
            m.original_tokens = 10000
            m.compressed_tokens = 5000
            m.tokens_saved = 5000
            m.turns_removed = 5
            m.was_compressed = True
            m.compression_ratio = 0.5
            agg.add_trajectory_metrics(m)
        d = agg.to_dict()
        assert d["summary"]["total_trajectories"] == 3
        assert d["summary"]["trajectories_compressed"] == 3
        assert d["tokens"]["total_saved"] == 15000
        assert d["averages"]["avg_compression_ratio"] == 0.5

    def test_to_dict_no_division_by_zero(self):
        """Ensure no ZeroDivisionError with empty data."""
        agg = AggregateMetrics()
        d = agg.to_dict()
        assert d["summarization"]["success_rate"] == 1.0
        assert d["tokens"]["overall_compression_ratio"] == 0.0


# ---------------------------------------------------------------------------
# TrajectoryCompressor._find_protected_indices
# ---------------------------------------------------------------------------


def _make_compressor(config=None):
    """Create a TrajectoryCompressor with mocked tokenizer and summarizer."""
    if config is None:
        config = CompressionConfig()
    with patch.object(TrajectoryCompressor, '_init_tokenizer'), \
         patch.object(TrajectoryCompressor, '_init_summarizer'):
        compressor = TrajectoryCompressor(config)
    # Provide a simple token counter for tests (1 token per 4 chars)
    compressor.tokenizer = MagicMock()
    compressor.tokenizer.encode = lambda text: [0] * (len(text) // 4)
    return compressor


class TestFindProtectedIndices:
    def test_basic_trajectory(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "system", "value": "You are an agent."},
            {"from": "human", "value": "Do something."},
            {"from": "gpt", "value": "I will use a tool."},
            {"from": "tool", "value": "Tool result."},
            {"from": "gpt", "value": "More work."},
            {"from": "tool", "value": "Another result."},
            {"from": "gpt", "value": "Work continues."},
            {"from": "tool", "value": "Result 3."},
            {"from": "gpt", "value": "Done."},
            {"from": "human", "value": "Thanks."},
        ]
        protected, start, end = tc._find_protected_indices(trajectory)
        # First system (0), human (1), gpt (2), tool (3) are protected
        assert 0 in protected
        assert 1 in protected
        assert 2 in protected
        assert 3 in protected
        # Last 4 turns (6,7,8,9) are protected
        assert 6 in protected
        assert 7 in protected
        assert 8 in protected
        assert 9 in protected
        # Compressible region should be between head and tail
        assert start >= 4
        assert end <= 6

    def test_short_trajectory_all_protected(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ]
        protected, start, end = tc._find_protected_indices(trajectory)
        # All 3 turns should be protected (first of each + last 4 covers all)
        assert len(protected) == 3
        assert start >= end  # Nothing to compress

    def test_protect_last_n_zero(self):
        config = CompressionConfig()
        config.protect_last_n_turns = 0
        tc = _make_compressor(config)
        trajectory = [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
            {"from": "tool", "value": "r"},
            {"from": "gpt", "value": "b"},
            {"from": "tool", "value": "r2"},
            {"from": "gpt", "value": "c"},
            {"from": "tool", "value": "r3"},
        ]
        protected, start, end = tc._find_protected_indices(trajectory)
        # Only first occurrences protected, no tail protection
        assert 0 in protected
        assert 1 in protected
        assert 2 in protected
        assert 3 in protected
        assert 7 not in protected

    def test_no_system_turn(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
            {"from": "tool", "value": "data"},
            {"from": "gpt", "value": "result"},
            {"from": "human", "value": "thanks"},
        ]
        protected, start, end = tc._find_protected_indices(trajectory)
        assert 0 in protected  # first human

    def test_disable_protect_first_system(self):
        config = CompressionConfig()
        config.protect_first_system = False
        tc = _make_compressor(config)
        trajectory = [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
            {"from": "tool", "value": "r"},
            {"from": "gpt", "value": "b"},
            {"from": "tool", "value": "r2"},
            {"from": "gpt", "value": "c"},
            {"from": "tool", "value": "r3"},
        ]
        protected, _, _ = tc._find_protected_indices(trajectory)
        assert 0 not in protected  # system not protected


# ---------------------------------------------------------------------------
# TrajectoryCompressor._extract_turn_content_for_summary
# ---------------------------------------------------------------------------


class TestExtractTurnContent:
    def test_basic_extraction(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "gpt", "value": "I will search."},
            {"from": "tool", "value": "Search result: found it."},
            {"from": "gpt", "value": "Great, done."},
        ]
        content = tc._extract_turn_content_for_summary(trajectory, 0, 2)
        assert "[Turn 0 - GPT]" in content
        assert "I will search." in content
        assert "[Turn 1 - TOOL]" in content
        assert "Search result: found it." in content
        # Turn 2 should NOT be included (end is exclusive)
        assert "[Turn 2" not in content

    def test_long_content_truncated(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "tool", "value": "x" * 5000},
        ]
        content = tc._extract_turn_content_for_summary(trajectory, 0, 1)
        assert "...[truncated]..." in content
        assert len(content) < 5000

    def test_empty_range(self):
        tc = _make_compressor()
        trajectory = [{"from": "gpt", "value": "hello"}]
        content = tc._extract_turn_content_for_summary(trajectory, 0, 0)
        assert content == ""


# ---------------------------------------------------------------------------
# TrajectoryCompressor.count_tokens / count_trajectory_tokens
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_count_tokens_empty(self):
        tc = _make_compressor()
        assert tc.count_tokens("") == 0

    def test_count_tokens_basic(self):
        tc = _make_compressor()
        # Our mock: 1 token per 4 chars
        assert tc.count_tokens("12345678") == 2

    def test_count_trajectory_tokens(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "system", "value": "12345678"},   # 2 tokens
            {"from": "human", "value": "1234567890ab"}, # 3 tokens
        ]
        assert tc.count_trajectory_tokens(trajectory) == 5

    def test_count_turn_tokens(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "system", "value": "1234"},     # 1 token
            {"from": "human", "value": "12345678"},  # 2 tokens
        ]
        result = tc.count_turn_tokens(trajectory)
        assert result == [1, 2]

    def test_count_tokens_fallback_on_error(self):
        tc = _make_compressor()
        tc.tokenizer.encode = MagicMock(side_effect=Exception("fail"))
        # Should fallback to len(text) // 4
        assert tc.count_tokens("12345678") == 2
