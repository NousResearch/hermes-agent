"""Tests for TrajectoryCompressor compression paths (compress_trajectory / async).

Covers the compress_trajectory (sync, lines ~743-889) and
compress_trajectory_async (async, lines ~891-1015) methods, plus the private
helpers they rely on (_find_protected_indices, _snap_boundary,
_extract_turn_content_for_summary).

All scenarios are parametrized over ``branch='sync'/'async'`` and driven through
``asyncio.run`` for the async branch so the SAME matrix exercises both paths.
The compressor is built with the established ``__new__`` idiom (see
tests/test_trajectory_compressor.py) and a stubbed deterministic token counter
(word count) so no real tokenizer is downloaded and no network is touched.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from trajectory_compressor import CompressionConfig, TrajectoryCompressor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _words(n: int) -> str:
    """Return a value with exactly ``n`` words, so count_tokens == n."""
    return " ".join(f"x{i}" for i in range(n))


def _build_compressor(config: CompressionConfig) -> TrajectoryCompressor:
    """Build a TrajectoryCompressor via __new__ with deterministic stubs.

    - ``count_tokens`` is replaced by a word-count stub (instance attribute
      shadows the class method) so token numbers are fully deterministic.
    - ``_generate_summary`` / ``_generate_summary_async`` return a fixed summary
      via stubbed clients; no network and no real tokenizer.
    """
    comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
    comp.config = config
    comp.logger = MagicMock()
    comp._use_call_llm = False

    # Sync client stub.
    comp.client = MagicMock()
    comp.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: test summary"))]
    )

    # Async client stub (AsyncMock so await create(...) resolves).
    async_conn = MagicMock()
    async_conn.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: test summary"))]
        )
    )
    comp._get_async_client = MagicMock(return_value=async_conn)

    # Deterministic token counter: 1 token per whitespace-delimited word,
    # 0 for empty strings (matches TrajectoryCompressor.count_tokens contract).
    comp.count_tokens = lambda text: len(text.split()) if text else 0
    return comp


def _compress(comp: TrajectoryCompressor, trajectory, branch: str):
    """Run one compression branch. ``branch`` is 'sync' or 'async'."""
    if branch == "async":
        return asyncio.run(comp.compress_trajectory_async(trajectory))
    return comp.compress_trajectory(trajectory)


def _happy_path_trajectory():
    """A 12-turn trajectory whose middle turns (indices 4..9) are compressible.

    Protected head = first system/human/gpt/tool (indices 0-3); protected tail
    = last 2 (indices 10-11). Region is [4, 10). With a word-count token stub
    this totals 54 tokens; target 20 forces compression.
    """
    return (
        [
            {"from": "system", "value": "SYSTEM"},
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "gpt0"},
            {"from": "tool", "value": "tool0"},
        ]
        + [{"from": "gpt", "value": _words(8)} for _ in range(6)]
        + [
            {"from": "tool", "value": "toolX"},
            {"from": "gpt", "value": "gpt11"},
        ]
    )


# ---------------------------------------------------------------------------
# compress_trajectory / compress_trajectory_async — scenario matrix
# ---------------------------------------------------------------------------

class TestCompressSkipsUnderTarget:
    """total_tokens <= target_max_tokens -> unchanged, no API call."""

    @pytest.mark.parametrize("branch", ["sync", "async"])
    def test_returns_same_trajectory_and_skip_metrics(self, branch):
        comp = _build_compressor(CompressionConfig(target_max_tokens=1000, protect_last_n_turns=2))
        traj = [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "hi"},
        ]

        result, metrics = _compress(comp, traj, branch)

        assert result is traj
        assert metrics.skipped_under_target is True
        assert metrics.was_compressed is False
        assert metrics.compression_ratio == 1.0
        assert metrics.original_tokens == comp.count_trajectory_tokens(traj)
        comp.client.chat.completions.create.assert_not_called()
        comp._get_async_client.assert_not_called()


class TestCompressEmptyRegion:
    """compress_start >= compress_end -> unchanged, still_over_limit True."""

    @pytest.mark.parametrize("branch", ["sync", "async"])
    def test_all_turns_protected_leaves_empty_region(self, branch):
        # target 2 keeps the trajectory over limit, protect_last_n_turns >= len
        # protects every turn so the head/tail brackets leave no compressible
        # region (compress_start == compress_end).
        comp = _build_compressor(
            CompressionConfig(target_max_tokens=2, summary_target_tokens=5, protect_last_n_turns=10)
        )
        traj = [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "gpt0"},
            {"from": "tool", "value": "tool0"},
        ]

        result, metrics = _compress(comp, traj, branch)

        assert result is traj
        assert metrics.was_compressed is False
        assert metrics.still_over_limit is True
        assert metrics.compression_ratio == 1.0
        comp.client.chat.completions.create.assert_not_called()
        comp._get_async_client.assert_not_called()


class TestCompressHappyPath:
    """A real mid-trajectory compression: summary inserted, notice appended."""

    @pytest.mark.parametrize("branch", ["sync", "async"])
    def test_compresses_middle_region(self, branch):
        comp = _build_compressor(
            CompressionConfig(
                target_max_tokens=20,
                summary_target_tokens=5,
                protect_last_n_turns=2,
                add_summary_notice=True,
                summary_notice_text="[notice]",
            )
        )
        traj = _happy_path_trajectory()

        result, metrics = _compress(comp, traj, branch)

        # 12 original turns -> 4 head + 1 summary + 3 tail = 8.
        assert len(result) == 8
        assert metrics.original_turns == 12
        assert metrics.original_tokens == 54

        # Head system turn got the notice appended.
        assert result[0]["from"] == "system"
        assert result[0]["value"] == "SYSTEM[notice]"

        # Summary inserted as a single human turn with the context prefix.
        summary_turns = [t for t in result if t["value"].startswith("[CONTEXT SUMMARY]:")]
        assert len(summary_turns) == 1
        assert summary_turns[0]["from"] == "human"
        assert result[4]["value"].startswith("[CONTEXT SUMMARY]:")

        # Tail turns preserved verbatim.
        assert result[5]["value"] == _words(8)
        assert result[6]["value"] == "toolX"
        assert result[7]["value"] == "gpt11"

        # Metrics invariants.
        assert metrics.was_compressed is True
        assert metrics.turns_removed == metrics.original_turns - metrics.compressed_turns == 4
        assert metrics.tokens_saved > 0
        assert metrics.compression_ratio < 1.0
        assert metrics.turns_compressed_start_idx == 4
        assert metrics.turns_compressed_end_idx == 9
        assert metrics.turns_in_compressed_region == 5
        assert metrics.still_over_limit is False

    def test_add_summary_notice_false_leaves_system_untouched(self):
        comp = _build_compressor(
            CompressionConfig(
                target_max_tokens=20,
                summary_target_tokens=5,
                protect_last_n_turns=2,
                add_summary_notice=False,
                summary_notice_text="[notice]",
            )
        )
        traj = _happy_path_trajectory()

        result, metrics = _compress(comp, traj, "sync")

        assert result[0]["from"] == "system"
        assert result[0]["value"] == "SYSTEM"
        assert metrics.was_compressed is True


class TestCompressWholeRegionFallback:
    """Region tokens below target but region non-empty -> whole region compressed."""

    @pytest.mark.parametrize("branch", ["sync", "async"])
    def test_region_sum_below_target_compresses_entire_region(self, branch):
        # Region [4, 10) sums to 18 tokens, below target_tokens_to_compress
        # (higher than 18), so the whole region is consumed. The tail boundary
        # lands on a clean (gpt) turn so compress_until stays at compress_end.
        comp = _build_compressor(
            CompressionConfig(target_max_tokens=10, summary_target_tokens=5, protect_last_n_turns=2)
        )
        traj = (
            [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "gpt0"},
                {"from": "tool", "value": "tool0"},
            ]
            + [{"from": "gpt", "value": _words(3)} for _ in range(6)]
            + [
                {"from": "gpt", "value": "gpt10"},
                {"from": "gpt", "value": "gpt11"},
            ]
        )

        result, metrics = _compress(comp, traj, branch)

        assert metrics.was_compressed is True
        assert metrics.turns_compressed_start_idx == 4
        assert metrics.turns_compressed_end_idx == 10
        assert metrics.turns_in_compressed_region == 6
        # Only the 4 head turns remain before the summary.
        assert result[4]["from"] == "human"


class TestCompressSnapCollapse:
    """Tail boundary snaps back onto the region start -> unchanged, still over."""

    @pytest.mark.parametrize("branch", ["sync", "async"])
    def test_collapse_returns_unchanged(self, branch):
        # Region [4, 7) is entirely tool turns. compress_until ends at the tail
        # boundary (index 7), which is itself a tool turn, so _snap_boundary
        # slides backward and collapses onto compress_start (4).
        comp = _build_compressor(
            CompressionConfig(target_max_tokens=10, summary_target_tokens=5, protect_last_n_turns=1)
        )
        traj = (
            [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "gpt0"},
                {"from": "tool", "value": "tool0"},
            ]
            + [{"from": "tool", "value": _words(5)} for _ in range(4)]
        )

        result, metrics = _compress(comp, traj, branch)

        assert result is traj
        assert metrics.was_compressed is False
        assert metrics.still_over_limit is True
        comp.client.chat.completions.create.assert_not_called()
        comp._get_async_client.assert_not_called()


class TestCompressTinyRegionGuard:
    """Region sum <= summary_target_tokens -> compression would grow, skip."""

    @pytest.mark.parametrize("branch", ["sync", "async"])
    def test_tiny_region_returns_unchanged(self, branch):
        # Region [4, 7) sums to 3 tokens <= summary_target_tokens (5); the
        # summary that would replace it costs 5 tokens, so the function returns
        # the trajectory untouched and still over limit.
        comp = _build_compressor(
            CompressionConfig(target_max_tokens=3, summary_target_tokens=5, protect_last_n_turns=1)
        )
        traj = [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "gpt0"},
            {"from": "tool", "value": "tool0"},
            {"from": "gpt", "value": "g"},
            {"from": "tool", "value": "t1"},
            {"from": "tool", "value": "t2"},
            {"from": "gpt", "value": "gt"},
        ]

        result, metrics = _compress(comp, traj, branch)

        assert result is traj
        assert metrics.was_compressed is False
        assert metrics.still_over_limit is True
        comp.client.chat.completions.create.assert_not_called()
        comp._get_async_client.assert_not_called()


class TestCompressHappyPathMetrics:
    """Exact metrics relationship on the happy path."""

    def test_compression_ratio_relationship(self):
        comp = _build_compressor(
            CompressionConfig(
                target_max_tokens=20,
                summary_target_tokens=5,
                protect_last_n_turns=2,
                add_summary_notice=True,
                summary_notice_text="[notice]",
            )
        )
        traj = _happy_path_trajectory()

        result, metrics = _compress(comp, traj, "sync")

        assert metrics.was_compressed is True
        assert metrics.compression_ratio == pytest.approx(
            metrics.compressed_tokens / metrics.original_tokens
        )
        assert metrics.compressed_tokens == comp.count_trajectory_tokens(result)
        assert metrics.compressed_tokens < metrics.original_tokens
        assert metrics.still_over_limit is False
        assert metrics.tokens_saved == metrics.original_tokens - metrics.compressed_tokens


# ---------------------------------------------------------------------------
# _extract_turn_content_for_summary (564-588)
# ---------------------------------------------------------------------------

class TestExtractTurnContent:
    def setup_method(self):
        self.comp = _build_compressor(CompressionConfig())

    def test_truncates_long_values_around_mid_marker(self):
        big = "A" * 3001
        content = self.comp._extract_turn_content_for_summary(
            [{"from": "gpt", "value": big}], 0, 1
        )

        assert content.startswith("[Turn 0 - GPT]:\n")
        assert "A" * 1500 in content
        assert "A" * 500 in content
        assert ("A" * 1500 + "\n...[truncated]...\n" + "A" * 500) in content
        assert content.endswith("A" * 500)

    def test_joins_turns_with_index_role_headers(self):
        content = self.comp._extract_turn_content_for_summary(
            [
                {"from": "system", "value": "hello"},
                {"from": "human", "value": "world"},
            ],
            0,
            2,
        )

        assert content == "[Turn 0 - SYSTEM]:\nhello\n\n[Turn 1 - HUMAN]:\nworld"

    def test_non_truncating_values_kept_verbatim(self):
        content = self.comp._extract_turn_content_for_summary(
            [{"from": "gpt", "value": "short"}], 0, 1
        )
        assert content == "[Turn 0 - GPT]:\nshort"


# ---------------------------------------------------------------------------
# _find_protected_indices (477-523)
# ---------------------------------------------------------------------------

class TestFindProtectedIndices:
    def setup_method(self):
        self.comp = _build_compressor(CompressionConfig())

    def test_tracks_first_occurrence_of_each_role(self):
        # protect_last_n_turns must be below n (default is 4) so middle turns
        # are not swallowed by the tail protection.
        comp = _build_compressor(CompressionConfig(protect_last_n_turns=2))
        traj = [
            {"from": "system", "value": "a"},
            {"from": "human", "value": "b"},
            {"from": "gpt", "value": "c"},
            {"from": "tool", "value": "d"},
            {"from": "gpt", "value": "e"},
            {"from": "gpt", "value": "f"},
            {"from": "human", "value": "g"},
            {"from": "tool", "value": "h"},
        ]

        protected, compress_start, compress_end = comp._find_protected_indices(traj)

        # First occurrences + last 2 (indices 6, 7).
        assert {0, 1, 2, 3, 6, 7} <= protected
        assert 4 not in protected
        assert 5 not in protected
        assert compress_start == 4
        assert compress_end == 6

    def test_protect_first_flags_off_disables_head_protection(self):
        comp = _build_compressor(
            CompressionConfig(
                protect_first_system=False,
                protect_first_human=False,
                protect_first_gpt=False,
                protect_first_tool=False,
                protect_last_n_turns=2,
            )
        )
        traj = [
            {"from": "system", "value": "a"},
            {"from": "human", "value": "b"},
            {"from": "gpt", "value": "c"},
            {"from": "tool", "value": "d"},
            {"from": "gpt", "value": "e"},
            {"from": "human", "value": "f"},
        ]

        protected, compress_start, compress_end = comp._find_protected_indices(traj)

        assert protected == {4, 5}
        assert compress_start == 0
        assert compress_end == 4

    def test_empty_trajectory_returns_empty_set(self):
        protected, compress_start, compress_end = self.comp._find_protected_indices([])

        assert protected == set()
        assert compress_start == 0
        assert compress_end == 0

    def test_protect_last_n_zero_disables_tail_protection(self):
        comp = _build_compressor(CompressionConfig(protect_last_n_turns=0))
        traj = [
            {"from": "system", "value": "a"},
            {"from": "human", "value": "b"},
            {"from": "gpt", "value": "c"},
            {"from": "tool", "value": "d"},
            {"from": "gpt", "value": "e"},
        ]

        protected, _, _ = comp._find_protected_indices(traj)

        assert protected == {0, 1, 2, 3}
        assert 4 not in protected

    def test_protect_last_n_ge_length_protects_every_turn(self):
        comp = _build_compressor(CompressionConfig(protect_last_n_turns=10))
        traj = [
            {"from": "system", "value": "a"},
            {"from": "human", "value": "b"},
            {"from": "gpt", "value": "c"},
            {"from": "tool", "value": "d"},
        ]

        protected, _, _ = comp._find_protected_indices(traj)

        assert protected == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# _snap_boundary (539-562)
# ---------------------------------------------------------------------------

class TestSnapBoundary:
    def test_forward_snap_onto_clean_tool_following_turn(self):
        traj = [
            {"from": "system", "value": "a"},
            {"from": "human", "value": "b"},
            {"from": "tool", "value": "c"},
            {"from": "tool", "value": "d"},
            {"from": "gpt", "value": "e"},
        ]
        assert TrajectoryCompressor._snap_boundary(traj, 2, 1, 4) == 4

    def test_backward_snap_when_tail_starts_with_tool(self):
        traj = [
            {"from": "human", "value": "b"},
            {"from": "gpt", "value": "e"},
            {"from": "tool", "value": "c"},
        ]
        assert TrajectoryCompressor._snap_boundary(traj, 2, 1, 2) == 1

    def test_clean_boundary_returns_same_index(self):
        traj = [
            {"from": "system", "value": "a"},
            {"from": "human", "value": "b"},
            {"from": "gpt", "value": "e"},
        ]
        assert TrajectoryCompressor._snap_boundary(traj, 1, 1, 3) == 1

    def test_clamps_forward_to_max_idx(self):
        traj = [
            {"from": "gpt", "value": "a"},
            {"from": "tool", "value": "b"},
            {"from": "gpt", "value": "c"},
        ]
        assert TrajectoryCompressor._snap_boundary(traj, 1, 0, 2) == 2

    def test_clamps_backward_to_min_idx(self):
        traj = [
            {"from": "tool", "value": "a"},
            {"from": "tool", "value": "b"},
            {"from": "tool", "value": "c"},
        ]
        assert TrajectoryCompressor._snap_boundary(traj, 2, 1, 2) == 1


# ---------------------------------------------------------------------------
# Async parity (compress_trajectory_async) — explicit named tests
# ---------------------------------------------------------------------------

class TestCompressAsyncParity:
    """Direct async assertions, including the async client wiring."""

    @pytest.mark.asyncio
    async def test_async_happy_path_uses_get_async_client(self):
        comp = _build_compressor(
            CompressionConfig(
                target_max_tokens=20,
                summary_target_tokens=5,
                protect_last_n_turns=2,
                add_summary_notice=True,
                summary_notice_text="[notice]",
            )
        )
        traj = _happy_path_trajectory()

        result, metrics = await comp.compress_trajectory_async(traj)

        comp._get_async_client.assert_called_once()
        assert metrics.was_compressed is True
        assert metrics.compression_ratio < 1.0
        assert result[4]["value"].startswith("[CONTEXT SUMMARY]:")
