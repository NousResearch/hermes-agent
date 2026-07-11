"""Tests for agent/tool_repair_stats.py — repair observability module."""

from __future__ import annotations

import threading


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from agent.tool_repair_stats import (
    RepairPattern,
    ToolRepairStats,
    get_stats,
    record_repair,
    get_current_model,
    set_current_model,
)


# ---------------------------------------------------------------------------
# RepairPattern enum
# ---------------------------------------------------------------------------

class TestRepairPattern:
    """Verify the enum covers the known failure modes."""

    def test_has_core_patterns(self):
        core = [
            "EMPTY_ARGS", "NONE_LITERAL", "CONTROL_CHAR_ESCAPE",
            "TRAILING_COMMA", "UNREPAIRABLE",
            "BARE_STRING_WRAP", "BARE_OBJECT_WRAP",
            "STRING_TO_INT", "STRING_TO_BOOL",
        ]
        for name in core:
            assert hasattr(RepairPattern, name), f"Missing pattern: {name}"

    def test_string_values(self):
        """Enum values must be plain strings for serialization."""
        for p in RepairPattern:
            assert isinstance(p.value, str)


# ---------------------------------------------------------------------------
# ToolRepairStats — basic operations
# ---------------------------------------------------------------------------

class TestToolRepairStatsBasic:

    def test_empty_stats(self):
        stats = ToolRepairStats()
        assert stats.total() == 0
        assert stats.all_models() == {}
        assert "No tool-call repairs" in stats.summary()

    def test_record_single_event(self):
        stats = ToolRepairStats()
        stats.record(RepairPattern.BARE_STRING_WRAP, "terminal", "deepseek-v4")
        assert stats.total() == 1
        assert stats.by_model("deepseek-v4") == {"bare_string_wrap": 1}

    def test_record_multiple_patterns(self):
        stats = ToolRepairStats()
        stats.record(RepairPattern.BARE_STRING_WRAP, "terminal", "ds")
        stats.record(RepairPattern.BARE_STRING_WRAP, "read_file", "ds")
        stats.record(RepairPattern.UNREPAIRABLE, "patch", "ds")
        assert stats.total() == 3
        by_model = stats.by_model("ds")
        assert by_model["bare_string_wrap"] == 2
        assert by_model["unrepairable"] == 1

    def test_multiple_models(self):
        stats = ToolRepairStats()
        stats.record(RepairPattern.BARE_STRING_WRAP, "t", "model-a")
        stats.record(RepairPattern.UNREPAIRABLE, "t", "model-b")
        all_m = stats.all_models()
        assert "model-a" in all_m
        assert "model-b" in all_m

    def test_top_patterns(self):
        stats = ToolRepairStats()
        for _ in range(10):
            stats.record(RepairPattern.BARE_STRING_WRAP, "t", "m")
        for _ in range(3):
            stats.record(RepairPattern.UNREPAIRABLE, "t", "m")
        top = stats.top_patterns(2)
        assert top[0][0] == "bare_string_wrap"
        assert top[0][1] == 10

    def test_recent_events(self):
        stats = ToolRepairStats()
        for i in range(5):
            stats.record(RepairPattern.OTHER, f"tool_{i}", "m")
        recent = stats.recent(3)
        assert len(recent) == 3
        assert recent[-1].tool_name == "tool_4"

    def test_reset(self):
        stats = ToolRepairStats()
        stats.record(RepairPattern.BARE_STRING_WRAP, "t", "m")
        assert stats.total() == 1
        stats.reset()
        assert stats.total() == 0
        assert stats.all_models() == {}


# ---------------------------------------------------------------------------
# Ring buffer cap
# ---------------------------------------------------------------------------

class TestRingBuffer:

    def test_bounded_memory(self):
        stats = ToolRepairStats()
        # Override max for faster test
        stats._MAX_EVENTS = 100
        for i in range(150):
            stats.record(RepairPattern.OTHER, "t", "m")
        assert stats.total() == 100


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_recording(self):
        stats = ToolRepairStats()
        errors = []

        def record_many(n: int):
            try:
                for _ in range(n):
                    stats.record(RepairPattern.BARE_STRING_WRAP, "t", "m")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many, args=(100,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert stats.total() == 1000
        assert stats.by_model("m")["bare_string_wrap"] == 1000


# ---------------------------------------------------------------------------
# Failure resilience
# ---------------------------------------------------------------------------

class TestFailureResilience:

    def test_record_never_raises(self):
        """record() must NEVER raise, even with garbage input."""
        stats = ToolRepairStats()
        # Should not raise
        stats.record(None, None, None)  # type: ignore[arg-type]
        stats.record("not_an_enum", "", "")

    def test_import_failure_noop(self):
        """When _record_repair is None, _stat should be a no-op."""
        # This is tested indirectly — if the import fails in
        # message_sanitization.py, the fallback record_repair does nothing.
        # Here we just verify the module-level record_repair works.
        record_repair(RepairPattern.OTHER, "test", "test")  # should not raise


# ---------------------------------------------------------------------------
# Summary format
# ---------------------------------------------------------------------------

class TestSummary:

    def test_summary_contains_model_name(self):
        stats = ToolRepairStats()
        stats.record(RepairPattern.BARE_STRING_WRAP, "terminal", "deepseek-v4")
        s = stats.summary()
        assert "deepseek-v4" in s
        assert "bare_string_wrap" in s

    def test_summary_contains_totals(self):
        stats = ToolRepairStats()
        stats.record(RepairPattern.BARE_STRING_WRAP, "t", "m")
        stats.record(RepairPattern.UNREPAIRABLE, "t", "m")
        s = stats.summary()
        assert "Total events: 2" in s


# ---------------------------------------------------------------------------
# Model context (set_current_model / get_current_model)
# ---------------------------------------------------------------------------

class TestModelContext:

    def test_set_and_get(self):
        set_current_model("deepseek/deepseek-v4-pro")
        assert get_current_model() == "deepseek/deepseek-v4-pro"

    def test_default_is_unknown(self):
        import agent.tool_repair_stats as mod
        mod._current_model = "unknown"
        assert get_current_model() == "unknown"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:

    def test_get_stats_returns_same_instance(self):
        a = get_stats()
        b = get_stats()
        assert a is b

    def test_singleton_survives_reset(self):
        stats = get_stats()
        stats.record(RepairPattern.OTHER, "t", "m")
        stats.reset()
        assert get_stats().total() == 0
        # Same object
        assert get_stats() is stats
