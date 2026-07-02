"""Tests for tools.tool_output_limits.

Covers:
1. Default values when no config is provided.
2. Config override picks up user-supplied max_bytes / max_lines /
   max_line_length.
3. Malformed values (None, negative, wrong type) fall back to defaults
   rather than raising.
4. Integration: the helpers return what the terminal_tool and
   file_operations call paths will actually consume.

Port-tracking: anomalyco/opencode PR #23770
(feat(truncate): allow configuring tool output truncation limits).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tools import tool_output_limits as tol


@pytest.fixture(autouse=True)
def _reset_limits_cache():
    """get_tool_output_limits() now memoizes its result for the process
    lifetime, so each test must start from a clean cache to observe the
    config value it patches in."""
    tol._reset_tool_output_limits_cache()
    yield
    tol._reset_tool_output_limits_cache()


class TestDefaults:
    def test_defaults_match_previous_hardcoded_values(self):
        assert tol.DEFAULT_MAX_BYTES == 50_000
        assert tol.DEFAULT_MAX_LINES == 2000
        assert tol.DEFAULT_MAX_LINE_LENGTH == 2000

    def test_get_limits_returns_defaults_when_config_missing(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            limits = tol.get_tool_output_limits()
        assert limits == {
            "max_bytes": tol.DEFAULT_MAX_BYTES,
            "max_lines": tol.DEFAULT_MAX_LINES,
            "max_line_length": tol.DEFAULT_MAX_LINE_LENGTH,
        }

    def test_get_limits_returns_defaults_when_config_not_a_dict(self):
        # load_config should always return a dict but be defensive anyway.
        with patch("hermes_cli.config.load_config", return_value="not a dict"):
            limits = tol.get_tool_output_limits()
        assert limits["max_bytes"] == tol.DEFAULT_MAX_BYTES

    def test_get_limits_returns_defaults_when_load_config_raises(self):
        def _boom():
            raise RuntimeError("boom")

        with patch("hermes_cli.config.load_config", side_effect=_boom):
            limits = tol.get_tool_output_limits()
        assert limits["max_lines"] == tol.DEFAULT_MAX_LINES


class TestOverrides:
    def test_user_config_overrides_all_three(self):
        cfg = {
            "tool_output": {
                "max_bytes": 100_000,
                "max_lines": 5000,
                "max_line_length": 4096,
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = tol.get_tool_output_limits()
        assert limits == {
            "max_bytes": 100_000,
            "max_lines": 5000,
            "max_line_length": 4096,
        }

    def test_partial_override_preserves_other_defaults(self):
        cfg = {"tool_output": {"max_bytes": 200_000}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = tol.get_tool_output_limits()
        assert limits["max_bytes"] == 200_000
        assert limits["max_lines"] == tol.DEFAULT_MAX_LINES
        assert limits["max_line_length"] == tol.DEFAULT_MAX_LINE_LENGTH

    def test_section_not_a_dict_falls_back(self):
        cfg = {"tool_output": "nonsense"}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = tol.get_tool_output_limits()
        assert limits["max_bytes"] == tol.DEFAULT_MAX_BYTES


class TestCoercion:
    @pytest.mark.parametrize("bad", [None, "not a number", -1, 0, [], {}])
    def test_invalid_values_fall_back_to_defaults(self, bad):
        cfg = {"tool_output": {"max_bytes": bad, "max_lines": bad, "max_line_length": bad}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = tol.get_tool_output_limits()
        assert limits["max_bytes"] == tol.DEFAULT_MAX_BYTES
        assert limits["max_lines"] == tol.DEFAULT_MAX_LINES
        assert limits["max_line_length"] == tol.DEFAULT_MAX_LINE_LENGTH

    def test_string_integer_is_coerced(self):
        cfg = {"tool_output": {"max_bytes": "75000"}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            limits = tol.get_tool_output_limits()
        assert limits["max_bytes"] == 75_000


class TestShortcuts:
    def test_individual_accessors_delegate_to_get_tool_output_limits(self):
        cfg = {
            "tool_output": {
                "max_bytes": 111,
                "max_lines": 222,
                "max_line_length": 333,
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert tol.get_max_bytes() == 111
            assert tol.get_max_lines() == 222
            assert tol.get_max_line_length() == 333


class TestDefaultConfigHasSection:
    """The DEFAULT_CONFIG in hermes_cli.config must expose tool_output so
    that ``hermes setup`` and default installs stay in sync with the
    helpers here."""

    def test_default_config_contains_tool_output_section(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert "tool_output" in DEFAULT_CONFIG
        section = DEFAULT_CONFIG["tool_output"]
        assert isinstance(section, dict)
        assert section["max_bytes"] == tol.DEFAULT_MAX_BYTES
        assert section["max_lines"] == tol.DEFAULT_MAX_LINES
        assert section["max_line_length"] == tol.DEFAULT_MAX_LINE_LENGTH


class TestIntegrationReadPagination:
    """normalize_read_pagination uses get_max_lines() — verify the plumbing."""

    def test_pagination_limit_clamped_by_config_value(self):
        from tools.file_operations import normalize_read_pagination
        cfg = {"tool_output": {"max_lines": 50}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            offset, limit = normalize_read_pagination(offset=1, limit=1000)
        # limit should have been clamped to 50 (the configured max_lines)
        assert limit == 50
        assert offset == 1

    def test_pagination_default_when_config_missing(self):
        from tools.file_operations import normalize_read_pagination
        with patch("hermes_cli.config.load_config", return_value={}):
            offset, limit = normalize_read_pagination(offset=10, limit=100000)
        # Clamped to default MAX_LINES (2000).
        assert limit == tol.DEFAULT_MAX_LINES
        assert offset == 10


class TestUsageGuardCompactMode:
    def test_compact_terminal_output_preserves_exit_code_and_edges(self):
        from hermes_cli.usage_guard import compact_terminal_output_after_warning

        output = "\n".join(f"line-{i:04d}" for i in range(800))

        compacted = compact_terminal_output_after_warning(
            output,
            exit_code=7,
            active=True,
            max_chars=500,
        )

        assert len(compacted) <= 700
        assert "exit_code=7" in compacted
        assert "OUTPUT COMPACTED AFTER USAGE WARNING" in compacted
        assert "line-0000" in compacted
        assert "line-0799" in compacted

    def test_broad_read_denial_requires_narrow_range(self):
        from hermes_cli.usage_guard import read_request_denial_after_warning

        denial = read_request_denial_after_warning(
            path="large.py",
            offset=1,
            limit=500,
            active=True,
        )

        assert denial is not None
        assert denial["usage_guard"] == "active"
        assert "narrow" in denial["error"].lower()


class TestPerModeOutputPolicies:
    def test_builder_mode_uses_normal_caps_until_usage_warning(self):
        builder = tol.get_mode_output_policy("builder", usage_guard_active=False)
        builder_warned = tol.get_mode_output_policy("builder", usage_guard_active=True)

        assert builder.terminal_max_chars == tol.DEFAULT_MAX_BYTES
        assert builder.read_max_lines == tol.DEFAULT_MAX_LINES
        assert builder_warned.terminal_max_chars < builder.terminal_max_chars
        assert builder_warned.read_max_lines < builder.read_max_lines

    def test_reviewer_mode_caps_only_after_usage_warning(self):
        reviewer = tol.get_mode_output_policy("reviewer", usage_guard_active=False)
        reviewer_warned = tol.get_mode_output_policy("reviewer", usage_guard_active=True)

        assert reviewer.terminal_max_chars == tol.DEFAULT_MAX_BYTES
        assert reviewer_warned.terminal_max_chars < reviewer.terminal_max_chars
        assert reviewer_warned.search_max_results <= 25

    def test_synthesizer_and_monitor_modes_are_always_bounded(self):
        synthesizer = tol.get_mode_output_policy("synthesizer", usage_guard_active=False)
        monitor = tol.get_mode_output_policy("monitor", usage_guard_active=False)

        assert synthesizer.terminal_max_chars < tol.DEFAULT_MAX_BYTES
        assert synthesizer.read_max_lines <= 200
        assert monitor.terminal_max_chars <= synthesizer.terminal_max_chars
        assert monitor.search_max_results <= synthesizer.search_max_results

    def test_resolve_mode_from_environment(self, monkeypatch):
        monkeypatch.setenv("HERMES_TOOL_OUTPUT_MODE", "synthesis")

        assert tol.resolve_tool_output_mode() == "synthesizer"

    def test_synthesizer_mode_compacts_terminal_without_usage_warning(self):
        from hermes_cli.usage_guard import compact_terminal_output_after_warning

        output = "\n".join(f"row-{i:04d}" for i in range(2000))

        compacted = compact_terminal_output_after_warning(
            output,
            exit_code=0,
            active=False,
            mode="synthesizer",
        )

        assert len(compacted) < len(output)
        assert "OUTPUT COMPACTED" in compacted
        assert "row-0000" in compacted
        assert "row-1999" in compacted

    def test_synthesizer_terminal_compaction_names_command_and_artifacts(self):
        from hermes_cli.usage_guard import compact_terminal_output_after_warning

        artifact = "C:/workspace/dist/final-report.md"
        output = (
            f"created artifact {artifact}\n"
            + "\n".join(f"verbose-row-{i:04d}" for i in range(2000))
            + "\nall done"
        )

        compacted = compact_terminal_output_after_warning(
            output,
            exit_code=0,
            active=False,
            mode="synthesizer",
            command="python build_report.py --out C:/workspace/dist/final-report.md",
        )

        assert "SYNTHESIZER TERMINAL SUMMARY" in compacted
        assert "command=python build_report.py --out C:/workspace/dist/final-report.md" in compacted
        assert "exit_code=0" in compacted
        assert artifact in compacted
        assert "verbose-row-0000" in compacted
        assert "all done" in compacted

    def test_audit_first_large_read_requires_range_or_symbol(self):
        from hermes_cli.usage_guard import read_request_denial_after_warning

        denial = read_request_denial_after_warning(
            path="large.py",
            offset=1,
            limit=500,
            active=False,
            mode="synthesizer",
        )

        assert denial is not None
        assert denial["mode"] == "synthesizer"
        assert "audit-first" in denial["error"]

    def test_synthesizer_broad_search_requests_missing_curated_evidence(self):
        from hermes_cli.usage_guard import search_request_denial_after_warning

        denial = search_request_denial_after_warning(
            pattern="*",
            target="content",
            path=".",
            file_glob=None,
            limit=20,
            active=False,
            mode="synthesizer",
        )

        assert denial is not None
        assert denial["mode"] == "synthesizer"
        assert "curated parent summaries" in denial["error"]
        assert "missing-evidence request" in denial["error"]

    def test_synthesizer_instruction_and_missing_evidence_packet_contract(self):
        from hermes_cli.usage_guard import (
            synthesizer_missing_evidence_request,
            synthesizer_mode_instruction,
            validate_synthesizer_evidence_packet,
        )

        instruction = synthesizer_mode_instruction(task_id="t_synth")
        assert "curated parent summaries" in instruction
        assert "explicit artifacts" in instruction
        assert "capped worker logs" in instruction
        assert "missing-evidence request" in instruction
        assert "Do not run broad find" in instruction

        packet = synthesizer_missing_evidence_request(
            task_id="t_synth",
            missing=["verifier gate metadata"],
            checked_sources=["parent summaries", "artifacts"],
        )
        assert packet["task_id"] == "t_synth"
        assert packet["missing_evidence"] == ["verifier gate metadata"]
        assert packet["checked_sources"] == ["parent summaries", "artifacts"]
        assert validate_synthesizer_evidence_packet(packet) == []

        bad = dict(packet)
        bad["checked_sources"] = ["entire repository"]
        assert "unsupported synthesizer evidence source: entire repository" in (
            validate_synthesizer_evidence_packet(bad)
        )

    def test_reviewer_after_warning_caps_search_results(self):
        from hermes_cli.usage_guard import search_request_denial_after_warning

        denial = search_request_denial_after_warning(
            pattern="TODO",
            target="content",
            path=".",
            file_glob="*.py",
            limit=50,
            active=True,
            mode="reviewer",
        )

        assert denial is not None
        assert denial["mode"] == "reviewer"
        assert denial["max_limit"] <= 25
        assert "verdict comment first" in denial["error"]

    def test_reviewer_after_warning_blocks_broad_read_with_verdict_instruction(self):
        from hermes_cli.usage_guard import read_request_denial_after_warning

        denial = read_request_denial_after_warning(
            path="large.py",
            offset=1,
            limit=500,
            active=True,
            mode="reviewer",
        )

        assert denial is not None
        assert denial["mode"] == "reviewer"
        assert "verdict comment first" in denial["error"]

    def test_reviewer_verdict_packet_requires_categories_and_one_followup(self):
        from hermes_cli.usage_guard import (
            REVIEWER_VERDICT_CATEGORIES,
            reviewer_verdict_instruction,
            validate_reviewer_verdict_packet,
        )

        assert REVIEWER_VERDICT_CATEGORIES == (
            "actionable",
            "trade-off",
            "contract-misread",
            "noise",
        )
        instruction = reviewer_verdict_instruction(task_id="t_review")
        for category in REVIEWER_VERDICT_CATEGORIES:
            assert category in instruction
        assert "one narrow follow-up" in instruction
        assert "scoped acceptance" in instruction

        valid = {
            "task_id": "t_review",
            "categories": {category: [] for category in REVIEWER_VERDICT_CATEGORIES},
            "follow_up": {"scope": "tests/test_target.py", "question": "rerun one test"},
        }
        assert validate_reviewer_verdict_packet(valid) == []

        too_many_followups = dict(valid)
        too_many_followups["follow_up"] = [
            {"scope": "a.py", "question": "first"},
            {"scope": "b.py", "question": "second"},
        ]
        assert "at most one narrow follow-up" in validate_reviewer_verdict_packet(
            too_many_followups
        )

        missing = {"task_id": "t_review", "categories": {"actionable": []}}
        errors = validate_reviewer_verdict_packet(missing)
        assert "missing verdict category: trade-off" in errors

    def test_reviewer_after_warning_blocks_broad_delegation(self, monkeypatch):
        from hermes_cli.usage_guard import reset_usage_guard_for_tests
        from tools import delegate_tool as dt

        reset_usage_guard_for_tests()
        monkeypatch.setenv("HERMES_USAGE_GUARD_ACTIVE", "1")
        monkeypatch.setenv("HERMES_TOOL_OUTPUT_MODE", "reviewer")

        out = dt.delegate_task(goal="review the entire repository for issues")
        data = json.loads(out)

        assert data.get("error")
        assert "verdict comment first" in data["error"]
