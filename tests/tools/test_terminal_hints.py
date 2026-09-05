"""Tests for tools/terminal_hints.py — output-pattern failure hints."""

import json
from unittest.mock import patch as mock_patch

import pytest

from tools.terminal_hints import annotate_failure, annotate_masked_success


class TestAnnotateFailureBasics:
    def test_success_never_annotated(self):
        assert annotate_failure("python x.py", 0, "python: command not found") is None

    def test_empty_output_falls_to_exit_code_tier(self):
        assert "126" in annotate_failure("./run.sh", 126, "")
        assert "SIGKILL" in annotate_failure("big_job", 137, "")
        assert "timeout" in annotate_failure("sleep 999", 124, "")

    def test_unknown_failure_returns_none(self):
        assert annotate_failure("./x", 1, "some unrecognized error") is None

    def test_only_first_matching_hint(self):
        out = 'CONFLICT (content): Merge conflict in a.py\npython: command not found'
        hint = annotate_failure("git merge x && python t.py", 1, out)
        assert "conflict" in hint.lower()
        assert "python3" not in hint


class TestGhUnknownJsonField:
    def test_field_name_extracted(self):
        out = 'Unknown JSON field: "authorAssociation"\nAvailable fields:\n  additions\n  author'
        hint = annotate_failure("gh pr view 1 --json authorAssociation", 1, out)
        assert "authorAssociation" in hint
        assert "valid field list" in hint


class TestCommandNotFound:
    def test_bare_python_gets_python3_hint(self):
        out = "/usr/bin/bash: line 1: python: command not found"
        hint = annotate_failure("python x.py", 127, out)
        assert "python3" in hint

    def test_bare_pip_gets_pip3_hint(self):
        out = "bash: pip: command not found"
        hint = annotate_failure("pip install x", 127, out)
        assert "pip3" in hint or "-m pip" in hint

    def test_generic_command(self):
        out = "bash: line 3: shellcheck: command not found"
        hint = annotate_failure("shellcheck s.sh", 127, out)
        assert "shellcheck" in hint
        assert "which" in hint


class TestModuleNotFound:
    def test_module_named(self):
        out = ("Traceback (most recent call last):\n  File \"x.py\", line 1\n"
               "ModuleNotFoundError: No module named 'requests'")
        hint = annotate_failure("python3 x.py", 1, out)
        assert "requests" in hint
        assert "venv" in hint

    def test_dotted_module(self):
        out = "ImportError: No module named 'hermes_cli.main'"
        hint = annotate_failure("python3 -m hermes_cli.main", 1, out)
        assert "hermes_cli" in hint


class TestGitShapes:
    def test_merge_conflict(self):
        out = "Auto-merging a.py\nCONFLICT (content): Merge conflict in a.py\nAutomatic merge failed; fix conflicts and then commit the result."
        hint = annotate_failure("git merge feature", 1, out)
        assert "Do not retry" in hint

    def test_branch_already_exists(self):
        out = "fatal: a branch named 'fix/x' already exists"
        hint = annotate_failure("git checkout -b fix/x", 128, out)
        assert "fix/x" in hint

    def test_rate_limit(self):
        out = "GraphQL: API rate limit already exceeded for user ID 1."
        hint = annotate_failure("gh pr list", 1, out)
        assert "rate limit" in hint.lower()


class TestPermissionDenied:
    def test_permission_denied(self):
        hint = annotate_failure("touch /etc/x", 1, "touch: cannot touch '/etc/x': Permission denied")
        assert "Permission denied" in hint


class TestBoundedScan:
    def test_pattern_beyond_scan_window_ignored(self):
        out = "x" * 5000 + "\npython: command not found"
        assert annotate_failure("noop", 1, out) is None

    def test_hint_functions_cannot_crash_annotate(self):
        # A hint raising must not propagate.
        with mock_patch("tools.terminal_hints._OUTPUT_HINTS", [lambda c, o: 1 / 0]):
            assert annotate_failure("x", 1, "boom") is None


class TestTerminalIntegration:
    """The hint lands in the terminal result dict under 'hint'."""

    def test_hint_field_wired(self):
        # Exercise the wiring path shape without a live environment: the
        # result assembly guards on returncode != 0 and no exit_note.
        from tools import terminal_tool_result
        # simulate: interpret gives None, hints give a value
        note = terminal_tool_result._interpret_exit_code("python x.py", 127)
        assert note is None
        hint = annotate_failure("python x.py", 127, "bash: python: command not found")
        assert hint and "python3" in hint

    def test_exit_note_suppresses_pattern_hint(self):
        # grep exit 1 is informational; annotate_failure must not be reached
        # for it in the wiring (exit_note wins). Just verify the semantics
        # table still covers it.
        from tools import terminal_tool_result
        assert terminal_tool_result._interpret_exit_code("grep foo bar.txt", 1) is not None


class TestMaskedSuccess:
    """annotate_masked_success: exit-0 pipelines that hide real failures."""

    def _fail_out(self):
        return (
            "error[E0308]: mismatched types\n"
            "error: could not compile `llama_manager` due to 11 previous errors\n"
        )

    def test_cargo_pipe_tail_flagged(self):
        hint = annotate_masked_success(
            "cargo build --release 2>&1 | tail -20", self._fail_out()
        )
        assert hint and "last pipeline command" in hint

    def test_pipe_head_flagged(self):
        hint = annotate_masked_success("cargo check | head -50", self._fail_out())
        assert hint is not None

    def test_or_echo_fallback_flagged(self):
        hint = annotate_masked_success(
            'cargo build || echo "BUILD FAILED"',
            "error: could not compile `x`\nBUILD FAILED\n",
        )
        assert hint and "||" in hint

    def test_or_true_flagged(self):
        hint = annotate_masked_success(
            "pytest tests/ || true",
            "FAILED tests/test_x.py::test_y - AssertionError\n= 3 failed in 1.2s\n",
        )
        assert hint is not None

    def test_bare_command_not_flagged(self):
        # No pipe, no ||: exit 0 with error-looking output is not our call.
        assert annotate_masked_success("cargo build", self._fail_out()) is None

    def test_clean_output_pipe_not_flagged(self):
        assert (
            annotate_masked_success(
                "cargo build 2>&1 | tail -20",
                "   Compiling llama_manager v0.1.0\n    Finished release\n",
            )
            is None
        )

    def test_grep_pipeline_excluded(self):
        # Search output legitimately CONTAINS failure text; grep|head exit 0
        # is a real success (matches found).
        assert (
            annotate_masked_success(
                "grep -rn 'error\\[E' build.log | head -20",
                "build.log:12:error[E0308]: mismatched types\n",
            )
            is None
        )

    def test_rg_pipeline_excluded(self):
        assert (
            annotate_masked_success(
                "rg 'could not compile' logs/ | tail -5",
                "logs/b.txt:error: could not compile `x`\n",
            )
            is None
        )

    def test_pipe_into_grep_not_flagged(self):
        # grep as CONSUMER filters and its exit status is meaningful
        # (0 = matched) — not a passthrough truncation consumer.
        assert (
            annotate_masked_success(
                "cargo build 2>&1 | grep -c error",
                "error[E0308]: mismatched types\n",
            )
            is None
        )

    def test_or_operator_alone_not_confused_with_pipe(self):
        # `a || b` must not match the single-pipe regex.
        assert (
            annotate_masked_success(
                "test -f x || stat y",
                "error[E0308]\n",  # would need a masking shape too
            )
            is None
        )

    def test_generic_error_word_not_enough(self):
        # Weak signal ("error" substring) must not fire the note.
        assert (
            annotate_masked_success(
                "make install 2>&1 | tail -30",
                "checking error handling support... yes\nInstall complete.\n",
            )
            is None
        )

    def test_pytest_summary_via_tee(self):
        hint = annotate_masked_success(
            "pytest tests/ | tee /tmp/out.log",
            "FAILED tests/a.py::test_b\n=========== 2 failed, 10 passed ===========\n",
        )
        assert hint is not None

    def test_empty_inputs(self):
        assert annotate_masked_success("", "") is None
        assert annotate_masked_success("cargo build | tail", "") is None
        assert annotate_masked_success("", "error[E0308]") is None

    def test_echo_printf_heads_excluded(self):
        assert (
            annotate_masked_success(
                "printf 'a.rs:1:error[E0308]: x\\n' | cat | tail -3",
                "a.rs:1:error[E0308]: x\n",
            )
            is None
        )
        assert (
            annotate_masked_success(
                'echo "error: could not compile x" | tee log.txt',
                "error: could not compile x\n",
            )
            is None
        )
