"""Tests for edge cases in tools/file_operations.py.

Covers:
- ``_is_likely_binary()`` content-analysis branch (dead-code removal regression guard)
- ``_check_lint()`` robustness against file paths containing curly braces
"""

import pytest
from unittest.mock import MagicMock, patch

from tools.file_operations import ShellFileOperations, _parse_search_context_line


# =========================================================================
# _is_likely_binary edge cases
# =========================================================================


class TestIsLikelyBinary:
    """Verify content-analysis logic after dead-code removal."""

    @pytest.fixture()
    def ops(self):
        return ShellFileOperations.__new__(ShellFileOperations)

    def test_binary_extension_returns_true(self, ops):
        """Known binary extensions should short-circuit without content analysis."""
        assert ops._is_likely_binary("image.png") is True
        assert ops._is_likely_binary("archive.tar.gz", content_sample="hello") is True

    def test_text_content_returns_false(self, ops):
        """Normal printable text should not be classified as binary."""
        sample = "Hello, world!\nThis is a normal text file.\n"
        assert ops._is_likely_binary("unknown.xyz", content_sample=sample) is False

    def test_binary_content_returns_true(self, ops):
        """Content with >30% non-printable characters should be classified as binary."""
        # 500 NUL bytes + 500 printable = 50% non-printable → binary
        # Use .xyz extension (not in BINARY_EXTENSIONS) to ensure content analysis runs
        sample = "\x00" * 500 + "a" * 500
        assert ops._is_likely_binary("data.xyz", content_sample=sample) is True

    def test_no_content_sample_returns_false(self, ops):
        """When no content sample is provided and extension is unknown → not binary."""
        assert ops._is_likely_binary("mystery_file") is False

    def test_none_content_sample_returns_false(self, ops):
        """Explicit ``None`` content_sample should behave the same as missing."""
        assert ops._is_likely_binary("mystery_file", content_sample=None) is False

    def test_empty_string_content_sample_returns_false(self, ops):
        """Empty string is falsy, so content analysis should be skipped → not binary."""
        assert ops._is_likely_binary("mystery_file", content_sample="") is False

    def test_threshold_boundary(self, ops):
        """Exactly 30% non-printable should NOT trigger binary classification (> 0.30, not >=)."""
        # 300 NUL bytes + 700 printable = 30.0% → should be False (uses strict >)
        sample = "\x00" * 300 + "a" * 700
        assert ops._is_likely_binary("data.xyz", content_sample=sample) is False

    def test_just_above_threshold(self, ops):
        """301/1000 = 30.1% non-printable → should be binary."""
        sample = "\x00" * 301 + "a" * 699
        assert ops._is_likely_binary("data.xyz", content_sample=sample) is True

    def test_tabs_and_newlines_excluded(self, ops):
        """Tabs, carriage returns, and newlines should not count as non-printable."""
        sample = "\t" * 400 + "\n" * 300 + "\r" * 200 + "a" * 100
        assert ops._is_likely_binary("file.txt", content_sample=sample) is False

    def test_content_sample_longer_than_1000(self, ops):
        """Only the first 1000 characters should be analysed."""
        # First 1000 chars: 200 NUL + 800 printable = 20% → not binary
        # Remaining 1000 chars: all NUL → ignored by [:1000] slice
        sample = "\x00" * 200 + "a" * 800 + "\x00" * 1000
        assert ops._is_likely_binary("file.xyz", content_sample=sample) is False


# =========================================================================
# _check_lint edge cases
# =========================================================================


class TestCheckLintBracePaths:
    """Verify _check_lint handles file paths with curly braces safely.

    Uses ``.js`` to exercise the shell-linter path since ``.py`` now goes
    through the in-process ast.parse linter (see TestCheckLintInproc).
    """

    @pytest.fixture()
    def ops(self):
        obj = ShellFileOperations.__new__(ShellFileOperations)
        obj._command_cache = {}
        return obj

    def test_normal_path(self, ops):
        """Normal path without braces should work as before."""
        with patch.object(ops, "_has_command", return_value=True), \
             patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(exit_code=0, stdout="")
            result = ops._check_lint("/tmp/test_file.js")

        assert result.success is True
        # Verify the command was built correctly
        cmd_arg = mock_exec.call_args[0][0]
        assert "'/tmp/test_file.js'" in cmd_arg

    def test_path_with_curly_braces(self, ops):
        """Path containing ``{`` and ``}`` must not raise KeyError/ValueError."""
        with patch.object(ops, "_has_command", return_value=True), \
             patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(exit_code=0, stdout="")
            # This would raise KeyError with .format() but works with .replace()
            result = ops._check_lint("/tmp/{test}_file.js")

        assert result.success is True
        cmd_arg = mock_exec.call_args[0][0]
        assert "{test}" in cmd_arg

    def test_path_with_nested_braces(self, ops):
        """Path with complex brace patterns like ``{{var}}`` should be safe."""
        with patch.object(ops, "_has_command", return_value=True), \
             patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(exit_code=0, stdout="")
            result = ops._check_lint("/tmp/{{var}}.js")

        assert result.success is True

    def test_unsupported_extension_skipped(self, ops):
        """Extensions without a linter should return a skipped result."""
        result = ops._check_lint("/tmp/file.unknown_ext")
        assert result.skipped is True

    def test_missing_linter_skipped(self, ops):
        """When the linter binary is not installed, skip gracefully."""
        with patch.object(ops, "_has_command", return_value=False):
            result = ops._check_lint("/tmp/test.js")
        assert result.skipped is True

    def test_lint_failure_returns_output(self, ops):
        """When the linter exits non-zero, result should capture output."""
        with patch.object(ops, "_has_command", return_value=True), \
             patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(
                exit_code=1,
                stdout="SyntaxError: invalid syntax",
            )
            result = ops._check_lint("/tmp/bad.js")

        assert result.success is False
        assert "SyntaxError" in result.output


class TestCheckLintInproc:
    """Verify in-process linters (.py via ast.parse, .json, .yaml, .toml).

    These bypass the shell linter table entirely and parse content
    directly in Python — no subprocess, no toolchain dependency.
    """

    @pytest.fixture()
    def ops(self):
        obj = ShellFileOperations.__new__(ShellFileOperations)
        obj._command_cache = {}
        return obj

    def test_python_inproc_clean(self, ops):
        """Valid Python content passes in-process ast.parse."""
        result = ops._check_lint("/tmp/ok.py", content="x = 1\n")
        assert result.success is True
        assert not result.skipped
        assert result.output == ""

    def test_python_inproc_syntax_error(self, ops):
        """Invalid Python content fails with SyntaxError + line info."""
        result = ops._check_lint("/tmp/bad.py", content="def foo(:\n    pass\n")
        assert result.success is False
        assert "SyntaxError" in result.output
        assert "line" in result.output.lower()

    def test_python_inproc_content_explicit(self, ops):
        """When content is passed explicitly, the file is not re-read."""
        with patch.object(ops, "_exec") as mock_exec:
            result = ops._check_lint("/tmp/explicit.py", content="y = 2\n")
            # _exec must not have been called — content was supplied
            mock_exec.assert_not_called()
        assert result.success is True

    def test_json_inproc_clean(self, ops):
        result = ops._check_lint("/tmp/a.json", content='{"a": 1}')
        assert result.success is True

    def test_json_inproc_error(self, ops):
        result = ops._check_lint("/tmp/b.json", content='{"a": 1')
        assert result.success is False
        assert "JSONDecodeError" in result.output

    def test_yaml_inproc_clean(self, ops):
        result = ops._check_lint("/tmp/a.yaml", content="a: 1\nb: 2\n")
        assert result.success is True

    def test_yaml_inproc_error(self, ops):
        result = ops._check_lint("/tmp/b.yaml", content='key: "unclosed\n')
        assert result.success is False
        assert "YAMLError" in result.output

    def test_toml_inproc_clean(self, ops):
        result = ops._check_lint("/tmp/a.toml", content='[section]\nk = "v"\n')
        assert result.success is True

    def test_toml_inproc_error(self, ops):
        result = ops._check_lint("/tmp/b.toml", content='[section\nk = "v"')
        assert result.success is False
        assert "TOMLDecodeError" in result.output


class TestCheckLintDelta:
    """Verify _check_lint_delta() filters pre-existing errors from post-edit output."""

    @pytest.fixture()
    def ops(self):
        obj = ShellFileOperations.__new__(ShellFileOperations)
        obj._command_cache = {}
        return obj

    def test_clean_post_no_pre_lint(self, ops):
        """Hot path: post-write is clean, pre-lint should be skipped entirely."""
        with patch.object(ops, "_check_lint", wraps=ops._check_lint) as wrapped:
            r = ops._check_lint_delta("/tmp/a.py", pre_content="x = 0\n", post_content="x = 1\n")
            # Post-lint called exactly once (clean), pre-lint never called.
            assert wrapped.call_count == 1
        assert r.success is True

    def test_new_file_reports_all_errors(self, ops):
        """No pre-content means no delta refinement — all post errors surface."""
        r = ops._check_lint_delta("/tmp/new.py", pre_content=None, post_content="def x(:\n")
        assert r.success is False
        assert "SyntaxError" in r.output

    def test_broken_file_becomes_good(self, ops):
        """Post-clean short-circuits without any delta refinement."""
        r = ops._check_lint_delta("/tmp/fix.py", pre_content="def x(:\n", post_content="def x():\n    pass\n")
        assert r.success is True

    def test_introduces_new_error_filters_pre(self, ops):
        """Delta filter drops pre-existing errors, surfaces only new ones."""
        pre = 'def a(:\n    pass\n'  # line 1 broken
        post = 'def a():\n    pass\n\ndef b(:\n    pass\n'  # line 1 fixed, line 4 broken
        r = ops._check_lint_delta("/tmp/d.py", pre_content=pre, post_content=post)
        assert r.success is False
        assert "New lint errors" in r.output or "line 4" in r.output

    def test_pre_existing_remains_flagged_but_not_new(self, ops):
        """Single-error parsers (ast) may miss that post is OK — be cautious."""
        # Pre has line-1 error, post keeps it (and doesn't add anything new)
        pre = 'def a(:\n    pass\n'
        post = 'def a(:\n    pass\n\nprint(42)\n'  # still line 1 broken
        r = ops._check_lint_delta("/tmp/d.py", pre_content=pre, post_content=post)
        # File is still broken — don't lie and claim success — but flag it as pre-existing
        assert r.success is False
        assert "pre-existing" in (r.message or "").lower()


# =========================================================================
# Pagination bounds
# =========================================================================


class TestPaginationBounds:
    """Invalid pagination inputs should not leak into shell commands."""

    def test_read_file_clamps_offset_and_limit_before_building_sed_range(self):
        env = MagicMock()
        env.cwd = "/tmp"
        ops = ShellFileOperations(env)
        commands = []

        def fake_exec(command, *args, **kwargs):
            commands.append(command)
            if command.startswith("wc -c"):
                return MagicMock(exit_code=0, stdout="12")
            if command.startswith("head -c"):
                return MagicMock(exit_code=0, stdout="line1\nline2\n")
            if command.startswith("sed -n"):
                return MagicMock(exit_code=0, stdout="line1\n")
            if command.startswith("wc -l"):
                return MagicMock(exit_code=0, stdout="2")
            return MagicMock(exit_code=0, stdout="")

        with patch.object(ops, "_exec", side_effect=fake_exec):
            result = ops.read_file("notes.txt", offset=0, limit=0)

        assert result.error is None
        assert "     1|line1" in result.content
        sed_commands = [cmd for cmd in commands if cmd.startswith("sed -n")]
        assert sed_commands == ["sed -n '1,1p' 'notes.txt'"]

    def test_search_clamps_offset_and_limit_before_building_head_pipeline(self):
        env = MagicMock()
        env.cwd = "/tmp"
        ops = ShellFileOperations(env)
        commands = []

        def fake_exec(command, *args, **kwargs):
            commands.append(command)
            if command.startswith("test -e"):
                return MagicMock(exit_code=0, stdout="exists")
            if command.startswith("rg --files"):
                return MagicMock(exit_code=0, stdout="a.py\n")
            return MagicMock(exit_code=0, stdout="")

        with patch.object(ops, "_has_command", side_effect=lambda cmd: cmd == "rg"), \
             patch.object(ops, "_exec", side_effect=fake_exec):
            result = ops.search("*.py", target="files", path=".", offset=-4, limit=-2)

        assert result.files == ["a.py"]
        rg_commands = [cmd for cmd in commands if cmd.startswith("rg --files")]
        assert rg_commands
        assert "| head -n 1" in rg_commands[0]


# =========================================================================
# Search context parsing
# =========================================================================


class TestSearchContextParsing:
    def test_parse_search_context_line_prefers_rightmost_numeric_separator(self):
        parsed = _parse_search_context_line("dir/file-12-name.py-8-context here")

        assert parsed == ("dir/file-12-name.py", 8, "context here")

    def test_search_with_rg_context_handles_filename_with_dash_digits(self):
        env = MagicMock()
        env.cwd = "/tmp"
        ops = ShellFileOperations(env)

        with patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(
                exit_code=0,
                stdout="dir/file-12-name.py-8-context here\n",
            )
            result = ops._search_with_rg(
                "needle",
                path=".",
                file_glob=None,
                limit=10,
                offset=0,
                output_mode="content",
                context=1,
            )

        assert result.error is None
        assert result.total_count == 1
        assert result.matches[0].path == "dir/file-12-name.py"
        assert result.matches[0].line_number == 8
        assert result.matches[0].content == "context here"

    def test_search_with_grep_context_handles_filename_with_dash_digits(self):
        env = MagicMock()
        env.cwd = "/tmp"
        ops = ShellFileOperations(env)

        with patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(
                exit_code=0,
                stdout="dir/file-12-name.py-8-context here\n",
            )
            result = ops._search_with_grep(
                "needle",
                path=".",
                file_glob=None,
                limit=10,
                offset=0,
                output_mode="content",
                context=1,
            )

        assert result.error is None
        assert result.total_count == 1
        assert result.matches[0].path == "dir/file-12-name.py"
        assert result.matches[0].line_number == 8
        assert result.matches[0].content == "context here"
