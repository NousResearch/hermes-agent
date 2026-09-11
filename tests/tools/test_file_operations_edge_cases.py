"""Tests for edge cases in tools/file_operations.py.

Covers:
- ``_is_likely_binary()`` content-analysis branch (dead-code removal regression guard)
- ``_check_lint()`` robustness against file paths containing curly braces
"""

import pytest
from unittest.mock import MagicMock, patch

from tests.tools.file_ops_fakes import READ_SENTINEL_RE, compound_read_output
from tools.file_operations import ShellFileOperations
from tools.file_operations_search import _parse_search_context_line


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


    def test_json_inproc_clean(self, ops):
        result = ops._check_lint("/tmp/a.json", content='{"a": 1}')
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
            m = READ_SENTINEL_RE.search(command)
            if m:
                return MagicMock(
                    exit_code=0,
                    stdout=compound_read_output(
                        m.group(0), size=12, sample=b"line1\nline2\n",
                        content="line1\n", total_lines=2,
                    ),
                )
            return MagicMock(exit_code=0, stdout="")

        with patch.object(ops, "_exec", side_effect=fake_exec):
            result = ops.read_file("notes.txt", offset=0, limit=0)

        assert result.error is None
        assert "1|line1" in result.content
        # The clamped range rides the single compound probe.
        assert len(commands) == 1
        assert "sed -n '1,1p' 'notes.txt' 2>/dev/null | cut -b1-8001" in commands[0]

    def test_search_clamps_offset_and_limit_before_building_head_pipeline(self):
        env = MagicMock()
        env.cwd = "/tmp"
        ops = ShellFileOperations(env)
        commands = []

        def fake_exec(command, *args, **kwargs):
            commands.append(command)
            if command.startswith("test -e"):
                return MagicMock(exit_code=0, stdout="exists")
            if "--files" in command:
                return MagicMock(exit_code=0, stdout="a.py\n")
            return MagicMock(exit_code=0, stdout="")

        with patch.object(ops, "_has_command", side_effect=lambda cmd: cmd == "rg"), \
             patch.object(ops, "_exec", side_effect=fake_exec):
            result = ops.search("*.py", target="files", path=".", offset=-4, limit=-2)

        assert result.files == ["a.py"]
        rg_commands = [cmd for cmd in commands if "--files" in cmd]
        assert rg_commands
        assert "| head -n 2" in rg_commands[0]


# =========================================================================
# Search context parsing
# =========================================================================


class TestSearchContextParsing:
    def test_search_with_grep_uses_extended_regex(self):
        env = MagicMock()
        env.cwd = "/tmp"
        ops = ShellFileOperations(env)

        with patch.object(ops, "_exec") as mock_exec:
            mock_exec.return_value = MagicMock(
                exit_code=0,
                stdout="./first.txt:1:foo\n./second.txt:1:bar\n",
            )
            result = ops._search_with_grep(
                "foo|bar",
                path=".",
                file_glob=None,
                limit=10,
                offset=0,
                output_mode="content",
                context=0,
            )

        cmd_arg = mock_exec.call_args[0][0]
        assert cmd_arg.startswith("set -o pipefail; grep -rnHE ")
        assert result.error is None
        assert result.total_count == 2
        assert [match.content for match in result.matches] == ["foo", "bar"]

    def test_parse_search_context_line_prefers_rightmost_numeric_separator(self):
        parsed = _parse_search_context_line("dir/file-12-name.py-8-context here")

        assert parsed == ("dir/file-12-name.py", 8, "context here")


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


# =========================================================================
# read_file line counting for files without a trailing newline
# =========================================================================


class _RealShellEnv:
    """Minimal env that runs commands through bash for end-to-end read_file tests."""

    def __init__(self, cwd):
        self.cwd = cwd

    def execute(self, command, cwd=None, **kwargs):
        import subprocess
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=cwd or self.cwd,
            capture_output=True,
            text=True,
        )
        return {"output": proc.stdout, "returncode": proc.returncode}


# ``read_file`` reaches a file three different ways and each one counts lines on
# its own: the single-round-trip compound probe, the sequential fallback that
# asks one question per call, and the native reader that never starts a shell.
# Every case below runs against all three, so a count fix landing in one of them
# cannot leave the others behind.
READ_PATHS = ("compound", "sequential", "native")


def _read_by_path(ops, path, *, offset, limit, via):
    """``read_file`` through one named path. ``path`` is absolute and the
    pagination already normalized, which is what the two inner entry points
    expect (``read_file`` does both before it dispatches)."""
    if via == "compound":
        return ops.read_file(path, offset=offset, limit=limit)
    if via == "sequential":
        return ops._read_file_sequential(path, offset, limit)
    return ops._read_file_native(path, offset, limit)


@pytest.mark.parametrize("via", READ_PATHS)
class TestReadFileTrailingNewlineCount:
    """``read_file`` must count a final line that lacks a trailing newline."""

    def _ops(self, tmp_path):
        return ShellFileOperations(_RealShellEnv(str(tmp_path)))

    def test_total_lines_counts_final_unterminated_line(self, tmp_path, via):
        # 3 real lines, no trailing newline.
        path = tmp_path / "a.txt"
        path.write_text("l1\nl2\nl3")
        result = _read_by_path(self._ops(tmp_path), str(path), offset=1, limit=500, via=via)
        assert result.error is None
        assert result.total_lines == 3
        assert "3|l3" in result.content

    def test_final_line_at_page_boundary_is_not_lost(self, tmp_path, via):
        # 3 real lines, no trailing newline, page size 2: line 3 lands just
        # past the first page. Counting newlines reported 2, so truncated came
        # out False and the continuation hint was suppressed. The page had
        # already stopped at line 2, so line 3 was never read and the caller was
        # told the file was complete.
        path = tmp_path / "c.txt"
        path.write_text("x1\nx2\nx3")
        ops = self._ops(tmp_path)
        result = _read_by_path(ops, str(path), offset=1, limit=2, via=via)
        assert result.error is None
        assert result.total_lines == 3
        assert result.truncated is True
        assert result.hint and "offset=3" in result.hint
        # And the final line is reachable on the next page.
        page2 = _read_by_path(ops, str(path), offset=3, limit=2, via=via)
        assert page2.error is None
        assert "3|x3" in page2.content

    def test_trailing_newline_file_count_unchanged(self, tmp_path, via):
        # Control: a file that ends in a newline still counts correctly.
        path = tmp_path / "b.txt"
        path.write_text("l1\nl2\nl3\n")
        result = _read_by_path(self._ops(tmp_path), str(path), offset=1, limit=500, via=via)
        assert result.error is None
        assert result.total_lines == 3


class TestNativeReadTrailingLineAcrossChunks:
    """The native reader stops per-line work and bulk-counts newlines once the
    page is behind it; the trailing unterminated line must survive that switch."""

    def test_unterminated_tail_counted_past_the_page(self, tmp_path):
        line = b"x" * 63 + b"\n"   # 64 bytes, so exactly 16384 lines per 1 MiB chunk
        lines = 16400              # enough to push the tail into a second chunk
        path = tmp_path / "big.txt"
        path.write_bytes(line * lines + b"tail")
        assert path.stat().st_size > (1 << 20)
        ops = ShellFileOperations(_RealShellEnv(str(tmp_path)))

        result = ops._read_file_native(str(path), 1, 2)
        assert result.error is None
        assert result.total_lines == lines + 1
        assert result.truncated is True

        last = ops._read_file_native(str(path), lines + 1, 2)
        assert last.error is None
        assert f"{lines + 1}|tail" in last.content
