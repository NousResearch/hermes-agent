#!/usr/bin/env python3
"""Unit tests for tools/tty_detector.py"""

import pytest
from tools.tty_detector import (
    contains_ansi_escape_codes,
    is_interactive_command,
    should_skip_compression,
    ANSI_ESCAPE_PATTERNS,
    INTERACTIVE_SIZE_THRESHOLD,
)


# -----------------------------------------------------------------------
# ANSI Escape Detection
# -----------------------------------------------------------------------

class TestAnsiEscapeDetection:
    def test_sgr_color_sequence(self):
        """SGR sequences (color/bold) are common in interactive output."""
        assert contains_ansi_escape_codes("\x1b[31mred\x1b[0m") is True

    def test_cursor_position_sequence(self):
        assert contains_ansi_escape_codes("\x1b[10;20H\x1b[25;80H") is True

    def test_dec_private_mode(self):
        """tmux and screen use DEC private mode sequences."""
        assert contains_ansi_escape_codes("\x1b[?1h\x1b[?12h\x1b[?12l") is True

    def test_tmux_window_title(self):
        """OSC window title sequences from tmux."""
        assert contains_ansi_escape_codes("\x1b]0;my window\x07") is True

    def test_plain_text_no_ansi(self):
        """Normal text output should not trigger ANSI detection."""
        assert contains_ansi_escape_codes("Hello, world!\nTest passed.\n") is False

    def test_git_diff_no_ansi(self):
        assert contains_ansi_escape_codes(
            "diff --git a/file.txt b/file.txt\n"
            "-old line\n+new line\n"
        ) is False

    def test_pytest_output_no_ansi(self):
        assert contains_ansi_escape_codes(
            "tests/test_foo.py::test_one PASSED\n"
            "======= 1 passed in 0.05s =======\n"
        ) is False

    def test_empty_string(self):
        assert contains_ansi_escape_codes("") is False

    def test_only_bell_character(self):
        """BEL (\\x07) alone is not an ANSI escape sequence."""
        assert contains_ansi_escape_codes("\x07") is False

    def test_vim_escape_sequence(self):
        """Vim emits ESC followed by letter commands."""
        assert contains_ansi_escape_codes("\x1b[G\x1b[3D") is True


# -----------------------------------------------------------------------
# Command Blocklist
# -----------------------------------------------------------------------

class TestInteractiveCommandBlocklist:
    # Editors
    def test_vim(self):
        assert is_interactive_command("vim") is True
        assert is_interactive_command("vim file.txt") is True
        assert is_interactive_command("/usr/bin/vim") is True

    def test_nvim(self):
        assert is_interactive_command("nvim") is True
        assert is_interactive_command("sudo nvim README.md") is True

    def test_nano(self):
        assert is_interactive_command("nano") is True
        assert is_interactive_command("nano config.ini") is True

    def test_emacs_no_flag(self):
        """emacs in terminal mode requires -nw flag."""
        assert is_interactive_command("emacs") is False

    def test_emacs_nw(self):
        assert is_interactive_command("emacs -nw") is True
        assert is_interactive_command("emacs -nw file.txt") is True

    # Pagers
    def test_less(self):
        assert is_interactive_command("less") is True
        assert is_interactive_command("less /var/log/syslog") is True

    def test_more(self):
        assert is_interactive_command("more") is True

    # Remote access
    def test_ssh(self):
        assert is_interactive_command("ssh user@host") is True
        assert is_interactive_command("ssh -i key.pub user@host") is True

    def test_mosh(self):
        assert is_interactive_command("mosh user@host") is True

    # System monitors
    def test_top(self):
        assert is_interactive_command("top") is True

    def test_htop(self):
        assert is_interactive_command("htop") is True

    def test_tmux(self):
        assert is_interactive_command("tmux new-session") is True

    def test_screen(self):
        assert is_interactive_command("screen -S mysession") is True

    # Git TUI
    def test_tig(self):
        assert is_interactive_command("tig") is True

    # SQL clients
    def test_psql(self):
        assert is_interactive_command("psql") is True
        assert is_interactive_command("psql -d mydb") is True

    def test_mysql(self):
        assert is_interactive_command("mysql -u root") is True

    def test_sqlite3(self):
        assert is_interactive_command("sqlite3") is True
        assert is_interactive_command("sqlite3 mydb.db") is True

    # REPLs
    def test_node(self):
        """node REPL is interactive by default."""
        assert is_interactive_command("node") is True

    def test_python_no_i(self):
        """python without -i flag is not necessarily interactive."""
        assert is_interactive_command("python") is False
        assert is_interactive_command("python script.py") is False

    def test_python_i_flag(self):
        assert is_interactive_command("python -i") is True
        assert is_interactive_command("python3 -i script.py") is True

    def test_pry(self):
        assert is_interactive_command("pry") is True

    # Docker interactive
    def test_docker_run_it(self):
        assert is_interactive_command("docker run -it ubuntu bash") is True
        assert is_interactive_command("docker run --rm -it python:3 bash") is True

    def test_docker_run_detached(self):
        """docker run without -it is not interactive."""
        assert is_interactive_command("docker run ubuntu") is False
        assert is_interactive_command("docker run -d nginx") is False

    def test_kubectl_exec_it(self):
        assert is_interactive_command("kubectl exec -it pod-name -- bash") is True

    # Non-interactive commands should return False
    def test_git(self):
        assert is_interactive_command("git status") is False
        assert is_interactive_command("git diff") is False

    def test_pytest(self):
        assert is_interactive_command("pytest") is False
        assert is_interactive_command("pytest tests/") is False

    def test_cargo(self):
        assert is_interactive_command("cargo test") is False

    def test_ls(self):
        assert is_interactive_command("ls -la") is False

    def test_cat(self):
        assert is_interactive_command("cat file.txt") is False

    def test_find(self):
        assert is_interactive_command("find . -name '*.py'") is False

    def test_grep(self):
        assert is_interactive_command("grep -r 'pattern' .") is False

    def test_curl(self):
        assert is_interactive_command("curl https://example.com") is False

    def test_aws_cli(self):
        assert is_interactive_command("aws ec2 describe-instances") is False

    def test_normalized_cmd_strips_flags(self):
        """Commands with flags that aren't interactive should not match."""
        assert is_interactive_command("docker ps") is False
        assert is_interactive_command("docker logs abc123") is False
        assert is_interactive_command("docker build -t myimage .") is False


# -----------------------------------------------------------------------
# Combined Adaptive Detection
# -----------------------------------------------------------------------

class TestShouldSkipCompression:
    def test_ansi_output_skips(self):
        result, reason = should_skip_compression(
            "vim file.txt",
            "\x1b[31merror\x1b[0m\n",
            "",
        )
        assert result is True
        assert reason == "ansi"

    def test_blocklisted_command_skips(self):
        result, reason = should_skip_compression(
            "ssh user@host",
            "password: ",
            "",
        )
        assert result is True
        assert reason == "blocklist"

    def test_small_non_blocklisted_command_skips(self):
        """Small output is treated as potentially interactive — skip compression."""
        result, reason = should_skip_compression(
            "ls -la",
            "total 8\ndrwxr-xr-x  2 pi pi 4096 Apr 19 10:00 .\n",
            "",
        )
        assert result is True
        assert reason == "size"

    def test_large_output_from_unknown_command_compresses(self):
        """Large output (even from unknown commands) should not be skipped."""
        large_output = "x" * 10000
        result, reason = should_skip_compression(
            "my_custom_command",
            large_output,
            "",
        )
        assert result is False

    def test_large_batch_output_compresses(self):
        """Large pytest/git output should NOT be skipped."""
        large_output = "x" * 10000
        result, reason = should_skip_compression(
            "pytest",
            large_output,
            "",
        )
        assert result is False
        assert reason == "none"

    def test_git_diff_small_skips_size_threshold(self):
        """Small git diff output is skipped (below INTERACTIVE_SIZE_THRESHOLD)."""
        git_diff = (
            "diff --git a/file.txt b/file.txt\n"
            "-old line\n+new line\n"
            "This is a realistic git diff output.\n" * 50  # ~4500 chars < 5KB
        )
        result, reason = should_skip_compression("git diff", git_diff, "")
        assert result is True
        assert reason == "size"

    def test_pytest_compresses(self):
        """Large pytest output (above 5KB) does not skip."""
        result, reason = should_skip_compression(
            "pytest tests/",
            "tests/test_foo.py::test_one PASSED\n" * 150,  # ~5700 chars > 5KB
            "",
        )
        assert result is False

    def test_unknown_command_small_output(self):
        """Small unknown command output skips (below size threshold)."""
        result, reason = should_skip_compression(
            "my_custom_command",
            "short output",
            "",
        )
        assert result is True
        assert reason == "size"

    def test_empty_stdout_stderr(self):
        result, reason = should_skip_compression("echo hello", "", "")
        assert result is True
        assert reason == "size"

    def test_top_large_output(self):
        """top running with huge output (misuse) is still detected as interactive command."""
        # Even if output is large, the blocklist should catch it
        result, reason = should_skip_compression(
            "top",
            "x" * 50000,
            "",
        )
        assert result is True
        assert reason == "blocklist"
