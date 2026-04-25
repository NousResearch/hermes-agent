"""Tests for BaseEnvironment unified execution model.

Tests _wrap_command(), _extract_cwd_from_output(), _embed_stdin_heredoc(),
init_session() failure handling, and the CWD marker contract.
"""

import uuid
from unittest.mock import MagicMock

from tools.environments.base import BaseEnvironment, _cwd_marker


class _TestableEnv(BaseEnvironment):
    """Concrete subclass for testing base class methods."""

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("Use mock")

    def cleanup(self):
        pass


class TestWrapCommand:
    def test_basic_shape(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hello", "/tmp")

        assert "source" in wrapped
        assert "cd /tmp" in wrapped or "cd '/tmp'" in wrapped
        assert "eval 'echo hello'" in wrapped
        assert "__hermes_ec=$?" in wrapped
        assert "export -p >" in wrapped
        assert "pwd -P >" in wrapped
        assert env._cwd_marker in wrapped
        assert "exit $__hermes_ec" in wrapped

    def test_no_snapshot_skips_source(self):
        env = _TestableEnv()
        env._snapshot_ready = False
        wrapped = env._wrap_command("echo hello", "/tmp")

        assert "source" not in wrapped

    def test_single_quote_escaping(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo 'hello world'", "/tmp")

        assert "eval 'echo '\\''hello world'\\'''" in wrapped

    def test_tilde_not_quoted(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~")

        assert "cd ~" in wrapped
        assert "cd '~'" not in wrapped

    def test_tilde_subpath_with_spaces_uses_home_and_quotes_suffix(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~/my repo")

        assert "cd $HOME/'my repo'" in wrapped
        assert "cd ~/my repo" not in wrapped

    def test_tilde_slash_maps_to_home(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~/")

        assert "cd $HOME" in wrapped
        assert "cd ~/" not in wrapped

    def test_cd_failure_exit_126(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "/nonexistent")

        assert "exit 126" in wrapped


class TestExtractCwdFromOutput:
    def test_happy_path(self):
        env = _TestableEnv()
        marker = env._cwd_marker
        result = {
            "output": f"hello\n{marker}/home/user{marker}\n",
        }
        env._extract_cwd_from_output(result)

        assert env.cwd == "/home/user"
        assert marker not in result["output"]

    def test_missing_marker(self):
        env = _TestableEnv()
        result = {"output": "hello world\n"}
        env._extract_cwd_from_output(result)

        assert env.cwd == "/tmp"  # unchanged

    def test_marker_in_command_output(self):
        """If the marker appears in command output AND as the real marker,
        rfind grabs the last (real) one."""
        env = _TestableEnv()
        marker = env._cwd_marker
        result = {
            "output": f"user typed {marker} in their output\nreal output\n{marker}/correct/path{marker}\n",
        }
        env._extract_cwd_from_output(result)

        assert env.cwd == "/correct/path"

    def test_output_cleaned(self):
        env = _TestableEnv()
        marker = env._cwd_marker
        result = {
            "output": f"hello\n{marker}/tmp{marker}\n",
        }
        env._extract_cwd_from_output(result)

        assert "hello" in result["output"]
        assert marker not in result["output"]


class TestEmbedStdinHeredoc:
    def test_heredoc_format(self):
        result = BaseEnvironment._embed_stdin_heredoc("cat", "hello world")

        assert result.startswith("cat << '")
        assert "hello world" in result
        assert "HERMES_STDIN_" in result

    def test_unique_delimiter_each_call(self):
        r1 = BaseEnvironment._embed_stdin_heredoc("cat", "data")
        r2 = BaseEnvironment._embed_stdin_heredoc("cat", "data")

        # Extract delimiters
        d1 = r1.split("'")[1]
        d2 = r2.split("'")[1]
        assert d1 != d2  # UUID-based, should be unique


class TestInitSessionFailure:
    def test_snapshot_ready_false_on_failure(self):
        env = _TestableEnv()

        def failing_run_bash(*args, **kwargs):
            raise RuntimeError("bash not found")

        env._run_bash = failing_run_bash
        env.init_session()

        assert env._snapshot_ready is False

    def test_login_flag_when_snapshot_not_ready(self):
        """When _snapshot_ready=False, execute() should pass login=True to _run_bash."""
        env = _TestableEnv()
        env._snapshot_ready = False

        calls = []
        def mock_run_bash(cmd, *, login=False, timeout=120, stdin_data=None):
            calls.append({"login": login})
            # Return a mock process handle
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.returncode = 0
            mock.stdout = iter([])
            return mock

        env._run_bash = mock_run_bash
        env.execute("echo test")

        assert len(calls) == 1
        assert calls[0]["login"] is True


class TestStripStateSyncLeak:
    """Test that _strip_state_sync_leak removes leaked declare -x / export
    lines from command output (defense-in-depth for bug #15459).
    
    When the persistent-shell snapshot redirect ('export -p > file') fails
    (permissions, deleted temp dir, race condition, etc.), the full env
    dump leaks into stdout.  The method should strip these lines while
    preserving legitimate command output.
    """

    def test_strips_declare_x_lines(self):
        env = _TestableEnv()
        result = {
            "output": (
                'declare -x PATH="/usr/bin:/bin"\n'
                'declare -x HOME="/Users/test"\n'
                'declare -x SHELL="/bin/zsh"\n'
                "Hello World\n"
            ),
        }
        env._strip_state_sync_leak(result)
        assert result["output"] == "Hello World\n"

    def test_strips_export_lines(self):
        """export VAR=value format (zsh, POSIX sh) should also be stripped."""
        env = _TestableEnv()
        result = {
            "output": (
                'export PATH="/usr/bin:/bin"\n'
                'export HOME="/Users/test"\n'
                "actual output\n"
            ),
        }
        env._strip_state_sync_leak(result)
        assert result["output"] == "actual output\n"

    def test_preserves_normal_output(self):
        """Lines that don't match declare -x or export patterns are kept."""
        env = _TestableEnv()
        result = {
            "output": "Hello World\nexport DATA=good\nsecond line\n",
        }
        env._strip_state_sync_leak(result)
        # "export DATA=good" looks like an env export line — stripped.
        # In practice, legitimate output rarely starts with "declare -x " or
        # "export " at column zero, and defense-in-depth means we err on
        # the side of stripping.
        assert "Hello World" in result["output"]
        assert "second line" in result["output"]

    def test_mixed_leak_and_real_output(self):
        """Leaked lines interspersed with real output."""
        env = _TestableEnv()
        result = {
            "output": (
                'declare -x BROWSER_INACTIVITY_TIMEOUT="120"\n'
                "real line 1\n"
                'declare -x COLORTERM="truecolor"\n'
                "real line 2\n"
            ),
        }
        env._strip_state_sync_leak(result)
        assert result["output"] == "real line 1\nreal line 2\n"

    def test_mixed_export_formats(self):
        """Mix of declare -x and export lines."""
        env = _TestableEnv()
        result = {
            "output": (
                'declare -x PATH="/usr/bin"\n'
                'export HOME="/Users/test"\n'
                "actual output\n"
            ),
        }
        env._strip_state_sync_leak(result)
        assert result["output"] == "actual output\n"

    def test_no_leak_passes_through(self):
        """Output with no leaked lines passes through unchanged."""
        env = _TestableEnv()
        original = "Hello World\nsecond line\n"
        result = {"output": original}
        env._strip_state_sync_leak(result)
        assert result["output"] == original

    def test_empty_output(self):
        env = _TestableEnv()
        result = {"output": ""}
        env._strip_state_sync_leak(result)
        assert result["output"] == ""

    def test_missing_output_key(self):
        env = _TestableEnv()
        result = {}
        env._strip_state_sync_leak(result)
        # Should not crash, should default to empty
        assert result.get("output", "") == ""

    def test_declare_x_with_multiline_value(self):
        """declare -x values can span multiple lines with escaped newlines."""
        env = _TestableEnv()
        result = {
            "output": (
                "declare -x MULTILINE=\"line1\\nline2\"\n"
                "real output here\n"
            ),
        }
        env._strip_state_sync_leak(result)
        # The declare line is stripped; real output preserved.
        # Note: declare -x multiline values use \\n within the same line,
        # so each declare line is still a single physical line.
        assert result["output"] == "real output here\n"

    def test_declare_x_without_value(self):
        """Some declare -x lines have no = (e.g., declare -x OLDPWD)."""
        env = _TestableEnv()
        result = {
            "output": (
                "declare -x OLDPWD\n"
                "declare -x PATH=\"/usr/bin\"\n"
                "real output\n"
            ),
        }
        env._strip_state_sync_leak(result)
        assert result["output"] == "real output\n"

    def test_declare_fn_stripped(self):
        """declare -f (function definitions) can also leak."""
        env = _TestableEnv()
        result = {
            "output": (
                "my_func ()\n"
                "{\n"
                "    echo hello\n"
                "}\n"
                "declare -f _my_helper\n"
                "real output\n"
            ),
        }
        # Function definitions are trickier — they span multiple lines.
        # The simple line-based stripper handles only single-line patterns.
        # Multi-line function definitions would need more advanced parsing.
        # For now, just verify single-line declare -f is stripped.
        env._strip_state_sync_leak(result)
        assert "real output" in result["output"]
        # declare -f lines are single-line and should be stripped
        assert "declare -f" not in result["output"]

    def test_alias_p_stripped(self):
        """alias -p output (alias lines) can also leak."""
        env = _TestableEnv()
        result = {
            "output": (
                "alias ll='ls -la'\n"
                "alias gs='git status'\n"
                "real command output\n"
            ),
        }
        env._strip_state_sync_leak(result)
        # Aliases are stripped
        assert result["output"] == "real command output\n"

    def test_preserves_echo_with_declare_in_it(self):
        """A real command output like 'declare -x' in a string is preserved
        if it doesn't start at column zero."""
        env = _TestableEnv()
        result = {
            "output": "The variable is declare -x formatted\n",
        }
        env._strip_state_sync_leak(result)
        assert result["output"] == "The variable is declare -x formatted\n"


class TestCwdMarker:
    def test_marker_contains_session_id(self):
        env = _TestableEnv()
        assert env._session_id in env._cwd_marker

    def test_unique_per_instance(self):
        env1 = _TestableEnv()
        env2 = _TestableEnv()
        assert env1._cwd_marker != env2._cwd_marker
