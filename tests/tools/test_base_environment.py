"""Tests for BaseEnvironment unified execution model.

Tests _wrap_command(), _extract_cwd_from_output(), _embed_stdin_heredoc(),
init_session() failure handling, and the CWD marker contract.
"""

from unittest.mock import MagicMock

from tools.environments.base import BaseEnvironment


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
        assert "cd -- /tmp" in wrapped or "cd -- '/tmp'" in wrapped
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

        assert "cd -- ~" in wrapped
        assert "cd -- '~'" not in wrapped

    def test_tilde_subpath_with_spaces_uses_home_and_quotes_suffix(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~/my repo")

        assert "cd -- $HOME/'my repo'" in wrapped
        assert "cd -- ~/my repo" not in wrapped

    def test_tilde_slash_maps_to_home(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "~/")

        assert "cd -- $HOME" in wrapped
        assert "cd -- ~/" not in wrapped

    def test_hyphen_prefixed_workdir_is_passed_after_double_dash(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("pwd", "-demo")

        assert "builtin cd -- -demo || exit 126" in wrapped

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


class TestStatelessExecute:
    """Finding 1 (#38249): allowlisted concurrent terminal calls must run on a
    snapshot-free path so two calls sharing one environment cannot race the
    per-session snapshot/cwd read-modify-write.

    Why: ``BaseEnvironment.execute`` persists session state by read-modify-write
    of shared snapshot/cwd files; the parallel-prefix feature introduces
    concurrency, so allowlisted calls must opt out of that shared state.
    What: ``persist_session=False`` makes ``_wrap_command`` omit the snapshot
    source/rewrite and the cwd-file write, and makes ``execute`` skip the
    cwd read-back so ``self.cwd`` is not mutated.
    Test: assert the wrapped script and the post-execute ``self.cwd`` for both
    the persisted (default) and stateless paths.
    """

    def test_wrap_command_stateless_omits_snapshot_and_cwd_file(self):
        env = _TestableEnv()
        env._snapshot_ready = True

        persisted = env._wrap_command("echo hi", "/tmp")
        assert "source" in persisted
        assert "export -p >" in persisted
        assert env._snapshot_path in persisted
        assert env._cwd_file in persisted

        stateless = env._wrap_command("echo hi", "/tmp", persist_session=False)
        # No snapshot source, no snapshot rewrite, no cwd-file write.
        assert "source" not in stateless
        assert "export -p >" not in stateless
        assert env._snapshot_path not in stateless
        assert env._cwd_file not in stateless
        # The command still runs and still cd's into the configured cwd.
        assert "eval 'echo hi'" in stateless
        assert "cd -- /tmp" in stateless or "cd -- '/tmp'" in stateless

    def _run_execute(self, env, *, persist_session, new_cwd):
        """Drive execute() with mocked shell I/O; the fake shell 'reports' a new
        cwd via the stdout marker the wrapper would emit."""
        marker = env._cwd_marker
        output = f"done\n{marker}{new_cwd}{marker}\n"

        env._run_bash = lambda *a, **k: MagicMock()
        env._wait_for_process = lambda proc, timeout=120: {
            "output": output,
            "returncode": 0,
        }
        return env.execute("echo hi", persist_session=persist_session)

    def test_execute_persisted_updates_cwd(self):
        """Sanity: the default (persisted) path DOES propagate cwd from output."""
        env = _TestableEnv(cwd="/start")
        env._snapshot_ready = True
        self._run_execute(env, persist_session=True, new_cwd="/moved")
        assert env.cwd == "/moved"

    def test_execute_stateless_does_not_mutate_cwd(self):
        """Stateless path leaves self.cwd untouched — no shared-state write that
        a concurrent call could race."""
        env = _TestableEnv(cwd="/start")
        env._snapshot_ready = True
        result = self._run_execute(env, persist_session=False, new_cwd="/moved")
        assert env.cwd == "/start"
        # The marker is still stripped from user-visible output (parity).
        assert env._cwd_marker not in result["output"]

    def test_execute_stateless_concurrent_calls_keep_cwd_stable(self):
        """Two concurrent stateless calls reporting different cwds must not
        corrupt the shared self.cwd — it stays at its initial value."""
        import threading

        env = _TestableEnv(cwd="/start")
        env._snapshot_ready = True

        def worker(target_cwd):
            marker = env._cwd_marker
            local = _TestableEnv  # noqa: F841  # readability only
            env._run_bash = lambda *a, **k: MagicMock()
            env._wait_for_process = lambda proc, timeout=120, _c=target_cwd: {
                "output": f"x\n{marker}{_c}{marker}\n",
                "returncode": 0,
            }
            env.execute("echo hi", persist_session=False)

        threads = [threading.Thread(target=worker, args=(f"/c{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert env.cwd == "/start"


class TestCwdMarker:
    def test_marker_contains_session_id(self):
        env = _TestableEnv()
        assert env._session_id in env._cwd_marker

    def test_unique_per_instance(self):
        env1 = _TestableEnv()
        env2 = _TestableEnv()
        assert env1._cwd_marker != env2._cwd_marker
