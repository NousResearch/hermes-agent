"""Tests for BaseEnvironment unified execution model.

Tests _wrap_command(), _extract_cwd_from_output(), _embed_stdin_heredoc(),
init_session() failure handling, and the CWD marker contract.
"""

from unittest.mock import MagicMock

from tools.environments.base import BaseEnvironment, _BoundedOutputCollector


class _TestableEnv(BaseEnvironment):
    """Concrete subclass for testing base class methods."""

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("Use mock")

    def cleanup(self):
        pass


class TestBoundedOutputCollector:
    def test_large_stream_retains_bounded_head_and_tail(self):
        collector = _BoundedOutputCollector(1_000)
        collector.append("HEAD-SENTINEL\n")
        for _ in range(2_000):
            collector.append("x" * 4_096)
        collector.append("\nTAIL-SENTINEL")

        rendered = collector.render()

        assert collector.total_chars > 8_000_000
        assert collector.buffered_chars <= 1_000
        assert len(rendered) <= 1_000
        assert rendered.startswith("HEAD-SENTINEL")
        assert rendered.endswith("TAIL-SENTINEL")
        assert "[OUTPUT TRUNCATED" in rendered


    def test_required_status_suffix_stays_inside_limit(self):
        collector = _BoundedOutputCollector(120)
        collector.append("A" * 10_000)

        rendered = collector.render(suffix="\n[Command timed out after 1s]")

        assert len(rendered) <= 120
        assert rendered.endswith("[Command timed out after 1s]")
        assert "[OUTPUT TRUNCATED" in rendered


class TestWrapCommand:
    def test_basic_shape(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hello", "/tmp")

        assert "source" in wrapped
        assert "cd -- /tmp" in wrapped or "cd -- '/tmp'" in wrapped
        assert "eval 'echo hello'" in wrapped
        assert "__hermes_ec=$?" in wrapped
        assert "export -p" in wrapped and "> " in wrapped
        # cwd travels via the stdout marker only — no temp-file write.
        assert "pwd -P >" not in wrapped
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


    def test_cd_failure_exit_126(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("ls", "/nonexistent")

        assert "exit 126" in wrapped


class TestAtomicSnapshotWrite:
    """Regression for #38249: concurrent terminal calls in one session both
    source AND rewrite the shared env snapshot. A non-atomic ``export -p >
    snap`` truncates-then-writes in place, so a concurrent ``source snap`` can
    read a half-written file and embed ``declare -x``/``export`` fragments into
    PATH, breaking ``ls``/``git``/``tr`` with command-not-found. The write must
    assemble in a temp file and ``mv -f`` it into place (mv is atomic on POSIX
    same-fs), so a reader sees the old-or-new complete file, never a torn one.
    """

    def test_wrap_command_uses_atomic_temp_then_mv(self):
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hi", "/tmp")
        # Env dump goes to a temp file, not directly over the live snapshot.
        assert "export -p" in wrapped and "> " in wrapped
        assert ".tmp." in wrapped
        # Then an atomic rename onto the real snapshot path.
        assert "mv -f " in wrapped
        # The env-dump must NOT write the live snapshot in place (the bug).
        snap = env._snapshot_path
        assert f"> {snap} " not in wrapped
        assert f"> '{snap}'" not in wrapped
        assert f"> {snap}\n" not in wrapped

    def test_temp_path_uses_mktemp_not_pid_variables(self):
        """The temp name MUST be allocated by ``mktemp`` — never ``$$`` (in
        ``&``-launched concurrent subshells it stays the parent shell's PID, so
        two writers would pick the same temp name and publish a torn file) and
        never ``$BASHPID`` (macOS ships bash 3.2, which lacks it — the name
        expands empty, collapsing every writer onto one temp path and
        reopening the #38249 race).  Regression for PR #54314."""
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hi", "/tmp")
        assert "mktemp " in wrapped
        assert ".tmp.XXXXXXXXXX" in wrapped
        assert "$BASHPID" not in wrapped
        # The bare $$ temp form must be gone.
        assert ".tmp.$$" not in wrapped


    def test_wrap_command_mv_chained_on_export_success(self):
        """A failed export must not replace a valid snapshot."""
        env = _TestableEnv()
        env._snapshot_ready = True
        wrapped = env._wrap_command("echo hi", "/tmp")
        assert "export -p" in wrapped and "> " in wrapped and "&& mv -f " in wrapped
        assert "rm -f " in wrapped


    def test_init_session_bootstrap_also_atomic_and_mktemp(self):
        """The init_session bootstrap (first snapshot write) is the same shared
        file a concurrent command could source — it must be atomic and use
        ``mktemp`` too (no ``$BASHPID``: absent on macOS bash 3.2)."""
        env = _TestableEnv()
        captured = {}

        def fake_run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
            captured.setdefault("cmd", cmd_string)  # only the bootstrap; ignore the failure-path probe
            raise RuntimeError("stop after capture")

        env._run_bash = fake_run_bash  # type: ignore[assignment]
        try:
            env.init_session()
        except Exception:
            pass
        boot = captured.get("cmd", "")
        assert ".tmp." in boot and "mv -f " in boot, boot
        assert "mktemp " in boot
        assert "$BASHPID" not in boot
        assert ".tmp.$$" not in boot


    def test_init_session_bootstrap_uses_private_umask(self):
        env = _TestableEnv()
        captured = {}

        def fake_run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
            captured.setdefault("cmd", cmd_string)  # only the bootstrap; ignore the failure-path probe
            raise RuntimeError("stop after capture")

        env._run_bash = fake_run_bash  # type: ignore[assignment]
        try:
            env.init_session()
        except Exception:
            pass
        boot = captured.get("cmd", "")
        assert "umask 077" in boot
        assert boot.index("umask 077") < boot.index("export -p")


class TestAtomicSnapshotConcurrencyBehavioral:
    """Behavioral regression for #38249 — actually EXECUTES the generated
    snapshot write/read concurrently and asserts the file never tears.

    The string-inspection tests prove the right script is emitted; this proves
    the emitted script's guarantee holds under real concurrency: N concurrent
    writers + readers, and the snapshot is ALWAYS a complete, parseable env
    dump — never truncated mid-line with a ``declare -x`` / ``export`` fragment
    that would corrupt PATH.  Crucially it allocates the temp with ``mktemp``
    (per-writer unique, works on macOS bash 3.2 which lacks ``$BASHPID``),
    which is what closes the race; ``$$`` would still tear here.
    """

    def _run(self, script):
        import subprocess
        return subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True)

    def test_concurrent_writes_never_tear_the_snapshot(self, tmp_path):
        import shutil
        if not shutil.which("bash"):
            import pytest
            pytest.skip("bash required")
        import shlex
        snap = str(tmp_path / "hermes-snap-x.sh")
        _q = shlex.quote
        _tmpl = _q(snap + ".tmp.XXXXXXXXXX")
        # One writer iteration = the exact atomic sequence _wrap_command emits.
        writer = (
            "for i in $(seq 1 80); do "
            "export BIG_$i=$(head -c 600 /dev/zero | tr '\\0' x); "
            f"__hermes_snap_tmp=$(mktemp {_tmpl}) && "
            f"{{ export -p > \"$__hermes_snap_tmp\" && mv -f \"$__hermes_snap_tmp\" {_q(snap)}; }} "
            f"2>/dev/null || rm -f \"$__hermes_snap_tmp\" 2>/dev/null || true; "
            "done"
        )
        # Reader: repeatedly source the snapshot and check PATH never absorbs
        # an `export `/`declare -x` fragment (the corruption signature).
        reader = (
            "export PATH=/usr/bin:/bin; "
            "for i in $(seq 1 160); do "
            f"( source {_q(snap)} >/dev/null 2>&1 || true; "
            "case \"$PATH\" in *'declare -x'*|*'export '*) echo CORRUPT;; esac ); "
            "done"
        )
        self._run(f"export -p > {_q(snap)}")  # seed a valid snapshot
        # 4 concurrent writers + 4 readers, repeated.
        w = " & ".join([writer] * 4)
        r = " & ".join([reader] * 4)
        procs = [self._run(f"{w} & {r} & wait") for _ in range(3)]
        corrupt = any("CORRUPT" in p.stdout for p in procs)
        assert not corrupt, "snapshot tore — PATH absorbed a declare-x/export fragment"
        final = self._run(f"source {_q(snap)} >/dev/null 2>&1 && echo OK || echo BROKEN")
        assert "OK" in final.stdout, f"final snapshot not sourceable: {final.stdout} {final.stderr}"

    def test_failed_export_does_not_destroy_good_snapshot(self, tmp_path):
        """If ``export -p`` fails, the ``&&``-chained mv must NOT clobber the
        existing good snapshot."""
        import shutil
        if not shutil.which("bash"):
            import pytest
            pytest.skip("bash required")
        import shlex
        snap = str(tmp_path / "snap.sh")
        _q = shlex.quote
        self._run(f"echo 'export GOOD=1' > {_q(snap)}")  # seed good snapshot
        # Redirect export into an unwritable dir so the export side fails; mv
        # must then NOT run (&&) and not clobber snap.
        bad_tmp = _q("/nonexistent-dir/snap.tmp.XXXXXXXXXX")
        script = (
            f"__hermes_snap_tmp=$(mktemp {bad_tmp}) && "
            f"{{ export -p > \"$__hermes_snap_tmp\" && mv -f \"$__hermes_snap_tmp\" {_q(snap)}; }} "
            f"2>/dev/null || rm -f \"$__hermes_snap_tmp\" 2>/dev/null || true"
        )
        self._run(script)
        out = self._run(f"cat {_q(snap)}")
        assert "export GOOD=1" in out.stdout, "good snapshot was destroyed by a failed export"


class TestSnapshotFileModes:
    """Snapshot metadata files are private without changing user command umask."""

    def test_snapshot_and_cwd_files_are_0600(self, tmp_path):
        import os
        from pathlib import Path
        import shutil
        import stat
        import subprocess
        if not shutil.which("bash"):
            import pytest
            pytest.skip("bash required")

        class ExecutableEnv(BaseEnvironment):
            def __init__(self, temp_dir):
                self._temp_dir = str(temp_dir)
                super().__init__(cwd=str(temp_dir), timeout=10)

            def get_temp_dir(self):
                return self._temp_dir

            def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
                proc = subprocess.Popen(
                    ["/bin/bash", "-lc", cmd_string],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    cwd=self.cwd,
                )
                proc.communicate(timeout=timeout)
                return proc

            def cleanup(self):
                pass

        old_umask = os.umask(0o022)
        try:
            env = ExecutableEnv(tmp_path)
            env.init_session()

            user_file = tmp_path / "user-created.txt"
            env.execute(f"touch {user_file}")

            assert stat.S_IMODE(user_file.stat().st_mode) == 0o644
            assert stat.S_IMODE(Path(env._snapshot_path).stat().st_mode) == 0o600
            # The cwd temp file is no longer written (cwd travels via the
            # stdout marker for every backend) — nothing to leak on disk.
            assert not Path(env._cwd_file).exists()
        finally:
            os.umask(old_umask)


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


    def test_prefer_nonlogin_when_login_bash_is_dead(self):
        """Login snapshot failure + working non-login probe → don't use bash -l."""
        env = _TestableEnv()

        def mock_run_bash(cmd, *, login=False, timeout=120, stdin_data=None):
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.stdout = iter([])
            if login:
                mock.returncode = 1
            else:
                mock.returncode = 0
            return mock

        env._run_bash = mock_run_bash
        env.init_session()

        assert env._snapshot_ready is False
        assert env._prefer_nonlogin is True

        calls = []

        def track_run_bash(cmd, *, login=False, timeout=120, stdin_data=None):
            calls.append({"login": login})
            mock = MagicMock()
            mock.poll.return_value = 0
            mock.returncode = 0
            mock.stdout = iter([])
            return mock

        env._run_bash = track_run_bash
        env.execute("echo test")

        assert calls[0]["login"] is False


class TestCwdMarker:
    def test_marker_contains_session_id(self):
        env = _TestableEnv()
        assert env._session_id in env._cwd_marker

    def test_unique_per_instance(self):
        env1 = _TestableEnv()
        env2 = _TestableEnv()
        assert env1._cwd_marker != env2._cwd_marker


class TestSanitizeTaskIdForPath:
    """sanitize_task_id_for_path must yield mountable, collision-free segments.

    A raw task id like ``session:agent:main:telegram:dm:12345`` used as a
    sandbox directory name made docker -v split the bind-mount on the embedded
    colons and the daemon rejected it with "invalid mode" / exit 125 (#92414).
    The helper is shared by every backend that builds host paths from task_id
    (docker persistent sandboxes, singularity overlays), fixing the class once.
    """

    def test_docker_unsafe_characters_are_replaced(self):
        from tools.environments.base import sanitize_task_id_for_path

        out = sanitize_task_id_for_path("session:agent:main:telegram:dm:12345")
        assert ":" not in out
        assert "/" not in out and "\\" not in out

    def test_safe_ids_pass_through_verbatim(self):
        """Existing sandboxes keep resolving to their current directory."""
        from tools.environments.base import sanitize_task_id_for_path

        for value in ("default", "task-01.abc_def", "astropy__astropy-12907"):
            assert sanitize_task_id_for_path(value) == value

    def test_deterministic_and_collision_free_for_distinct_inputs(self):
        from tools.environments.base import sanitize_task_id_for_path

        assert sanitize_task_id_for_path("a:b") == sanitize_task_id_for_path("a:b")
        # substitution alone is not injective — the digest must disambiguate
        assert sanitize_task_id_for_path("a:b") != sanitize_task_id_for_path("a_b")
        assert sanitize_task_id_for_path("!!!") != sanitize_task_id_for_path("@@@")

    def test_empty_and_traversal_inputs_are_neutralized(self):
        from tools.environments.base import sanitize_task_id_for_path

        assert sanitize_task_id_for_path("") == "default"
        for value in (".", "..", "../../etc", "..\\..\\escape"):
            out = sanitize_task_id_for_path(value)
            assert out not in {".", ".."}
            assert "/" not in out and "\\" not in out

    def test_oversized_input_truncates_with_unique_digest(self):
        from tools.environments.base import (
            _SANDBOX_DIR_MAX_LEN,
            sanitize_task_id_for_path,
        )

        long_a = "a" * 300 + ":1"
        long_b = "a" * 300 + ":2"
        out_a = sanitize_task_id_for_path(long_a)
        out_b = sanitize_task_id_for_path(long_b)
        assert len(out_a) <= _SANDBOX_DIR_MAX_LEN
        assert ":" not in out_a
        assert out_a != out_b

    def test_sanitized_dir_is_creatable(self, tmp_path):
        from tools.environments.base import sanitize_task_id_for_path

        target = tmp_path / "docker" / sanitize_task_id_for_path(
            "session:agent:main:telegram:dm:12345"
        )
        target.mkdir(parents=True)
        assert target.is_dir()
