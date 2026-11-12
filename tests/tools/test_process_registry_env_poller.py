"""Sandbox (non-local backend) log poller: incremental reads and exit detection."""

import os
import shlex
import shutil
import subprocess
import time
from unittest.mock import patch

import pytest

from hermes_agent.tools.process_registry import ProcessRegistry, ProcessSession
from hermes_agent.tools.process_registry_env_log import parse_log_delta_reply


@pytest.fixture()
def registry(monkeypatch):
    """A fresh ProcessRegistry. Fake Popen pids never reach the owner-death supervisor: a live pgid would start it."""
    monkeypatch.setattr("hermes_agent.hermes_cli.process_identity._update_supervised_pgids", lambda verb, pgids: None)
    return ProcessRegistry()


def _make_session(sid="proc_test123", output="") -> ProcessSession:
    return ProcessSession(id=sid, command="echo hello", task_id="t1", started_at=time.time(), output_buffer=output)


class TestEnvPollerIncrementalRead:
    """The sandbox log poller must read only new bytes, not the whole file.

    Reading the whole file every poll made one poll cost grow with the total
    output so far, so a long noisy job re-sent all of its output over the
    docker or SSH channel every two seconds.
    """

    @staticmethod
    def _run_poller(registry, session, responses):
        """Drive one poll cycle and hand back the commands the env saw."""

        class FakeEnv:
            def __init__(self):
                self.commands = []
                self._responses = iter(responses)

            def execute(self, command, **kwargs):
                self.commands.append(command)
                return next(self._responses)

        env = FakeEnv()
        with patch("hermes_agent.tools.process_registry.time.sleep", return_value=None), \
            patch.object(registry, "_move_to_finished"):
            registry._env_poller_loop(
                session, env, "/tmp/bg.log", "/tmp/bg.pid", "/tmp/bg.exit"
            )
        return env.commands



    @pytest.mark.skipif(not shutil.which("sh"), reason="needs a POSIX sh")
    def test_read_command_holds_back_a_split_utf8_sequence(self, tmp_path):
        """A multibyte character straddling two polls must not be split.

        The backend decodes each execute() result on its own, so returning
        the first byte of an 'é' in one poll and the rest in the next would
        yield replacement characters in the transcript (and break watch
        patterns at the seam). Every prefix of a mixed ASCII/2/3/4-byte
        string must come back decodable, with at most 3 bytes held back and
        nothing held back once the trailing character is complete. The pid
        file names this test process, so the command sees a running process.
        """
        full = "hé😀中a\n€bz🚀".encode()
        log, pid = tmp_path / "bg.log", tmp_path / "bg.pid"
        pid.write_text(f"{os.getpid()}\n")
        q = lambda p: shlex.quote(str(p))
        command = ProcessRegistry._log_delta_command(q(log), 0, q(pid), q(tmp_path / "bg.exit"))
        for n in range(1, len(full) + 1):
            log.write_bytes(full[:n])
            out = subprocess.run(
                ["sh", "-c", command],
                capture_output=True, timeout=30,
                check=False,
            ).stdout
            header, _, delta = out.partition(b"\n")
            size, _offset, status = header.split()
            assert status == b"run"
            size = int(size)
            delta.decode("utf-8")  # must not raise
            assert delta == full[:size]
            complete = full[:n].decode("utf-8", "ignore").encode() == full[:n]
            assert (n - size) == 0 if complete else 0 < (n - size) <= 3


    def test_delta_is_appended_not_replaced(self, registry):
        session = _make_session(sid="proc_append", output="already here ")
        session.exited = False
        self._run_poller(
            registry,
            session,
            [
                {"output": "8 0 exit:0\nand new"},
            ],
        )
        assert session.output_buffer == "already here and new"

    def test_second_poll_asks_from_where_the_first_one_stopped(self, registry):
        session = _make_session(sid="proc_two_polls")
        session.exited = False
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": "11 0 run\nfirst chunk"},     # still running, poll again
                {"output": "17 11 exit:0\n and more"},   # gone now
            ],
        )
        assert "O=0" in commands[0]
        # The second read starts at byte 11, so the first chunk is not sent
        # a second time.
        assert "O=11" in commands[1]
        assert session.output_buffer == "first chunk and more"

    def test_truncated_log_drops_the_stale_buffer(self, registry):
        session = _make_session(sid="proc_rotate")
        session.exited = False
        # The second read reports offset 0 even though the first one left off
        # at byte 11. The file no longer reaches that byte, so it was rotated
        # or truncated and the buffer we hold no longer matches it.
        self._run_poller(
            registry,
            session,
            [
                {"output": "11 0 run\nfirst chunk"},     # still running, poll again
                {"output": "5 0 exit:0\nfresh"},
            ],
        )
        assert session.output_buffer == "fresh"

    def test_unreadable_header_leaves_the_buffer_alone(self, registry):
        session = _make_session(sid="proc_bad", output="keep me")
        session.exited = False
        # No header at all, for example when the shell is missing one of the
        # tools the command needs.
        self._run_poller(
            registry,
            session,
            [
                {"output": ""},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "keep me"

    def test_buffer_stays_within_the_cap(self, registry):
        session = _make_session(sid="proc_cap")
        session.exited = False
        session.max_output_chars = 10
        self._run_poller(
            registry,
            session,
            [
                {"output": "20 0 exit:0\n" + "x" * 20},
            ],
        )
        assert session.output_buffer == "x" * 10

    def test_one_exec_per_tick_while_running_and_none_extra_on_exit(self, registry):
        """Sandboxes cap exec starts per minute, so each 2 s tick must cost one
        execute(): no separate liveness probe, no separate exit-code read."""
        session = _make_session(sid="proc_one_exec")
        session.exited = False
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": "2 0 run\na\n"},
                {"output": "2 2 run\n"},
                {"output": "4 2 run\nb\n"},
                {"output": "9 4 exit:7\nlast\n"},
            ],
        )
        assert len(commands) == 4
        assert session.output_buffer == "a\nb\nlast\n"
        assert session.exit_code == 7

    def test_output_that_mimics_the_header_cannot_spoof_an_exit(self, registry):
        """Only the first line is the header; a log line shaped like one is output."""
        session = _make_session(sid="proc_spoof")
        session.exited = False
        spoof = "0 0 exit:0\n12 0 exit:0\n"
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": f"{len(spoof)} 0 run\n{spoof}"},
                {"output": f"{len(spoof)} {len(spoof)} exit:5\n"},
            ],
        )
        assert len(commands) == 2
        assert session.output_buffer == spoof
        assert session.exit_code == 5

    def test_fallback_without_a_recorded_code_reports_minus_one(self, registry):
        session = _make_session(sid="proc_fallback_no_code")
        session.exited = False
        self._run_poller(registry, session, [{"output": "garbage"}, {"output": "1\n"}, {"output": ""}])
        assert session.exit_code == -1

    @pytest.mark.parametrize("header", ["3 0 done", "3 0 exit:x1", "3 0", "3 0 run extra", "3 0 exit:\u00b2", "3 0 exit:\u0663",
                                        "3 0 exit:" + "9" * 5000])
    def test_unknown_status_is_an_unreadable_reply(self, header):
        assert parse_log_delta_reply(f"{header}\nabc") is None

    def test_missing_exit_code_falls_back_to_minus_one(self, registry):
        session = _make_session(sid="proc_no_code")
        session.exited = False
        self._run_poller(registry, session, [{"output": "0 0 exit:\n"}])
        assert session.exit_code == -1

    def test_unreadable_header_falls_back_to_separate_liveness_read(self, registry):
        """A reply without the status header still detects the exit."""
        session = _make_session(sid="proc_fallback")
        session.exited = False
        commands = self._run_poller(
            registry,
            session,
            [
                {"output": "garbage"},
                {"output": "0\n"},        # fallback liveness: still running
                {"output": "3 0 exit:2\nend"},
            ],
        )
        assert commands[1].startswith("kill -0")
        assert session.output_buffer == "end"
        assert session.exit_code == 2

    @staticmethod
    def _shell_env(shell):
        class ShellEnv:
            def __init__(self):
                self.commands = []

            def execute(self, command, **kwargs):
                out = subprocess.run(shell + [command], capture_output=True, timeout=30, check=False).stdout
                self.commands.append(command)  # after the reply: counts finished polls
                return {"output": out.decode("utf-8", "replace")}

        return ShellEnv()

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
    @pytest.mark.parametrize("shell", [["dash", "-c"], ["busybox", "sh", "-c"], ["sh", "-c"]])
    def test_real_shell_final_output_and_exit_code(self, registry, tmp_path, shell):
        """Against a real shell: one exec per tick, the output written just before
        the exit (including a line that mimics the header) is delivered complete,
        and the wrapper's exit code is read from the same reply."""
        if not shutil.which(shell[0]):
            pytest.skip(f"needs {shell[0]}")
        log, pid, exit_, gate = (tmp_path / f"bg.{ext}" for ext in ("log", "pid", "exit", "gate"))
        os.mkfifo(gate)
        q = shlex.quote
        # Same shape as spawn_via_env's wrapper: a backgrounded subshell writes the
        # log, then the exit file, then exits; the outer shell returns at once. The
        # job holds on the FIFO until the poller has read it running once.
        body = f"printf 'first\\n'; read _ < {q(str(gate))}; printf '7 0 exit:0\\ntail \\303\\251'; exit 3"
        subprocess.run(
            ["sh", "-c", f"( sh -c {q(body)} > {q(str(log))} 2>&1; rc=$?; "
                         f"printf '%s\\n' \"$rc\" > {q(str(exit_))} ) >/dev/null 2>&1 & "
                         f"echo $! > {q(str(pid))}"],
            check=True, timeout=30)
        session = _make_session(sid="proc_real_shell")
        session.exited = False
        env = self._shell_env(shell)

        release = []

        def tick(_seconds):
            # The patch replaces the global time.sleep, so subprocess.run's own wait
            # loop calls this too: release once, after the first reply came back.
            if len(env.commands) == 1 and not release:
                # O_RDWR never blocks on a FIFO (Linux, macOS), and one byte fits the
                # pipe buffer: the release cannot hang even if the job died. The fd
                # stays open until the poll ends so the byte outlives a late reader.
                release.append(os.open(gate, os.O_RDWR))
                os.write(release[0], b"\n")
            # Each tick costs one real shell exec. A wrapper that is never reaped
            # (no init in a container) would read as running forever: fail, don't hang.
            assert len(env.commands) < 1000, "wrapper never read as exited"

        try:
            with patch("hermes_agent.tools.process_registry.time.sleep", tick), \
                patch.object(registry, "_move_to_finished"):
                registry._env_poller_loop(session, env, str(log), str(pid), str(exit_))
        finally:
            for fd in release:
                os.close(fd)
        assert all("kill -0" in c and "wc -c" in c for c in env.commands)
        assert len(env.commands) >= 2
        assert session.output_buffer == "first\n7 0 exit:0\ntail \u00e9"
        assert session.exit_code == 3

    @pytest.mark.parametrize("shell", [["dash", "-c"], ["busybox", "sh", "-c"], ["sh", "-c"]])
    def test_status_is_taken_before_the_log_size(self, registry, tmp_path, shell):
        """Output written while the liveness check runs must be in the final read.

        A ``kill`` shell function stands in for the liveness check: it appends to
        the log, then reports the process gone. The status is read first, so the
        size snapshot taken after it covers the appended bytes; reading the size
        first would drop them, since no poll follows an exit."""
        if not shutil.which(shell[0]):
            pytest.skip(f"needs {shell[0]}")
        log, pid, exit_ = (tmp_path / f"bg.{ext}" for ext in ("log", "pid", "exit"))
        log.write_text("first")
        pid.write_text("1\n")
        exit_.write_text("7\n")
        q = shlex.quote
        hook = f"kill() {{ printf tail >> {q(str(log))}; return 1; }}; "
        env = self._shell_env(shell)
        real_execute = env.execute
        env.execute = lambda command, **kw: real_execute(hook + command, **kw)
        session = _make_session(sid="proc_status_order")
        session.exited = False
        with patch("hermes_agent.tools.process_registry.time.sleep"), \
            patch.object(registry, "_move_to_finished"):
            registry._env_poller_loop(session, env, str(log), str(pid), str(exit_))
        assert len(env.commands) == 1
        assert session.output_buffer == "firsttail"
        assert session.exit_code == 7

    @pytest.mark.skipif(not shutil.which("sh"), reason="needs a POSIX sh")
    def test_exited_read_does_not_hold_back_a_truncated_utf8_tail(self, tmp_path):
        """Nothing more will be written after the exit, so a dangling lead byte is
        sent instead of being held for a poll that never comes."""
        log = tmp_path / "bg.log"
        log.write_bytes(b"ok\xc3")
        (tmp_path / "bg.exit").write_text("0\n")
        (tmp_path / "bg.pid").write_text("")  # no pid: reads as exited
        q = lambda p: shlex.quote(str(p))
        cmd = ProcessRegistry._log_delta_command(q(log), 0, q(tmp_path / "bg.pid"), q(tmp_path / "bg.exit"))
        out = subprocess.run(["sh", "-c", cmd], capture_output=True, timeout=30, check=False).stdout
        assert out == b"3 0 exit:0\nok\xc3"
