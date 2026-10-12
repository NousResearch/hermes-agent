"""Env-poller incremental-read tests for tools/process_registry.py.

Split from test_process_registry.py to hold that file under its code-health
line cap; the moved tests keep their semantics unchanged.
"""

import shlex
import shutil
import subprocess
import time
from unittest.mock import patch

import pytest

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


@pytest.fixture(autouse=True)
def _reset_systemd_scope_cache():
    """Reset the cached ``systemd-run --user --scope`` availability flag
    before each test so a probe run on a real systemd host (where
    ``INVOCATION_ID`` is set) doesn't leak into tests that mock
    ``subprocess.Popen``. Tests that exercise the probe directly reset the
    cache themselves."""
    import tools.process_registry as _pr

    original = _pr._SYSTEMD_SCOPE_AVAILABLE
    _pr._SYSTEMD_SCOPE_AVAILABLE = False
    yield
    _pr._SYSTEMD_SCOPE_AVAILABLE = original


def _make_session(
    sid="proc_test123",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    started_at=None,
) -> ProcessSession:
    """Helper to create a ProcessSession for testing."""
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=started_at or time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
    )
    return s


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
        with patch("tools.process_registry.time.sleep", return_value=None), \
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
        nothing held back once the trailing character is complete.
        """
        full = "hé😀中a\n€bz🚀".encode()
        log = tmp_path / "bg.log"
        quoted = shlex.quote(str(log))
        for n in range(1, len(full) + 1):
            log.write_bytes(full[:n])
            out = subprocess.run(
                ["sh", "-c", ProcessRegistry._log_delta_command(quoted, 0)],
                capture_output=True, timeout=30,
                check=False,
            ).stdout
            header, _, delta = out.partition(b"\n")
            size, _offset = map(int, header.split())
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
                {"output": "8 0\nand new"},
                {"output": "1\n"},
                {"output": "0\n"},
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
                {"output": "11 0\nfirst chunk"},
                {"output": "0\n"},          # still running, poll again
                {"output": "17 11\n and more"},
                {"output": "1\n"},          # gone now
                {"output": "0\n"},
            ],
        )
        assert "O=0" in commands[0]
        # The second read starts at byte 11, so the first chunk is not sent
        # a second time.
        assert "O=11" in commands[2]
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
                {"output": "11 0\nfirst chunk"},
                {"output": "0\n"},          # still running, poll again
                {"output": "5 0\nfresh"},
                {"output": "1\n"},
                {"output": "0\n"},
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
                {"output": "20 0\n" + "x" * 20},
                {"output": "1\n"},
                {"output": "0\n"},
            ],
        )
        assert session.output_buffer == "x" * 10
