"""RED repro for issue #121354 (background/headless stdout loss + no heartbeat).

Part 1: a background process terminated at a cap must return whatever it
printed so far (like a foreground timeout does). Sandbox-backend spawns
(``spawn_via_env``) run ``python`` block-buffered, so a kill loses everything.
Part 2: a headless operator polling a running job cannot tell "working" from
"wedged" -- poll must report output recency (a "last line received" heartbeat
for polling clients).
"""

import os
import subprocess
import sys
import time

import pytest

from tools.process_registry import ProcessRegistry


def _posix(path: str) -> str:
    return path.replace("\\", "/")


class _ShellEnv:
    """Minimal sandbox-style backend: runs each wrapper command through a
    NON-LOGIN shell (an outer ``-lic`` would add interactive history/job
    quirks on Windows), with PYTHONUNBUFFERED scrubbed (sandbox envs do not
    inherit the local spawn's unbuffered env)."""

    def __init__(self, name: str):
        import uuid

        self._tmpdir = f"/tmp/hermes_121354_{uuid.uuid4().hex[:8]}"
        self._name = name
        self.commands = []
        self._ncalls = 0

    def get_temp_dir(self):
        return self._tmpdir

    def execute(self, command, timeout=10, **kwargs):
        from tools.environments.local import _find_shell

        self.commands.append(command)
        shell = _find_shell()
        env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
        if sys.platform == "win32":
            # Force msysgit tools (bash/nohup/mkdir) ahead of system32 so the
            # wrapper's inner ``bash`` never hits the WSL stub.
            git_usr = os.path.join(
                os.path.dirname(os.path.dirname(shell)), "usr", "bin"
            )
            if os.path.isdir(git_usr):
                env["PATH"] = git_usr + os.pathsep + env.get("PATH", "")
        # Capture to a FILE, never a pipe: the wrapper backgrounds a subshell
        # that outlives the outer shell, so pipe-EOF only arrives when the
        # child itself exits (communicate() would block until then).
        import tempfile

        self._ncalls += 1
        out_path = os.path.join(
            tempfile.gettempdir(), f"hermes_121354_{self._name}_{self._ncalls}.log"
        )
        with open(out_path, "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(
                [shell, "-c", command],
                stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, env=env,
            )
        with open(out_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        return {"output": text, "returncode": proc.returncode}


def _read_log_until(registry, session_id, marker, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        log = registry.read_log(session_id)
        if marker in log.get("output", ""):
            return True
        time.sleep(0.2)
    return False


def test_kill_returns_partially_printed_stdout(tmp_path):
    """Sandbox-background python printing then sleeping: kill at the cap must
    return the printed line, not an empty transcript (#121354 part 1)."""
    registry = ProcessRegistry()
    env = _ShellEnv("part1")
    exe = _posix(sys.executable)
    marker = "MARKER_121354_PARTIAL"
    cmd = f"{exe} -c \"import time; print('{marker}'); time.sleep(30)\""
    # Pre-create the log dir: the spawn wrapper's ``mkdir -p DIR && (...) &``
    # backgrounds mkdir alongside the worker (``A && B &`` == ``(A && B) &``),
    # so the pid-file redirect can race a cold mkdir. Pre-creating removes
    # that unrelated race from this repro (product-side, mkdir usually wins).
    env.execute(f"mkdir -p {env.get_temp_dir()}")
    session = registry.spawn_via_env(env, cmd)
    try:
        assert not session.exited, f"launch failed: {session.output_buffer!r}"
        assert _read_log_until(registry, session.id, marker), (
            "buffered stdout never reached the log before the kill "
            "(block-buffered child lost its output)"
        )
        result = registry.kill_process(session.id)
        assert marker in result.get("output", ""), (
            f"kill at the cap lost buffered stdout: {result!r}"
        )
    finally:
        registry.kill_process(session.id)


def test_poll_reports_output_recency(tmp_path):
    """A ticking job vs a silent sleeper must be distinguishable via poll
    without shelling out to ps (#121354 part 2)."""
    registry = ProcessRegistry()
    exe = _posix(sys.executable)
    ticker = registry.spawn_local(
        f"{exe} -u -c \"import time; print('TICK_121354'); time.sleep(30)\"",
        cwd=str(tmp_path),
    )
    silent = registry.spawn_local(
        f"{exe} -u -c \"import time; time.sleep(30)\"",
        cwd=str(tmp_path),
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if "TICK_121354" in registry.poll(ticker.id).get("output_preview", ""):
                break
            time.sleep(0.2)
        tick_poll = registry.poll(ticker.id)
        silent_poll = registry.poll(silent.id)
        assert tick_poll.get("status") == "running"
        assert silent_poll.get("status") == "running"
        tick_age = tick_poll["last_output_age_s"]
        silent_age = silent_poll["last_output_age_s"]
        assert tick_age < 5.0, f"stale liveness for a ticking job: {tick_age!r}"
        assert tick_age < silent_age, (
            f"working vs wedged indistinguishable: tick={tick_age!r} silent={silent_age!r}"
        )
    finally:
        registry.kill_process(ticker.id)
        registry.kill_process(silent.id)
