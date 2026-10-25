"""The sandbox Chromium is signalled only while its pid still names the browser this process launched.

The "sandbox" here is this host through a plugin environment's ``exec_prefix`` (``env``), so the real launch,
liveness and stop scripts run in a real shell against real processes. The stand-in Chromium and the unrelated
job both block SIGTERM, so a delivered signal stays pending in ``/proc/<pid>/status`` (no timing involved).
"""

import signal
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.platforms("linux")

_SIGTERM_BIT = 1 << (signal.SIGTERM - 1)

_STUB = textwrap.dedent("""\
    #!{python}
    import os, signal, socket, sys, time
    signal.pthread_sigmask(signal.SIG_BLOCK, {{signal.SIGTERM}})
    udd = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(8)
    tmp = os.path.join(udd, "DevToolsActivePort.tmp")
    with open(tmp, "w") as f:
        f.write(f"{{s.getsockname()[1]}}\\n/devtools/browser/stub\\n")
    os.replace(tmp, os.path.join(udd, "DevToolsActivePort"))
    time.sleep(120)
    """)


class _LocalSandbox:
    """A plugin terminal environment whose exec prefix runs argv on this host."""

    def exec_prefix(self, *, user=None, interactive=True):
        return ["env"]


def _sigterm_pending(pid: int) -> bool:
    with open(f"/proc/{pid}/status", encoding="ascii") as fh:
        fields = dict(line.split(":", 1) for line in fh if ":" in line)
    return bool(int(fields["ShdPnd"].strip(), 16) & _SIGTERM_BIT)


@pytest.fixture
def local_sandbox(tmp_path, monkeypatch):
    from tools import browser_session_sandbox
    from tools.bot_desktop import runtime, sandbox_host
    from tools.environments import streams

    stub = tmp_path / "chromium"
    stub.write_text(_STUB.format(python=sys.executable), encoding="utf-8")
    stub.chmod(0o755)
    env = _LocalSandbox()
    monkeypatch.setattr(browser_session_sandbox, "_launched", {})
    monkeypatch.setattr(browser_session_sandbox, "_confirmed", {})
    monkeypatch.setattr(streams, "forward_port", lambda env, port, user=None: port)
    monkeypatch.setattr(streams, "forward_failures", lambda env, port: 0)
    monkeypatch.setattr(streams, "sandbox_temp_dir", lambda env: str(tmp_path))
    monkeypatch.setattr(runtime, "_sandbox_env", lambda create=False: env)
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":5"})
    monkeypatch.setattr(sandbox_host, "_user_for", lambda env: None)
    monkeypatch.setattr(sandbox_host, "browser_profile_dir", lambda env: str(tmp_path / "profile"))
    monkeypatch.setattr(sandbox_host, "chromium_executable", lambda env: str(stub))
    yield browser_session_sandbox
    for entry in list(browser_session_sandbox._launched.values()):
        _kill_if_ours(entry[1], tmp_path)


def _kill_if_ours(pid: int, tmp_path) -> None:
    """SIGKILL the stand-in Chromium from inside the "sandbox" (its own ``kill``, as a crash or the sandbox's
    end would): it was started detached there, outside this test's process tree."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            ours = str(tmp_path).encode() in fh.read()
    except OSError:
        return
    if ours:
        subprocess.run(["kill", "-KILL", str(pid)], check=False)


def _launched_pid(sandbox_module, info) -> int:
    return sandbox_module._launched[info["launcher_key"]][1]


def test_stop_signals_the_chromium_it_launched(local_sandbox, tmp_path):
    info = local_sandbox.create_sandbox_session("task-A")
    pid = _launched_pid(local_sandbox, info)
    try:
        local_sandbox.stop_sandbox_browser(info)
        assert _sigterm_pending(pid)
    finally:
        _kill_if_ours(pid, tmp_path)


def test_stop_leaves_alone_a_pid_that_now_names_another_process(local_sandbox, tmp_path, caplog):
    """Chromium crashed and the sandbox reused its pid for an unrelated job: stale-session eviction or idle
    cleanup must not SIGTERM that job."""
    info = local_sandbox.create_sandbox_session("task-A")
    key = info["launcher_key"]
    pid = _launched_pid(local_sandbox, info)
    _kill_if_ours(pid, tmp_path)  # the crash
    job = subprocess.Popen(  # the unrelated job now holding the recorded pid number
        [sys.executable, "-c", "import signal,sys,time; signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM}); "
         "print('ready', flush=True); time.sleep(120)"], stdout=subprocess.PIPE, text=True)
    try:
        assert job.stdout.readline().strip() == "ready"
        env, _pid, *identity = local_sandbox._launched[key]
        local_sandbox._launched[key] = (env, job.pid, *identity)
        with caplog.at_level("INFO", logger="tools.browser_session_sandbox"):
            local_sandbox.stop_sandbox_browser(info)
        assert not _sigterm_pending(job.pid), "an unrelated process on a reused pid was signalled"
        assert str(job.pid) in caplog.text
    finally:
        job.kill()
        job.wait()
