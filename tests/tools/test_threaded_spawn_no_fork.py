"""Tool spawn sites must not fork() a threaded macOS parent (#97296).

The Desktop backend and the gateway are threaded. On macOS a child fork()ed from such a
parent can die in a Network.framework atfork handler before exec (SIGSEGV, exit -11, empty
output: every terminal call, background process, search, cron script and kanban liveness
probe failing at once). Each site below asks for a cwd, a new session or close_fds, which
takes CPython 3.11 off posix_spawn unless the spawn goes through ``fork_safe_popen``; the
kanban liveness probe must not spawn ``ps`` at all.

Observed through CPython's own audit events in a fresh threaded interpreter
(``subprocess.Popen`` fires for every spawn, ``os.posix_spawn`` only on the no-fork path).
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_HARNESS = textwrap.dedent("""
    import json, os, subprocess, sys, threading
    events, current = {}, [None]

    def hook(event, args):
        if current[0] and event in ("subprocess.Popen", "os.posix_spawn"):
            events.setdefault(current[0], []).append(event)

    sys.addaudithook(hook)
    threading.Thread(target=threading.Event().wait, daemon=True).start()  # a threaded parent
    from cron.scheduler_script import _run_job_script
    from hermes_cli.kanban_db_dispatch import _pid_alive
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    from tools.process_registry import ProcessRegistry

    def site(name, fn):
        current[0] = name
        try:
            return fn()
        finally:
            current[0] = None

    work, out = sys.argv[1], {}
    env = site("terminal", lambda: LocalEnvironment(cwd=work))
    out["terminal"] = site("terminal", lambda: env.execute("pwd", timeout=60)["output"].strip())
    ops = ShellFileOperations(env, cwd=work)
    out["search_files"] = site(
        "search_files", lambda: ops._run_rg_native(["sh", "-c", "pwd"], 5, timeout=60).stdout.strip())
    out["process_registry"] = site(
        "process_registry", lambda: ProcessRegistry().spawn_local("pwd", cwd=work).process.wait(60))
    scripts = os.path.join(os.environ["HERMES_HOME"], "scripts")
    os.makedirs(scripts, exist_ok=True)
    with open(os.path.join(scripts, "probe.sh"), "w") as f:
        f.write("pwd\\n")
    out["cron_script"] = site("cron_script", lambda: _run_job_script("probe.sh", workdir=work))
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    out["kanban_pid_alive"] = site("kanban_pid_alive", lambda: _pid_alive(sleeper.pid))
    sleeper.kill()
    sleeper.wait()
    forked = {name: evs for name, evs in events.items()
              if evs.count("subprocess.Popen") != evs.count("os.posix_spawn")}
    print(json.dumps({"forked": forked, "out": out}))
""")


@pytest.mark.macos_only
@pytest.mark.live_system_guard_bypass
def test_tool_spawn_sites_never_fork_a_threaded_parent(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    harness = subprocess.run(
        [sys.executable, "-c", _HARNESS, str(work)], cwd=_REPO_ROOT,
        capture_output=True, text=True, timeout=300, start_new_session=True)
    assert harness.returncode == 0, harness.stderr[-3000:]
    report = json.loads(harness.stdout.splitlines()[-1])
    assert report["forked"] == {}, f"fork()ed at: {sorted(report['forked'])}"
    out, real = report["out"], os.path.realpath(work)
    assert (out["terminal"], out["search_files"], out["process_registry"]) == (real, real, 0)
    assert out["cron_script"][0] is True and real in out["cron_script"][1]
    assert out["kanban_pid_alive"] is True
