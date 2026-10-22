"""Lightpanda servers a pre-update Hermes left behind are reaped once; a live owner's are never touched.

Servers are owned children now (spawn_owned) and write no record. Before, each launch wrote
``<state_dir>/<session>.json`` and an orphan sweep reaped the server after its owner died. A server
left by a crashed pre-update Hermes has only that record, so the browser janitor still sweeps the
old format, identity-checked against the real process.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from tests.waiting import wait_until

# The orphan stand-in is killed by the code under test; it is detached like the real server.
pytestmark = [pytest.mark.platforms("posix"), pytest.mark.live_system_guard_bypass]




def _legacy_lightpanda(exe: Path, port: int) -> subprocess.Popen:
    """A stand-in ``lightpanda serve`` a pre-update Hermes started (name + argv are what identity checks)."""
    return subprocess.Popen([str(exe), "-c", "sleep 120; :", "serve", "--port", str(port)],  # detached-spawn: ok — test stand-in, killed in finally
                            stdin=subprocess.DEVNULL, start_new_session=True)

def test_a_server_left_by_a_crashed_pre_update_hermes_is_reaped_but_a_live_owners_is_not(tmp_path, monkeypatch):
    from gateway.status import get_process_start_time
    from tools import browser_lightpanda as lp

    monkeypatch.setattr(lp, "_state_dir", lambda: tmp_path)
    # A bash copy keeps its own name (macOS /bin/sh re-execs). Written once: overwriting a running
    # executable on macOS kills the processes running it. Resolved on PATH: NixOS has no /bin/bash.
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash on PATH")
    exe = tmp_path / "bin" / "lightpanda"
    exe.parent.mkdir()
    exe.write_bytes(Path(bash).read_bytes())
    exe.chmod(0o755)
    orphan, kept = _legacy_lightpanda(exe, 43111), _legacy_lightpanda(exe, 43112)
    dead_owner = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_owner.wait(timeout=10)
    live_owner = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        for name, proc, port, owner in (("lp_old", orphan, 43111, dead_owner.pid), ("lp_live", kept, 43112, live_owner.pid)):
            wait_until(lambda proc=proc: "serve" in psutil.Process(proc.pid).cmdline(), what="stand-in started")
            (tmp_path / f"{name}.json").write_text(json.dumps({  # the pre-update record format
                "pid": proc.pid, "port": port, "owner_pid": owner,
                "start_time": get_process_start_time(proc.pid), "started_at": 0}), encoding="utf-8")

        assert lp.reap_legacy_lightpanda_records() == 1
        assert orphan.wait(timeout=10) is not None
        assert kept.poll() is None, "a live pre-update owner's server was killed"
        assert not (tmp_path / "lp_old.json").exists() and (tmp_path / "lp_live.json").exists()
    finally:
        live_owner.stdin.close()
        live_owner.wait(timeout=10)
        for proc in (orphan, kept):
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
