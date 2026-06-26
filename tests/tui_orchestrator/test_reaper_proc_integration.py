"""Stage 3 reaper — REAL /proc integration test (no mocks for the scan).

Spawns real `sleep` processes carrying the orchestrator marker env, then proves:
  (A) scan_processes() finds them via /proc with the right owner/role.
  (B) an ORPHAN (marker owner pid that is dead) is planned + actually SIGTERMed.
  (C) a marked process whose owner is ALIVE and is in live_pids is NEVER touched.
  (D) a completely unmarked process is invisible to the scan.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def _spawn_marked(owner_pid: int, role: str, hb_file: str | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    env["HERMES_TUI_ORCH_OWNER_PID"] = str(owner_pid)
    env["HERMES_TUI_ORCH_ROLE"] = role
    if hb_file:
        env["HERMES_TUI_HEARTBEAT_FILE"] = hb_file
    return subprocess.Popen(["sleep", "60"], env=env)


def main() -> int:
    from tui_gateway import reaper

    me = os.getpid()
    procs: list[subprocess.Popen] = []

    # A DEAD owner pid: spawn a short-lived proc, get its pid, let it exit.
    dead_owner = subprocess.Popen(["true"])
    dead_owner.wait()
    dead_owner_pid = dead_owner.pid
    # Make sure it's really gone.
    assert not reaper.pid_alive(dead_owner_pid), "dead-owner pid still alive?"

    # (B) ORPHAN: marked child whose owner (dead_owner_pid) is gone.
    orphan = _spawn_marked(dead_owner_pid, "renderer")
    procs.append(orphan)

    # (C) KEEP: marked child owned by US (alive), and declared live.
    keep = _spawn_marked(me, "renderer")
    procs.append(keep)

    # (D) UNMARKED: a plain sleep with no marker — must be invisible.
    unmarked = subprocess.Popen(["sleep", "60"])
    procs.append(unmarked)

    time.sleep(0.5)  # let /proc settle

    try:
        # (A) scan finds the two marked ones, not the unmarked.
        snap = reaper.scan_processes()
        found = {p.pid: p for p in snap}
        assert orphan.pid in found, "scan missed the orphan"
        assert keep.pid in found, "scan missed the keep proc"
        assert unmarked.pid not in found, "scan wrongly included an UNMARKED process!"
        assert found[orphan.pid].owner_pid == dead_owner_pid
        assert found[orphan.pid].role == "renderer"
        print(f"PASS (A) scan: found orphan={orphan.pid} keep={keep.pid}, ignored unmarked={unmarked.pid}")

        # Plan: orphan's owner dead → orphan; keep is live → safe.
        owners = {p.owner_pid for p in snap if p.owner_pid is not None}
        alive_orchs = {me} | {o for o in owners if reaper.pid_alive(o)}
        plan = reaper.plan_reap(
            snap,
            my_orchestrator_pid=me,
            live_pids=[me, keep.pid],
            alive_orchestrator_pids=alive_orchs,
        )
        planned = set(plan.all_pids)
        assert orphan.pid in planned, "orphan not planned for reap"
        assert keep.pid not in planned, "LIVE keep proc wrongly planned for reap!"
        assert unmarked.pid not in planned, "unmarked wrongly planned!"
        print(f"PASS (B/C) plan: orphan={orphan.pid} reaped, keep={keep.pid} + unmarked spared")

        # Execute for real — SIGTERM the orphan.
        signalled = reaper.execute_reap(plan, log=lambda m: print(f"   {m}"))
        assert orphan.pid in signalled
        time.sleep(0.8)
        assert orphan.poll() is not None, "orphan survived SIGTERM"
        assert keep.poll() is None, "keep proc was killed — SAFETY VIOLATION"
        assert unmarked.poll() is None, "unmarked proc was killed — SAFETY VIOLATION"
        print(f"PASS (D) execute: orphan {orphan.pid} REALLY died; keep + unmarked still alive")

        print("\nREAPER /PROC INTEGRATION ALL-GREEN: real orphan reaped, live + unmarked untouched.")
        rc = 0
    except AssertionError as e:
        print(f"[FAIL] {e}")
        rc = 1
    finally:
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
