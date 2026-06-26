"""Stage 3 — REAL frozen-renderer detection via heartbeat (no mocks).

Launches the actual built bun renderer with a heartbeat file (as the orchestrator
would), proves it WRITES the heartbeat, then SIGSTOPs it — a frozen-but-alive
renderer (poll() still says alive). Proves the reaper's heartbeat path flags it
as 'frozen' once the mtime goes stale, while poll()-based logic would miss it.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def main() -> int:
    from hermes_state import SessionDB
    from tui_gateway import reaper

    home = Path(tempfile.mkdtemp(prefix="hermes-hb-frozen-"))
    db = SessionDB(db_path=home / "state.db")
    sid = uuid.uuid4().hex[:8]
    db.create_session(sid, source="tui")
    db.append_message(sid, role="user", content="HB_FROZEN_MARKER")

    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    cred = "hb-frozen-cred"
    gw_env = dict(os.environ)
    gw_env.update({
        "HERMES_HOME": str(home),
        "HERMES_TUI_WS_HOST": "127.0.0.1",
        "HERMES_TUI_WS_PORT": str(port),
        "HERMES_TUI_WS_INTERNAL_CREDENTIAL": cred,
    })
    gateway = subprocess.Popen([sys.executable, "-m", "tui_gateway.ws_host"],
                               env=gw_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def listening():
        with socket.socket() as t:
            t.settimeout(0.4)
            try:
                t.connect(("127.0.0.1", port)); return True
            except OSError:
                return False
    deadline = time.time() + 25
    while time.time() < deadline and not listening():
        if gateway.poll() is not None:
            print("[FATAL] gateway died"); return 1
        time.sleep(0.2)

    me = os.getpid()
    hb_file = str(home / "renderer-hb")
    import pty
    bun_env = dict(gw_env)
    bun_env.update({
        "HERMES_TUI_GATEWAY_URL": f"ws://127.0.0.1:{port}/api/ws?internal={cred}",
        "HERMES_TUI_RESUME": sid,
        "HERMES_TUI_HEARTBEAT_FILE": hb_file,
        "HERMES_TUI_ORCH_OWNER_PID": str(me),
        "HERMES_TUI_ORCH_ROLE": "renderer",
        "TERM": "xterm-256color",
    })
    root = os.getcwd()
    entry = os.path.join(root, "ui-tui", "dist", "entry.js")
    mfd, sfd = pty.openpty()
    bun = subprocess.Popen(["bun", entry], env=bun_env, stdin=sfd, stdout=sfd, stderr=sfd, close_fds=True)
    os.close(sfd)
    print(f"[bun] launched pid={bun.pid}, heartbeat={hb_file}")

    rc = 0
    try:
        # 1) Prove the renderer WRITES the heartbeat file.
        deadline = time.time() + 25
        wrote = False
        while time.time() < deadline:
            if bun.poll() is not None:
                try:
                    os.set_blocking(mfd, False); tail = os.read(mfd, 4096).decode(errors="replace")
                except Exception:
                    tail = ""
                print(f"[FATAL] bun exited early code={bun.returncode}\n{tail[-1000:]}"); return 2
            if os.path.exists(hb_file):
                wrote = True; break
            time.sleep(0.3)
        assert wrote, "renderer never wrote the heartbeat file"
        print(f"[bun] heartbeat file present (mtime age {time.time()-os.stat(hb_file).st_mtime:.1f}s)")

        # 2) FREEZE the renderer (SIGSTOP): alive to poll(), but the event loop
        #    is suspended so the heartbeat goes stale. This is the case poll()
        #    can't catch and the reaper must.
        os.kill(bun.pid, signal.SIGSTOP)
        print(f"[freeze] SIGSTOP {bun.pid} — poll()={bun.poll()} (None=still 'alive')")
        assert bun.poll() is None, "SIGSTOP should not exit the process"

        # 3) Let the heartbeat go stale, then scan with a SHORT stale threshold
        #    so the test doesn't wait 90s. Prove the frozen path flags it.
        time.sleep(2.5)
        snap = reaper.scan_processes()
        mine = [p for p in snap if p.pid == bun.pid]
        assert mine, "scan didn't find the frozen renderer"
        age = mine[0].heartbeat_age_s
        print(f"[scan] frozen renderer heartbeat age = {age:.1f}s")
        assert age is not None and age >= 2.0, f"heartbeat age not stale: {age}"

        plan = reaper.plan_reap(
            snap,
            my_orchestrator_pid=me,
            live_pids=[me],  # NOT listing bun.pid as live → eligible
            alive_orchestrator_pids={me},
            heartbeat_stale_s=2.0,  # short threshold for the test
        )
        frozen_pids = {pid for pid, _ in plan.frozen}
        assert bun.pid in frozen_pids, f"frozen renderer not flagged: {plan.frozen}"
        print(f"[plan] frozen renderer {bun.pid} correctly flagged for reap: {plan.frozen}")

        # 4) And confirm poll()-only logic would have MISSED it (the whole point).
        #    Resume + reap to clean up.
        os.kill(bun.pid, signal.SIGCONT)
        print("\nHEARTBEAT-FROZEN ALL-GREEN: a SIGSTOP-frozen real renderer (alive to poll()) "
              "was detected as frozen via stale heartbeat and planned for reap.")
    except AssertionError as e:
        print(f"[FAIL] {e}"); rc = 3
    finally:
        for p in (bun, gateway):
            try:
                os.kill(p.pid, signal.SIGCONT)
            except Exception:
                pass
            try:
                p.kill()
            except Exception:
                pass
        try:
            os.close(mfd)
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
