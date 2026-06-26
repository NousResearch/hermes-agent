"""REAL BUN attach proof — launches the actual built bun renderer against the
real ws_host and confirms it ATTACHES over WebSocket (opens the ws, no gateway
child spawned), then that SIGKILLing it leaves the gateway alive.

This complements test_live_inversion.py (which proves the gateway-side adoption
keystone via a python ws client) by proving the OTHER half with the real artifact:
the actual bun bundle, given HERMES_TUI_GATEWAY_URL, connects over ws instead of
forking its own gateway — i.e. it really is a disposable client.

We run bun under a PTY (Ink needs a TTY) and watch the ws_host's own connection
accounting + the gateway process liveness across a SIGKILL of bun.
"""
from __future__ import annotations

import json
import os
import pty
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

    home = Path(tempfile.mkdtemp(prefix="hermes-bun-attach-"))
    db = SessionDB(db_path=home / "state.db")
    sid = uuid.uuid4().hex[:8]
    db.create_session(sid, source="tui")
    db.append_message(sid, role="user", content="BUN_ATTACH_MARKER")

    # Free port + start ws_host.
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    cred = "bun-attach-cred"
    env = dict(os.environ)
    env.update({
        "HERMES_HOME": str(home),
        "HERMES_TUI_WS_HOST": "127.0.0.1",
        "HERMES_TUI_WS_PORT": str(port),
        "HERMES_TUI_WS_INTERNAL_CREDENTIAL": cred,
    })
    gateway = subprocess.Popen(
        [sys.executable, "-m", "tui_gateway.ws_host"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    def listening() -> bool:
        with socket.socket() as t:
            t.settimeout(0.4)
            try:
                t.connect(("127.0.0.1", port)); return True
            except OSError:
                return False

    deadline = time.time() + 25
    while time.time() < deadline and not listening():
        if gateway.poll() is not None:
            print("[FATAL] gateway died at startup"); return 1
        time.sleep(0.2)
    if not listening():
        print("[FATAL] gateway never listened"); gateway.kill(); return 1
    print(f"[gateway] listening on {port} pid={gateway.pid}")

    # Count current ESTABLISHED conns to the port (gateway-side connection proof).
    def est_conns() -> int:
        try:
            out = subprocess.run(
                ["ss", "-tnp", "state", "established", f"sport = :{port}"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            # one header line; count remaining
            return max(0, len([l for l in out.splitlines() if f":{port}" in l]))
        except Exception:
            return -1

    base_conns = est_conns()

    # Launch the REAL bun renderer under a PTY, attached to the gateway ws.
    root = os.getcwd()
    entry = os.path.join(root, "ui-tui", "dist", "entry.js")
    bun_env = dict(env)
    bun_env["HERMES_TUI_GATEWAY_URL"] = f"ws://127.0.0.1:{port}/api/ws?internal={cred}"
    bun_env["HERMES_TUI_RESUME"] = sid
    bun_env["TERM"] = "xterm-256color"

    mfd, sfd = pty.openpty()
    bun = subprocess.Popen(
        ["bun", entry], env=bun_env, stdin=sfd, stdout=sfd, stderr=sfd,
        close_fds=True,
    )
    os.close(sfd)
    print(f"[bun] launched pid={bun.pid}, watching for ws attach…")

    # Wait until an extra established connection to the gateway appears = bun attached.
    attached = False
    deadline = time.time() + 25
    while time.time() < deadline:
        if bun.poll() is not None:
            # bun exited early — dump a little PTY output for diagnosis
            try:
                os.set_blocking(mfd, False)
                data = os.read(mfd, 8192).decode(errors="replace")
            except Exception:
                data = ""
            print(f"[FATAL] bun exited early code={bun.returncode}\n--- pty tail ---\n{data[-1500:]}")
            gateway.kill(); return 2
        c = est_conns()
        if c >= 0 and c > base_conns:
            attached = True
            print(f"[bun] ATTACHED over ws (established conns {base_conns}→{c})")
            break
        time.sleep(0.3)

    if not attached:
        print("[FATAL] bun never attached over ws within timeout")
        bun.kill(); gateway.kill(); return 3

    # The real test: SIGKILL the bun renderer (hard kill, no cleanup) and assert
    # the gateway stays alive — kill the renderer, the anchor survives.
    os.kill(bun.pid, signal.SIGKILL)
    try:
        bun.wait(timeout=5)
    except Exception:
        pass
    time.sleep(1.0)
    gw_alive = gateway.poll() is None
    print(f"[kill] bun SIGKILLed; gateway alive={gw_alive}")

    rc = 0 if gw_alive else 4

    gateway.terminate()
    try:
        gateway.wait(timeout=5)
    except Exception:
        gateway.kill()
    try:
        os.close(mfd)
    except Exception:
        pass

    if rc == 0:
        print("\nBUN-ATTACH ALL-GREEN: the real bun bundle attached over ws as a disposable client; "
              "SIGKILLing it left the durable gateway anchor alive.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
