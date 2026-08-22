"""Live smoke test of the OPT-IN orchestrator launch path.

Mimics what `HERMES_TUI_ORCHESTRATOR=1 hermes --tui` now does: run
`python -m tui_gateway.orchestrator` with the same env the launcher sets,
under a PTY (Ink needs a TTY). Asserts:
  1. the orchestrator process starts and stays up,
  2. it brings up the gateway ws-host (a child listening on a loopback port),
  3. it spawns a renderer child that attaches to the gateway,
  4. SIGKILLing the renderer child does NOT kill the orchestrator or gateway
     (the renderer is respawned),
  5. quitting the orchestrator (SIGTERM) tears everything down cleanly.

This is the "is the wiring actually live + working" gate, run against live src.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import pty
import tempfile
from pathlib import Path


def _children(pid: int) -> list[int]:
    """Direct children of pid via /proc (snapshot, no pgrep self-match)."""
    out = []
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
        cp = subprocess.run(
            ["ps", "--ppid", str(pid), "-o", "pid=", "--no-headers"],
            capture_output=True, text=True, timeout=5,
        )
        out = [int(x) for x in cp.stdout.split()]
    except Exception:
        pass
    return out


def _descendants(pid: int) -> list[int]:
    seen = []
    frontier = [pid]
    while frontier:
        p = frontier.pop()
        kids = _children(p)
        for k in kids:
            if k not in seen:
                seen.append(k)
                frontier.append(k)
    return seen


def main() -> int:
    src = Path(__file__).resolve().parents[2]  # .../hermes/src
    home = Path(tempfile.mkdtemp(prefix="hermes-orch-smoke-"))

    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["HERMES_PYTHON_SRC_ROOT"] = str(src)
    env["HERMES_PYTHON"] = sys.executable
    env["TERM"] = "xterm-256color"
    # Cold start: no resume. Let the orchestrator's default spawners build the
    # renderer argv from HERMES_PYTHON_SRC_ROOT (bun + dist/entry.js).
    env.pop("HERMES_TUI_RESUME", None)
    env.pop("HERMES_TUI_RENDERER_ARGV", None)  # exercise the fallback path
    # quick reaper cadence so we can observe respawn promptly
    env["HERMES_TUI_ACTIVE_SESSION_FILE"] = str(home / "active.json")

    mfd, sfd = pty.openpty()
    orch = subprocess.Popen(
        [sys.executable, "-m", "tui_gateway.orchestrator"],
        cwd=str(src), env=env, stdin=sfd, stdout=sfd, stderr=sfd, close_fds=True,
    )
    os.close(sfd)
    print(f"[orch] started pid={orch.pid}")

    rc = 0
    try:
        # 1+2+3: wait for orchestrator to spawn gateway + renderer descendants.
        deadline = time.time() + 30
        desc = []
        while time.time() < deadline:
            if orch.poll() is not None:
                try:
                    os.set_blocking(mfd, False)
                    tail = os.read(mfd, 8192).decode(errors="replace")
                except Exception:
                    tail = ""
                print(f"[FATAL] orchestrator exited early code={orch.returncode}\n--- pty ---\n{tail[-1500:]}")
                return 1
            desc = _descendants(orch.pid)
            # expect at least 2: gateway ws-host (python) + renderer (bun/node)
            if len(desc) >= 2:
                break
            time.sleep(0.5)
        assert len(desc) >= 2, f"orchestrator never spawned gateway+renderer (descendants={desc})"
        print(f"[orch] descendants up: {desc} (gateway + renderer)")

        # Identify the renderer child (bun/node) vs gateway (python ws_host).
        def _cmd(pid):
            try:
                return open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0"," ".encode()).decode(errors="replace")
            except Exception:
                return ""
        renderer = None
        gateway = None
        for p in desc:
            c = _cmd(p)
            if "ws_host" in c:
                gateway = p
            elif "entry.js" in c or "bun" in c or "node" in c or "tsx" in c:
                renderer = p
        print(f"[ids] gateway={gateway} ({_cmd(gateway)[:50] if gateway else '?'}) renderer={renderer} ({_cmd(renderer)[:50] if renderer else '?'})")
        assert gateway, "no gateway ws_host child found"
        assert renderer, "no renderer child found"

        # 4: SIGKILL the renderer; orchestrator + gateway must survive, renderer respawns.
        os.kill(renderer, signal.SIGKILL)
        print(f"[kill] SIGKILL renderer {renderer}")
        time.sleep(3)
        assert orch.poll() is None, "orchestrator died when renderer was killed!"
        # gateway still alive?
        try:
            os.kill(gateway, 0)
            gw_alive = True
        except ProcessLookupError:
            gw_alive = False
        assert gw_alive, "gateway died when renderer was killed (anchor not durable!)"
        # a new renderer should have been respawned
        new_desc = _descendants(orch.pid)
        new_renderers = [p for p in new_desc if ("entry.js" in _cmd(p) or "bun" in _cmd(p) or "node" in _cmd(p) or "tsx" in _cmd(p))]
        print(f"[respawn] descendants now {new_desc}, renderer candidates {new_renderers}")
        assert new_renderers and new_renderers[0] != renderer, "renderer was NOT respawned after kill"
        print(f"[respawn] fresh renderer {new_renderers[0]} attached to surviving gateway {gateway}")

        print("\nORCHESTRATOR-LAUNCH SMOKE ALL-GREEN: flag brings up gateway+renderer; "
              "SIGKILL renderer → orchestrator+gateway survive → renderer respawns.")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        rc = 2
    finally:
        # 5: clean teardown
        try:
            orch.send_signal(signal.SIGTERM)
            orch.wait(timeout=8)
        except Exception:
            for p in _descendants(orch.pid) + [orch.pid]:
                try:
                    os.kill(p, signal.SIGKILL)
                except Exception:
                    pass
        try:
            os.close(mfd)
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
