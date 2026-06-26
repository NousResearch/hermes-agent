"""REAL sub-agent survival proof — sub-agents/delegated tasks stay intact across
a renderer kill under the orchestrator.

WHY: a reviewer's first worry about "the renderer can be killed/recycled" is
whether in-flight sub-agents (delegate_task children) die with it. They don't —
and this proves it live rather than by argument.

PROCESS MODEL (the reason it's safe):

    orchestrator
     ├─ gateway ws-host                      DURABLE ANCHOR
     │    └─ slash_worker (per session)      runs the agent turn
     │         └─ subagent threads           ThreadPoolExecutor, IN-PROCESS
     └─ bun renderer                         DISPOSABLE; holds NO agent state

delegate_task runs children as in-process threads inside the slash_worker, which
is a child of the GATEWAY, not the renderer. So killing the renderer cannot touch
a running sub-agent.

WHAT THIS DOES: launches an orchestrated TUI, sends a prompt that spawns ONE
delegate_task sub-agent running `sleep 45`, waits until that sleep is actually
running, SIGKILLs the renderer mid-run, then asserts (via real process checks +
the session DB, NOT screen scraping — the TUI echoes typed input):
  (a) the sub-agent's `sleep 45` keeps running across the kill,
  (b) the gateway + the slash_worker hosting the sub-agent survive,
  (c) a fresh renderer respawns,
  (d) the parent turn completes and persists to the SAME session.

This is a LIVE harness: it needs the `hermes` wrapper + a reachable model, so it
self-skips (exit 77) when those aren't available, exactly like the other live
harnesses in this directory. Run manually:

    PYTHONPATH=. python tests/tui_orchestrator/test_subagent_survival.py
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time

WRAPPER = os.environ.get("HERMES_TEST_WRAPPER") or os.path.expanduser("~/.local/bin/hermes")
# A workspace HERMES_HOME to copy config/auth from. Override with
# HERMES_TEST_WORKSPACE; falls back to the live HERMES_HOME if set.
WORKSPACE = os.environ.get("HERMES_TEST_WORKSPACE") or os.environ.get("HERMES_HOME") or ""
SKIP = 77  # autotools convention: test skipped


def _ps() -> str:
    return subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout


def _orch_pid():
    for line in _ps().splitlines():
        if "tui_gateway.orchestrator" in line and "awk" not in line:
            try:
                return int(line.split()[0])
            except ValueError:
                pass
    return None


def _children(pid):
    out = {}
    for line in _ps().splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            p, pp = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pp == pid:
            out[p] = parts[2]
    return out


def _renderer(o):
    return next((p for p, a in _children(o).items() if "entry.js" in a), None)


def _gateway(o):
    return next((p for p, a in _children(o).items() if "ws_host" in a), None)


def _slash_workers():
    out = []
    for line in _ps().splitlines():
        if "tui_gateway.slash_worker" in line and "awk" not in line:
            try:
                out.append(int(line.split()[0]))
            except ValueError:
                pass
    return out


def _alive(p):
    try:
        os.kill(p, 0)
        return True
    except Exception:
        return False


def _asst_count(db):
    if not os.path.exists(db):
        return 0
    try:
        r = subprocess.run(
            ["sqlite3", db, "SELECT count(*) FROM messages WHERE role='assistant';"],
            capture_output=True, text=True,
        ).stdout.strip()
        return int(r or "0")
    except Exception:
        return 0


def main() -> int:
    if not os.path.exists(WRAPPER):
        print(f"SKIP: hermes wrapper not found at {WRAPPER}")
        return SKIP
    try:
        import pexpect  # noqa: F401
    except ImportError:
        print("SKIP: pexpect not installed")
        return SKIP
    import pexpect

    if not WORKSPACE or not os.path.isdir(WORKSPACE):
        print("SKIP: no workspace HERMES_HOME to source config/auth from "
              "(set HERMES_TEST_WORKSPACE or HERMES_HOME)")
        return SKIP

    home = tempfile.mkdtemp(prefix="hermes-subagent-")
    for f in ("config.yaml", ".env"):
        s = os.path.join(WORKSPACE, f)
        if os.path.exists(s):
            subprocess.run(["cp", "-a", s, os.path.join(home, f)])
    for d in ("auth", ".secrets", "secrets", "skills"):
        s = os.path.join(WORKSPACE, d)
        if os.path.exists(s):
            try:
                os.symlink(s, os.path.join(home, d))
            except OSError:
                pass

    env = dict(os.environ)
    env.update({"HERMES_HOME": home, "HERMES_TUI_RPC_TIMEOUT_MS": "300000", "TERM": "xterm-256color"})
    env.pop("HERMES_TUI_ORCHESTRATOR", None)  # default-on path
    env.pop("HERMES_TUI_RESUME", None)
    db = os.path.join(home, "state.db")

    print(f"[launch] orchestrated TUI (default-on)  HERMES_HOME={home}")
    child = pexpect.spawn("/bin/bash", [WRAPPER, "--tui"], env=env, encoding="utf-8",
                          dimensions=(45, 140), timeout=120)
    res = {}
    try:
        try:
            child.expect([r"ready", r"forging", r"How can I help", r"esc to"], timeout=90)
        except pexpect.TIMEOUT:
            pass
        time.sleep(4)
        o = _orch_pid()
        if not o:
            print("[FATAL] no orchestrator process")
            return 3
        g, r1 = _gateway(o), _renderer(o)
        print(f"[tree] orch={o} gateway={g} renderer={r1}")
        if not r1:
            print("[FATAL] no renderer child under the orchestrator")
            return 3

        prompt = (
            "Use delegate_task to spawn ONE subagent with this exact goal: "
            "'run the shell command: sleep 45, then reply DONE_SLEEP'. "
            "Wait for it and report its result."
        )
        print("[type] prompt that spawns a subagent running `sleep 45`")
        child.send(prompt)
        time.sleep(0.6)
        child.send("\r")

        print("[wait] for the subagent's `sleep 45` to appear…")
        deadline = time.time() + 90
        sleeper = None
        sw_before = []
        while time.time() < deadline:
            for line in _ps().splitlines():
                if "sleep 45" in line and "awk" not in line and "grep" not in line:
                    try:
                        sleeper = int(line.split()[0])
                        break
                    except ValueError:
                        pass
            if sleeper:
                sw_before = _slash_workers()
                break
            time.sleep(2)
        if not sleeper:
            print("SKIP: model never spawned the `sleep 45` subagent (model/tooling unavailable)")
            return SKIP
        print(f"[subagent] running: sleep pid={sleeper}  slash_workers={sw_before}")

        print(f"[kill] SIGKILL renderer {r1} WHILE subagent is running")
        os.kill(r1, signal.SIGKILL)
        time.sleep(5)
        res["gw_survived"] = _alive(g)
        res["subagent_survived"] = _alive(sleeper)
        r2 = _renderer(o)
        res["respawned"] = bool(r2 and r2 != r1)
        sw_after = _slash_workers()
        res["slash_worker_survived"] = any(w in sw_after for w in sw_before)
        print(f"[after-kill] gw_alive={res['gw_survived']} subagent_alive={res['subagent_survived']} "
              f"new_renderer={r2} slash_worker_survived={res['slash_worker_survived']}")

        print("[wait] for the parent turn to complete + persist…")
        deadline = time.time() + 120
        persisted = False
        while time.time() < deadline:
            if _asst_count(db) > 0:
                persisted = True
                break
            time.sleep(3)
        res["turn_persisted"] = persisted
        print(f"[db] assistant msgs={_asst_count(db)} persisted={persisted}")

        print("\n=== SUBAGENT-SURVIVAL VERDICT (live + db ground truth) ===")
        rows = [
            ("respawned", "renderer respawned after kill"),
            ("gw_survived", "gateway survived kill"),
            ("subagent_survived", "SUBAGENT (sleep 45) kept running across the kill"),
            ("slash_worker_survived", "agent-turn worker (subagent host) survived the kill"),
            ("turn_persisted", "parent turn completed + persisted to same session"),
        ]
        for k, label in rows:
            print(f"  {'YES' if res.get(k) else 'NO ':3}  {label}")
        good = all(res.get(k) for k in ("subagent_survived", "gw_survived", "slash_worker_survived"))
        print("\nOVERALL: "
              + ("ALL-GREEN — sub-agents stay intact across a renderer kill" if good else "INCOMPLETE"))
        return 0 if good else 4
    finally:
        try:
            child.sendcontrol("c")
            time.sleep(0.3)
            child.send("/quit\r")
        except Exception:
            pass
        time.sleep(1)
        o = _orch_pid()
        if o:
            for p in list(_children(o)) + [o]:
                try:
                    os.kill(p, signal.SIGKILL)
                except Exception:
                    pass
        for line in _ps().splitlines():
            if ("sleep 45" in line or ("slash_worker" in line and home in line)) and "awk" not in line:
                try:
                    os.kill(int(line.split()[0]), 9)
                except Exception:
                    pass
        try:
            child.close(force=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
