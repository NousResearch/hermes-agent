"""Region E — browser_exec guard orchestration (integration surface).

Wires Regions A (CDP network listener), B (socket egress interposer),
C (CDP Fetch page guard) and D (shared URL-safety helper) into
``browser_exec`` per the adjudicated consensus contract:

* **H1** ``_prepare_guard`` — armed components (floor always; private unless
  ``allow_private_urls``), policy snapshot, per-exec token, host-owned state
  dir, ``BH_RUNTIME_DIR`` pinning, endpoint tier resolution (T1 env /
  T1b provider re-resolution / T2 OS-attested loopback), and pre-spawn
  attachment of the network listener + the Fetch guard process. Any arm
  failure fails the exec closed BEFORE the CLI spawns.
* **H2** ``_guard_env`` — PYTHONPATH (egress interposer) + policy JSON +
  token + state dir + ``BH_RUNTIME_DIR`` pin; injection failure → self-test
  fails → withhold.
* **H3** ``_guard_self_test`` — policy-parameterized probe subprocess in the
  same interposer env (workspace-stripped); any failed assertion withholds
  before spawn.
* **H6** ``_BrowserExecGuardListener`` — browser-level CDP listener
  (``Target.setAutoAttach`` + per-session ``Network.enable``/``Page.enable``,
  per-target last-known-URL map, WS-drop attestation gap).
* **H9** ``_guard_endstate_verdict`` — the §10 agreement truth table
  (coverage precondition + V/L/P/M rows).
* **H11** ``_strip_guard_markers`` — only report lines stripped; page content
  echoing markers preserved.

The in-CLI preamble (``_GUARD_PREAMBLE``) is ADVISORY by construction: a
poisoned workspace ``agent_helpers.py`` runs before it (import-time seam),
so nothing authoritative lives in the CLI namespace — the host never learns
an attach target from stdout, and every marker claim is cross-checked
against daemon state / the trusted probe before it influences the verdict.
"""

import json
import logging
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Hosted tri-state armed markers emitted by the in-CLI preamble (advisory).
ARMED_MARKER = "__HERMES_BROWSER_EXEC_ARMED__:"
ANNOUNCE_MARKER = "__HERMES_BROWSER_EXEC_ANNOUNCE__:"
SELF_TEST_OK_MARKER = "__HERMES_BROWSER_EXEC_GUARD_OK__"

# Bounded wait for the honest-path attach-ack file.
_GUARD_ATTACH_TIMEOUT_S = 20.0

# Env keys the model can control and that must never reach the guard's
# self-test probe or the trusted landing probe (extends the P1 strip set).
_HERMES_CONTROLLED_ENV_KEYS = frozenset({
    "BH_AGENT_WORKSPACE",
    "BU_WORKSPACE",
})

# Marker lines that report guard state (stripped from returned output).
_GUARD_REPORT_LINE_RE = re.compile(
    r"^__(HERMES_BROWSER_EXEC_(ARMED|ANNOUNCE)|HERMES_EGRESS_GUARD__:|"
    r"HERMES_SSRF_GUARD_(READY|ARM_FAILED))"
)


def _read_guard_config() -> dict:
    """Guard config knobs (Region A + E) from the ``browser:`` section."""
    try:
        from tools.browser_use_cli import _read_browser_cfg

        cfg = _read_browser_cfg()
    except Exception:
        cfg = {}
    enabled = not (
        str(cfg.get("exec_network_monitor") or "").strip().lower() == "off"
        or cfg.get("exec_network_monitor") is False
    )
    return {
        "enabled": enabled,
        "fail_open": bool(cfg.get("exec_monitor_fail_open")),
        "grace_s": float(cfg.get("exec_monitor_grace_s") or 1.0),
        "attach_timeout_s": float(cfg.get("exec_monitor_attach_timeout_s") or 15.0),
        "allow_private": bool(cfg.get("allow_private_urls")),
    }


# ── Endpoint trust tiers (H12) ─────────────────────────────────────────────

def _resolve_endpoint_tier(env: dict, task_id: Optional[str], session: Optional[str]) -> tuple:
    """Resolve the browser endpoint and its attestation tier.

    Returns ``(endpoint, tier, error)``. Every endpoint the guard attaches
    to must be host-attested:

    * ``t1`` — endpoint present in the spawn env (``BU_CDP_WS``/``BU_CDP_URL``,
      set by ``_ensure_exec_cdp_endpoint``: operator override, cloud provider
      session, or the Hermes-supervised local Chrome fallback). Attach
      happens PRE-SPAWN.
    * ``t1b`` — named ``BU_NAME`` sessions whose endpoint is not yet in env:
      the host re-resolves the same browser via the provider machinery.
    * ``t2`` — OS-attested loopback discovery (daemon port file under the
      pinned ``BH_RUNTIME_DIR``; ping/pid cross-check). Used only when no
      env endpoint exists.

    UNATTESTED (stdout-derived-only) origins are never attached to: the
    caller withholds.
    """
    endpoint = (env.get("BU_CDP_WS") or env.get("BU_CDP_URL") or "").strip()
    if endpoint:
        return endpoint, "t1", None

    # T1b: provider re-resolution for named/cloud sessions.
    try:
        from tools.browser_tool import _get_cloud_provider, _get_session_info

        provider = _get_cloud_provider()
        if provider is not None:
            info = _get_session_info(task_id or "browser-exec-default")
            cdp = str((info or {}).get("cdp_url") or "").strip()
            if cdp:
                env["BU_CDP_URL" if cdp.startswith(("http://", "https://")) else "BU_CDP_WS"] = cdp
                return cdp, "t1b", None
            return "", "", (
                f"cannot attest the browser endpoint for session {session or ''}: "
                "provider returned no CDP URL"
            )
    except Exception as e:
        return "", "", (
            f"cannot attest the browser endpoint for session {session or ''}: {e}"
        )

    # T2: OS-attested loopback discovery via the pinned runtime dir.
    runtime_dir = env.get("BH_RUNTIME_DIR", "")
    port_file = Path(runtime_dir) / "port.json" if runtime_dir else None
    if port_file is not None and port_file.is_file():
        try:
            info = json.loads(port_file.read_text(encoding="utf-8"))
            port = int(info.get("port") or 0)
            if port:
                # Probe every plausible browser listener is heavy; the
                # supervised-Chrome fallback already guarantees an endpoint
                # via C1. A port file without an env endpoint is treated as
                # unattestable here — the caller withholds.
                return "", "t2", (
                    "local daemon port file found but no Hermes-attested CDP "
                    "endpoint; cannot attest the browser endpoint"
                )
        except Exception:
            pass
    return "", "", "no Hermes-attested CDP endpoint available for browser_exec"


# ── C-guard subprocess (Region C) ──────────────────────────────────────────

def _guard_process_env() -> dict:
    """Env for the Fetch guard subprocess: model-controlled keys stripped."""
    env = dict(os.environ)
    for key in _HERMES_CONTROLLED_ENV_KEYS:
        env.pop(key, None)
    # Ensure the guard can import tools.browser_ssrf_guard regardless of cwd.
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _spawn_ssrf_guard(endpoint: str, token: str, popen_extra: Optional[dict] = None) -> dict:
    """Spawn the CDP Fetch guard process and open its report channel.

    Returns a dict with ``proc``, ``report_sock``, ``blocked`` (threading
    Event), ``arm_failed`` (Event), ``died`` (Event), ``markers`` (list) —
    or None when the guard could not be spawned/armed (fail-closed).
    """
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        report_port = server.getsockname()[1]
    except OSError as e:
        logger.warning("ssrf guard report channel unavailable: %s", e)
        return None

    env = _guard_process_env()

    cmd = [
        sys.executable, "-m", "tools.browser_ssrf_guard",
        "--cdp-url", endpoint,
        "--report-port", str(report_port),
        "--report-token", token,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **(popen_extra or {}),
        )
    except OSError as e:
        logger.warning("ssrf guard spawn failed: %s", e)
        return None

    blocked = threading.Event()
    arm_failed = threading.Event()
    died = threading.Event()
    markers: list[str] = []

    def _accept_and_read() -> None:
        try:
            server.settimeout(15.0)
            conn, _ = server.accept()
        except OSError:
            died.set()
            return
        try:
            buf = b""
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", "replace").strip()
                    if not text:
                        continue
                    markers.append(text)
                    if text.startswith("__HERMES_BROWSER_EXEC_SSRF_BLOCK__:"):
                        blocked.set()
                    elif text.startswith("__HERMES_SSRF_GUARD_ARM_FAILED__"):
                        arm_failed.set()
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
            if proc.poll() is None:
                pass  # report channel closed but guard may still run
            died.set()

    threading.Thread(
        target=_accept_and_read, name="ssrf-guard-report", daemon=True
    ).start()

    # Wait for the arm signal (bounded). The guard emits the READY marker
    # ONLY after arm() has completed on every current target (defect-2 fix),
    # so a missing READY within the window — or an arm-failure/block marker —
    # is an arm failure, never a pass.
    ready_seen = False
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            died.set()
            break
        if arm_failed.is_set() or blocked.is_set():
            break
        if f"__HERMES_SSRF_GUARD_READY__:{token}" in markers:
            ready_seen = True
            break
        time.sleep(0.1)

    if arm_failed.is_set() or proc.poll() is not None or not ready_seen:
        # Fail closed: a not-yet-armed (or dead) guard must not let the CLI
        # spawn. Kill the guard process so it cannot linger unarmed.
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        except OSError:
            pass
        return None
    return {
        "proc": proc,
        "report_sock": server,
        "blocked": blocked,
        "arm_failed": arm_failed,
        "died": died,
        "markers": markers,
    }


def _teardown_ssrf_guard(guard: Optional[dict]) -> None:
    if not guard:
        return
    try:
        if guard.get("proc") is not None and guard["proc"].poll() is None:
            guard["proc"].terminate()
            try:
                guard["proc"].wait(timeout=5)
            except Exception:
                pass
    except Exception:
        pass
    try:
        if guard.get("report_sock") is not None:
            guard["report_sock"].close()
    except Exception:
        pass


# ── H1: guard preparation ──────────────────────────────────────────────────

def _prepare_guard(env: dict, task_id: Optional[str], session: Optional[str],
                   popen_extra: Optional[dict] = None) -> dict:
    """Arm the full guard stack for one exec window (Region E §9).

    Returns a guard context dict. ``ctx["enabled"]`` is False when the
    operator disabled the monitor (``browser.exec_network_monitor: off``).
    ``ctx["error"]`` is set (fail-closed) when any arm step failed.
    """
    cfg = _read_guard_config()
    if not cfg["enabled"]:
        return {"enabled": False, "config": cfg}

    endpoint, tier, tier_err = _resolve_endpoint_tier(env, task_id, session)
    if tier_err:
        return {"enabled": True, "config": cfg, "error": tier_err}

    token = secrets.token_hex(8)
    safe_task = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_id or "default"))[:40]
    state_dir = (
        Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        / "cache" / "browser-use" / "guard-state"
        / f"{safe_task}-{token[:8]}"
    )
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {"enabled": True, "config": cfg, "error": f"guard state dir unavailable: {e}"}

    ctx: Dict[str, Any] = {
        "enabled": True,
        "config": cfg,
        "endpoint": endpoint,
        "tier": tier,
        "token": token,
        "state_dir": str(state_dir),
        "exec_started": None,
        "monitor": None,
        "ssrf_guard": None,
        "error": None,
    }

    # Region A listener — attach PRE-SPAWN (T1).
    try:
        from tools.browser_exec_monitor import NetworkExecMonitor

        monitor = _BrowserExecGuardListener(endpoint, task_id=str(task_id or "default"))
        monitor.start(timeout=cfg["attach_timeout_s"])
        ctx["monitor"] = monitor
        if monitor.armed():
            try:
                (state_dir / "attached").write_text(token, encoding="utf-8")
            except OSError:
                pass
    except Exception as e:
        ctx["error"] = f"network listener failed to start: {e}"
        return ctx

    # Region C Fetch guard process — armed before the CLI spawns.
    guard = _spawn_ssrf_guard(endpoint, token, popen_extra=popen_extra)
    if guard is None:
        ctx["error"] = (
            "browser_exec requires a Hermes-resolvable CDP endpoint for its "
            "SSRF guard; the Fetch guard could not be armed"
        )
        return ctx
    ctx["ssrf_guard"] = guard
    return ctx


# ── H2: guard env construction ─────────────────────────────────────────────

def _guard_env(env: dict, ctx: dict) -> dict:
    """Build the CLI subprocess env: egress interposer + guard state keys.

    Raises RuntimeError when the interposer cannot be installed (the caller
    withholds — the self-test would fail anyway).
    """
    from tools.browser_exec_egress_guard import _install_egress_guard

    guard_env = dict(env)
    installed = _install_egress_guard(guard_env)
    if not installed:
        logger.warning(
            "browser_exec egress guard disabled by config — subprocess "
            "egress is not interposed; marker checks are skipped."
        )
    guard_env["HERMES_BROWSER_EXEC_GUARD_TOKEN"] = ctx["token"]
    guard_env["HERMES_BROWSER_EXEC_GUARD_STATE_DIR"] = ctx["state_dir"]
    # Policy snapshot keyed by the SAME per-spawn nonce the interposer echoes
    # in :installed: (the egress installer already set it when armed).
    if "HERMES_BROWSER_EXEC_EGRESS_POLICY" not in guard_env:
        guard_env["HERMES_BROWSER_EXEC_EGRESS_POLICY"] = json.dumps(
            _egress_policy_snapshot(
                guard_env.get("HERMES_BROWSER_EXEC_EGRESS_GUARD_NONCE", "")
            ),
            sort_keys=True,
        )
    # BH_RUNTIME_DIR pinning (Region E §4.4): the host always knows where
    # the harness daemon's port file lives.
    if not guard_env.get("BH_RUNTIME_DIR"):
        try:
            from hermes_constants import get_hermes_home

            runtime = str(Path(get_hermes_home()) / "cache" / "browser-use" / "runtime")
            Path(runtime).mkdir(parents=True, exist_ok=True)
            guard_env["BH_RUNTIME_DIR"] = runtime
        except OSError:
            pass
    return guard_env


def _egress_policy_snapshot(nonce: str) -> dict:
    from tools.browser_exec_egress_guard import _policy_snapshot

    return _policy_snapshot(nonce=nonce)


# ── H3: policy-parameterized self-test ─────────────────────────────────────

_SELF_TEST_PROBE = r"""
import os, socket as _socket, sys

def _ok(msg):
    print(msg, flush=True)

def _is_egress_blocked(exc):
    # The interposer's EgressBlocked is an OSError subclass; distinguish it
    # from an OS-level failure by the message.
    return "egress guard blocked" in str(exc) or "egress guard: " in str(exc)

results = []
# (1) blocked-hostname floor fires BEFORE DNS (works on any host).
try:
    _socket.getaddrinfo("metadata.google.internal", None, _socket.AF_UNSPEC, _socket.SOCK_STREAM)
    results.append(("floor-hostname", False))
except OSError as e:
    results.append(("floor-hostname", _is_egress_blocked(e)))
# (2) IMDS literal connect → immediate block.
try:
    _socket.create_connection(("169.254.169.254", 1), timeout=2)
    results.append(("imds-literal", False))
except OSError as e:
    results.append(("imds-literal", _is_egress_blocked(e)))
# (3) private-external — required unless allow_private opt-out.
if os.environ.get("HERMES_BROWSER_EXEC_EGRESS_ALLOW_PRIVATE", "") == "1":
    # Must NOT be blocked at the policy layer (proceeds to an OS result).
    try:
        _socket.create_connection(("10.255.255.1", 1), timeout=1)
        results.append(("private-external", True))
    except OSError as e:
        results.append(("private-external", not _is_egress_blocked(e)))
else:
    try:
        _socket.create_connection(("10.255.255.1", 1), timeout=2)
        results.append(("private-external", False))
    except OSError as e:
        results.append(("private-external", _is_egress_blocked(e)))
# (4) loopback positive control: OS result, never EgressBlocked.
try:
    _socket.create_connection(("127.0.0.1", 1), timeout=1)
    results.append(("loopback", True))  # connected (unlikely) — policy passed
except OSError as e:
    results.append(("loopback", not _is_egress_blocked(e)))

for name, passed in results:
    print("SELFTEST:%s:%s" % (name, "PASS" if passed else "FAIL"), flush=True)
if all(passed for _, passed in results):
    print("__HERMES_BROWSER_EXEC_GUARD_OK__", flush=True)
    sys.exit(0)
sys.exit(1)
"""


def _guard_self_test(ctx: dict, guard_env: dict) -> Optional[str]:
    """Run the interposer self-test in a fresh, workspace-stripped probe.

    Returns None on success, or a withhold reason string. Called only when
    the guard is enabled; a failure withholds BEFORE the CLI spawns.
    """
    probe_env = dict(guard_env)
    for key in _HERMES_CONTROLLED_ENV_KEYS:
        probe_env.pop(key, None)
    probe_env["HERMES_BROWSER_EXEC_EGRESS_ALLOW_PRIVATE"] = (
        "1" if ctx["config"].get("allow_private") else "0"
    )
    try:
        p = subprocess.run(
            [sys.executable, "-c", _SELF_TEST_PROBE],
            capture_output=True, text=True, timeout=30, env=probe_env,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"egress guard self-test could not run: {e}"
    if p.returncode != 0 or SELF_TEST_OK_MARKER not in (p.stdout or ""):
        fails = [ln for ln in (p.stdout or "").splitlines() if "SELFTEST:" in ln and "FAIL" in ln]
        return (
            "browser_exec egress guard self-test failed (interposer did not "
            "install — the CLI launcher may ignore PYTHONPATH)"
            + (f": {', '.join(fails)}" if fails else "")
        )
    return None


# ── Advisory in-CLI preamble (H5/H7/H8) ────────────────────────────────────

_GUARD_PREAMBLE = r'''
# Hermes browser_exec guard preamble — ADVISORY. Authoritative enforcement
# lives in the host listener, the socket interposer, the trusted probe, and
# the host verdict; nothing in this preamble is trusted. A poisoned
# workspace agent_helpers.py may already have run (import-time seam), so
# this preamble self-isolates and touches no helper names.
import os as __hermes_os, time as __hermes_time

def __hermes_guard_preamble():
    def _flush(msg):
        try:
            print(msg, flush=True)
        except Exception:
            pass
    _announce = __hermes_os.environ.get("BU_CDP_WS", "") or __hermes_os.environ.get("BU_CDP_URL", "")
    _flush("__HERMES_BROWSER_EXEC_ANNOUNCE__:" + str(_announce))
    _session = False
    try:
        _runtime = __hermes_os.environ.get("BH_RUNTIME_DIR", "")
        if _runtime:
            import json as _json
            _port_file = __hermes_os.path.join(_runtime, "port.json")
            if __hermes_os.path.exists(_port_file):
                _info = _json.load(open(_port_file, encoding="utf-8"))
                _port = int(_info.get("port") or 0)
                _token = str(_info.get("token") or "")
                if _port:
                    import socket as _socket
                    _s = _socket.create_connection(("127.0.0.1", _port), timeout=2)
                    try:
                        _payload = _json.dumps({"meta": "ping", "token": _token}) + "\n"
                        _s.sendall(_payload.encode("utf-8"))
                        _s.settimeout(2)
                        _resp = _s.recv(4096).decode("utf-8", "replace")
                        _session = ("pong" in _resp) or (_resp.strip() != "")
                    finally:
                        _s.close()
    except Exception:
        _session = False
    if _session:
        _flush("__HERMES_BROWSER_EXEC_ARMED__:full")
    else:
        _flush("__HERMES_BROWSER_EXEC_ARMED__:no-session")
    _state_dir = __hermes_os.environ.get("HERMES_BROWSER_EXEC_GUARD_STATE_DIR", "")
    if _state_dir:
        _deadline = __hermes_time.monotonic() + 20.0
        while __hermes_time.monotonic() < _deadline:
            if __hermes_os.path.exists(__hermes_os.path.join(_state_dir, "attached")):
                break
            __hermes_time.sleep(0.2)

__hermes_guard_preamble()
'''

_MARKER_ARMED_RE = re.compile(r"^" + re.escape(ARMED_MARKER) + r"(full|no-session)\s*$")
_MARKER_ANNOUNCE_RE = re.compile(r"^" + re.escape(ANNOUNCE_MARKER) + r"(.*)$")


def _parse_preamble_markers(stdout: str) -> dict:
    """Extract the advisory preamble markers from CLI stdout."""
    armed = None
    announce = None
    for line in (stdout or "").splitlines():
        m = _MARKER_ARMED_RE.match(line.strip())
        if m:
            armed = m.group(1)
            continue
        m = _MARKER_ANNOUNCE_RE.match(line.strip())
        if m:
            announce = m.group(1).strip()
    return {"armed": armed, "announce": announce}


# ── Guarded CLI runner (replaces subprocess.run in browser_exec) ───────────

def _run_guarded_cli(cmd, env, code, popen_extra, timeout, guard_ctx, preamble=True) -> dict:
    """Run the CLI subprocess while draining the C-guard report channel.

    Returns ``{returncode, stdout, stderr, timed_out, guard_blocked,
    guard_died, markers}``. On a C-guard block marker the CLI is killed
    immediately; on timeout the CLI is killed and ``timed_out`` is set.
    """
    import io

    from tools.browser_exec_egress_guard import _parse_guard_markers, _strip_guard_markers

    popen_kwargs = dict(popen_extra or {})
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            **popen_kwargs,
        )
    except OSError as e:
        return {"returncode": None, "stdout": "", "stderr": "", "timed_out": False,
                "guard_blocked": False, "guard_died": False, "launch_error": str(e),
                "markers": {"armed": None, "announce": None}}

    input_data = ((_GUARD_PREAMBLE + "\n") if preamble else "") + code
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    lock = threading.Lock()

    def _reader(stream, sink):
        try:
            for line in stream:
                with lock:
                    sink.append(line)
        except Exception:
            pass

    t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks), daemon=True)
    t_out.start()
    t_err.start()

    try:
        proc.stdin.write(input_data)
        proc.stdin.close()
    except OSError:
        pass

    guard = guard_ctx.get("ssrf_guard") if guard_ctx else None
    deadline = time.monotonic() + timeout
    timed_out = False
    guard_blocked = False
    while True:
        if guard is not None and guard.get("blocked") and guard["blocked"].is_set():
            guard_blocked = True
            try:
                proc.kill()
            except OSError:
                pass
            break
        rc = proc.poll()
        if rc is not None:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            try:
                proc.kill()
            except OSError:
                pass
            break
        time.sleep(0.05)

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait(timeout=5)

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    markers = _parse_preamble_markers(stdout)

    # Region B marker parse — fail closed on block/tamper/missing-nonce.
    nonce = env.get("HERMES_BROWSER_EXEC_EGRESS_GUARD_NONCE", "")
    guard_enabled = env.get("HERMES_BROWSER_EXEC_EGRESS_GUARD", "") == "1"
    b_reason = None
    if guard_enabled:
        b_reason = _parse_guard_markers(stderr, nonce)
    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "guard_blocked": guard_blocked,
        "guard_died": bool(guard is not None and guard.get("died") and guard["died"].is_set()),
        "markers": markers,
        "egress_reason": b_reason,
    }


def _strip_guard_markers(text: str) -> str:
    """Strip guard report lines (advisory markers) from returned output."""
    if not text:
        return text
    kept = [ln for ln in text.splitlines() if not _GUARD_REPORT_LINE_RE.match(ln.strip())]
    stripped = "\n".join(kept)
    return stripped + "\n" if text.endswith("\n") and stripped else stripped


# ── H6: browser-level listener ─────────────────────────────────────────────

class _BrowserExecGuardListener:
    """Browser-level CDP listener (Region E §3): auto-attach + per-session
    Network/Page, per-target last-known URL map, WS-drop attestation gap.

    Wraps the Region A ``NetworkExecMonitor`` (the actual CDP machinery).
    """

    def __init__(self, cdp_url: str, *, task_id: str, tier: str = "t1") -> None:
        from tools.browser_exec_monitor import NetworkExecMonitor

        self._monitor = NetworkExecMonitor(cdp_url, task_id=task_id)
        self.tier = tier
        self.attach_error: Optional[str] = None

    # Delegated monitor API.
    def start(self, timeout: float = 15.0) -> None:
        try:
            self._monitor.start(timeout=timeout)
        except Exception as e:  # pragma: no cover — defensive
            self.attach_error = str(e)

    def stop(self, timeout: float = 5.0) -> None:
        self._monitor.stop(timeout=timeout)

    def attach_failed(self) -> bool:
        return self._monitor.attach_failed() or self.attach_error is not None

    def armed(self) -> bool:
        return self._monitor.armed()

    def saw_activity(self, exec_started: float) -> bool:
        return self._monitor.saw_activity(exec_started)

    def mark_probe_success(self) -> None:
        self._monitor.mark_probe_success()

    def violation(self) -> Optional[dict]:
        return self._monitor.violation()

    def reset(self) -> None:
        self._monitor.reset()

    def last_known_url(self) -> str:
        return self._monitor.last_known_url()

    def event_count(self) -> int:
        return self._monitor.event_count()

    def dropped(self) -> bool:
        return self._monitor.dropped()

    def request_log(self, limit: int = 200) -> list:
        return self._monitor.request_log(limit=limit)


# ── H9: end-state verdict (the §10 agreement truth table) ──────────────────

def _landed_is_blocked(landed: str) -> Optional[str]:
    """Region D verdict on the landed URL; returns the reason when blocked."""
    try:
        from tools.browser_tool import _resolve_and_check_url

        v = _resolve_and_check_url(landed)
        if not v.ok:
            return v.reason
    except Exception:
        return "error:internal"
    return None


def _browser_observed(ctx: dict, landed: Optional[str]) -> bool:
    """Cross-check: is a live browser known to exist/be active?"""
    monitor = ctx.get("monitor")
    if monitor is not None and (monitor.saw_activity(ctx.get("exec_started") or 0.0)
                                or monitor.event_count() > 0):
        return True
    if landed:
        return True
    return False


def _guard_endstate_verdict(ctx: dict, landed: Optional[str], run: dict) -> dict:
    """Apply the Region E §10 truth table; returns a verdict dict.

    ``{"verdict": "return"|"withhold", "reason": str, "note": str|None}``.
    """
    cfg = ctx.get("config") or {}
    monitor = ctx.get("monitor")
    exec_started = ctx.get("exec_started") or 0.0
    fail_open = bool(cfg.get("fail_open"))

    # Violations (V): monitor latch, C-guard block, B markers.
    violation = monitor.violation() if monitor is not None else None
    if run.get("egress_reason"):
        return {"verdict": "withhold", "reason": run["egress_reason"], "note": "b"}
    if run.get("guard_blocked"):
        urls = [m for m in (ctx.get("ssrf_guard") or {}).get("markers", [])
                if m.startswith("__HERMES_BROWSER_EXEC_SSRF_BLOCK__:")]
        url = urls[0].split(":", 1)[1] if urls else "<unknown>"
        return {
            "verdict": "withhold",
            "reason": (
                f"Blocked: during execution the browser requested a URL the "
                f"navigation policy rejects ({url}); all output was withheld."
            ),
            "note": "c",
        }
    if violation is not None:
        url = str(violation.get("url") or "")
        policy = str(violation.get("policy") or "private")
        return {
            "verdict": "withhold",
            "reason": (
                f"Blocked: during execution the browser requested a URL the "
                f"navigation policy rejects ({url} — {policy}); all output "
                f"was withheld. Intermediate requests are monitored via CDP "
                f"Network events ({violation.get('event')}); the final "
                f"landing check alone cannot detect this."
            ),
            "note": "a",
        }
    if run.get("guard_died"):
        return {"verdict": "withhold",
                "reason": "the browser_exec SSRF guard process died mid-exec; output withheld",
                "note": "c"}

    # Precondition C — coverage verified: A attached to a host-attested
    # endpoint with no attach failure/drop mid-exec. (Absence of exec-window
    # activity is NOT a coverage failure — it is handled by the P/L/M rows
    # via the browser_observed cross-check, rows 6/9.)
    if monitor is not None:
        coverage_ok = (
            not monitor.attach_failed()
            and monitor.armed()
            and not monitor.dropped()
        )
        if not coverage_ok:
            if fail_open:
                logger.warning(
                    "browser network monitoring could not be verified "
                    "attached and active (fail_open); continuing",
                )
                return {"verdict": "return", "reason": "", "note": "unverified"}
            return {
                "verdict": "withhold",
                "reason": (
                    "browser network monitoring could not be verified "
                    "attached and active; output withheld"
                ),
                "note": "coverage",
            }

    # P / L / M rows.
    markers = run.get("markers") or {}
    armed_marker = markers.get("armed")
    announce = markers.get("announce")
    endpoint = ctx.get("endpoint") or ""
    if announce and endpoint and announce != endpoint:
        return {
            "verdict": "withhold",
            "reason": (
                "browser_exec guard announcement does not match the "
                "Hermes-attested endpoint; output withheld"
            ),
            "note": "announce",
        }

    p_reason = _landed_is_blocked(landed) if landed else None
    if p_reason:
        if p_reason in ("blocked:metadata-host", "blocked:metadata-ip",
                        "blocked:link-local", "blocked:ipv4-compatible"):
            return {"verdict": "withhold",
                    "reason": (
                        "Blocked: URL targets a cloud metadata endpoint — the "
                        "browser ended on this address after the code ran, so "
                        "the page output was withheld."),
                    "note": "p"}
        return {"verdict": "withhold",
                "reason": (
                    "Blocked: URL targets a private or internal address — the "
                    "browser ended on this address after the code ran, so the "
                    "page output was withheld."),
                "note": "p"}

    l_url = monitor.last_known_url() if monitor is not None else ""
    l_reason = _landed_is_blocked(l_url) if l_url else None
    if l_reason:
        return {
            "verdict": "withhold",
            "reason": (
                "Blocked: during execution the browser was observed on a URL "
                f"the navigation policy rejects ({l_url}); all output was "
                "withheld."
            ),
            "note": "l",
        }

    browser_live = _browser_observed(ctx, landed)
    if armed_marker == "no-session":
        if not browser_live:
            return {"verdict": "return", "reason": "", "note": "no-session"}
        return {
            "verdict": "withhold",
            "reason": (
                "browser_exec armed marker claims no session, but a browser "
                "was observed; output withheld"
            ),
            "note": "m",
        }
    if armed_marker == "full":
        # Rows 5/7: the 'full' marker is model-writable CLI stdout
        # (advisory). It can release output ONLY when the host-side monitor
        # verifies the claim — armed AND exec-window activity observed (its
        # own traffic or the trusted landing probe) — AND a trusted landing
        # exists (P). Without a trusted landing (row 7: P=None + M='full' +
        # browser observed), a forged stdout marker cannot flip a withhold
        # into a return.
        monitor_verified = (
            monitor is not None
            and monitor.armed()
            and monitor.saw_activity(exec_started)
        )
        if landed and monitor_verified:
            return {"verdict": "return", "reason": "", "note": "full"}
        return {
            "verdict": "withhold",
            "reason": (
                "browser_exec guard reported a full armed state, but no "
                "trusted landing was verified against the host-side "
                "monitor; output withheld"
            ),
            "note": "m",
        }
    # (nothing) — C-install/tri-state absent.
    if browser_live:
        return {
            "verdict": "withhold",
            "reason": (
                "browser_exec guard did not report a verified armed state "
                "while a browser was observed; output withheld"
            ),
            "note": "m",
        }
    return {"verdict": "return", "reason": "", "note": "no-browser"}
