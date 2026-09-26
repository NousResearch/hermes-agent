"""Browser Use harness daemon lifecycle: owner claim, idle/orphan reaping, exit teardown.

``browser_harness`` starts its daemon with ``start_new_session=True`` and expects it to
OUTLIVE the CLI invocation that spawned it — that detachment is what makes a session
persist across ``browser_exec`` calls. So the daemon is reparented to init on purpose;
``ppid=1`` is the design, not the leak, and a parent-death signal (pipe EOF, process
group, ``PR_SET_PDEATHSIG``) would kill the browser after every single tool call.

The leak is that nothing ever reclaimed those daemons. ``browser_tool_lifecycle`` owns
the same three mechanisms for the ``agent-browser`` lane — an owner-pid claim, an idle
grace, and teardown at exit — and the harness lane had none of them, so every distinct
``BU_NAME`` left a daemon holding a CDP connection until reboot.

Stopping is IPC-first: ``meta: shutdown`` is the only path that lets the daemon stop a
billable Browser Use cloud browser before it dies, and answering ``{"pong": true}`` on
its own socket is itself proof of identity. A signal is the fallback for a daemon whose
socket is already gone, and is sent only to a PID that is ours, still alive, and running
``browser_harness.daemon``. Chrome is NOT a child of this daemon (Hermes always exports a
``BU_CDP_*`` endpoint, so the harness never launches its own browser) — the agent-browser
lane reaps the browser, this one reaps the daemon holding the socket to it.
"""

import contextlib
import os
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from tools.browser_tool_origin import origin as _bt

# Runtime dirs we have claimed a session in, so the janitor sweeps the same dirs the CLI
# actually used even when an operator points a profile at a custom ``BH_RUNTIME_DIR``.
_claimed_runtime_dirs: Set[str] = set()
_claim_lock = threading.Lock()

# A daemon answering ping gets this long to run its cloud-stop and exit before the signal
# fallback; a wedged one must not hold the janitor thread.
_SHUTDOWN_IPC_TIMEOUT_S = 5.0
_SHUTDOWN_EXIT_WAIT_S = 10.0


def _truthy_env(env: Dict[str, str], key: str) -> bool:
    return str(env.get(key, "")).strip() == "1"


def harness_runtime_dir(env: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """Resolve the dir holding ``bu-*.sock``/``.pid``, mirroring ``browser_harness.paths``.

    Precedence (``_ipc.py``): ``BH_RUNTIME_DIR`` → ``BH_TMP_DIR`` → ``BH_HOME`` /
    ``BROWSER_HARNESS_HOME`` / ``XDG_CONFIG_HOME``-derived home + ``runtime``. Resolved from
    the env dict handed to the CLI so a profile-scoped override lands in the same dir the
    subprocess will use.
    """
    src = os.environ if env is None else env
    for key in ("BH_RUNTIME_DIR", "BH_TMP_DIR"):
        raw = src.get(key)
        if raw:
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                return Path(raw).expanduser().resolve()
            return None

    home_raw = src.get("BH_HOME") or src.get("BROWSER_HARNESS_HOME")
    try:
        if home_raw:
            return Path(home_raw).expanduser().resolve() / "runtime"
        xdg = src.get("XDG_CONFIG_HOME")
        if xdg:
            return (Path(xdg).expanduser() / "browser-harness").resolve() / "runtime"
        return (Path.home() / ".config" / "browser-harness").resolve() / "runtime"
    except (OSError, RuntimeError, ValueError) as exc:
        _bt.logger.debug("Could not resolve browser-harness runtime dir: %s", exc)
        return None


def _runtime_stem(env: Dict[str, str], session: str) -> str:
    """``bu`` when a caller-supplied runtime dir isolates one daemon, else ``bu-<name>``
    (``browser_harness._ipc._runtime_stem``)."""
    isolated = bool(env.get("BH_RUNTIME_DIR") or env.get("BH_TMP_DIR"))
    shared = _truthy_env(env, "BH_RUNTIME_DIR_SHARED") or _truthy_env(env, "BH_TMP_DIR_SHARED")
    return "bu" if isolated and not shared else f"bu-{session or 'default'}"


def _owner_path(runtime_dir: Path, stem: str) -> Path:
    return runtime_dir / f"{stem}.owner_pid"


def claim_session(env: Dict[str, str], session: str) -> None:
    """Record this process as the owner of ``session``'s daemon, before the CLI spawns it.

    Rewritten on EVERY call, so the file's mtime doubles as a restart-proof activity marker
    (the socket's mtime never moves, and the daemon log is only appended on notable events).
    Best-effort: an unwritable runtime dir just leaves the daemon on the legacy
    idle-only path rather than failing the tool call.
    """
    runtime_dir = harness_runtime_dir(env)
    if runtime_dir is None:
        return
    stem = _runtime_stem(env, session)
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        _owner_path(runtime_dir, stem).write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        _bt.logger.debug("Could not claim harness session %s: %s", stem, exc)
        return
    with _claim_lock:
        _claimed_runtime_dirs.add(str(runtime_dir))


def _sweep_dirs() -> List[Path]:
    """Runtime dirs to scan: every dir we claimed in, plus this env's default."""
    with _claim_lock:
        dirs = {d for d in _claimed_runtime_dirs}
    default = harness_runtime_dir()
    if default is not None:
        dirs.add(str(default))
    return [Path(d) for d in sorted(dirs)]


def _read_pid(path: Path) -> Optional[int]:
    """Integer PID from ``path``; None when missing, corrupt, or out of range.

    The upper bound matches ``browser_harness._ipc.identify``: a value outside signed
    32-bit ``pid_t`` makes ``os.kill`` raise ``OverflowError``, and 0/negatives address a
    process GROUP rather than one process.
    """
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if 0 < pid < (1 << 31) else None


def _pid_alive(pid: int) -> bool:
    from gateway.status import _pid_exists
    return bool(_pid_exists(pid))


def _idle_seconds(paths: Iterable[Path]) -> Optional[float]:
    """Seconds since the most recent of ``paths`` was written; None when none exist."""
    newest = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        newest = mtime if newest is None else max(newest, mtime)
    return None if newest is None else max(0.0, time.time() - newest)


def _ipc_shutdown(runtime_dir: Path, stem: str) -> Tuple[bool, Optional[int]]:
    """Ask the daemon to stop over its own socket. Returns ``(acknowledged, daemon_pid)``.

    Answering ``{"pong": true, "pid": N}`` on a 0600 socket in a 0700 dir IS the identity
    proof, so no separate verification is needed on this path. This is also the ONLY stop
    that runs the daemon's cloud-browser cleanup — a signal here would leave a billable
    browser running.
    """
    sock = runtime_dir / f"{stem}.sock"
    if not sock.exists():
        return False, None

    import json
    import socket as _socket

    def _request(payload: dict) -> Optional[dict]:
        try:
            conn = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        except (AttributeError, OSError):
            return None  # no AF_UNIX (Windows): caller falls back to the signal path
        try:
            conn.settimeout(_SHUTDOWN_IPC_TIMEOUT_S)
            conn.connect(str(sock))
            conn.sendall((json.dumps(payload) + "\n").encode())
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(1 << 16)
                if not chunk:
                    break
                data += chunk
            parsed = json.loads(data or b"{}")
            return parsed if isinstance(parsed, dict) else None
        except (OSError, ValueError):
            return None
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    pong = _request({"meta": "ping"})
    if not pong or pong.get("pong") is not True:
        return False, None
    reported = pong.get("pid")
    # ``type(...) is int`` rejects bool: isinstance(True, int) is True, and {"pid": True}
    # would be read as PID 1.
    daemon_pid = reported if type(reported) is int and 0 < reported < (1 << 31) else None

    resp = _request({"meta": "shutdown"})
    return bool(resp and resp.get("ok") is True), daemon_pid


def _is_harness_daemon_cmdline(cmdline: Optional[List[str]]) -> bool:
    """True only for a genuine ``python -m browser_harness.daemon`` invocation.

    Never a substring test on the joined command line — that also matches a process
    whose argv merely MENTIONS the text (a ``-c`` script literal, a config value, a
    log path) without actually running the module (the exact bug class root
    ``AGENTS.md`` calls out: "Never infer process identity from argv substrings").
    Matches the ``-m <module>`` token pair exactly, or a direct script invocation
    whose last argv element is a path ending in ``browser_harness/daemon.py``.
    """
    if not cmdline:
        return False
    tokens = list(cmdline)
    for i, tok in enumerate(tokens[:-1]):
        if tok == "-m" and tokens[i + 1] == "browser_harness.daemon":
            return True
    last = tokens[-1].replace("\\", "/")
    return last == "browser_harness/daemon.py" or last.endswith("/browser_harness/daemon.py")


def _verified_daemon_pid(pid: int) -> bool:
    """Whether ``pid`` is a live ``browser_harness.daemon`` owned by this user (fail-closed)."""
    try:
        import psutil
    except ImportError:  # psutil is a hard dep; defensive only
        return False
    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline() or []
        same_user = proc.uids().real == os.getuid() if hasattr(os, "getuid") else True
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError) as exc:
        _bt.logger.warning("Refusing to signal harness daemon PID %d: identity unreadable (%s)", pid, exc)
        return False
    if not same_user:
        return False
    if not _is_harness_daemon_cmdline(cmdline):
        _bt.logger.warning("Refusing to signal PID %d: not a browser_harness daemon", pid)
        return False
    return True


def _signal_daemon(pid: int) -> bool:
    """SIGTERM a verified daemon, escalating to SIGKILL. True when it is gone.

    No tree kill: the harness daemon never parents the browser under Hermes (a
    ``BU_CDP_*`` endpoint is always exported), so its descendants are not ours to take
    down — the agent-browser lane owns Chrome.
    """
    if not _verified_daemon_pid(pid):
        return False

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError) as exc:
            _bt.logger.warning("Could not signal harness daemon PID %d: %s", pid, exc)
            return False

        deadline = time.monotonic() + _SHUTDOWN_EXIT_WAIT_S
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                return True
            time.sleep(0.2)
        if sig is signal.SIGTERM:
            _bt.logger.warning("Harness daemon PID %d ignored SIGTERM; escalating to SIGKILL", pid)
    return not _pid_alive(pid)


def _clear_endpoint(runtime_dir: Path, stem: str) -> None:
    for suffix in (".sock", ".pid", ".port", ".owner_pid"):
        with contextlib.suppress(OSError):
            (runtime_dir / f"{stem}{suffix}").unlink()


def _stop_daemon(runtime_dir: Path, stem: str, pid: Optional[int]) -> bool:
    """Stop one daemon IPC-first, signal as fallback, then clear its endpoint files."""
    acknowledged, reported_pid = _ipc_shutdown(runtime_dir, stem)
    target = reported_pid or pid

    if acknowledged and target is not None:
        deadline = time.monotonic() + _SHUTDOWN_EXIT_WAIT_S
        while time.monotonic() < deadline and _pid_alive(target):
            time.sleep(0.2)

    stopped = acknowledged and (target is None or not _pid_alive(target))
    if not stopped and target is not None and _pid_alive(target):
        stopped = _signal_daemon(target)

    if stopped or target is None or not _pid_alive(target):
        _clear_endpoint(runtime_dir, stem)
        return stopped
    return False


def _sessions_in(runtime_dir: Path) -> List[str]:
    """Stems of every daemon with a pid file in ``runtime_dir``."""
    try:
        return sorted(p.name[: -len(".pid")] for p in runtime_dir.glob("bu*.pid"))
    except OSError:
        return []


def _should_reap(runtime_dir: Path, stem: str, *, owner_pid: Optional[int],
                 only_owner: Optional[int]) -> Tuple[bool, str]:
    """Reap decision for one daemon: ``(reap, reason)``.

    ``only_owner`` restricts the sweep to daemons claimed by that PID (exit teardown).
    Otherwise: a dead owner means nobody will ever reclaim the daemon — reap now; a live
    owner (this process or another Hermes sharing the runtime dir) keeps its daemon until
    it goes idle past the grace, which is the same escape hatch the agent-browser lane
    uses so a leaked claim cannot make a daemon immortal; a missing/corrupt claim (a
    legacy daemon, or a failed claim write) is idle-gated rather than trusted.
    """
    if only_owner is not None:
        return (owner_pid == only_owner, "owner exiting")

    if owner_pid is not None and not _pid_alive(owner_pid):
        return True, f"owner PID {owner_pid} is gone"

    idle = _idle_seconds([
        _owner_path(runtime_dir, stem),
        runtime_dir / f"{stem}.pid",
        runtime_dir / f"{stem}.sock",
    ])
    grace = _bt.BROWSER_ORPHAN_GRACE_SECONDS
    if idle is None or idle < grace:
        return False, ""  # unknown age or within grace — fail safe
    if owner_pid is None:
        return True, f"unclaimed and idle for {int(idle)}s (grace {grace}s)"
    return True, f"owner PID {owner_pid} is live but the session is idle for {int(idle)}s (grace {grace}s)"


def _sweep(only_owner: Optional[int]) -> int:
    reaped = 0
    for runtime_dir in _sweep_dirs():
        for stem in _sessions_in(runtime_dir):
            pid = _read_pid(runtime_dir / f"{stem}.pid")
            owner_pid = _read_pid(_owner_path(runtime_dir, stem))

            if pid is None or not _pid_alive(pid):
                # Daemon already gone; drop the endpoint so the next call starts clean.
                _clear_endpoint(runtime_dir, stem)
                continue

            reap, reason = _should_reap(runtime_dir, stem, owner_pid=owner_pid, only_owner=only_owner)
            if not reap:
                continue
            try:
                if _stop_daemon(runtime_dir, stem, pid):
                    _bt.logger.info("Stopped browser-harness daemon PID %d (%s): %s", pid, stem, reason)
                    reaped += 1
                else:
                    _bt.logger.warning("Browser-harness daemon PID %d (%s) did not stop; will retry", pid, stem)
            except Exception as exc:  # a stuck daemon must never abort the sweep
                _bt.logger.warning("Error stopping browser-harness daemon %s: %s", stem, exc)
    return reaped


def _sweep_unindexed() -> int:
    """Stop harness daemons that no longer have a pid file anywhere we sweep.

    Observed live: a daemon kept running with its ``bu-<name>.sock``/``.pid`` already
    unlinked (a ``--reload``/restart path removes the endpoint, and a daemon that outlives
    that removal is unreachable over IPC and invisible to every file-based scan). Such a
    daemon can never be reused — nothing can find its socket — so it is pure leak, and only
    the process table can see it.

    Fail-closed: only same-user ``browser_harness.daemon`` processes, only when older than
    the grace (so a daemon mid-startup, before it publishes its pid file, is never killed),
    and never one whose PID is still referenced by a pid file we know about.
    """
    try:
        import psutil
    except ImportError:  # psutil is a hard dep; defensive only
        return 0

    indexed = set()
    for runtime_dir in _sweep_dirs():
        for stem in _sessions_in(runtime_dir):
            pid = _read_pid(runtime_dir / f"{stem}.pid")
            if pid is not None:
                indexed.add(pid)

    uid = os.getuid() if hasattr(os, "getuid") else None
    grace = _bt.BROWSER_ORPHAN_GRACE_SECONDS
    reaped = 0
    for proc in psutil.process_iter(["pid", "cmdline", "create_time", "uids"]):
        try:
            info = proc.info
            pid = info["pid"]
            if pid in indexed or pid == os.getpid():
                continue
            if not _is_harness_daemon_cmdline(info.get("cmdline")):
                continue
            if uid is not None and info.get("uids") and info["uids"].real != uid:
                continue
            if time.time() - (info.get("create_time") or 0) < grace:
                continue  # may still be publishing its pid file
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError):
            continue

        if _signal_daemon(pid):
            _bt.logger.info("Stopped browser-harness daemon PID %d: running with no endpoint "
                            "files (unreachable, cannot be reused)", pid)
            reaped += 1
    return reaped


def reap_orphaned_harness_daemons() -> None:
    """Stop harness daemons whose owning hermes process is gone, or that went idle past the
    grace. Safe from any context; called on the janitor's orphan-reap cycle."""
    reaped = _sweep(only_owner=None) + _sweep_unindexed()
    if reaped:
        _bt.logger.info("Reaped %d orphaned browser-harness daemon(s)", reaped)


def shutdown_owned_harness_daemons() -> None:
    """atexit: stop the daemons THIS process claimed.

    Only our own claims — several hermes processes (gateway multiplex, CLI, kanban worker)
    share one runtime dir, and the last caller of a name owns it. A process that lost a
    claim leaves the daemon to its new owner; ``ensure_daemon`` is idempotent, so a caller
    whose daemon was stopped under it just gets a fresh one on its next call.
    """
    _sweep(only_owner=os.getpid())
