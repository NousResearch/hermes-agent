"""Process-tree termination and post-kill survivor verification for ProcessRegistry."""

import logging
import math
import os
import signal
import subprocess
import time
from contextlib import suppress
from typing import TYPE_CHECKING, List, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.process_registry_env_log import group_liveness_shell

if TYPE_CHECKING:
    from tools.process_registry import ProcessSession

logger = logging.getLogger("tools.process_registry")

ENV_GONE_MARKER = "__HERMES_PROCESS_TERMINATED__"
ENV_SURVIVORS_MARKER = "__HERMES_PROCESS_SURVIVORS__"
_ENV_TICK = 0.05
# Exec round trip to a remote sandbox plus two member scans, on top of the script's signal waits.
_ENV_KILL_TRANSPORT_SECONDS = 15.0


def env_group_kill_command(pgid: int, grace: float) -> str:
    """POSIX shell command that stops sandbox process group *pgid* and proves it is gone.

    SIGTERM, then after *grace* seconds SIGKILL (no SIGKILL when *grace* is 0,
    matching ``terminal.daemon_term_grace_seconds``). The signal waits poll the cheap
    ``_hq`` (no process-table scan, which takes seconds on a host with ~10k processes),
    so the scan cost never stretches the grace past the exec timeout. Then at most two
    scans settle it: ``ENV_GONE_MARKER`` once ``group_liveness_shell`` finds no live
    member (zombies do not count), else ``ENV_SURVIVORS_MARKER`` and the live member
    PIDs (``?`` when the sandbox cannot list them).

    Fractional ``sleep`` is not POSIX: where it fails, each wait tick sleeps a whole
    second and counts as that many ticks, so the grace still holds."""
    if not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 1:
        # kill -- -1 signals every process the sandbox user can reach.
        raise ValueError(f"not a sandbox process group id: {pgid!r}")
    term_ticks = max(1, math.ceil(grace / _ENV_TICK)) if grace else round(1 / _ENV_TICK)
    whole_second = round(1 / _ENV_TICK)
    gone_exit = f"{{ echo {ENV_GONE_MARKER}; exit 0; }}"
    escalate = f'kill -KILL "-$_hg" 2>/dev/null; _hw 20 && {gone_exit}; ' if grace else ""
    return (
        group_liveness_shell(str(pgid))
        + '_hz() { if [ "$_hs" = 1 ]; then sleep "$1"; else sleep 1; fi; }; '
        f'_hw() {{ _i=0; while [ "$_i" -lt "$1" ]; do _hq || return 0; _hz {_ENV_TICK}; _i=$((_i + _hs)); done; return 1; }}; '
        # The fractional-sleep check is the first tick of the grace, after the signal.
        f'kill -TERM "-$_hg" 2>/dev/null; _hs=1; sleep {_ENV_TICK} 2>/dev/null || _hs={whole_second}; '
        f'_hw {term_ticks} && {gone_exit}; '
        f"{escalate}"
        # Zombies keep _hq "alive"; the member scan settles it (once more after a beat for
        # members the kernel is still tearing down).
        f'_hl || {gone_exit}; _hz 0.2; _m=$(_hm); case "$_m" in "") {gone_exit};; "!") _hp || {gone_exit};; "?") _hq || {gone_exit};; esac; '
        f'echo {ENV_SURVIVORS_MARKER} $_m; exit 1'
    )


def parse_env_group_kill(result: dict, pgid: int) -> "list[int] | None":
    """Survivors of an ``env_group_kill_command`` run on group *pgid*: ``[]`` when the
    group is proven gone, the listed PIDs when it outlived the signals (``[pgid]`` when
    the sandbox could not list them), ``None`` when the reply proves neither (transport
    failure, no bash)."""
    lines = str(result.get("output", "")).splitlines()
    if result.get("returncode") == 0 and ENV_GONE_MARKER in lines:
        return []
    for line in lines:
        if line.startswith(ENV_SURVIVORS_MARKER):
            listed = [int(tok) for tok in line[len(ENV_SURVIVORS_MARKER):].split() if tok.isascii() and tok.isdigit()]
            return listed or [pgid]
    return None


def survivors_result(session: "ProcessSession", survivors: list[int]) -> dict:
    """The kill result for a session left running because *survivors* are alive."""
    alive = ", ".join(map(str, survivors))
    logger.warning("Kill incomplete for %s: %d process(es) still alive (%s) — session kept running",
                   session.id, len(survivors), alive)
    return {
        "status": "error",
        "error": f"Kill incomplete: {len(survivors)} process(es) still alive ({alive}); session running",
        "session_id": session.id, "survivors": survivors, "process_running": True}


class ProcessTerminationMixin:
    """The subclass supplies ``_host_pid_is_ours``, ``_is_host_pid_alive``,
    ``_detached_host_fate`` and ``_daemon_term_grace_seconds``. See ProcessRegistry."""

    @staticmethod
    def _proc_alive(proc) -> bool:
        """True if a psutil.Process is running and not a zombie (already dead, just unreaped)."""
        try:
            import psutil
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    @classmethod
    def _terminate_host_pid(cls, pid: int, expected_start: Optional[int] = None) -> None:
        """Terminate a host-visible PID and its descendants.
        ``expected_start`` (kernel start time at spawn) is re-validated first: a mismatch
        or dead PID means the number was recycled onto a stranger and we refuse to touch
        it — a leaked orphan beats tree-killing someone's browser. POSIX: snapshot descendants,
        SIGTERM the parent alone so it can perform an orderly shutdown, then clean up snapshot
        descendants that survive its grace window. Survivors are SIGKILLed after a second
        ``terminal.daemon_term_grace_seconds`` window. Windows:
        ``taskkill /T /F`` (psutil's stale PPID links miss orphans there); ``os.kill``
        is the fallback."""
        from tools.process_registry import _IS_WINDOWS

        if expected_start is not None and not cls._host_pid_is_ours(pid, expected_start):
            logger.warning(
                "Refusing to terminate host pid %d: start-time mismatch — "
                "PID was recycled onto an unrelated process.", pid)
            return

        def _sigterm_quietly():
            with suppress(OSError, ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)
        if _IS_WINDOWS:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True,
                    encoding='utf-8', errors='replace', timeout=10, creationflags=windows_hide_flags(),
                    stdin=subprocess.DEVNULL, check=False)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                _sigterm_quietly()
            return
        import psutil
        gone = (psutil.NoSuchProcess, psutil.AccessDenied, OSError)
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        except (OSError, PermissionError):
            _sigterm_quietly()
            return
        # Snapshot before signalling: once the parent exits, psutil can no longer
        # reliably find children that it failed to reap.
        try:
            descendants = parent.children(recursive=True)
        except gone:
            descendants = []

        # Let self-managing parents (notably Chromium/Electron) shut down their
        # tree before touching children. Killing their zygotes first can turn a
        # graceful browser shutdown into a crash dump.
        with suppress(gone):
            parent.terminate()

        grace = cls._daemon_term_grace_seconds()

        def _wait_for_exit(targets) -> None:
            if grace <= 0:
                return
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and any(cls._proc_alive(p) for p in targets):
                time.sleep(0.05)

        # Preserve descendants during the parent's configured shutdown window.
        _wait_for_exit([parent])

        # The snapshot is an anti-orphan guarantee: only descendants still alive
        # after the parent had its chance are asked to terminate themselves.
        remaining = descendants if grace <= 0 else [
            proc for proc in descendants if cls._proc_alive(proc)
        ]
        for proc in remaining:
            with suppress(gone):
                proc.terminate()

        # Preserve the existing SIGKILL escalation semantics for every owned
        # process that remains after its SIGTERM grace window. The parent is
        # included in case it ignored the first signal.
        targets = [parent, *remaining]
        # Escalate to SIGKILL for anything that ignored SIGTERM within the grace window.
        # ``psutil.wait_procs``' gone/alive partition is deliberately NOT trusted: it
        # reaps via ``Process.wait()`` and mis-partitions across zombie transitions in a
        # parent/child tree, leaving survivors un-killed. Re-probing every target is
        # deterministic.
        if grace <= 0:
            return
        _wait_for_exit(targets)
        # A parent that ignored SIGTERM (the interactive ``bash -lic`` wrapper does) keeps
        # running its script through both grace windows and can spawn children the first
        # snapshot never saw. Re-snapshot while it is still alive: once it is SIGKILLed
        # they reparent to init and nothing can find them again.
        with suppress(gone):
            if cls._proc_alive(parent):
                known = {proc.pid for proc in targets}
                targets.extend(p for p in parent.children(recursive=True) if p.pid not in known)
        for proc in targets:
            with suppress(gone):
                if cls._proc_alive(proc):
                    proc.kill()  # SIGKILL on POSIX
                    logger.info("Escalated to SIGKILL for pid %d (ignored SIGTERM within %.1fs grace)", proc.pid, grace)

    @staticmethod
    def _live_descendants(pid: int) -> list[int]:
        """PIDs of living non-zombie descendants of host PID ``pid`` (best-effort)."""
        try:
            import psutil
            children = psutil.Process(pid).children(recursive=True)
        except Exception:
            logger.debug("Could not list descendants of pid %s", pid, exc_info=True)
            return []
        return [c.pid for c in children if ProcessTerminationMixin._proc_alive(c)]

    def _already_exited_result(self, session: "ProcessSession", consume_output: bool) -> dict:
        """The kill result for a session that already exited.

        A double-forked descendant may still be alive in the session's systemd scope
        (#70716), so the scope is stopped first: ``systemctl --user stop`` signals every
        process in the cgroup and escalates to SIGKILL after TimeoutStopSec."""
        scope_stopped = self._stop_scope(session)
        with session._lock:
            result = self._exit_snapshot(session, "already_exited")
            finalizing = self._reader_finalizing(session)
        if not scope_stopped:
            result["scope_stop_failed"] = session.systemd_unit
        # Only suppress the autonomous turn after its output is present in
        # the explicit kill result, matching wait/log consumption.
        if consume_output and not finalizing:
            self._completion_consumed.add(session.id)
        return result

    # Sandbox kills a sweep runs at once: each waits on its own remote exec (up to the TERM
    # grace plus transport), so one at a time a few stalled backends use up a gateway's stop
    # budget. Capped, since sandboxes rate-limit exec starts.
    _SWEEP_SANDBOX_CONCURRENCY = 8

    def _kill_targets(self, targets: "list[ProcessSession]", *, source: str, consume_output: bool) -> list[dict]:
        """``kill_process`` each target: host sessions in order on this thread, sandbox
        sessions concurrently (each worker in a copy of this context, so it sees the same
        sandbox scope)."""
        from concurrent.futures import ThreadPoolExecutor
        from contextvars import copy_context

        def kill(s: "ProcessSession") -> dict:
            return self.kill_process(s.id, source=source, consume_output=consume_output)

        remote_ids = {s.id for s in targets if s.pid_scope == "sandbox" and s.env_ref is not None}
        if len(remote_ids) < 2:
            return [kill(s) for s in targets]
        with ThreadPoolExecutor(max_workers=min(len(remote_ids), self._SWEEP_SANDBOX_CONCURRENCY),
                                thread_name_prefix="proc-sweep") as pool:
            futures = [pool.submit(copy_context().run, kill, s) for s in targets if s.id in remote_ids]
            results = [kill(s) for s in targets if s.id not in remote_ids]
            results += [f.result() for f in futures]
        return results

    def _kill_env_group(self, session: "ProcessSession") -> Optional[dict]:
        """Stop a sandbox job's process group from inside the sandbox, once
        ``session.begin_group_kill`` switched the poller to the group. None once the group
        is proven gone; else the result to return, with the session left running and killable."""
        grace = self._daemon_term_grace_seconds()
        pgid = session.pid
        assert pgid is not None, "a sandbox kill needs the job's recorded PID"
        result = session.env_ref.execute(
            # Whole seconds: Modal and Daytona validate the exec timeout as an integer.
            env_group_kill_command(pgid, grace), timeout=math.ceil(grace + _ENV_KILL_TRANSPORT_SECONDS))
        survivors = parse_env_group_kill(result, pgid)
        if survivors == []:
            return None
        if survivors is None:
            logger.warning("Kill unconfirmed for %s: sandbox reply %r", session.id, result)
            return {
                "status": "error",
                "error": (f"Kill unconfirmed: the sandbox did not report process group {pgid} "
                          f"gone (exit {result.get('returncode')}); session running"),
                "session_id": session.id, "process_running": True}
        return survivors_result(session, survivors)

    # SIGKILL / taskkill are asynchronous: the kernel needs a scheduling tick to
    # tear the process down and the parent must reap it before poll()/isalive()
    # stop saying "alive". Verifying survivors in that window flagged every
    # escalated kill as incomplete.
    _KILL_SETTLE_SECONDS = 1.0

    def _post_kill_survivors(self, session: "ProcessSession") -> list[int]:
        """Host PIDs still alive once the kill signals have had time to land (#115490).

        Fail-closed: anything unverifiable counts as a survivor, so a kill
        that leaves a live tree can never write a killed receipt. Sandbox
        (env) sessions have no host-visible tree: ``_kill_env_group`` proves
        their group gone inside the sandbox, so they return no survivors here."""
        deadline = time.monotonic() + self._KILL_SETTLE_SECONDS
        while True:
            survivors = self._probe_survivors(session)
            if not survivors or time.monotonic() >= deadline:
                return survivors
            time.sleep(0.05)

    def _probe_survivors(self, session: "ProcessSession") -> list[int]:
        survivors: list[int] = []
        proc = getattr(session, "process", None)
        if proc is not None:
            try:
                root_alive = proc.poll() is None
            except Exception:
                root_alive = True
            if root_alive:
                survivors.append(getattr(proc, "pid", None) or session.pid)
        pty = getattr(session, "_pty", None)
        if pty is not None:
            try:
                pty_alive = bool(pty.isalive())
            except Exception:
                pty_alive = self._is_host_pid_alive(session.pid)
            if pty_alive:
                survivors.append(session.pid)
        if session.pid_scope == "host" and session.pid:
            if self._detached_host_fate(session.pid, session.host_start_time) == "running":
                if session.pid not in survivors:
                    survivors.append(session.pid)
                survivors.extend(
                    pid for pid in self._live_descendants(session.pid)
                    if pid not in survivors)
            # A dead/recycled root has no PID-scope descendants left to find:
            # reparented orphans are outside PID scope (systemd scope stop,
            # issued before this check, covers the cgroup case).
        return [pid for pid in survivors if pid]
