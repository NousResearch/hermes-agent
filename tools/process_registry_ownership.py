"""Which local background jobs die with their owner, for ProcessRegistry (process_identity layer 4)."""

import subprocess
import threading
from contextlib import suppress
from typing import Any

from hermes_cli.process_identity import own_process_groups, release_process_groups, spawn_owned


class ProcessOwnershipMixin:
    """A local job's process group goes to the owner-death supervisor while the job's own process runs,
    so it dies with this process however this process dies (SIGKILL, OOM, crash): the agent-lifecycle
    sweeps only run while we are alive, and a killed CLI/TUI/worker left every job running on init.
    ``ProcessSession._owned_pgid`` is that group, cleared by the one release so a recycled pgid is
    never unregistered on someone else's behalf. The subclass supplies ``_terminate_host_pid``."""

    # Set by recover_from_checkpoint: this host re-adopts the jobs a crashed predecessor left
    # running, so its own jobs must outlive its crash the same way (see _owns_job).
    _jobs_outlive_host = False
    _owned_pgid_lock = threading.Lock()

    def _owns_job(self, persist_on_release: bool) -> bool:
        """Not owned, on purpose: ``persist_on_release`` jobs, which must outlive the agent (#41225);
        every job of a gateway, which re-adopts its jobs from the checkpoint after a crash (flagged by
        recover_from_checkpoint, which every gateway boot runs, checkpoint or not; a supervised gateway
        also before it gets there). The classic CLI, the TUI, the Desktop, ACP, cron and messaging all
        run their tools in a gateway, so their background jobs are not owned; a process that is not a
        gateway (a standalone serve with no session authority, the slash worker, an embedder of
        AIAgent) owns its jobs, and foreground commands are owned everywhere. Also not owned: every job of a
        managed worker, a per-turn interpreter whose jobs the next turn's worker adopts with their
        output log (worker_output_log); and Windows, where the layer-3 job object covers ownership."""
        from agent.runtime_session_store import is_worker_process
        from tools.process_registry import _IS_WINDOWS, _is_supervised_gateway_process

        return not (_IS_WINDOWS or persist_on_release or self._jobs_outlive_host
                    or is_worker_process() or _is_supervised_gateway_process())

    def _spawn_job(self, session: Any, argv: list[str], persist_on_release: bool, **popen_kwargs) -> subprocess.Popen:
        """Start a pipe job in its own session, owned unless ``_owns_job`` says it must outlive us."""
        if self._owns_job(persist_on_release):
            proc = spawn_owned(argv, purpose="terminal-background", **popen_kwargs)
            session._owned_pgid = proc.pid
            return proc
        # noqa: subprocess-stdin — a forwarder: spawn_local passes stdin=DEVNULL in popen_kwargs
        return subprocess.Popen(argv, start_new_session=True, **popen_kwargs)  # detached-spawn: ok — see _owns_job

    def _own_pty_job(self, session: Any, pty_proc: Any) -> None:
        """The PTY child leads its own session and group (pty.fork). Its hangup reaches only jobs that
        keep the default SIGHUP action; `nohup`/trap-HUP servers outlived a killed owner."""
        if self._owns_job(session.persist_on_release):
            own_process_groups([pty_proc.pid])
            session._owned_pgid = pty_proc.pid

    def _take_handoff_group(self, session: Any, proc: subprocess.Popen) -> None:
        """A foreground command that yields to background hands its supervised group over with the
        pipe (LocalEnvironment._run_bash): the job keeps it, unless jobs here must outlive us."""
        owned_pgid, proc._hermes_owned_pgid = getattr(proc, "_hermes_owned_pgid", None), None
        if owned_pgid is not None and self._owns_job(False):
            session._owned_pgid = owned_pgid
        elif owned_pgid is not None:
            release_process_groups([owned_pgid])

    def _return_handoff_group(self, session: Any, proc: subprocess.Popen, owned_pgid: Any) -> None:
        """The handoff failed (the reader could not start): the foreground wait resumes and must own
        the group again, covered by the supervisor, so its own completion releases it."""
        with self._owned_pgid_lock:
            session._owned_pgid = None
        if owned_pgid is not None:
            own_process_groups([owned_pgid])  # re-covers a group the gateway path released
            proc._hermes_owned_pgid = owned_pgid

    def _release_job_group(self, session: Any) -> None:
        """The job's own process exited or was killed: stop covering its group. What it left behind on
        purpose (`server &`) is no longer this job's, exactly as before."""
        with self._owned_pgid_lock:
            owned_pgid, session._owned_pgid = session._owned_pgid, None
        if owned_pgid is not None:
            release_process_groups([owned_pgid])

    def _reap_untracked(self, session: Any, proc: subprocess.Popen) -> None:
        """Post-Popen setup failed: kill the orphaned subprocess (and any setsid
        descendants) so nothing leaks untracked."""
        from tools.process_registry import _stop_systemd_unit

        with suppress(Exception):
            if session.systemd_unit:
                # Scope teardown is the authoritative cleanup for the worker cgroup;
                # the systemd-run wrapper PID is terminated below as fallback.
                _stop_systemd_unit(session.systemd_unit)
            # _terminate_host_pid revalidates the kernel start time captured at
            # spawn before signalling: pgid == pid would only prove the pid leads a
            # group, and a reaped child can be recycled onto an unrelated leader.
            self._terminate_host_pid(proc.pid, session.host_start_time)
        with suppress(Exception):
            proc.wait(timeout=5)
        self._release_job_group(session)
