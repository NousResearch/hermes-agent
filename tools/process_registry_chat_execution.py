"""Local pipe-process launch evidence, never process-tree settlement proof."""
import os
from typing import Any


def spawn_local_pipe(registry, session, safe_command, env_vars, pty_scope_attempted, *, log=None):
    # Late facade lookup preserves existing platform and subprocess test seams.
    from tools import process_registry as facade
    popen_kwargs: dict[str, Any] = {"creationflags": facade.windows_hide_flags()} if facade._IS_WINDOWS else {}
    suffix = f"{session.id}-pipe-fallback" if pty_scope_attempted else session.id
    argv = registry._scope_argv(
        session, facade.record_exit_command(log, safe_command) if log else safe_command, suffix, "Local")
    if session.native_job_scope:
        from tools.process_registry_job import job_receipt_path
        from hermes_dir_acl import make_user_dir
        log = job_receipt_path(session.id).parent / (session.id + '.log')
        make_user_dir(log.parent, mode=0o700, parents=True, exist_ok=True)
        argv[-1] = f'set +m; {safe_command}'  # The helper alone records scope completion.
    env = registry._spawn_env(env_vars)
    if session.systemd_unit:
        env = facade.systemd_user_bus_env(env)
    # The scope does not create a new session; preserve TTY/process-group isolation.
    output = facade.open_output_log(log) if log else facade.subprocess.PIPE
    try:
        proc = facade.subprocess.Popen(
            argv, text=True, cwd=session.cwd, env=env, encoding="utf-8", errors="replace",
            stdout=output, stderr=facade.subprocess.STDOUT,
            stdin=facade.subprocess.DEVNULL, start_new_session=True, **popen_kwargs)
    finally:
        if log:
            os.close(output)
    session.process, session.output_log = proc, str(log or "")
    session.pid = proc.pid
    session.host_start_time = registry._safe_host_start_time(session.pid)
    try:
        registry._track_started(
            session, registry._log_reader_loop if log else registry._reader_loop, f"proc-reader-{session.id}")
    except Exception:
        registry._reap_untracked(session, proc)
        raise
    # Reader exit and checkpoint cleanup cannot settle descendants or external effects.
    return session


def reap_untracked(registry, session, proc):
    """Post-Popen failure cannot force-kill a native job's only reaper."""
    from contextlib import suppress
    from tools import process_registry as facade
    if session.native_job_scope:
        from tools.process_registry_job import request_job_stop, job_quiescent
        retain_failed_start(registry, session)
        request_job_stop(registry, session)
        reader = registry._pty_reader_loop if session._pty is not None else registry._log_reader_loop
        if job_quiescent(session):
            reader(session)  # Collect this original job's status/output, never replay its command.
        elif session._reader_thread is None or not session._reader_thread.is_alive():
            try:
                registry._track_started(session, reader, f'proc-start-recovery-{session.id}')
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Native job reader recovery failed; retained checkpoint %s", session.id)
        return
    with suppress(Exception):
        if session.systemd_unit:
            # Scope teardown is the authoritative cleanup for the worker cgroup;
            # the systemd-run wrapper PID is terminated below as fallback.
            registry._stop_scope(session)
        # _terminate_host_pid revalidates the kernel start time captured at
        # spawn before signalling: pgid == pid would only prove the pid leads a
        # group, and a reaped child can be recycled onto an unrelated leader.
        registry._terminate_host_pid(proc.pid, session.host_start_time)
    with suppress(Exception):
        proc.wait(timeout=5)


def retain_failed_start(registry, session):
    """Keep partial native launch evidence; failure never authorizes command replay."""
    from tools.process_registry_job import job_quiescent
    session._scope_stop_pending = not job_quiescent(session)
    with registry._lock:
        if session.id not in registry._finished:
            registry._running[session.id] = session
    registry._write_checkpoint()
    if session.systemd_unit:
        registry._stop_scope(session)
