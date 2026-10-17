"""Local pipe-process launch evidence, never process-tree settlement proof."""
from uuid import uuid4
from typing import Any


def reserve_local_process():
    from agent.chat_execution_context import concurrent_tool_resource_for_submit
    return concurrent_tool_resource_for_submit(str(uuid4()), kind="terminal_local_process")


def start_local_process(handle):
    if handle is None:
        return
    from agent.chat_execution_context import start_reserved_tool_resource
    if not start_reserved_tool_resource(handle):
        raise RuntimeError("Durable terminal process start refused")


def record_local_process(handle, process_session_id, pid, host_start_time):
    if handle is None:
        return
    from gateway.chat_execution_recovery import ChatExecutionStore
    if host_start_time is None or not ChatExecutionStore(handle.home).record_terminal_process_lineage(
            handle.session_key, handle.generation, handle.resource_id, handle.session_id,
            process_session_id, pid, host_start_time):
        raise RuntimeError("Durable terminal process lineage refused")


def spawn_local_pipe(registry, session, safe_command, env_vars, pty_scope_attempted, resource):
    # Late facade lookup preserves existing platform and subprocess test seams.
    from tools import process_registry as facade
    popen_kwargs: dict[str, Any] = {"creationflags": facade.windows_hide_flags()} if facade._IS_WINDOWS else {}
    suffix = f"{session.id}-pipe-fallback" if pty_scope_attempted else session.id
    argv = registry._scope_argv(session, safe_command, suffix, "Local")
    env = registry._spawn_env(env_vars)
    if session.systemd_unit:
        env = facade.systemd_user_bus_env(env)
    start_local_process(resource)
    # The scope does not create a new session; preserve TTY/process-group isolation.
    proc = facade.subprocess.Popen(
        argv, text=True, cwd=session.cwd, env=env, encoding="utf-8", errors="replace",
        stdout=facade.subprocess.PIPE, stderr=facade.subprocess.STDOUT,
        stdin=facade.subprocess.DEVNULL, start_new_session=True, **popen_kwargs)
    session.process = proc
    session.pid = proc.pid
    session.host_start_time = registry._safe_host_start_time(session.pid)
    try:
        record_local_process(resource, session.id, session.pid, session.host_start_time)
        registry._track_started(session, registry._reader_loop, f"proc-reader-{session.id}")
    except Exception:
        registry._reap_untracked(session, proc)
        raise
    # Reader exit and checkpoint cleanup cannot settle descendants or external effects.
    return session
