"""Session-persistent kernels for REMOTE terminal backends (docker/ssh/modal).

Closes the gap tracked in hermes-agent#96873: local execute_code holds a
persistent kernel child (tools/code_kernel.py); remote backends previously
re-shipped and re-ran a fresh script per call, losing all interpreter state.

The remote transport offers exactly one primitive — ``env.execute(cmd)``,
run-to-completion — so the three things the local kernel gets from owning a
child process are rebuilt on top of it:

1. **A process that outlives one env.execute():** the kernel runner is
   started detached (``nohup ... &``) and its PID recorded; each later cell
   first probes liveness with ``kill -0``.
2. **A conversation channel:** a file-based CELL protocol in the kernel dir
   (``cell_req_NNNNNN.json`` / ``cell_res_NNNNNN.json``), sibling to the
   existing file-based TOOL-RPC protocol (req_/res_ files) which is reused
   unchanged — the host-side ``_rpc_poll_loop`` is started per cell with the
   calling thread's context, which is what gives per-cell tool authority.
3. **Death detection:** a failed liveness probe (transport drop, container
   restart, OOM-killed runner) reads as *kernel died: state lost*; the next
   call respawns fresh and says so — never a hung poll loop, because every
   wait is bounded by the cell timeout.

Same invariants as local: owner = approval session key with the
``::child::{id}`` qualifier for delegated children (imported from
tools.code_kernel — one resolver, cannot drift), same generated tool stubs,
same output post-processing in the caller. ``reset=true`` kills and
respawns. Spawn failure fails OPEN to the per-call path with a note, so a
degraded remote host never blocks execution entirely.
"""
from __future__ import annotations

import atexit
import base64
import json
import logging
import shlex
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# One lock guards the registry; teardown runs outside it (mirrors code_kernel).
_REMOTE_KERNELS: Dict[Tuple, "RemoteKernel"] = {}
_REMOTE_KERNELS_LOCK = threading.Lock()

# How often the host polls the remote for a cell result file. Each poll is
# one env.execute round-trip (typically 0.1-0.4s on ssh/docker), so this is
# a floor, not a rate.
_CELL_POLL_INTERVAL = 0.5

# The remote runner: a tiny forever-loop that polls for cell request files,
# execs them in one persistent namespace, and writes response files. It is
# deliberately transport-agnostic (pure files) and stdlib-only. Cells and
# tool-RPC share the kernel dir but use distinct prefixes.
REMOTE_KERNEL_RUNNER_SOURCE = '''\
"""Auto-generated Hermes REMOTE session-kernel runner (file cell protocol)."""
import contextlib
import io
import json
import os
import sys
import time
import traceback

KDIR = os.environ["HERMES_KERNEL_DIR"]
CELLS = os.path.join(KDIR, "cells")
CAPTURE_LIMIT = {capture_limit}
IDLE_EXIT_SECONDS = {idle_exit}

GLOBALS = {{"__name__": "__main__", "__builtins__": __builtins__}}


def _bounded(text):
    if len(text) <= CAPTURE_LIMIT:
        return text, False
    return text[:CAPTURE_LIMIT], True


def main():
    execution_count = 0
    last_activity = time.time()
    while True:
        pending = sorted(
            f for f in os.listdir(CELLS)
            if f.startswith("cell_req_") and f.endswith(".json")
        )
        if not pending:
            if time.time() - last_activity > IDLE_EXIT_SECONDS:
                return  # self-reap: nobody is talking to us anymore
            time.sleep(0.2)
            continue
        for name in pending:
            req_path = os.path.join(CELLS, name)
            try:
                with open(req_path, "r", encoding="utf-8") as f:
                    request = json.load(f)
            except Exception:
                # Partially-written request (ship in progress): retry next tick.
                continue
            os.remove(req_path)
            last_activity = time.time()
            execution_count += 1
            out, err = io.StringIO(), io.StringIO()
            status = "ok"
            trace = ""
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    exec(compile(request["code"], "<cell>", "exec"), GLOBALS)
            except SystemExit as exc:
                status = "exit"
                trace = "SystemExit: " + repr(exc.code)
            except BaseException:
                status = "error"
                trace = traceback.format_exc()
            stdout_text, stdout_clipped = _bounded(out.getvalue())
            stderr_text, stderr_clipped = _bounded(err.getvalue())
            payload = {{
                "id": request.get("id", ""),
                "status": status,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdout_clipped": stdout_clipped,
                "stderr_clipped": stderr_clipped,
                "traceback": trace,
                "execution_count": execution_count,
            }}
            res_name = name.replace("cell_req_", "cell_res_")
            tmp = os.path.join(CELLS, res_name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, os.path.join(CELLS, res_name))
            if status == "exit":
                return


if __name__ == "__main__":
    main()
'''


@dataclass
class RemoteKernel:
    """Host-side record of one detached remote kernel process."""

    env: Any
    env_type: str
    kernel_dir: str
    pid: str
    rpc_token: str
    owner: str
    created: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    execution_count: int = 0
    cell_seq: int = 0


def _kernel_key(owner: str, env_type: str, task_env_id: str) -> Tuple:
    return (owner, "remote", env_type, task_env_id)


def _is_alive(kernel: RemoteKernel) -> bool:
    """Bounded liveness probe: kill -0 through the transport.

    Any transport failure counts as dead — the caller respawns. This is the
    "death detection" leg: a dropped ssh connection and a dead runner are
    indistinguishable from here, and both have the same correct answer.
    """
    try:
        probe = kernel.env.execute(
            f"kill -0 {shlex.quote(kernel.pid)} 2>/dev/null && echo ALIVE",
            cwd="/", timeout=15,
        )
        return "ALIVE" in (probe.get("output", "") or "")
    except Exception:
        return False


def _kill(kernel: RemoteKernel) -> None:
    """Best-effort kill of the runner and its subprocesses, then rm -rf."""
    try:
        kernel.env.execute(
            # Kill the runner's process group if the shell gave it one,
            # falling back to the single PID.
            f"pkill -TERM -P {shlex.quote(kernel.pid)} 2>/dev/null; "
            f"kill {shlex.quote(kernel.pid)} 2>/dev/null; true",
            cwd="/", timeout=15,
        )
    except Exception:
        logger.debug("remote kernel kill failed (transport?)", exc_info=True)
    try:
        kernel.env.execute(
            f"rm -rf {shlex.quote(kernel.kernel_dir)}", cwd="/", timeout=15,
        )
    except Exception:
        logger.debug("remote kernel dir cleanup failed", exc_info=True)


def shutdown_all_remote_kernels() -> None:
    with _REMOTE_KERNELS_LOCK:
        kernels = list(_REMOTE_KERNELS.values())
        _REMOTE_KERNELS.clear()
    for kernel in kernels:
        _kill(kernel)


def shutdown_remote_kernels_for_owner(owner: str) -> None:
    """Session-boundary disposal — wired to the same clear_session hook as
    local kernels, so /new and session close reap both kinds."""
    if not owner:
        return
    with _REMOTE_KERNELS_LOCK:
        doomed = [k for k in _REMOTE_KERNELS if k[0] == owner]
        kernels = [_REMOTE_KERNELS.pop(k) for k in doomed]
    for kernel in kernels:
        _kill(kernel)


atexit.register(shutdown_all_remote_kernels)


def _spawn_remote_kernel(env, env_type: str, owner: str, task_env_id: str,
                         sandbox_tools: frozenset, *,
                         idle_exit: int) -> Optional[RemoteKernel]:
    """Start a detached kernel runner on the remote. None on failure."""
    from tools.code_execution_tool import (
        MAX_STDOUT_BYTES,
        _ship_file_to_remote,
        _env_temp_dir,
        generate_hermes_tools_module,
    )
    import secrets as _secrets

    kernel_dir = f"{_env_temp_dir(env)}/hermes_rkernel_{uuid.uuid4().hex[:12]}"
    q_dir = shlex.quote(kernel_dir)
    try:
        env.execute(f"mkdir -p {q_dir}/cells {q_dir}/rpc", cwd="/", timeout=15)

        rpc_token = _secrets.token_urlsafe(32)
        runner_src = REMOTE_KERNEL_RUNNER_SOURCE.format(
            capture_limit=MAX_STDOUT_BYTES,
            idle_exit=idle_exit,
        )
        _ship_file_to_remote(env, f"{kernel_dir}/kernel_runner.py", runner_src)
        tools_src = generate_hermes_tools_module(
            list(sandbox_tools), transport="file",
        )
        _ship_file_to_remote(env, f"{kernel_dir}/hermes_tools.py", tools_src)

        env_prefix = (
            f"HERMES_KERNEL_DIR={q_dir} "
            f"HERMES_RPC_DIR={shlex.quote(kernel_dir + '/rpc')} "
            f"HERMES_RPC_TOKEN={shlex.quote(rpc_token)} "
            f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={q_dir}"
        )
        started = env.execute(
            f"cd {q_dir} && nohup env {env_prefix} python3 kernel_runner.py "
            f"> {q_dir}/runner.log 2>&1 & echo PID:$!",
            cwd="/", timeout=20,
        )
        pid = ""
        for line in (started.get("output", "") or "").splitlines():
            if line.strip().startswith("PID:"):
                pid = line.strip()[4:].strip()
                break
        if not pid.isdigit():
            logger.warning("remote kernel spawn returned no PID: %r",
                           started.get("output", ""))
            env.execute(f"rm -rf {q_dir}", cwd="/", timeout=15)
            return None

        kernel = RemoteKernel(
            env=env, env_type=env_type, kernel_dir=kernel_dir,
            pid=pid, rpc_token=rpc_token, owner=owner,
        )
        if not _is_alive(kernel):
            # Died instantly (missing python3 was pre-checked by the caller,
            # so this is unexpected) — surface the runner log at debug.
            try:
                log = env.execute(f"cat {q_dir}/runner.log", cwd="/", timeout=10)
                logger.warning("remote kernel died at spawn: %s",
                               (log.get("output", "") or "")[:500])
            except Exception:
                pass
            env.execute(f"rm -rf {q_dir}", cwd="/", timeout=15)
            return None
        return kernel
    except Exception:
        logger.warning("remote kernel spawn failed", exc_info=True)
        try:
            env.execute(f"rm -rf {q_dir}", cwd="/", timeout=15)
        except Exception:
            pass
        return None


def execute_in_remote_kernel(
    code: str,
    *,
    env,
    env_type: str,
    task_env_id: str,
    sandbox_tools: frozenset,
    timeout: int,
    max_tool_calls: int,
    reset: bool,
    idle_exit: int = 1800,
) -> Optional[Dict[str, Any]]:
    """Run one cell in the owner's remote kernel.

    Returns the raw cell result dict (caller does output post-processing),
    or ``None`` when no kernel could be spawned — the caller falls open to
    the per-call path. ``state_lost`` / ``state_reset`` / ``reused`` ride in
    the ``kernel`` sub-dict, matching the local kernel's result shape.
    """
    from tools.code_kernel import _resolve_owner
    from tools.code_execution_tool import (
        _rpc_poll_loop,
        _ship_file_to_remote,
    )
    from tools.thread_context import propagate_context_to_thread

    owner = _resolve_owner(task_env_id)
    key = _kernel_key(owner, env_type, task_env_id)
    state_lost = False
    state_reset = False

    with _REMOTE_KERNELS_LOCK:
        kernel = _REMOTE_KERNELS.get(key)

    if kernel is not None and reset:
        with _REMOTE_KERNELS_LOCK:
            _REMOTE_KERNELS.pop(key, None)
        _kill(kernel)
        kernel = None
        state_reset = True

    if kernel is not None and not _is_alive(kernel):
        # Transport drop, container restart, self-reaped on idle, OOM — all
        # the same answer: report the loss, respawn fresh.
        with _REMOTE_KERNELS_LOCK:
            _REMOTE_KERNELS.pop(key, None)
        _kill(kernel)  # best-effort dir cleanup; process is already gone
        kernel = None
        state_lost = True

    reused = kernel is not None
    if kernel is None:
        kernel = _spawn_remote_kernel(
            env, env_type, owner, task_env_id, sandbox_tools,
            idle_exit=idle_exit,
        )
        if kernel is None:
            return None  # fail open to per-call
        with _REMOTE_KERNELS_LOCK:
            _REMOTE_KERNELS[key] = kernel

    kernel.last_used = time.monotonic()
    kernel.cell_seq += 1
    seq = f"{kernel.cell_seq:06d}"
    q_cells = shlex.quote(f"{kernel.kernel_dir}/cells")

    # Clean stale tool-RPC requests from a previous cell before arming this
    # cell's poll loop, so a background thread the last cell leaked cannot
    # smuggle a call into this cell's authority window.
    try:
        env.execute(
            f"rm -f {shlex.quote(kernel.kernel_dir + '/rpc')}/req_* "
            f"{shlex.quote(kernel.kernel_dir + '/rpc')}/res_*",
            cwd="/", timeout=10,
        )
    except Exception:
        pass

    tool_call_log: list = []
    tool_call_counter = [0]
    stop_event = threading.Event()
    # Per-cell RPC thread carrying THIS call's approval/session context —
    # the remote analogue of CellAuthority: authority lives exactly as long
    # as the cell's poll loop.
    rpc_thread = threading.Thread(
        target=propagate_context_to_thread(_rpc_poll_loop),
        args=(
            env, f"{kernel.kernel_dir}/rpc", task_env_id,
            tool_call_log, tool_call_counter, max_tool_calls,
            sandbox_tools, stop_event, kernel.rpc_token,
        ),
        daemon=True,
    )
    rpc_thread.start()

    cell_status = "no-result"
    cell_payload: Dict[str, Any] = {}
    try:
        request = json.dumps({"id": seq, "code": code}, ensure_ascii=False)
        _ship_file_to_remote(
            env, f"{kernel.kernel_dir}/cells/cell_req_{seq}.json.tmp", request,
        )
        env.execute(
            f"mv {q_cells}/cell_req_{seq}.json.tmp {q_cells}/cell_req_{seq}.json",
            cwd="/", timeout=10,
        )

        deadline = time.monotonic() + timeout
        res_name = f"cell_res_{seq}.json"
        while time.monotonic() < deadline:
            try:
                probe = env.execute(
                    f"cat {q_cells}/{shlex.quote(res_name)} 2>/dev/null",
                    cwd="/", timeout=20,
                )
            except Exception:
                # One flaky round-trip is not kernel death; liveness decides.
                time.sleep(_CELL_POLL_INTERVAL)
                continue
            body = (probe.get("output", "") or "").strip()
            if body:
                try:
                    cell_payload = json.loads(body)
                    cell_status = cell_payload.get("status", "error")
                except ValueError:
                    cell_status = "protocol-error"
                env.execute(
                    f"rm -f {q_cells}/{shlex.quote(res_name)}",
                    cwd="/", timeout=10,
                )
                break
            time.sleep(_CELL_POLL_INTERVAL)
        else:
            cell_status = "timeout"
    finally:
        stop_event.set()
        rpc_thread.join(timeout=5)

    if cell_status in ("timeout", "protocol-error", "no-result"):
        # No safe way to interrupt one cell in place (same contract as
        # local): kill the kernel, report the loss, respawn next call.
        with _REMOTE_KERNELS_LOCK:
            _REMOTE_KERNELS.pop(key, None)
        _kill(kernel)
        return {
            "status": "timeout" if cell_status == "timeout" else "error",
            "stdout": "",
            "stderr": "",
            "traceback": "",
            "tool_calls_made": tool_call_counter[0],
            "kernel": {
                "reused": reused,
                "remote": True,
                "ended": True,
                "state_lost": True,
                "note": (
                    "Cell timed out; the remote session kernel was killed and "
                    "its state was lost. The next call starts a fresh kernel."
                    if cell_status == "timeout" else
                    "Remote kernel protocol failure; kernel killed, state lost."
                ),
            },
        }

    if cell_status == "exit":
        with _REMOTE_KERNELS_LOCK:
            _REMOTE_KERNELS.pop(key, None)
        _kill(kernel)

    kernel.execution_count = int(cell_payload.get("execution_count", 0) or 0)

    result: Dict[str, Any] = {
        "status": "success" if cell_status in ("ok", "exit") else "error",
        "stdout": cell_payload.get("stdout", ""),
        "stderr": cell_payload.get("stderr", ""),
        "traceback": cell_payload.get("traceback", ""),
        "stdout_clipped": bool(cell_payload.get("stdout_clipped")),
        "stderr_clipped": bool(cell_payload.get("stderr_clipped")),
        "tool_calls_made": tool_call_counter[0],
        "kernel": {
            "reused": reused,
            "remote": True,
            "execution_count": kernel.execution_count,
        },
    }
    if cell_status == "exit":
        result["kernel"]["ended"] = True
    if state_reset:
        result["kernel"]["state_reset"] = True
    if state_lost:
        result["kernel"]["state_lost"] = True
        result["kernel"]["note"] = (
            "The previous remote kernel was gone (transport drop, container "
            "restart, or idle self-exit); state from earlier calls was lost "
            "and a fresh kernel was started."
        )
    if cell_status == "error" and result["traceback"]:
        result["error"] = result["traceback"].strip().splitlines()[-1]
    return result
