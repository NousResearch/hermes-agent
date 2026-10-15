"""Shell process groups for env-backed background commands."""

import shlex
from typing import Any


def background_command(temp_dir: str, session_id: str, command: str) -> str:
    """Create one job group without requiring setsid/pkill in minimal images."""
    q = shlex.quote
    log_path, pid_path, exit_path = (
        f"{temp_dir}/hermes_bg_{session_id}.{ext}" for ext in ("log", "pid", "exit")
    )
    # Monitor mode isolates the wrapper; disable it inside the job so the
    # command and its ordinary descendants stay in that same process group.
    # The TERM trap lets the wrapper record a normally terminated child's exit.
    return (
        f"mkdir -p {q(temp_dir)} && "
        f"( set -m; ( set +m; trap : TERM; nohup bash -lc {q(command)} > {q(log_path)} 2>&1; "
        f"rc=$?; printf '%s\\n' \"$rc\" > {q(exit_path)} ) > /dev/null 2>&1 < /dev/null & "
        f"echo $! > {q(pid_path)} ) && cat {q(pid_path)}"
    )


def kill_process_group(env: Any, pid: int) -> None:
    """Terminate the owned job group, including children that ignore SIGTERM."""
    # Always escalate the group, even if the wrapper has already exited:
    # a TERM-ignoring writer can outlive it. A vanished group is normal.
    command = (
        f"kill -TERM -- -{pid} 2>/dev/null; "
        f"sleep 0.2; kill -KILL -- -{pid} 2>/dev/null || "
        f"{{ ! kill -0 -- -{pid} 2>/dev/null; }}"
    )
    result = env.execute(command, timeout=5)
    if result.get("returncode", 0) != 0:
        raise RuntimeError("Failed to terminate background process group")
