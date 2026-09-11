"""Linux ``no_new_privs`` escape for local sudo commands.

Electron enables the one-way ``PR_SET_NO_NEW_PRIVS`` latch.  A Desktop-spawned
backend inherits it, so the kernel ignores sudo's setuid bit.  A transient user
service is created by the systemd user manager rather than by that process tree
and therefore does not inherit the latch.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


_STATUS_PATH = Path("/proc/self/status")
_ENV_LOADER = """while IFS= read -r -d '' entry; do
    [ -z "$entry" ] && break
    export "$entry"
done
exec "$@"
"""
_SYSTEMD_CLIENT_ENV_KEYS = frozenset({"HOME", "LANG", "LOGNAME", "PATH", "TERM", "USER"})


@dataclass(frozen=True)
class SystemdRunEscape:
    args: list[str]
    env: dict[str, str]
    stdin_data: str


def no_new_privs_enabled(status_path: Path = _STATUS_PATH) -> bool:
    """Read the current process's kernel latch from procfs."""
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("NoNewPrivs:"):
                return line.split(maxsplit=1)[1].strip() == "1"
    except (OSError, IndexError):
        return False
    return False


def _systemd_client_env(source: Mapping[str, str], runtime_dir: str) -> dict[str, str]:
    """Minimal non-secret environment needed by the systemd-run client."""
    env = {
        key: value
        for key, value in source.items()
        if key in _SYSTEMD_CLIENT_ENV_KEYS or key.startswith("LC_")
    }
    env["XDG_RUNTIME_DIR"] = runtime_dir
    env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime_dir}/bus"
    return env


def prepare_systemd_run_escape(
    args: Sequence[str],
    run_env: Mapping[str, str],
    cwd: str,
    stdin_data: str | None,
    *,
    has_sudo: bool,
    platform: str = os.sys.platform,
    status_path: Path = _STATUS_PATH,
    which: Callable[[str], str | None] = shutil.which,
    getuid: Callable[[], int] | None = getattr(os, "getuid", None),
    path_exists: Callable[[str], bool] = os.path.exists,
) -> SystemdRunEscape | None:
    """Return a systemd-manager launch when inherited NNP would break sudo.

    The sanitized terminal environment is transferred over stdin as NUL-delimited
    entries.  Unlike ``systemd-run --setenv``, this keeps credentials out of argv
    and the journal.  A blank NUL entry terminates the prefix; the command then
    inherits the still-open stdin at the exact byte where the caller's original
    payload begins.
    """
    if platform != "linux" or not has_sudo or not no_new_privs_enabled(status_path):
        return None
    if getuid is None:
        return None

    systemd_run = which("systemd-run")
    runtime_dir = f"/run/user/{getuid()}"
    if not systemd_run or not path_exists(f"{runtime_dir}/bus"):
        return None

    env_prefix = "".join(f"{key}={value}\0" for key, value in sorted(run_env.items())) + "\0"
    escaped_args = [
        systemd_run,
        "--user",
        "--pipe",
        "--quiet",
        "--collect",
        "--wait",
        f"--working-directory={cwd}",
        "--",
        "/bin/bash",
        "-c",
        _ENV_LOADER,
        "hermes-no-new-privs",
        *args,
    ]
    return SystemdRunEscape(
        args=escaped_args,
        env=_systemd_client_env(os.environ, runtime_dir),
        stdin_data=env_prefix + (stdin_data or ""),
    )
