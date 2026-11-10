"""WSL2 execution environment — all tools execute inside WSL2 via ``wsl -e bash``.

``ShellFileOperations`` routes ``read_file`` / ``write_file`` / ``patch`` /
``search_files`` through ``self.env.execute()``, so file I/O also runs inside
WSL.  Use Linux paths for all tools.  Windows files are accessible via the
``/mnt/c/`` mount point.

.. note::
    ``self.cwd`` stores a **Linux** path (e.g. ``/home/agents``), not a
    Windows path.  This is correct because ``cd`` commands dispatch
    through WSL's bash.  It differs from ``LocalEnvironment`` where
    ``self.cwd`` is a Windows native path.
"""

import logging
import os
import re
import shutil
import subprocess
import sys

from tools.environments.base import BaseEnvironment
from tools.environments.base_output import _pipe_stdin
from hermes_cli._subprocess_compat import windows_hide_flags


logger = logging.getLogger(__name__)

# Passed to ``wsl -d``. A leading ``-`` would be another wsl flag.
_DISTRO_RE = re.compile(r"^[A-Za-z0-9._-]+\Z")


def _validated_distro(distro: str | None) -> str:
    """Return a distro name safe to pass as ``wsl -d <name>``.

    Empty means "the default distro". Anything else must match
    ``^[A-Za-z0-9._-]+$`` and must not start with ``-``.
    """
    name = str(distro or "").strip()
    if not name:
        return ""
    if name.startswith("-") or _DISTRO_RE.fullmatch(name) is None:
        raise ValueError(
            f"Invalid WSL distro name {name!r}. Use letters, digits, '.', '_' or '-'; "
            "the name must not start with '-'."
        )
    return name


def _distro_args(distro: str | None) -> list[str]:
    name = _validated_distro(distro)
    return ["-d", name] if name else []


def _hidden_window_kwargs() -> dict:
    """``creationflags`` is Windows-only; passing it elsewhere raises TypeError."""
    if sys.platform != "win32":
        return {}
    return {"creationflags": windows_hide_flags()}


def _find_wsl() -> str:
    """Locate wsl.exe on the Windows host.

    Returns the absolute path to wsl.exe.  Raises RuntimeError if
    WSL is not installed.
    """
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    system32_wsl = os.path.join(system_root, "System32", "wsl.exe")
    if os.path.isfile(system32_wsl):
        return system32_wsl
    found = shutil.which("wsl")
    if found:
        return found
    raise RuntimeError(
        "wsl.exe not found. Install WSL2: "
        "https://learn.microsoft.com/en-us/windows/wsl/install"
    )


def _probe_wsl_home(wsl_exe: str, distro: str = "") -> str:
    args = [wsl_exe, *_distro_args(distro), "-e", "bash", "-c", "echo $HOME"]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=5,
            **_hidden_window_kwargs(),
        )
        home = result.stdout.strip()
        if home and home.startswith("/"):
            return home
    except Exception:
        pass
    return "/root"


def _ensure_wsl_available() -> str:
    """Verify wsl.exe is reachable and return its path.

    Raises RuntimeError with actionable messages depending on the failure:
      - wsl.exe not found → install WSL2
      - wsl.exe times out → WSL distro may be stopped, run 'wsl' first
    """
    wsl_exe = _find_wsl()
    try:
        result = subprocess.run(
            [wsl_exe, "--version"],
            capture_output=True, text=True, timeout=10,
            **_hidden_window_kwargs(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"wsl.exe --version failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "wsl.exe is installed but did not respond. "
            "The WSL distro may be stopped — open a terminal and run 'wsl' first."
        ) from None
    except FileNotFoundError:
        raise RuntimeError(
            "wsl.exe not found. Install WSL2: "
            "https://learn.microsoft.com/en-us/windows/wsl/install"
        ) from None
    return wsl_exe


class WslEnvironment(BaseEnvironment):
    """Run terminal commands inside WSL2 via ``wsl -e bash``.

    Windows CWDs are auto-converted to /mnt/c/... by _get_env_config()
    on backend switch.  The agent receives prompt hints to use Linux
    paths for all tools (terminal + file I/O).

    Optional env vars:
      TERMINAL_WSL_DISTRO  - WSL distribution name (e.g. "Debian")
    """

    _snapshot_timeout: int = 30

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None,
                 distro: str = ""):
        # distro priority (first non-empty wins):
        #   explicit distro param → env dict → process environment
        # Validate before touching wsl.exe so a bad name fails closed.
        _env = env or {}
        self._distro = _validated_distro(
            distro or _env.get("TERMINAL_WSL_DISTRO") or os.getenv("TERMINAL_WSL_DISTRO", "")
        )
        self._wsl = _ensure_wsl_available()
        if not cwd:
            cwd = _probe_wsl_home(self._wsl, self._distro)
        super().__init__(cwd=cwd, timeout=timeout, env=env)
        self.init_session()

    # _before_execute is intentionally NOT overridden — unlike SSH/Daytona,
    # WSL mounts the Windows filesystem at /mnt/c/, so ~/.hermes/ (on C:)
    # is accessible natively without FileSyncManager syncing.

    @staticmethod
    def get_temp_dir() -> str:
        """Return the backend temp directory.  WSL's /tmp is always writable
        and cleaned on distro restart, so temp file leaks are bounded."""
        # no-tmp: ok — path is inside the WSL distro, not the Windows host filesystem
        return "/tmp"

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,   # enforced by BaseEnvironment._wait_for_process
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        wsl_args = [self._wsl, *_distro_args(self._distro)]
        if login:
            wsl_args.extend(["-e", "bash", "-l", "-c", cmd_string])
        else:
            wsl_args.extend(["-e", "bash", "-c", cmd_string])

        # Merge stderr into stdout so _wait_for_process drains a single
        # stream.  WSL may emit "Installing..." progress to stderr on first
        # launch of a distro — merging avoids interleaved-output races.
        proc = subprocess.Popen(
            wsl_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            text=True,
            **_hidden_window_kwargs(),
        )

        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)

        return proc

    def _update_cwd(self, result: dict):
        self._extract_cwd_from_output(result)

    def cleanup(self):
        # Temp files live in WSL's /tmp — unreachable from Windows Python.
        # Remove them via wsl -e rm instead of os.unlink.  Best-effort: if
        # the distro has already stopped, the files die with its /tmp anyway.
        try:
            args = [self._wsl, *_distro_args(self._distro), "-e", "rm", "-f", self._snapshot_path, self._cwd_file]
            subprocess.run(
                args,
                timeout=5, capture_output=True,
                **_hidden_window_kwargs(),
            )
        except Exception:
            logger.debug("Cleanup rm failed (distro may have stopped)", exc_info=True)

    def _kill_process(self, proc):
        # wsl.exe is the root of its process tree inside WSL — killing it
        # with terminate() (Ctrl+C) then kill() (force) is sufficient to
        # clean up the entire bash session.  os.killpg is a Windows footgun
        # and unnecessary here; the WSL init process handles orphan cleanup.
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
