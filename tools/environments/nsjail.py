"""Nsjail execution environment — lightweight namespace sandbox (Linux-only).

Wraps each command with ``nsjail -Mo`` (once mode) to execute inside a fresh
Linux user/mount/(net/pid) namespace set. No daemon, no container image, no
root required: spawn latency is ~20 ms on a Pi-class device.

Threat model: commands see a read-only view of the host filesystem with a
single rw bindmount for the working directory, a tmpfs-backed session dir for
snapshot state, and (by default) no network. Suitable for running untrusted
short-lived code — the LLM equivalent of a CTF judge jail — without the boot
overhead of Docker.

Escape hatch: set ``terminal.nsjail_config`` to the path of a user-supplied
``nsjail.cfg`` (protobuf text-format) and Hermes will delegate all policy to
that file, only injecting the working directory and timeout.
"""

import logging
import os
import platform
import shutil
import subprocess
from typing import Optional

from tools.environments.base import (
    BaseEnvironment,
    _pipe_stdin,
    get_sandbox_dir,
)
from tools.environments.local import _SANE_PATH, _sanitize_subprocess_env

logger = logging.getLogger(__name__)

_IS_LINUX = platform.system() == "Linux"


_nsjail_executable: Optional[str] = None  # resolved once, cached


def find_nsjail() -> Optional[str]:
    """Locate the nsjail CLI binary.

    Resolution order:
    1. ``HERMES_NSJAIL_BINARY`` env var — explicit override
    2. ``nsjail`` on PATH via ``shutil.which``

    Returns the absolute path, or ``None`` if nsjail cannot be found.
    """
    global _nsjail_executable
    if _nsjail_executable is not None:
        return _nsjail_executable

    override = os.getenv("HERMES_NSJAIL_BINARY")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        _nsjail_executable = override
        logger.info("Using HERMES_NSJAIL_BINARY override: %s", override)
        return override

    found = shutil.which("nsjail")
    if found:
        _nsjail_executable = found
        return found

    return None


def _ensure_nsjail_available() -> str:
    """Best-effort preflight that nsjail is usable before the first exec."""
    if not _IS_LINUX:
        raise RuntimeError(
            "Nsjail backend is only supported on Linux. "
            "Set terminal.backend to 'local', 'docker', or 'singularity'."
        )

    exe = find_nsjail()
    if not exe:
        raise RuntimeError(
            "nsjail binary not found. Install nsjail "
            "(https://github.com/google/nsjail) and ensure it is on PATH, "
            "or set HERMES_NSJAIL_BINARY to its absolute path."
        )

    try:
        result = subprocess.run(
            [exe, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Resolved nsjail binary {exe!r} could not be executed: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"'{exe} --help' timed out after 5s; the binary appears unresponsive."
        ) from exc

    # nsjail prints help to stderr and exits non-zero; both are fine as long
    # as we can tell the binary responds at all.
    if result.returncode not in (0, 1) and "Usage:" not in (result.stderr + result.stdout):
        raise RuntimeError(
            f"'{exe} --help' returned an unexpected error "
            f"(rc={result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )

    return exe


# Environment variables every sandboxed bash session needs to behave normally.
# Forwarded from the sanitized host env when present so locale/paths work.
_ALWAYS_FORWARD_ENV = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "USER",
    "LOGNAME",
    "SHELL",
)


class NsjailEnvironment(BaseEnvironment):
    """Spawn-per-call execution inside an nsjail user-namespace jail.

    The host filesystem is bindmounted read-only; the working directory is
    bindmounted read-write at the same path; a host-side session directory is
    bindmounted at ``/tmp`` so the env-snapshot file written by
    :class:`BaseEnvironment` persists across invocations.
    """

    def __init__(
        self,
        cwd: str = "",
        timeout: int = 60,
        env: dict | None = None,
        *,
        cpu: int = 1,
        memory: int = 5120,
        disk: int = 51200,
        task_id: str = "default",
        config_file: str | None = None,
        allow_net: bool = False,
        forward_env: list[str] | None = None,
    ):
        resolved_cwd = os.path.abspath(os.path.expanduser(cwd or os.getcwd()))
        super().__init__(cwd=resolved_cwd, timeout=timeout, env=env)

        # Pin the read-write bindmount to the cwd captured at init. If the
        # user cd's elsewhere mid-session, that new location stays read-only
        # (the /-level bindmount_ro still covers it for reads). Without this
        # pin, a ``cd /etc`` followed by another exec would re-bindmount /etc
        # as rw and silently weaken the sandbox.
        self._initial_cwd = resolved_cwd

        self.executable = _ensure_nsjail_available()

        self._config_file = (
            os.path.abspath(os.path.expanduser(config_file))
            if config_file
            else None
        )
        if self._config_file and not os.path.isfile(self._config_file):
            raise RuntimeError(
                f"terminal.nsjail_config points to a non-existent file: "
                f"{self._config_file}"
            )

        self._allow_net = bool(allow_net)
        self._cpu = max(int(cpu), 0)
        self._memory = max(int(memory), 0)
        self._disk = max(int(disk), 0)
        self._forward_env = list(forward_env or [])
        self._task_id = str(task_id)

        # Host-side session dir bindmounted into the jail as /tmp, so the env
        # snapshot + cwd marker file persist across spawn-per-call invocations.
        # Spawn-per-call otherwise wipes a tmpfs /tmp between each command.
        self._session_dir = (
            get_sandbox_dir() / "nsjail" / f"{self._task_id}-{self._session_id}"
        )
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self.init_session()

    def get_temp_dir(self) -> str:
        # Paths in base.BaseEnvironment are computed against this. We bindmount
        # self._session_dir → /tmp inside the jail, so these paths are writable
        # from inside and readable from the host.
        return "/tmp"

    # ------------------------------------------------------------------
    # nsjail CLI construction
    # ------------------------------------------------------------------

    def _build_nsjail_args(self, timeout: int) -> list[str]:
        """Assemble the nsjail invocation flags for a single command run."""
        # Deadline passed to nsjail is given in whole seconds; give it a small
        # cushion over our own timeout so our Python-side poll wins the race
        # and we get a clean stdout drain instead of a kernel SIGKILL.
        nsjail_time_limit = max(int(timeout) + 5, 5)

        if self._config_file:
            # User-supplied config owns all policy. We still inject the
            # time limit so runaway commands can't outlive the tool call.
            return [
                self.executable,
                "--config", self._config_file,
                "--time_limit", str(nsjail_time_limit),
            ]

        rw_cwd = (
            self._initial_cwd if os.path.isdir(self._initial_cwd) else "/"
        )
        start_cwd = self.cwd if os.path.isdir(self.cwd) else rw_cwd

        args: list[str] = [
            self.executable,
            "-Mo",                    # once mode: single exec, no listener
            "--quiet",                # suppress nsjail's [I] chatter
            "--bindmount_ro", "/",
            "--bindmount", f"{self._session_dir}:/tmp",
            "--bindmount", rw_cwd,
            "--cwd", start_cwd,
            "--time_limit", str(nsjail_time_limit),
        ]

        if self._allow_net:
            args.append("--disable_clone_newnet")

        if self._memory > 0:
            args += ["--rlimit_as", str(self._memory)]
        if self._disk > 0:
            args += ["--rlimit_fsize", str(self._disk)]
        args += ["--rlimit_nofile", "1024"]

        # Build a sanitized env and forward the explicit subset into the jail.
        # Provider API keys and messaging-platform tokens are stripped by
        # _sanitize_subprocess_env before we even construct the --env flags.
        host_env = dict(os.environ)
        sanitized = _sanitize_subprocess_env(host_env, self.env)
        ensured_path = sanitized.get("PATH") or _SANE_PATH

        keep: dict[str, str] = {"PATH": ensured_path}
        for name in _ALWAYS_FORWARD_ENV:
            value = sanitized.get(name)
            if value is not None:
                keep[name] = value
        for name in self._forward_env:
            value = sanitized.get(name)
            if value is not None:
                keep[name] = value

        for key, value in keep.items():
            args += ["--env", f"{key}={value}"]

        return args

    # ------------------------------------------------------------------
    # Bash wrapper entrypoint
    # ------------------------------------------------------------------

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ) -> subprocess.Popen:
        nsjail_args = self._build_nsjail_args(timeout)
        bash_argv = ["/bin/bash"]
        if login:
            bash_argv.append("-l")
        bash_argv += ["-c", cmd_string]

        full_args = nsjail_args + ["--"] + bash_argv

        proc = subprocess.Popen(
            full_args,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        )
        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)
        return proc

    # ------------------------------------------------------------------
    # CWD extraction — the cwd marker file lives in the host-side session
    # dir thanks to the /tmp bindmount, so we can read it without a round
    # trip through the sandbox.
    # ------------------------------------------------------------------

    def _update_cwd(self, result: dict):
        host_cwd_file = self._session_dir / os.path.basename(self._cwd_file)
        try:
            cwd_path = host_cwd_file.read_text().strip()
            if cwd_path:
                self.cwd = cwd_path
        except (OSError, FileNotFoundError):
            pass
        self._extract_cwd_from_output(result)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        session_dir = getattr(self, "_session_dir", None)
        if session_dir is not None:
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except OSError:
                pass
