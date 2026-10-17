"""Private embedded cua-driver daemon for non-standard permission modes, plus the macOS CuaDriver.app identity
checks its launch path depends on. Config/policy helpers are looked up lazily through the facade."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use import cua_backend_driver as _driver
from tools.computer_use.permissions import CUA_DRIVER_BUNDLE_ID

logger = logging.getLogger("tools.computer_use.cua_backend")

# The only bundle identity the private daemon may launch through, and the teams that sign official
# releases. Exact matches only: a suffixed identifier or other team is an impostor.
_CUA_DRIVER_BUNDLE_ID = CUA_DRIVER_BUNDLE_ID
_CUA_DRIVER_TEAM_IDS = ("4YEC26S9KF", "YCK386LBJ7")
_QUIET_ERRORS = (OSError, subprocess.SubprocessError)

def _cb():
    """Facade module (config/policy helpers), looked up lazily to avoid the import cycle."""
    from tools.computer_use import cua_backend
    return cua_backend

def _resolve_cua_driver_app_path(driver_cmd: str) -> Optional[str]:
    """Return the CuaDriver.app bundle that CARRIES *driver_cmd*, if any. Derived from the resolved binary path
    only — no /Applications fallback, which could be a DIFFERENT install than the one the manifest resolved,
    running code the resolution chain never validated."""
    head, marker, _ = os.path.realpath(driver_cmd).partition(".app/Contents/MacOS/")
    executable = os.path.join(head + ".app", "Contents", "MacOS", "cua-driver")
    return head + ".app" if marker and os.path.isfile(executable) and os.access(executable, os.X_OK) else None

def _validate_cua_driver_app_signature(app_path: str) -> None:
    """Fail closed unless *app_path* is the genuinely-signed CuaDriver.app. ``/usr/bin/open`` hands LaunchServices
    whatever bundle sits at the path, so ``codesign -dv`` must report EXACTLY ``Identifier=com.trycua.driver`` and
    an expected TeamIdentifier. ``TeamIdentifier=not set`` (ad-hoc dev builds) is allowed only with
    ``computer_use.allow_unsigned_driver: true``. Raises RuntimeError on any mismatch or when codesign is
    unavailable/fails."""
    codesign = shutil.which("codesign")
    if not codesign:
        raise RuntimeError("codesign is required to verify CuaDriver.app before launching it.")
    try:
        proc = _cb()._run_quiet([codesign, "-dv", app_path], timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not verify CuaDriver.app signature: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"CuaDriver.app at {app_path} is not code-signed; refusing to launch it ({(proc.stderr or '').strip()})")
    parts = [line.partition("=") for line in (proc.stderr or "").splitlines()]  # codesign -dv reports on stderr
    fields = {k.strip(): v.strip() for k, sep, v in reversed(parts) if sep}  # first occurrence of a key wins
    identifier, team = fields.get("Identifier", ""), fields.get("TeamIdentifier", "")
    if identifier != _CUA_DRIVER_BUNDLE_ID:
        raise RuntimeError(f"CuaDriver.app at {app_path} has identifier {identifier!r}, expected {_CUA_DRIVER_BUNDLE_ID!r}; "
                           "refusing to launch it.")
    if team in _CUA_DRIVER_TEAM_IDS or (team in ("", "not set") and _cb()._computer_use_cfg().get("allow_unsigned_driver") is True):
        return
    raise RuntimeError(f"CuaDriver.app at {app_path} is signed by team {team!r}, expected one of {_CUA_DRIVER_TEAM_IDS!r}; "
                       "refusing to launch it. (Set computer_use.allow_unsigned_driver: true in config.yaml only for "
                       "local unsigned driver builds.)")

def _embedded_daemon_spawn_command(driver_cmd: str, serve_args: list[str], *, platform: str,
                                   app_path: Optional[str] = None) -> list[str]:
    """Build the private-daemon launch while preserving macOS TCC identity."""
    if platform != "darwin":
        return [driver_cmd, *serve_args]
    resolved_app = app_path or _resolve_cua_driver_app_path(driver_cmd)
    if not resolved_app:
        raise RuntimeError("CuaDriver.app is required for private computer-use sessions on macOS. Run `hermes computer-use install` to restore it.")
    _validate_cua_driver_app_signature(resolved_app)
    return ["/usr/bin/open", "-n", "-g", "-a", resolved_app, "--args", *serve_args]

def _socket_on_windows_side(command: Optional[str]) -> bool:
    """Whether the processes that will receive ``--socket`` (``serve``/``status``/``mcp``/``stop`` are all the
    same driver) run on Windows and thus need a named pipe rather than a Unix socket file.

    Native Windows always does. A POSIX host still can, because WSL runs a Windows driver through interop
    (``/mnt/c/.../cua-driver.exe``, or the ``C:\\...`` its manifest reports); such a process cannot open a
    ``.sock`` and answers ``os error 123`` instead (NousResearch/hermes-agent#80227)."""
    return sys.platform == "win32" or _driver.cua_driver_command_is_windows(command)

_WINDOWS_PIPE_PREFIX = "\\\\.\\pipe\\"

def _windows_pipe_path(token: str) -> str:
    return f"{_WINDOWS_PIPE_PREFIX}hermes-cua-{token}"

def _unix_socket_path(token: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"hc-{token}.sock")

def _wait_or_kill(process: Any) -> None:
    """Wait 5s for a graceful exit, then terminate (2s), then kill."""
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


class _EmbeddedCuaDaemon:
    """Private daemon for a non-standard permission mode. cua-driver's permission mode is immutable after daemon
    startup, so reusing the machine-wide daemon would let one Hermes session's YOLO choice affect another. A
    private daemon gives the session its own socket, runtime and launch-time authorization; on macOS it is
    launched through CuaDriver.app so TCC stays attached to ``com.trycua.driver``. ``unrestricted`` = explicit
    Hermes YOLO (``--dangerously-bypass-approvals``); ``bounded`` = a user-reviewed capability manifest approved
    at launch is the authorization boundary, not a runtime prompt. The manifest is a ceiling, not a mode: it "can
    narrow a profile but never widen it", so a configured v3 manifest is forwarded even for ``unrestricted``
    (bounding an approval-bypassed run). Mandatory for ``bounded``, optional everywhere else."""

    _START_TIMEOUT_SECONDS = 15.0
    # A driver on the far side of the WSL interop boundary is started as a fresh Windows process per call:
    # measured on the reporting host at 1.9-2.8s for a `status --socket` probe and 4.1-5.3s for `serve` to
    # answer at all. The native budgets (2.0s probe, 3.0s stop, 15.0s start) are sized for a local process
    # and would report a healthy daemon as unreachable.
    _CROSS_OS_CALL_TIMEOUT_SECONDS = 15.0
    _CROSS_OS_START_TIMEOUT_SECONDS = 60.0

    def __init__(self, driver_cmd: str, permission_mode: str, capability_manifest: Optional[str] = None) -> None:
        if permission_mode not in {"unrestricted", "bounded"}:
            raise ValueError("embedded permission override supports unrestricted or bounded only")
        manifest = str(capability_manifest or "").strip()
        if not manifest and permission_mode == "bounded":
            raise ValueError("bounded permission mode requires computer_use.capability_manifest")
        manifest = os.path.abspath(os.path.expanduser(manifest)) if manifest else ""
        if manifest and not os.path.isfile(manifest):
            raise ValueError(f"capability manifest not found: {manifest}")
        self.capability_manifest: Optional[str] = manifest or None
        # bounded always forwards (driver validates it); other modes accept only v3 — a legacy manifest aborts startup.
        self.manifest_applies = bool(manifest) and (
            permission_mode == "bounded" or _cb()._manifest_is_mode_independent(manifest))
        if manifest and not self.manifest_applies:
            logger.warning("computer_use.capability_manifest is a legacy (v1/v2) manifest, which cua-driver only accepts in "
                           "bounded mode — it will NOT bound this %s session. Migrate the manifest to version 3 to keep a "
                           "ceiling on approval-bypassed runs.", permission_mode)
        self.permission_mode, self._driver_cmd, self._command = permission_mode, driver_cmd, driver_cmd
        self._mcp_args: list[str] = list(_driver._CUA_DRIVER_ARGS)
        self._process: Any = None
        self._owns_runtime = self._running = False
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._socket_token = uuid.uuid4().hex[:12]
        self.socket_path = self._socket_path_for(_socket_on_windows_side(driver_cmd))

    def _socket_path_for(self, windows_side: bool) -> str:
        """The private daemon's socket, named for the OS that has to open it: a named pipe on Windows, a Unix
        socket otherwise. ``--socket`` is only ever handed to the driver's own processes, so the transport
        follows the driver's OS rather than the host's — for a Windows driver under WSL that is the difference
        between a working daemon and ``os error 123``. ``start`` re-derives this once the command is resolved."""
        return _windows_pipe_path(self._socket_token) if windows_side else _unix_socket_path(self._socket_token)

    def _driver_via_interop(self) -> bool:
        """True when the driver runs on Windows while Hermes does not (WSL interop): every driver call then
        crosses an OS boundary as a fresh Windows process launch, so its latency is not a local one."""
        return sys.platform != "win32" and _socket_on_windows_side(self._command)

    def _driver_call_timeout(self, native: float) -> float:
        """Budget for one driver call (readiness probe, graceful stop): the native value for a local process."""
        return self._CROSS_OS_CALL_TIMEOUT_SECONDS if self._driver_via_interop() else native

    def _start_timeout(self) -> float:
        return self._CROSS_OS_START_TIMEOUT_SECONDS if self._driver_via_interop() else self._START_TIMEOUT_SECONDS

    def child_env(self) -> dict[str, str]:
        env = {**_cb().cua_driver_child_env(), "CUA_DRIVER_PERMISSION_MODE": self.permission_mode}
        if self.permission_mode == "unrestricted":
            env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] = "1"
        return env

    def _sanitized_env(self) -> dict[str, str]:
        from tools.environments.local import _sanitize_subprocess_env
        return _sanitize_subprocess_env(self.child_env())

    def _drain_stderr(self, process: Any) -> None:
        with contextlib.suppress(Exception):
            for line in getattr(process, "stderr", None) or ():
                text = str(line).strip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug("embedded cua-driver: %s", text)

    def _serve_args(self) -> list[str]:
        serve_args = ["serve", "--embedded", "--socket", self.socket_path, "--no-permissions-gate", "--permission-mode",
                      self.permission_mode, *(["--dangerously-bypass-approvals"] if self.permission_mode == "unrestricted" else [])]
        if self.manifest_applies:
            serve_args += ["--capability-manifest", str(self.capability_manifest), "--approve-capability-manifest"]
        # The private daemon owns the cursor overlay, so the overlay policy must apply to this long-lived serve
        # process, not only its MCP proxy. Appended BEFORE the macOS app-launch wrapping so the flag travels inside
        # `open ... --args` with the rest of the serve args.
        return _driver._mcp_args_with_overlay_flag(serve_args, driver_cmd=self._command)

    def start(self) -> None:
        if self._running:
            return
        # Keep an explicit command fixed, but reselect managed drivers on every
        # launch: PM may have acquired a new pin since construction or last stop.
        driver_cmd = self._driver_cmd or _driver.resolve_cua_driver_cmd()
        if not driver_cmd:
            raise RuntimeError(_driver.cua_driver_install_hint())
        self._command, self._mcp_args = _driver._resolve_mcp_invocation(driver_cmd)
        # Only now is the driver's OS known: an empty override resolves to PM's binary, and a Windows manifest
        # command replaces the POSIX path we started from. The socket must match the process that receives it.
        self.socket_path = self._socket_path_for(_socket_on_windows_side(self._command))
        env = self._sanitized_env()
        command = _embedded_daemon_spawn_command(self._command, self._serve_args(), platform=sys.platform)
        self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                         env=env)
        self._owns_runtime = True
        threading.Thread(target=self._drain_stderr, args=(self._process,), name="hermes-cua-daemon-stderr", daemon=True).start()
        deadline = time.monotonic() + self._start_timeout()
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            # `open` exits 0 once LaunchServices took the request: on macOS only a non-zero exit means the daemon died.
            if return_code is not None and (sys.platform != "darwin" or return_code != 0):
                # Release the failed launch exactly like the timeout path below, so a startup failure never
                # leaves the dead handle or the runtime ownership (and the socket it implies) behind.
                self.stop()
                self._startup_failure("embedded cua-driver exited during startup", "no diagnostic output")
            if self._socket_ready(env):
                self._running = True
                return
            time.sleep(0.1)
        self.stop()
        self._startup_failure("embedded cua-driver startup timed out", "daemon did not become ready")

    def _startup_failure(self, what: str, fallback: str) -> None:
        raise RuntimeError(f"{what}: {'; '.join(self._stderr_tail) or fallback}")

    def _socket_ready(self, env: dict[str, str]) -> bool:
        """``cua-driver status --socket`` exits 0 once the private daemon accepts connections."""
        probe = _cb()._run_quiet([self._command, "status", "--socket", self.socket_path],
                                 timeout=self._driver_call_timeout(2.0), env=env, swallow=_QUIET_ERRORS)
        return probe is not None and probe.returncode == 0

    def proxy_invocation(self) -> tuple[str, list[str]]:
        if not self._running:
            raise RuntimeError("embedded cua-driver daemon is not running")
        return self._command, [*self._mcp_args, "--embedded", "--socket", self.socket_path]

    def stop(self) -> None:
        process, self._process = self._process, None
        owns_runtime, self._owns_runtime, self._running = self._owns_runtime, False, False
        if owns_runtime:
            _cb()._run_quiet([self._command, "stop", "--socket", self.socket_path], timeout=self._driver_call_timeout(3.0),
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, env=self._sanitized_env(), swallow=_QUIET_ERRORS)
        if process is not None:
            _wait_or_kill(process)
        # A named pipe belongs to the driver that created it; only a filesystem socket is ours to unlink.
        if (sys.platform != "win32" and not self.socket_path.startswith(_WINDOWS_PIPE_PREFIX)
                and os.path.exists(self.socket_path)):
            with contextlib.suppress(OSError):
                os.remove(self.socket_path)
