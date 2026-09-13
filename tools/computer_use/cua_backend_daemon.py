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

logger = logging.getLogger("tools.computer_use.cua_backend")

# The only bundle identity the private daemon may launch through, and the teams that sign official
# releases. Exact matches only: a suffixed identifier or other team is an impostor.
_CUA_DRIVER_BUNDLE_ID = "com.trycua.driver"
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

def _embedded_daemon_spawn_command(driver_cmd: str, serve_args: List[str], *, platform: str,
                                   app_path: Optional[str] = None) -> List[str]:
    """Build the private-daemon launch while preserving macOS TCC identity."""
    if platform != "darwin":
        return [driver_cmd, *serve_args]
    resolved_app = app_path or _resolve_cua_driver_app_path(driver_cmd)
    if not resolved_app:
        raise RuntimeError("CuaDriver.app is required for private computer-use sessions on macOS. Run `hermes computer-use install` to restore it.")
    _validate_cua_driver_app_signature(resolved_app)
    return ["/usr/bin/open", "-n", "-g", "-a", resolved_app, "--args", *serve_args]

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

    def __init__(self, driver_cmd: str, permission_mode: str, capability_manifest: Optional[str] = None,
                 *, execution_context=None) -> None:
        self.execution_context = execution_context
        self.launch_context = execution_context.context.computer_use if execution_context else None
        private_standard = bool(self.launch_context and self.launch_context.private_daemon and permission_mode == "standard")
        if permission_mode not in {"unrestricted", "bounded"} and not private_standard:
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
        self._mcp_args: List[str] = list(_driver._CUA_DRIVER_ARGS)
        self._process: Any = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._launch_env = None
        self._stop_argv = None
        self._staged_manifest = None
        self._owns_runtime = self._running = False
        self._stderr_tail: deque[str] = deque(maxlen=20)
        token = uuid.uuid4().hex[:12]
        self.socket_path = (rf"\\.\pipe\hermes-cua-{token}" if sys.platform == "win32"
                            else os.path.join(tempfile.gettempdir(), f"hc-{token}.sock"))
        if self.launch_context and self.launch_context.runtime_dir:
            self.execution_context.check()
            self.socket_path = os.path.join(self.launch_context.runtime_dir, f"hc-{token}.sock")
            if len(os.fsencode(self.socket_path)) >= 104:
                raise ValueError("runtime_dir path is too long for a private Unix socket")

    def child_env(self) -> Dict[str, str]:
        env = {**_cb().cua_driver_child_env(**({"execution_context": self.execution_context} if self.execution_context else {})),
               "CUA_DRIVER_PERMISSION_MODE": self.permission_mode}
        if self.permission_mode == "unrestricted":
            env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] = "1"
        return env

    def _sanitized_env(self) -> Dict[str, str]:
        from tools.environments.local import _sanitize_subprocess_env
        if self.execution_context is not None:
            return _cb().sanitize_cua_child_env(self.child_env(), self.execution_context)
        return _sanitize_subprocess_env(self.child_env())

    def _drain_stderr(self, process: Any) -> None:
        stream = getattr(process, "stderr", None)
        if stream is None:
            return
        with contextlib.suppress(Exception), contextlib.closing(stream):
            for line in stream:
                text = str(line).strip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug("embedded cua-driver: %s", text)

    def _serve_args(self) -> List[str]:
        serve_args = ["serve", "--embedded", "--socket", self.socket_path, "--no-permissions-gate", "--permission-mode",
                      self.permission_mode, *(["--dangerously-bypass-approvals"] if self.permission_mode == "unrestricted" else [])]
        if self.manifest_applies:
            manifest = self.capability_manifest
            if self.launch_context and self.launch_context.runtime_dir:
                self.execution_context.check()
                if self._staged_manifest is None:
                    with open(manifest, "rb") as source:
                        contents = source.read()
                    fd, path = tempfile.mkstemp(prefix="hc-manifest-", suffix=".json", dir=self.launch_context.runtime_dir)
                    try:
                        with os.fdopen(fd, "wb") as destination:
                            destination.write(contents)
                    except BaseException:
                        os.unlink(path)
                        raise
                    self._staged_manifest = path
                manifest = self._staged_manifest
            serve_args += ["--capability-manifest", str(manifest), "--approve-capability-manifest"]
        # The private daemon owns the cursor overlay, so the overlay policy must apply to this long-lived serve
        # process, not only its MCP proxy. Appended BEFORE the macOS app-launch wrapping so the flag travels inside
        # `open ... --args` with the rest of the serve args.
        if self.launch_context and self.launch_context.theme:
            serve_args += ["--cursor-theme", self.launch_context.theme]
        return _driver._mcp_args_with_overlay_flag(serve_args, driver_cmd=self._command,
                    **({"execution_context": self.execution_context} if self.execution_context else {}))

    def start(self) -> None:
        if self._running:
            return
        self._driver_cmd = self._driver_cmd or _driver.resolve_cua_driver_cmd() or ""
        if not self._driver_cmd:
            raise RuntimeError(_driver.cua_driver_install_hint())
        self._command, self._mcp_args = _driver._resolve_mcp_invocation(self._driver_cmd,
                    **({"execution_context": self.execution_context} if self.execution_context else {}))
        env = self._sanitized_env()
        self._launch_env = env
        try:
            command = _embedded_daemon_spawn_command(self._command, self._serve_args(), platform=sys.platform)
            stop_argv = [self._command, "stop", "--socket", self.socket_path]
            if self.execution_context is not None:
                command = self.execution_context.wrap_argv(command, computer_use=True)
                stop_argv = self.execution_context.wrap_argv(stop_argv, computer_use=True)
            self._stop_argv = stop_argv
            self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.PIPE, text=True, env=env)
            self._owns_runtime = True
            self._stderr_thread = threading.Thread(target=self._drain_stderr, args=(self._process,),
                                                   name="hermes-cua-daemon-stderr", daemon=True)
            self._stderr_thread.start()
            deadline = time.monotonic() + self._START_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                return_code = self._process.poll()
                # `open` exits 0 once LaunchServices took the request: on macOS only a non-zero exit means the daemon died.
                if return_code is not None and (sys.platform != "darwin" or return_code != 0):
                    self._startup_failure("embedded cua-driver exited during startup", "no diagnostic output")
                if self._socket_ready(env):
                    self._running = True
                    return
                time.sleep(0.1)
            self._startup_failure("embedded cua-driver startup timed out", "daemon did not become ready")
        except BaseException:
            # Lease checks and thread launch can fail after we already own the child.
            try:
                self.stop()
            except Exception:
                logger.exception("failed to clean up embedded cua-driver startup")
            raise

    def _startup_failure(self, what: str, fallback: str) -> None:
        try:
            self.stop()  # reap and finish reading diagnostics before formatting the error
        except Exception:
            logger.exception("failed to clean up embedded cua-driver startup")
        raise RuntimeError(f"{what}: {'; '.join(self._stderr_tail) or fallback}")

    def _socket_ready(self, env: Dict[str, str]) -> bool:
        """``cua-driver status --socket`` exits 0 once the private daemon accepts connections."""
        argv = [self._command, "status", "--socket", self.socket_path]
        if self.execution_context is not None:
            argv = self.execution_context.wrap_argv(argv, computer_use=True)
        probe = _cb()._run_quiet(argv, timeout=2.0, env=env, swallow=_QUIET_ERRORS)
        return probe is not None and probe.returncode == 0

    def proxy_invocation(self) -> Tuple[str, List[str]]:
        if not self._running:
            raise RuntimeError("embedded cua-driver daemon is not running")
        return self._command, [*self._mcp_args, "--embedded", "--socket", self.socket_path]

    def stop(self) -> None:
        process, self._process = self._process, None
        owns_runtime, self._owns_runtime, self._running = self._owns_runtime, False, False
        try:
            if owns_runtime:
                _cb()._run_quiet(self._stop_argv or [self._command, "stop", "--socket", self.socket_path], timeout=3.0, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, env=self._launch_env or self._sanitized_env(), swallow=_QUIET_ERRORS)
        finally:
            try:
                if process is not None:
                    _wait_or_kill(process)
            finally:
                reader, self._stderr_thread = self._stderr_thread, None
                if reader is not None and reader.ident is not None:
                    # A descendant may retain stderr. Never block shutdown on its EOF, or
                    # close a TextIOWrapper concurrently with a reader holding its lock.
                    if reader is not threading.current_thread():
                        reader.join(timeout=2.0)
                elif process is not None and process.stderr is not None:
                    process.stderr.close()  # thread.start() failed; nobody else owns the pipe
                if sys.platform != "win32" and os.path.exists(self.socket_path):
                    with contextlib.suppress(OSError):
                        os.remove(self.socket_path)
                if self._staged_manifest is not None:
                    with contextlib.suppress(OSError):
                        os.remove(self._staged_manifest)
                    self._staged_manifest = None
