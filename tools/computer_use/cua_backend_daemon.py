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
        self._mcp_args: List[str] = list(_driver._CUA_DRIVER_ARGS)
        self._process: Any = None
        self._owns_runtime = self._running = False
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._registry_session_id: Optional[str] = None
        token = uuid.uuid4().hex[:12]
        self.socket_path = (rf"\\.\pipe\hermes-cua-{token}" if sys.platform == "win32"
                            else os.path.join(tempfile.gettempdir(), f"hc-{token}.sock"))

    def child_env(self) -> Dict[str, str]:
        env = {**_cb().cua_driver_child_env(), "CUA_DRIVER_PERMISSION_MODE": self.permission_mode}
        if self.permission_mode == "unrestricted":
            env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] = "1"
        return env

    def _sanitized_env(self) -> Dict[str, str]:
        from tools.environments.local import _sanitize_subprocess_env
        return _sanitize_subprocess_env(self.child_env())

    def _drain_stdout(self, process: Any) -> None:
        """Fallback stdout drainer for when process_registry adoption is skipped or fails.
        stdout is a real pipe (see start()), so someone MUST keep reading it or a chatty
        daemon blocks once the pipe buffer fills."""
        with contextlib.suppress(Exception):
            for _ in getattr(process, "stdout", None) or ():
                pass

    def _drain_stderr(self, process: Any) -> None:
        with contextlib.suppress(Exception):
            for line in getattr(process, "stderr", None) or ():
                text = str(line).strip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug("embedded cua-driver: %s", text)

    def _serve_args(self) -> List[str]:
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
        self._driver_cmd = self._driver_cmd or _driver.resolve_cua_driver_cmd() or ""
        if not self._driver_cmd:
            raise RuntimeError(_driver.cua_driver_install_hint())
        self._command, self._mcp_args = _driver._resolve_mcp_invocation(self._driver_cmd)
        env = self._sanitized_env()
        command = _embedded_daemon_spawn_command(self._command, self._serve_args(), platform=sys.platform)
        # stdout is a real pipe (not DEVNULL) so process_registry's adopted reader thread can follow
        # this long-lived daemon to its REAL exit: a DEVNULL fd reads EOF immediately, which made the
        # registry's reader thread finish right away and mark the still-running daemon "exited" after
        # its wait(timeout=5) grace lapsed (observed empirically — see #104953 card). The reader drains
        # this pipe continuously, so a chatty daemon can never block on a full stdout buffer either.
        self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, env=env)
        self._owns_runtime = True
        threading.Thread(target=self._drain_stderr, args=(self._process,), name="hermes-cua-daemon-stderr", daemon=True).start()
        # Register with process_registry (#104953): this cua-driver subprocess previously had zero
        # references to the registry, so a hard crash/SIGKILL of the parent Hermes process (atexit
        # below never runs) left it running forever, invisible to `/stop`, `process list`, and crash
        # recovery. adopt_local() takes over the already-started Popen: it persists pid+host_start_time
        # to the crash-recovery checkpoint (tools/process_registry.py:_write_checkpoint), so a restarted
        # gateway's `recover_from_checkpoint()` re-discovers and can kill it, and `kill_all()` at normal
        # shutdown now tree-kills it as an extra safety net alongside the existing atexit hook. It
        # inherits this worker's own systemd-scope MemoryMax cgroup cap (subprocess children share their
        # parent's cgroup — confirmed empirically, see #104953) rather than getting a second cap of its
        # own. notify_on_complete=False: nothing should chat-notify when an internal daemon exits.
        # Skipped on macOS: there ``self._process`` is ``/usr/bin/open`` (see
        # _embedded_daemon_spawn_command), a short-lived launch helper that exits once it hands the
        # request to LaunchServices, NOT the actual cua-driver daemon — adopting it would checkpoint
        # and later "recover" the wrong, already-exited PID while the real daemon keeps running
        # untracked (coderabbit finding, #104953). Registering the true daemon PID on macOS needs its
        # own resolution path (e.g. from CuaDriver.app's process list); out of scope here.
        # isinstance-gated: test suites patch subprocess.Popen with a Mock() (no real stdout pipe/fd,
        # and even replace the ``subprocess.Popen`` class itself), so isinstance() and the adoption both
        # go inside the broad suppress — a mocked process must never spin a real reader thread against
        # fake stream attributes, but a mocking failure here must never break daemon startup either.
        registered = False
        if sys.platform != "darwin":
            with contextlib.suppress(Exception):
                if isinstance(self._process, subprocess.Popen):
                    from tools.process_registry import process_registry
                    session = process_registry.adopt_local(
                        self._process, command=" ".join(command), cwd=None, notify_on_complete=False)
                    self._registry_session_id = session.id
                    registered = True
        if not registered:
            # Registry adoption owns draining stdout via its reader thread; without it (macOS, or a
            # failed/suppressed adoption above) something must still drain the real pipe or a chatty
            # daemon blocks once the OS pipe buffer fills.
            threading.Thread(target=self._drain_stdout, args=(self._process,),
                             name="hermes-cua-daemon-stdout", daemon=True).start()
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
        self.stop()
        self._startup_failure("embedded cua-driver startup timed out", "daemon did not become ready")

    def _startup_failure(self, what: str, fallback: str) -> None:
        raise RuntimeError(f"{what}: {'; '.join(self._stderr_tail) or fallback}")

    def _socket_ready(self, env: Dict[str, str]) -> bool:
        """``cua-driver status --socket`` exits 0 once the private daemon accepts connections."""
        probe = _cb()._run_quiet([self._command, "status", "--socket", self.socket_path], timeout=2.0, env=env, swallow=_QUIET_ERRORS)
        return probe is not None and probe.returncode == 0

    def proxy_invocation(self) -> Tuple[str, List[str]]:
        if not self._running:
            raise RuntimeError("embedded cua-driver daemon is not running")
        return self._command, [*self._mcp_args, "--embedded", "--socket", self.socket_path]

    def stop(self) -> None:
        process, self._process = self._process, None
        owns_runtime, self._owns_runtime, self._running = self._owns_runtime, False, False
        if owns_runtime:
            _cb()._run_quiet([self._command, "stop", "--socket", self.socket_path], timeout=3.0, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, env=self._sanitized_env(), swallow=_QUIET_ERRORS)
        # Deregister from process_registry FIRST (#104953): `cua-driver stop` above already asked the
        # daemon to exit gracefully, so by the time _wait_or_kill's tree-kill fallback would fire the
        # registry's own reader thread may already have observed EOF and moved the session to
        # "finished" on its own -- kill_process() on an already-exited session is a no-op status
        # lookup, never an error. Never let a stale registry entry outlive the Popen this class still
        # owns and is about to _wait_or_kill directly.
        if self._registry_session_id is not None:
            with contextlib.suppress(Exception):
                from tools.process_registry import process_registry
                process_registry.kill_process(self._registry_session_id, source="cua_backend_daemon.stop",
                                               consume_output=False)
            self._registry_session_id = None
        if process is not None:
            _wait_or_kill(process)
        if sys.platform != "win32" and os.path.exists(self.socket_path):
            with contextlib.suppress(OSError):
                os.remove(self.socket_path)
