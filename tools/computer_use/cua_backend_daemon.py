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
from hermes_cli.process_identity import release_process_group_once_empty, spawn_owned

logger = logging.getLogger("tools.computer_use.cua_backend")

# The only bundle identity the private daemon may launch through, and the teams that sign official
# releases. Exact matches only: a suffixed identifier or other team is an impostor.
_CUA_DRIVER_BUNDLE_ID = CUA_DRIVER_BUNDLE_ID
_CUA_DRIVER_TEAM_IDS = ("4YEC26S9KF", "YCK386LBJ7")
_QUIET_ERRORS = (OSError, subprocess.SubprocessError)
# Inside a sandbox, killing the exec client does not end the daemon behind it. Hold the client's stdin open and stop
# the daemon when it closes, so a sandboxed daemon dies with the Hermes that started it, as a host one does.
# (A backgrounded command's stdin is /dev/null unless redirected, hence the explicit fd 3.)
_SANDBOX_DAEMON_SH = 'exec 3<&0; "$@" 3<&- & p=$!; (cat <&3 >/dev/null; kill $p 2>/dev/null) & wait $p'

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
    (bounding an approval-bypassed run). Mandatory for ``bounded``, optional everywhere else.

    Where it runs follows the Bot Desktop placement, never the permission mode: with a sandboxed terminal
    backend the daemon is spawned INSIDE the sandbox (next to the screen it drives) through the backend's exec
    prefix, so approvals off / ``--yolo`` widens what the driver may do in the sandbox without moving it to the
    host. Only a gateway-hosted desktop spawns the host binary."""

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
        self._mcp_args: list[str] = list(_driver._CUA_DRIVER_ARGS)
        self._process: Any = None
        self._owns_runtime = self._running = False
        self._sandboxed = False  # decided by start(): the daemon, its socket and every probe live in the sandbox
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._socket_token = uuid.uuid4().hex[:12]
        self.socket_path = self._host_socket_path()

    def _host_socket_path(self) -> str:
        return (rf"\\.\pipe\hermes-cua-{self._socket_token}" if sys.platform == "win32"
                else os.path.join(tempfile.gettempdir(), f"hc-{self._socket_token}.sock"))

    def _sandbox_socket_path(self) -> str:
        # The sandbox's own /tmp: a host TMPDIR (or the Hermes scratch dir) is not mounted there.
        return f"/tmp/hc-{self._socket_token}.sock"  # no-tmp: ok — sandbox-side path

    def _permission_env(self) -> dict[str, str]:
        env = {"CUA_DRIVER_PERMISSION_MODE": self.permission_mode}
        if self.permission_mode == "unrestricted":
            env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] = "1"
        return env

    def child_env(self) -> dict[str, str]:
        if self._sandboxed:  # the driver's env is exported inside the sandbox; the exec client needs only PATH
            return {"PATH": os.environ.get("PATH", "")}
        return {**_cb().cua_driver_child_env(), **self._permission_env()}

    def _in_sandbox(self, argv: list[str], *, interactive: bool, env: Optional[dict[str, str]] = None
                    ) -> tuple[list[str], dict[str, str]]:
        """``(host argv, host env)`` running ``cua-driver <argv>`` inside the sandbox this daemon lives in."""
        placed = _cb().sandbox_invocation(["cua-driver", *argv], interactive=interactive, extra_env=env)
        if placed is None:
            raise RuntimeError("the terminal backend's sandbox is gone; start the session again")
        (command, args), host_env = placed
        return [command, *args], host_env

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

    def _serve_args(self, *, sandboxed: bool = False) -> list[str]:
        serve_args = ["serve", "--embedded", "--socket", self.socket_path, "--no-permissions-gate", "--permission-mode",
                      self.permission_mode, *(["--dangerously-bypass-approvals"] if self.permission_mode == "unrestricted" else [])]
        if sandboxed:  # no host binary to probe; the sandboxed MCP server makes the same unconditional choice
            return [*serve_args, "--no-overlay"]
        if self.manifest_applies:
            serve_args += ["--capability-manifest", str(self.capability_manifest), "--approve-capability-manifest"]
        # The private daemon owns the cursor overlay, so the overlay policy must apply to this long-lived serve
        # process, not only its MCP proxy. Appended BEFORE the macOS app-launch wrapping so the flag travels inside
        # `open ... --args` with the rest of the serve args.
        return _driver._mcp_args_with_overlay_flag(serve_args, driver_cmd=self._command)

    def start(self) -> None:
        if self._running:
            return
        # Placement decides where it runs, not the permission mode (see the class docstring).
        self.socket_path = self._sandbox_socket_path()
        placed = _cb().sandbox_invocation(["sh", "-c", _SANDBOX_DAEMON_SH, "sh", "cua-driver", *self._serve_args(sandboxed=True)],
                                          interactive=True, extra_env=self._permission_env())
        self._sandboxed = placed is not None
        if placed is not None:
            if self.manifest_applies:
                # The manifest is a host path the sandbox cannot see; dropping it would silently widen the
                # session, so refuse instead of guessing.
                raise RuntimeError(
                    "computer_use.capability_manifest cannot be applied to a cua-driver running inside the "
                    "terminal backend's sandbox (the manifest is a host file). Remove the manifest and "
                    "permission_mode: bounded, or set bot_desktop.placement: gateway to run the driver on the "
                    "gateway host.")
            (command, args), env = placed
            command = [command, *args]
        else:
            self.socket_path = self._host_socket_path()
            # Keep an explicit command fixed, but reselect managed drivers on every
            # launch: PM may have acquired a new pin since construction or last stop.
            driver_cmd = self._driver_cmd or _driver.resolve_cua_driver_cmd()
            if not driver_cmd:
                raise RuntimeError(_driver.cua_driver_install_hint())
            self._command, self._mcp_args = _driver._resolve_mcp_invocation(driver_cmd)
            env = self._sanitized_env()
            command = _embedded_daemon_spawn_command(self._command, self._serve_args(), platform=sys.platform)
        stdin = subprocess.PIPE if self._sandboxed else subprocess.DEVNULL
        popen_kwargs: dict[str, Any] = dict(stdin=stdin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                            text=True, encoding="utf-8", errors="replace", env=env)
        if sys.platform == "darwin" and not self._sandboxed:
            # `open` hands the daemon to LaunchServices: it is launchd's child, never in our tree, so there is no
            # group of ours to own. The daemon's own --embedded lifetime is the only bound there.
            self._process = subprocess.Popen(command, **popen_kwargs)
        else:
            # Owned: the daemon dies with this process however it dies. It used to be stopped only by stop()
            # (session release / atexit), so a SIGKILLed or OOM-killed Hermes left a YOLO-mode daemon running.
            # (For a sandboxed daemon this owns the exec client; stop() ends the daemon inside the sandbox.)
            self._process = spawn_owned(command, purpose="cua-driver-embedded", **popen_kwargs)
        self._owns_runtime = True
        threading.Thread(target=self._drain_stderr, args=(self._process,), name="hermes-cua-daemon-stderr", daemon=True).start()
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

    def _socket_ready(self, env: dict[str, str]) -> bool:
        """``cua-driver status --socket`` exits 0 once the private daemon accepts connections."""
        if self._sandboxed:
            argv, env = self._in_sandbox(["status", "--socket", self.socket_path], interactive=False)
        else:
            argv = [self._command, "status", "--socket", self.socket_path]
        probe = _cb()._run_quiet(argv, timeout=5.0 if self._sandboxed else 2.0, env=env, swallow=_QUIET_ERRORS)
        return probe is not None and probe.returncode == 0

    def proxy_invocation(self) -> tuple[str, list[str]]:
        if not self._running:
            raise RuntimeError("embedded cua-driver daemon is not running")
        if self._sandboxed:
            argv, _ = self._in_sandbox(["mcp", "--embedded", "--socket", self.socket_path], interactive=True)
            return argv[0], argv[1:]
        return self._command, [*self._mcp_args, "--embedded", "--socket", self.socket_path]

    def stop(self) -> None:
        process, self._process = self._process, None
        owns_runtime, self._owns_runtime, self._running = self._owns_runtime, False, False
        if owns_runtime:
            if self._sandboxed:
                # Killing the exec client does not end the daemon inside the sandbox, so ask it to stop. The sandbox
                # may already be gone, and stop() must not raise.
                with contextlib.suppress(Exception):
                    argv, env = self._in_sandbox(["stop", "--socket", self.socket_path], interactive=False)
                    _cb()._run_quiet(argv, timeout=5.0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     env=env, swallow=_QUIET_ERRORS)
            else:
                _cb()._run_quiet([self._command, "stop", "--socket", self.socket_path], timeout=3.0, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, env=self._sanitized_env(), swallow=_QUIET_ERRORS)
        if process is not None:
            if self._sandboxed:
                process.stdin.close()  # the sandboxed daemon's lifetime tether (_SANDBOX_DAEMON_SH)
            _wait_or_kill(process)
            release_process_group_once_empty(process.pid, proc=process)  # stopped it ourselves (a no-op for the macOS `open` launcher)
        if sys.platform != "win32" and not self._sandboxed and os.path.exists(self.socket_path):
            with contextlib.suppress(OSError):
                os.remove(self.socket_path)
