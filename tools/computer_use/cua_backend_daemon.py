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

import json

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
        self._mcp_args: list[str] = list(_driver._CUA_DRIVER_ARGS)
        self._process: Any = None
        self._owns_runtime = self._running = False
        # True once start() resolved a Docker/ssh/apptainer-sandboxed Bot Desktop placement and routed
        # the daemon's serve process (and every probe against it) through the sandbox's exec prefix
        # instead of spawning the host binary directly (#t_39df9245: the daemon used to be spawned on
        # the host UNCONDITIONALLY for any non-standard permission mode, even when the screen it was
        # supposed to drive lived inside a sandbox -- every call hit a nonexistent host DISPLAY).
        self._sandboxed = False
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._socket_token = uuid.uuid4().hex[:12]
        self.socket_path = self._host_socket_path()

    def _host_socket_path(self) -> str:
        """Socket path on THIS (host/gateway) process's own filesystem."""
        return (rf"\\.\pipe\hermes-cua-{self._socket_token}" if sys.platform == "win32"
                else os.path.join(tempfile.gettempdir(), f"hc-{self._socket_token}.sock"))

    def _sandbox_socket_path(self) -> str:
        """Socket path INSIDE the sandbox container's own ``/tmp`` -- never derived from this (host)
        process's ``tempfile.gettempdir()`` (#t_39df9245: a host ``TMPDIR`` override pointing at a
        per-agent scratch directory does not exist inside the container, so the sandboxed
        ``cua-driver serve`` failed with \"Permission denied\" trying to bind a socket under a path
        that was never mounted there)."""
        return f"/tmp/hc-{self._socket_token}.sock"  # no-tmp: ok -- the SANDBOX container's own /tmp, deliberately never this host's TMPDIR/get_scratch_dir()

    def _permission_env(self) -> dict[str, str]:
        """``CUA_DRIVER_PERMISSION_MODE`` / the bypass flag this daemon's process needs, wherever it runs."""
        env = {"CUA_DRIVER_PERMISSION_MODE": self.permission_mode}
        if self.permission_mode == "unrestricted":
            env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] = "1"
        return env

    def child_env(self) -> dict[str, str]:
        """Env for the long-lived MCP proxy connecting to this daemon's socket. A sandboxed daemon's own
        env (DISPLAY, permission mode, ...) is exported inside the sandbox's exec script
        (``_sandbox_invocation``), so the proxy subprocess on THIS (host/gateway) process needs only
        PATH -- the full host env would be meaningless for a proxy that never touches this process's
        environment."""
        if self._sandboxed:
            return {"PATH": os.environ.get("PATH", "")}
        return {**_cb().cua_driver_child_env(), **self._permission_env()}

    def _sandbox_invocation(self, argv: list[str], *, interactive: bool
                            ) -> Optional[tuple[tuple[str, list[str]], dict[str, str]]]:
        """Route *argv* through the terminal backend's sandbox (see ``cua_backend.sandbox_serve_invocation``
        / ``sandbox_cli_invocation``); None when the Bot Desktop is gateway-hosted, where this daemon
        spawns the host binary directly."""
        if interactive:
            return _cb().sandbox_serve_invocation(argv, extra_env=self._permission_env())
        return _cb().sandbox_cli_invocation(argv)

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
        if self.manifest_applies:
            serve_args += ["--capability-manifest", str(self.capability_manifest), "--approve-capability-manifest"]
        if sandboxed:
            # No host binary to probe for --no-overlay support (the probe runs `<driver_cmd> --help`
            # locally); the sandboxed MCP path (cua_backend.sandbox_mcp_invocation) makes the same
            # unconditional call for the same reason.
            return [*serve_args, "--no-overlay"]
        # The private daemon owns the cursor overlay, so the overlay policy must apply to this long-lived serve
        # process, not only its MCP proxy. Appended BEFORE the macOS app-launch wrapping so the flag travels inside
        # `open ... --args` with the rest of the serve args.
        return _driver._mcp_args_with_overlay_flag(serve_args, driver_cmd=self._command)

    def start(self) -> None:
        if self._running:
            return
        if self.permission_mode == "bounded":
            from tools.bot_desktop import placement, runtime as _bd_runtime
            if _bd_runtime.tool_placement() != placement.GATEWAY:
                # computer_use.capability_manifest is a HOST filesystem path; nothing mounts it into the
                # sandbox, so a bounded daemon there would either fail driver-side on a missing file or,
                # worse, silently validate against one the sandbox happens to have at that path. Fail
                # closed instead of guessing (unrestricted has no such file and is handled below).
                raise RuntimeError(
                    "computer_use.permission_mode: bounded is not yet supported with a Docker/ssh/"
                    "apptainer-sandboxed Bot Desktop (the capability manifest is a host file path the "
                    "sandbox cannot see). Use permission_mode: standard, or run the Bot Desktop on the "
                    "gateway host (bot_desktop.placement: gateway).")
        self.socket_path = self._sandbox_socket_path()  # tentative; _serve_args(sandboxed=True) reads it
        sandboxed_invocation = self._sandbox_invocation(["cua-driver", *self._serve_args(sandboxed=True)], interactive=True)
        if sandboxed_invocation is not None:
            self._sandboxed = True
            (command, args), env = sandboxed_invocation
            spawn_argv = [command, *args]
        else:
            self._sandboxed = False
            self.socket_path = self._host_socket_path()
            # Keep an explicit command fixed, but reselect managed drivers on every
            # launch: PM may have acquired a new pin since construction or last stop.
            driver_cmd = self._driver_cmd or _driver.resolve_cua_driver_cmd()
            if not driver_cmd:
                raise RuntimeError(_driver.cua_driver_install_hint())
            self._command, self._mcp_args = _driver._resolve_mcp_invocation(driver_cmd)
            env = self._sanitized_env()
            spawn_argv = _embedded_daemon_spawn_command(self._command, self._serve_args(), platform=sys.platform)
        self._process = subprocess.Popen(spawn_argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                         env=env)
        self._owns_runtime = True
        threading.Thread(target=self._drain_stderr, args=(self._process,), name="hermes-cua-daemon-stderr", daemon=True).start()
        deadline = time.monotonic() + self._START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            # `open` exits 0 once LaunchServices took the request: on macOS only a non-zero exit means the daemon died.
            if return_code is not None and (sys.platform != "darwin" or return_code != 0):
                self._startup_failure("embedded cua-driver exited during startup", "no diagnostic output")
            if self._socket_ready():
                self._running = True
                return
            time.sleep(0.1)
        self.stop()
        self._startup_failure("embedded cua-driver startup timed out", "daemon did not become ready")

    def _startup_failure(self, what: str, fallback: str) -> None:
        raise RuntimeError(f"{what}: {'; '.join(self._stderr_tail) or fallback}")

    def _socket_ready(self) -> bool:
        """``cua-driver status --socket`` exits 0 once the private daemon accepts connections -- routed
        through the SAME transport (sandbox exec or host subprocess) the daemon itself was spawned with,
        since the socket lives on that filesystem."""
        if self._sandboxed:
            probe = self._sandbox_invocation(["cua-driver", "status", "--socket", self.socket_path], interactive=False)
            if probe is None:
                return False
            (command, args), env = probe
            result = _cb()._run_quiet([command, *args], timeout=5.0, env=env, swallow=_QUIET_ERRORS)
            return result is not None and result.returncode == 0
        result = _cb()._run_quiet([self._command, "status", "--socket", self.socket_path], timeout=2.0,
                                  env=self._sanitized_env(), swallow=_QUIET_ERRORS)
        return result is not None and result.returncode == 0

    def proxy_invocation(self) -> tuple[str, list[str]]:
        if not self._running:
            raise RuntimeError("embedded cua-driver daemon is not running")
        if self._sandboxed:
            invocation = self._sandbox_invocation(["cua-driver", "mcp", "--embedded", "--socket", self.socket_path],
                                                  interactive=True)
            if invocation is None:
                raise RuntimeError("the terminal backend's sandbox is gone; start it again")
            (command, args), _env = invocation
            return command, args
        return self._command, [*self._mcp_args, "--embedded", "--socket", self.socket_path]

    def call_invocation(self, name: str, call_args: dict[str, Any]) -> tuple[list[str], dict[str, str], Optional[str]]:
        """``(cmd, env, shot_file)`` for a one-shot ``cua-driver call`` against this daemon's socket --
        routed through the sandbox exec prefix when the daemon is sandboxed (``shot_file`` always None
        there: a HOST temp path the sandboxed driver process could never see), else the bare host
        binary with the ``get_window_state`` ``screenshot_out_file`` optimization."""
        call_args = dict(call_args)
        if self._sandboxed:
            invocation = self._sandbox_invocation(
                ["cua-driver", "call", name, json.dumps(call_args), "--socket", self.socket_path], interactive=False)
            if invocation is None:
                raise RuntimeError("the terminal backend's sandbox is gone; start it again")
            (command, args), env = invocation
            return [command, *args], env, None
        shot_file: Optional[str] = None
        if name == "get_window_state" and "screenshot_out_file" not in call_args:
            fd, shot_file = tempfile.mkstemp(prefix="cua_shot_", suffix=".png")
            os.close(fd)
            call_args["screenshot_out_file"] = shot_file
        return [self._command, "call", name, json.dumps(call_args), "--socket", self.socket_path], self.child_env(), shot_file

    def stop(self) -> None:
        process, self._process = self._process, None
        owns_runtime, self._owns_runtime, self._running = self._owns_runtime, False, False
        if owns_runtime:
            if self._sandboxed:
                with contextlib.suppress(Exception):
                    probe = self._sandbox_invocation(["cua-driver", "stop", "--socket", self.socket_path], interactive=False)
                    if probe is not None:
                        (command, args), env = probe
                        _cb()._run_quiet([command, *args], timeout=5.0, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, env=env, swallow=_QUIET_ERRORS)
            else:
                _cb()._run_quiet([self._command, "stop", "--socket", self.socket_path], timeout=3.0, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, env=self._sanitized_env(), swallow=_QUIET_ERRORS)
        if process is not None:
            _wait_or_kill(process)
        # The sandboxed socket lives in the container's own filesystem namespace, not on this host.
        if sys.platform != "win32" and not self._sandboxed and os.path.exists(self.socket_path):
            with contextlib.suppress(OSError):
                os.remove(self.socket_path)
