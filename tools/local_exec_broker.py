#!/usr/bin/env python3
"""Local execution broker for #59293. Linux/POSIX, stdlib only.

    python -m tools.local_exec_broker \
        --socket /run/hermes-broker/broker.sock --staging-root /run/hermes-worker/stage \
        --allow-uid 1003 --socket-mode 0666

The problem it exists to solve: the argv-only ``sudo -u`` carrier closed the same-UID policy
escape but broke ``execute_code``, because ``sudo`` is a *privilege* tool, not a *transport*.
Everything ``tools/code_kernel.py:_spawn`` hands its child today crosses the boundary
implicitly, and ``sudo`` drops all three:

  * the explicit ``env=`` dict         -> discarded under ``env_reset``
  * ``pass_fds=(death_r,)``            -> non-std descriptors are closed
  * the 0700 ``mkdtemp`` staging dir   -> the new uid cannot traverse it

So the broker owns child lifetime on the trusted side and transports each resource on a
channel that survives a uid switch:

  environment   the client sends the approved dict as JSON; the child env is built from it
                alone, never from the broker's own ``os.environ``.
  descriptors   the client passes open fds over ``SCM_RIGHTS``. The broker forwards them by
                number via ``pass_fds`` and publishes those numbers as ``HERMES_BROKER_FDS``,
                then closes its own copies so it never holds a peer's channel open.
  runner        the client passes its already-open regular runner as a distinct
                ``SCM_RIGHTS`` descriptor. A tiny interpreter bootstrap reads that descriptor
                with ``pread`` (not the caller's shared cursor) and compiles it with
                ``/proc/self/fd/<n>`` as the script identity, so no pathname or staging-directory
                traversal occurs. The legacy ``runner`` path request remains available for
                compatibility and is resolved inside the broker-owned staging root before being
                opened.
  lifetime      the client connection IS the lease. Its EOF — close, crash, SIGKILL — is what
                kills the child process group, the same signal shape as the inherited
                parent-death pipe, but owned by the broker rather than inherited through sudo.

Everything the child does not receive explicitly, it does not get. Legacy runner requests keep
stdin, stdout and stderr on ``DEVNULL``. The local-terminal argv request carries explicit stdin
and merged stdout/stderr descriptors, and nothing is inherited from the broker process.

**The request frame is the only untrusted surface, so it is the one that is bounded.** A
request is one newline-terminated JSON object of at most ``MAX_REQUEST_BYTES`` carrying at
most ``MAX_FDS`` child descriptors plus one runner descriptor, read under a handshake
timeout. Every refusal is a structured ``{"ok": false, "error", "message"}`` reply, and —
the invariant that matters more — every refusal closes the descriptors the kernel already
installed on the broker's behalf. A retained copy is not merely a leaked fd: it is the
peer's channel, and it keeps their pipe from ever reaching EOF.

Every accepted connection is authenticated with Linux ``SO_PEERCRED`` before a request is
read. ``--allow-uid`` is repeatable and defaults to the broker's effective uid; socket
publication defaults to 0600, while ``--socket-mode`` can explicitly publish 0660 (when the
service and client share the socket's group) or 0666 for an authorized cross-uid client.
Wider publication grants only reachability: the peer uid allowlist remains mandatory
authorization. The socket directory must be owned by the broker, not writable by group or
other, and at most traverse-only for unrelated users. The staging root must still be
broker-owned 0700 for legacy pathname requests. The client's ``env`` payload is passed to the
child verbatim because at this boundary it is the approved child environment.

Every allowlisted peer is intentionally authorized for full code execution as the broker
uid. The broker must therefore run as the dedicated, less-privileged worker uid, never as a
more privileged account. Direct runs own and clean up only the process group they create.
``--systemd-cgroup`` instead fails closed unless a user manager can create transient scopes,
then contains every command in its own scope so a descendant cannot escape lease teardown
with ``setsid()``. The ``--install-user-service`` operator action installs and starts that
contained mode under the calling user's systemd manager.

Integration seams:

  * ``tools/environments/local.py:_run_bash`` now opts in when
    ``terminal.local_exec_broker.socket`` and ``terminal.local_exec_broker.uid`` are configured.
    Its argv, cwd, scrubbed environment, stdin, and merged stdout/stderr cross explicitly;
    broker failure is fatal rather than a same-uid fallback.
  * ``tools/code_kernel.py:_spawn`` sends the open runner, selected project interpreter and cwd,
    scrubbed child environment, and explicit stdio. The returned connection replaces
    ``kernel.death_pipe_w`` as the liveness handle held by ``SessionKernel``.
  * Per-command systemd scopes are created on the broker side, where the trusted uid still has
    a user bus. The scope launcher gets trusted broker bus locators; an ``env`` wrapper restores
    the peer-approved locator values for the command, or removes only values the peer omitted.

This Linux-only runtime surface has its own behavior test
(``tests/scripts/test_local_exec_broker.py``).
"""

from __future__ import annotations

import argparse
import array
import contextlib
import errno
import fcntl
import json
import os
import selectors
import secrets
import signal
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import NamedTuple

# The user service owns the stable broker endpoint. Each command gets its own transient scope.
USER_SERVICE_NAME = "hermes-local-exec-broker.service"
USER_RUNTIME_DIRECTORY = "hermes-local-exec-broker"

# Child-visible fd numbers the broker forwarded on its behalf, comma separated.
FDS_ENV = "HERMES_BROKER_FDS"

# Caps on the one untrusted surface. These bound the TOTAL for a request, not a single recv.
MAX_FDS = 8
_MAX_STDIO_FDS = 3
# One runner, up to three stdio descriptors, plus MAX_FDS child descriptors.
MAX_RECEIVED_FDS = MAX_FDS + 1 + _MAX_STDIO_FDS
# Linux permits roughly 2 MiB across argv+env and expands non-ASCII JSON characters to as
# many as six bytes. Keep the protocol bounded while leaving room for every kernel-admissible
# launch plus its JSON structure. Replies remain tiny and keep their own tighter cap.
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_REPLY_BYTES = 65536
MAX_SOCKET_PATH_BYTES = 107
DEFAULT_HANDSHAKE_TIMEOUT = 10.0
DEFAULT_EXEC_TIMEOUT = 10.0
_SYSTEMD_LOCATORS = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")

_INT_SIZE = array.array("i").itemsize
_UCRED = struct.Struct("=iII")
_RECV_CHUNK = 65536
_TERM_GRACE_SECONDS = 2.0
_KILL_GRACE_SECONDS = 2.0
_SYSTEMCTL_TIMEOUT_SECONDS = 5.0
_PUBLISH_SUFFIXES = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_RUNNER_BOOTSTRAP = """\
import sys

# ``python -c`` prepends the process cwd (as "") to sys.path, but the direct spawn runs the
# runner as a script, so there sys.path[0] is the staging dir and the caller's project
# directory is never importable. Left in place, the injected entry lets a project file named
# for a stdlib module (json.py, os.py) shadow the real one for every cell the kernel runs.
# Drop it before importing anything but sys — which is builtin and cannot be shadowed. The
# staging dir reaches the child on PYTHONPATH, and the child's cwd is untouched, so user code
# still resolves relative paths against the project.
if sys.path and sys.path[0] == "":
    del sys.path[0]

import os

_runner_fd = int(sys.argv.pop())
_runner_path = sys.argv.pop()
sys.argv[:] = [_runner_path]
_runner_parts = []
_runner_offset = 0
while True:
    _runner_chunk = os.pread(_runner_fd, 65536, _runner_offset)
    if not _runner_chunk:
        break
    _runner_parts.append(_runner_chunk)
    _runner_offset += len(_runner_chunk)
_runner_code = compile(b"".join(_runner_parts), _runner_path, "exec")
exec(
    _runner_code,
    {
        "__name__": "__main__",
        "__file__": _runner_path,
        "__package__": None,
        "__cached__": None,
    },
)
"""
_ENV_EXEC_BOOTSTRAP = """\
import json
import os
import sys

_env_fd = int(sys.argv.pop(1))
_status_fd = int(sys.argv.pop(1))
_stderr_fd = int(sys.argv.pop(1))
with os.fdopen(_env_fd, encoding="utf-8") as _env_file:
    _child_env = json.load(_env_file)
_child_argv = sys.argv[1:]
os.write(_status_fd, b"ready\\n")
os.set_inheritable(_status_fd, False)
try:
    # The launcher's diagnostics have a private pipe. Restore the requested stderr stream only
    # for the requested program; fd 1 preserves the historical merged default.
    os.dup2(_stderr_fd, 2)
    if _stderr_fd not in (1, 2):
        os.close(_stderr_fd)
    os.execvpe(_child_argv[0], _child_argv, _child_env)
except OSError as _exc:
    _message = str(_exc).encode("utf-8", "replace").replace(b"\\n", b" ")
    os.write(_status_fd, b"error " + str(_exc.errno or 0).encode() + b" " + _message + b"\\n")
    raise SystemExit(127)
"""


class BrokerError(RuntimeError):
    """A refusal, carrying the same ``code`` on both sides of the socket.

    The broker raises it, replies with ``code``/``message``, and the client re-raises it from
    that reply — so a caller sees a typed failure it can branch on instead of whatever
    exception happens to fall out of parsing an empty read.
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class _SystemdContainment(NamedTuple):
    systemd_run: str
    env: str
    broker_env: dict[str, str]


def _resolve_systemd_containment() -> _SystemdContainment:
    """Resolve containment executables and trusted bus state once at startup."""
    systemd_run = shutil.which("systemd-run")
    env_binary = shutil.which("env")
    if systemd_run is None or env_binary is None:
        raise SystemExit("systemd cgroup containment executables are unavailable")
    broker_env = {
        name: os.environ[name] for name in _SYSTEMD_LOCATORS if name in os.environ
    }
    return _SystemdContainment(systemd_run, env_binary, broker_env)


def _systemd_quote(value: str) -> str:
    """Quote one literal argument in an ``ExecStart=`` directive."""
    if "\n" in value or "\r" in value:
        raise ValueError("ExecStart argument must not contain a line break")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def _systemd_working_directory(value: str) -> str:
    """Encode a path for ``WorkingDirectory=``, whose parser does not unquote it."""
    if "\n" in value or "\r" in value:
        raise ValueError("WorkingDirectory path must not contain a line break")
    return value.replace("%", "%%")


def _verified_import_root() -> str:
    """Return the directory from which this exact broker module imports as ``tools``."""
    module_path = Path(__file__).resolve()
    import_root = module_path.parent.parent
    expected = import_root / "tools" / "local_exec_broker.py"
    try:
        if not expected.samefile(module_path):
            raise OSError("module path does not match import root")
    except OSError as exc:
        raise SystemExit(f"could not verify broker import root: {import_root}") from exc
    return str(import_root)


def _render_user_service(
    *,
    python_path: str,
    import_root: str,
    allowed_uids: frozenset[int] | None = None,
    socket_mode: int = 0o600,
) -> str:
    """Render the operator-owned user unit that enables mandatory containment."""
    if allowed_uids is None:
        allowed_uids = frozenset({os.geteuid()})
    socket_mode = _validated_socket_mode(socket_mode)
    authorization_args = tuple(
        argument
        for uid in sorted(allowed_uids)
        for argument in ("--allow-uid", str(uid))
    )
    exec_start = " ".join((
        _systemd_quote(python_path),
        "-m",
        "tools.local_exec_broker",
        "--systemd-cgroup",
        *authorization_args,
        "--socket-mode",
        f"{socket_mode:04o}",
        f"--socket=%t/{USER_RUNTIME_DIRECTORY}/broker.sock",
        f"--staging-root=%t/{USER_RUNTIME_DIRECTORY}",
    ))
    return f"""[Unit]
Description=Hermes local execution broker
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory={_systemd_working_directory(import_root)}
ExecStart={exec_start}
RuntimeDirectory={USER_RUNTIME_DIRECTORY}
RuntimeDirectoryMode=0700
KillMode=control-group
Restart=on-failure

[Install]
WantedBy=default.target
"""


def _install_user_service(
    *,
    service_dir: Path,
    python_path: str,
    import_root: str,
    allowed_uids: frozenset[int] | None = None,
    socket_mode: int = 0o600,
) -> Path:
    """Install and start the explicit systemd user-service contract."""
    # Do not leave a unit behind when this login has no reachable user manager.
    subprocess.run(
        ["systemctl", "--user", "show-environment"],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    service_dir.mkdir(parents=True, exist_ok=True)
    unit_path = service_dir / USER_SERVICE_NAME
    unit_path.write_text(
        _render_user_service(
            python_path=python_path,
            import_root=import_root,
            allowed_uids=allowed_uids,
            socket_mode=socket_mode,
        ),
        encoding="utf-8",
    )
    unit_path.chmod(0o644)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", USER_SERVICE_NAME],
        check=True,
    )
    return unit_path


def _require_systemd_cgroup(containment: _SystemdContainment) -> None:
    """Prove the user manager can create scopes; requested containment is mandatory."""
    unit = f"hermes-local-exec-probe-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        subprocess.run(
            [
                containment.systemd_run,
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                f"--unit={unit}",
                "--",
                "/bin/true",
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=containment.broker_env,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise SystemExit(
            f"systemd cgroup containment is unavailable: {detail.strip()}"
        ) from exc


def request_launch(
    sock_path: str,
    *,
    expected_peer_uid: int,
    runner: str | None = None,
    runner_fd: int | None = None,
    runner_python: str | None = None,
    argv: list[str] | None = None,
    cwd: str | None = None,
    env: dict,
    fds: list,
    stdin_fd: int | None = None,
    stdout_fd: int | None = None,
    stderr_fd: int | None = None,
    scratch: bool = False,
    timeout: float = 30.0,
    detached: bool = False,
):
    """Ask the broker to launch *runner*; return ``(connection, reply, remainder)``.

    Ordinarily the caller MUST hold the returned connection open for as long as the child
    should live. ``detached=True`` is restricted to argv launches: the broker retains and
    reaps that child independently until it exits or the broker shuts down.
    """
    if (
        not isinstance(expected_peer_uid, int)
        or isinstance(expected_peer_uid, bool)
        or expected_peer_uid < 0
    ):
        raise BrokerError(
            "peer_authentication_failed",
            "expected broker uid must be a non-negative integer",
        )

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        launch_kinds = sum((
            runner is not None,
            runner_fd is not None,
            argv is not None,
        ))
        if launch_kinds != 1:
            raise ValueError("exactly one of runner, runner_fd, or argv is required")
        request = {"op": "launch", "env": env}
        if not isinstance(scratch, bool):
            raise ValueError("scratch must be a boolean")
        if scratch:
            if runner_fd is None:
                raise ValueError("scratch is valid only for runner fd launches")
            request["scratch"] = True
        if detached:
            if argv is None or fds or any(
                fd is not None for fd in (stdin_fd, stdout_fd, stderr_fd)
            ):
                raise ValueError(
                    "detached launches require argv and cannot carry descriptors"
                )
            request["detached"] = True
        rights = list(fds)
        if runner_fd is not None:
            request["runner_fd"] = True
            if runner_python is not None:
                request["runner_python"] = runner_python
            if cwd is not None:
                request["cwd"] = cwd
            rights.insert(0, runner_fd)
        elif argv is not None:
            request["argv"] = argv
            request["cwd"] = cwd
        else:
            request["runner"] = runner
        for name, fd in (
            ("stdin_fd", stdin_fd),
            ("stdout_fd", stdout_fd),
            ("stderr_fd", stderr_fd),
        ):
            if fd is not None:
                request[name] = len(rights)
                rights.append(fd)
        body = json.dumps(request).encode("utf-8") + b"\n"
        if len(body) > MAX_REQUEST_BYTES:
            raise BrokerError(
                "request_too_large",
                f"launch request exceeds the {MAX_REQUEST_BYTES}-byte frame limit",
            )
        socket_info = _validated_client_socket(sock_path, expected_peer_uid)
        conn.settimeout(timeout)
        conn.connect(sock_path)
        _authenticate_broker_peer(conn, sock_path, socket_info, expected_peer_uid)
        _sendmsg_all(conn, body, _ancillary(rights))
        reply, remainder = _read_reply(conn)
        if not reply.get("ok"):
            raise BrokerError(
                reply.get("error") or "unknown",
                reply.get("message") or "launch refused",
            )
    except BaseException:
        conn.close()
        raise
    return conn, reply, remainder


def _validated_client_socket(sock_path: str, expected_peer_uid: int):
    """Return the trusted socket identity required by the documented deployment."""
    directory = os.path.dirname(sock_path) or "."
    try:
        directory_info = os.lstat(directory)
        socket_info = os.lstat(sock_path)
    except OSError as exc:
        raise BrokerError(
            "peer_authentication_failed",
            f"could not inspect broker socket ownership: {exc}",
        ) from exc
    if (
        not stat.S_ISDIR(directory_info.st_mode)
        or stat.S_IMODE(directory_info.st_mode) & 0o022
    ):
        raise BrokerError(
            "peer_authentication_failed",
            "broker socket directory must not be writable by group or other",
        )
    if not stat.S_ISSOCK(socket_info.st_mode):
        raise BrokerError(
            "peer_authentication_failed", "configured broker path is not a socket"
        )
    if socket_info.st_uid != expected_peer_uid:
        raise BrokerError(
            "peer_authentication_failed",
            f"broker socket uid {socket_info.st_uid} does not match configured uid "
            f"{expected_peer_uid}",
        )
    if socket_info.st_uid != directory_info.st_uid:
        raise BrokerError(
            "peer_authentication_failed",
            "broker socket and its directory have different owners",
        )
    return socket_info


def _authenticate_broker_peer(
    conn, sock_path: str, socket_info, expected_peer_uid: int
) -> None:
    """Bind the connected Linux peer to the pre-connect filesystem identity."""
    try:
        current = os.lstat(sock_path)
        _pid, peer_uid, _gid = _UCRED.unpack(
            conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size)
        )
    except (OSError, struct.error) as exc:
        raise BrokerError(
            "peer_authentication_failed", f"could not authenticate broker peer: {exc}"
        ) from exc
    if (current.st_dev, current.st_ino) != (socket_info.st_dev, socket_info.st_ino):
        raise BrokerError(
            "peer_authentication_failed", "broker socket changed while connecting"
        )
    if peer_uid != expected_peer_uid:
        raise BrokerError(
            "peer_authentication_failed",
            f"broker peer uid {peer_uid} does not match configured uid "
            f"{expected_peer_uid}",
        )


def _ancillary(fds):
    return (
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", list(fds)))]
        if fds
        else []
    )


def _sendmsg_all(conn, body: bytes, ancillary) -> None:
    """Send a complete frame, attaching descriptor rights to its first bytes only."""
    sent = 0
    first = True
    while sent < len(body):
        written = conn.sendmsg([body[sent:]], ancillary if first else [])
        if written <= 0:
            raise ConnectionError("sendmsg made no progress")
        sent += written
        first = False


def _read_reply(conn) -> tuple[dict, bytes]:
    """Read one newline-terminated reply frame (no ancillary data expected)."""
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(_RECV_CHUNK)
        if not chunk:
            raise BrokerError(
                "no_reply", "broker closed the connection without replying"
            )
        buf += chunk
        if len(buf) > MAX_REPLY_BYTES:
            raise BrokerError("bad_reply", "broker reply exceeded the frame cap")
    try:
        frame, remainder = buf.split(b"\n", 1)
        return json.loads(frame), remainder
    except json.JSONDecodeError as exc:
        raise BrokerError("bad_reply", f"broker reply was not JSON: {exc}") from exc


def _await_detach_ack(conn, timeout: float) -> bool:
    """Accept one exact client acknowledgement before releasing a detached child."""
    conn.settimeout(timeout)
    try:
        frame, remainder = _read_reply(conn)
    except (BrokerError, OSError, TimeoutError):
        return False
    finally:
        with contextlib.suppress(OSError):
            conn.settimeout(None)
    return frame == {"op": "detach_ack"} and not remainder


def _close_all(fds) -> None:
    while fds:
        with contextlib.suppress(OSError):
            os.close(fds.pop())


def _recv_request(conn, fds: list, handshake_timeout: float):
    """Read one bounded request frame, appending every received descriptor to *fds*.

    *fds* is the CALLER's list on purpose. The kernel installs descriptors into this process
    the moment they arrive, including on a request that turns out to be garbage; making the
    caller the owner from the first byte is what keeps "who closes this" answerable on every
    path out of here.
    """
    buf = b""
    oversized = False
    deadline = time.monotonic() + handshake_timeout
    while True:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            conn.settimeout(remaining)
            msg, ancdata, flags, _addr = conn.recvmsg(
                _RECV_CHUNK, socket.CMSG_SPACE(MAX_RECEIVED_FDS * _INT_SIZE)
            )
        except TimeoutError as exc:
            if oversized:
                raise BrokerError(
                    "request_too_large",
                    f"request exceeded {MAX_REQUEST_BYTES} bytes without a frame terminator",
                ) from exc
            raise BrokerError(
                "handshake_timeout", "no complete request within the handshake window"
            ) from exc
        for level, kind, data in ancdata:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                continue
            if len(data) % _INT_SIZE:
                raise BrokerError(
                    "truncated_ancillary", "partial descriptor in ancillary data"
                )
            received = array.array("i")
            received.frombytes(data)
            fds.extend(received)
        # The kernel installs as many descriptors as the control buffer holds, CLOSES the
        # rest and sets MSG_CTRUNC. Ignoring it means launching a child whose
        # HERMES_BROKER_FDS is silently shorter than what the client passed.
        if flags & socket.MSG_CTRUNC:
            raise BrokerError(
                "truncated_ancillary",
                "ancillary data was truncated; at most "
                f"{MAX_RECEIVED_FDS} descriptors per request",
            )
        if len(fds) > MAX_RECEIVED_FDS:
            raise BrokerError(
                "too_many_fds",
                f"at most {MAX_RECEIVED_FDS} descriptors may be passed per request",
            )
        if not msg:
            raise BrokerError(
                "incomplete_request", "peer closed before sending a complete request"
            )
        if oversized or len(buf) + len(msg) > MAX_REQUEST_BYTES:
            # Stop retaining attacker-controlled bytes, but consume the remainder of this
            # frame so a well-behaved peer can finish sendmsg and read the structured refusal.
            oversized = True
            buf = b""
            if b"\n" in msg:
                raise BrokerError(
                    "request_too_large",
                    f"request exceeded {MAX_REQUEST_BYTES} bytes before its frame terminator",
                )
            continue
        buf += msg
        if b"\n" in buf:
            break
    try:
        request = json.loads(buf.split(b"\n", 1)[0])
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BrokerError("bad_request", f"request was not JSON: {exc}") from exc
    return _validated(request)


def _validated(request):
    """Structural validation only — return ``(runner, env)``.

    This checks the SHAPE of the payload, not its content: the env dict is the approved child
    environment and is forwarded verbatim (see the trust-boundary note in the module
    docstring). NUL is rejected because it would truncate silently at ``execve``.
    """
    if not isinstance(request, dict):
        raise BrokerError("bad_request", "request must be a JSON object")
    if request.get("op") != "launch":
        raise BrokerError("bad_request", f"unsupported op {request.get('op')!r}")
    runner = request.get("runner")
    argv = request.get("argv")
    uses_runner_fd = request.get("runner_fd") is True
    detached = request.get("detached", False)
    if not isinstance(detached, bool):
        raise BrokerError("bad_request", "'detached' must be a boolean")
    if sum((uses_runner_fd, runner is not None, argv is not None)) != 1:
        raise BrokerError(
            "bad_request",
            "request must carry exactly one of 'runner', 'runner_fd', or 'argv'",
        )
    cwd = request.get("cwd")
    runner_python = request.get("runner_python")
    if argv is not None:
        if "runner_python" in request:
            raise BrokerError(
                "bad_request", "'runner_python' is valid only for runner fd launches"
            )
        if (
            not isinstance(argv, list)
            or not argv
            or not all(
                isinstance(part, str) and part and "\0" not in part for part in argv
            )
        ):
            raise BrokerError(
                "bad_request", "'argv' must be a non-empty string array without NUL"
            )
        try:
            for part in argv:
                os.fsencode(part)
        except UnicodeEncodeError as exc:
            raise BrokerError(
                "bad_request", "'argv' entries must be OS-encodable"
            ) from exc
        if not isinstance(cwd, str) or not cwd or "\0" in cwd:
            raise BrokerError(
                "bad_request", "argv launches require a non-empty 'cwd' without NUL"
            )
        if not os.path.isabs(cwd):
            raise BrokerError(
                "bad_request", "argv launch 'cwd' must be an absolute path"
            )
        try:
            os.fsencode(cwd)
        except UnicodeEncodeError as exc:
            raise BrokerError(
                "bad_request", "argv launch 'cwd' is not filesystem-encodable"
            ) from exc
    elif uses_runner_fd:
        if runner_python is not None and (
            not isinstance(runner_python, str)
            or not runner_python
            or "\0" in runner_python
        ):
            raise BrokerError(
                "bad_request", "'runner_python' must be a non-empty string without NUL"
            )
        if runner_python is not None and not os.path.isabs(runner_python):
            raise BrokerError("bad_request", "'runner_python' must be an absolute path")
        if cwd is not None and (
            not isinstance(cwd, str) or not cwd or "\0" in cwd or not os.path.isabs(cwd)
        ):
            raise BrokerError(
                "bad_request", "'cwd' must be an absolute non-empty string without NUL"
            )
        try:
            if runner_python is not None:
                os.fsencode(runner_python)
            if cwd is not None:
                os.fsencode(cwd)
        except UnicodeEncodeError as exc:
            raise BrokerError(
                "bad_request", "runner fd launch paths must be OS-encodable"
            ) from exc
    elif "cwd" in request or "runner_python" in request:
        raise BrokerError(
            "bad_request",
            "'cwd' and 'runner_python' are valid only for argv or runner fd launches",
        )
    elif not uses_runner_fd:
        if not isinstance(runner, str) or not runner:
            raise BrokerError("bad_request", "'runner' must be a non-empty string")
        if "\0" in runner:
            raise BrokerError("bad_request", "'runner' must not contain NUL")
        try:
            os.fsencode(runner)
        except UnicodeEncodeError as exc:
            raise BrokerError(
                "bad_request", "'runner' is not filesystem-encodable"
            ) from exc
    if detached and argv is None:
        raise BrokerError("bad_request", "'detached' is valid only for argv launches")
    scratch = request.get("scratch", False)
    if not isinstance(scratch, bool):
        raise BrokerError("bad_request", "'scratch' must be a boolean")
    if scratch and not uses_runner_fd:
        raise BrokerError(
            "bad_request", "'scratch' is valid only for runner fd launches"
        )
    if "cleanup_dirs" in request or "cleanup_fds" in request:
        raise BrokerError(
            "bad_request", "peer-named cleanup targets are not accepted"
        )
    if "env" not in request:
        env = {}
    else:
        env = request["env"]
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise BrokerError(
            "bad_request", "'env' must be a JSON object of string to string"
        )
    if any("\0" in key or "\0" in value or "=" in key for key, value in env.items()):
        raise BrokerError(
            "bad_request", "environment entries must not contain NUL or '=' in names"
        )
    try:
        for key, value in env.items():
            os.fsencode(key)
            os.fsencode(value)
    except UnicodeEncodeError as exc:
        raise BrokerError(
            "bad_request", "environment entries must be OS-encodable"
        ) from exc
    stdio = {}
    for name in ("stdin_fd", "stdout_fd", "stderr_fd"):
        if name not in request:
            continue
        index = request[name]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise BrokerError("bad_request", f"'{name}' must be a descriptor index")
        stdio[name] = index
    if detached and stdio:
        raise BrokerError(
            "bad_request", "detached launches cannot carry stdio descriptors"
        )
    return (
        runner,
        uses_runner_fd,
        runner_python,
        argv,
        cwd,
        env,
        stdio,
        detached,
        scratch,
    )


def _resolve_runner(staging_root: str, runner: str) -> str:
    """Resolve *runner* to a real path strictly inside *staging_root*.

    The runner transport only means anything if the BROKER is the side that resolves the
    path. ``realpath`` collapses ``..`` and follows symlinks BEFORE the containment test, so
    a symlink staged inside the root but pointing out of it is refused rather than followed.
    """
    resolved = os.path.realpath(runner)
    if not resolved.startswith(staging_root + os.sep):
        raise BrokerError(
            "runner_outside_root",
            f"runner resolves outside the staging root {staging_root}",
        )
    return resolved


def _open_runner(path: str) -> int:
    """Open the staged runner. ``O_NOFOLLOW`` closes the realpath-then-open swap window."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise BrokerError("runner_not_regular", "runner is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _validate_runner_fd(fd: int) -> None:
    """Require an open, readable regular file suitable for ``/proc/self/fd`` execution."""
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise BrokerError("runner_not_regular", "runner is not a regular file")
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if flags & getattr(os, "O_PATH", 0) or flags & os.O_ACCMODE != os.O_RDONLY:
        raise BrokerError("runner_not_readable", "runner descriptor must be read-only")


def _child_env_memfd(child_env: dict[str, str]) -> int:
    """Serialize an exec environment to an anonymous descriptor without leaking failures."""
    env_fd = os.memfd_create("hermes-child-env", os.MFD_CLOEXEC)
    try:
        with os.fdopen(os.dup(env_fd), "w", encoding="utf-8") as payload_file:
            # ASCII escapes preserve surrogateescaped environment bytes through JSON; json.load
            # reconstructs them before os.execvpe encodes them back with surrogateescape.
            json.dump(child_env, payload_file)
        os.lseek(env_fd, 0, os.SEEK_SET)
        return env_fd
    except BaseException:
        os.close(env_fd)
        raise


def _validate_execve_payload(argv: list[str], env: dict[str, str]) -> None:
    """Reject Linux execve payloads that the kernel will refuse with ``E2BIG``."""
    try:
        arg_max = int(os.sysconf("SC_ARG_MAX"))
        max_string = int(os.sysconf("SC_PAGE_SIZE")) * 32
    except (OSError, ValueError):
        return
    encoded = [*(os.fsencode(part) + b"\0" for part in argv)]
    encoded.extend(
        os.fsencode(f"{name}={value}") + b"\0" for name, value in env.items()
    )
    pointer_bytes = (len(encoded) + 2) * struct.calcsize("P")
    if any(len(item) > max_string for item in encoded) or (
        sum(map(len, encoded)) + pointer_bytes > arg_max
    ):
        raise OSError(errno.E2BIG, os.strerror(errno.E2BIG))


def _current_service_unit(
    *, pid: int | None = None, proc_root: Path = Path("/proc")
) -> str | None:
    """Return the innermost service below this user's systemd manager, if any."""
    try:
        membership = (
            proc_root / str(os.getpid() if pid is None else pid) / "cgroup"
        ).read_text(encoding="utf-8")
    except OSError:
        return None
    user_manager = f"user@{os.geteuid()}.service"
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or fields[0] != "0" or fields[1]:
            continue
        components = Path(fields[2]).parts
        try:
            manager_index = components.index(user_manager)
        except ValueError:
            continue
        for component in reversed(components[manager_index + 1 :]):
            if component.endswith(".service") and component != ".service":
                return component
    return None


def _launch(
    runner_fd: int | None,
    env: dict,
    fds: list,
    *,
    runner_python=None,
    argv=None,
    cwd=None,
    stdio_fds=None,
    systemd_containment: _SystemdContainment | None = None,
    exec_timeout: float = DEFAULT_EXEC_TIMEOUT,
):
    """Spawn the child with everything handed over explicitly."""
    child_env = dict(env)
    child_env[FDS_ENV] = ",".join(str(fd) for fd in fds)
    # pass_fds keeps each descriptor at its own number in the child, which is what makes both
    # FDS_ENV and the /proc/self/fd runner path resolvable on the far side.
    # The bootstrap reads the already-open descriptor directly. Asking the interpreter to open
    # ``/proc/self/fd/N`` as a script would re-check the inode's mode bits and fail after a
    # legitimate cross-UID SCM_RIGHTS handoff, even though the descriptor itself is readable.
    if argv is None:
        runner_path = f"/proc/self/fd/{runner_fd}"
        child_argv = [
            runner_python or sys.executable,
            "-c",
            _RUNNER_BOOTSTRAP,
            runner_path,
            str(runner_fd),
        ]
    else:
        child_argv = argv
    _validate_execve_payload(child_argv, child_env)
    stdio_fds = stdio_fds or {}
    child_stdin = stdio_fds.get("stdin_fd", subprocess.DEVNULL)
    child_stdout = stdio_fds.get("stdout_fd", subprocess.DEVNULL)
    requested_stderr = stdio_fds.get("stderr_fd")
    child_stderr = requested_stderr if requested_stderr is not None else child_stdout
    unit = None
    env_fd = None
    status_r = status_w = diagnostics_r = diagnostics_w = None
    popen_env = child_env
    if systemd_containment is not None:
        unit = f"hermes-local-exec-{os.getpid()}-{secrets.token_hex(8)}"
        service_unit = _current_service_unit()
        service_properties = []
        if service_unit is not None:
            service_properties = [
                f"--property=BindsTo={service_unit}",
                f"--property=After={service_unit}",
            ]
        env_fd = _child_env_memfd(child_env)
        try:
            status_r, status_w = os.pipe2(os.O_CLOEXEC)
            diagnostics_r, diagnostics_w = os.pipe2(os.O_CLOEXEC)
        except BaseException:
            _close_all([
                fd
                for fd in (env_fd, status_r, status_w, diagnostics_r, diagnostics_w)
                if fd is not None
            ])
            env_fd = None
            raise
        child_argv = [
            systemd_containment.systemd_run,
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            *service_properties,
            "--",
            systemd_containment.env,
            "-i",
            "--",
            sys.executable,
            "-I",
            "-c",
            _ENV_EXEC_BOOTSTRAP,
            str(env_fd),
            str(status_w),
            str(requested_stderr if requested_stderr is not None else 1),
            *child_argv,
        ]
        # No peer value reaches either trusted launcher's argv or envp. The bootstrap reads
        # the complete approved child environment from an inherited anonymous descriptor only
        # after systemd has placed it in the command scope, then execs the requested program.
        popen_env = systemd_containment.broker_env
    pass_fds = tuple(
        fd
        for fd in (
            runner_fd,
            *fds,
            env_fd,
            status_w,
            requested_stderr if systemd_containment is not None else None,
        )
        if fd is not None
    )
    try:
        proc = subprocess.Popen(
            child_argv,
            env=popen_env,
            pass_fds=pass_fds,
            close_fds=True,
            stdin=child_stdin,
            stdout=child_stdout,
            # The scope launcher keeps private diagnostics until its bootstrap reports a clean
            # exec. The bootstrap then dup2s the separately requested stderr descriptor; direct
            # launches can wire it here without sharing the sentinel-framed stdout pipe.
            stderr=diagnostics_w if diagnostics_w is not None else child_stderr,
            start_new_session=True,
            cwd=cwd,
        )
        if unit is not None:
            proc._hermes_systemd_unit = f"{unit}.scope"
        if status_w is not None:
            os.close(status_w)
            status_w = None
        if diagnostics_w is not None:
            os.close(diagnostics_w)
            diagnostics_w = None
        if status_r is not None and diagnostics_r is not None:
            try:
                _await_contained_exec(status_r, diagnostics_r, exec_timeout)
            except BaseException:
                _terminate(proc)
                raise
    finally:
        if env_fd is not None:
            os.close(env_fd)
        _close_all([
            fd
            for fd in (status_r, status_w, diagnostics_r, diagnostics_w)
            if fd is not None
        ])
    return proc


def _await_contained_exec(status_fd: int, diagnostics_fd: int, timeout: float) -> None:
    """Wait until the in-scope bootstrap execs, or refuse the launch fail-closed."""
    deadline = time.monotonic() + timeout
    status = bytearray()
    diagnostics = bytearray()
    open_fds = {status_fd: status, diagnostics_fd: diagnostics}
    with selectors.DefaultSelector() as sel:
        for fd in open_fds:
            sel.register(fd, selectors.EVENT_READ)
        while open_fds:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BrokerError(
                    "launch_failed", "contained exec acknowledgement timed out"
                )
            events = sel.select(remaining)
            if not events:
                raise BrokerError(
                    "launch_failed", "contained exec acknowledgement timed out"
                )
            for key, _mask in events:
                chunk = os.read(key.fd, 4096)
                if chunk:
                    limit = 4096 if key.fd == status_fd else 65536
                    if len(open_fds[key.fd]) < limit:
                        open_fds[key.fd].extend(chunk[: limit - len(open_fds[key.fd])])
                    continue
                sel.unregister(key.fd)
                open_fds.pop(key.fd)
    lines = bytes(status).splitlines()
    if lines == [b"ready"]:
        return
    detail = bytes(diagnostics).decode("utf-8", "replace").strip()
    if len(lines) >= 2 and lines[0] == b"ready" and lines[1].startswith(b"error "):
        _, raw_errno, raw_message = lines[1].split(b" ", 2)
        error_number = int(raw_errno)
        message = raw_message.decode("utf-8", "replace")
        code = "request_too_large" if error_number == errno.E2BIG else "launch_failed"
        raise BrokerError(code, message)
    raise BrokerError(
        "launch_failed", detail or "contained launcher exited before exec"
    )


def _cgroup_v2_membership(pid: int, proc_root: Path) -> str:
    membership = (proc_root / str(pid) / "cgroup").read_text(encoding="utf-8")
    return next(
        path
        for hierarchy, controllers, path in (
            line.split(":", 2) for line in membership.splitlines()
        )
        if hierarchy == "0" and not controllers
    )


def _kill_cgroup_v2(
    pid: int,
    *,
    broker_pid: int | None = None,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> bool:
    """SIGKILL *pid*'s cgroup-v2 membership, but never the broker's own cgroup."""
    try:
        relative = _cgroup_v2_membership(pid, proc_root)
        own_relative = _cgroup_v2_membership(
            os.getpid() if broker_pid is None else broker_pid,
            proc_root,
        )
        relative_path = Path(relative)
        own_relative_path = Path(own_relative)
        if (
            relative_path == own_relative_path
            or relative_path in own_relative_path.parents
        ):
            return False
        root = cgroup_root.resolve()
        kill_file = (root / relative.lstrip("/") / "cgroup.kill").resolve()
        if not kill_file.is_relative_to(root):
            return False
        kill_file.write_text("1", encoding="utf-8")
        return True
    except (OSError, StopIteration, ValueError):
        return False


def _signal_group(pid: int, sig) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)  # windows-footgun: ok — Linux-only broker


def _process_start_time(pid: int) -> int:
    """Return Linux kernel start ticks while the broker still owns *pid*."""
    stat_fields = (
        Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
    )
    return int(stat_fields[19])


def _signal_process_tree(proc, sig) -> None:
    """Signal either the requested cgroup or the legacy process group."""
    unit = getattr(proc, "_hermes_systemd_unit", None)
    if unit is None:
        _signal_group(proc.pid, sig)
        return
    command = [
        "systemctl",
        "--user",
        "kill",
        f"--signal={signal.Signals(sig).name}",
        unit,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_SYSTEMCTL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        return
    # A dead user bus must not turn a failed systemctl invocation into a successful lease
    # teardown. cgroup.kill reaches setsid descendants on v2; the process-group fallback
    # still covers the legacy/non-v2 path and gives SIGTERM its best available delivery.
    if sig == signal.SIGKILL and _kill_cgroup_v2(proc.pid):
        return
    _signal_group(proc.pid, sig)


def _wait_unreaped(pid: int, timeout: float) -> None:
    """Wait up to *timeout* for *pid* to exit, deliberately leaving it UNREAPED.

    ``Popen.wait``/``poll`` would reap it, and a reaped pid can be recycled — after which the
    pgid we are about to sweep may belong to somebody else entirely.
    """
    try:
        pidfd = os.pidfd_open(pid)
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(pidfd, selectors.EVENT_READ)
                sel.select(timeout)
        finally:
            os.close(pidfd)
    except OSError:
        deadline = time.monotonic() + timeout
        while not _exited_unreaped(pid):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.05, remaining))


def _exited_unreaped(pid: int) -> bool:
    """Observe child exit without releasing its pid for reuse."""
    try:
        return (
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        )
    except ChildProcessError:
        return True


def _unreaped_returncode(pid: int) -> int | None:
    """Read a child's exit status without releasing its pid for the final group sweep."""
    try:
        info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return None
    if info is None:
        return None
    if info.si_code == os.CLD_EXITED:
        return info.si_status
    return -info.si_status


def _terminate(proc) -> None:
    """Tear the child's process GROUP down and REAP the leader.

    ``start_new_session=True`` made the child its own group leader, so its pid is the pgid —
    no ``getpgid`` lookup to race against an already-exited child. Reaping is part of the
    contract, not cleanup: an unreaped child is still a ``/proc`` entry that answers
    ``kill(pid, 0)``, i.e. indistinguishable from one that outlived its lease.
    """
    _terminate_many([proc])


def _terminate_many(procs) -> None:
    """Broadcast teardown phases to *procs* under shared grace deadlines."""
    procs = [proc for proc in procs if proc.returncode is None]
    for proc in procs:
        _signal_process_tree(proc, signal.SIGTERM)
    term_deadline = time.monotonic() + _TERM_GRACE_SECONDS
    for proc in procs:
        _wait_unreaped(proc.pid, max(0.0, term_deadline - time.monotonic()))
    # The leader is still unreaped, so its pid — and with it the pgid — cannot have been
    # recycled. This is the only safe moment to sweep descendants that outlived the leader or
    # ignored the SIGTERM; they are not our children, so we signal them and let init reap.
    for proc in procs:
        _signal_process_tree(
            proc,
            signal.SIGKILL,  # windows-footgun: ok — Linux-only broker
        )
    kill_deadline = time.monotonic() + _KILL_GRACE_SECONDS
    for proc in procs:
        try:
            proc.wait(timeout=max(0.0, kill_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            # Keep the Popen object (and therefore waitpid ownership) alive until the
            # uninterruptible syscall returns and SIGKILL can complete.
            threading.Thread(target=proc.wait, daemon=True).start()


class _Leases:
    """Every accepted connection, worker and child, so shutdown can drain all three.

    Registration precedes ``Thread.start`` so even a worker blocked inside ``Popen`` remains
    visible to shutdown. Closing every accepted connection breaks incomplete handshakes;
    waiting for the registry to empty covers children spawned after the initial snapshot.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._workers = {}
        self._shutting_down = False

    def register(self, conn, worker) -> bool:
        with self._condition:
            if self._shutting_down:
                return False
            self._workers[conn] = [worker, None, []]
            return True

    def add(self, conn, proc, cleanup_dirs=()) -> bool:
        with self._condition:
            entry = self._workers.get(conn)
            if self._shutting_down or entry is None:
                return False
            entry[1] = proc
            entry[2] = list(cleanup_dirs)
            return True

    def claim(self, conn, proc) -> bool:
        with self._condition:
            entry = self._workers.get(conn)
            if entry is None or entry[1] is not proc:
                return False
            entry[1] = None
            entry[2] = []
            return True

    def finished(self, conn) -> None:
        with self._condition:
            self._workers.pop(conn, None)
            self._condition.notify_all()

    def drain(self) -> None:
        with self._condition:
            self._shutting_down = True
            entries = list(self._workers.items())
            owned = []
            for _conn, entry in entries:
                if entry[1] is not None:
                    owned.append((entry[1], entry[2]))
                    entry[1] = None
                    entry[2] = []
        for conn, _entry in entries:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
        _terminate_many([proc for proc, _cleanup_dirs in owned])
        for _proc, cleanup_dirs in owned:
            with contextlib.suppress(BrokerError):
                _cleanup_scratch_directories(cleanup_dirs)
        with self._condition:
            while self._workers:
                self._condition.wait()


def _reply(conn, payload: dict) -> bool:
    try:
        conn.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        return True
    except OSError:
        return False


def _await_lease_end_without_selector(conn, pid: int) -> None:
    """Wait for lease EOF or child exit without allocating another descriptor."""
    conn.settimeout(0.05)
    try:
        while not _exited_unreaped(pid):
            try:
                if not conn.recv(_RECV_CHUNK):
                    return
            except TimeoutError:
                continue
    finally:
        conn.settimeout(None)


def _await_lease_end(conn, proc) -> None:
    """Block until the lease ends — the client's EOF, or the child exiting on its own.

    Watching only the connection makes the lease one-directional: a child that finishes
    normally would leave this thread parked in ``recv`` until the client happened to
    disconnect, holding an unreaped child the whole time. The pidfd is the other half, and it
    reports the exit WITHOUT reaping, so ``_terminate`` still owns the group sweep.
    """
    conn.settimeout(None)
    try:
        pidfd = os.pidfd_open(proc.pid)
    except ProcessLookupError:
        return
    except OSError:
        _await_lease_end_without_selector(conn, proc.pid)
        return
    try:
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(pidfd, selectors.EVENT_READ)
                sel.register(conn, selectors.EVENT_READ)
                while True:
                    for key, _mask in sel.select():
                        if key.fd == pidfd or not conn.recv(_RECV_CHUNK):
                            return
        except OSError:
            _await_lease_end_without_selector(conn, proc.pid)
    finally:
        os.close(pidfd)


def _cleanup_scratch_directories(paths: list[str]) -> None:
    """Clear and remove broker-created per-lease scratch directories.

    Every path originates from ``mkdtemp`` below the validated broker staging root; peers cannot
    name a cleanup target. Descriptor-relative deletion and no-follow opens keep traversal pinned
    to the directory created for this lease even if worker code leaves symlinks behind.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    for path in paths:
        try:
            directory_fd = os.open(path, flags)
        except OSError as exc:
            raise BrokerError(
                "cleanup_failed", f"could not open cleanup directory: {exc}"
            ) from exc
        try:
            for name in os.listdir(directory_fd):
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        shutil.rmtree(name, dir_fd=directory_fd)
                    else:
                        os.unlink(name, dir_fd=directory_fd)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise BrokerError(
                        "cleanup_failed", f"could not clear cleanup directory: {exc}"
                    ) from exc
        finally:
            os.close(directory_fd)
        try:
            os.rmdir(path)
        except OSError as exc:
            raise BrokerError(
                "cleanup_failed", f"could not remove scratch directory: {exc}"
            ) from exc


def _serve_connection(
    conn,
    staging_root: str,
    handshake_timeout: float,
    leases: _Leases,
    allowed_uids: frozenset[int] | None = None,
    systemd_containment: _SystemdContainment | None = None,
) -> None:
    proc, runner_fd = None, None
    process_start_time = None
    cleanup_dirs: list[str] = []
    scratch = False
    cleanup_armed = False
    registered = False
    exit_reply: int | None = None
    legacy_completion = False
    connection_closed = False
    # Descriptors the kernel installed on our behalf. One owner, one close, every path out.
    fds: list = []
    stdio_fds: dict[str, int] = {}
    try:
        try:
            if allowed_uids is None:
                allowed_uids = frozenset({os.geteuid()})
            _pid, peer_uid, _gid = _UCRED.unpack(
                conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size)
            )
            if peer_uid not in allowed_uids:
                raise BrokerError(
                    "peer_uid_not_allowed",
                    f"peer uid {peer_uid} is not allowed",
                )
            (
                runner,
                uses_runner_fd,
                runner_python,
                argv,
                cwd,
                env,
                stdio,
                detached,
                scratch,
            ) = _recv_request(conn, fds, handshake_timeout)
            legacy_completion = argv is None and not scratch
            indexes = list(stdio.values())
            if (
                len(set(indexes)) != len(indexes)
                or any(index >= len(fds) for index in indexes)
                or (uses_runner_fd and 0 in indexes)
            ):
                raise BrokerError(
                    "bad_request", "stdio descriptor index was not received"
                )
            stdio_fds = {name: fds[index] for name, index in stdio.items()}
            for index in sorted(indexes, reverse=True):
                fds.pop(index)
            if detached and fds:
                raise BrokerError(
                    "bad_request", "detached launches cannot carry descriptors"
                )
            if uses_runner_fd:
                if not fds:
                    raise BrokerError(
                        "runner_fd_missing", "runner descriptor was not received"
                    )
                runner_fd = fds.pop(0)
                _validate_runner_fd(runner_fd)
            elif runner is not None:
                if len(fds) > MAX_FDS:
                    raise BrokerError(
                        "too_many_fds",
                        f"at most {MAX_FDS} descriptors may be passed per request",
                    )
                runner_fd = _open_runner(_resolve_runner(staging_root, runner))
            if len(fds) > MAX_FDS:
                raise BrokerError(
                    "too_many_fds",
                    f"at most {MAX_FDS} descriptors may be passed per request",
                )
            if scratch:
                scratch_dir = tempfile.mkdtemp(
                    prefix="hermes-kernel-", dir=staging_root
                )
                cleanup_dirs = [scratch_dir]
                cleanup_armed = True
                os.chmod(scratch_dir, 0o700)
                env = dict(env)
                env["TMPDIR"] = scratch_dir
                if cwd is None:
                    cwd = scratch_dir
            proc = _launch(
                runner_fd,
                env,
                fds,
                runner_python=runner_python,
                argv=argv,
                cwd=cwd,
                stdio_fds=stdio_fds,
                systemd_containment=systemd_containment,
                exec_timeout=DEFAULT_EXEC_TIMEOUT,
            )
            cleanup_armed = True
            try:
                process_start_time = _process_start_time(proc.pid)
            except BaseException:
                # The child exists but has not entered the lease table yet. Do not let an
                # identity-read failure turn a refused launch into an unowned process.
                _terminate(proc)
                proc = None
                raise
        except BrokerError as exc:
            _reply(conn, {"ok": False, "error": exc.code, "message": exc.message})
            return
        except OSError as exc:
            error = "request_too_large" if exc.errno == errno.E2BIG else "launch_failed"
            _reply(conn, {"ok": False, "error": error, "message": str(exc)})
            return
        finally:
            # Every forwarded descriptor is the peer's channel, not ours: the child holds its
            # own copies, and a retained copy here would keep a pipe from ever reaching EOF.
            _close_all(fds)
            _close_all(list(stdio_fds.values()))
            if runner_fd is not None:
                os.close(runner_fd)
        # Registered BEFORE the reply: a SIGTERM racing the handshake must still find this
        # child, or it is orphaned in the one window where nobody is watching it.
        registered = leases.add(conn, proc, cleanup_dirs)
        if not registered:
            _terminate(proc)
            return
        reply = {
            "ok": True,
            "pid": proc.pid,
            "start_time": process_start_time,
        }
        if detached:
            reply.update(
                detached=True,
                systemd_unit=getattr(proc, "_hermes_systemd_unit", ""),
            )
        reply_delivered = _reply(conn, reply)
        if detached:
            if not reply_delivered or not _await_detach_ack(conn, handshake_timeout):
                return
            # A detached argv launch is the background worker itself, not a child hidden
            # behind a foreground lease. Keep its Popen registered so broker shutdown still
            # owns containment and reaping, but a client EOF no longer orders it killed.
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
            while not _exited_unreaped(proc.pid):
                _wait_unreaped(proc.pid, 60.0)
            return
        # The connection IS the child's lease. A clean close, a crashed client or a SIGKILLed
        # one all surface the same way, which is exactly the signal the inherited
        # parent-death pipe gave us before sudo started closing it.
        with contextlib.suppress(OSError):
            _await_lease_end(conn, proc)
        # Clients that need a completion frame opt in either through argv execution or broker-side
        # cleanup. Observe without reaping: the finalizer must first sweep the whole process tree,
        # then clear worker-owned files, and only then acknowledge completion to the controller.
        if argv is not None or cleanup_dirs:
            exit_reply = _unreaped_returncode(proc.pid)
    finally:
        cleanup_error = None
        cleanup_owned_here = cleanup_armed and not registered
        try:
            if proc is not None and registered:
                cleanup_owned_here = leases.claim(conn, proc)
                if cleanup_owned_here:
                    if legacy_completion:
                        with contextlib.suppress(OSError):
                            conn.close()
                        connection_closed = True
                    _terminate(proc)
            if cleanup_owned_here and cleanup_dirs:
                try:
                    _cleanup_scratch_directories(cleanup_dirs)
                except BrokerError as exc:
                    cleanup_error = exc
            if exit_reply is not None:
                if cleanup_error is None:
                    _reply(conn, {"exit": exit_reply})
                else:
                    _reply(
                        conn,
                        {
                            "error": cleanup_error.code,
                            "message": cleanup_error.message,
                        },
                    )
        finally:
            if not connection_closed:
                with contextlib.suppress(OSError):
                    conn.close()
            leases.finished(conn)


def _validated_staging_root(path: str) -> str:
    root = os.path.realpath(path)
    if not os.path.isdir(root):
        raise SystemExit(f"staging root is not a directory: {path}")
    info = os.stat(root)
    broker_uid = os.geteuid()  # windows-footgun: ok — Linux-only broker
    if info.st_uid != broker_uid or stat.S_IMODE(info.st_mode) != 0o700:
        raise SystemExit(
            "staging root must be owned by the broker uid with permissions 0700: "
            f"{path}"
        )
    return root


def _validated_socket_path(path: str) -> str:
    length = len(os.fsencode(path))
    if length > MAX_SOCKET_PATH_BYTES:
        raise SystemExit(
            f"AF_UNIX socket path must be at most {MAX_SOCKET_PATH_BYTES} bytes: "
            f"got {length}: {path}"
        )
    return path


def _validated_socket_directory(path: str) -> str:
    directory = os.path.dirname(path) or "."
    try:
        info = os.stat(directory, follow_symlinks=False)
    except OSError as exc:
        raise SystemExit(
            f"socket directory is not accessible: {directory}: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()  # windows-footgun: ok — Linux-only broker
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        raise SystemExit(
            "socket directory must be broker-owned and not writable by group/other: "
            f"{directory}"
        )
    return directory


def _validated_socket_mode(mode: int) -> int:
    if mode not in (0o600, 0o660, 0o666):
        raise ValueError("socket mode must be 0600, 0660, or 0666")
    return mode


def _clear_stale_socket(sock_path: str) -> None:
    """Remove a socket left behind by an unclean shutdown — and nothing else.

    ``lstat``, not ``stat``: the decision is about the path itself, so a symlink parked here
    pointing at something that matters is refused rather than followed and unlinked. Anything
    that is not a socket is somebody else's file; the broker declines to start instead.
    """
    try:
        mode = os.lstat(sock_path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise SystemExit(
            f"refusing to replace a path that is not a socket: {sock_path}"
        )

    # A socket pathname is not stale merely because it already exists. Probe it before
    # unlinking: stealing a live broker's path leaves that process running but unreachable.
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(sock_path)
    except (ConnectionRefusedError, FileNotFoundError):
        pass
    except OSError as exc:
        raise SystemExit(
            f"refusing to replace an uncertain socket: {sock_path}: {exc}"
        ) from exc
    else:
        raise SystemExit(f"refusing to replace a live broker socket: {sock_path}")
    finally:
        probe.close()
    os.unlink(sock_path)


def _reclaim_stale_publish_dir(path: str) -> bool:
    """Reclaim an empty private slot or one unreachable broker socket."""
    try:
        info = os.lstat(path)
        entries = os.listdir(path)
    except OSError:
        return False
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()  # windows-footgun: ok — Linux-only broker
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        return False
    if not entries:
        try:
            os.rmdir(path)
        except OSError:
            return False
        return True
    if entries != ["s"]:
        return False
    socket_path = os.path.join(path, "s")
    try:
        if not stat.S_ISSOCK(os.lstat(socket_path).st_mode):
            return False
    except OSError:
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(socket_path)
    except (ConnectionRefusedError, FileNotFoundError):
        pass
    except OSError:
        return False
    else:
        return False
    finally:
        probe.close()
    try:
        os.unlink(socket_path)
        os.rmdir(path)
    except OSError:
        return False
    return True


def _unlink_owned_socket(sock_path: str, owned_fd: int) -> None:
    """Unlink *sock_path* only while it still names the socket we bound."""
    try:
        socket_dir = os.path.dirname(sock_path) or "."
        lock_fd = os.open(socket_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                current = os.lstat(sock_path)
            except FileNotFoundError:
                return
            owned = os.fstat(owned_fd)
            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                os.unlink(sock_path)
        finally:
            os.close(lock_fd)
    finally:
        os.close(owned_fd)


def _install_shutdown(listener) -> None:
    """Turn SIGTERM/SIGINT into an ordinary exit from the accept loop.

    Closing the listener is what breaks ``accept``: under PEP 475 Python retries an
    EINTR-interrupted syscall itself, so a handler that merely set a flag would go unnoticed
    until the next connection happened to arrive.
    """

    def _shutdown(_signum, _frame):
        listener.close()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _shutdown)


def _start_worker(
    conn,
    root: str,
    handshake_timeout: float,
    leases: _Leases,
    allowed_uids: frozenset[int] | None = None,
    systemd_containment: _SystemdContainment | None = None,
) -> None:
    worker = threading.Thread(
        target=_serve_connection,
        args=(conn, root, handshake_timeout, leases, allowed_uids, systemd_containment),
        daemon=True,
    )
    if not leases.register(conn, worker):
        conn.close()
        return
    try:
        worker.start()
    except RuntimeError:
        leases.finished(conn)
        conn.close()


def serve(
    sock_path: str,
    staging_root: str,
    *,
    handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT,
    allowed_uids: frozenset[int] | None = None,
    socket_mode: int = 0o600,
    systemd_cgroup: bool = False,
) -> None:
    """Bind, announce readiness, then serve one connection per thread."""
    systemd_containment = None
    if systemd_cgroup:
        systemd_containment = _resolve_systemd_containment()
        _require_systemd_cgroup(systemd_containment)
    sock_path = _validated_socket_path(sock_path)
    socket_dir = _validated_socket_directory(sock_path)
    root = _validated_staging_root(staging_root)
    socket_mode = _validated_socket_mode(socket_mode)
    if allowed_uids is None:
        allowed_uids = frozenset({os.geteuid()})
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    leases = _Leases()
    owned_fd = None
    publish_dir = None
    temporary_socket = None
    try:
        lock_fd = os.open(socket_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            _clear_stale_socket(sock_path)
            for suffix in _PUBLISH_SUFFIXES:
                if suffix == os.path.basename(sock_path):
                    continue
                candidate = os.path.join(socket_dir, suffix)
                try:
                    os.mkdir(candidate, 0o700)
                except FileExistsError:
                    if not _reclaim_stale_publish_dir(candidate):
                        continue
                    os.mkdir(candidate, 0o700)
                publish_dir = candidate
                break
            else:
                raise SystemExit(
                    "no compact private socket publication directory was available",
                )
            os.chmod(publish_dir, 0o700)
            temporary_socket = os.path.join(publish_dir, "s")
            listener.bind(f"/proc/self/fd/{lock_fd}/{suffix}/s")
            listener.listen(16)
            os.chmod(temporary_socket, socket_mode)
            owned_fd = os.open(
                temporary_socket,
                os.O_PATH
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,  # windows-footgun: ok — Linux-only broker
            )
            os.link(temporary_socket, sock_path)
            os.unlink(temporary_socket)
            temporary_socket = None
            os.rmdir(publish_dir)
            publish_dir = None
        finally:
            os.close(lock_fd)
        _install_shutdown(listener)
        print(json.dumps({"ready": True, "socket": sock_path}), flush=True)
        while True:
            try:
                conn, _addr = listener.accept()
            except OSError as exc:
                if listener.fileno() == -1:
                    break  # the shutdown handler closed the listener
                if exc.errno == errno.ECONNABORTED:
                    continue
                if exc.errno in (
                    errno.EMFILE,
                    errno.ENFILE,
                    errno.ENOBUFS,
                    errno.ENOMEM,
                ):
                    time.sleep(0.05)
                    continue
                raise
            _start_worker(
                conn,
                root,
                handshake_timeout,
                leases,
                allowed_uids,
                systemd_containment,
            )
    finally:
        with contextlib.suppress(OSError):
            listener.close()
        # Daemon threads will not unwind, so shutdown — not the workers — is what keeps the
        # "nothing outlives its lease" promise when the broker itself is the one going away.
        leases.drain()
        if owned_fd is not None:
            with contextlib.suppress(OSError):
                _unlink_owned_socket(sock_path, owned_fd)
        if temporary_socket is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary_socket)
        if publish_dir is not None:
            with contextlib.suppress(OSError):
                os.rmdir(publish_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", help="AF_UNIX path to bind")
    parser.add_argument(
        "--staging-root",
        help="directory the broker owns; runners outside it are refused",
    )
    parser.add_argument(
        "--install-user-service",
        action="store_true",
        help="install and start the contained broker as this operator's user service",
    )
    parser.add_argument(
        "--handshake-timeout",
        type=float,
        default=DEFAULT_HANDSHAKE_TIMEOUT,
        help="seconds a connection may take to send one complete request",
    )
    parser.add_argument(
        "--allow-uid",
        action="append",
        type=int,
        dest="allowed_uids",
        help="uid authorized to request launches; repeatable (default: broker euid)",
    )
    parser.add_argument(
        "--socket-mode",
        choices=("0600", "0660", "0666"),
        default="0600",
        help="published socket permissions; peer uid authorization still applies",
    )
    parser.add_argument(
        "--systemd-cgroup",
        action="store_true",
        help="require every command to run in a transient systemd user scope",
    )
    args = parser.parse_args(argv)
    allowed_uids = frozenset(
        args.allowed_uids if args.allowed_uids is not None else [os.geteuid()]
    )
    if any(uid < 0 for uid in allowed_uids):
        parser.error("--allow-uid must be non-negative")
    socket_mode = int(args.socket_mode, 8)
    if args.install_user_service:
        if socket_mode != 0o600 or allowed_uids != frozenset({os.geteuid()}):
            parser.error(
                "the installed user service is owner-only; use the inline broker "
                "with an operator-deployed socket directory for cross-uid access"
            )
        config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        unit_path = _install_user_service(
            service_dir=config_home / "systemd" / "user",
            python_path=sys.executable,
            import_root=_verified_import_root(),
            allowed_uids=allowed_uids,
            socket_mode=socket_mode,
        )
        print(f"installed and started {unit_path}")
        print(
            "broker socket: "
            f"/run/user/{os.geteuid()}/{USER_RUNTIME_DIRECTORY}/broker.sock"
        )
        return 0
    if args.socket is None or args.staging_root is None:
        parser.error(
            "--socket and --staging-root are required unless installing the service"
        )
    serve(
        args.socket,
        args.staging_root,
        handshake_timeout=args.handshake_timeout,
        allowed_uids=allowed_uids,
        socket_mode=socket_mode,
        systemd_cgroup=args.systemd_cgroup,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
