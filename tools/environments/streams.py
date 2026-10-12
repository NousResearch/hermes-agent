"""A long-lived process INSIDE a terminal backend with both stdio pipes open.

``BaseEnvironment.execute`` is request/response: it captures output and returns. The desktop that
follows the terminal backend needs three things that are not: the RFB byte stream between the
Desktop pane and the sandbox's Xvnc, cua-driver's MCP-over-stdio server, and agent-browser
invocations whose daemon must outlive the call. All three are "spawn argv in the sandbox, keep
stdin/stdout as pipes". Each spawn-per-call backend has a local argv prefix that gives exactly that
(``docker exec -i``, ``ssh``, ``apptainer exec``); this module is the one place that knows it.

SDK backends (modal, daytona, vercel) have no argv prefix: ``exec_prefix`` returns None there and
the desktop stays on the gateway host (or refuses, per ``bot_desktop.placement``).
"""
from __future__ import annotations

import shlex
import socket
import subprocess
import threading
from typing import Optional, Sequence

from tools.environments.base import BaseEnvironment


def exec_prefix(env: BaseEnvironment, *, user: Optional[str] = None, interactive: bool = True) -> Optional[list[str]]:
    """Local argv that runs its remainder inside ``env``, or None for backends without one.

    ``user`` selects the sandbox-side account (docker only; ssh runs as the configured login, apptainer as
    the caller). ``interactive`` keeps stdin open (``docker exec -i``); ssh and apptainer always do.
    """
    from tools.environments.docker import DockerEnvironment
    from tools.environments.singularity import SingularityEnvironment
    from tools.environments.ssh import SSHEnvironment

    if isinstance(env, DockerEnvironment):
        container = getattr(env, "_container_id", None)
        if not container:
            return None
        argv = [env._docker_exe, "exec"]
        if interactive:
            argv.append("-i")
        if user:
            argv += ["-u", user]
        return argv + [container]
    if isinstance(env, SSHEnvironment):
        return env._build_ssh_command()
    if isinstance(env, SingularityEnvironment):
        if not getattr(env, "_instance_started", False):
            return None
        return [env.executable, "exec", f"instance://{env.instance_id}"]
    return None


def supports_streams(env: BaseEnvironment) -> bool:
    return exec_prefix(env, interactive=True) is not None


def remote_argv(prefix: Sequence[str], argv: Sequence[str], *, env: Optional[dict] = None,
                shell_joined: bool = False) -> list[str]:
    """``prefix`` + a ``bash -c`` that exports ``env`` and execs ``argv``.

    The script is built once, then delivered the way the client needs it: argv-preserving clients (docker,
    apptainer) get ``["bash", "-c", script]``; ``shell_joined`` clients (OpenSSH joins the words after the host
    with spaces and the remote login shell splits them again) get the whole ``bash -c '<script>'`` as ONE
    quoted word, or the remote side runs ``bash -c export`` and hands the rest to the login shell
    (``SSHEnvironment.run_bash`` quotes for the same reason). Prefer :func:`remote_command`, which knows."""
    exports = " ".join(f"export {k}={shlex.quote(v)};" for k, v in (env or {}).items())
    script = f"{exports} exec {' '.join(shlex.quote(a) for a in argv)}"
    if shell_joined:
        return [*prefix, shlex.join(["bash", "-c", script])]
    return [*prefix, "bash", "-c", script]


def shell_joined(env: BaseEnvironment) -> bool:
    from tools.environments.ssh import SSHEnvironment
    return isinstance(env, SSHEnvironment)


def remote_command(env: BaseEnvironment, argv: Sequence[str], *, child_env: Optional[dict] = None,
                   user: Optional[str] = None, interactive: bool = True) -> Optional[list[str]]:
    """Full local argv running ``argv`` inside ``env`` with ``child_env`` exported, quoted for that backend's
    client; None for a backend without an exec prefix."""
    prefix = exec_prefix(env, user=user, interactive=interactive)
    if prefix is None:
        return None
    return remote_argv(prefix, argv, env=child_env, shell_joined=shell_joined(env))


def open_stream(env: BaseEnvironment, argv: Sequence[str], *, child_env: Optional[dict] = None,
                user: Optional[str] = None, stderr=subprocess.DEVNULL) -> subprocess.Popen:
    """Spawn ``argv`` inside ``env`` with stdin and stdout as pipes (bytes). Raises ``RuntimeError`` for a
    backend that cannot host a stream."""
    command = remote_command(env, argv, child_env=child_env, user=user, interactive=True)
    if command is None:
        raise RuntimeError(f"{type(env).__name__} cannot host a long-lived stdio stream")
    return subprocess.Popen(  # windows-footgun: ok — the prefix is a Linux sandbox's client (docker/ssh/apptainer)
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, close_fds=True)


def run_in(env: BaseEnvironment, argv: Sequence[str], *, child_env: Optional[dict] = None, user: Optional[str] = None,
           timeout: float = 30.0, stdin: Optional[bytes] = None) -> subprocess.CompletedProcess:
    """One short command inside ``env`` with captured bytes output (probes: ``command -v``, ``cat env``)."""
    command = remote_command(env, argv, child_env=child_env, user=user, interactive=stdin is not None)
    if command is None:
        raise RuntimeError(f"{type(env).__name__} cannot exec argv")
    return subprocess.run(  # windows-footgun: ok — Linux sandbox client
        command, input=stdin, capture_output=True, timeout=timeout, check=False)


# ── TCP port forwarding through the exec stream ──────────────────────────────────────────────────────
# A sandbox's loopback ports (Chromium's CDP endpoint, a dev server) are unreachable from the gateway host:
# Docker bridge / ssh remote / apptainer netns. Rather than teach every client a backend-specific tunnel
# (`docker port` needs a published port at create time; ssh -L is ssh-only), a local listener proxies each
# accepted connection over ONE exec stream running a tiny relay inside the sandbox — the same mechanism
# the Bot Screen's RFB relay uses, so it works on every backend that can exec.

_TCP_RELAY = (
    "import os,socket,sys,threading\n"
    "s=socket.create_connection((sys.argv[1],int(sys.argv[2])))\n"
    "def up():\n"
    "  while True:\n"
    "    d=os.read(0,65536)\n"
    "    if not d: break\n"
    "    s.sendall(d)\n"
    "  s.shutdown(socket.SHUT_WR)\n"
    "threading.Thread(target=up,daemon=True).start()\n"
    "while True:\n"
    "  d=s.recv(65536)\n"
    "  if not d: break\n"
    "  os.write(1,d)\n"
)

_forwards: dict[tuple[int, str, int], "PortForward"] = {}
_forwards_lock = threading.Lock()


class PortForward:
    """Local ``127.0.0.1:<local_port>`` whose connections land on ``remote_host:remote_port`` inside ``env``."""

    def __init__(self, env: BaseEnvironment, remote_host: str, remote_port: int, *, user: Optional[str]):
        self.env, self.remote_host, self.remote_port, self.user = env, remote_host, remote_port, user
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(16)
        self.local_port: int = self._server.getsockname()[1]
        self._closed = False
        threading.Thread(target=self._accept_loop, name=f"port-forward-{self.local_port}", daemon=True).start()

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            proc = open_stream(self.env, ["python3", "-c", _TCP_RELAY, self.remote_host, str(self.remote_port)],
                               user=self.user)
        except Exception:
            conn.close()
            return

        def to_remote() -> None:
            try:
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    proc.stdin.write(data)
                    proc.stdin.flush()
            except (OSError, ValueError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        threading.Thread(target=to_remote, daemon=True).start()
        try:
            while True:
                data = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(65536)
                if not data:
                    break
                conn.sendall(data)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            finally:
                proc.kill()

    def close(self) -> None:
        self._closed = True
        try:
            self._server.close()
        except OSError:
            pass


def forward_port(env: BaseEnvironment, remote_port: int, *, remote_host: str = "127.0.0.1",
                 user: Optional[str] = None) -> int:
    """Local port on 127.0.0.1 that reaches ``remote_host:remote_port`` inside ``env``; one listener per
    (env, host, port) is kept for the process lifetime (daemon threads, nothing to clean up)."""
    key = (id(env), remote_host, int(remote_port))
    with _forwards_lock:
        fwd = _forwards.get(key)
        if fwd is None or fwd._closed:
            fwd = _forwards[key] = PortForward(env, remote_host, int(remote_port), user=user)
    return fwd.local_port
