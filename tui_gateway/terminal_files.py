"""Private, bounded file reads on the compute host's existing terminal environment.

This is not a public JSON-RPC method and accepts no commands or environment factories.
"""
from __future__ import annotations

import base64
import uuid

from hermes_cli.ssh_workspace_fs import SshWorkspaceFsError
from hermes_cli.terminal_workspace_fs import TerminalWorkspaceFs, TerminalWorkspaceFsError

CHUNK_BYTES = 64 * 1024


def _unavailable():
    return TerminalWorkspaceFsError("ECONN", "Terminal environment is unavailable")


def scoped_session_fs(session, backend):
    """Resolve only the owning profile's exact durable terminal key; never create."""
    from tools import terminal_tool
    from tools.environments.sandbox_scope import admit_env, current_sandbox_key

    durable = str(session.get("session_key") or session.get("id") or "")
    sandbox = current_sandbox_key()
    keys = ([sandbox] if sandbox else
            [terminal_tool._qualify_task_key(key) for key in (durable, f"session:{durable}")])
    with terminal_tool._env_lock:
        from tools.terminal_file_handles import resolve
        for key in keys:
            env = admit_env(resolve(key))
            if env is not None and getattr(env, "_hermes_backend_name", backend) == backend:
                if not getattr(env, "_hermes_file_read_token", None):
                    env._hermes_file_read_token = uuid.uuid4().hex
                return TerminalWorkspaceFs(env, env_key=key)
    raise _unavailable()


def handle_file_request(server, frame):
    """Execute one structured inspect/read against an already-owned live environment."""
    session = server._sessions.get(frame.get("sid"))
    if (session is None or not frame.get("session_key") or not frame.get("profile_home")
            or session.get("session_key") != frame["session_key"]
            or str(session.get("profile_home") or "") != frame["profile_home"]):
        raise _unavailable()
    operation = frame.get("operation")
    if operation not in {"open", "inspect", "read"}:
        raise SshWorkspaceFsError("EINVAL")
    with server._session_profile_runtime_scope(session):
        from tools import terminal_tool
        backend = terminal_tool._session_scope().env_type
        if backend != frame.get("backend") or backend in {"local", "ssh", "docker"}:
            raise _unavailable()
        fs = scoped_session_fs(session, backend)
        token = fs._env._hermes_file_read_token
        if operation == "open":
            return {"cwd": fs.cwd, "env_token": token}
        if frame.get("env_token") != token:
            raise _unavailable()
        path = frame.get("path")
        if not isinstance(path, str) or len(path) > 4096:
            raise SshWorkspaceFsError("EINVAL")
        if operation == "inspect":
            target, size = fs.inspect_file(path)
            return {"path": target, "size": size}
        offset, length = frame.get("offset"), frame.get("length")
        if (type(offset) is not int or offset < 0 or offset > 2**63 - 1
                or type(length) is not int or not 0 < length <= CHUNK_BYTES):
            raise SshWorkspaceFsError("EINVAL")
        data = b"".join(fs.stream_file(path, offset=offset, length=length))
        return {"data": base64.b64encode(data).decode("ascii")}


class ComputeTerminalWorkspaceFs:
    """HTTP reader pinned to one live process, session and environment generation."""
    def __init__(self, supervisor, sid, session_key, profile_home, backend):
        self._supervisor = supervisor
        self._process = supervisor._proc
        self._identity = dict(sid=sid, session_key=session_key, profile_home=profile_home, backend=backend)
        opened = self._request("open")
        self.cwd = opened["cwd"]
        self._identity["env_token"] = opened["env_token"]

    def _request(self, operation, **params):
        if self._process is None or self._supervisor._proc is not self._process:
            raise _unavailable()
        try:
            reply = self._supervisor.terminal_file({**self._identity, "operation": operation, **params},
                                                   process=self._process)
        except (RuntimeError, OSError, TimeoutError) as exc:
            raise _unavailable() from exc
        if reply.get("type") != "terminal.file.ack":
            raise TerminalWorkspaceFsError(reply.get("code") or "ECONN", reply.get("message") or "Terminal environment is unavailable")
        return reply["result"]

    def inspect_file(self, path):
        result = self._request("inspect", path=path)
        return result["path"], result["size"]

    def stream_file(self, path, chunk_size=CHUNK_BYTES, *, offset=0, length=None):
        target, size = self.inspect_file(path)
        remaining = max(0, size - offset)
        if length is not None:
            remaining = min(remaining, length)
        while remaining:
            count = min(CHUNK_BYTES, chunk_size, remaining)
            result = self._request("read", path=target, offset=offset, length=count)
            data = base64.b64decode(result["data"], validate=True)
            if len(data) != count:
                raise SshWorkspaceFsError("EIO", "Terminal file changed during read")
            yield data
            offset += count
            remaining -= count

    def read_bytes(self, path, *, max_bytes, read_limit=None):
        target, size = self.inspect_file(path)
        if size > max_bytes:
            raise SshWorkspaceFsError("EFBIG")
        return b"".join(self.stream_file(target, length=read_limit)), size, target

    def write_text(self, *args, **kwargs):
        raise SshWorkspaceFsError("EROFS", "Terminal attachment access is read-only")


def compute_session_fs(durable, backend):
    """Map durable id + bound profile to the actual live UI sid, not the DB row id."""
    from hermes_constants import get_hermes_home
    from tui_gateway import server

    home = str(get_hermes_home())
    owners = [(sid, sess) for sid, sess in list(server._sessions.items())
              if sess.get("session_key") == durable and str(sess.get("profile_home") or "") == home]
    if not owners:
        return None
    if len(owners) != 1:
        raise _unavailable()
    sid, session = owners[0]
    if not session.get("_compute_host_active"):
        return None
    supervisor = server._compute_host_supervisor
    if supervisor is None or not supervisor.is_running():
        raise _unavailable()
    return ComputeTerminalWorkspaceFs(supervisor, sid, durable, home, backend)
