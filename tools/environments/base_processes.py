"""Long-lived processes inside an environment, with both stdio pipes open.

``execute`` is request/response. A user MCP server (stdio transport) needs the other shape: start a
process once, keep talking to it over stdin/stdout from later calls in the same session, stop it when
it goes idle. Each environment owns the processes it started, by name; nothing here outlives the
environment's own object, and a name is only meaningful inside the environment that started it.

Spawning goes through :meth:`LongLivedProcessMixin._spawn_long_lived`: the local backend runs the
argv on this host with its own run environment; a backend with an exec prefix (docker, ssh, a plugin
declaring ``supports_streams``) runs it inside the sandbox through ``tools.environments.streams``;
SDK backends with neither raise :class:`NotImplementedError`. A hosted backend implements the three
public methods itself (they receive the bound scope, see ``sandbox_scope.ScopedEnvironment``).
"""

from __future__ import annotations

import contextlib
import logging
import shlex
import subprocess
import threading
import uuid
from typing import Any, Optional

from tools.environments.sandbox_scope import ProcessRef, ProcessStream

logger = logging.getLogger("tools.environments")


class LongLivedProcessMixin:
    """``start_process`` / ``attach_process`` / ``stop_process`` for a :class:`BaseEnvironment`."""

    def _long_lived(self) -> dict[str, subprocess.Popen]:
        state = self.__dict__.get("_long_lived_state")
        if state is None:
            # One owned state object, installed by a single setdefault: a concurrent first start
            # can never see the table without the lock, and no fallback lock can differ from
            # the starter's.
            state = self.__dict__.setdefault(
                "_long_lived_state", {"table": {}, "lock": threading.Lock()})
        return state["table"]

    def _long_lived_lock(self) -> threading.Lock:
        state = self.__dict__.get("_long_lived_state")
        if state is None:
            state = self.__dict__.setdefault(
                "_long_lived_state", {"table": {}, "lock": threading.Lock()})
        return state["lock"]

    def _spawn_long_lived(self, argv: list[str], child_env: Optional[dict]) -> subprocess.Popen:
        from tools.environments.streams import (exec_prefix, open_stream, sandbox_temp_dir, shell_joined)
        try:
            prefix = exec_prefix(self)  # type: ignore[arg-type]  # a BaseEnvironment at runtime, the mixin in isolation
            if prefix is None:
                return open_stream(self, argv, child_env=child_env)  # type: ignore[arg-type]
            # A backend with an exec prefix (docker exec -i, ssh, a plugin prefix) runs the service
            # inside the sandbox behind a LOCAL client; killing that client need not stop it. The
            # service's pid is therefore recorded inside the sandbox before it execs, and
            # _reap_long_lived stops it there before closing the client. The transport shape is
            # open_stream's own (``exports; exec argv``): stdin/stdout stay the service's.
            quoted = " ".join(shlex.quote(str(word)) for word in argv)
            pidfile = f"{sandbox_temp_dir(self)}/hermes_llp_{uuid.uuid4().hex[:16]}.pid"  # type: ignore[arg-type]
            exports = " ".join(f"export {key}={shlex.quote(str(value))};" for key, value in (child_env or {}).items())
            script = f"{exports} echo $$ > {shlex.quote(pidfile)} 2>/dev/null ; exec {quoted}"
            if shell_joined(self):  # type: ignore[arg-type]
                command = [*prefix, shlex.join(["bash", "-c", script])]
            else:
                command = [*prefix, "bash", "-c", script]
            proc = subprocess.Popen(  # windows-footgun: ok — the prefix is a Linux sandbox's client (docker/ssh/apptainer)
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, close_fds=True)
            proc._hermes_remote_pidfile = pidfile  # type: ignore[attr-defined]  # teardown metadata, not a pipe
            return proc
        except RuntimeError as exc:
            raise NotImplementedError(str(exc)) from exc

    def start_process(self, name: str, argv: list[str], *, child_env: Optional[dict] = None,
                      scope: Any = None) -> ProcessStream:
        """Start *argv* as the long-lived process *name* (stdin/stdout as byte pipes). A name that is
        still running is refused: attach to it instead. A name whose record is left over from a
        launcher that exited goes to the new process only after that record is reaped: its children
        and pipes are this environment's to stop, and overwriting them would orphan both."""
        table = self._long_lived()
        with self._long_lived_lock():
            running = table.get(name)
            if running is not None and running.poll() is None:
                raise ValueError(f"a long-lived process named {name!r} is already running; attach to it")
            stale = table.pop(name, None)
            table[name] = proc = self._spawn_long_lived(list(argv), child_env)
        if stale is not None:
            self._reap_long_lived(stale, scope)
        return self._stream(name, proc)

    def attach_process(self, name: str, *, scope: Any = None) -> Optional[ProcessStream]:
        """The running process *name* this environment started, or ``None``."""
        with self._long_lived_lock():
            proc = self._long_lived().get(name)
        return self._stream(name, proc) if proc is not None and proc.poll() is None else None

    def stop_process(self, name: str, *, scope: Any = None) -> bool:
        """Stop *name* (its whole process tree where the backend can); ``False`` if none was running.
        A leader that already exited still has its group swept: a descendant it left behind is
        stopped too, and the record goes only once that is done."""
        with self._long_lived_lock():
            proc = self._long_lived().pop(name, None)
        if proc is None:
            return False
        was_running = proc.poll() is None
        self._reap_long_lived(proc, scope)
        return was_running

    def stop_all_processes(self, *, scope: Any = None) -> None:
        """Stop and reap every long-lived process this environment started (environment cleanup)."""
        with self._long_lived_lock():
            procs = list(self._long_lived().values())
            self._long_lived().clear()
        for proc in procs:
            self._reap_long_lived(proc, scope)

    def _stop_remote_service(self, proc: subprocess.Popen, scope: Any) -> None:
        """Best-effort stop of a service that runs inside a backend (docker exec / ssh / a plugin
        prefix): its pid was recorded in the sandbox at spawn; kill it there, where the local
        client's death alone would leave it running."""
        pidfile = getattr(proc, "_hermes_remote_pidfile", None)
        execute = getattr(self, "execute", None)
        if pidfile is None or not callable(execute):
            return
        quoted = shlex.quote(pidfile)
        command = (f"pkill -TERM -P $(cat {quoted}) 2>/dev/null; kill -TERM $(cat {quoted}) 2>/dev/null; "
                   f"rm -f {quoted}")
        try:
            from tools.environments.sandbox_scope import accepts_scope
            if scope is not None and accepts_scope(execute):
                execute(command, scope=scope)
            else:
                execute(command)
        except Exception:
            # Best-effort by design (the client kill below still ends the transport), but never a
            # silent one: a service that survived its own stop is the next bug report.
            logger.warning("long-lived service stop command failed for %s", pidfile, exc_info=True)

    def _reap_long_lived(self, proc: subprocess.Popen, scope: Any = None) -> None:
        """Kill *proc*'s tree, reap it, close its pipes, then release a host-owned group. A backend
        service (pid recorded in the sandbox at spawn) is stopped there first: the local client's
        death need not stop it."""
        self._stop_remote_service(proc, scope)
        owned_pgid = getattr(proc, "_hermes_owned_pgid", None)
        if proc.poll() is None or owned_pgid is not None:
            self._kill_process(proc)  # type: ignore[attr-defined]  # signals the group even with the leader gone
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()
        if owned_pgid is not None:
            # A host-owned group (``spawn_owned``): stopped, so the owner-death supervisor stops covering it
            # once it is empty (a member that survived both signals stays covered and dies with Hermes).
            from hermes_cli.process_identity import release_process_group_once_empty
            release_process_group_once_empty(owned_pgid, proc=proc)

    @staticmethod
    def _stream(name: str, proc: subprocess.Popen) -> ProcessStream:
        assert proc.stdin is not None and proc.stdout is not None  # spawned with both pipes
        return ProcessStream(ref=ProcessRef("", name), stdin=proc.stdin, stdout=proc.stdout, poll=proc.poll)
