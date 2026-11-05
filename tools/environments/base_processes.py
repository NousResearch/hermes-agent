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

import subprocess
import threading
from typing import Any, Optional

from tools.environments.sandbox_scope import ProcessRef, ProcessStream


class LongLivedProcessMixin:
    """``start_process`` / ``attach_process`` / ``stop_process`` for a :class:`BaseEnvironment`."""

    def _long_lived(self) -> dict[str, subprocess.Popen]:
        table = self.__dict__.get("_long_lived_processes")
        if table is None:
            table = self.__dict__.setdefault("_long_lived_processes", {})
            self.__dict__.setdefault("_long_lived_lock", threading.Lock())
        return table

    def _spawn_long_lived(self, argv: list[str], child_env: Optional[dict]) -> subprocess.Popen:
        from tools.environments.streams import open_stream
        try:
            return open_stream(self, argv, child_env=child_env)  # type: ignore[arg-type]
        except RuntimeError as exc:
            raise NotImplementedError(str(exc)) from exc

    def start_process(self, name: str, argv: list[str], *, child_env: Optional[dict] = None,
                      scope: Any = None) -> ProcessStream:
        """Start *argv* as the long-lived process *name* (stdin/stdout as byte pipes). A name that is
        still running is refused: attach to it instead."""
        table = self._long_lived()
        with self.__dict__["_long_lived_lock"]:
            running = table.get(name)
            if running is not None and running.poll() is None:
                raise ValueError(f"a long-lived process named {name!r} is already running; attach to it")
            table[name] = proc = self._spawn_long_lived(list(argv), child_env)
        return self._stream(name, proc)

    def attach_process(self, name: str, *, scope: Any = None) -> Optional[ProcessStream]:
        """The running process *name* this environment started, or ``None``."""
        with self.__dict__.get("_long_lived_lock") or threading.Lock():
            proc = self._long_lived().get(name)
        return self._stream(name, proc) if proc is not None and proc.poll() is None else None

    def stop_process(self, name: str, *, scope: Any = None) -> bool:
        """Stop *name* (its whole process tree where the backend can); ``False`` if none was running."""
        with self.__dict__.get("_long_lived_lock") or threading.Lock():
            proc = self._long_lived().pop(name, None)
        if proc is None or proc.poll() is not None:
            return False
        self._kill_process(proc)  # type: ignore[attr-defined]
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if (owned_pgid := getattr(proc, "_hermes_owned_pgid", None)) is not None:
            # A host-owned group (``spawn_owned``): we stopped it, so the owner-death supervisor stops covering it.
            from hermes_cli.process_identity import release_process_group_once_empty
            release_process_group_once_empty(owned_pgid, proc=proc)
        return True

    @staticmethod
    def _stream(name: str, proc: subprocess.Popen) -> ProcessStream:
        assert proc.stdin is not None and proc.stdout is not None  # spawned with both pipes
        return ProcessStream(ref=ProcessRef("", name), stdin=proc.stdin, stdout=proc.stdout, poll=proc.poll)
