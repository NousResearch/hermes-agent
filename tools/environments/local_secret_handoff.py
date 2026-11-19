"""Secret exports a local shell carries between commands in memory, never in the snapshot file.

The session snapshot (``hermes-snap-*.sh``) carries a terminal session's exported variables
from one command to the next on disk. Secret values never go there (#62336): the dump writes
each secret export to an inherited pipe (``base_session_env._secret_handoff_snippet``), this
store keeps the values per snapshot path, and the next command receives them as process env.
The shell sees the same variables as before; the file on disk has none of the values.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable

from tools.environments.base_session_env import _SHELL_ENV_NAME_RE

# One dump's ceiling: a runaway export must not grow the host process without bound.
_MAX_HANDOFF_BYTES = 16 * 1024 * 1024


def profile_secret_names() -> tuple[str, ...]:
    """Env names the served profile's ``.env`` and external secret sources define. ``.env`` is
    secrets-only, so these are credentials whatever they are called (``PORTAL_CLIENT_SECRET``
    needs no naming convention). The bound scope under multiplex, else the active home's file;
    ``load_env_file`` is memoised on the file's bytes, so a call per command costs one read."""
    from agent.secret_scope import current_secret_scope, load_env_file
    from hermes_cli.env_loader import source_supplied_names
    from hermes_constants import get_hermes_home

    scope = current_secret_scope()
    names = set(scope) if scope is not None else set(load_env_file(get_hermes_home() / ".env"))
    names.update(source_supplied_names())
    return tuple(sorted(n for n in names if isinstance(n, str) and _SHELL_ENV_NAME_RE.fullmatch(n)))


def _parse_records(data: bytes) -> dict[str, str] | None:
    """``NAME=value\\0`` records closed by one empty record; None when the dump never closed them
    (killed mid-write, or it did not run), so a cut-off record never plants a partial value."""
    if data.startswith(b"\0"):
        body = b""
    else:
        end = data.find(b"\0\0")
        if end < 0:
            return None
        body = data[:end]
    values: dict[str, str] = {}
    for record in body.split(b"\0") if body else ():
        name, sep, value = record.decode("utf-8", "surrogateescape").partition("=")
        if sep and _SHELL_ENV_NAME_RE.fullmatch(name):
            values[name] = value
    return values


class SecretHandoff:
    """Per environment: snapshot path -> the secret exports that snapshot's last dump handed off."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_snapshot: dict[str, dict[str, str]] = {}

    def values_for(self, snapshot_paths: Iterable[str]) -> dict[str, str]:
        """The exports recorded for the first of *snapshot_paths* that has a record: a delegated
        child's own snapshot, then its ancestors', the order the wrapper sources the files in."""
        with self._lock:
            for path in snapshot_paths:
                if path in self._by_snapshot:
                    return dict(self._by_snapshot[path])
        return {}

    def start_reader(self, snapshot_path: str) -> tuple[int, threading.Thread]:
        """Open the channel for one spawn: returns the inheritable write fd to hand the child and the
        reader thread (already running) that records what arrives for *snapshot_path*. The thread
        must run while the child does, since a pipe holds only ~64 KB before the writer blocks."""
        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)
        # health: allow HX012 -- reads one pipe into this object; no secret scope, config or home is read
        reader = threading.Thread(
            target=self._collect, args=(read_fd, snapshot_path), name="hermes-secret-handoff", daemon=True)
        reader.start()
        return write_fd, reader

    def _collect(self, read_fd: int, snapshot_path: str) -> None:
        """Read to the closing empty record or EOF. A dump that never reached the hand-off keeps the
        previous values, as the file on disk keeps its last good snapshot when a re-dump fails."""
        data = b""
        try:
            while len(data) <= _MAX_HANDOFF_BYTES and not (data == b"\0" or data.endswith(b"\0\0")):
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                data += chunk
        except OSError:
            return
        finally:
            os.close(read_fd)
        values = _parse_records(data)
        if values is not None:
            with self._lock:
                self._by_snapshot[snapshot_path] = values

    def forget(self) -> None:
        with self._lock:
            self._by_snapshot.clear()
