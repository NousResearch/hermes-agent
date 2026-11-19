"""Secret exports a local shell carries between commands in memory, never in the snapshot file.

The session snapshot (``hermes-snap-*.sh``) carries a terminal session's exported variables
from one command to the next on disk. Secret values never go there (#62336): the dump sends each
secret export's ``declare -p`` text to an inherited pipe and writes only a generation id into the
file (``base_session_env._secret_handoff_snippet``). This store keeps the declarations per
generation; the next command receives them as ``__HERMES_SD_<generation>_<NAME>`` env entries and
applies only those of the generation its sourced file names, so the file and its secrets always
come from one dump. The shell sees the same variables (arrays and attributes included) as before.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Iterable

from tools.environments.base_session_env import _SHELL_ENV_NAME_RE

# One dump's ceiling: a runaway export must not grow the host process without bound.
_MAX_HANDOFF_BYTES = 16 * 1024 * 1024
# Generations kept per snapshot file: the file on disk names one of its recent dumps, and with
# overlapping commands the last one to publish need not be the last one to report.
_GENERATIONS_KEPT = 8
_DECL_ENV_PREFIX = "__HERMES_SD_"


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
    """``NAME=<declaration>\\0`` records closed by one empty record; None when the dump never
    closed them (killed mid-write, a failed write, or it did not run), so a cut-off record never
    plants a partial value."""
    if data.startswith(b"\0"):
        body = b""
    else:
        end = data.find(b"\0\0")
        if end < 0:
            return None
        body = data[:end]
    decls: dict[str, str] = {}
    for record in body.split(b"\0") if body else ():
        name, sep, decl = record.decode("utf-8", "surrogateescape").partition("=")
        if sep and _SHELL_ENV_NAME_RE.fullmatch(name):
            decls[name] = decl
    return decls


class SecretHandoff:
    """Per environment: snapshot path -> its recent dumps' secret declarations, by generation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_snapshot: dict[str, dict[str, dict[str, str]]] = {}
        self._names: set[str] = set()

    @staticmethod
    def new_generation() -> str:
        return uuid.uuid4().hex

    def declaration_env(self, snapshot_paths: Iterable[str], skip: Iterable[str] = ()) -> dict[str, str]:
        """``__HERMES_SD_<generation>_<NAME>`` entries for every kept generation of the files this
        command may source; the shell applies only the generation its file names."""
        skipped = set(skip)
        with self._lock:
            return {f"{_DECL_ENV_PREFIX}{gen}_{name}": decl
                    for path in snapshot_paths for gen, decls in self._by_snapshot.get(path, {}).items()
                    for name, decl in decls.items() if name not in skipped}

    def start_reader(self, snapshot_path: str, generation: str) -> tuple[int, threading.Thread]:
        """Open the channel for one spawn: returns the inheritable write fd to hand the child and the
        reader thread (already running) that records what arrives as *generation* of *snapshot_path*.
        The thread must run while the child does, since a pipe holds only ~64 KB before the writer
        blocks."""
        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)
        # health: allow HX012 -- reads one pipe into this object; no secret scope, config or home is read
        reader = threading.Thread(
            target=self._collect, args=(read_fd, snapshot_path, generation), name="hermes-secret-handoff",
            daemon=True)
        reader.start()
        return write_fd, reader

    def _collect(self, read_fd: int, snapshot_path: str, generation: str) -> None:
        """Read to the closing empty record or EOF, then keep the set under its generation."""
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
        decls = _parse_records(data)
        if decls is None:
            return
        with self._lock:
            kept = self._by_snapshot.setdefault(snapshot_path, {})
            kept[generation] = decls
            while len(kept) > _GENERATIONS_KEPT:
                kept.pop(next(iter(kept)))
            self._names.update(decls)

    def names(self) -> set[str]:
        """Every name a dump ever handed off: it stays a secret for the environment's lifetime, even
        after it is unset and leaves ``.env``, so a later re-export never lands on disk."""
        with self._lock:
            return set(self._names)

    def forget(self) -> None:
        with self._lock:
            self._by_snapshot.clear()
            self._names.clear()
