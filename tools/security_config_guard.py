"""Detect active-config mutations across unrestricted execution boundaries.

File tools reject the active profile's ``config.yaml`` before writing, but code
and shell children can use ordinary filesystem APIs.  A parent-owned snapshot
can report mutations without treating an old snapshot as authority to overwrite
a newer configuration generation.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path


_REFUSAL = (
    "Blocked: the execution modified the active Hermes config.yaml. The change "
    "was not rolled back because it cannot be distinguished safely from a "
    "concurrent trusted edit. Inspect config.yaml before continuing, and use "
    "'hermes config' outside the agent for intended changes."
)


@dataclass(frozen=True)
class _ConfigState:
    """Namespace-entry and resolved-target state for one config path."""

    entry_identity: tuple[int, int, int] | None
    link_target: str | None
    resolved_path: Path
    target_identity: tuple[int, int, int] | None
    content: bytes | None


@dataclass(frozen=True)
class ActiveConfigSnapshot:
    """A mutation detector for the active config namespace entry and its target."""

    path: Path
    state: _ConfigState

    @staticmethod
    def _read_state(path: Path) -> _ConfigState:
        if not os.path.lexists(path):
            return _ConfigState(None, None, path.resolve(strict=False), None, None)

        entry = path.lstat()
        link_target = os.readlink(path) if stat.S_ISLNK(entry.st_mode) else None
        resolved = path.resolve(strict=False)
        if not resolved.exists():
            return _ConfigState(
                (entry.st_dev, entry.st_ino, entry.st_mode),
                link_target,
                resolved,
                None,
                None,
            )

        target = resolved.stat()
        if not stat.S_ISREG(target.st_mode):
            raise OSError(f"Hermes config path is not a regular file: {path}")
        return _ConfigState(
            (entry.st_dev, entry.st_ino, entry.st_mode),
            link_target,
            resolved,
            (target.st_dev, target.st_ino, target.st_mode),
            resolved.read_bytes(),
        )

    @classmethod
    def capture(cls) -> tuple[ActiveConfigSnapshot | None, str | None]:
        """Capture the active profile config, failing closed on unreadable state."""
        try:
            from hermes_cli.config import get_config_path

            path = get_config_path().absolute()
            return cls(path=path, state=cls._read_state(path)), None
        except OSError as exc:
            return None, f"execute tool refused: could not snapshot Hermes config.yaml: {exc}"

    def mutation_error(self) -> str | None:
        """Return a refusal when either the namespace entry or target changed."""
        try:
            current = self._read_state(self.path)
        except OSError as exc:
            return f"{_REFUSAL} Snapshot comparison failed: {exc}"
        return None if current == self.state else _REFUSAL
