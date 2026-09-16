"""Integrity boundary for execution tools that can mutate Hermes config directly.

File tools reject the active profile's ``config.yaml`` before writing, but code
and shell children can use ordinary filesystem APIs.  Snapshotting at the
trusted parent boundary lets those tools detect and roll back a write without
trying to statically parse arbitrary Python or shell syntax.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path


_REFUSAL = (
    "Blocked: the execution modified the active Hermes config.yaml. The change "
    "was rolled back because agents cannot modify security-sensitive "
    "configuration. Edit config.yaml directly or use 'hermes config' outside "
    "the agent."
)


@dataclass
class ActiveConfigSnapshot:
    """Bytes and metadata needed to verify and restore one active config file."""

    path: Path
    existed: bool
    content: bytes
    mode: int | None
    identity: tuple[int, int] | None

    @classmethod
    def capture(cls) -> tuple[ActiveConfigSnapshot | None, str | None]:
        """Capture the active profile config, failing closed on unreadable state."""
        try:
            from hermes_cli.config import get_config_path

            path = get_config_path().resolve(strict=False)
            try:
                info = path.stat()
            except FileNotFoundError:
                return cls(
                    path=path, existed=False, content=b"", mode=None, identity=None
                ), None
            if path.is_symlink() or not path.is_file():
                return None, f"execute tool refused: Hermes config path is not a regular file: {path}"
            return cls(
                path=path,
                existed=True,
                content=path.read_bytes(),
                mode=stat.S_IMODE(info.st_mode),
                identity=(info.st_dev, info.st_ino),
            ), None
        except OSError as exc:
            return None, f"execute tool refused: could not snapshot Hermes config.yaml: {exc}"

    def restore_if_changed(self) -> str | None:
        """Restore the snapshot atomically and return a refusal when it changed."""
        try:
            current_exists = self.path.exists()
            current_info = self.path.stat() if current_exists else None
            current_regular = current_exists and not self.path.is_symlink() and self.path.is_file()
            current = self.path.read_bytes() if current_regular else None
            current_mode = stat.S_IMODE(current_info.st_mode) if current_info else None
            current_identity = (
                (current_info.st_dev, current_info.st_ino) if current_info else None
            )
        except OSError:
            current_exists, current = True, None
            current_mode, current_identity = None, None

        expected = self.content if self.existed else None
        if (
            current_exists == self.existed
            and current == expected
            and current_mode == self.mode
            and current_identity == self.identity
        ):
            return None

        try:
            if self.existed:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{self.path.name}.restore-", dir=str(self.path.parent)
                )
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(self.content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if self.mode is not None:
                        os.chmod(temp_name, self.mode)
                    os.replace(temp_name, self.path)
                finally:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
            elif self.path.is_file() or self.path.is_symlink():
                self.path.unlink()
            else:
                raise OSError("config path was replaced by a non-file entry")
        except OSError as exc:
            return f"{_REFUSAL} WARNING: automatic rollback failed: {exc}"
        return _REFUSAL
