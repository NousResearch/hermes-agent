"""Opaque identity for a repeatable terminal approval; never a permission by itself."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any


@dataclass
class ApprovalOperation:
    cwd: str
    backend: str
    target: tuple[str, ...]
    requested: bool = False

    @property
    def description(self) -> str:
        connection = "Local" if self.backend == "local" else f"SSH {self.target[2]}@{self.target[0]}:{self.target[1]}"
        return f"{connection}, folder {self.cwd}"

    def matches(self, *, environment: Any, backend: str, cwd: str) -> bool:
        config = approval_environment_config(environment, backend)
        return (config is not None and backend == self.backend and cwd == self.cwd
                and _target(backend, config) == self.target)


_operation: ContextVar[ApprovalOperation | None] = ContextVar("approval_operation", default=None)
MAX_REMEMBER_COMMAND_CHARS = 512
MAX_REMEMBER_CONTEXT_CHARS = 384


def valid_operation_key(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def valid_operation_context(value: Any) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= MAX_REMEMBER_CONTEXT_CHARS
            and bool(value.strip()) and all(ord(char) >= 32 for char in value))


def approval_environment_config(environment: Any, backend: str) -> dict[str, Any] | None:
    """Read the acquired terminal, which can predate the currently loaded config."""
    from tools.environments.local import LocalEnvironment
    from tools.environments.ssh import SSHEnvironment

    if backend == "local" and isinstance(environment, LocalEnvironment):
        return {}
    if backend == "ssh" and isinstance(environment, SSHEnvironment):
        return {f"ssh_{key}": getattr(environment, key) for key in ("host", "port", "user", "key_path")}
    return None


def _target(backend: str, config: Mapping[str, Any]) -> tuple[str, ...] | None:
    if backend == "local":
        return ("local",)
    if backend == "ssh":
        values = tuple(str(config.get(key) or "") for key in ("ssh_host", "ssh_port", "ssh_user", "ssh_key_path"))
        if values[0] and values[2] and all(len(value) <= 1024 for value in values):
            return values
    return None


@contextmanager
def approval_operation(
    *, cwd: str, backend: str, config: Mapping[str, Any], enabled: bool = True,
) -> Iterator[ApprovalOperation | None]:
    """Bind the actual execution context only around a supported foreground guard."""
    target = _target(backend, config) if enabled else None
    absolute = isinstance(cwd, str) and len(cwd) <= 4096 and (
        PurePosixPath(cwd).is_absolute() or PureWindowsPath(cwd).is_absolute())
    context = ApprovalOperation(cwd, backend, target) if target is not None and absolute else None
    token = _operation.set(context)
    try:
        yield context
    finally:
        _operation.reset(token)


def approval_operation_key(command: str, pattern_keys: Sequence[str]) -> str:
    """Hash original bytes before display redaction, including the bound execution context."""
    context = _operation.get()
    if (context is None or not isinstance(command, str) or not command
            or len(command) > MAX_REMEMBER_COMMAND_CHARS or "\n" in command or "\r" in command):
        return ""
    if not valid_operation_context(context.description):
        return ""
    if not pattern_keys or len(pattern_keys) > 32 or any(
        not isinstance(key, str) or not key or len(key) > 4096 for key in pattern_keys
    ):
        return ""
    try:
        encoded = json.dumps({
            "version": 2, "command": command, "cwd": context.cwd, "backend": context.backend,
            "target": context.target, "patterns": sorted(set(pattern_keys)),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except UnicodeEncodeError:
        return ""
    context.requested = True
    return hashlib.sha256(encoded).hexdigest()


def approval_operation_description() -> str:
    context = _operation.get()
    return context.description if context is not None and context.requested else ""
