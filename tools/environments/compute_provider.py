"""Compute-provider contracts for task-scoped, capability-aware sandboxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ProcessHandle(Protocol):
    """Minimal process contract shared by local and remote executions."""

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


@runtime_checkable
class DuplexProcess(ProcessHandle, Protocol):
    """A process with readable stdout/stderr and writable stdin."""

    stdin: Any
    stdout: Any
    stderr: Any


@dataclass(frozen=True)
class ServiceRequest:
    """A request routed to a named service in a leased sandbox."""

    service: str
    method: str
    params: Mapping[str, Any] = field(default_factory=dict)
    timeout: float | None = None


@dataclass(frozen=True)
class ServiceResponse:
    """Response returned by a sandbox service."""

    ok: bool
    result: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentCapabilities:
    """Verified capabilities supplied by an environment image."""

    terminal: bool = True
    files: bool = True
    computer_use: bool = False
    process: bool = True
    extras: frozenset[str] = field(default_factory=frozenset)

    def to_capabilities(self) -> frozenset[str]:
        capabilities = set(self.extras)
        for name, enabled in (
            ("terminal", self.terminal),
            ("files", self.files),
            ("computer_use", self.computer_use),
            ("process", self.process),
        ):
            if enabled:
                capabilities.add(name)
        return frozenset(capabilities)


@dataclass(frozen=True)
class ComputeLease:
    """A task-scoped reservation; all task tools must use this lease."""

    task_id: str
    lease_id: str
    provider: str
    image: str
    capabilities: EnvironmentCapabilities
    endpoint: str | None = None
    expires_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ComputeProvider(Protocol):
    """Owns placement, leasing, execution routing, and release."""

    name: str

    def acquire(
        self,
        task_id: str,
        *,
        image: str | None = None,
        capabilities: Sequence[str] | None = None,
    ) -> ComputeLease: ...

    def release(self, lease: ComputeLease) -> None: ...


@runtime_checkable
class ComputerCapableEnvironment(Protocol):
    """Environment extension used by computer_use to share a task lease."""

    def get_computer_backend(self) -> Any: ...

    @property
    def compute_lease(self) -> ComputeLease: ...
