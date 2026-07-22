"""Process-wide task-to-desktop lease manager."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Sequence

from tools.environments.compute_provider import ComputeLease


@dataclass
class DesktopSandboxLease:
    """One task's environment and the provider reservation backing it."""

    task_id: str
    lease: ComputeLease
    environment: Any
    references: int = 1


class DesktopSandboxManager:
    """Ensures terminal and computer-use resolve the same task lease."""

    def __init__(self, provider: Any):
        self.provider = provider
        self._leases: dict[str, DesktopSandboxLease] = {}
        self._lock = threading.RLock()

    def acquire(self, task_id: str, *, image: str | None = None,
                capabilities: Sequence[str] | None = None) -> DesktopSandboxLease:
        with self._lock:
            existing = self._leases.get(task_id)
            if existing is not None:
                existing.references += 1
                return existing
            lease = self.provider.acquire(task_id, image=image, capabilities=capabilities)
            try:
                environment = self.provider.create_environment(lease)
            except Exception:
                self.provider.release(lease)
                raise
            managed = DesktopSandboxLease(task_id=task_id, lease=lease, environment=environment)
            self._leases[task_id] = managed
            return managed

    def get(self, task_id: str) -> DesktopSandboxLease | None:
        with self._lock:
            return self._leases.get(task_id)

    def release(self, task_id: str) -> None:
        with self._lock:
            managed = self._leases.get(task_id)
            if managed is None:
                return
            managed.references -= 1
            if managed.references > 0:
                return
            self._leases.pop(task_id, None)
        try:
            cleanup = getattr(managed.environment, "cleanup", None)
            if callable(cleanup):
                cleanup()
        finally:
            self.provider.release(managed.lease)

    def status(self, task_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            leases = [self._leases[task_id]] if task_id in self._leases else list(self._leases.values()) if task_id is None else []
            return {
                "count": len(leases),
                "leases": [
                    {"task_id": item.task_id, "lease_id": item.lease.lease_id,
                     "provider": item.lease.provider, "image": item.lease.image,
                     "capabilities": sorted(item.lease.capabilities.to_capabilities()),
                     "references": item.references}
                    for item in leases
                ],
            }


_manager: DesktopSandboxManager | None = None
_manager_lock = threading.Lock()


def get_desktop_sandbox_manager(provider: Any | None = None) -> DesktopSandboxManager:
    """Return the process-wide manager, constructing the Modal PoC by default."""
    global _manager
    with _manager_lock:
        if _manager is None:
            if provider is None:
                from tools.environments.modal_desktop import ModalDesktopProvider
                provider = ModalDesktopProvider()
            _manager = DesktopSandboxManager(provider)
        return _manager
