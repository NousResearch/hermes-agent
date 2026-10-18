"""Private, bounded tool-to-execution-environment correlation for Relay.

Docker's native container ID is hashed into a stable join key that an external
collector can also compute. The added annotation never includes the native ID;
tool output is unchanged and is not sanitized here.
"""

from __future__ import annotations

import contextvars
import hashlib
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

_DOCKER_ID = re.compile(r"[0-9a-f]{64}\Z")
_DOCKER_DOMAIN = b"hermes.execution_environment/oci-container/v1\0"
_DOCKER_SCHEME = "oci_container_id_sha256_128_v1"


def opaque_docker_resource_id(container_id: str | None) -> str | None:
    """Return a versioned hash join key only for a full Docker ID."""
    if not isinstance(container_id, str) or _DOCKER_ID.fullmatch(container_id) is None:
        return None
    digest = hashlib.sha256(_DOCKER_DOMAIN + bytes.fromhex(container_id)).digest()
    return digest[:16].hex()


@dataclass
class ToolExecutionResources:
    """Collect observed Docker exec targets within one callback, conservatively."""

    docker_ids: set[str] = field(default_factory=set)
    unknown_docker_attempt: bool = False
    closed: bool = False
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def record_docker(self, container_id: str | None) -> None:
        resource_id = opaque_docker_resource_id(container_id)
        with self._lock:
            if self.closed:
                return
            if resource_id is None:
                self.unknown_docker_attempt = True
            elif len(self.docker_ids) < 2:
                self.docker_ids.add(resource_id)

    def seal(self) -> None:
        """Ignore late deadline/background work after the callback has returned."""
        with self._lock:
            self.closed = True

    def annotation(self) -> dict | None:
        with self._lock:
            docker_ids = self.docker_ids.copy()
            unknown_attempt = self.unknown_docker_attempt
        if not docker_ids and not unknown_attempt:
            return None
        payload = {
            "schema_version": "1",
            "backend": "docker",
            "association": "docker_exec_target",
        }
        if len(docker_ids) == 1 and not unknown_attempt:
            payload.update(
                resource_id=next(iter(docker_ids)),
                resource_id_scheme=_DOCKER_SCHEME,
                resource_id_status="available",
            )
        elif len(docker_ids) > 1:
            payload["resource_id_status"] = "ambiguous_multiple"
        elif docker_ids:
            payload["resource_id_status"] = "ambiguous_unknown"
        else:
            payload["resource_id_status"] = "unavailable"
        return {"hermes.execution_environment": payload}


_TOOL_RESOURCES: contextvars.ContextVar[ToolExecutionResources | None] = (
    contextvars.ContextVar("hermes_tool_execution_resources", default=None)
)
_DOCKER_ATTEMPT: contextvars.ContextVar[list[str | None] | None] = (
    contextvars.ContextVar("hermes_docker_exec_attempt", default=None)
)


@contextmanager
def collect_tool_resources() -> Iterator[ToolExecutionResources]:
    """Bind a fresh collector across copied callback and deadline contexts."""
    collector = ToolExecutionResources()
    token = _TOOL_RESOURCES.set(collector)
    try:
        yield collector
    finally:
        collector.seal()
        _TOOL_RESOURCES.reset(token)


@contextmanager
def capture_docker_attempt() -> Iterator[list[str | None]]:
    """Capture the ID in the actual ``docker exec`` argv, including deadline workers."""
    attempts: list[str | None] = []
    token = _DOCKER_ATTEMPT.set(attempts)
    try:
        yield attempts
    finally:
        _DOCKER_ATTEMPT.reset(token)


@contextmanager
def suppress_docker_attempt_capture() -> Iterator[None]:
    """Do not mistake a preparation probe for the tool's main command."""
    token = _DOCKER_ATTEMPT.set(None)
    try:
        yield
    finally:
        _DOCKER_ATTEMPT.reset(token)


def docker_exec_started(container_id: str | None) -> None:
    """Called after spawning docker exec; the holder survives context copies."""
    attempts = _DOCKER_ATTEMPT.get()
    if attempts is not None:
        attempts.append(container_id)


def record_docker_result(container_id: str | None) -> None:
    """Record one spawned Docker CLI target, including pre-recovery attempts."""
    collector = _TOOL_RESOURCES.get()
    if collector is not None:
        collector.record_docker(container_id)
