"""Progressive Operational Capabilities: deterministic operational procedures and composition.

Capabilities are learned from verified experience, promoted through validation,
composed deterministically, and reused automatically without paying LLM tokens.
Reuses existing ArtifactStore and ProceduralMemory/RecipeStore persistence planes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable

from hermes_constants import get_hermes_home
from workstation.artifacts import ArtifactStore
from workstation.recipes import sanitize, digest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CapabilityLifecycle(str, Enum):
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    RETIRED = "retired"


class CapabilityError(Exception):
    """Base error for operational capabilities."""
    pass


class CapabilityNotFoundError(CapabilityError):
    """Requested capability was not found."""
    pass


class CapabilityCycleError(CapabilityError):
    """Circular dependency detected in capability composition."""
    pass


class CapabilityDepthExceededError(CapabilityError):
    """Capability dependency resolution exceeded maximum allowed depth."""
    pass


class CapabilityDriftError(CapabilityError):
    """Capability execution drifted or failed preconditions/postconditions."""
    pass


class CapabilityValidationError(CapabilityError):
    """Capability structure or contract is invalid."""
    pass


def matches_version(version: str, constraint: str | None) -> bool:
    """Check if version matches a semver constraint (*, ^X.Y, >=X.Y, exact)."""
    if not constraint or constraint in {"*", "", "latest"}:
        return True
    version = str(version).strip().lstrip("v")
    constraint = str(constraint).strip().lstrip("v")
    if constraint == version:
        return True
    if constraint.startswith("^"):
        base = constraint[1:]
        base_major = base.split(".")[0]
        v_major = version.split(".")[0]
        return base_major == v_major
    if constraint.startswith(">="):
        min_v = constraint[2:]
        try:
            v_parts = [int(p) for p in version.split(".")[:3]]
            min_parts = [int(p) for p in min_v.split(".")[:3]]
            return v_parts >= min_parts
        except ValueError:
            return version >= min_v
    return version.startswith(constraint)


@dataclass
class CapabilityDependency:
    capability_id: str
    version_constraint: str = "*"
    input_mappings: dict[str, str] = field(default_factory=dict)
    output_alias: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "version_constraint": self.version_constraint,
            "input_mappings": dict(self.input_mappings),
            "output_alias": self.output_alias,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityDependency:
        return cls(
            capability_id=str(data.get("capability_id", "")),
            version_constraint=str(data.get("version_constraint", "*")),
            input_mappings=dict(data.get("input_mappings", {})),
            output_alias=data.get("output_alias"),
        )


@dataclass
class OperationalCapability:
    id: str
    name: str
    version: str = "1.0.0"
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    effect: str = "state_mutation"  # read_only, state_mutation, external_mutation
    route: str = "native_browser"  # native_browser, filesystem, host_process, composite
    scope: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    verifier_contract: dict[str, Any] = field(default_factory=dict)
    dependencies: list[CapabilityDependency] = field(default_factory=list)
    implementation: dict[str, Any] = field(default_factory=dict)
    lifecycle: CapabilityLifecycle = CapabilityLifecycle.DISCOVERED
    provenance: dict[str, Any] = field(default_factory=dict)
    semantic_fingerprint: str = ""
    compatibility_fingerprint: str = ""
    validation_evidence: list[dict[str, Any]] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    drift_state: str = "healthy"  # healthy, quarantined, retired
    savings: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "effect": self.effect,
            "route": self.route,
            "scope": self.scope,
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "verifier_contract": self.verifier_contract,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "implementation": self.implementation,
            "lifecycle": self.lifecycle.value,
            "provenance": self.provenance,
            "semantic_fingerprint": self.semantic_fingerprint,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "validation_evidence": list(self.validation_evidence),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "drift_state": self.drift_state,
            "savings": dict(self.savings),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperationalCapability:
        lifecycle_raw = data.get("lifecycle", CapabilityLifecycle.DISCOVERED.value)
        try:
            lifecycle = CapabilityLifecycle(lifecycle_raw)
        except ValueError:
            lifecycle = CapabilityLifecycle.DISCOVERED
        deps_data = data.get("dependencies", [])
        deps = [CapabilityDependency.from_dict(d) if isinstance(d, dict) else d for d in deps_data]
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            version=str(data.get("version", "1.0.0")),
            input_schema=dict(data.get("input_schema", {})),
            output_schema=dict(data.get("output_schema", {})),
            effect=str(data.get("effect", "state_mutation")),
            route=str(data.get("route", "native_browser")),
            scope=dict(data.get("scope", {})),
            preconditions=list(data.get("preconditions", [])),
            postconditions=list(data.get("postconditions", [])),
            verifier_contract=dict(data.get("verifier_contract", {})),
            dependencies=deps,
            implementation=dict(data.get("implementation", {})),
            lifecycle=lifecycle,
            provenance=dict(data.get("provenance", {})),
            semantic_fingerprint=str(data.get("semantic_fingerprint", "")),
            compatibility_fingerprint=str(data.get("compatibility_fingerprint", "")),
            validation_evidence=list(data.get("validation_evidence", [])),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            drift_state=str(data.get("drift_state", "healthy")),
            savings=dict(data.get("savings", {})),
            created_at=str(data.get("created_at", _utc_now())),
            updated_at=str(data.get("updated_at", _utc_now())),
        )


class OperationalCapabilityRegistry:
    """Registry for deterministic operational capabilities.
    
    Persistence is backed by the existing ArtifactStore and a filesystem index,
    maintaining compatibility with ProceduralMemory and RecipeStore.
    """
    def __init__(self, artifacts: ArtifactStore | None = None, root: Path | str | None = None):
        self.artifacts = artifacts or ArtifactStore()
        self.root = Path(root) if root else self.artifacts.root.parent / "operational_capabilities"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.json"
        self._lock = threading.RLock()

    @contextmanager
    def _transaction(self):
        with self._lock, (self.root / "index.lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                if not lock.read(1):
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_file.exists():
            return {}
        try:
            return json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def register(self, capability: OperationalCapability) -> OperationalCapability:
        """Register a new or updated operational capability."""
        if not capability.id:
            raise CapabilityValidationError("Capability must have a non-empty id")
        capability.updated_at = _utc_now()
        sanitized = sanitize(capability.to_dict())
        sha = digest(sanitized)
        with self._transaction():
            ref = self.artifacts.store("operational_capabilities", f"{capability.id}_{capability.version}_{sha[:16]}.json", sanitized)
            index = self._load_index()
            entry_key = f"{capability.id}@{capability.version}"
            index[entry_key] = {
                "id": capability.id,
                "version": capability.version,
                "name": capability.name,
                "route": capability.route,
                "lifecycle": capability.lifecycle.value,
                "drift_state": capability.drift_state,
                "semantic_fingerprint": capability.semantic_fingerprint,
                "compatibility_fingerprint": capability.compatibility_fingerprint,
                "scope": capability.scope,
                "ref": ref.ref,
                "sha256": sha,
                "updated_at": capability.updated_at,
            }
            temp = self.index_file.with_name(f"index.{os.getpid()}.{threading.get_ident()}.tmp")
            with temp.open("w", encoding="utf-8") as stream:
                json.dump(index, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            temp.replace(self.index_file)
        return capability

    def get(self, capability_id: str, version: str | None = None) -> OperationalCapability | None:
        """Retrieve capability by id and optional version/constraint."""
        with self._transaction():
            index = self._load_index()
            candidates = []
            for key, entry in index.items():
                if entry.get("id") == capability_id:
                    if matches_version(entry.get("version", "1.0.0"), version):
                        candidates.append(entry)
            if not candidates:
                return None
            candidates.sort(key=lambda e: e.get("version", "0"), reverse=True)
            chosen = candidates[0]
            body = self.artifacts.read_json(chosen["ref"])
            return OperationalCapability.from_dict(body)

    def list_capabilities(self, lifecycle: CapabilityLifecycle | None = None, route: str | None = None) -> list[OperationalCapability]:
        with self._transaction():
            index = self._load_index()
            results = []
            for entry in index.values():
                if lifecycle and entry.get("lifecycle") != lifecycle.value:
                    continue
                if route and entry.get("route") != route:
                    continue
                body = self.artifacts.read_json(entry["ref"])
                results.append(OperationalCapability.from_dict(body))
            return results

    def record_validation(self, capability_id: str, evidence: dict[str, Any], *, auto_promote_threshold: int = 2) -> OperationalCapability:
        cap = self.get(capability_id)
        if not cap:
            raise CapabilityNotFoundError(f"Capability '{capability_id}' not found")
        cap.validation_evidence.append(sanitize(evidence))
        cap.success_count += 1
        if cap.lifecycle == CapabilityLifecycle.DISCOVERED:
            cap.lifecycle = CapabilityLifecycle.VALIDATED
        if cap.success_count >= auto_promote_threshold and cap.lifecycle == CapabilityLifecycle.VALIDATED and cap.drift_state == "healthy":
            cap.lifecycle = CapabilityLifecycle.PROMOTED
        return self.register(cap)

    def promote(self, capability_id: str, *, version: str | None = None) -> OperationalCapability:
        cap = self.get(capability_id, version)
        if not cap:
            raise CapabilityNotFoundError(f"Capability '{capability_id}' not found")
        cap.lifecycle = CapabilityLifecycle.PROMOTED
        return self.register(cap)

    def record_drift(self, capability_id: str, reason: str, *, quarantine: bool = True) -> OperationalCapability:
        cap = self.get(capability_id)
        if not cap:
            raise CapabilityNotFoundError(f"Capability '{capability_id}' not found")
        cap.failure_count += 1
        if quarantine:
            cap.drift_state = "quarantined"
            cap.lifecycle = CapabilityLifecycle.VALIDATED
        evidence = {"timestamp": _utc_now(), "drift_reason": reason, "status": "quarantined" if quarantine else "failed"}
        cap.validation_evidence.append(evidence)
        return self.register(cap)

    def retire(self, capability_id: str, reason: str = "") -> OperationalCapability:
        cap = self.get(capability_id)
        if not cap:
            raise CapabilityNotFoundError(f"Capability '{capability_id}' not found")
        cap.lifecycle = CapabilityLifecycle.RETIRED
        cap.drift_state = "retired"
        return self.register(cap)

    def find_matching(self, route: str, semantic_fingerprint: str, scope: dict[str, Any] | None = None, promoted_only: bool = True) -> OperationalCapability | None:
        """Find matching capability by route, semantic fingerprint, and scope."""
        with self._transaction():
            index = self._load_index()
            for entry in index.values():
                if entry.get("route") != route:
                    continue
                if entry.get("semantic_fingerprint") != semantic_fingerprint:
                    continue
                if promoted_only and entry.get("lifecycle") != CapabilityLifecycle.PROMOTED.value:
                    continue
                if entry.get("drift_state") != "healthy":
                    continue
                if scope:
                    entry_scope = entry.get("scope", {})
                    if any(entry_scope.get(k) != v for k, v in scope.items()):
                        continue
                body = self.artifacts.read_json(entry["ref"])
                return OperationalCapability.from_dict(body)
        return None


class CapabilityResolver:
    """Resolves and linearizes capability dependencies with cycle detection and depth bounds."""
    def __init__(self, registry: OperationalCapabilityRegistry, max_depth: int = 8):
        self.registry = registry
        self.max_depth = max_depth

    def resolve(
        self,
        capability_id: str,
        version_constraint: str = "*",
        visited: list[str] | None = None,
        depth: int = 0
    ) -> OperationalCapability:
        if depth > self.max_depth:
            raise CapabilityDepthExceededError(f"Maximum capability depth of {self.max_depth} exceeded resolving '{capability_id}'")
        visited = list(visited or [])
        if capability_id in visited:
            cycle = " -> ".join(visited + [capability_id])
            raise CapabilityCycleError(f"Cycle detected in capability dependencies: {cycle}")
        cap = self.registry.get(capability_id, version_constraint)
        if not cap:
            raise CapabilityNotFoundError(f"Capability '{capability_id}' matching '{version_constraint}' not found")
        if cap.drift_state == "quarantined":
            raise CapabilityDriftError(f"Capability '{capability_id}' is quarantined due to operational drift")

        current_visited = visited + [capability_id]
        for dep in cap.dependencies:
            self.resolve(dep.capability_id, dep.version_constraint, visited=current_visited, depth=depth + 1)
        return cap

    def linearize(self, capability_id: str, version_constraint: str = "*") -> list[OperationalCapability]:
        """Produce an ordered, cycle-free list of dependencies (topological sort) followed by the root."""
        resolved: list[OperationalCapability] = []
        visited_ids: set[str] = set()

        def _dfs(cap_id: str, v_constraint: str, path: list[str], depth: int):
            if depth > self.max_depth:
                raise CapabilityDepthExceededError(f"Maximum depth of {self.max_depth} exceeded resolving '{cap_id}'")
            if cap_id in path:
                cycle = " -> ".join(path + [cap_id])
                raise CapabilityCycleError(f"Cycle detected in capability dependencies: {cycle}")
            cap = self.registry.get(cap_id, v_constraint)
            if not cap:
                raise CapabilityNotFoundError(f"Capability '{cap_id}' not found")
            for dep in cap.dependencies:
                _dfs(dep.capability_id, dep.version_constraint, path + [cap_id], depth + 1)
            if cap.id not in visited_ids:
                visited_ids.add(cap.id)
                resolved.append(cap)

        _dfs(capability_id, version_constraint, [], 0)
        return resolved


def learn_operational_capability(
    registry: OperationalCapabilityRegistry,
    *,
    name: str,
    route: str,
    steps: list[dict[str, Any]],
    scope: dict[str, Any] | None = None,
    preconditions: list[str] | None = None,
    postconditions: list[str] | None = None,
    semantic_fingerprint: str = "",
    compatibility_fingerprint: str = "",
    provenance: dict[str, Any] | None = None,
    dependencies: list[CapabilityDependency] | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    version: str = "1.0.0",
    auto_id: str | None = None,
) -> OperationalCapability:
    """Learn a new OperationalCapability candidate from verified execution experience."""
    import uuid
    cap_id = auto_id or f"cap_{uuid.uuid4().hex[:12]}"
    capability = OperationalCapability(
        id=cap_id,
        name=name,
        version=version,
        route=route,
        input_schema=input_schema or {},
        output_schema=output_schema or {},
        scope=scope or {},
        preconditions=preconditions or [],
        postconditions=postconditions or [],
        dependencies=dependencies or [],
        implementation={"steps": steps},
        lifecycle=CapabilityLifecycle.DISCOVERED,
        provenance=provenance or {},
        semantic_fingerprint=semantic_fingerprint,
        compatibility_fingerprint=compatibility_fingerprint,
    )
    return registry.register(capability)

