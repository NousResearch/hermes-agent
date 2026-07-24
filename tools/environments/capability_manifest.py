"""Capability manifests declared by sandbox images."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from tools.environments.compute_provider import EnvironmentCapabilities


@dataclass(frozen=True)
class CapabilitySpec:
    """A single image capability and optional service endpoint."""

    name: str
    enabled: bool = True
    service: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, value: Any) -> "CapabilitySpec":
        if isinstance(value, bool):
            return cls(name=name, enabled=value)
        if value is None:
            return cls(name=name)
        if not isinstance(value, Mapping):
            raise ValueError(f"capability {name!r} must be a bool or object")
        return cls(
            name=name,
            enabled=bool(value.get("enabled", True)),
            service=value.get("service"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class CapabilityManifest:
    """Describes the services installed in a compute image."""

    image: str = ""
    capabilities: Mapping[str, CapabilitySpec] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilityManifest":
        raw = value.get("capabilities", {})
        if isinstance(raw, list):
            raw = {name: True for name in raw}
        if not isinstance(raw, Mapping):
            raise ValueError("manifest capabilities must be an object or list")
        capabilities = {
            str(name): CapabilitySpec.from_dict(str(name), spec)
            for name, spec in raw.items()
        }
        return cls(
            image=str(value.get("image", "")),
            capabilities=capabilities,
            metadata=dict(value.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, value: str) -> "CapabilityManifest":
        parsed = json.loads(value)
        if not isinstance(parsed, Mapping):
            raise ValueError("manifest JSON must contain an object")
        return cls.from_dict(parsed)

    def to_environment_capabilities(self) -> EnvironmentCapabilities:
        enabled = {name for name, spec in self.capabilities.items() if spec.enabled}
        return EnvironmentCapabilities(
            terminal="terminal" in enabled,
            files="files" in enabled,
            computer_use="computer_use" in enabled,
            process="process" in enabled or "terminal" in enabled,
            extras=frozenset(enabled - {"terminal", "files", "computer_use", "process"}),
        )

    def enabled_capabilities(self) -> frozenset[str]:
        return self.to_environment_capabilities().to_capabilities()


def default_manifest(image: str = "") -> CapabilityManifest:
    return CapabilityManifest.from_dict({
        "image": image,
        "capabilities": {"terminal": True, "files": True, "process": True},
    })


def desktop_manifest(image: str = "") -> CapabilityManifest:
    return CapabilityManifest.from_dict({
        "image": image,
        "capabilities": {
            "terminal": True,
            "files": True,
            "process": True,
            "computer_use": {"service": "cua-driver"},
        },
    })
