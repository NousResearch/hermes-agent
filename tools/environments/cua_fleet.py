"""Placeholder provider for a remotely managed CUA desktop fleet."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Sequence

from tools.environments.compute_provider import ComputeLease, EnvironmentCapabilities


@dataclass(frozen=True)
class CuaFleetConfig:
    endpoint: str = ""
    image: str = "trycua/cua:latest"
    pool: str = "default"
    token: str = ""


class CuaFleetDesktopProvider:
    """PoC lifecycle surface for fleet pool/claim/release integration."""

    name = "cua_fleet"

    def __init__(self, config: CuaFleetConfig | None = None):
        self.config = config or CuaFleetConfig()

    def acquire(self, task_id: str, *, image: str | None = None,
                capabilities: Sequence[str] | None = None) -> ComputeLease:
        if not self.config.endpoint:
            raise RuntimeError("cua_fleet.endpoint must be configured before acquiring a desktop")
        enabled = EnvironmentCapabilities(computer_use=True)
        requested = frozenset(capabilities or enabled.to_capabilities())
        if not requested <= enabled.to_capabilities():
            raise ValueError("Cua fleet desktop does not support all requested capabilities")
        return ComputeLease(
            task_id=task_id, lease_id=uuid.uuid4().hex, provider=self.name,
            image=image or self.config.image, capabilities=enabled,
            endpoint=self.config.endpoint, metadata={"pool": self.config.pool},
        )

    def create_environment(self, lease: ComputeLease):
        raise NotImplementedError("CuaFleetDesktopProvider environment attachment is not implemented in this PoC")

    def release(self, lease: ComputeLease) -> None:
        return None
