"""Migration facade for selecting the legacy or experimental environment runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import logging
import os
from threading import RLock
from typing import TYPE_CHECKING, Any

from tools.environments.definitions import BackendFactoryRequest
from tools.environments.manager import EnvironmentManager

if TYPE_CHECKING:
    from tools.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

LegacyEnvironmentFactory = Callable[[BackendFactoryRequest], Any]


class EnvironmentFacade(ABC):
    """Process-scoped entry point for environment lifecycle operations."""

    @abstractmethod
    def create_environment(self, request: BackendFactoryRequest) -> "BaseEnvironment":
        """Create an environment through the selected runtime."""


class LegacyEnvironmentFacade(EnvironmentFacade):
    """Adapter that preserves the existing terminal backend implementation."""

    def __init__(self, legacy_factory: LegacyEnvironmentFactory) -> None:
        self._legacy_factory = legacy_factory

    def create_environment(self, request: BackendFactoryRequest) -> "BaseEnvironment":
        return self._legacy_factory(request)


class ExperimentalEnvironmentFacade(EnvironmentFacade):
    """Adapter for the registry-backed EnvironmentManager runtime."""

    def __init__(self, manager: EnvironmentManager) -> None:
        self.manager = manager

    def create_environment(self, request: BackendFactoryRequest) -> "BaseEnvironment":
        return self.manager.create_environment(request)


_facade: EnvironmentFacade | None = None
_facade_lock = RLock()


def get_environment_facade(
    legacy_factory: LegacyEnvironmentFactory,
    manager_factory: Callable[[], EnvironmentManager] = EnvironmentManager,
) -> EnvironmentFacade:
    """Return the process-scoped environment facade."""
    global _facade
    with _facade_lock:
        if _facade is None:
            exp_backend = os.getenv("EXP_BACKEND")
            if exp_backend == "1":
                _facade = ExperimentalEnvironmentFacade(manager_factory())
            else:
                if exp_backend not in {None, "0"}:
                    logger.warning(
                        "Invalid EXP_BACKEND=%r; using the legacy backend runtime",
                        exp_backend,
                    )
                _facade = LegacyEnvironmentFacade(legacy_factory)
        return _facade


def reset_environment_facade() -> None:
    """Reset the process-scoped facade for tests and controlled reloads."""
    global _facade
    with _facade_lock:
        _facade = None
