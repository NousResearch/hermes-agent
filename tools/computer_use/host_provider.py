"""The built-in computer-use providers: the gateway host, and a test stub.

``local`` is what computer_use has always done — spawn cua-driver on whatever
machine runs the gateway and drive its display. Registering it as a provider
rather than leaving it as the fallback branch means the dispatcher has exactly
one path to a backend, so a plugin provider is exercised by the same code the
default is, instead of by a branch nobody runs.

``noop`` replaces the ``HERMES_COMPUTER_USE_BACKEND=noop`` escape hatch that
tests and CI used to reach for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.computer_use_provider import ComputerUseProvider
from agent.computer_use_registry import HOST_PROVIDER_NAME, register_provider

if TYPE_CHECKING:
    from tools.computer_use.backend import ComputerUseBackend


class HostCuaProvider(ComputerUseProvider):
    """cua-driver on the machine running the gateway."""

    @property
    def name(self) -> str:
        return HOST_PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "This machine (cua-driver)"

    def is_available(self) -> bool:
        from tools.computer_use.cua_backend_driver import cua_driver_binary_available

        return cua_driver_binary_available()

    def create_backend(self, session_id: str, permission_mode: str) -> "ComputerUseBackend":
        from tools.computer_use.cua_backend import CuaDriverBackend

        return CuaDriverBackend(permission_mode=permission_mode)


class NoopCuaProvider(ComputerUseProvider):
    """Records calls and returns trivial results. For tests and CI."""

    @property
    def name(self) -> str:
        return "noop"

    def is_available(self) -> bool:
        return True

    def create_backend(self, session_id: str, permission_mode: str) -> "ComputerUseBackend":
        from tools.computer_use.tool import _NoopBackend

        return _NoopBackend()


register_provider(HostCuaProvider())
register_provider(NoopCuaProvider())
