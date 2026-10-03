"""The two bridge providers: reach a keyboard on a machine that is not the host.

Both answer the same question the host provider does — whose screen does
``computer_use`` act on — for the case where the agent and the screen are on
different machines. They differ in who dials whom:

``http-bridge``
    The operator's own plumbing. ``hermes computer-use bridge`` listens on the
    desktop, an SSH/VPN tunnel carries it, and the backend is pointed at the
    resulting loopback URL in config.yaml. Selected explicitly, like any other
    provider, and it stays selected for every session on that backend.

``desktop-bridge``
    The productised path. Hermes Desktop opens a WebSocket back to the backend
    and proxies calls to a sidecar next to the app, so it works from behind NAT
    with nothing to tunnel. Unlike the others it is not a property of the
    backend at all: one gateway serves a Desktop client with a bridge, a phone
    on Telegram, and a cron job in the same process, and which of those may
    drive the user's Mac is a question about the *session*. So it is normally
    left out of config entirely and ``tools.computer_use.tool`` resolves it per
    call from the connection's own verified scope (the surface-capability rule
    in AGENTS.md).

    Naming it in config is still meaningful, and is the right setting for a
    shared gateway: it pins the answer to "a Desktop I authenticated, or
    nothing", so a cron tick or a Telegram turn fails closed instead of falling
    back to driving the server's own screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from agent.computer_use_provider import ComputerUseProvider
from agent.computer_use_registry import register_provider

if TYPE_CHECKING:
    from tools.computer_use.backend import ComputerUseBackend

#: Config name for the tunnelled HTTP bridge.
HTTP_BRIDGE_PROVIDER_NAME = "http-bridge"

#: Not a config name — resolved from the live session. See the module docstring.
DESKTOP_BRIDGE_PROVIDER_NAME = "desktop-bridge"


class HttpBridgeProvider(ComputerUseProvider):
    """cua-driver on a desktop reachable at a configured bridge URL."""

    @property
    def name(self) -> str:
        return HTTP_BRIDGE_PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "Tunnelled bridge (hermes computer-use bridge)"

    def is_available(self) -> bool:
        from tools.computer_use.bridge import bridge_backend_configured

        return bridge_backend_configured()

    def create_backend(self, session_id: str, permission_mode: str) -> "ComputerUseBackend":
        from tools.computer_use.bridge import HttpComputerUseBridgeBackend

        return HttpComputerUseBridgeBackend()

    def get_status(self) -> Dict[str, Any]:
        from tools.computer_use.bridge import bridge_computer_use_status

        return bridge_computer_use_status()

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "tunnel",
            "tag": "remote desktop",
            "env_vars": ["HERMES_COMPUTER_USE_BRIDGE_TOKEN"],
        }


class DesktopBridgeProvider(ComputerUseProvider):
    """The Hermes Desktop client on the other end of this session's socket."""

    @property
    def name(self) -> str:
        return DESKTOP_BRIDGE_PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "Hermes Desktop (this machine)"

    def is_available(self) -> bool:
        """Whether *this caller's* Desktop bridge is live.

        Scope-aware and fail-closed: with no verified principal bound to the
        execution — a cron tick, a Telegram turn, another user's session on the
        same gateway — there is no socket to match and this is False. That is
        the check that keeps one person's Desktop off another person's calls,
        so it must stay the same question ``create_backend`` will ask.
        """
        from tools.computer_use.desktop_bridge import desktop_bridge_connected

        return desktop_bridge_connected()

    def routing_identity(self) -> str:
        """Keyed by the caller's scope, so a backend never outlives its owner.

        One started backend is cached per Hermes session, and a session id can
        outlast the identity behind it — the shared empty session id that
        integrations reuse, a Desktop that reconnects as a different user.
        Naming the scope here makes that a cache miss and a fresh handshake
        instead of a second principal inheriting the first one's socket.
        """
        from tools.computer_use.desktop_bridge import current_desktop_bridge_scope

        try:
            scope = current_desktop_bridge_scope()
        except RuntimeError:
            return self.name

        return f"{self.name}:{scope.provider}:{scope.principal}:{scope.profile}"

    def unavailable_reason(self) -> str:
        return (
            "Hermes Desktop's Computer Use bridge is not connected for the "
            "authenticated principal of this session"
        )

    def create_backend(self, session_id: str, permission_mode: str) -> "ComputerUseBackend":
        from tools.computer_use.desktop_bridge import DesktopComputerUseBridgeBackend

        # Refuse here rather than hand back a backend with nothing on the far
        # end. This provider is only reached by name when config pinned it, so
        # the caller asked for a Desktop specifically: say that none answered
        # instead of letting a request time out against a dead socket.
        if not self.is_available():
            raise RuntimeError(self.unavailable_reason())

        return DesktopComputerUseBridgeBackend()

    def get_status(self) -> Dict[str, Any]:
        from tools.computer_use.desktop_bridge import desktop_bridge_computer_use_status

        return desktop_bridge_computer_use_status()

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "desktop",
            "tag": "auto",
            "env_vars": [],
        }


register_provider(HttpBridgeProvider())
register_provider(DesktopBridgeProvider())
