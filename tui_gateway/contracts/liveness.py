"""Transport liveness + build capability probes (``methods_voice.py`` hosts them)."""

from __future__ import annotations

from .base import Params, Result, WireEnum
from .registry import method


class PingParams(Params):
    pass


class PingResult(Result):
    pong: bool


method("ping", params=PingParams, result=PingResult,
       doc="Cheapest liveness probe; answered on the WS reader thread even while every agent is mid-turn.")


class GatewayCapabilitiesResult(Result):
    per_session_exclusive_submit: bool


method("gateway.capabilities", params=PingParams, result=GatewayCapabilitiesResult,
       doc="What THIS build enforces (a client withholds a feature unless advertised).")


class DeviceCapability(WireEnum):
    """What a handheld client can offer the agent on request — the closed vocabulary a client
    advertises in ``client.capabilities`` and the backend may then ask for over ``device.request``.
    Deliberately device-shaped, not tool-shaped: a desktop client advertises none of these."""

    camera = "camera"
    microphone = "microphone"
    location = "location"
    screen_capture = "screen_capture"
    notifications = "notifications"
    photo_library = "photo_library"
    files = "files"


class ClientCapabilitiesParams(Params):
    #: The client answers server→client requests (clarify, approval, sudo, …) — with a result or a -32601
    #: error for methods it has no handler for. A WebSocket client that never says so is treated as a
    #: build older than server→client requests and every such request fails fast for it.
    server_requests: bool = False
    #: Device capabilities this client can serve (a phone: camera, location, …). Absent/empty means
    #: "no device behind me" — the desktop and the TUI advertise nothing, and ``device.request``
    #: fails fast for them instead of waiting for a capability that will never arrive.
    device_capabilities: list[DeviceCapability] | None = None


class ClientCapabilitiesResult(Result):
    #: Server→client request methods this backend may send.
    server_requests: list[str]
    #: This backend counts a ``4404`` "no window here shows this session" error from a window-owned
    #: bridge (preview/terminal/window read, preview act, tour) as that client declining, not as the
    #: answer: the request settles only once every attached client declined (server_requests.py).
    #: An older backend settles on the first error, so a client sends the decline only when this is true.
    declines_not_shown: bool = False
    #: Device-shaped server→client requests this backend may send — a client advertises
    #: ``device_capabilities`` and reads this to know which of them to expect.
    device_requests: list[str]


method("client.capabilities", params=ClientCapabilitiesParams, result=ClientCapabilitiesResult,
       doc="What the calling client handles, sent once per connection (after gateway.ready); returns the "
           "server→client request methods this backend may send.")


class DeviceCapabilitiesParams(Params):
    #: The session whose attached client is being described — the agent asks about the chat it is in.
    session_id: str


class DeviceCapabilitiesResult(Result):
    session_id: str
    #: Sorted union of the device capabilities the session's live clients advertised; empty when none
    #: did (a desktop window or the TUI), which is the agent's cue to not offer them at all.
    capabilities: list[DeviceCapability]


method("device.capabilities", params=DeviceCapabilitiesParams, result=DeviceCapabilitiesResult,
       doc="The device capabilities a session's attached client advertised, so an agent knows what it "
           "may ask for over device.request.")
