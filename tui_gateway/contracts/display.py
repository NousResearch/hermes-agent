"""Bot Screen (``methods_display.py`` / ``methods_display_watch.py``): a profile's headless Xfce
desktop, its takeover lease, the noVNC ticket and the on-host package install.

Every method rides ``profile`` (the desktop routes through ``requestGatewayForProfile``) and answers
for THAT profile's screen. The lease is the whole control model: ``holder`` is ``agent`` or
``human``; ``epoch`` moves on every real transition, so a client (or the agent's fence) can tell an
in-flight result was produced under a lease that has since changed.
"""

from __future__ import annotations

from .base import Params, Payload, Result, WireEnum
from .common import ProfileParams
from .registry import event, method, server_request
from .server_requests import ServerRequestParams, ValueResult

# ── shapes ────────────────────────────────────────────────────────────────────────────────────


class LeaseHolder(WireEnum):
    agent = "agent"
    human = "human"


class DisplayLease(Result):
    """``tools/bot_desktop/lease.py::Lease`` as clients may see it: the holder's viewer id is a
    capability and never leaves the gateway; ``viewer_hash`` lets the holder recognise itself."""

    holder: LeaseHolder
    viewer_id: None = None
    viewer_hash: str | None = None
    since: float
    epoch: int
    reason: str = ""


class DisplayStatus(Result):
    """``tools/bot_desktop/runtime.py::DesktopStatus`` plus the lease and the profile it speaks for."""

    profile: str
    supported: bool
    installed: bool
    missing: list[str]
    running: bool
    pid: int | None = None
    display: str | None = None
    socket: str | None = None
    geometry: str
    install_command: str | None = None
    browser: str | None = None
    blocker: str | None = None  # why display.start would refuse now (host memory); the pane shows it instead of Start
    memory_available_mb: int | None = None
    memory_limit_mb: int | None = None
    lease: DisplayLease
    profile_key: str


# ── methods ───────────────────────────────────────────────────────────────────────────────────


method("display.status", params=ProfileParams, result=DisplayStatus,
       doc="Runtime + lease snapshot for this profile's screen.")


class DisplayThumbnailResult(Result):
    """``data_url`` is null while the screen is stopped or while a human holds the lease
    (``suppressed``): the frame may show what they are typing."""

    data_url: str | None = None
    suppressed: str | None = None


method("display.thumbnail", params=ProfileParams, result=DisplayThumbnailResult,
       doc="One JPEG grab of the bot's screen; read-only, never changes the lease.")

method("display.start", params=ProfileParams, result=DisplayStatus,
       doc="Start this profile's Xvnc + Xfce (idempotent); blocks until the display is published.")


class DisplayStopParams(ProfileParams):
    force: bool | None = None  # stop even while a human holds control


class DisplayStopResult(DisplayStatus):
    stopped: bool


method("display.stop", params=DisplayStopParams, result=DisplayStopResult,
       doc="Stop the screen. Refused (5300, code viewer_mismatch) while a human holds unless force.")


class DisplayObserveParams(ProfileParams):
    viewer_id: str | None = None  # an id THIS connection minted earlier keeps its identity across a reconnect


class DisplayObserveResult(DisplayStatus):
    ticket: str
    path: str
    viewer_id: str


method("display.observe", params=DisplayObserveParams, result=DisplayObserveResult,
       doc="Mint a single-use ticket for /api/display/ws and the server-minted viewer id for this connection.")


class DisplayInstallResult(Result):
    started: bool
    command: str | None = None
    profile_key: str


method("display.install", params=ProfileParams, result=DisplayInstallResult,
       doc="Run the distro package install on the gateway host; progress streams as display.install.log/.done.")


class DisplayLeaseAcquireParams(ProfileParams):
    viewer_id: str
    reason: str | None = None


class DisplayLeaseResult(Result):
    lease: DisplayLease


method("display.lease.acquire", params=DisplayLeaseAcquireParams, result=DisplayLeaseResult,
       doc="Take over: the human named by a viewer id this connection minted controls the screen.")


class DisplayLeaseReleaseParams(ProfileParams):
    viewer_id: str | None = None
    force: bool | None = None


method("display.lease.release", params=DisplayLeaseReleaseParams, result=DisplayLeaseResult,
       doc="Hand back. Without a viewer id the release is refused while a human holds unless force.")


# ── server → client ───────────────────────────────────────────────────────────────────────────


class DisplayInstallSudoParams(ServerRequestParams):
    profile_key: str


server_request("display.install.sudo", params=DisplayInstallSudoParams, result=ValueResult,
               doc="Masked sudo password for the Bot Screen package install; app-level (empty session).")


# ── events ────────────────────────────────────────────────────────────────────────────────────


class DisplayStatusPayload(DisplayStatus, Payload):
    pass


event("display.status", DisplayStatusPayload,
      doc="This profile's screen started or stopped (also for transitions made outside hermes serve).")


class DisplayLeasePayload(Payload):
    profile_key: str
    lease: DisplayLease


event("display.lease", DisplayLeasePayload, doc="The takeover lease changed hands; every client repaints.")


class DisplayInstallLogPayload(Payload):
    profile_key: str
    line: str


event("display.install.log", DisplayInstallLogPayload, doc="One line of package-manager output.")


class DisplayInstallDonePayload(Payload):
    profile_key: str
    code: int
    status: DisplayStatus


event("display.install.done", DisplayInstallDonePayload,
      doc="The install ended (0 ok, -1 cancelled, -2 no sudo: the command to run by hand was streamed).")
