"""Transport liveness + build capability probes (``methods_voice.py`` hosts them)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .base import Params, Result
from .common import CanonicalClientIdentity, EmptyResult
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


class ClientCapabilitiesParams(Params):
    #: The client answers server→client requests (clarify, approval, sudo, …) — with a result or a -32601
    #: error for methods it has no handler for. A WebSocket client that never says so is treated as a
    #: build older than server→client requests and every such request fails fast for it.
    server_requests: bool = False
    #: The client is the Desktop app connected to a backend on its own machine (its connection mode is
    #: local). Counted only on a connection that redeemed a native ticket minted over this machine's
    #: private control socket; it is what makes the welcome trial and first-run onboarding available
    #: in a gateway that also serves the CLI, the TUI and bots.
    desktop_local: bool = False
    #: Which client this connection is (``{name, version?, kind?}``), kept for the connection's life. A
    #: self-declared label for attribution only: it grants nothing. Sent as its own call, so a backend
    #: older than the field refuses only that call. A malformed one keeps the identity the connection
    #: already had: an unknown key in it answers ``4000`` (the generic unknown-key check), any other
    #: problem ``4001``.
    client: CanonicalClientIdentity | None = None


class ClientCapabilitiesResult(Result):
    #: Server→client request methods this backend may send.
    server_requests: list[str]
    #: This backend counts a ``4404`` "no window here shows this session" error from a window-owned
    #: bridge (preview/terminal/window read, preview act, tour) as that client declining, not as the
    #: answer: the request settles only once every attached client declined (server_requests.py).
    #: An older backend settles on the first error, so a client sends the decline only when this is true.
    declines_not_shown: bool = False
    #: A canonical owner on this backend admits ``clarify_batch`` on ``prompt.submit``: the turn's
    #: multi-question clarify arrives as ONE ``clarify`` request answered with ``clarify.lock``, not
    #: one card per question. An older owner refuses the unknown key, so a client declares it only when
    #: this is true. (The legacy sidecar always asks the batch at once and is never sent the key.)
    canonical_clarify_batch: bool = False
    #: This connection keeps ``client.activity`` reports (false on one that keeps no presence, such as
    #: stdio). An older backend refuses the method, so a client reports only on a connection whose
    #: answer said this is true.
    client_activity: bool = False


method("client.capabilities", params=ClientCapabilitiesParams, result=ClientCapabilitiesResult,
       doc="What the calling client handles, sent once per connection (after gateway.ready); returns the "
           "server→client request methods this backend may send.")


class ClientActivityParams(Params):
    #: Every session this client has in view right now: the connection's whole statement. The owner
    #: drops this connection's records of any session not listed, input it saw itself included, so
    #: an empty list withdraws them all. At most 16 ids, each at most 256 characters.
    sessions: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(max_length=16)
    #: How long ago the user's last input was, by the client's clock source.
    idle_seconds: float = Field(ge=0)
    #: ``system``: the OS idle clock (input in any app counts). ``page``: input inside the Hermes window only.
    basis: Literal["system", "page"]
    #: This client can get the user's attention for these sessions itself (on screen in a focused
    #: window, or its own turn-done notification is on).
    can_alert: bool = True
    #: The user is gone now (screen locked, machine suspending): clears this connection's reports.
    away: bool = False


method("client.activity", params=ClientActivityParams, result=EmptyResult,
       doc="Whether the user is at this client, sent only when ``client.capabilities`` answered "
           "``client_activity: true``; each report replaces the connection's last one, is kept in memory "
           "per connection and session, and grants nothing.")
