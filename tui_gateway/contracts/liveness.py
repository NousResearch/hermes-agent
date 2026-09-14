"""Transport liveness + build capability probes (``methods_voice.py`` hosts them)."""

from __future__ import annotations

from .base import Params, Result
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
