"""Wire contract primitives for the TUI/desktop gateway.

Python is the single source of truth for the JSON-RPC wire: every client→server method
(params + result), every server→client request (params + result) and every notification
payload is a Pydantic model declared in this package. ``scripts/gen_gateway_contracts.py``
renders them into ``apps/shared/src/gateway-contract.generated.ts`` and
``apps/shared/src/gateway-contract.openrpc.json``; ``tests/tui_gateway/contracts/test_generated.py``
regenerates in memory and diffs the committed files, so a model edited without regenerating
fails CI on the Python side, and TS that reads a phantom field fails ``tsc``.

Modelling rules (they keep the generated TS clean and the wire stable):

- ``snake_case`` field names, exactly as they travel.
- Closed sets are ``StrEnum`` (rendered as literal unions); discriminators are ``Literal``.
- Inbound models (method params and server-request results) render defaulted fields as optional.
  Outbound models (method results, server-request params and event payloads) render every field
  required; use ``X | None`` when the wire may carry ``null``.
- Every model is ``extra="forbid"``. An unknown key in params is a client bug and answers
  ``4000`` instead of being silently ignored; a genuinely open field is ``JsonValue``, never
  ``dict[str, Any]`` or ``extra="allow"``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, JsonValue


class Params(BaseModel):
    """Inbound client→server method params / outbound server-request params; unknown keys reject."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MethodParams(Params):
    """Top-level params of a client→server method (``registry.method`` requires this base)."""

    # The desktop routes any method to a named profile by injecting ``profile`` (``requestGatewayForProfile``,
    # ``session-request-router.routeParams``) and ``server._profile_scoped`` reads it via getattr: transport, not
    # surface. Nested inputs and server-request params stay on bare ``Params`` so the key never travels outbound.
    profile: str | None = None


class Result(BaseModel):
    """Outbound method / inbound server-request result; ``None`` serializes as wire ``null``."""

    model_config = ConfigDict(extra="forbid")


class Payload(BaseModel):
    """Outbound notification payload (``event`` frame ``params.payload``)."""

    model_config = ConfigDict(extra="forbid")


class WireEnum(StrEnum):
    """A closed string set on the wire; renders as a TS literal union."""


__all__ = ["JsonValue", "MethodParams", "Params", "Payload", "Result", "WireEnum"]
