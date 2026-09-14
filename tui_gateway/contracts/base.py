"""Wire contract primitives for the TUI/desktop gateway.

Python is the single source of truth for the JSON-RPC wire: every client→server method
(params + result), every server→client request (params + result) and every notification
payload is a Pydantic model declared in this package. ``scripts/gen_gateway_contracts.py``
renders them into ``apps/shared/src/gateway-contract.generated.ts`` and
``apps/shared/src/gateway-contract.openrpc.json``; ``tests/contracts/test_generated.py``
regenerates in memory and diffs the committed files, so a model edited without regenerating
fails CI on the Python side, and TS that reads a phantom field fails ``tsc``.

Modelling rules (they keep the generated TS clean and the wire stable):

- ``snake_case`` field names, exactly as they travel.
- Closed sets are ``StrEnum`` (rendered as literal unions); discriminators are ``Literal``.
- ``X | None = None`` renders ``x?: X | null``; a plain default renders ``x?: X``.
- Params models are ``extra="forbid"``: an unknown key is a client bug and answers ``4000``
  instead of being silently ignored. Result and payload models are ``extra="allow"`` only
  while a field is genuinely open (``dict[str, Any]`` is banned in a contract — declare the
  shape or use ``JsonValue``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

JsonValue = Any  # a JSON scalar/array/object the contract deliberately leaves open (renders ``unknown``)


class Params(BaseModel):
    """Client→server method params / server→client request params. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Result(BaseModel):
    """Method / server-request result. Serialised with ``exclude_none=False`` so an explicit
    ``null`` stays a ``null`` on the wire (clients distinguish absent from null)."""

    model_config = ConfigDict(extra="forbid")


class Payload(BaseModel):
    """Notification payload (``event`` frame ``params.payload``)."""

    model_config = ConfigDict(extra="forbid")


class WireEnum(StrEnum):
    """A closed string set on the wire; renders as a TS literal union."""


__all__ = ["JsonValue", "Params", "Payload", "Result", "WireEnum"]
