"""Repo-local validated tool I/O helper.

Pydantic v2 ``BaseModel`` is already a core dependency, so this module uses it
directly. It deliberately does *not* add any new runtime dependency.

Design goals
------------
* One import root for relay-style envelopes::

    from tools.validation_helper import Success, Failure, relay_success, relay_failure

* Build validated structures without adding tool-schema dependencies.
* A drop-in outcome helper so callers can wrap a result dict and let Pydantic
  v2 normalize/validate it before it leaves the process.

Usage
-----
.. code-block:: python

    from tools.validation_helper import Success, Failure, relay_success, relay_failure, validate_model_or_failure
    from pydantic import BaseModel

    class QueryResult(BaseModel):
        ok: bool
        count: int = 0

    def handle(args):
        raw = args.get("query") or ""
        try:
            result = QueryResult(ok=True, count=len(raw.split())).model_dump(mode="json")
            return relay_success(result)
        except Exception as exc:
            return relay_failure(str(exc), code="parse_error")

    # or, when a payload exists and should be normalized:
    def validate_and_result(model_type: type, payload: dict, namespace: str = "payload"):
        return validate_model_or_failure(model_type, payload, namespace)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class Success(BaseModel, frozen=True):
    """Dispatched when a handler successfully handled a tool call."""

    kind: str = "success"
    data: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None


class Failure(BaseModel, frozen=True):
    """Dispatched when a handler failed to handle a tool call.

    Failures stay outside the happy-path return envelope so callers never
    mis-handle a structured failure as payload data.
    """

    kind: str = "failure"
    error: str
    code: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


def relay_success(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a zodern:relay-style success envelope dict.

    Keeps the API serialization-agnostic so caller/runtime can JSON-encode
    it however the gateway expects.
    """
    return Success(data=data).model_dump(mode="json")


def relay_failure(
    error: str,
    code: Optional[str] = None,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a zodern:relay-style failure envelope dict."""
    return Failure(error=error, code=code, meta=meta).model_dump(mode="json")


def validate_model_or_failure(
    model_type: type,
    payload: Dict[str, Any],
    field_namespace: str = "payload",
) -> Dict[str, Any]:
    """Validate *payload* against *model_type*.

    Returns a success envelope with the normalized payload on success, or a
    failure envelope with a code=``validation_error`` on failure.
    """
    try:
        model_type.model_validate(payload)
    except Exception as exc:
        return relay_failure(
            f"invalid {field_namespace}: {exc}", code="validation_error"
        )
    return Success(data=payload).model_dump(mode="json")

__all__ = [
    "F",
    "Success",
    "Failure",
    "relay_success",
    "relay_failure",
    "validate_model_or_failure",
]
