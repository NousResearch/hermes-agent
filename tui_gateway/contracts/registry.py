"""The contract catalog: every method, server→client request and event the gateway speaks.

Three tables, filled by the ``contracts.*`` topic modules at import time and read by
``tui_gateway/server.py`` (runtime dispatch) and ``scripts/gen_gateway_contracts.py``
(TypeScript + OpenRPC rendering). A handler registered with ``@method`` for a name that has
no contract here fails at import: the wire has no undeclared surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import UnionType
from typing import TypeAlias

from .base import MethodParams, Params, Payload, Result


ResultType: TypeAlias = type[Result] | UnionType


@dataclass(frozen=True)
class MethodContract:
    name: str
    params: type[Params]
    result: ResultType
    doc: str = ""


@dataclass(frozen=True)
class ServerRequestContract:
    """A question the backend asks the client (``server_requests.send``)."""

    name: str
    params: type[Params]
    result: ResultType
    doc: str = ""


@dataclass(frozen=True)
class EventContract:
    name: str
    payload: type[Payload] | None  # None: the frame carries no payload
    doc: str = ""


METHODS: dict[str, MethodContract] = {}
SERVER_REQUESTS: dict[str, ServerRequestContract] = {}
EVENTS: dict[str, EventContract] = {}


def _declare(table: dict, entry) -> None:
    if entry.name in table:
        raise RuntimeError(f"contract declared twice: {entry.name}")
    table[entry.name] = entry


def method(name: str, *, params: type[MethodParams], result: ResultType, doc: str = "") -> MethodContract:
    if not issubclass(params, MethodParams):
        raise TypeError(f"{name}: method params must subclass MethodParams (carries the transport ``profile`` key)")
    entry = MethodContract(name, params, result, doc)
    _declare(METHODS, entry)
    return entry


def server_request(name: str, *, params: type[Params], result: ResultType,
                   doc: str = "") -> ServerRequestContract:
    if isinstance(params, type) and issubclass(params, MethodParams):  # clarify passes a union alias
        raise TypeError(f"{name}: server-request params must not carry the client-only ``profile`` key")
    entry = ServerRequestContract(name, params, result, doc)
    _declare(SERVER_REQUESTS, entry)
    return entry


def event(name: str, payload: type[Payload] | None = None, *, doc: str = "") -> EventContract:
    entry = EventContract(name, payload, doc)
    _declare(EVENTS, entry)
    return entry


def assert_complete(methods: dict[str, object], server_requests: set[str]) -> None:
    """Every registered method and sent server request has exactly one contract."""
    problems = []
    if missing := sorted(set(methods) - set(METHODS)):
        problems.append(f"methods without a contract: {missing}")
    if orphan := sorted(set(METHODS) - set(methods)):
        problems.append(f"method contracts with no handler: {orphan}")
    if missing := sorted(server_requests - set(SERVER_REQUESTS)):
        problems.append(f"server requests without a contract: {missing}")
    if orphan := sorted(set(SERVER_REQUESTS) - server_requests):
        problems.append(f"server request contracts nothing sends: {orphan}")
    if problems:
        raise RuntimeError("tui_gateway/contracts is incomplete:\n  " + "\n  ".join(problems))
