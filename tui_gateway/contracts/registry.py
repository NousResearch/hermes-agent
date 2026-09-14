"""The contract catalog: every method, server→client request and event the gateway speaks.

Three tables, filled by the ``contracts.*`` topic modules at import time and read by
``tui_gateway/server.py`` (runtime validation) and ``scripts/gen_gateway_contracts.py``
(TypeScript + OpenRPC rendering). A handler registered with ``@method`` for a name that has
no contract here fails at import: the wire has no undeclared surface.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from pydantic import ValidationError

from .base import Params, Payload, Result

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MethodContract:
    name: str
    params: type[Params]
    result: type[Result]
    doc: str = ""


@dataclass(frozen=True)
class ServerRequestContract:
    """A question the backend asks the client (``server_requests.send``)."""

    name: str
    params: type[Params]
    result: type[Result]
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


def method(name: str, *, params: type[Params], result: type[Result], doc: str = "") -> MethodContract:
    entry = MethodContract(name, params, result, doc)
    _declare(METHODS, entry)
    return entry


def server_request(name: str, *, params: type[Params], result: type[Result], doc: str = "") -> ServerRequestContract:
    entry = ServerRequestContract(name, params, result, doc)
    _declare(SERVER_REQUESTS, entry)
    return entry


def event(name: str, payload: type[Payload] | None = None, *, doc: str = "") -> EventContract:
    entry = EventContract(name, payload, doc)
    _declare(EVENTS, entry)
    return entry


# ── runtime checks ────────────────────────────────────────────────────────────────────────────
#
# Params are validated on every call: an unknown or mistyped key is the CLIENT's bug and answers
# JSON-RPC ``4000`` with the field path, never a silent ignore. Results and payloads are OUR bug when
# they drift, so they never break a user's turn: outside the test suite a mismatch is logged once per
# name; under ``HERMES_TEST_ISOLATION`` (set by ``scripts/run_tests.sh`` / ``tests/conftest.py``) it
# raises, which is what makes the suite the gate.

STRICT = bool(os.environ.get("HERMES_TEST_ISOLATION"))
_reported: set[str] = set()


class ContractViolation(AssertionError):
    """A result/payload the gateway produced does not match its declared contract."""


def _report(kind: str, name: str, exc: ValidationError) -> None:
    if STRICT:
        raise ContractViolation(f"{kind} {name!r} violates its contract: {exc}") from exc
    if name not in _reported:
        _reported.add(name)
        logger.warning("%s %r violates its wire contract: %s", kind, name, exc)


def validate_params(contract: MethodContract | ServerRequestContract, params: dict) -> tuple[dict | None, str | None]:
    """``(params, None)`` when valid — the ORIGINAL dict, so handlers keep reading what the client
    sent; ``(None, message)`` for the ``4000`` error."""
    try:
        contract.params.model_validate(params)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "params"
        return None, f"invalid params for {contract.name}: {loc}: {first.get('msg')}"
    return params, None


def check_result(contract: MethodContract | ServerRequestContract, result: dict) -> None:
    try:
        contract.result.model_validate(result)
    except ValidationError as exc:
        _report("result of", contract.name, exc)


def check_payload(name: str, payload: dict | None) -> None:
    contract = EVENTS.get(name)
    if contract is None:
        # Completeness (every emitted name has a contract) is the generator's / the contract
        # test's job; at emit time only a DECLARED contract can be violated.
        return
    if contract.payload is None:
        if payload:
            _report("payload of", name, _no_payload_error(payload))
        return
    try:
        contract.payload.model_validate(payload or {})
    except ValidationError as exc:
        _report("payload of", name, exc)


def assert_complete(methods: dict[str, object], emitted_events: set[str], server_requests: set[str]) -> None:
    """Every registered method / emitted event name / sent server request has a contract, and no
    contract is orphaned. Raises with the full lists."""
    problems = []
    if missing := sorted(set(methods) - set(METHODS)):
        problems.append(f"methods without a contract: {missing}")
    if orphan := sorted(set(METHODS) - set(methods)):
        problems.append(f"method contracts with no handler: {orphan}")
    if missing := sorted(emitted_events - set(EVENTS)):
        problems.append(f"events without a contract: {missing}")
    if orphan := sorted(set(EVENTS) - emitted_events):
        problems.append(f"event contracts nothing emits: {orphan}")
    if missing := sorted(server_requests - set(SERVER_REQUESTS)):
        problems.append(f"server requests without a contract: {missing}")
    if orphan := sorted(set(SERVER_REQUESTS) - server_requests):
        problems.append(f"server request contracts nothing sends: {orphan}")
    if problems:
        raise ContractViolation("tui_gateway/contracts is incomplete:\n  " + "\n  ".join(problems))


def _no_payload_error(payload: dict) -> ValidationError:
    class _Empty(Payload):
        pass

    try:
        _Empty.model_validate(payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")
