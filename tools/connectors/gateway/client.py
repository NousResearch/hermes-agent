"""HTTP client for managed gateway connector routes.

A dispatch-local idempotency key permits one execute retry; connections are never retried because the gateway cannot deduplicate authorization starts.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

import requests
from pydantic import ValidationError

from tools.connectors.gateway import wire
from tools.connectors.gateway.config import session_platform
from tools.connectors.gateway.errors import (
    GatewayAuthError,
    GatewayUnavailable,
    ToolGatewayError,
    parse_gateway_error,
)

from tools.connectors.gateway.merge import PlannedCall
from tools.managed_gateway_identity import CallRef, connector_headers, log_connector_outcome, response_request_id

logger = logging.getLogger(__name__)

__all__ = ["ConnectorClient", "Transport", "return_to_args"]

DEFAULT_TIMEOUT_SECONDS = 30.0
EXECUTE_TIMEOUT_SECONDS = 60.0
# Search is on every enabled tool-search path and must degrade to local-only results.
SEARCH_TIMEOUT_SECONDS = 30.0
SCHEMAS_TIMEOUT_SECONDS = 10.0

_MAX_RETRIES = 1


@dataclass(frozen=True)
class _Reply:
    """A 2xx answer before its body is checked; the outcome line is written once the body is read."""

    payload: Any
    status: int
    request_id: Optional[str]
    route: str


class Transport(Protocol):

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> Any: ...


def _default_transport() -> Transport:
    return requests


def _default_endpoint_resolver() -> Optional[str]:
    """Resolve the connector deployment origin directly, not as a vendor passthrough."""
    from tools.managed_gateway_auth import connector_gateway_origin

    try:
        return connector_gateway_origin() or None
    except ValueError:
        return None


def _default_header_provider(url: str) -> dict:
    from tools.managed_gateway_auth import managed_gateway_auth_headers

    return managed_gateway_auth_headers(url)


class ConnectorClient:
    def __init__(
        self,
        *,
        transport: Optional[Transport] = None,
        endpoint_resolver: Optional[Callable[[], Optional[str]]] = None,
        header_provider: Optional[Callable[[str], dict]] = None,
        call: Optional[CallRef] = None,
        initiator: Optional[str] = None,
    ) -> None:
        self._transport = transport or _default_transport()
        self._endpoint_resolver = endpoint_resolver or _default_endpoint_resolver
        self._header_provider = header_provider or _default_header_provider
        # The model call this client serves, and who asked for a connect; both go on the wire.
        self._call = call
        self._initiator = initiator

    def search(self, queries: Sequence[dict[str, Any]]) -> dict[str, Any]:
        body = wire.ConnectorSearchRequest(
            queries=[wire.ConnectorSearchQuery(**q) for q in queries]
        ).model_dump(by_alias=True, exclude_none=True)
        reply = self._post(
            wire.CONNECTOR_SEARCH_PATH, body,
            timeout=SEARCH_TIMEOUT_SECONDS, retries=0,
        )
        return self._checked(wire.ConnectorSearchResponse, reply, "connector search").model_dump()

    def schemas(self, tools: Sequence[str]) -> dict[str, Any]:
        body = wire.ConnectorSchemasRequest(tools=list(tools)).model_dump(
            by_alias=True
        )
        reply = self._post(
            wire.CONNECTOR_SCHEMAS_PATH, body, timeout=SCHEMAS_TIMEOUT_SECONDS
        )
        return self._checked(wire.ConnectorSchemasResponse, reply, "connector schemas").model_dump()

    def connections(
        self, connectors: Sequence[str], *, reinitiate: bool = False, alias: Optional[str] = None,
        connection_id: Optional[str] = None, return_to: Optional[str] = None, op: Optional[str] = None,
    ) -> dict[str, Any]:
        """Never retry: the gateway cannot deduplicate authorization starts. ``alias`` names the one
        account the request connects (or, with ``reinitiate``, repairs); ``connection_id`` names an
        existing account to repair, named or not."""
        body = wire.ConnectorConnectionsRequest(
            connectors=list(connectors), reinitiate=reinitiate, alias=alias, connection_id=connection_id,
            return_to=return_to, op=op, initiator=self._initiator,
        ).model_dump(by_alias=True, exclude_none=True)
        reply = self._post(wire.CONNECTOR_CONNECTIONS_PATH, body, retries=0)
        return self._checked(wire.ConnectorConnectionsResponse, reply, "connector connections").model_dump()

    def list_connectors(self, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> list[dict[str, Any]]:
        """Every page of the session's toolkit list, each typed whole. ``timeout`` applies per page: the
        watcher bounds it by the operation's remaining deadline so a stalled page cannot hold the
        operation open."""
        items: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(20):
            path = f"{wire.CONNECTORS_PATH}?limit=50"
            if cursor:
                path += f"&cursor={cursor}"
            page = self._checked(wire.ConnectorListResponse, self._request("GET", path, None, timeout=timeout),
                                 "connector list page")
            items.extend(item.model_dump(by_alias=True) for item in page.items)
            cursor = page.next_cursor
            if not cursor:
                return items
        raise ToolGatewayError("connector list pagination incomplete", code="INVALID_RESPONSE")

    def account_status(
        self, connection_id: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
    ) -> Optional[dict[str, Any]]:
        """One account's row; ``None`` when the gateway no longer knows it. A 429 raises ``RateLimited``.
        ``timeout`` is the watcher's remaining deadline, so a stalled read cannot outlive its operation."""
        try:
            reply = self._request("GET", f"{wire.CONNECTOR_ACCOUNTS_PATH}/{connection_id}", None, timeout=timeout)
        except GatewayUnavailable as exc:
            if exc.code == "connection_not_found":
                return None
            raise
        return self._checked(wire.ConnectorAccount, reply, "connector account").model_dump(by_alias=True)

    def _checked(self, model: Any, reply: _Reply, what: str) -> Any:
        """Parse a 2xx body and write its outcome line: a body that does not parse is a failure,
        logged with the gateway's request id like any failed status."""
        try:
            parsed = self._parse(model, reply.payload, what)
        except ToolGatewayError as exc:
            exc.status, exc.request_id = reply.status, reply.request_id
            self._log(reply, exc.code)
            raise
        if not isinstance(parsed, wire.ConnectorExecuteResponse):
            self._log(reply)
            return parsed
        # The execute envelope answers 200 even when a call failed; each failed call is a failure line.
        failed = [result.error.code for result in parsed.results if result.error is not None]
        for code in failed or [None]:
            self._log(reply, code)
        return parsed

    def _log(self, reply: _Reply, code: Optional[str] = None) -> None:
        log_connector_outcome(reply.route, reply.status, code=code, request_id=reply.request_id, call=self._call)

    @staticmethod
    def _parse(model: Any, payload: Any, what: str) -> Any:
        if not isinstance(payload, dict) or "error" in payload:
            raise ToolGatewayError(f"invalid {what}", code="INVALID_RESPONSE")
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            raise ToolGatewayError(f"invalid {what}: {'.'.join(str(p) for p in first.get('loc', ()))}: {first.get('msg')}",
                                   code="INVALID_RESPONSE") from exc

    def execute(
        self, planned: Sequence[PlannedCall], *, return_to: Optional[str] = None, op: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return gateway results in request order; merge owns length mismatches."""
        body = wire.ConnectorExecuteRequest(
            tools=[
                wire.ConnectorExecuteCall(
                    connector=plan.connector, tool=plan.tool, arguments=plan.arguments
                )
                for plan in planned
            ],
            return_to=return_to, op=op,
        ).model_dump(by_alias=True, exclude_none=True)
        # Keep the key dispatch-local so its retry reuses it without a shared store.
        idempotency_key = str(uuid.uuid4())
        reply = self._post(
            wire.CONNECTOR_EXECUTE_PATH,
            body,
            timeout=EXECUTE_TIMEOUT_SECONDS,
            idempotency_key=idempotency_key,
        )
        parsed = self._checked(wire.ConnectorExecuteResponse, reply, "connector execute response")
        return [_result_dict(result) for result in parsed.results]

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        idempotency_key: Optional[str] = None,
        retries: int = _MAX_RETRIES,
    ) -> _Reply:
        return self._request(
            "POST", path, body,
            timeout=timeout, idempotency_key=idempotency_key, retries=retries,
        )

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict[str, Any]],
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        idempotency_key: Optional[str] = None,
        retries: int = _MAX_RETRIES,
    ) -> _Reply:
        """The 2xx reply, unchecked; the caller logs its outcome once the body is read. A failed
        status is logged and raised here."""
        origin = self._endpoint_resolver()
        if not origin:
            raise GatewayUnavailable(
                "no tool gateway origin resolves", code="NO_ORIGIN"
            )
        url = f"{origin.rstrip('/')}/{path}"

        route = _route_key(path)
        last_error: Optional[ToolGatewayError] = None
        for attempt in range(1 + retries):
            headers = self._attempt_headers(url, idempotency_key)
            try:
                response = self._transport.request(
                    method, url, headers=headers, json=body, timeout=timeout
                )
            except Exception as exc:
                last_error = ToolGatewayError(
                    f"transport failure: {exc}", code="TRANSPORT_ERROR", retryable=True
                )
                log_connector_outcome(route, None, code="TRANSPORT_ERROR", call=self._call)
                continue

            status = int(getattr(response, "status_code", 0))
            request_id = response_request_id(response)
            if 200 <= status < 300:
                return _Reply(_safe_json(response), status, request_id, route)

            error = parse_gateway_error(status, _safe_json(response), getattr(response, "headers", None))
            error.request_id = error.request_id or request_id
            log_connector_outcome(route, status, code=error.code, request_id=error.request_id, call=self._call)
            if error.retryable and attempt < retries:
                last_error = error
                logger.debug(
                    "Connector %s attempt %d got %d; retrying with same key",
                    path,
                    attempt + 1,
                    status,
                )
                continue
            raise error

        assert last_error is not None
        raise last_error

    def _attempt_headers(self, url: str, idempotency_key: Optional[str]) -> dict[str, str]:
        """Every header of one HTTP attempt: auth, caller identity, and the idempotency key."""
        headers = dict(self._header_provider(url))
        if not headers:
            raise GatewayAuthError(
                "no portal access token available", code="NO_TOKEN", status=401
            )
        headers.update(connector_headers(self._call))
        headers["Content-Type"] = "application/json"
        if idempotency_key:
            headers["x-idempotency-key"] = idempotency_key
        return headers


def return_to_args(*, op: Optional[str] = None) -> dict[str, Any]:
    """The ``returnTo`` / ``op`` arguments a connect or execute call carries so the vendor's done page
    can send the browser back to the app that asked for the connection.

    Only the desktop registers a URL scheme for that return, so every other surface sends neither
    field and keeps the vendor's own done page. The dev build registers ``hermes-dev://`` instead, and
    announces itself to its backend with ``HERMES_DESKTOP_DEV_SERVER``."""
    if session_platform() != "desktop":
        return {}
    target = "hermes-desktop-dev" if os.environ.get("HERMES_DESKTOP_DEV_SERVER") else "hermes-desktop"
    return {"return_to": target, "op": op} if op else {"return_to": target}


def _route_key(path: str) -> str:
    """``v1/connectors/accounts/<id>?x`` -> ``connectors.accounts``: the route without ids or query."""
    parts = path.split("?", 1)[0].strip("/").split("/")[1:]
    if len(parts) > 2:
        parts = parts[:2]
    return ".".join(parts) or path


def _result_dict(result: wire.ConnectorExecuteResult) -> dict[str, Any]:
    error = None
    if result.error is not None:
        error = {"code": result.error.code, "message": result.error.message}
        if result.error.connector:
            error["connector"] = result.error.connector
        if result.error.connect_url:
            error["connect_url"] = result.error.connect_url
        if result.error.connection_id:
            error["connection_id"] = result.error.connection_id
        if result.error.hint:
            error["hint"] = result.error.hint
    entry = {"data": result.data, "error": error}
    if result.account:
        entry["account"] = result.account
    return entry


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return getattr(response, "text", None)
