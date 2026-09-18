"""HTTP client for managed gateway connector routes.

A dispatch-local idempotency key permits one execute retry; connections are never retried because the gateway cannot deduplicate authorization starts.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Optional, Protocol, Sequence

import requests
from pydantic import ValidationError

from tools.connectors.gateway import wire
from tools.connectors.gateway.errors import (
    GatewayAuthError,
    GatewayUnavailable,
    ToolGatewayError,
    parse_gateway_error,
)
from tools.connectors.gateway.merge import PlannedCall

logger = logging.getLogger(__name__)

__all__ = ["ConnectorClient", "Transport"]

DEFAULT_TIMEOUT_SECONDS = 30.0
EXECUTE_TIMEOUT_SECONDS = 60.0
# Search is on every enabled tool-search path and must degrade to local-only results.
SEARCH_TIMEOUT_SECONDS = 30.0
SCHEMAS_TIMEOUT_SECONDS = 10.0

_MAX_RETRIES = 1


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
    ) -> None:
        self._transport = transport or _default_transport()
        self._endpoint_resolver = endpoint_resolver or _default_endpoint_resolver
        self._header_provider = header_provider or _default_header_provider

    def search(self, queries: Sequence[dict[str, Any]]) -> dict[str, Any]:
        body = wire.ConnectorSearchRequest(
            queries=[wire.ConnectorSearchQuery(**q) for q in queries]
        ).model_dump(by_alias=True, exclude_none=True)
        payload = self._post(
            wire.CONNECTOR_SEARCH_PATH, body,
            timeout=SEARCH_TIMEOUT_SECONDS, retries=0,
        )
        parsed = wire.ConnectorSearchResponse.model_validate(payload)
        return parsed.model_dump()

    def schemas(self, tools: Sequence[str]) -> dict[str, Any]:
        body = wire.ConnectorSchemasRequest(tools=list(tools)).model_dump(
            by_alias=True
        )
        payload = self._post(
            wire.CONNECTOR_SCHEMAS_PATH, body, timeout=SCHEMAS_TIMEOUT_SECONDS
        )
        return wire.ConnectorSchemasResponse.model_validate(payload).model_dump()

    def connections(
        self, connectors: Sequence[str], *, reinitiate: bool = False
    ) -> dict[str, Any]:
        """Never retry: the gateway cannot deduplicate authorization starts."""
        body = wire.ConnectorConnectionsRequest(
            connectors=list(connectors), reinitiate=reinitiate
        ).model_dump(by_alias=True)
        payload = self._post(wire.CONNECTOR_CONNECTIONS_PATH, body, retries=0)
        return wire.ConnectorConnectionsResponse.model_validate(payload).model_dump()

    def list_connectors(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(20):
            path = f"{wire.CONNECTORS_PATH}?limit=50"
            if cursor:
                path += f"&cursor={cursor}"
            payload = self._request("GET", path, None)
            if not isinstance(payload, dict) or "error" in payload:
                raise ToolGatewayError("invalid connector list page", code="INVALID_RESPONSE")
            page = payload.get("items")
            if not isinstance(page, list) or any(not isinstance(entry, dict) for entry in page):
                raise ToolGatewayError("invalid connector list items", code="INVALID_RESPONSE")
            try:
                items.extend(wire.ConnectorListItem.model_validate(entry).model_dump(by_alias=True) for entry in page)
            except ValidationError as exc:
                raise ToolGatewayError(f"invalid connector list item: {exc.errors()[0].get('msg')}",
                                       code="INVALID_RESPONSE") from exc
            cursor = payload.get("nextCursor")
            if not cursor:
                return items
            if not isinstance(cursor, str):
                raise ToolGatewayError("invalid connector list cursor", code="INVALID_RESPONSE")
        raise ToolGatewayError("connector list pagination incomplete", code="INVALID_RESPONSE")

    def execute(self, planned: Sequence[PlannedCall]) -> list[dict[str, Any]]:
        """Return gateway results in request order; merge owns length mismatches."""
        body = wire.ConnectorExecuteRequest(
            tools=[
                wire.ConnectorExecuteCall(
                    connector=plan.connector, tool=plan.tool, arguments=plan.arguments
                )
                for plan in planned
            ]
        ).model_dump(by_alias=True)
        # Keep the key dispatch-local so its retry reuses it without a shared store.
        idempotency_key = str(uuid.uuid4())
        payload = self._post(
            wire.CONNECTOR_EXECUTE_PATH,
            body,
            timeout=EXECUTE_TIMEOUT_SECONDS,
            idempotency_key=idempotency_key,
        )
        parsed = wire.ConnectorExecuteResponse.model_validate(payload)
        return [_result_dict(result) for result in parsed.results]

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        idempotency_key: Optional[str] = None,
        retries: int = _MAX_RETRIES,
    ) -> Any:
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
    ) -> Any:
        origin = self._endpoint_resolver()
        if not origin:
            raise GatewayUnavailable(
                "no tool gateway origin resolves", code="NO_ORIGIN"
            )
        url = f"{origin.rstrip('/')}/{path}"

        last_error: Optional[ToolGatewayError] = None
        for attempt in range(1 + retries):
            headers = dict(self._header_provider(url))
            if not headers:
                raise GatewayAuthError(
                    "no portal access token available", code="NO_TOKEN", status=401
                )
            headers["Content-Type"] = "application/json"
            if idempotency_key:
                headers["x-idempotency-key"] = idempotency_key

            try:
                response = self._transport.request(
                    method, url, headers=headers, json=body, timeout=timeout
                )
            except Exception as exc:
                last_error = ToolGatewayError(
                    f"transport failure: {exc}", code="TRANSPORT_ERROR", retryable=True
                )
                logger.debug(
                    "Connector %s attempt %d transport failure: %s", path, attempt + 1, exc
                )
                continue

            status = int(getattr(response, "status_code", 0))
            if 200 <= status < 300:
                return response.json()

            error = parse_gateway_error(status, _safe_json(response))
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


def _result_dict(result: wire.ConnectorExecuteResult) -> dict[str, Any]:
    error = None
    if result.error is not None:
        error = {"code": result.error.code, "message": result.error.message}
        if result.error.connector:
            error["connector"] = result.error.connector
        if result.error.connect_url:
            error["connect_url"] = result.error.connect_url
        if result.error.hint:
            error["hint"] = result.error.hint
    return {"data": result.data, "error": error}


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return getattr(response, "text", None)
