"""HTTP client for the connector routes on the managed tool gateway.

Constructed per dispatch — a portal access token expires within the hour, so
auth headers are read fresh on every call (the ``managed_gateway_auth_headers``
idiom). Sync ``requests`` on purpose: the bridge branch cannot reach the
registry's async bridge, so every call here runs on the calling thread.

Injectable seams (``transport`` / ``endpoint_resolver`` / ``header_provider``)
default to the real ones; tests inject fakes instead of patching modules.

Retry policy (D29): at most ONE retry, on transport failure or 5xx only,
reusing the SAME ``x-idempotency-key``. The key is a local variable in
:meth:`ConnectorClient.execute` — one dispatch is one call frame, nothing
outside it ever retries the same dispatch, so scope guarantees same-key-on-
retry with no store to clean up. 409 means the key was reused with a
different body — always a client bug, never retried.

Only the gateway's execute route supports idempotency keys, so only routes
that are safe to repeat are retried: search and schemas are read-only, and
execute dedupes on its key. The connections route is state-changing WITHOUT
dedup support (it starts or restarts an authorization flow), so it is never
retried automatically — a lost response surfaces as an error the caller can
deliberately re-ask.

Wire models stay inside this module: callers receive plain dicts shaped for
``merge.splice_remote_results``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Optional, Protocol, Sequence

import requests

from tools.tool_gateway import wire
from tools.tool_gateway.errors import (
    GatewayAuthError,
    GatewayUnavailable,
    ToolGatewayError,
    parse_gateway_error,
)
from tools.tool_gateway.merge import PlannedCall

logger = logging.getLogger(__name__)

__all__ = ["ConnectorClient", "Transport"]

DEFAULT_TIMEOUT_SECONDS = 30.0
# One execute request carries up to MAX_CALLS_PER_DISPATCH remote tool runs;
# measured batch latency is seconds, not minutes, but give slow tools room.
EXECUTE_TIMEOUT_SECONDS = 60.0
# Search rides the availability path of EVERY tool_search once connectors are
# lit. A hung gateway degrades silently to local-only results, with no retry.
# Measured: one request with 6 use_cases takes about 7 s, so an 8 s budget sat
# on the edge and cut real answers off; 30 s tolerates a slow gateway and still
# bounds the wait. Schemas (tool_describe) is user-initiated; a short budget
# with one retry keeps its worst case at 2x this value.
SEARCH_TIMEOUT_SECONDS = 30.0
SCHEMAS_TIMEOUT_SECONDS = 10.0

_MAX_RETRIES = 1  # D29: at most one retry, same key.


class Transport(Protocol):
    """The slice of ``requests`` the client uses; tests inject a fake."""

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
    return requests  # module satisfies the protocol


def _default_endpoint_resolver() -> Optional[str]:
    """The connectors origin, or ``None`` when none resolves (scheme misconfig).

    Asks the connectors host resolver directly. This used to go through
    ``managed_vendor_endpoints("connectors")``, which invented a vendor that
    does not exist — and then discarded the ``base_url``/``upload_path`` it
    built for it. Connector routes are their own deployment's own paths
    (``v1/connectors/*``) on its own host, not a vendor passthrough and not the
    media host.
    """
    from tools.managed_gateway_auth import connector_gateway_origin

    try:
        return connector_gateway_origin() or None
    except ValueError:
        # Misconfigured TOOL_GATEWAY_SCHEME: there is no origin to call.
        return None


def _default_header_provider(url: str) -> dict:
    from tools.managed_gateway_auth import managed_gateway_auth_headers

    return managed_gateway_auth_headers(url)


class ConnectorClient:
    """One dispatch's connection to the gateway's connector routes."""

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

    # -- routes ---------------------------------------------------------

    def search(self, queries: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """POST v1/connectors/search. Returns the response as a plain dict."""
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
        """POST v1/connectors/schemas."""
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
        """POST v1/connectors/connections. Never auto-retried: this route
        starts/restarts authorization flows and the gateway offers no dedup
        key for it — a blind retry could double-submit a restart."""
        body = wire.ConnectorConnectionsRequest(
            connectors=list(connectors), reinitiate=reinitiate
        ).model_dump(by_alias=True)
        payload = self._post(wire.CONNECTOR_CONNECTIONS_PATH, body, retries=0)
        return wire.ConnectorConnectionsResponse.model_validate(payload).model_dump()

    def list_connectors(self) -> list[dict[str, Any]]:
        """GET v1/connectors, following pagination. Read-only.

        Returns the raw item dicts (``{"connector", "enabled", "connected"}``
        today; tolerant of additions). The page size cap is the gateway's.
        """
        items: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(20):  # generous page bound; the catalog is small
            path = f"{wire.CONNECTORS_PATH}?limit=50"
            if cursor:
                path += f"&cursor={cursor}"
            payload = self._request("GET", path, None)
            if not isinstance(payload, dict):
                break
            page = payload.get("items")
            if isinstance(page, list):
                items.extend(entry for entry in page if isinstance(entry, dict))
            cursor = payload.get("nextCursor")
            if not cursor:
                break
        return items

    def execute(self, planned: Sequence[PlannedCall]) -> list[dict[str, Any]]:
        """POST v1/connectors/execute — ONE request for the whole slice.

        Returns one dict per wire result, in wire order (slot ``i`` is the
        response to request ``tools[i]``): ``{"data": ..., "error": None |
        {code, message, connector, connect_url, hint}}``. Length mismatches
        are the merge layer's problem, by design.
        """
        body = wire.ConnectorExecuteRequest(
            tools=[
                wire.ConnectorExecuteCall(
                    connector=plan.connector, tool=plan.tool, arguments=plan.arguments
                )
                for plan in planned
            ]
        ).model_dump(by_alias=True)
        # Local by design (no store): one dispatch = one call frame, and a
        # transport retry below re-presents this same key by scope.
        idempotency_key = str(uuid.uuid4())
        payload = self._post(
            wire.CONNECTOR_EXECUTE_PATH,
            body,
            timeout=EXECUTE_TIMEOUT_SECONDS,
            idempotency_key=idempotency_key,
        )
        parsed = wire.ConnectorExecuteResponse.model_validate(payload)
        return [_result_dict(result) for result in parsed.results]

    # -- plumbing -------------------------------------------------------

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
                # No usable portal token; an unauthenticated request would
                # only 401 — fail fast with the same meaning.
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

        assert last_error is not None  # loop ran at least once
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
