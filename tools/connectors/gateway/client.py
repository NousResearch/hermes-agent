"""HTTP client for managed gateway connector routes.

A dispatch-local idempotency key permits one execute retry; connections are never retried because the gateway cannot deduplicate authorization starts.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

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
STORAGE_TIMEOUT_SECONDS = 60.0
_STORAGE_CHUNK_BYTES = 64 * 1024
_STORAGE_PUT_ROUTE = "connectors.storage.put"
_STORAGE_GET_ROUTE = "connectors.storage.get"

_MAX_RETRIES = 1
# Execute retries these once under the same idempotency key; the gateway replays a settled call.
_RETRY_ONCE_STATUSES = frozenset({502, 503, 504})
# A 429 asking for a longer wait is returned to the model rather than holding the turn.
_MAX_RETRY_AFTER_SECONDS = 10.0
# Gateway origins that rejected the single-call execute form; they get the array form for the process.
_LEGACY_EXECUTE_ORIGINS: set[str] = set()
_EXECUTE_ROUTE = "connectors.execute"


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


class StorageTransport(Protocol):
    """Plain HTTP to presigned storage URLs: never carries the gateway's auth headers."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict] = None,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
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
        sleep: Callable[[float], None] = time.sleep,
        storage: Optional[StorageTransport] = None,
    ) -> None:
        self._transport = transport or _default_transport()
        self._storage = storage or requests
        self._endpoint_resolver = endpoint_resolver or _default_endpoint_resolver
        self._header_provider = header_provider or _default_header_provider
        # The model call this client serves, and who asked for a connect; both go on the wire.
        self._call = call
        self._initiator = initiator
        self._sleep = sleep

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

    def stage_file(self, *, toolkit: str, tool: str, filename: str, mimetype: str, size: int) -> dict[str, Any]:
        """A presigned upload for one file input: ``{upload_url, headers, descriptor, expires_at}``.
        Never retried: each stage mints a new storage key."""
        body = wire.ConnectorFileUploadRequest(
            toolkit=toolkit, tool=tool, filename=filename, mimetype=mimetype, size=size,
        ).model_dump(by_alias=True)
        reply = self._post(wire.CONNECTOR_FILE_UPLOAD_PATH, body, retries=0)
        return self._checked(wire.ConnectorFileUploadResponse, reply, "connector file upload").model_dump()

    def put_file(self, url: str, headers: Mapping[str, str], body: bytes) -> None:
        """PUT the bytes to a presigned storage URL with exactly the headers the gateway returned."""
        try:
            response = self._storage.request("PUT", url, headers=dict(headers), data=body,
                                             timeout=STORAGE_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            log_connector_outcome(_STORAGE_PUT_ROUTE, None, code="TRANSPORT_ERROR", call=self._call)
            raise ToolGatewayError(f"storage upload failed: {type(exc).__name__}", code="TRANSPORT_ERROR") from exc
        status = int(getattr(response, "status_code", 0))
        failed = not 200 <= status < 300
        log_connector_outcome(_STORAGE_PUT_ROUTE, status, code=f"HTTP_{status}" if failed else None, call=self._call)
        if failed:
            raise ToolGatewayError(f"storage upload answered {status}", code=f"HTTP_{status}", status=status)

    def get_file(self, url: str, sink: Callable[[bytes], None], *, max_bytes: int) -> None:
        """Stream a presigned storage GET into ``sink``; more than ``max_bytes`` raises ``TOO_LARGE``."""
        try:
            response = self._storage.request("GET", url, headers={}, timeout=STORAGE_TIMEOUT_SECONDS, stream=True)
        except requests.RequestException as exc:
            log_connector_outcome(_STORAGE_GET_ROUTE, None, code="TRANSPORT_ERROR", call=self._call)
            raise ToolGatewayError(f"download failed: {type(exc).__name__}", code="TRANSPORT_ERROR") from exc
        status = int(getattr(response, "status_code", 0))
        try:
            if not 200 <= status < 300:
                log_connector_outcome(_STORAGE_GET_ROUTE, status, code=f"HTTP_{status}", call=self._call)
                raise ToolGatewayError(f"download answered {status}", code=f"HTTP_{status}", status=status)
            received = 0
            for chunk in response.iter_content(chunk_size=_STORAGE_CHUNK_BYTES):
                received += len(chunk)
                if received > max_bytes:
                    log_connector_outcome(_STORAGE_GET_ROUTE, status, code="TOO_LARGE", call=self._call)
                    raise ToolGatewayError(f"file is larger than {max_bytes // (1024 * 1024)} MiB",
                                           code="TOO_LARGE", status=status)
                sink(chunk)
        except requests.RequestException as exc:
            log_connector_outcome(_STORAGE_GET_ROUTE, status, code="TRANSPORT_ERROR", call=self._call)
            raise ToolGatewayError(f"download failed: {type(exc).__name__}", code="TRANSPORT_ERROR") from exc
        finally:
            response.close()
        log_connector_outcome(_STORAGE_GET_ROUTE, status, call=self._call)

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
        """One result per planned call, in request order: ``{"data", "error"}``.

        Each call is its own single-call request; a failure status with an error body is that call's
        error entry. A gateway that predates the single-call form gets the legacy array once and is
        remembered for the process."""
        origin = self._endpoint_resolver()
        if origin and origin in _LEGACY_EXECUTE_ORIGINS:
            return self._execute_legacy(planned, return_to=return_to, op=op)
        results: list[dict[str, Any]] = []
        for index, plan in enumerate(planned):
            result = self._execute_one(plan, return_to=return_to, op=op)
            if result is None:
                if origin:
                    _LEGACY_EXECUTE_ORIGINS.add(origin)
                logger.info("Connector gateway predates single-call execute; using the array form")
                return results + self._execute_legacy(planned[index:], return_to=return_to, op=op)
            results.append(result)
        return results

    def _execute_one(
        self, plan: PlannedCall, *, return_to: Optional[str], op: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """``None`` when the gateway rejected the single-call form itself (an older gateway)."""
        body = wire.ConnectorExecuteSingleRequest(
            call=wire.ConnectorExecuteCall(connector=plan.connector, tool=plan.tool, arguments=plan.arguments),
            return_to=return_to, op=op,
        ).model_dump(by_alias=True, exclude_none=True)
        url = self._url(wire.CONNECTOR_EXECUTE_PATH)
        idempotency_key = str(uuid.uuid4())
        for attempt in range(1 + _MAX_RETRIES):
            response = self._send("POST", url, body, timeout=EXECUTE_TIMEOUT_SECONDS, idempotency_key=idempotency_key,
                                  retry_transport=attempt < _MAX_RETRIES)
            if response is None:
                continue
            status = int(getattr(response, "status_code", 0))
            payload = _safe_json(response)
            request_id = response_request_id(response)
            if 200 <= status < 300:
                reply = _Reply(payload, status, request_id, _EXECUTE_ROUTE)
                parsed = self._checked(wire.ConnectorExecuteSingleResponse, reply, "connector execute result")
                return _result_dict(parsed.result)
            headers = getattr(response, "headers", None)
            failure = _call_failure(payload)
            code = failure.code if failure else parse_gateway_error(status, payload, headers).code
            log_connector_outcome(_EXECUTE_ROUTE, status, code=code,
                                  request_id=(failure.request_id if failure else None) or request_id, call=self._call)
            if status == 400 and _rejects_single_call_form(payload):
                return None
            wait = _retry_wait(status, headers) if attempt < _MAX_RETRIES else None
            if wait is not None:
                logger.debug("Connector execute attempt %d got %d; retrying with same key", attempt + 1, status)
                if wait:
                    self._sleep(wait)
                continue
            if failure is None:
                raise parse_gateway_error(status, payload, headers)
            return {"data": None, "error": _error_dict(failure, status=status, request_id=request_id)}
        raise ToolGatewayError("connector execute retries exhausted", code="TRANSPORT_ERROR", retryable=True)

    def _execute_legacy(
        self, planned: Sequence[PlannedCall], *, return_to: Optional[str] = None, op: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """The array form an older gateway accepts: a 200 envelope with per-call errors inside."""
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

    def _url(self, path: str) -> str:
        origin = self._endpoint_resolver()
        if not origin:
            raise GatewayUnavailable(
                "no tool gateway origin resolves", code="NO_ORIGIN"
            )
        return f"{origin.rstrip('/')}/{path}"

    def _send(
        self, method: str, url: str, body: Optional[dict[str, Any]], *,
        timeout: float, idempotency_key: Optional[str], retry_transport: bool,
    ) -> Any:
        """One HTTP attempt. A transport failure returns ``None`` when the caller may retry, else raises."""
        headers = self._attempt_headers(url, idempotency_key)
        try:
            return self._transport.request(method, url, headers=headers, json=body, timeout=timeout)
        except Exception as exc:
            log_connector_outcome(_route_key(url.split("/", 3)[-1]), None, code="TRANSPORT_ERROR", call=self._call)
            if retry_transport:
                return None
            raise ToolGatewayError(
                f"transport failure: {exc}", code="TRANSPORT_ERROR", retryable=True
            ) from exc

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
        url = self._url(path)
        route = _route_key(path)
        for attempt in range(1 + retries):
            response = self._send(method, url, body, timeout=timeout, idempotency_key=idempotency_key,
                                  retry_transport=attempt < retries)
            if response is None:
                continue
            status = int(getattr(response, "status_code", 0))
            request_id = response_request_id(response)
            if 200 <= status < 300:
                return _Reply(_safe_json(response), status, request_id, route)

            error = parse_gateway_error(status, _safe_json(response), getattr(response, "headers", None))
            error.request_id = error.request_id or request_id
            log_connector_outcome(route, status, code=error.code, request_id=error.request_id, call=self._call)
            if error.retryable and attempt < retries:
                logger.debug(
                    "Connector %s attempt %d got %d; retrying with same key",
                    path,
                    attempt + 1,
                    status,
                )
                continue
            raise error
        raise ToolGatewayError("connector request retries exhausted", code="TRANSPORT_ERROR", retryable=True)

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


def _result_dict(result: wire.ConnectorExecuteResult | wire.ConnectorExecuteSingleResult) -> dict[str, Any]:
    error = getattr(result, "error", None)
    entry = {"data": result.data, "error": _error_dict(error) if error is not None else None}
    if result.account:
        entry["account"] = result.account
    if result.data_use is not None and result.data_use.requires:
        entry["data_use"] = list(result.data_use.requires)
    return entry


def _error_dict(
    failure: wire.ConnectorToolError, *, status: Optional[int] = None, request_id: Optional[str] = None,
) -> dict[str, Any]:
    """The plain error entry merge renders. ``status`` and ``request_id`` are for logs, never the model."""
    error: dict[str, Any] = {"code": failure.code, "message": failure.message}
    family = wire.execute_error_family(failure.code, failure.family)
    if family:
        error["family"] = family
    for key, value in (("connector", failure.connector), ("connect_url", failure.connect_url),
                       ("connection_id", failure.connection_id), ("hint", failure.hint)):
        if value:
            error[key] = value
    if status is not None:
        error["status"] = status
    request_id = getattr(failure, "request_id", None) or request_id
    if request_id:
        error["request_id"] = str(request_id)
    return error


def _call_failure(payload: Any) -> Optional[wire.ConnectorExecuteFailure]:
    """A single-call failure body: a flat ``{code, message}`` the gateway marks with a family or a code
    this client knows. Anything else (auth, dark route, idempotency conflict) stays a route failure."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get("code"), str):
        return None
    if payload.get("family") is None and payload["code"] not in wire.EXECUTE_ERROR_CODES:
        return None
    try:
        return wire.ConnectorExecuteFailure.model_validate(
            {"message": "The connector gateway reported an error.", **payload})
    except ValidationError:
        return None


def _rejects_single_call_form(payload: Any) -> bool:
    """An older gateway validates every body as the array form and names ``tools`` in its 400."""
    if not isinstance(payload, Mapping):
        return False
    issues = payload.get("issues")
    if isinstance(issues, list):
        for issue in issues:
            path = issue.get("path") if isinstance(issue, Mapping) else None
            if isinstance(path, list) and path[:1] == ["tools"]:
                return True
    envelope = payload.get("error")
    code = envelope.get("code") if isinstance(envelope, Mapping) else payload.get("code")
    if code != "VALIDATION_ERROR":
        return False
    text = envelope if isinstance(envelope, Mapping) else payload
    return "tools" in f"{text.get('message', '')} {text.get('details', '')}"


def _retry_wait(status: int, headers: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Seconds to wait before the one retry, or ``None`` when this status is not retried.

    A 429 is retried only when the gateway names a wait, and only a short one; a longer wait goes back
    to the model as the call's error instead of holding the turn."""
    if status in _RETRY_ONCE_STATUSES:
        return 0.0
    if status != 429:
        return None
    raw = (headers or {}).get("Retry-After")
    try:
        wait = float(raw)
    except (TypeError, ValueError):
        return None
    return wait if 0 <= wait <= _MAX_RETRY_AFTER_SECONDS else None


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return getattr(response, "text", None)
