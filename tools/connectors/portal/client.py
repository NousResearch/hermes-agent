from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import quote

import requests
from pydantic import ValidationError

from hermes_cli.nous_account import resolve_nous_portal_base_url
from tools.connectors.gateway.errors import (
    GatewayAuthError,
    GatewayUnavailable,
    ToolGatewayError,
    parse_gateway_error,
)
from tools.connectors.gateway.wire import ConnectorAccount, ConnectorAccountsResponse, RemovedConnectorAccount
from tools.connectors.portal.errors import InvalidConnectorSlug, PortalConnectorUnavailable, PortalToolsUnavailable
from tools.connectors.portal.wire import (
    ConnectorCatalogResponse,
    ConnectorPolicyResponse,
    ConnectorPolicyWriteResponse,
    ConnectorToolsListing,
)
from tools.managed_gateway_auth import read_nous_access_token
from tools.managed_gateway_identity import connector_headers, log_connector_outcome, response_request_id


DEFAULT_TIMEOUT_SECONDS = 30.0
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class NotModified:
    pass


def validate_slug(slug: str) -> None:
    if not _SLUG_RE.fullmatch(slug):
        raise InvalidConnectorSlug("connector must be a slug")


def _default_transport() -> Transport:
    return requests


def _default_endpoint_resolver() -> str:
    return resolve_nous_portal_base_url()


def _default_header_provider(_url: str) -> dict[str, str]:
    token = read_nous_access_token()
    return {"Authorization": f"Bearer {token}"} if isinstance(token, str) and token.strip() else {}


class PortalConnectorClient:

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        endpoint_resolver: Callable[[], str] | None = None,
        header_provider: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        self._transport = transport or _default_transport()
        self._endpoint_resolver = endpoint_resolver or _default_endpoint_resolver
        self._header_provider = header_provider or _default_header_provider

    def origin(self) -> str:
        return self._endpoint_resolver().rstrip("/")

    def require_authentication(self) -> None:
        self._authorized_headers(self.origin())

    def authorization_token(self) -> str | None:
        authorization = self._authorized_headers(self.origin()).get("Authorization")
        if not isinstance(authorization, str):
            return None
        scheme, _, token = authorization.partition(" ")
        return token.strip() if scheme.lower() == "bearer" and token.strip() else None

    def _authorized_headers(self, url: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json", **(extra or {}), **connector_headers(), **self._header_provider(url)}
        if not isinstance(headers.get("Authorization"), str) or not headers["Authorization"].strip():
            raise GatewayAuthError("portal authorization required", code="NO_TOKEN", status=401)
        return headers

    def tools(self, slug: str, *, if_none_match: str | None = None) -> ConnectorToolsListing | NotModified:
        validate_slug(slug)
        headers = {"If-None-Match": if_none_match} if if_none_match else None
        try:
            response, status, route = self._request("GET", f"/api/v1/connectors/{slug}/tools", headers=headers)
        except ToolGatewayError as exc:
            missing_route = isinstance(exc, GatewayUnavailable) and exc.code != "connector_not_found"
            if type(exc) is not ToolGatewayError and not missing_route:
                raise
            raise PortalToolsUnavailable(
                "portal tools unavailable",
                code=exc.code,
                status=exc.status,
                request_id=exc.request_id,
                retryable=exc.retryable,
            ) from exc
        if status == 304:
            return NotModified()
        listing = self._parse(ConnectorToolsListing, response, status, route, PortalToolsUnavailable, "portal tools unavailable")
        etag = getattr(response, "headers", {}).get("etag")
        return listing.model_copy(update={"etag": etag}) if isinstance(etag, str) and etag else listing

    def catalog(self) -> ConnectorCatalogResponse:
        response, status, route = self._request("GET", "/api/v1/connectors/catalog")
        return self._parse(ConnectorCatalogResponse, response, status, route, PortalConnectorUnavailable, "portal catalog unavailable")

    def list_accounts(self) -> list[dict[str, Any]]:
        response, status, route = self._request("GET", "/api/v1/connectors/accounts")
        inventory = self._parse(ConnectorAccountsResponse, response, status, route, PortalConnectorUnavailable, "portal accounts unavailable")
        return [account.model_dump(by_alias=True) for account in inventory.accounts]

    def rename_account(self, connection_id: str, alias: str) -> dict[str, Any]:
        path = f"/api/v1/connectors/accounts/{quote(connection_id, safe='')}"
        response, status, route = self._request("PATCH", path, {"alias": alias})
        account = self._parse(ConnectorAccount, response, status, route, PortalConnectorUnavailable, "portal accounts unavailable")
        return account.model_dump(by_alias=True)

    def delete_account(self, connection_id: str) -> dict[str, Any]:
        path = f"/api/v1/connectors/accounts/{quote(connection_id, safe='')}"
        response, status, route = self._request("DELETE", path)
        removed = self._parse(RemovedConnectorAccount, response, status, route, PortalConnectorUnavailable, "portal accounts unavailable")
        return removed.model_dump(by_alias=True)

    def policy(self) -> ConnectorPolicyResponse:
        response, status, route = self._request("GET", "/api/v1/connectors/policy")
        return self._parse(ConnectorPolicyResponse, response, status, route, PortalConnectorUnavailable, "portal policy unavailable")

    def set_policy(self, body: dict[str, Any]) -> ConnectorPolicyWriteResponse:
        response, status, route = self._request("PUT", "/api/v1/connectors/policy", body)
        return self._parse(ConnectorPolicyWriteResponse, response, status, route, PortalConnectorUnavailable, "portal policy unavailable")

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[Any, int, str]:
        """The answer, its status and route key. A failed status is logged and raised here; a 2xx is
        logged by ``_parse`` once its body is checked."""
        url = f"{self.origin()}{path}"
        headers = self._authorized_headers(url, headers)
        try:
            response = self._transport.request(method, url, headers=headers, json=body, timeout=DEFAULT_TIMEOUT_SECONDS)
        except Exception as exc:
            log_connector_outcome(_route_key(path), None, code="TRANSPORT_ERROR")
            raise PortalConnectorUnavailable("portal connector metadata unavailable", code="TRANSPORT_ERROR", retryable=True) from exc
        status = int(getattr(response, "status_code", 0))
        request_id = response_request_id(response)
        if not 200 <= status < 300 and status != 304:
            error = parse_gateway_error(status, _safe_json(response), getattr(response, "headers", None))
            error.request_id = error.request_id or request_id
            log_connector_outcome(_route_key(path), status, code=error.code, request_id=error.request_id)
            raise error
        if status == 304:
            log_connector_outcome(_route_key(path), status, request_id=request_id)
        return response, status, _route_key(path)

    @staticmethod
    def _parse(model, response: Any, status: int, route: str, error_type, message: str):
        request_id = response_request_id(response)
        try:
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("response must be an object")
            parsed = model.model_validate(payload)
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            log_connector_outcome(route, status, code="INVALID_RESPONSE", request_id=request_id)
            raise error_type(message, code="INVALID_RESPONSE", status=status, request_id=request_id) from exc
        log_connector_outcome(route, status, request_id=request_id)
        return parsed


def _route_key(path: str) -> str:
    """``/api/v1/connectors/gmail/tools`` -> ``portal.connectors.tools``: no slugs or ids."""
    parts = path.strip("/").split("/")[2:]
    named = [part for part in parts if part in _PORTAL_ROUTE_WORDS]
    return ".".join(["portal", *named]) or "portal"


_PORTAL_ROUTE_WORDS = frozenset({"connectors", "tools", "catalog", "accounts", "policy"})


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return None
