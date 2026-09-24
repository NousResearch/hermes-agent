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
from tools.connectors.gateway.wire import ConnectorAccountsResponse, RemovedConnectorAccount
from tools.connectors.portal.errors import InvalidConnectorSlug, PortalConnectorUnavailable, PortalToolsUnavailable
from tools.connectors.portal.wire import (
    ConnectorCatalogResponse,
    ConnectorPolicyResponse,
    ConnectorPolicyWriteResponse,
    ConnectorToolsListing,
)
from tools.managed_gateway_auth import read_nous_access_token


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
        headers = {"Accept": "application/json", **(extra or {}), **self._header_provider(url)}
        if not isinstance(headers.get("Authorization"), str) or not headers["Authorization"].strip():
            raise GatewayAuthError("portal authorization required", code="NO_TOKEN", status=401)
        return headers

    def tools(self, slug: str, *, if_none_match: str | None = None) -> ConnectorToolsListing | NotModified:
        validate_slug(slug)
        headers = {"If-None-Match": if_none_match} if if_none_match else None
        try:
            response, status = self._request("GET", f"/api/v1/connectors/{slug}/tools", headers=headers)
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
        listing = self._parse(ConnectorToolsListing, response, status, PortalToolsUnavailable, "portal tools unavailable")
        etag = getattr(response, "headers", {}).get("etag")
        return listing.model_copy(update={"etag": etag}) if isinstance(etag, str) and etag else listing

    def catalog(self) -> ConnectorCatalogResponse:
        response, status = self._request("GET", "/api/v1/connectors/catalog")
        return self._parse(ConnectorCatalogResponse, response, status, PortalConnectorUnavailable, "portal catalog unavailable")

    def list_accounts(self) -> list[dict[str, Any]]:
        response, status = self._request("GET", "/api/v1/connectors/accounts")
        inventory = self._parse(ConnectorAccountsResponse, response, status, PortalConnectorUnavailable, "portal accounts unavailable")
        return [account.model_dump(by_alias=True) for account in inventory.accounts]

    def delete_account(self, connection_id: str) -> dict[str, Any]:
        path = f"/api/v1/connectors/accounts/{quote(connection_id, safe='')}"
        response, status = self._request("DELETE", path)
        removed = self._parse(RemovedConnectorAccount, response, status, PortalConnectorUnavailable, "portal accounts unavailable")
        return removed.model_dump(by_alias=True)

    def policy(self) -> ConnectorPolicyResponse:
        response, status = self._request("GET", "/api/v1/connectors/policy")
        return self._parse(ConnectorPolicyResponse, response, status, PortalConnectorUnavailable, "portal policy unavailable")

    def set_policy(self, body: dict[str, Any]) -> ConnectorPolicyWriteResponse:
        response, status = self._request("PUT", "/api/v1/connectors/policy", body)
        return self._parse(ConnectorPolicyWriteResponse, response, status, PortalConnectorUnavailable, "portal policy unavailable")

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[Any, int]:
        url = f"{self.origin()}{path}"
        headers = self._authorized_headers(url, headers)
        try:
            response = self._transport.request(method, url, headers=headers, json=body, timeout=DEFAULT_TIMEOUT_SECONDS)
        except Exception as exc:
            raise PortalConnectorUnavailable("portal connector metadata unavailable", code="TRANSPORT_ERROR", retryable=True) from exc
        status = int(getattr(response, "status_code", 0))
        if not 200 <= status < 300 and status != 304:
            error = parse_gateway_error(status, _safe_json(response), getattr(response, "headers", None))
            raise error
        return response, status

    @staticmethod
    def _parse(model, response: Any, status: int, error_type, message: str):
        try:
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("response must be an object")
            return model.model_validate(payload)
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise error_type(message, code="INVALID_RESPONSE", status=status) from exc


def _safe_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception:
        return None
