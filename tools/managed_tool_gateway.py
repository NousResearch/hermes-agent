"""Generic managed-tool gateway helpers for Nous-hosted vendor passthroughs."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

from hermes_constants import get_hermes_home
from tools.tool_backend_helpers import managed_nous_tools_enabled

_DEFAULT_TOOL_GATEWAY_DOMAIN = "nousresearch.com"
_DEFAULT_TOOL_GATEWAY_SCHEME = "https"
_NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120


@dataclass(frozen=True)
class ManagedToolGatewayConfig:
    vendor: str
    gateway_origin: str
    nous_user_token: str
    managed_mode: bool


def auth_json_path():
    """Return the Hermes auth store path, respecting HERMES_HOME overrides."""
    return get_hermes_home() / "auth.json"


def _read_nous_provider_state() -> Optional[dict]:
    try:
        path = auth_json_path()
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        providers = data.get("providers", {})
        if not isinstance(providers, dict):
            return None
        nous_provider = providers.get("nous", {})
        if isinstance(nous_provider, dict):
            return nous_provider
    except Exception:
        pass
    return None


def _parse_timestamp(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _access_token_is_expiring(expires_at: object, skew_seconds: int) -> bool:
    expires = _parse_timestamp(expires_at)
    if expires is None:
        return True
    remaining = (expires - datetime.now(timezone.utc)).total_seconds()
    return remaining <= max(0, int(skew_seconds))


def _read_user_token_override() -> Optional[str]:
    """Read the TOOL_GATEWAY_USER_TOKEN env override through the secret scope.

    Availability scans run both inside agent turns (scope installed) and in
    unscoped CLI paths, so this uses the Slack pattern: honor the scope's
    verdict when installed (a scoped miss does NOT borrow the process env
    under multiplex), fall back to ``os.environ`` only when unscoped.
    """
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            explicit = get_secret("TOOL_GATEWAY_USER_TOKEN")
        except UnscopedSecretError:
            explicit = os.getenv("TOOL_GATEWAY_USER_TOKEN")
    except Exception:
        explicit = os.getenv("TOOL_GATEWAY_USER_TOKEN")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return None


def peek_nous_access_token() -> Optional[str]:
    """Cheap probe for a Nous gateway token without triggering refresh.

    Availability scans (`hermes tools`, banner/status paint, provider
    `is_available()` checks) must stay off the synchronous OAuth refresh path.
    This helper therefore only inspects the explicit env override and the
    cached auth-store token, without checking expiry and without making any
    network calls. Truthful refresh handling stays in request/session paths
    that call :func:`read_nous_access_token`.
    """
    explicit = _read_user_token_override()
    if explicit:
        return explicit

    nous_provider = _read_nous_provider_state() or {}
    access_token = nous_provider.get("access_token")
    if isinstance(access_token, str) and access_token.strip():
        return access_token.strip()
    return None


def read_nous_access_token() -> Optional[str]:
    """Read a Nous Subscriber OAuth access token from auth store or env override."""
    explicit = _read_user_token_override()
    if explicit:
        return explicit
    nous_provider = _read_nous_provider_state() or {}
    cached_token = peek_nous_access_token()

    if cached_token and not _access_token_is_expiring(
        nous_provider.get("expires_at"),
        _NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    ):
        return cached_token

    try:
        from hermes_cli.auth import resolve_nous_access_token

        refreshed_token = resolve_nous_access_token(
            refresh_skew_seconds=_NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
        )
        if isinstance(refreshed_token, str) and refreshed_token.strip():
            return refreshed_token.strip()
    except Exception as exc:
        logger.debug("Nous access token refresh failed: %s", exc)

    return cached_token


def get_tool_gateway_scheme() -> str:
    """Return configured shared gateway URL scheme."""
    scheme = os.getenv("TOOL_GATEWAY_SCHEME", "").strip().lower()
    if not scheme:
        return _DEFAULT_TOOL_GATEWAY_SCHEME

    if scheme in {"http", "https"}:
        return scheme

    raise ValueError("TOOL_GATEWAY_SCHEME must be 'http' or 'https'")


def build_vendor_gateway_url(vendor: str) -> str:
    """Return the gateway origin for a specific vendor."""
    vendor_key = f"{vendor.upper().replace('-', '_')}_GATEWAY_URL"
    explicit_vendor_url = os.getenv(vendor_key, "").strip().rstrip("/")
    if explicit_vendor_url:
        return explicit_vendor_url

    shared_scheme = get_tool_gateway_scheme()
    shared_domain = os.getenv("TOOL_GATEWAY_DOMAIN", "").strip().strip("/")
    if shared_domain:
        return f"{shared_scheme}://{vendor}-gateway.{shared_domain}"

    return f"{shared_scheme}://{vendor}-gateway.{_DEFAULT_TOOL_GATEWAY_DOMAIN}"


def resolve_managed_tool_gateway(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader: Optional[Callable[[], Optional[str]]] = None,
) -> Optional[ManagedToolGatewayConfig]:
    """Resolve shared managed-tool gateway config for a vendor."""
    if not managed_nous_tools_enabled():
        return None

    resolved_gateway_builder = gateway_builder or build_vendor_gateway_url
    resolved_token_reader = token_reader or read_nous_access_token

    gateway_origin = resolved_gateway_builder(vendor)
    nous_user_token = resolved_token_reader()
    if not gateway_origin or not nous_user_token:
        return None

    return ManagedToolGatewayConfig(
        vendor=vendor,
        gateway_origin=gateway_origin,
        nous_user_token=nous_user_token,
        managed_mode=True,
    )


def is_managed_tool_gateway_ready(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader: Optional[Callable[[], Optional[str]]] = None,
) -> bool:
    """Return True when gateway URL and a likely-usable Nous token are present.

    Defaults to :func:`peek_nous_access_token` so read-only availability scans
    avoid synchronous OAuth refresh. Callers that are about to make a real
    gateway request should use :func:`resolve_managed_tool_gateway` (which
    still defaults to the refresh-aware :func:`read_nous_access_token`).
    """
    return resolve_managed_tool_gateway(
        vendor,
        gateway_builder=gateway_builder,
        token_reader=token_reader or peek_nous_access_token,
    ) is not None


# ---------------------------------------------------------------------------
# Managed vendor endpoints
# ---------------------------------------------------------------------------
#
# Vendors the gateway serves on its own origin (rather than on a
# `{vendor}-gateway` host) are pinned HERE, in code, the same way every other
# managed vendor's gateway URL is pinned: adding one is a Hermes release, and
# the exact URL a user's agent may connect to is reviewable in this file. A
# runtime discovery catalog was tried and deliberately removed — a remote
# endpoint that can add tools to every entitled install is a bigger trust
# surface than a code diff.
#
# The gateway exposes a Nous-owned REST contract per vendor; it names the
# vendor but not the vendor's own API, so nothing here needs to know the
# upstream's endpoint or field names.

# Pseudo-vendor used only to resolve the shared tool-gateway origin via
# build_vendor_gateway_url (honors TOOL_GATEWAY_URL / TOOL_GATEWAY_DOMAIN).
_MANAGED_GATEWAY_VENDOR = "tool"

def managed_vendor_base_path(vendor: str) -> str:
    """Base path for a managed vendor's REST routes on the gateway host."""
    return f"/api/{vendor}"


def managed_vendor_upload_path(vendor: str) -> str:
    """Media upload endpoint for a managed vendor, on the same host."""
    return f"/api/uploads/{vendor}"


def managed_vendor_endpoints(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
) -> Optional[dict]:
    """Absolute URLs for a managed vendor, or ``None`` when none resolves.

    Address resolution only: entitlement is deliberately not consulted here.
    What an account may spend on a managed vendor is the gateway's own
    decision, stated in its refusals, and re-deciding it on the client can only
    ever disagree with the server. A caller that wants to hide its tools from
    users who could not call them at all does that in its ``check_fn``.

    ``None`` means no origin could be resolved — a misconfigured
    ``TOOL_GATEWAY_SCHEME`` — so there is nothing to call.
    """
    builder = gateway_builder or build_vendor_gateway_url
    try:
        origin = builder(_MANAGED_GATEWAY_VENDOR).rstrip("/")
    except ValueError:
        return None
    if not origin:
        return None

    return {
        "origin": origin,
        "base_url": f"{origin}{managed_vendor_base_path(vendor)}",
        "upload_path": managed_vendor_upload_path(vendor),
    }


def is_managed_nous_gateway_url(
    url: object,
    gateway_builder: Optional[Callable[[str], str]] = None,
) -> bool:
    """True when ``url`` is on the Nous tool-gateway origin this client builds.

    Anything granting a URL extra trust — our bearer, reading files off disk to
    upload — must gate on this rather than on a name, so an arbitrary URL can
    never inherit that trust.
    """
    if not isinstance(url, str) or not url.strip():
        return False

    builder = gateway_builder or build_vendor_gateway_url
    try:
        expected = urlsplit(builder(_MANAGED_GATEWAY_VENDOR))
        actual = urlsplit(url.strip())
    except ValueError:
        return False

    return bool(actual.scheme) and (actual.scheme, actual.netloc) == (expected.scheme, expected.netloc)


def managed_gateway_auth_headers(
    url: object,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader: Optional[Callable[[], Optional[str]]] = None,
) -> dict:
    """Live auth headers for a managed gateway URL, or ``{}`` when not managed.

    Read fresh on every call rather than cached: a Nous access token expires
    within the hour, and a long session would otherwise keep presenting a dead
    bearer. Returns ``{}`` rather than raising when no token is available, so a
    caller can report "sign in" instead of sending an unauthenticated request.
    """
    if not is_managed_nous_gateway_url(url, gateway_builder):
        return {}

    resolved_token_reader = token_reader or read_nous_access_token
    try:
        token = resolved_token_reader()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Managed gateway token read failed for %s: %s", url, exc)
        return {}
    if not isinstance(token, str) or not token.strip():
        return {}

    return {"Authorization": f"Bearer {token.strip()}"}


# ---------------------------------------------------------------------------
# Managed media uploads
# ---------------------------------------------------------------------------
#
# Media arguments used to be inlined as base64, which capped a whole tool call
# at ~2MB of real bytes under the gateway's request ceiling and ruled out video
# entirely. Each pinned managed server carries an upload endpoint
# (`upload_path`); the bytes go straight to storage via a presigned URL, and
# the tool argument carries an opaque `nous-upload:<token>` reference instead.
#
# The protocol lives HERE rather than in a vendor tool module: the presign
# request shape, the response contract, and the `nous-upload:` scheme are Nous
# gateway specifics shared by every managed vendor that takes media.

_MEDIA_UPLOAD_PRESIGN_TIMEOUT_SECONDS = 15.0
# The PUT carries up to 50MB of video; a flat 60s would fail a legitimate
# clip on an ordinary residential uplink, so only the write phase is long.
_MEDIA_UPLOAD_PUT_READ_TIMEOUT_SECONDS = 60.0
_MEDIA_UPLOAD_PUT_WRITE_TIMEOUT_SECONDS = 300.0


def _describe_media_upload_refusal(response) -> str:
    """A model-actionable reason from a gateway refusal, or a generic one.

    The gateway's 4xx bodies carry deliberate guidance (rate-limit waits, size
    caps, "you could not submit anyway"), so surface `error.message` verbatim
    rather than a bare status code.
    """
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    except Exception:
        pass
    return f"the gateway refused the upload (HTTP {response.status_code})"


def build_managed_media_uploader(
    server_url: object,
    upload_path: object,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader: Optional[Callable[[], Optional[str]]] = None,
) -> Optional[Callable]:
    """Async ``(data, mime) -> argument value`` uploader for one managed vendor.

    Returns ``None`` when there is no usable upload endpoint (not a managed
    Nous URL, or no ``upload_path``); callers then refuse local paths with a
    clear message instead of silently forwarding them.

    The three steps of the protocol:

    1. POST ``origin + upload_path`` with the declared content type and exact
       byte length, using the same live auth headers as the vendor calls.
       The gateway answers with a presigned single-object PUT URL (short
       expiry; type and length are signed into it) and an upload token.
    2. PUT the bytes to that URL. This goes directly to storage — never
       through the gateway — which is what removes the request-size ceiling.
    3. Return ``nous-upload:<token>`` for the tool argument. The token is
       bound to this Nous principal and is redeemable only through the
       gateway, so it is inert anywhere else it might end up.
    """
    if not is_managed_nous_gateway_url(server_url, gateway_builder):
        return None
    if not isinstance(upload_path, str) or not upload_path.startswith("/"):
        return None

    parts = urlsplit(str(server_url).strip())
    origin = f"{parts.scheme}://{parts.netloc}"
    presign_url = f"{origin}{upload_path}"

    async def upload(data: bytes, mime: str) -> str:
        import httpx

        from tools.url_safety import create_ssrf_safe_async_client

        headers = managed_gateway_auth_headers(server_url, gateway_builder, token_reader)
        if not headers:
            raise RuntimeError("no Nous credential is available for the upload")

        # Two clients on purpose, split by whose address we are trusting.
        #
        # The presign POST goes to `presign_url`, which is entirely determined
        # by the managed gateway origin (already validated by
        # is_managed_nous_gateway_url) plus the pinned upload_path — the same
        # first-party host the vendor calls go to freely. SSRF-guarding it
        # protects against nothing and would reject a local gateway on
        # 127.0.0.1, so it uses a plain client. The PUT target, by contrast, is
        # a URL the gateway *returned*, so it keeps the SSRF-safe client as
        # defense in depth (real presigned URLs are public R2, which it allows).
        presign_timeout = httpx.Timeout(_MEDIA_UPLOAD_PRESIGN_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=presign_timeout) as client:
            presign = await client.post(
                presign_url,
                headers=headers,
                json={"contentType": mime, "contentLength": len(data)},
            )
        if presign.status_code != 200:
            raise RuntimeError(_describe_media_upload_refusal(presign))

        try:
            payload = presign.json()
        except Exception:
            payload = None
        upload_url = payload.get("uploadUrl") if isinstance(payload, dict) else None
        token = payload.get("token") if isinstance(payload, dict) else None
        if not (isinstance(upload_url, str) and upload_url and isinstance(token, str) and token):
            raise RuntimeError("the gateway's upload response was malformed")

        put_timeout = httpx.Timeout(
            _MEDIA_UPLOAD_PRESIGN_TIMEOUT_SECONDS,
            read=_MEDIA_UPLOAD_PUT_READ_TIMEOUT_SECONDS,
            write=_MEDIA_UPLOAD_PUT_WRITE_TIMEOUT_SECONDS,
        )
        async with create_ssrf_safe_async_client(timeout=put_timeout) as client:
            # The presigned URL signs the exact Content-Type and Content-Length,
            # so this PUT must send precisely what was declared above.
            put = await client.put(upload_url, content=data, headers={"Content-Type": mime})
        if put.status_code != 200:
            raise RuntimeError(f"storage refused the upload (HTTP {put.status_code})")

        return f"nous-upload:{token}"

    return upload

