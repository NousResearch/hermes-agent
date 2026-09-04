"""URL safety checks — blocks requests to private/internal network addresses (SSRF).

``security.allow_private_urls: true`` disables private-IP blocking (DNS that resolves public
names to private ranges); cloud metadata hostnames/IPs are **always** blocked. DNS rebinding
(TOCTOU) is closed for Hermes-owned httpx paths by ``create_ssrf_safe_[async_]client()``, which
re-apply the policy at TCP connect and dial the validated IP while preserving Host/SNI. Redirect
bypass is mitigated by response hooks re-validating each target (``redirect_target_from_response``).
"""

import ipaddress
import logging
import os
import socket
import asyncio
import re
from contextlib import contextmanager
from typing import Any, Optional
from urllib.parse import parse_qsl, quote, unquote, urljoin, urlparse, urlsplit, urlunsplit

from hermes_constants import get_hermes_home_override
from utils import is_truthy_value

logger = logging.getLogger(__name__)

# Proxy env vars: when set, the runtime should delegate DNS to the proxy.
_PROXY_ENV_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
_HTTP_SCHEMES = frozenset({"http", "https"})
_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _proxy_is_configured() -> bool:
    return any(os.environ.get(v) for v in _PROXY_ENV_VARS)


def normalize_url_for_request(url: str) -> str:
    """ASCII-safe HTTP URL for Hermes-owned URL tools (IRI -> URI, e.g. ``https://wttr.in/Köln``).
    Preserves URL syntax and existing percent escapes while IDNA-encoding the host and
    percent-encoding non-ASCII path/query/fragment text. URL tool inputs only — never shell commands."""
    if not isinstance(url, str):
        return url
    raw = url.strip()
    if not raw:
        return raw

    # Repair model-emitted whitespace between scheme separator and authority
    # (``https:// docs.example``); that position is never meaningful in HTTP URLs.
    raw = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)\s+", r"\1", raw)
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if parsed.scheme.lower() not in _HTTP_SCHEMES:
        return raw
    netloc, hostname = parsed.netloc, parsed.hostname
    if hostname:
        try:
            ascii_host = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            ascii_host = hostname
        if ascii_host != hostname:
            netloc = netloc.replace(hostname, ascii_host, 1)
    safe = "/%:@!$&'()*+,;="
    return urlunsplit((parsed.scheme, netloc, quote(parsed.path, safe=safe),
                       quote(parsed.query, safe=safe + "?"), quote(parsed.fragment, safe=safe + "?")))


# Unambiguously credential-bearing query param names. Deliberately narrow: bare
# English words that double as page facets (``code``, ``key``, ``auth``,
# ``session``, ``sig``) are EXCLUDED so ordinary browsing is not blocked.
_SENSITIVE_QUERY_PARAM_NAMES = frozenset({
    "access_token", "api_key", "apikey", "auth_token", "authorization", "awsaccesskeyid",
    "client_secret", "credential", "credentials", "jwt", "password", "passwd", "secret",
    "session_id", "signature", "token", "x_amz_security_token", "x_amz_signature",
    "x-amz-security-token", "x-amz-signature"})


def sensitive_query_param_name(url: str) -> Optional[str]:
    """First credential-named query parameter in ``url`` (with a value), if any. Checked before
    handing URLs to third-party fetch/browser backends: catches opaque magic links, OAuth codes,
    signed-URL signatures and custom ``?token=...`` values that prefix-based redaction misses."""
    if not isinstance(url, str) or "?" not in url:
        return None
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in _HTTP_SCHEMES or not parsed.query:
        return None
    return next((key for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                 if value and unquote(key).lower() in _SENSITIVE_QUERY_PARAM_NAMES), None)


# Cloud metadata hostnames — always blocked regardless of DNS or config toggle.
_BLOCKED_HOSTNAMES = frozenset({"metadata.google.internal", "metadata.goog"})

# Cloud metadata / credential endpoints (the #1 SSRF target) — always blocked, also in
# IPv4-mapped IPv6 form (resolvers may return ``::ffff:x.x.x.x``; ipaddress treats those as distinct).
_METADATA_V4 = (
    "169.254.169.254",  # AWS/GCP/Azure/DO/Oracle metadata
    "169.254.170.2",    # AWS ECS task metadata (task IAM creds)
    "169.254.169.253",  # Azure IMDS wire server
    "100.100.100.200",  # Alibaba Cloud metadata
)
_ALWAYS_BLOCKED_IPS = frozenset(
    {ipaddress.ip_address(ip) for ip in _METADATA_V4}
    | {ipaddress.ip_address("::ffff:" + ip) for ip in _METADATA_V4}
    | {ipaddress.ip_address("fd00:ec2::254")}  # AWS metadata (IPv6)
)
# Entire link-local range (no legit agent target), plus its IPv4-mapped form.
_ALWAYS_BLOCKED_NETWORKS = tuple(ipaddress.ip_network(n) for n in ("169.254.0.0/16", "::ffff:169.254.0.0/112"))

# Exact HTTPS hostnames allowed to resolve to private/benchmark-space IPs
# (QQ media legitimately resolves to 198.18.0.0/15 behind local proxy infra).
_TRUSTED_PRIVATE_IP_HOSTS = frozenset({"multimedia.nt.qq.com.cn"})
_MAX_SSRF_CONNECT_IPS = 8

# 100.64.0.0/10 (CGNAT, RFC 6598) is neither is_private nor is_global in
# ipaddress — must be blocked explicitly (Tailscale/WireGuard, cloud internal nets).
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# Global toggle cache (process lifetime; see _global_allow_private_urls).
_allow_private_resolved, _cached_allow_private = False, False


def _global_allow_private_urls() -> bool:
    """True when the user has opted out of private-IP blocking. Priority: ``HERMES_ALLOW_PRIVATE_URLS``
    env, ``security.allow_private_urls``, legacy ``browser.allow_private_urls``. Profile-scoped turns
    (``get_hermes_home_override()`` set) bypass the process-global cache — a multiplex gateway serves
    several profiles in one process; the first profile's opt-out must not disable blocking for later ones."""
    global _allow_private_resolved, _cached_allow_private
    if get_hermes_home_override() is not None:
        return _resolve_allow_private_urls()
    if not _allow_private_resolved:
        _allow_private_resolved, _cached_allow_private = True, _resolve_allow_private_urls()
    return _cached_allow_private


def _resolve_allow_private_urls() -> bool:
    """Resolve the effective private-URL toggle from the active config scope."""
    env_val = os.getenv("HERMES_ALLOW_PRIVATE_URLS", "").strip().lower()
    if env_val in {"true", "1", "yes"}:
        return True
    if env_val in {"false", "0", "no"}:
        return False  # explicit false does not fall through to config
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        for section in ("security", "browser"):  # preferred, then legacy
            block = cfg.get(section, {})
            if isinstance(block, dict) and is_truthy_value(block.get("allow_private_urls"), default=False):
                return True
    except Exception:
        pass  # config unavailable (tests, early import) — keep default
    return False


def _reset_allow_private_cache() -> None:
    """Reset the cached toggle — only for tests."""
    global _allow_private_resolved, _cached_allow_private
    _allow_private_resolved = _cached_allow_private = False


def _normalize_hostname(host: Optional[str]) -> str:
    return (host or "").strip().lower().rstrip(".")


def _parse_ip(hostname: str) -> Optional[_IPAddress]:
    """IP object for a literal-IP hostname, else None."""
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        return None


def _iter_resolved_ips(addr_info: Any):
    """Yield ``(raw, ip_str, ip)`` per getaddrinfo answer. ``ip_str`` has any IPv6 scope ID
    (``%eth0``) stripped; ``ip`` is None when still unparseable — each caller decides skip/fail/raise."""
    for _family, _, _, _, sockaddr in addr_info:
        raw = sockaddr[0]
        ip_str = raw.split("%")[0]
        yield raw, ip_str, _parse_ip(ip_str)


def _getaddrinfo(hostname: str, port: Optional[int] = None):
    return socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)


def _is_always_blocked_ip(ip: _IPAddress) -> bool:
    return ip in _ALWAYS_BLOCKED_IPS or any(ip in net for net in _ALWAYS_BLOCKED_NETWORKS)


def _is_blocked_ip(ip: _IPAddress) -> bool:
    """Return True if the IP should be blocked for SSRF protection."""
    # IPv4-mapped IPv6 (``::ffff:x.x.x.x``) is classified by its embedded IPv4.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified or ip in _CGNAT_NETWORK)


def is_always_blocked_url(url: str) -> bool:
    """True when the URL targets the always-blocked floor (cloud metadata) — only the sentinel
    hostnames/IPs, regardless of backend, routing, or ``allow_private_urls``. For callers that
    deliberately bypass the full check (e.g. hybrid cloud browser routing private URLs to a local
    sidecar) but must still enforce the floor. False for ordinary private/loopback URLs, DNS
    failures and parse errors (the caller's ordinary fail-closed path handles those)."""
    try:
        hostname = _normalize_hostname(urlparse(url).hostname)
        if not hostname:
            return False
        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning("Blocked request to internal hostname (always-blocked floor): %s", hostname)
            return True
        ip = _parse_ip(hostname)
        if ip is not None:
            if _is_always_blocked_ip(ip):
                logger.warning("Blocked request to cloud metadata address (always-blocked floor): %s", hostname)
                return True
            return False
        try:
            addr_info = _getaddrinfo(hostname)
        except socket.gaierror:
            return False  # DNS failure is not part of the floor; caller's path handles it
        for raw, ip_str, resolved in _iter_resolved_ips(addr_info):
            if resolved is None:
                logger.warning("Unparseable IP address %r for hostname %s — skipping address", raw, hostname)
                continue
            if _is_always_blocked_ip(resolved):
                logger.warning(
                    "Blocked request to cloud metadata address (always-blocked floor): %s -> %s", hostname, ip_str
                )
                return True
        return False
    except Exception as exc:
        # Parse/unexpected errors are not "always blocked"; caller decides fail-open/closed.
        logger.debug("is_always_blocked_url error for %s: %s", url, exc)
        return False


def _allows_private_ip_resolution(hostname: str, scheme: str) -> bool:
    """Return True when a trusted HTTPS hostname may bypass IP-class blocking."""
    return scheme == "https" and hostname in _TRUSTED_PRIVATE_IP_HOSTS


def _resolved_ip_block_reason(ip: _IPAddress, allow_private: bool) -> Optional[str]:
    """Why a resolved answer must be rejected, or None if it may be dialed. The metadata floor
    ignores ``allow_private``; ordinary private/internal classes are blocked only when it is False."""
    if _is_always_blocked_ip(ip):
        return "cloud metadata address"
    if not allow_private and _is_blocked_ip(ip):
        return "private/internal address"
    return None


def is_safe_url(url: str) -> bool:
    """True if the URL target is not a private/internal address. Resolves the hostname and checks
    every answer; fails closed on DNS errors and unexpected exceptions. ``allow_private_urls``
    skips private-IP blocking, but cloud metadata endpoints remain blocked regardless."""
    try:
        parsed = urlparse(url)
        hostname = _normalize_hostname(parsed.hostname)
        scheme = (parsed.scheme or "").strip().lower()
        if scheme not in _HTTP_SCHEMES:
            logger.warning("Blocked request — unsupported URL scheme: %s", scheme or "<empty>")
            return False
        if not hostname:
            return False

        # Metadata hostnames are blocked BEFORE consulting the toggle.
        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning("Blocked request to internal hostname: %s", hostname)
            return False
        allow_all_private = _global_allow_private_urls()
        allow_private_ip = _allows_private_ip_resolution(hostname, scheme)
        allow_private = allow_all_private or allow_private_ip
        try:
            addr_info = _getaddrinfo(hostname)
        except socket.gaierror:
            # Sandbox/proxy environments may block direct DNS; when a proxy is
            # configured, delegate resolution to it (metadata hostnames were already
            # rejected above). Literal IPs need no DNS, so a failure on one is not a
            # proxy symptom — keep them fail-closed.
            if _parse_ip(hostname) is None and _proxy_is_configured():
                logger.debug(
                    "DNS resolution failed for %s — proxy configured, allowing through for proxy-side resolution",
                    hostname)
                return True
            logger.warning("Blocked request — DNS resolution failed for: %s", hostname)
            return False
        for raw, ip_str, ip in _iter_resolved_ips(addr_info):
            if ip is None:
                logger.warning("Blocked request — unparseable IP address %r for hostname %s", raw, hostname)
                return False
            reason = _resolved_ip_block_reason(ip, allow_private)
            if reason is not None:
                logger.warning("Blocked request to %s: %s -> %s", reason, hostname, ip_str)
                return False
        if allow_all_private:
            logger.debug("Allowing private/internal resolution (security.allow_private_urls=true): %s", hostname)
        elif allow_private_ip:
            logger.debug("Allowing trusted hostname despite private/internal resolution: %s", hostname)
        return True
    except Exception as exc:
        # Fail closed: parsing edge cases must not become SSRF bypass vectors.
        logger.warning("Blocked request — URL safety check error for %s: %s", url, exc)
        return False


async def async_is_safe_url(url: str) -> bool:
    """:func:`is_safe_url` with the blocking DNS work off the event loop."""
    return await asyncio.to_thread(is_safe_url, url)


class SSRFConnectionBlocked(ValueError):
    """Raised when connect-time DNS resolution violates the URL safety policy."""


def _resolved_http_connect_ips(host: str, port: int, scheme: str) -> list[str]:
    """Resolve and validate *host* at TCP-connect time; return dialable IP strings. Closes the
    DNS-rebinding gap between pre-flight validation and connect for direct httpx clients. The
    result is capped at ``_MAX_SSRF_CONNECT_IPS``, but EVERY answer is validated."""
    hostname = _normalize_hostname(host)
    if not hostname:
        raise SSRFConnectionBlocked("Blocked request with empty hostname")
    if hostname in _BLOCKED_HOSTNAMES:
        raise SSRFConnectionBlocked(f"Blocked request to internal hostname: {hostname}")
    allow_private = _global_allow_private_urls() or _allows_private_ip_resolution(hostname, scheme)
    try:
        addr_info = _getaddrinfo(hostname, port)
    except socket.gaierror as exc:
        raise SSRFConnectionBlocked(f"Blocked request - DNS resolution failed for: {hostname}") from exc
    safe_ips: list[str] = []
    for raw, ip_str, ip in _iter_resolved_ips(addr_info):
        if ip is None:
            raise SSRFConnectionBlocked(
                f"Blocked request - unparseable IP address {raw!r} for hostname {hostname}"
            ) from ValueError(f"{ip_str!r} does not appear to be an IPv4 or IPv6 address")
        reason = _resolved_ip_block_reason(ip, allow_private)
        if reason is not None:
            raise SSRFConnectionBlocked(f"Blocked request to {reason} during connect: {hostname} -> {ip_str}")
        if ip_str not in safe_ips and len(safe_ips) < _MAX_SSRF_CONNECT_IPS:
            safe_ips.append(ip_str)
    if not safe_ips:
        raise SSRFConnectionBlocked(f"Blocked request - DNS returned no results for: {hostname}")
    return safe_ips


class _SSRFGuardedBackendBase:
    """httpcore backend that re-resolves + validates at connect time and dials a vetted IP.
    Host/SNI stay on the original hostname; Unix sockets are refused outright. Candidate IPs are
    tried in order and the last connect error is re-raised so callers see the real failure."""

    def __init__(self, backend: Any, schemes_by_origin_var: Any):
        self._backend = backend
        self._schemes_by_origin_var = schemes_by_origin_var

    def _connect_ips(self, host: str, port: int) -> list[str]:
        scheme = self._schemes_by_origin_var.get({}).get((host, port)) or ("https" if port == 443 else "http")
        return _resolved_http_connect_ips(host, port, scheme)

    @staticmethod
    def _connect_errors() -> tuple:
        import httpcore
        return (httpcore.ConnectError, httpcore.ConnectTimeout)

    @staticmethod
    def _no_usable_ips(host: str, last_exc: Exception | None) -> Exception:
        if last_exc is not None:
            return last_exc
        return SSRFConnectionBlocked(f"Blocked request - DNS returned no usable IPs for: {host}")

    def connect_unix_socket(self, path: str, timeout: float | None = None, socket_options: Any = None) -> Any:
        raise SSRFConnectionBlocked("Blocked Unix socket connection in SSRF-safe transport")


class _SSRFGuardedAsyncNetworkBackend(_SSRFGuardedBackendBase):
    def __init__(self, schemes_by_origin_var: Any):
        from httpcore._backends.auto import AutoBackend
        super().__init__(AutoBackend(), schemes_by_origin_var)

    async def connect_tcp(self, host: str, port: int, timeout: float | None = None,
                          local_address: str | None = None, socket_options: Any = None) -> Any:
        last_exc: Exception | None = None
        for ip in await asyncio.to_thread(self._connect_ips, host, port):
            try:
                return await self._backend.connect_tcp(
                    ip, port, timeout=timeout, local_address=local_address, socket_options=socket_options)
            except self._connect_errors() as exc:
                last_exc = exc
        raise self._no_usable_ips(host, last_exc)

    async def connect_unix_socket(self, path: str, timeout: float | None = None, socket_options: Any = None) -> Any:
        raise SSRFConnectionBlocked("Blocked Unix socket connection in SSRF-safe transport")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _SSRFGuardedNetworkBackend(_SSRFGuardedBackendBase):
    def __init__(self, schemes_by_origin_var: Any):
        from httpcore._backends.sync import SyncBackend
        super().__init__(SyncBackend(), schemes_by_origin_var)

    def connect_tcp(self, host: str, port: int, timeout: float | None = None,
                    local_address: str | None = None, socket_options: Any = None) -> Any:
        last_exc: Exception | None = None
        for ip in self._connect_ips(host, port):
            try:
                return self._backend.connect_tcp(
                    ip, port, timeout=timeout, local_address=local_address, socket_options=socket_options)
            except self._connect_errors() as exc:
                last_exc = exc
        raise self._no_usable_ips(host, last_exc)

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(seconds)


def _origin_scheme_context(request: Any) -> dict[tuple[str, int], str]:
    host, port, scheme = request.url.host, request.url.port, request.url.scheme
    return {(host, port): scheme} if host and port is not None and scheme in _HTTP_SCHEMES else {}


@contextmanager
def _origin_scope(schemes_by_origin_var: Any, request: Any):
    """Expose the request's origin scheme to the connect-time backend for this request only."""
    token = schemes_by_origin_var.set(_origin_scheme_context(request))
    try:
        yield
    finally:
        schemes_by_origin_var.reset(token)


def _install_ssrf_guard_on_transport(transport: Any, schemes_by_origin_var: Any, *, is_async: bool = False) -> None:
    """Swap the transport's pool network backend for the SSRF-guarded one (idempotent). Only the
    direct transport is guarded; proxy mounts delegate final-target resolution to the trusted proxy."""
    state = getattr(transport, "__dict__", {}) if transport is not None else {}
    if transport is None or state.get("_hermes_ssrf_guarded", False):
        return
    label = "async httpx transport" if is_async else "httpx transport"
    pool = state.get("_pool")
    if pool is None or not hasattr(pool, "_network_backend"):
        raise SSRFConnectionBlocked(f"Unsupported {label} cannot be made SSRF-safe")
    backend_cls = _SSRFGuardedAsyncNetworkBackend if is_async else _SSRFGuardedNetworkBackend
    pool._network_backend = backend_cls(schemes_by_origin_var)
    method_name = "handle_async_request" if is_async else "handle_request"
    handle = getattr(transport, method_name, None)
    if handle is None:
        raise SSRFConnectionBlocked(f"Unsupported {label} cannot be made SSRF-safe")

    async def guarded_async(request: Any) -> Any:
        with _origin_scope(schemes_by_origin_var, request):
            return await handle(request)

    def guarded_sync(request: Any) -> Any:
        with _origin_scope(schemes_by_origin_var, request):
            return handle(request)
    setattr(transport, method_name, guarded_async if is_async else guarded_sync)
    transport._hermes_ssrf_guarded = True


def _install_ssrf_guard_on_client(client: Any, *, is_async: bool = False) -> None:
    """Guard ``client._transport`` only; ``_mounts`` (env/explicit proxies) stay untouched."""
    import contextvars
    var_name = "hermes_ssrf_async_origin_schemes" if is_async else "hermes_ssrf_origin_schemes"
    _install_ssrf_guard_on_transport(
        getattr(client, "__dict__", {}).get("_transport"), contextvars.ContextVar(var_name), is_async=is_async)


def create_ssrf_safe_async_client(**kwargs: Any) -> Any:
    """``httpx.AsyncClient`` with connect-time SSRF validation: direct HTTP(S) connections are
    resolved, validated, and dialed by IP while the hostname is preserved for Host, SNI, and
    certificate verification. Proxied requests delegate resolution to the proxy."""
    import httpx
    client = httpx.AsyncClient(**kwargs)
    _install_ssrf_guard_on_client(client, is_async=True)
    return client


def create_ssrf_safe_client(**kwargs: Any) -> Any:
    """Create an ``httpx.Client`` with connect-time SSRF validation."""
    import httpx
    client = httpx.Client(**kwargs)
    _install_ssrf_guard_on_client(client)
    return client


def redirect_target_from_response(response: Any) -> Optional[str]:
    """Redirect target visible from inside an httpx response hook. ``response.next_request`` is
    frequently ``None`` inside hooks (populated later by the follower), which would make an SSRF
    redirect guard silently never fire — so resolve from ``Location`` first, then ``next_request``."""
    if not getattr(response, "is_redirect", False):
        return None
    location = (getattr(response, "headers", {}) or {}).get("location")
    if location:
        return urljoin(str(getattr(response, "url", "")), str(location))
    next_request = getattr(response, "next_request", None)
    return str(next_request.url) if next_request else None


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def has_sensitive_query_params(url: str) -> bool:
    """Return True when ``url`` carries likely credential-bearing query params."""
    return sensitive_query_param_name(url) is not None

def ssrf_safe_async_http_transport(**kwargs: Any) -> Any:
    """Return an httpx async transport that pins direct TCP connects to vetted IPs."""
    import contextvars
    import httpx

    schemes_by_origin_var = contextvars.ContextVar("hermes_ssrf_async_origin_schemes")

    class _Transport(httpx.AsyncHTTPTransport):
        def __init__(self, **transport_kwargs: Any):
            super().__init__(**transport_kwargs)
            self._pool._network_backend = _SSRFGuardedAsyncNetworkBackend(  # type: ignore[attr-defined]
                schemes_by_origin_var
            )

        async def handle_async_request(self, request: Any) -> Any:
            token = schemes_by_origin_var.set(_origin_scheme_context(request))
            try:
                return await super().handle_async_request(request)
            finally:
                schemes_by_origin_var.reset(token)

    return _Transport(**kwargs)

def ssrf_safe_http_transport(**kwargs: Any) -> Any:
    """Return an httpx sync transport that pins direct TCP connects to vetted IPs."""
    import contextvars
    import httpx

    schemes_by_origin_var = contextvars.ContextVar("hermes_ssrf_origin_schemes")

    class _Transport(httpx.HTTPTransport):
        def __init__(self, **transport_kwargs: Any):
            super().__init__(**transport_kwargs)
            self._pool._network_backend = _SSRFGuardedNetworkBackend(  # type: ignore[attr-defined]
                schemes_by_origin_var
            )

        def handle_request(self, request: Any) -> Any:
            token = schemes_by_origin_var.set(_origin_scheme_context(request))
            try:
                return super().handle_request(request)
            finally:
                schemes_by_origin_var.reset(token)

    return _Transport(**kwargs)
# ---- END PLUGIN-COMPAT ----
