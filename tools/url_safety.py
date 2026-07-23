"""URL safety checks — blocks requests to private/internal network addresses.

Prevents SSRF (Server-Side Request Forgery) where a malicious prompt or
skill could trick the agent into fetching internal resources like cloud
metadata endpoints (169.254.169.254), localhost services, or private
network hosts.

The check can be globally disabled via ``security.allow_private_urls: true``
in config.yaml for environments where DNS resolves external domains to
private/benchmark-range IPs (OpenWrt routers, corporate proxies, VPNs
that use 198.18.0.0/15 or 100.64.0.0/10).  Even when disabled, cloud
metadata hostnames (metadata.google.internal, 169.254.169.254) are
**always** blocked — those are never legitimate agent targets.

Limitations:
  - DNS rebinding (TOCTOU): an attacker-controlled DNS server with TTL=0
    can return a public IP for the check, then a private IP for the actual
    connection. Hermes-owned direct httpx request paths should use
    ``create_ssrf_safe_client()`` / ``create_ssrf_safe_async_client()`` so the
    same policy is applied immediately before TCP connect and the client
    connects to the validated IP while preserving Host/SNI semantics.
  - Redirect-based bypass is mitigated by httpx event hooks that re-validate
    each redirect target in vision_tools, gateway platform adapters, and
    media cache helpers. Web tools use third-party SDKs (Firecrawl/Tavily)
    where redirect handling is on their servers.
"""

import ipaddress
import logging
import os
import socket
import asyncio
import re
from typing import Any, Optional
from urllib.parse import parse_qsl, quote, unquote, urljoin, urlparse, urlsplit, urlunsplit

from utils import is_truthy_value

logger = logging.getLogger(__name__)


def normalize_url_for_request(url: str) -> str:
    """Return an ASCII-safe HTTP URL for Hermes-owned URL tools.

    Browsers and HTTP clients expect URIs, but users and models often provide
    IRIs such as ``https://wttr.in/Köln``.  Preserve URL syntax and existing
    percent escapes while encoding non-ASCII host/path/query/fragment text.
    This is intentionally for URL tool inputs only; arbitrary shell commands
    must not be rewritten.
    """
    if not isinstance(url, str):
        return url

    raw = url.strip()
    if not raw:
        return raw

    # Models sometimes emit otherwise valid URLs with whitespace between the
    # scheme separator and authority (``https:// docs.example``). That position
    # is never meaningful in HTTP(S) URLs, and repairing it before parsing keeps
    # web tools from failing on a formatting artifact while leaving path/query
    # whitespace to the normal percent-encoding path below.
    raw = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)\s+", r"\1", raw)

    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw

    if parsed.scheme.lower() not in {"http", "https"}:
        return raw

    netloc = parsed.netloc
    hostname = parsed.hostname
    if hostname:
        try:
            ascii_host = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            ascii_host = hostname
        if ascii_host != hostname:
            netloc = netloc.replace(hostname, ascii_host, 1)

    path = quote(parsed.path, safe="/%:@!$&'()*+,;=")
    query = quote(parsed.query, safe="/%:@!$&'()*+,;=?")
    fragment = quote(parsed.fragment, safe="/%:@!$&'()*+,;=?")

    return urlunsplit((parsed.scheme, netloc, path, query, fragment))


# Query parameter names that are unambiguously credential-bearing. Kept
# deliberately narrow: bare English words that double as normal page facets
# (``code`` on promo/challenge pages, ``key``/``auth``/``session``/``sig`` as
# search or routing params) are intentionally EXCLUDED to avoid blocking
# ordinary browsing. Prefix-based token redaction (``is_safe_url``) still
# catches recognizable vendor key shapes; this set is the belt-and-suspenders
# for opaque secrets that carry an explicit credential-named parameter.
_SENSITIVE_QUERY_PARAM_NAMES = frozenset({
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "awsaccesskeyid",
    "client_secret",
    "credential",
    "credentials",
    "jwt",
    "password",
    "passwd",
    "secret",
    "session_id",
    "signature",
    "token",
    "x_amz_security_token",
    "x_amz_signature",
    "x-amz-security-token",
    "x-amz-signature",
})


def sensitive_query_param_name(url: str) -> Optional[str]:
    """Return the first sensitive query parameter name in ``url``, if any.

    Used before handing URLs to third-party fetch/browser backends. Prefix-based
    token redaction catches known credential shapes; this catches opaque magic
    links, OAuth codes, signed URL signatures, and custom ``?token=...`` values
    that do not have a recognizable vendor prefix.
    """
    if not isinstance(url, str) or "?" not in url:
        return None
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.query:
        return None
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if value and unquote(key).lower() in _SENSITIVE_QUERY_PARAM_NAMES:
            return key
    return None


def has_sensitive_query_params(url: str) -> bool:
    """Return True when ``url`` carries likely credential-bearing query params."""
    return sensitive_query_param_name(url) is not None

# Hostnames that should always be blocked regardless of IP resolution
# or any config toggle.  These are cloud metadata endpoints that an
# attacker could use to steal instance credentials.
_BLOCKED_HOSTNAMES = frozenset({
    "metadata.google.internal",
    "metadata.goog",
})

# IPs and networks that should always be blocked regardless of the
# allow_private_urls toggle.  These are cloud metadata / credential
# endpoints — the #1 SSRF target — and the link-local range where
# they all live.
#
# IPv4-mapped IPv6 variants are included because DNS resolvers may
# return ``::ffff:x.x.x.x`` for IPv4-only hosts, and Python's
# ipaddress module treats these as distinct from the plain IPv4
# address (they won't match ``ip in frozenset`` or ``ip in network``).
_ALWAYS_BLOCKED_IPS = frozenset({
    ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure/DO/Oracle metadata
    ipaddress.ip_address("169.254.170.2"),     # AWS ECS task metadata (task IAM creds)
    ipaddress.ip_address("169.254.169.253"),   # Azure IMDS wire server
    ipaddress.ip_address("fd00:ec2::254"),     # AWS metadata (IPv6)
    ipaddress.ip_address("100.100.100.200"),   # Alibaba Cloud metadata
    # IPv4-mapped IPv6 variants — same endpoints reachable via ::ffff:x.x.x.x
    ipaddress.ip_address("::ffff:169.254.169.254"),
    ipaddress.ip_address("::ffff:169.254.170.2"),
    ipaddress.ip_address("::ffff:169.254.169.253"),
    ipaddress.ip_address("::ffff:100.100.100.200"),
})
_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),    # Entire link-local range (no legit agent target)
    ipaddress.ip_network("::ffff:169.254.0.0/112"), # IPv4-mapped link-local range
)

# Exact HTTPS hostnames allowed to resolve to private/benchmark-space IPs.
# This is intentionally narrow: QQ media downloads can legitimately resolve
# to 198.18.0.0/15 behind local proxy/benchmark infrastructure.
_TRUSTED_PRIVATE_IP_HOSTS = frozenset({
    "multimedia.nt.qq.com.cn",
})

_MAX_SSRF_CONNECT_IPS = 8

# 100.64.0.0/10 (CGNAT / Shared Address Space, RFC 6598) is NOT covered by
# ipaddress.is_private — it returns False for both is_private and is_global.
# Must be blocked explicitly. Used by carrier-grade NAT, Tailscale/WireGuard
# VPNs, and some cloud internal networks.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# ---------------------------------------------------------------------------
# Global toggle: allow private/internal IP resolution
# ---------------------------------------------------------------------------
# Cached after first read so we don't hit the filesystem on every URL check.
_allow_private_resolved = False
_cached_allow_private: bool = False


def _global_allow_private_urls() -> bool:
    """Return True when the user has opted out of private-IP blocking.

    Checks (in priority order):
    1. ``HERMES_ALLOW_PRIVATE_URLS`` env var  (``true``/``1``/``yes``)
    2. ``security.allow_private_urls`` in config.yaml
    3. ``browser.allow_private_urls`` in config.yaml  (legacy / backward compat)

    Result is cached for the process lifetime.
    """
    global _allow_private_resolved, _cached_allow_private
    if _allow_private_resolved:
        return _cached_allow_private

    _allow_private_resolved = True
    _cached_allow_private = False  # safe default

    # 1. Env var override (highest priority)
    env_val = os.getenv("HERMES_ALLOW_PRIVATE_URLS", "").strip().lower()
    if env_val in {"true", "1", "yes"}:
        _cached_allow_private = True
        return _cached_allow_private
    if env_val in {"false", "0", "no"}:
        # Explicit false — don't fall through to config
        return _cached_allow_private

    # 2. Config file
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        # security.allow_private_urls (preferred)
        sec = cfg.get("security", {})
        if isinstance(sec, dict) and is_truthy_value(
            sec.get("allow_private_urls"), default=False
        ):
            _cached_allow_private = True
            return _cached_allow_private
        # browser.allow_private_urls (legacy fallback)
        browser = cfg.get("browser", {})
        if isinstance(browser, dict) and is_truthy_value(
            browser.get("allow_private_urls"), default=False
        ):
            _cached_allow_private = True
            return _cached_allow_private
    except Exception:
        # Config unavailable (e.g. tests, early import) — keep default
        pass

    return _cached_allow_private


def _reset_allow_private_cache() -> None:
    """Reset the cached toggle — only for tests."""
    global _allow_private_resolved, _cached_allow_private
    _allow_private_resolved = False
    _cached_allow_private = False


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP should be blocked for SSRF protection."""
    # IPv4-mapped IPv6 addresses (``::ffff:x.x.x.x``) should be checked
    # by their embedded IPv4 address, not as IPv6
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        embedded_ip = ip.ipv4_mapped
        return (embedded_ip.is_private or embedded_ip.is_loopback or
                embedded_ip.is_link_local or embedded_ip.is_reserved or
                embedded_ip.is_multicast or embedded_ip.is_unspecified or
                embedded_ip in _CGNAT_NETWORK)

    # Standard IPv4/IPv6 address checking
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    if ip.is_multicast or ip.is_unspecified:
        return True
    # CGNAT range not covered by is_private
    if ip in _CGNAT_NETWORK:
        return True
    return False


def is_always_blocked_url(url: str) -> bool:
    """Return True when the URL targets an always-blocked endpoint.

    This is the security floor — cloud metadata IPs / hostnames
    (169.254.169.254, metadata.google.internal, ECS task metadata, etc.)
    that have no legitimate agent use regardless of backend, routing, or
    the ``allow_private_urls`` toggle.  Used by callers that bypass the
    full ``is_safe_url`` check for their own reasons (e.g. hybrid cloud
    browser routing to a local Chromium sidecar for private URLs) and
    still need to enforce the non-negotiable floor before letting the
    request proceed.

    Returns True (= blocked) on:
      - Hostnames in ``_BLOCKED_HOSTNAMES``
      - IPs / networks in ``_ALWAYS_BLOCKED_IPS`` / ``_ALWAYS_BLOCKED_NETWORKS``
      - URLs whose hostname resolves to any of the above

    Returns False (= not in the always-blocked floor) on:
      - Benign public / private / loopback URLs (whether or not they'd
        be blocked by the ordinary SSRF check)
      - DNS-resolution failures for non-sentinel hostnames (these are
        someone else's problem — the caller's ordinary fail-closed path
        will catch them if applicable)
      - Parse errors (caller decides fail-open vs fail-closed)

    Intentionally narrower than ``is_safe_url``: only blocks the sentinel
    set, not ordinary private addresses.  Callers that want the full
    SSRF check should still use ``is_safe_url``.
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False

        # Blocked-hostname check fires regardless of DNS resolution
        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning(
                "Blocked request to internal hostname (always-blocked floor): %s",
                hostname,
            )
            return True

        # Literal IP → check directly against the always-blocked set
        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            ip = None

        if ip is not None:
            if ip in _ALWAYS_BLOCKED_IPS or any(
                ip in net for net in _ALWAYS_BLOCKED_NETWORKS
            ):
                logger.warning(
                    "Blocked request to cloud metadata address "
                    "(always-blocked floor): %s",
                    hostname,
                )
                return True
            return False

        # Hostname → resolve and check every answer.  DNS failure is NOT
        # always-blocked (caller's ordinary path handles that).
        try:
            addr_info = socket.getaddrinfo(
                hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        except socket.gaierror:
            return False

        for _family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            if '%' in ip_str:
                ip_str = ip_str.split('%')[0]
            try:
                resolved = ipaddress.ip_address(ip_str)
            except ValueError:
                logger.warning("Unparseable IP address %r for hostname %s — skipping address", sockaddr[0], hostname)
                continue
            if resolved in _ALWAYS_BLOCKED_IPS or any(
                resolved in net for net in _ALWAYS_BLOCKED_NETWORKS
            ):
                logger.warning(
                    "Blocked request to cloud metadata address "
                    "(always-blocked floor): %s -> %s",
                    hostname,
                    ip_str,
                )
                return True

        return False

    except Exception as exc:
        # Parse failures or unexpected errors — don't claim the URL is
        # always-blocked.  Caller decides what to do with a malformed URL.
        logger.debug("is_always_blocked_url error for %s: %s", url, exc)
        return False


def _allows_private_ip_resolution(hostname: str, scheme: str) -> bool:
    """Return True when a trusted HTTPS hostname may bypass IP-class blocking."""
    return scheme == "https" and hostname in _TRUSTED_PRIVATE_IP_HOSTS


def is_safe_url(url: str) -> bool:
    """Return True if the URL target is not a private/internal address.

    Resolves the hostname to an IP and checks against private ranges.
    Fails closed: DNS errors and unexpected exceptions block the request.

    When ``security.allow_private_urls`` is enabled (or the env var
    ``HERMES_ALLOW_PRIVATE_URLS=true``), private-IP blocking is skipped.
    Cloud metadata endpoints (169.254.169.254, metadata.google.internal)
    remain blocked regardless — they are never legitimate agent targets.
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        scheme = (parsed.scheme or "").strip().lower()
        if scheme not in {"http", "https"}:
            logger.warning("Blocked request — unsupported URL scheme: %s", scheme or "<empty>")
            return False
        if not hostname:
            return False

        # Block known internal hostnames — ALWAYS, even with toggle on
        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning("Blocked request to internal hostname: %s", hostname)
            return False

        # Check the global toggle AFTER blocking metadata hostnames
        allow_all_private = _global_allow_private_urls()

        allow_private_ip = _allows_private_ip_resolution(hostname, scheme)

        # Try to resolve and check IP
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            # DNS resolution failed — fail closed. If DNS can't resolve it,
            # the HTTP client will also fail, so blocking loses nothing.
            logger.warning("Blocked request — DNS resolution failed for: %s", hostname)
            return False

        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            if '%' in ip_str:
                ip_str = ip_str.split('%')[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                # Still unparseable after scope ID strip — fail closed
                logger.warning("Blocked request — unparseable IP address %r for hostname %s", sockaddr[0], hostname)
                return False

            # Always block cloud metadata IPs and link-local, even with toggle on
            if ip in _ALWAYS_BLOCKED_IPS or any(ip in net for net in _ALWAYS_BLOCKED_NETWORKS):
                logger.warning(
                    "Blocked request to cloud metadata address: %s -> %s",
                    hostname, ip_str,
                )
                return False

            if not allow_all_private and not allow_private_ip and _is_blocked_ip(ip):
                logger.warning(
                    "Blocked request to private/internal address: %s -> %s",
                    hostname, ip_str,
                )
                return False

        if allow_all_private:
            logger.debug(
                "Allowing private/internal resolution (security.allow_private_urls=true): %s",
                hostname,
            )
        elif allow_private_ip:
            logger.debug(
                "Allowing trusted hostname despite private/internal resolution: %s",
                hostname,
            )

        return True

    except Exception as exc:
        # Fail closed on unexpected errors — don't let parsing edge cases
        # become SSRF bypass vectors
        logger.warning("Blocked request — URL safety check error for %s: %s", url, exc)
        return False


async def async_is_safe_url(url: str) -> bool:
    """Same rules as :func:`is_safe_url`, but run the DNS work off the event loop.

    ``socket.getaddrinfo`` can block; call this from async code paths (gateway,
    ``web_extract_tool``, vision download hooks) instead of ``is_safe_url``.
    """
    return await asyncio.to_thread(is_safe_url, url)


class SSRFConnectionBlocked(ValueError):
    """Raised when connect-time DNS resolution violates the URL safety policy."""


def _safe_connect_scheme(host: str, port: int, schemes_by_origin: dict[tuple[str, int], str]) -> str:
    return schemes_by_origin.get((host, port)) or ("https" if port == 443 else "http")


def _resolved_http_connect_ips(host: str, port: int, scheme: str) -> list[str]:
    """Resolve and validate *host* for one HTTP connect attempt.

    Unlike :func:`is_safe_url`, this is called from the HTTP transport at the
    time the TCP socket is about to be opened.  It returns concrete IP strings
    that the transport can dial directly, closing the DNS-rebinding gap between
    pre-flight validation and connection setup for direct httpx clients.
    """
    hostname = (host or "").strip().lower().rstrip(".")
    if not hostname:
        raise SSRFConnectionBlocked("Blocked request with empty hostname")

    if hostname in _BLOCKED_HOSTNAMES:
        raise SSRFConnectionBlocked(f"Blocked request to internal hostname: {hostname}")

    allow_all_private = _global_allow_private_urls()
    allow_private_ip = _allows_private_ip_resolution(hostname, scheme)

    try:
        addr_info = socket.getaddrinfo(
            hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise SSRFConnectionBlocked(
            f"Blocked request - DNS resolution failed for: {hostname}"
        ) from exc

    safe_ips: list[str] = []
    seen: set[str] = set()
    for _family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        if "%" in ip_str:
            ip_str = ip_str.split("%")[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise SSRFConnectionBlocked(
                f"Blocked request - unparseable IP address {sockaddr[0]!r} for hostname {hostname}"
            ) from exc

        if ip in _ALWAYS_BLOCKED_IPS or any(ip in net for net in _ALWAYS_BLOCKED_NETWORKS):
            raise SSRFConnectionBlocked(
                f"Blocked request to cloud metadata address during connect: {hostname} -> {ip_str}"
            )

        if not allow_all_private and not allow_private_ip and _is_blocked_ip(ip):
            raise SSRFConnectionBlocked(
                f"Blocked request to private/internal address during connect: {hostname} -> {ip_str}"
            )

        if ip_str not in seen and len(safe_ips) < _MAX_SSRF_CONNECT_IPS:
            safe_ips.append(ip_str)
            seen.add(ip_str)

    if not safe_ips:
        raise SSRFConnectionBlocked(f"Blocked request - DNS returned no results for: {hostname}")
    return safe_ips


class _SSRFGuardedAsyncNetworkBackend:
    def __init__(self, schemes_by_origin_var: Any):
        from httpcore._backends.auto import AutoBackend

        self._backend = AutoBackend()
        self._schemes_by_origin_var = schemes_by_origin_var

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        import httpcore

        schemes_by_origin = self._schemes_by_origin_var.get({})
        scheme = _safe_connect_scheme(host, port, schemes_by_origin)
        ips = await asyncio.to_thread(_resolved_http_connect_ips, host, port, scheme)

        last_exc: Exception | None = None
        for ip in ips:
            try:
                return await self._backend.connect_tcp(
                    ip,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise SSRFConnectionBlocked(f"Blocked request - DNS returned no usable IPs for: {host}")

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> Any:
        raise SSRFConnectionBlocked("Blocked Unix socket connection in SSRF-safe transport")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _SSRFGuardedNetworkBackend:
    def __init__(self, schemes_by_origin_var: Any):
        from httpcore._backends.sync import SyncBackend

        self._backend = SyncBackend()
        self._schemes_by_origin_var = schemes_by_origin_var

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        import httpcore

        schemes_by_origin = self._schemes_by_origin_var.get({})
        scheme = _safe_connect_scheme(host, port, schemes_by_origin)
        ips = _resolved_http_connect_ips(host, port, scheme)

        last_exc: Exception | None = None
        for ip in ips:
            try:
                return self._backend.connect_tcp(
                    ip,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise SSRFConnectionBlocked(f"Blocked request - DNS returned no usable IPs for: {host}")

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> Any:
        raise SSRFConnectionBlocked("Blocked Unix socket connection in SSRF-safe transport")

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(seconds)


def _origin_scheme_context(request: Any) -> dict[tuple[str, int], str]:
    host = request.url.host
    port = request.url.port
    scheme = request.url.scheme
    if not host or port is None or scheme not in {"http", "https"}:
        return {}
    return {(host, port): scheme}


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


def _install_ssrf_guard_on_async_transport(transport: Any, schemes_by_origin_var: Any) -> None:
    state = getattr(transport, "__dict__", {}) if transport is not None else {}
    if transport is None or state.get("_hermes_ssrf_guarded", False):
        return

    pool = state.get("_pool")
    if pool is None or not hasattr(pool, "_network_backend"):
        raise SSRFConnectionBlocked("Unsupported async httpx transport cannot be made SSRF-safe")
    pool._network_backend = _SSRFGuardedAsyncNetworkBackend(schemes_by_origin_var)

    handle_async_request = getattr(transport, "handle_async_request", None)
    if handle_async_request is None:
        raise SSRFConnectionBlocked("Unsupported async httpx transport cannot be made SSRF-safe")

    async def guarded_handle_async_request(request: Any) -> Any:
        token = schemes_by_origin_var.set(_origin_scheme_context(request))
        try:
            return await handle_async_request(request)
        finally:
            schemes_by_origin_var.reset(token)

    transport.handle_async_request = guarded_handle_async_request
    transport._hermes_ssrf_guarded = True


def _install_ssrf_guard_on_transport(transport: Any, schemes_by_origin_var: Any) -> None:
    state = getattr(transport, "__dict__", {}) if transport is not None else {}
    if transport is None or state.get("_hermes_ssrf_guarded", False):
        return

    pool = state.get("_pool")
    if pool is None or not hasattr(pool, "_network_backend"):
        raise SSRFConnectionBlocked("Unsupported httpx transport cannot be made SSRF-safe")
    pool._network_backend = _SSRFGuardedNetworkBackend(schemes_by_origin_var)

    handle_request = getattr(transport, "handle_request", None)
    if handle_request is None:
        raise SSRFConnectionBlocked("Unsupported httpx transport cannot be made SSRF-safe")

    def guarded_handle_request(request: Any) -> Any:
        token = schemes_by_origin_var.set(_origin_scheme_context(request))
        try:
            return handle_request(request)
        finally:
            schemes_by_origin_var.reset(token)

    transport.handle_request = guarded_handle_request
    transport._hermes_ssrf_guarded = True


def _install_ssrf_guard_on_async_client(client: Any) -> None:
    import contextvars

    schemes_by_origin_var = contextvars.ContextVar("hermes_ssrf_async_origin_schemes")
    state = getattr(client, "__dict__", {})
    _install_ssrf_guard_on_async_transport(
        state.get("_transport"), schemes_by_origin_var
    )


def _install_ssrf_guard_on_client(client: Any) -> None:
    import contextvars

    schemes_by_origin_var = contextvars.ContextVar("hermes_ssrf_origin_schemes")
    state = getattr(client, "__dict__", {})
    _install_ssrf_guard_on_transport(
        state.get("_transport"), schemes_by_origin_var
    )


def create_ssrf_safe_async_client(**kwargs: Any) -> Any:
    """Create an ``httpx.AsyncClient`` with connect-time SSRF validation.

    Direct HTTP(S) connections are resolved, validated, and dialed by IP at
    TCP-connect time while the original request hostname is preserved for Host,
    SNI, and certificate verification.  If httpx routes through a proxy, final
    target resolution is delegated to that configured proxy; treat the proxy as
    a trusted egress boundary.
    """
    import httpx

    client = httpx.AsyncClient(**kwargs)
    _install_ssrf_guard_on_async_client(client)
    return client


def create_ssrf_safe_client(**kwargs: Any) -> Any:
    """Create an ``httpx.Client`` with connect-time SSRF validation."""
    import httpx

    client = httpx.Client(**kwargs)
    _install_ssrf_guard_on_client(client)
    return client


def redirect_target_from_response(response: Any) -> Optional[str]:
    """Return the redirect target visible from inside an httpx response hook.

    In ``httpx.AsyncClient`` response event hooks, ``response.next_request`` is
    frequently ``None`` even for a genuine redirect (it is populated later by
    the redirect-following machinery). Relying on ``next_request`` alone means
    an SSRF redirect guard silently never fires: a public URL that 302s to
    ``http://169.254.169.254/`` gets followed anyway. The ``Location`` header,
    however, is already present on the response, so resolve the target from it
    first (handling relative Locations via ``urljoin``) and only fall back to
    ``next_request`` when no ``Location`` header is set.
    """
    if not getattr(response, "is_redirect", False):
        return None

    headers = getattr(response, "headers", {}) or {}
    location = headers.get("location")
    if location:
        return urljoin(str(getattr(response, "url", "")), str(location))

    next_request = getattr(response, "next_request", None)
    if next_request:
        return str(next_request.url)

    return None
