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
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
from urllib.parse import parse_qsl, quote, unquote, urljoin, urlparse, urlsplit, urlunsplit

from hermes_constants import get_hermes_home_override
from utils import is_truthy_value

logger = logging.getLogger(__name__)


# ── Proxy detection ──────────────────────────────────────────
# Proxy environment variables that indicate the runtime should
# delegate DNS to a proxy rather than attempting direct resolution.
_PROXY_ENV_VARS = (
    "HTTPS_PROXY", "https_proxy",
    "HTTP_PROXY", "http_proxy",
    "ALL_PROXY", "all_proxy",
)


def _proxy_is_configured() -> bool:
    """Return True when at least one HTTP proxy env var is set."""
    return any(os.environ.get(v) for v in _PROXY_ENV_VARS)


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

# Deprecated IPv4-compatible embedding (::/96). Python 3.11 flags these
# ``is_reserved``, but the class is named explicitly so the floor reason is
# pinned and toggle-independent (``::a9fe:a9fe`` == ``::169.254.169.254``
# must always block, even with allow_private_urls on).
_IPV4_COMPATIBLE_NETWORK = ipaddress.ip_network("::/96")

# Hostname suffix floor: names under these suffixes are always blocked as
# metadata/internal without any DNS work. Deliberately minimal (Region D
# pin): ``.internal`` only — it subsumes ``.compute.internal`` — while
# ``.local``/``.lan`` stay routing-only to protect benign mDNS/VPN setups.
_BLOCKED_HOSTNAME_SUFFIXES = (".internal",)

# Reason classes that are part of the non-negotiable security floor: they
# block regardless of the ``allow_private_urls`` toggle or any trusted-host
# exemption. See ``_classify_ip`` / ``resolve_and_check_url``.
FLOOR_REASONS = frozenset({
    "blocked:metadata-host",
    "blocked:metadata-ip",
    "blocked:link-local",
    "blocked:ipv4-compatible",
})

# Reason classes (without the ``blocked:`` prefix, as returned by
# ``_classify_ip``) that the ``allow_private_urls`` toggle / trusted-host
# exemption may skip. Floor classes are never in this set.
_TOGGLE_SKIP_REASONS = frozenset({
    "private-ip",
    "cgnat",
    "reserved",
    "multicast",
    "unspecified",
    "loopback",
})

# Reasons (as returned by ``_url_is_private``'s ``resolve_and_check_url``
# verdict) that mean "the URL routes to the private/LAN sidecar".
_PRIVATE_ROUTING_REASONS = frozenset({
    "blocked:private-ip",
    "blocked:cgnat",
    "blocked:reserved",
    "blocked:multicast",
    "blocked:unspecified",
    "blocked:loopback",
    "blocked:link-local",
    "blocked:metadata-host",
    "blocked:metadata-ip",
    "blocked:ipv4-compatible",
})

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

    The single-profile result is cached for the process lifetime. Multiplexed
    profile turns bypass that process-global cache because their config root is
    context-local; ``read_raw_config()`` already provides path/mtime caching.
    """
    global _allow_private_resolved, _cached_allow_private

    # A multiplex gateway serves several independently configured profiles in
    # one process. Reusing the first profile's opt-out here would let it disable
    # private-network blocking for every later profile in that process.
    if get_hermes_home_override() is not None:
        return _resolve_allow_private_urls()

    if _allow_private_resolved:
        return _cached_allow_private

    _allow_private_resolved = True
    _cached_allow_private = _resolve_allow_private_urls()
    return _cached_allow_private


def _resolve_allow_private_urls() -> bool:
    """Resolve the effective private-URL toggle from the active config scope."""

    # 1. Env var override (highest priority)
    env_val = os.getenv("HERMES_ALLOW_PRIVATE_URLS", "").strip().lower()
    if env_val in {"true", "1", "yes"}:
        return True
    if env_val in {"false", "0", "no"}:
        # Explicit false — don't fall through to config
        return False

    # 2. Config file
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        # security.allow_private_urls (preferred)
        sec = cfg.get("security", {})
        if isinstance(sec, dict) and is_truthy_value(
            sec.get("allow_private_urls"), default=False
        ):
            return True
        # browser.allow_private_urls (legacy fallback)
        browser = cfg.get("browser", {})
        if isinstance(browser, dict) and is_truthy_value(
            browser.get("allow_private_urls"), default=False
        ):
            return True
    except Exception:
        # Config unavailable (e.g. tests, early import) — keep default
        pass

    return False


def _reset_allow_private_cache() -> None:
    """Reset the cached toggle — only for tests."""
    global _allow_private_resolved, _cached_allow_private
    _allow_private_resolved = False
    _cached_allow_private = False


def _classify_ip(ip: ipaddress._BaseAddress) -> tuple[bool, str]:
    """Classify one parsed IP address: ``(blocked, reason)``.

    This is the single classification core for the whole policy surface.
    Reasons (pinned, deterministic evaluation order — see the Region D
    contract):

    ``'ok'`` | ``'metadata-ip'`` | ``'link-local'`` | ``'loopback'`` |
    ``'private-ip'`` | ``'cgnat'`` | ``'reserved'`` | ``'multicast'`` |
    ``'unspecified'`` | ``'ipv4-compatible'``

    IPv4-mapped IPv6 addresses are unwrapped first and the reason is
    prefixed with ``mapped:`` (e.g. ``mapped:cgnat``) so callers can tell
    the encoding from the class. ``'blocked:parse'`` is never produced here
    (that is ``ip_is_blocked``'s job for unparseable strings).
    """
    # 1. Mapped unwrap — check the embedded IPv4, tagged with the encoding.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        blocked, reason = _classify_ip(ip.ipv4_mapped)
        return blocked, f"mapped:{reason}"

    # 2. Security floor: exact sentinel metadata IPs, then link-local.
    if ip in _ALWAYS_BLOCKED_IPS:
        return True, "metadata-ip"
    if ip.is_link_local:  # IPv4 169.254.0.0/16 (remainder) + IPv6 fe80::/10
        return True, "link-local"

    # 3. Obvious non-routable classes.
    if ip.is_loopback:
        return True, "loopback"
    if ip.is_multicast:
        return True, "multicast"
    if ip.is_unspecified:
        return True, "unspecified"

    # 4. Deprecated IPv4-compatible embedding (::/96) — floor class.
    if isinstance(ip, ipaddress.IPv6Address) and ip in _IPV4_COMPATIBLE_NETWORK:
        return True, "ipv4-compatible"

    # 5. RFC1918 / IPv6 ULA (fc00::/7) — private.
    if ip.is_private:
        return True, "private-ip"

    # 6. Explicit IPv4-only range checks, type-guarded so IPv6 addresses
    #    never silently no-op through an IPv4 network containment test.
    if isinstance(ip, ipaddress.IPv4Address):
        if ip in _CGNAT_NETWORK:
            return True, "cgnat"
        if ip in ipaddress.ip_network("172.16.0.0/12"):
            return True, "private-ip"
        if ip in ipaddress.ip_network("198.18.0.0/15"):
            return True, "private-ip"

    # 7. Everything else reserved (incl. benchmark/documentation ranges).
    if ip.is_reserved:
        return True, "reserved"

    return False, "ok"


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP should be blocked for SSRF protection."""
    return _classify_ip(ip)[0]


# ---------------------------------------------------------------------------
# Shared resolve-and-validate core (Region D)
# ---------------------------------------------------------------------------
# The helpers below are the single source of truth the whole SSRF policy
# surface consumes: ``is_safe_url`` / ``is_always_blocked_url`` keep their
# documented boundary semantics on top, while the model-controlled surfaces
# (browser_exec pre-check + landing recheck, the CDP monitor, the egress
# interposer) use ``resolve_and_check_url`` / ``url_block_reason`` directly
# so ``error:dns`` fails closed there regardless of proxy environment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class URLSafetyVerdict:
    """Result of :func:`resolve_and_check_url`.

    ``reason`` is one of ``'ok'`` | ``'blocked:metadata-host'`` |
    ``'blocked:metadata-ip'`` | ``'blocked:link-local'`` |
    ``'blocked:loopback'`` | ``'blocked:private-ip'`` |
    ``'blocked:cgnat'`` | ``'blocked:reserved'`` | ``'blocked:multicast'`` |
    ``'blocked:unspecified'`` | ``'blocked:ipv4-compatible'`` |
    ``'blocked:numeric-ip'`` | ``'blocked:unsupported-scheme'`` |
    ``'blocked:parse'`` | ``'error:dns'`` | ``'error:internal'``.
    """

    ok: bool
    reason: str
    detail: str
    scheme: str = ""
    hostname: str = ""
    resolved_ips: tuple[str, ...] = ()
    checked_at: float = 0.0


def _coerce_numeric_host(hostname: str) -> Optional[str]:
    """Strictly decode numeric hostname forms into a dotted IPv4 string.

    Handles pure-decimal integers (``2130706433``), ``0x`` hex integers,
    dotted-hex (``0x7f.0.0.1``), dotted-octal (``0177.0.0.1``), mixed radix
    and short forms (``127.1``) the way Chromium/glibc dial them. Each octet
    is parsed with an explicit radix; any out-of-range or non-numeric
    component makes the whole host "not numeric" (returns None). The
    resolver is never consulted for these — see ``resolve_and_check_url``.
    """
    h = (hostname or "").strip().lower()
    if not h or h[0] in "+-":
        return None
    parts = h.split(".")
    if len(parts) > 4:
        return None

    numbers: list[int] = []
    for i, part in enumerate(parts):
        if not part:
            return None
        if part.startswith("0x"):
            try:
                n = int(part[2:], 16)
            except ValueError:
                return None
        elif len(part) > 1 and part.startswith("0"):
            try:
                n = int(part, 8)
            except ValueError:
                return None
        else:
            try:
                n = int(part, 10)
            except ValueError:
                return None
        if n < 0:
            return None
        if i == len(parts) - 1 and len(parts) > 1 and n > 255:
            return None
        if i < len(parts) - 1 and n > 255:
            return None
        numbers.append(n)

    if len(numbers) == 1:
        v = numbers[0]
        if v > 0xFFFFFFFF:
            return None
        return ".".join(str((v >> shift) & 0xFF) for shift in (24, 16, 8, 0))

    # Multi-part form: every component is an octet; missing middle octets
    # are zero (inet_aton semantics: ``127.1`` → 127.0.0.1).
    octets = [0, 0, 0, 0]
    octets[0] = numbers[0]
    octets[-1] = numbers[-1]
    if len(numbers) == 3:
        octets[1] = numbers[1]
    elif len(numbers) == 4:
        octets[1], octets[2] = numbers[1], numbers[2]
    return ".".join(str(o) for o in octets)


def _strict_parse_fail(url: str, reason: str, detail: str) -> URLSafetyVerdict:
    return URLSafetyVerdict(ok=False, reason=reason, detail=detail, checked_at=time.time())


def _verdict_reason(classify_reason: str) -> str:
    """Turn a ``_classify_ip`` reason into a verdict reason string.

    IPv4-mapped prefixes (``mapped:…``) are normalized away for the verdict
    so floor consumers (``FLOOR_REASONS``) match regardless of the address
    encoding; ``ip_is_blocked`` still exposes the raw tagged reason.
    """
    if classify_reason.startswith("mapped:"):
        classify_reason = classify_reason[len("mapped:"):]
    return f"blocked:{classify_reason}"


def resolve_and_check_url(
    url: str,
    *,
    port: Optional[int] = None,
    allow_private: Optional[bool] = None,
    trusted_private_hosts: Optional[frozenset[str]] = None,
    resolve: Optional[Callable[..., Any]] = None,
    now: Optional[Callable[[], float]] = None,
) -> URLSafetyVerdict:
    """Resolve, validate, and pin one URL — the shared SSRF verdict helper.

    Deterministic, fail-closed behavior (in order):

    1. **Strict parse.** The authority span (after ``scheme://``) is
       percent-decoded; any decoded backslash or ASCII control character
       (0x00-0x1F, 0x7F) → ``blocked:parse``. WHATWG canonicalizes those
       into authority separators the Python-visible hostname never shows,
       so rejection is the only fail-closed choice (``http://***@evil.com/``
       style parser-divergence).
    2. **Scheme** must be ``http``/``https`` → else ``blocked:unsupported-scheme``.
    3. **Hostname floor** — ``_BLOCKED_HOSTNAMES`` or ``_BLOCKED_HOSTNAME_SUFFIXES``
       (``.internal``) → ``blocked:metadata-host`` (no DNS).
    4. **Numeric-IP coercion** — literal IPs and their numeric encodings are
       classified directly; the resolver is never consulted for them.
    5. **Resolve once** — any ``gaierror`` or empty answer set →
       ``error:dns`` (no proxy-delegation fail-open inside this helper; the
       shim survives only at ``is_safe_url``'s own boundary).
    6. **Classify every answer** — any blocked answer fails the whole
       verdict with its first blocking reason (whole-set semantics,
       everything classified even beyond the returned cap).
    7. **Honor toggles at the boundary** — ``allow_private=True`` or an
       HTTPS trusted-private host skips the toggle-skip set only; floor
       classes always block.
    8. **Pin** — on ``ok``, ``resolved_ips`` are the deduped, zone-stripped,
       validated answers capped at ``_MAX_SSRF_CONNECT_IPS``. Callers that
       dial must connect to these exact IPs and never re-resolve.

    ``allow_private`` defaults to the global ``security.allow_private_urls``
    toggle; ``trusted_private_hosts`` defaults to ``_TRUSTED_PRIVATE_IP_HOSTS``.
    ``resolve`` and ``now`` are injectable seams for tests / async offload.
    """
    ts = now() if now is not None else time.time()

    raw = (url or "").strip()
    if not raw:
        return _strict_parse_fail(raw, "blocked:parse", "empty URL")

    # 1. Strict parse — authority span, decoded, checked for separators.
    scheme_match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", raw)
    if not scheme_match:
        return _strict_parse_fail(raw, "blocked:parse", "URL has no scheme")
    scheme = scheme_match.group(1).lower()
    rest = raw[scheme_match.end():]
    if rest.startswith("//"):
        authority = rest[2:]
        for sep in ("/", "?", "#"):
            idx = authority.find(sep)
            if idx != -1:
                authority = authority[:idx]
                break
        decoded_authority = unquote(authority)
        if "\\" in decoded_authority or any(
            ord(c) < 0x20 or ord(c) == 0x7F for c in decoded_authority
        ):
            return _strict_parse_fail(
                raw, "blocked:parse",
                f"authority contains a decoded separator or control character ({decoded_authority!r})",
            )

    # 2. Scheme gate.
    if scheme not in {"http", "https"}:
        return _strict_parse_fail(raw, "blocked:unsupported-scheme", f"unsupported scheme {scheme!r}")

    # 3. Hostname.
    try:
        parsed = urlparse(raw)
    except ValueError as exc:
        return _strict_parse_fail(raw, "blocked:parse", f"unparseable URL: {exc}")
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return _strict_parse_fail(raw, "blocked:parse", "URL has no hostname")

    # 4. Hostname floor (no DNS).
    if hostname in _BLOCKED_HOSTNAMES or hostname.endswith(_BLOCKED_HOSTNAME_SUFFIXES):
        return URLSafetyVerdict(
            ok=False, reason="blocked:metadata-host",
            detail=f"hostname {hostname!r} is an always-blocked metadata/internal name",
            scheme=scheme, hostname=hostname, checked_at=ts,
        )

    trusted = _TRUSTED_PRIVATE_IP_HOSTS if trusted_private_hosts is None else trusted_private_hosts
    allow_all_private = _global_allow_private_urls() if allow_private is None else allow_private
    allow_private_ip = scheme == "https" and hostname in trusted  # https gate pinned (D5)

    # 5. Literal-IP / numeric-coercion path — resolver never consulted.
    try:
        ip = ipaddress.ip_address(hostname)
        coerced = False
    except ValueError:
        coerced_host = _coerce_numeric_host(hostname)
        if coerced_host is None:
            ip = None
            coerced = False
        else:
            try:
                ip = ipaddress.ip_address(coerced_host)
                coerced = True
            except ValueError:
                ip = None
                coerced = False

    if ip is not None:
        blocked, reason = _classify_ip(ip)
        plain_reason = reason[len("mapped:"):] if reason.startswith("mapped:") else reason
        if blocked and (plain_reason not in _TOGGLE_SKIP_REASONS or not (allow_all_private or allow_private_ip)):
            # Numeric encodings are classified by their coerced address
            # class (2130706433 → 127.0.0.1 → blocked:loopback); the
            # resolver is never consulted for them (G5).
            numeric_tag = " (numeric hostname encoding)" if coerced else ""
            return URLSafetyVerdict(
                ok=False, reason=_verdict_reason(reason),
                detail=f"{hostname!r} (dialed as {ip}) is a blocked address class{numeric_tag}",
                scheme=scheme, hostname=hostname, checked_at=ts,
            )
        return URLSafetyVerdict(
            ok=True, reason="ok",
            detail=f"{hostname!r} resolves to the allowed literal {ip}",
            scheme=scheme, hostname=hostname,
            resolved_ips=(str(ip),), checked_at=ts,
        )

    # 6. Resolve once. The resolver is looked up at call time so test
    #    seams and runtime monkeypatches of ``socket.getaddrinfo`` apply.
    resolver = resolve if resolve is not None else socket.getaddrinfo
    try:
        addr_info = resolver(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return URLSafetyVerdict(
            ok=False, reason="error:dns",
            detail=f"DNS resolution failed for {hostname!r}: {exc}",
            scheme=scheme, hostname=hostname, checked_at=ts,
        )
    except Exception as exc:
        return URLSafetyVerdict(
            ok=False, reason="error:internal",
            detail=f"resolution raised for {hostname!r}: {exc}",
            scheme=scheme, hostname=hostname, checked_at=ts,
        )

    safe_ips: list[str] = []
    seen: set[str] = set()
    for _family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        if "%" in ip_str:
            ip_str = ip_str.split("%", 1)[0]
        try:
            resolved = ipaddress.ip_address(ip_str)
        except ValueError:
            return URLSafetyVerdict(
                ok=False, reason="error:internal",
                detail=f"unparseable address {sockaddr[0]!r} for {hostname!r}",
                scheme=scheme, hostname=hostname, checked_at=ts,
            )
        blocked, reason = _classify_ip(resolved)
        plain_reason = reason[len("mapped:"):] if reason.startswith("mapped:") else reason
        if blocked and (plain_reason not in _TOGGLE_SKIP_REASONS or not (allow_all_private or allow_private_ip)):
            return URLSafetyVerdict(
                ok=False, reason=_verdict_reason(reason),
                detail=f"{hostname!r} resolved to blocked address {ip_str}",
                scheme=scheme, hostname=hostname, checked_at=ts,
            )
        if ip_str not in seen:
            seen.add(ip_str)
            if len(safe_ips) < _MAX_SSRF_CONNECT_IPS:  # cap applies to the returned list only
                safe_ips.append(ip_str)

    if not safe_ips:
        return URLSafetyVerdict(
            ok=False, reason="error:dns",
            detail=f"DNS returned no usable answers for {hostname!r}",
            scheme=scheme, hostname=hostname, checked_at=ts,
        )

    return URLSafetyVerdict(
        ok=True, reason="ok",
        detail=f"{hostname!r} resolved to {len(safe_ips)} allowed address(es)",
        scheme=scheme, hostname=hostname,
        resolved_ips=tuple(safe_ips), checked_at=ts,
    )


async def async_resolve_and_check_url(
    url: str, **kwargs: Any
) -> URLSafetyVerdict:
    """Async twin of :func:`resolve_and_check_url` (DNS off the event loop)."""
    return await asyncio.to_thread(resolve_and_check_url, url, **kwargs)


def ip_is_blocked(ip: Union[str, ipaddress._BaseAddress]) -> tuple[bool, str]:
    """Classify an already-observed IP: ``(blocked, reason)``.

    For strings, a zone scope is stripped before parsing; an unparseable
    string fails closed with ``(True, 'blocked:parse')``. Reasons are the
    raw ``_classify_ip`` classes (e.g. ``'mapped:cgnat'``) plus
    ``'blocked:parse'``.
    """
    if isinstance(ip, str):
        host = ip.strip()
        if "%" in host:
            host = host.split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True, "blocked:parse"
    try:
        blocked, reason = _classify_ip(ip)
        return blocked, reason
    except Exception:
        return True, "blocked:parse"


def url_block_reason(url: str, *, allow_private: Optional[bool] = None) -> Optional[str]:
    """Strict predicate for model-controlled navigation surfaces.

    Returns the verdict ``reason`` when the URL is blocked, ``None`` when
    safe. Deliberately does NOT apply ``is_safe_url``'s proxy-delegation
    shim: ``error:dns`` / ``error:internal`` / ``blocked:parse`` all block
    here. Use this (or ``resolve_and_check_url``) wherever a third-party
    browser/CLI will dial the destination.
    """
    v = resolve_and_check_url(url, allow_private=allow_private)
    return None if v.ok else v.reason


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
      - Hostnames in ``_BLOCKED_HOSTNAMES`` or under ``_BLOCKED_HOSTNAME_SUFFIXES``
      - IPs / networks in ``_ALWAYS_BLOCKED_IPS`` / ``_ALWAYS_BLOCKED_NETWORKS``
      - IPv4-compatible ``::/96`` addresses (new floor)
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
        v = resolve_and_check_url(url, allow_private=True)
        if not v.ok and v.reason in FLOOR_REASONS:
            logger.warning(
                "Blocked request to cloud metadata address (always-blocked floor): %s",
                v.detail,
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

    **Delegation caveat (Region D pin):** this function keeps a
    proxy-environment DNS-delegation fail-open at ITS OWN boundary only
    (sandbox/Docker+Squid environments where direct DNS is blocked and
    only HTTP(S) through the proxy is permitted). Callers whose
    destination will be dialed by a third-party browser/CLI must use
    ``resolve_and_check_url`` / ``url_block_reason`` instead — never this
    function's delegation branch.
    """
    try:
        v = resolve_and_check_url(url)
        if v.ok:
            return True

        # Proxy-delegation shim, kept ONLY here: in proxy-only sandboxes
        # direct DNS is blocked at the network level and the proxy is the
        # egress boundary.  Literal IPs never qualify (no DNS needed), and
        # blocked hostnames/metadata floors already failed above.
        if v.reason == "error:dns" and _proxy_is_configured():
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").strip().lower().rstrip(".")
            try:
                ipaddress.ip_address(hostname)
                _is_literal_ip = True
            except ValueError:
                _is_literal_ip = False
            if not _is_literal_ip:
                logger.debug(
                    "DNS resolution failed for %s — proxy configured, "
                    "allowing through for proxy-side resolution",
                    hostname,
                )
                return True
            logger.warning("Blocked request — DNS resolution failed for: %s", hostname)
            return False

        logger.warning("Blocked request — URL safety check failed for %s (%s)", url, v.reason)
        return False

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

        blocked, reason = _classify_ip(ip)
        plain_reason = reason[7:] if reason.startswith("mapped:") else reason
        if plain_reason in ("metadata-ip", "link-local", "ipv4-compatible"):
            raise SSRFConnectionBlocked(
                f"Blocked request to cloud metadata address during connect: {hostname} -> {ip_str}"
            )

        if not allow_all_private and not allow_private_ip and blocked:
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
