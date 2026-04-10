"""URL safety checks — blocks requests to private/internal network addresses.

Prevents SSRF (Server-Side Request Forgery) where a malicious prompt or
skill could trick the agent into fetching internal resources like cloud
metadata endpoints (169.254.169.254), localhost services, or private
network hosts.

Limitations (documented, not fixable at pre-flight level):
  - DNS rebinding (TOCTOU): an attacker-controlled DNS server with TTL=0
    can return a public IP for the check, then a private IP for the actual
    connection. Fixing this requires connection-level validation (e.g.
    Python's Champion library or an egress proxy like Stripe's Smokescreen).
  - Redirect-based bypass is mitigated by httpx event hooks that re-validate
    each redirect target in vision_tools, gateway platform adapters, and
    media cache helpers. Web tools use third-party SDKs (Firecrawl/Tavily)
    where redirect handling is on their servers.
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Hostnames that should always be blocked regardless of IP resolution
_BLOCKED_HOSTNAMES = frozenset({
    "metadata.google.internal",
    "metadata.goog",
})

# 100.64.0.0/10 (CGNAT / Shared Address Space, RFC 6598) is NOT covered by
# ipaddress.is_private — it returns False for both is_private and is_global.
# Must be blocked explicitly. Used by carrier-grade NAT, Tailscale/WireGuard
# VPNs, and some cloud internal networks.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP should be blocked for SSRF protection."""
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    if ip.is_multicast or ip.is_unspecified:
        return True
    # CGNAT range not covered by is_private
    if ip in _CGNAT_NETWORK:
        return True
    return False


def is_safe_url(url: str) -> bool:
    """Return True if the URL target is not a private/internal address.

    Resolves the hostname to an IP and checks against private ranges.
    Fails closed: DNS errors and unexpected exceptions block the request.
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            return False

        # Block known internal hostnames
        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning("Blocked request to internal hostname: %s", hostname)
            return False

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
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue

            if _is_blocked_ip(ip):
                logger.warning(
                    "Blocked request to private/internal address: %s -> %s",
                    hostname, ip_str,
                )
                return False

        return True

    except Exception as exc:
        # Fail closed on unexpected errors — don't let parsing edge cases
        # become SSRF bypass vectors
        logger.warning("Blocked request — URL safety check error for %s: %s", url, exc)
        return False
