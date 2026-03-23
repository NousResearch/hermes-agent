"""URL safety checks — blocks requests to private/internal network addresses.

Prevents SSRF (Server-Side Request Forgery) where a malicious prompt or
skill could trick the agent into fetching internal resources like cloud
metadata endpoints (169.254.169.254), localhost services, or private
network hosts.
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


def is_safe_url(url: str) -> bool:
    """Return True if the URL target is not a private/internal address.

    Resolves the hostname to an IP and checks against private ranges.
    Returns True for unresolvable hosts (let the HTTP client handle DNS
    errors) to avoid false positives on legitimate CDN hostnames.
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
            # DNS resolution failed — let the HTTP client deal with it
            return True

        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue

            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                logger.warning(
                    "Blocked request to private/internal address: %s -> %s",
                    hostname, ip_str,
                )
                return False

        return True

    except Exception as exc:
        logger.debug("URL safety check error for %s: %s", url, exc)
        return True  # Fail open on unexpected errors
