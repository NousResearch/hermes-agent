"""URL safety for ``browser_exec`` (pre-scan of URL literals, hybrid local routing) and the SSRF helpers
``browser_cdp`` uses on raw CDP calls.

Ordered floors: cloud metadata / IMDS refused unconditionally (a local Chromium on a cloud VM still
reaches the host IMDS); private addresses refused unless the backend is local, the URL is served by the
hybrid local sidecar, or ``browser.allow_private_urls``; then the website allow/deny policy.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import urllib.parse
from typing import Optional

from tools.url_safety import (
    _is_declared_fake_ip,
    is_always_blocked_url,
    is_safe_url,
    normalize_url_for_request,
)
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

_PRIVATE_HOST_SUFFIXES = (".localhost", ".local", ".lan", ".internal")
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
# URL-shaped literals in a JS expression (http/https only). fetch/XHR to a private host never updates
# ``location.href``, so the literals are pre-screened instead.
_JS_URL_LITERAL_RE = re.compile(r"""https?://[^\s'"`)\]<>]+""", re.IGNORECASE)


def _err(error: str, **extra) -> dict:
    return {"success": False, "error": error, **extra}


def _private_ip(host: str) -> Optional[bool]:
    """None when ``host`` is not an IP literal."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if _is_declared_fake_ip(ip):
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip in _CGNAT


def url_is_private(url: str) -> bool:
    """True when the URL's host is (or resolves to) a private/LAN/loopback/CGNAT address.

    Routing oracle only: DNS failures are NOT private (the configured backend surfaces the error). A local
    proxy's declared fake-ip sentinel (``security.fake_ip_ranges``) is not private: the name is public and
    the cloud browser resolves it itself."""
    try:
        hostname = (urllib.parse.urlparse(url).hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False
        literal = _private_ip(hostname)
        if literal is not None:
            return literal
        if hostname == "localhost" or hostname.endswith(_PRIVATE_HOST_SUFFIXES):
            return True
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            return False
        return any(_private_ip(sockaddr[0]) for *_, sockaddr in addr_info)
    except (ValueError, OSError, UnicodeError) as exc:
        logger.debug("URL-privacy check failed for %s: %s", url, exc)
        return False


def hybrid_routes_locally(url: str) -> bool:
    """True when ``url`` belongs on the hybrid LOCAL sidecar instead of the cloud provider: a provider
    configured, ``browser.auto_local_for_private_urls`` on, a private URL, and no CDP override (it owns
    the whole session)."""
    from tools.browser_session_cdp import get_cdp_override_raw
    from tools.browser_session_cloud import auto_local_for_private_urls, get_cloud_provider
    return (
        not get_cdp_override_raw()
        and get_cloud_provider() is not None
        and auto_local_for_private_urls()
        and url_is_private(url)
    )


def _secret_url_error(url: str) -> Optional[dict]:
    """Refuse URLs embedding an API key/token (raw and URL-decoded, catching ``%2D`` tricks): a prompt
    injection could otherwise exfiltrate secrets through the URL."""
    from agent.redact import _PREFIX_RE
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(urllib.parse.unquote(url)):
        return _err("Blocked: URL contains what appears to be an API key or token. Secrets must not be sent in URLs.")
    return None


def _url_policy_error(url: str, *, auto_local: bool = False) -> Optional[dict]:
    """Backend-aware checks on a normalized URL; None if allowed.

    Credential-NAMED query params (``?token=``, ``?signature=``) are deliberately not a floor: magic links,
    OAuth callbacks and signed CDN assets are how the agent signs in, and a cloud browser already sees every
    cookie of the session. Hermes' own secrets in a URL are caught by ``_secret_url_error``."""
    from tools.browser_session_cloud import allow_private_urls, is_local_backend
    if is_always_blocked_url(url):  # unconditional: IMDS via any browser exfiltrates host credentials (#16234)
        return _err("Blocked: URL targets a cloud metadata endpoint")
    if not is_local_backend() and not auto_local and not allow_private_urls() and not is_safe_url(url):
        return _err("Blocked: URL targets a private or internal address")
    blocked = check_website_access(url)
    if blocked:
        return _err(blocked["message"],
                    blocked_by_policy={"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]})
    return None


def evaluate_url_safety(url: str, *, auto_local: bool = False) -> Optional[dict]:
    """None if ``url`` is safe to open, else an error dict. ``auto_local`` (served by the hybrid local
    sidecar) relaxes only the private-address floor, never the cloud-metadata one."""
    err = _secret_url_error(url)
    if err is None:
        url = normalize_url_for_request(url)
        err = _secret_url_error(url)
    return err or _url_policy_error(url, auto_local=auto_local)


def eval_ssrf_guard_active() -> bool:
    """True when page-driven private-network access must be guarded: a non-local backend (cloud browser,
    off-host CDP, or a containerized terminal whose host browser reaches networks it can't) without
    ``browser.allow_private_urls``."""
    from tools.browser_session_cloud import allow_private_urls, is_local_backend
    return not is_local_backend() and not allow_private_urls()


def url_blocked(url: str) -> bool:
    """True when ``url`` hits the always-blocked cloud-metadata floor or fails the SSRF guard."""
    return is_always_blocked_url(url) or not is_safe_url(url)


def expression_targets_private_url(expression: str) -> Optional[str]:
    """The first private/always-blocked ``http(s)://`` literal in a JS expression (best-effort), else None."""
    literals = _JS_URL_LITERAL_RE.findall(expression) if isinstance(expression, str) else []
    return next((c for c in (m.rstrip(".,;") for m in literals) if url_blocked(c)), None)
