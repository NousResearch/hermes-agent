"""First-party gateway bearer trust, ported from connectors-only 43608084f4."""

import logging
from typing import Callable, Optional
from urllib.parse import urlsplit

from tools.managed_tool_gateway import build_vendor_gateway_url, read_nous_access_token

logger = logging.getLogger(__name__)


def managed_gateway_origin() -> str:
    """Origin for on-origin managed vendors and media uploads."""
    return build_vendor_gateway_url("tool")


def connector_gateway_origin() -> str:
    """Separate connector deployment; honors CONNECTOR_GATEWAY_URL."""
    return build_vendor_gateway_url("connector")


def is_managed_nous_gateway_url(
    url: object,
    gateway_builder: Optional[Callable[[str], str]] = None,
) -> bool:
    """True when ``url`` is on one of the first-party gateway origins we build.

    Both first-party surfaces count: the media/on-origin-vendor host
    (:func:`managed_gateway_origin`) and the connectors host
    (:func:`connector_gateway_origin`). Each is compared as an exact
    ``(scheme, netloc)`` pair — never as a name or a domain suffix — so
    ``evil-connector-gateway.nousresearch.com.attacker.dev`` and an ``http``
    downgrade of a real host both stay outside the set.

    Anything granting a URL extra trust — our bearer, reading files off disk to
    upload — must gate on this, so an arbitrary URL can never inherit it.
    """
    if not isinstance(url, str) or not url.strip():
        return False

    build_origin = gateway_builder or build_vendor_gateway_url
    try:
        expected = {
            urlsplit(build_origin(label))[:2]
            for label in ("tool", "connector")
        }
        actual = urlsplit(url.strip())
    except ValueError:
        return False

    return bool(actual.scheme) and (actual.scheme, actual.netloc) in expected


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
