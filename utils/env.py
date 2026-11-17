"""Environment values: truthy strings, numeric env vars, and proxy env var normalization."""

import os
import re
from typing import Any


TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


def is_truthy_value(value: Any, default: bool = False) -> bool:
    """Coerce bool-ish values using the project's shared truthy string set."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY_STRINGS
    return bool(value)


def env_var_enabled(name: str, default: str = "") -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return is_truthy_value(os.getenv(name, default), default=False)


def _env_number(key: str, default, cast):
    raw = os.getenv(key, "").strip()
    try:
        return cast(raw) if raw else default
    except (ValueError, TypeError):
        return default


def env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer, with fallback."""
    return _env_number(key, default, int)


def env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable as a float, with fallback."""
    return _env_number(key, default, float)


_PROXY_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")
_NO_PROXY_ENV_KEYS = ("NO_PROXY", "no_proxy")


def normalize_proxy_url(proxy_url: str | None) -> str | None:
    """Normalize proxy URLs for httpx/aiohttp: WSL/Clash export ``socks://``, httpx needs ``socks5://``."""
    candidate = str(proxy_url or "").strip()
    if candidate.lower().startswith("socks://"):
        return f"socks5://{candidate[len('socks://'):]}"
    return candidate or None


def _bare_ipv6_literal(entry: str) -> str | None:
    """The bare IPv6 literal a NO_PROXY entry should become, or None when the entry is not an
    IPv6 form httpx cannot compile (``[::1]``, ``[::1]:8080``, ``[2001:db8::1]/64``, ``::1/128``).

    httpx 0.28.1's ``get_environment_proxies`` routes bracketed entries through its wildcard
    branch (``all://*[::1]`` → ``InvalidURL: Invalid port`` at ``Client.__init__``) and cannot
    compile IPv6 CIDR bypasses at all, so both degrade to the bare literal — which httpx compiles
    into a working ``all://[<v6>]`` bypass and urllib, requests and ``agent.proxy_bypass`` all
    understand. A ``[<v6>]:port`` entry loses its port: httpx has no port-scoped IPv6 bypass form,
    and the portless literal still bypasses every port (the operator's intent superset). Bracketed
    or CIDR IPv4 is left alone (httpx compiles those forms fine).
    """
    import ipaddress

    candidate = str(entry or "").strip()
    m = (re.fullmatch(r"\[([0-9A-Fa-f:.]+)\](?:[/:].*)?", candidate)
         or re.fullmatch(r"([0-9A-Fa-f:.]+)/\d+", candidate))
    if not m:
        return None
    try:
        ip = ipaddress.ip_address(m.group(1))
    except ValueError:
        return None  # bracketed junk: httpx's wildcard branch compiles it; not ours to touch
    return m.group(1) if ip.version == 6 else None


def sanitize_no_proxy_entries(no_proxy_value: str | None) -> str:
    """Rewrite NO_PROXY entries httpx 0.28.1 cannot compile into the equivalent bare-IPv6 forms.

    Every other entry passes through verbatim, and the original string is returned byte-stable
    when nothing needed rewriting (an env without bracketed/CIDR IPv6 entries is untouched).
    """
    raw = str(no_proxy_value or "")
    entries = [part for part in re.split(r"[\s,]+", raw.strip()) if part]
    if not entries or "*" in entries:  # a wildcard bypasses everything; httpx returns no mounts
        return raw
    rewritten: list[str] = []
    for entry in entries:
        bare = _bare_ipv6_literal(entry)
        if bare is None:
            rewritten.append(entry)  # not ours to touch: IPv4, IPv4 CIDR, domains, junk
        elif bare not in rewritten:
            rewritten.append(bare)
    if rewritten == entries:
        return raw
    return ",".join(rewritten)


def normalize_proxy_env_vars() -> None:
    """Rewrite supported proxy env vars to canonical URL forms in-place."""
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key, "")
        normalized = normalize_proxy_url(value)
        if normalized and normalized != value:
            os.environ[key] = normalized
    for key in _NO_PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value:
            sanitized = sanitize_no_proxy_entries(value)
            if sanitized != value:
                os.environ[key] = sanitized
