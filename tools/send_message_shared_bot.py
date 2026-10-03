"""Shared-bot satellite fallback for the ``send`` action (``hermes send`` + ``send_message``).

A multiplexed satellite profile whose bridge is served by a host profile's bot through
``gateway.profile_routes`` (``bot_profile`` unset) holds no platform credential of its own: a copy
of the host token in the satellite's ``.env`` is a ``duplicate_credential`` fatal for the gateway.
Inbound and cron delivery already drain through the host (``_is_shared_bot_satellite``,
``SharedRouteAdapters``); this module gives the outbound ``send`` path the same rule.

Eligibility (all must hold, else fail closed with a one-line reason):

* the current profile has NO own usable credential for the platform;
* the platform authenticates with a bot token (``PLATFORM_TOKEN_ENV_NAMES``);
* exactly one host profile has an enabled ``profile_routes`` entry for (current profile, platform)
  with ``bot_profile`` unset. With several such hosts, the one whose live gateway serves this
  profile wins; anything still ambiguous fails closed.

The host credential is read from the host's own secret scope (``build_profile_secret_scope``) and
lives only in the returned ``PlatformConfig``: nothing is written to ``os.environ``, the installed
secret scope, or any profile's ``.env``. The home channel and session mirroring stay with the
satellite (callers keep using the satellite's gateway config for those). In-process, the live
adapter lookup (``_live_adapter`` → ``_adapters_for_profile``) already resolves the host's adapter
for a shared-bot satellite, so the token here only feeds the standalone/DM-resolution paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedBotHost:
    name: str
    home: Path
    pconfig: object  # gateway.config.PlatformConfig carrying the HOST's token


def _same_home(a: Path, b: Path) -> bool:
    try:
        return Path(a).expanduser().resolve(strict=False) == Path(b).expanduser().resolve(strict=False)
    except OSError:
        return False


def _candidate_homes(current_home: Path) -> list[tuple[str, Path]]:
    """``(name, home)`` for every profile on this root except the current one."""
    from hermes_cli.profiles import _get_default_hermes_home, _iter_named_profile_dirs
    homes = [("default", _get_default_hermes_home())]
    homes += [(entry.name, entry) for entry in _iter_named_profile_dirs()]
    return [(name, home) for name, home in homes if not _same_home(home, current_home)]


def _raw_config(home: Path) -> dict:
    from hermes_cli.config import read_user_config_raw
    path = Path(home) / "config.yaml"
    if not path.exists():
        return {}
    raw = read_user_config_raw(path)
    return raw if isinstance(raw, dict) else {}


def _host_routes(raw: dict, platform_name: str, current_home: Path) -> list:
    """Enabled host-bot ``profile_routes`` in *raw* that target the current profile on *platform_name*."""
    routes_raw = raw.get("profile_routes")
    if routes_raw is None and isinstance(raw.get("gateway"), dict):
        routes_raw = raw["gateway"].get("profile_routes")
    if not isinstance(routes_raw, list):
        return []
    from gateway.profile_routing import parse_profile_routes
    from hermes_cli.profiles import profile_matches_home
    return [
        route for route in parse_profile_routes(routes_raw)
        if route.enabled and route.bot_profile is None
        and str(route.platform).lower() == platform_name
        and profile_matches_home(route.profile, current_home)
    ]


def _live_gateway_serves(home: Path, profile_name: str) -> bool:
    """True when *home*'s gateway is live and serves *profile_name*: either this very process is that
    gateway (in-process send), or its verified ``gateway_state.json`` lists the profile."""
    try:
        from gateway.run import _gateway_runner_ref
        from hermes_constants import get_routing_process_hermes_home
        if _gateway_runner_ref() is not None and _same_home(get_routing_process_hermes_home(), home):
            return True
    except Exception:
        logger.debug("in-process gateway lookup failed", exc_info=True)
    try:
        from gateway.status import live_gateway_pid_for_home, read_runtime_status
        if live_gateway_pid_for_home(home) is None:
            return False
        served = (read_runtime_status(Path(home) / "gateway_state.json") or {}).get("served_profiles")
        return isinstance(served, list) and profile_name in {str(p) for p in served}
    except Exception:
        logger.debug("live gateway lookup failed for %s", home, exc_info=True)
        return False


def own_credential_present(platform, config) -> bool:
    """Does the current profile hold its own credential for *platform* (config token or env/scope)?
    A profile with its own credential is its own boundary and never borrows a host bot, even when
    its platform block is disabled."""
    from gateway.config import _getenv
    from gateway.config_env import _ENV_ENABLE_CREDENTIALS
    pconfig = config.platforms.get(platform)
    if pconfig is not None and (getattr(pconfig, "token", None) or getattr(pconfig, "api_key", None)):
        return True
    return any((_getenv(name) or "").strip() for name in _ENV_ENABLE_CREDENTIALS.get(platform) or ())


def resolve_shared_bot_host(platform_name: str, platform, config) -> tuple[Optional[SharedBotHost], str]:
    """``(host, "")`` when the current profile may send through a host profile's bot, else
    ``(None, reason)`` with a one-line, secret-free reason."""
    from gateway.config import PLATFORM_TOKEN_ENV_NAMES, PlatformConfig
    from hermes_cli.profiles import get_active_profile_name
    from hermes_constants import get_hermes_home

    if own_credential_present(platform, config):
        return None, f"this profile has its own {platform_name} credential, so it never borrows a host bot"
    token_env = PLATFORM_TOKEN_ENV_NAMES.get(platform)
    if not token_env:
        return None, f"host-bot fallback only covers token platforms; {platform_name} is not one"
    current_home = get_hermes_home()
    current_name = get_active_profile_name()

    candidates = []
    for name, home in _candidate_homes(current_home):
        try:
            raw = _raw_config(home)
            if _host_routes(raw, platform_name, current_home):
                candidates.append((name, home, raw))
        except Exception:
            logger.debug("profile_routes unreadable for %s", home, exc_info=True)
    if not candidates:
        return None, (f"no profile has an enabled gateway.profile_routes {platform_name} entry for "
                      f"profile '{current_name}' with bot_profile unset")
    if len(candidates) > 1:
        live = [c for c in candidates if _live_gateway_serves(c[1], current_name)]
        if len(live) != 1:
            names = ", ".join(sorted(c[0] for c in candidates))
            return None, (f"host bot is ambiguous: profiles {names} all route {platform_name} to "
                          f"'{current_name}' and {len(live) or 'none'} of them is a live gateway serving it")
        candidates = live
    host_name, host_home, raw = candidates[0]

    from agent.secret_scope import build_profile_secret_scope
    host_scope = build_profile_secret_scope(host_home)
    block = (raw.get("platforms") or {}).get(platform_name) if isinstance(raw.get("platforms"), dict) else None
    pconfig = PlatformConfig.from_dict(block if isinstance(block, dict) else {})
    token = (host_scope.get(token_env) or "").strip() or (pconfig.token or "").strip()
    if not token:
        return None, f"host profile '{host_name}' routes {platform_name} here but has no {token_env}"
    pconfig.enabled = True
    pconfig.token = token
    pconfig.home_channel = None  # the satellite's own home channel stays authoritative
    return SharedBotHost(host_name, Path(host_home), pconfig), ""
