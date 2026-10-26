"""Task-owned page bindings with exact-object cleanup and creation lifetime fences."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import threading
import time
from typing import Any

from hermes_constants import get_hermes_home, hermes_home_key


class BrowserSessionRetired(RuntimeError):
    """The operation's task or profile was retired before it could publish."""


@dataclass
class SessionLifetime:
    home_key: str
    owner_home: str
    lock: Any = field(default_factory=threading.RLock)


def capture_lifetime(key: str) -> SessionLifetime:
    from tools import browser_session as bs
    with bs._cleanup_lock:
        lifetime = bs._session_lifetimes.setdefault(
            key, SessionLifetime(hermes_home_key(), str(get_hermes_home())))
        bs._session_owner_homes.setdefault(key, lifetime.owner_home)
        bs._session_last_activity[key] = time.time()
    from tools.browser_session_lifecycle import start_cleanup_thread
    start_cleanup_thread()
    return lifetime


def require_lifetime(key: str, lifetime: SessionLifetime) -> None:
    from tools import browser_session as bs
    with bs._cleanup_lock:
        current = bs._session_lifetimes.get(key) is lifetime
    if not current:
        raise BrowserSessionRetired("The browser task ended while this operation was preparing; its result was discarded.")


def binding_is_current(key: str, info: dict[str, Any]) -> bool:
    from tools import browser_session as bs
    from tools.browser_session_cdp import live_cdp_generation
    with bs._cleanup_lock:
        return (bs._active_sessions.get(key) is info
                and bs._session_lifetimes.get(key) is info["_lifetime"]
                and live_cdp_generation(info["home_key"]) == info["_endpoint_generation"])


def get_owned_session(key: str, *, lifetime: SessionLifetime | None = None,
                      endpoint: str | None = None, expected_generation: int | None = None) -> dict[str, Any]:
    """Resolve outside the global lock, publishing only into the captured live task."""
    from tools import browser_session as bs, browser_session_lifecycle as lifecycle
    from tools.browser_session_cdp import live_cdp_generation, resolve_cdp_override
    lifetime = lifetime or capture_lifetime(key)
    lifecycle.start_cleanup_thread()
    lifecycle.update_session_activity(key)
    with lifetime.lock:
        require_lifetime(key, lifetime)
        with bs._cleanup_lock:
            existing = bs._active_sessions.get(key)
        if existing is not None:
            matching_endpoint = endpoint is None or existing.get("cdp_url") == endpoint
            if (matching_endpoint and binding_is_current(key, existing)
                    and not lifecycle.session_has_expired(existing) and not bs._backend_dead(existing)):
                return existing
            lifecycle.cleanup_single_session(key, expected=existing, preserve_lifetime=True)
        for _attempt in range(bs._STALE_ENDPOINT_RETRIES + 1):
            require_lifetime(key, lifetime)
            generation = live_cdp_generation(lifetime.home_key)
            if expected_generation is not None and generation != expected_generation:
                raise BrowserSessionRetired("The profile changed browser endpoints before session publication.")
            if endpoint is None:
                candidate = bs._create_session_for_key(key, bs.is_local_sidecar_key(key))
            else:
                candidate = bs._session_record("cdp", resolve_cdp_override(endpoint), {"cdp_override": True})
            candidate.update(_lifetime=lifetime, _endpoint_generation=generation,
                             _owner_home=lifetime.owner_home, home_key=lifetime.home_key)
            with bs._cleanup_lock:
                live = bs._session_lifetimes.get(key) is lifetime
                same_endpoint = live_cdp_generation(lifetime.home_key) == generation
                if live and same_endpoint:
                    return bs._publish(key, candidate, lifetime.home_key)
            with bs._cleanup_lock:
                bs._retired_sessions.setdefault(key, []).append(candidate)
            lifecycle._dispose_retired(key, [candidate])
            if not live:
                raise BrowserSessionRetired("The browser task ended during endpoint discovery or launch.")
            if endpoint is not None:
                raise BrowserSessionRetired("The profile changed browser endpoints during this operation.")
        raise BrowserSessionRetired("The profile kept changing browser endpoints; no stale session was published.")


def cdp_call(endpoint: str, method: str, params: dict[str, Any], target_id: str | None = None):
    from tools.browser_cdp_tool import _cdp_call, _run_async
    return _run_async(_cdp_call(endpoint, method, params, target_id, 10.0))


def close_owned_target(info: dict[str, Any]) -> None:
    """Close recorded created target IDs; explicit selection never transfers ownership."""
    owned = info.get("_owned_target_ids")
    targets_to_close = set(owned) if owned is not None else {info["target_id"]} if info.get("target_id") else set()
    if not targets_to_close:
        return
    endpoint = info["cdp_url"]
    targets = cdp_call(endpoint, "Target.getTargets", {}).get("targetInfos", [])
    present = {row["targetId"] for row in targets}
    for target in targets_to_close & present:
        result = cdp_call(endpoint, "Target.closeTarget", {"targetId": target})
        if not result.get("success"):
            raise RuntimeError("The browser refused to close the task-owned target.")
    targets = cdp_call(endpoint, "Target.getTargets", {}).get("targetInfos", [])
    if targets_to_close & {row["targetId"] for row in targets}:
        raise RuntimeError("The task-owned target remains open after cleanup.")


def bind_owned_target(key: str, info: dict[str, Any], supervisor_key: str):
    """Create one owned page and attach the supervisor to that exact page, never an arbitrary tab."""
    from tools import browser_session as bs
    from tools.browser_session_cdp import get_dialog_policy_config
    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    with info["_lifetime"].lock:
        if not binding_is_current(key, info):
            raise BrowserSessionRetired("The browser binding was retired before page creation.")
        if not info.get("target_id"):
            endpoint = info["cdp_url"]
            pages = []
            available = [row for row in cdp_call(endpoint, "Target.getTargets", {}).get("targetInfos", [])
                         if row.get("type") == "page"]
            pages = [row for row in available if row["targetId"] in info.get("_owned_target_ids", ())]
            if not pages and (info.get("features") or {}).get("lightpanda"):
                pages = available
            target = (pages[0]["targetId"] if pages else
                      cdp_call(endpoint, "Target.createTarget", {"url": "about:blank"})["targetId"])
            info.setdefault("_owned_target_ids", set()).add(target)
            with bs._cleanup_lock:
                from tools.browser_session_cdp import live_cdp_generation
                live = (bs._active_sessions.get(key) is info
                        and bs._session_lifetimes.get(key) is info["_lifetime"]
                        and live_cdp_generation(info["home_key"]) == info["_endpoint_generation"])
                if live:
                    info["target_id"] = target
            if not live:
                close_owned_target({"cdp_url": endpoint, "target_id": target})
                raise BrowserSessionRetired("The browser task ended during page creation.")
        policy, timeout_s = get_dialog_policy_config()
        supervisor = SUPERVISOR_REGISTRY.get_or_start(
            supervisor_key, info["cdp_url"], target_id=info["target_id"],
            dialog_policy=policy, dialog_timeout_s=timeout_s,
            publish_guard=lambda: binding_is_current(key, info))
        if not binding_is_current(key, info) or SUPERVISOR_REGISTRY.get(supervisor_key) is not supervisor:
            SUPERVISOR_REGISTRY._pop(supervisor_key, expected=supervisor)
            supervisor.stop()
            raise BrowserSessionRetired("The browser task ended while its supervisor was attaching.")
        with bs._cleanup_lock:
            if bs._active_sessions.get(key) is not info:
                live = False
            else:
                info.setdefault("_supervisors", {})[supervisor_key] = supervisor
                live = True
        if not live:
            SUPERVISOR_REGISTRY._pop(supervisor_key, expected=supervisor)
            supervisor.stop()
            raise BrowserSessionRetired("The browser task ended before its supervisor could publish.")
        return supervisor


def pin_exec(env: dict, key: str, supervisor_key: str, operation_key: str,
             operation: SessionLifetime, generation: int) -> dict[str, Any] | None:
    """Pin a CDP-routed harness call before any user code is dispatched."""
    from tools.browser_session_cdp import live_cdp_generation
    require_lifetime(operation_key, operation)
    if live_cdp_generation(operation.home_key) != generation:
        raise BrowserSessionRetired("The profile changed browser endpoints before code dispatch.")
    endpoint = env.get("BU_CDP_WS") or env.get("BU_CDP_URL")
    if not endpoint:
        return None  # The harness owns its direct-cloud browser; no shared CDP endpoint exists.
    from tools import browser_session as bs
    with bs._cleanup_lock:
        if bs._session_lifetimes.get(operation_key) is not operation:
            raise BrowserSessionRetired("The browser task ended before its route could bind.")
        lifetime = bs._session_lifetimes.setdefault(key, SessionLifetime(operation.home_key, operation.owner_home))
    info = get_owned_session(key, lifetime=lifetime, endpoint=endpoint, expected_generation=generation)
    require_lifetime(operation_key, operation)
    bind_owned_target(key, info, supervisor_key)
    env.pop("BU_CDP_URL", None)
    env["BU_CDP_WS"] = info["cdp_url"]
    identity = hashlib.sha256((key + "\0" + info["session_name"]).encode()).hexdigest()[:32]
    env["BU_NAME"] = "hermes-" + identity
    return info


def dispose_owned_binding(info: dict[str, Any]) -> None:
    """Dispose captured resources without looking up a replacement by its task key."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    daemon = info.get("_harness_daemon")
    if daemon is not None:
        from tools.browser_use_cli import stop_owned_harness_daemon
        stop_owned_harness_daemon(daemon)
        info.pop("_harness_daemon", None)
    from tools.browser_owned_exec import consume_owned_exec
    consume_owned_exec(info)
    for key, supervisor in list(info.get("_supervisors", {}).items()):
        SUPERVISOR_REGISTRY._pop(key, expected=supervisor)
        supervisor.stop()
    info.pop("_supervisors", None)
    close_owned_target(info)
