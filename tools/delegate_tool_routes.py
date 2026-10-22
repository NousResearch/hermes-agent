"""Per-task child routes from the active delegation provider (``delegation.backend``).

``delegate_task`` asks the active :class:`agent.delegation_provider.DelegationProvider` for one
route per task, validates the model's provider-declared parameters, turns each route into the
credential bundle child construction already consumes (the same resolver ``delegation.*`` uses),
and binds a profile route's scope around the child's construction and conversation. The provider
never touches the lifecycle; this module is the whole contract between the two.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from agent.delegation_provider import (
    RESERVED_TASK_PARAMETERS, ROUTE_KEYS, DelegationProvider, DelegationRouteError,
)

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

_PARAM_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass
class TaskRoute:
    """How core builds one child: the credential bundle, the config block owning its fallback
    policy, an optional reasoning override, and the profile home it runs as (None = this one)."""

    creds: dict[str, Any]
    routing_cfg: dict[str, Any]
    reasoning_effort: Any = None
    profile_home: Optional[Path] = None
    label: Optional[str] = None


def _active_provider(config: Mapping[str, Any]) -> DelegationProvider:
    from agent.delegation_registry import get_active_provider
    return get_active_provider(config)


def _declared_parameters(provider: DelegationProvider, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The provider's per-task parameters, minus anything malformed or reserved (warned)."""
    try:
        raw = provider.task_parameters(config)
    except Exception:  # plugin boundary: a broken provider costs its parameters, never the delegate_task tool
        logger.warning("delegation provider %s.task_parameters() raised; no extra parameters", provider.name,
                       exc_info=True)
        return {}
    if not isinstance(raw, Mapping):
        return {}
    accepted: dict[str, dict[str, Any]] = {}
    for name, schema in raw.items():
        if not isinstance(name, str) or not _PARAM_NAME_RE.match(name) or name in RESERVED_TASK_PARAMETERS:
            logger.warning("delegation provider %s: task parameter %r refused (reserved or invalid name)",
                           provider.name, name)
            continue
        if isinstance(schema, Mapping):
            accepted[name] = json.loads(json.dumps(schema))  # plain, detached copy
    return accepted


def schema_extension(config: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    """``(extra task properties, description addendum)`` for the active provider; empty for the
    built-in default so its schema stays byte-identical."""
    from agent.delegation_registry import DEFAULT_BACKEND
    provider = _active_provider(config)
    if provider.name == DEFAULT_BACKEND:
        return {}, ""
    try:
        addendum = provider.description(config)
    except Exception:  # plugin boundary: a broken provider costs its guidance, never the delegate_task tool
        logger.warning("delegation provider %s.description() raised", provider.name, exc_info=True)
        addendum = ""
    return _declared_parameters(provider, config), addendum.strip() if isinstance(addendum, str) else ""


def _param_error(index: int, name: str, value: Any, schema: Mapping[str, Any]) -> Optional[str]:
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"Task {index}: '{name}' must be one of {', '.join(map(str, enum))} (got {value!r})."
    return None


def _clean_route(provider_name: str, route: Any) -> dict[str, Any]:
    """The route keys of a provider's answer. Other keys are dropped silently: returning the whole
    ``delegation`` section (``dict(config)``, the default route) legitimately carries lifecycle knobs."""
    if not isinstance(route, Mapping):
        raise DelegationRouteError(f"delegation backend '{provider_name}' returned no route.")
    return {k: v for k, v in route.items() if k in ROUTE_KEYS}


def _profile_home(name: Any) -> Path:
    from hermes_cli.profiles import get_profile_dir, profile_exists
    if not isinstance(name, str) or not name.strip() or not profile_exists(name):
        raise DelegationRouteError(f"Profile {name!r} does not exist (see `hermes profile list`).")
    return Path(get_profile_dir(name))


@contextlib.contextmanager
def delegated_profile_scope(home: Any) -> Iterator[None]:
    """Run the block as profile ``home``: HERMES_HOME override, that profile's secret scope,
    multiplex semantics when it is not this process's own profile, and its terminal policy - the
    binding a multiplexed gateway applies to a routed turn (gateway/AGENTS.md § Profile scope).
    A no-op for anything but a path (``None``: no profile route) and for the home already in effect."""
    from hermes_constants import get_hermes_home, get_routing_process_hermes_home, hermes_home_key
    if not isinstance(home, (str, Path)) or hermes_home_key(home) == hermes_home_key(get_hermes_home()):
        yield
        return
    from agent.secret_scope import (
        reset_multiplex_context, reset_secret_scope, set_multiplex_context, set_secret_scope,
    )
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
    from tui_gateway.launch_profile_policy import owner_secret_scope

    routed = hermes_home_key(home) != hermes_home_key(get_routing_process_hermes_home())
    hydrate_profile_secret_sources(home)
    home_token = set_hermes_home_override(str(home))
    secret_token = set_secret_scope(owner_secret_scope(home), profile_home=str(home))
    mux_token = set_multiplex_context(True) if routed else None
    terminal_token = install_profile_terminal_scope(home)
    try:
        yield
    finally:
        reset_terminal_scope(terminal_token)
        if mux_token is not None:
            reset_multiplex_context(mux_token)
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def child_profile_scope(child: Any) -> contextlib.AbstractContextManager:
    """The profile scope a built child's own work runs in (its startup credential lease, every conversation turn
    incl. the output_schema retry); a no-op for children without a profile route. Parent bookkeeping (registry,
    heartbeat, progress, result assembly) stays outside it."""
    return delegated_profile_scope(getattr(child, "_delegate_profile_home", None))


def _profile_route(home: Path, label: Optional[str]) -> TaskRoute:
    """The profile's own primary route, resolved inside its scope (its config.yaml + secrets). The
    child never borrows the launching profile's key, pool or fallback chain."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from tools.delegate_tool_config import _credential_bundle
    with delegated_profile_scope(home):
        cfg = load_config_readonly() or {}
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            model = model_cfg.get("default") or model_cfg.get("model")
        else:
            model = model_cfg if isinstance(model_cfg, str) else None
        try:
            rt = resolve_runtime_provider(target_model=model or None)
        except Exception as exc:
            raise DelegationRouteError(f"Profile at {home} has no usable model provider: {exc}") from exc
        agent_cfg = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}
        fallback = cfg.get("fallback_providers")
    creds = _credential_bundle(
        model or rt.get("model") or None, rt.get("provider"), rt.get("base_url"),
        rt.get("api_key") or "no-key-required", rt.get("api_mode"), dict(rt.get("request_overrides") or {}),
        command=rt.get("command"), args=list(rt.get("args") or []),
    )
    return TaskRoute(creds, {"fallback_providers": fallback if isinstance(fallback, list) else []},
                     agent_cfg.get("reasoning_effort"), home, label)


def _route_to_task_route(route: dict[str, Any], parent_agent: Any, cache: dict[str, TaskRoute]) -> TaskRoute:
    key = json.dumps(route, sort_keys=True, default=str)
    if key in cache:
        return cache[key]
    label = route.get("label") if isinstance(route.get("label"), str) else None
    if route.get("profile"):
        resolved = _profile_route(_profile_home(route["profile"]), label)
    else:
        from tools.delegate_tool import _resolve_delegation_credentials  # the facade is the patch seam
        resolved = TaskRoute(_resolve_delegation_credentials(route, parent_agent), route,
                             route.get("reasoning_effort"), None, label)
    cache[key] = resolved
    return resolved


def _route_one(provider: DelegationProvider, declared: dict[str, dict[str, Any]], index: int,
               task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    params = {name: task[name] for name in declared if task.get(name) is not None}
    for name, value in params.items():
        error = _param_error(index, name, value, declared[name])
        if error:
            raise DelegationRouteError(error)
    payload = {"goal": task.get("goal"), "context": task.get("context"), "index": index, **params}
    try:
        return _clean_route(provider.name, provider.route_task(payload, config))
    except DelegationRouteError as exc:
        raise DelegationRouteError(f"Task {index}: {exc}") from exc
    except Exception as exc:
        logger.warning("delegation provider %s.route_task() raised", provider.name, exc_info=True)
        raise DelegationRouteError(f"Task {index}: delegation backend '{provider.name}' failed: {exc}") from exc


def resolve_task_routes(task_list: list[dict[str, Any]], config: Mapping[str, Any], parent_agent: Any,
                        ) -> tuple[list[TaskRoute], Optional[str]]:
    """One :class:`TaskRoute` per task from the active provider, or ``([], error)``. Identical
    routes resolve credentials once per call (the default provider resolves exactly once, as
    before the seam)."""
    provider = _active_provider(config)
    declared = _declared_parameters(provider, config)
    cache: dict[str, TaskRoute] = {}
    routes: list[TaskRoute] = []
    try:
        for i, task in enumerate(task_list):
            routes.append(_route_to_task_route(_route_one(provider, declared, i, task, config), parent_agent, cache))
    except ValueError as exc:  # DelegationRouteError and the credential resolver's refusals
        return [], str(exc)
    return routes, None


def route_for_config(routing_cfg: dict[str, Any], parent_agent: Any) -> TaskRoute:
    """The route an internal caller's explicit config block describes (``/review`` ->
    ``auxiliary.review``); bypasses the provider, as those callers own their route."""
    from tools.delegate_tool import _resolve_delegation_credentials  # the facade is the patch seam
    return TaskRoute(_resolve_delegation_credentials(routing_cfg, parent_agent), routing_cfg)
