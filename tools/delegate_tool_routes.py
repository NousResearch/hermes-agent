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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from agent.delegation_provider import (
    RESERVED_TASK_PARAMETERS, ROUTE_KEYS, DelegationProvider, DelegationRouteError,
)

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

_PARAM_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass
class TaskRoute:
    """How core builds one child: the credential bundle (with a route's speed tier applied), the config block
    owning its fallback policy, an optional reasoning override, and the profile home it runs as (None = this one).
    ``error`` set = the backend could not route this task: it is refused alone and its siblings still run."""

    creds: dict[str, Any]
    routing_cfg: dict[str, Any]
    reasoning_effort: Any = None
    profile_home: Optional[Path] = None
    label: Optional[str] = None
    backend: str = "default"
    error: Optional[str] = None

    def report(self) -> Optional[dict[str, Any]]:
        """The ``route`` a backend-routed task's result carries (None for the default backend's unlabelled route)."""
        if self.backend == "default" and not self.label:
            return None
        return {"backend": self.backend, **({"label": self.label} if self.label else {})}

    def refused_entry(self, index: int) -> dict[str, Any]:
        """Result entry for a task this route refused: no child was built for it."""
        return {"task_index": index, "status": "error", "summary": None, "error": self.error, "api_calls": 0,
                "duration_seconds": 0, "route": {**(self.report() or {"backend": self.backend}), "refused": True}}


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
    """``(extra task properties, description addendum)`` for the active provider (none for the
    default strategy, whose schema is the base one)."""
    try:
        provider = _active_provider(config)
    except DelegationRouteError:
        return {}, ""  # fails closed at spawn time, with the fix; the schema just stays the base one
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
    # Each step is entered inside the stack, so one that raises unwinds the ones already bound and the
    # parent never stays in the child's profile (same discipline as the gateway's routed-turn scope).
    with contextlib.ExitStack() as stack:
        stack.callback(reset_hermes_home_override, set_hermes_home_override(str(home)))
        stack.callback(reset_secret_scope, set_secret_scope(owner_secret_scope(home), profile_home=str(home)))
        if routed:
            stack.callback(reset_multiplex_context, set_multiplex_context(True))
        stack.callback(reset_terminal_scope, install_profile_terminal_scope(home))
        yield


def child_profile_scope(child: Any) -> contextlib.AbstractContextManager:
    """The profile scope a built child's own work runs in (its startup credential lease, every conversation turn
    incl. the output_schema retry); a no-op for children without a profile route. Parent bookkeeping (registry,
    heartbeat, progress, result assembly) stays outside it."""
    return delegated_profile_scope(getattr(child, "_delegate_profile_home", None))


def _profile_model_request(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """``resolve_runtime_provider`` kwargs for a profile's primary model, decoded by the same owner CLI startup
    uses (``resolve_startup_model_route``): a ``model.default`` alias takes the alias's model, endpoint and key."""
    from hermes_cli.model_switch import resolve_startup_model_route
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        model = model_cfg.get("default") or model_cfg.get("model")
        cfg_provider = str(model_cfg.get("provider") or "").strip()
    else:
        model, cfg_provider = (model_cfg if isinstance(model_cfg, str) else None), ""
    request: dict[str, Any] = {"target_model": model or None}
    if not isinstance(model, str) or not model.strip():
        return request
    startup = resolve_startup_model_route(
        model, current_provider=cfg_provider.lower(), user_providers=cfg.get("providers"),
        custom_providers=cfg.get("custom_providers"), alias_config=dict(cfg))
    if startup is not None:
        request = {"target_model": startup.model, "requested": startup.provider or None,
                   "explicit_base_url": startup.base_url or None, "explicit_api_key": startup.api_key or None}
    return request


def _refuse_under_worker_snapshot(home: Path) -> None:
    """A managed worker (kanban, safe mode, managed sessions) runs on the session config frozen at creation:
    every config reader returns that snapshot and the process never opens profile files, so another profile's
    config.yaml is unreachable here. Refuse rather than pair the launch profile's model with the target's keys."""
    from agent.safe_worker_policy import worker_config_snapshot
    from hermes_constants import get_hermes_home, hermes_home_key
    if worker_config_snapshot() is not None and hermes_home_key(home) != hermes_home_key(get_hermes_home()):
        raise DelegationRouteError(
            f"Profile routes are unavailable in a managed worker session (its config is frozen at creation): "
            f"cannot run as the profile at {home}. Delegate without a profile.")


def _profile_route(home: Path) -> TaskRoute:
    """The profile's own primary route, resolved inside its scope (its config.yaml + secrets). The
    child never borrows the launching profile's key, pool, fallback chain or reasoning level."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_constants import resolve_reasoning_config
    from tools.delegate_tool_config import ResolvedReasoning, _credential_bundle, _require_pinned_command
    _refuse_under_worker_snapshot(home)
    with delegated_profile_scope(home):
        cfg = load_config_readonly() or {}
        request = _profile_model_request(cfg)
        try:
            rt = resolve_runtime_provider(**request)
        except Exception as exc:
            raise DelegationRouteError(f"Profile at {home} has no usable model provider: {exc}") from exc
        model = request["target_model"] or rt.get("model") or None
        fallback = cfg.get("fallback_providers")
    # A named custom provider resolves to provider "custom": keep its name as the identity (as delegation.provider
    # does) so two named providers sharing one URL never lease each other's pool.
    requested = str(rt.get("requested_provider") or "").strip()
    provider = requested if rt.get("provider") == "custom" and requested.lower() not in ("", "auto") else rt.get("provider")
    # Same rule as delegation.provider (#80450): a pinned ACP command that is not on PATH refuses the spawn.
    _require_pinned_command(
        rt.get("command"), f"Profile at {home} is pinned to the '{rt.get('command')}' command, which was not "
        f"found on PATH. Install it or route this task elsewhere.")
    creds = _credential_bundle(
        model, provider, rt.get("base_url"),
        rt.get("api_key") or "no-key-required", rt.get("api_mode"), dict(rt.get("request_overrides") or {}),
        command=rt.get("command"), args=list(rt.get("args") or []),
    )
    return TaskRoute(creds, {"fallback_providers": fallback if isinstance(fallback, list) else []},
                     ResolvedReasoning(resolve_reasoning_config(cfg, model or "")), home)


def _with_inherited_fallback(route: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """A route without a fallback chain of its own takes ``delegation.fallback_providers`` (an explicit ``[]`` on the
    route still disables fallback). A profile route keeps its profile's own chain."""
    if route.get("profile") or route.get("fallback_providers") is not None:
        return route
    inherited = config.get("fallback_providers")
    return route if inherited is None else {**route, "fallback_providers": inherited}


def _resolve_route(route: dict[str, Any], parent_agent: Any, backend: str) -> TaskRoute:
    if route.get("profile"):
        return replace(_profile_route(_profile_home(route["profile"])), backend=backend)
    from tools.delegate_tool import _resolve_delegation_credentials  # the facade is the patch seam
    creds = _resolve_delegation_credentials(route, parent_agent)
    if route.get("service_tier") not in (None, ""):
        from tools.delegate_tool_tuning import task_service_tier
        creds = {**creds, **task_service_tier(str(route["service_tier"]), creds, parent_agent)}
    return TaskRoute(creds, route, route.get("reasoning_effort"), None, None, backend)


def _route_to_task_route(route: dict[str, Any], parent_agent: Any, cache: dict[str, TaskRoute],
                         config: Optional[Mapping[str, Any]] = None, backend: str = "default") -> TaskRoute:
    label = route.get("label") if isinstance(route.get("label"), str) else None
    route = _with_inherited_fallback(route, config or {})
    # The label is display-only: two labels on one route (or one profile) resolve once.
    key = json.dumps({k: v for k, v in route.items() if k != "label"}, sort_keys=True, default=str)
    if key not in cache:
        cache[key] = _resolve_route(route, parent_agent, backend)
    return replace(cache[key], label=label)


def _check_backend_effort(index: int, route: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """A backend's ``reasoning_effort`` must be a known level: a typo is a tool error, never the parent's level.
    The ``delegation`` section's own value (the default route, ``dict(config)``) keeps its warn-and-inherit rule."""
    effort = route.get("reasoning_effort")
    if route.get("profile") or effort in (None, "") or effort == config.get("reasoning_effort"):
        return
    from hermes_constants import VALID_REASONING_EFFORTS, parse_reasoning_effort
    if parse_reasoning_effort(effort) is None:
        raise DelegationRouteError(
            f"Task {index}: unknown reasoning_effort {effort!r} on this route. "
            f"Use none or one of: {', '.join(VALID_REASONING_EFFORTS)}.")


def _undeclared_error(provider: DelegationProvider, index: int, names: list[str], config: Mapping[str, Any]) -> str:
    """Why a task carrying parameters the session's backend does not declare is refused (a model inventing them, or
    a session whose tools came from another build than its backend); dropping them would run the task on this
    backend's default route. ``/delegation <backend> --now`` or ``/new`` rebuilds the tools and the backend together."""
    from agent.delegation_registry import _registry
    active = provider.name
    listed = ", ".join(f"'{name}'" for name in names)
    owners = sorted({other.name for other in _registry.list_providers() if other.name != active
                     and set(names) & set(_declared_parameters(other, config))})
    head = f"Task {index}: the delegation backend is '{active}', which does not take {listed}"
    fix = (f"run `/delegation {active} --now` to rebuild this session's delegate_task for '{active}', "
           f"or start a new session with /new")
    if owners:
        quoted = ", ".join(f"'{owner}'" for owner in owners)
        return (f"{head} (declared by {quoted}), so this task was not started. This session's delegate_task was "
                f"built for another backend: {fix}.")
    return (f"{head}, and no installed delegation backend does, so this task was not started. Resend it without "
            f"{listed}; if this session's delegate_task tool offers {listed}, {fix}.")


def _route_one(provider: DelegationProvider, declared: dict[str, dict[str, Any]], index: int,
               task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    undeclared = sorted(name for name, value in task.items()
                        if value is not None and name not in declared and name not in RESERVED_TASK_PARAMETERS)
    if undeclared:
        raise DelegationRouteError(_undeclared_error(provider, index, undeclared, config))
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
    """One :class:`TaskRoute` per task from the backend the session's ``delegate_task`` was built from (else
    ``delegation.backend``), or ``([], error)`` when that backend cannot run (fails closed, never another backend).
    A task the backend cannot route gets a route with ``error`` set; its siblings are unaffected. Identical routes
    resolve credentials once per call (the default backend resolves exactly once)."""
    from agent.delegation_backends import pinned_backend
    if pinned := pinned_backend(parent_agent):
        config = {**config, "backend": pinned}
    try:
        provider = _active_provider(config)
    except DelegationRouteError as exc:
        return [], str(exc)
    declared = _declared_parameters(provider, config)
    cache: dict[str, TaskRoute] = {}
    routes: list[TaskRoute] = []
    for i, task in enumerate(task_list):
        try:
            route = _route_one(provider, declared, i, task, config)
            _check_backend_effort(i, route, config)
            routes.append(_route_to_task_route(route, parent_agent, cache, config, provider.name))
        except ValueError as exc:  # DelegationRouteError, the credential resolver's and the tier check's refusals
            routes.append(TaskRoute({}, {}, backend=provider.name, error=str(exc)))
    return routes, None


def route_for_config(routing_cfg: dict[str, Any], parent_agent: Any) -> TaskRoute:
    """The route an internal caller's explicit config block describes (``/review`` ->
    ``auxiliary.review``); bypasses the provider, as those callers own their route."""
    from tools.delegate_tool import _resolve_delegation_credentials  # the facade is the patch seam
    return TaskRoute(_resolve_delegation_credentials(routing_cfg, parent_agent), routing_cfg)
