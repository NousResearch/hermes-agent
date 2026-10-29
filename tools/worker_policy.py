"""Pluggable delegation worker policy host. Core ships no named routes: default-worker-only.

A plugin registers one policy with ``PluginContext.register_worker_policy(policy_id, policy)``.
The host calls it BEFORE any credential resolution, provider meter read or child construction,
gives it only a frozen snapshot of the parent's provenance (never the live agent), pins the
admitting policy on every child it configures, and runs each child inside that policy's lease
(``run_child``) so acquisition, cancellation, the whole conversation and release share one owner.

Fail-closed fallback: when no policy is loaded, more than one is loaded, or the loaded policy
raises anything other than a policy refusal (``ValueError``), admission is default-worker-only:
any per-task ``worker``/``effort``/``work_class`` selector is refused and children use the
configured delegation route unchanged. A missing or broken policy can never widen a route.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_ONLY_REFUSAL = ("Worker refused: no worker policy is available; use the configured default worker "
                        "(omit worker, effort and work_class).")
SELECTOR_KEYS = ("worker", "effort", "work_class")
REQUIRED_METHODS = ("admit", "task_route", "validate_credentials", "child_routing", "configure_child",
                    "run_child", "refresh_effort", "parent_authorized", "task_schema")


@dataclass(frozen=True)
class SessionDB:
    db_path: Any


@dataclass(frozen=True)
class ParentOrigin:
    """Frozen provenance snapshot of a delegating parent. Attribute names mirror the agent's."""
    _delegate_depth: int
    session_id: Optional[str]
    _session_db: Optional[SessionDB]
    provider: Optional[str]
    model: Optional[str]
    base_url: Optional[str]
    reasoning_config: Any


def parent_origin(agent) -> Optional[ParentOrigin]:
    if agent is None:
        return None
    if isinstance(agent, ParentOrigin):
        return agent
    db = getattr(getattr(agent, "_session_db", None), "db_path", None)
    depth = getattr(agent, "_delegate_depth", 0)
    return ParentOrigin(
        _delegate_depth=depth if isinstance(depth, int) and not isinstance(depth, bool) else 1,
        session_id=getattr(agent, "session_id", None),
        _session_db=SessionDB(db) if db is not None else None,
        provider=getattr(agent, "provider", None), model=getattr(agent, "model", None),
        base_url=getattr(agent, "base_url", None),
        reasoning_config=copy.deepcopy(getattr(agent, "reasoning_config", None)))


def _registered(slot):
    """``(provider_id, provider)`` pairs whose owning plugin is currently loaded without error."""
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    out = []
    for key, (owner, provider) in sorted(getattr(manager, slot, {}).items()):
        info = manager._plugins.get(owner)
        if info is not None and getattr(info, "enabled", True) and not getattr(info, "error", None):
            out.append((key, provider))
    return out


def _policy():
    """The single loaded registered policy, else None (default-worker-only)."""
    try:
        live = [policy for _key, policy in _registered("_worker_policies")]
        if len(live) > 1:
            logger.warning("worker policy refused: %d policies registered; default-worker-only", len(live))
            return None
        return live[0] if live else None
    except Exception:
        logger.debug("worker policy lookup failed; default-worker-only", exc_info=True)
        return None


def _default_only(tasks) -> None:
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if any(k in task for k in ("effort", "work_class")) or task.get("worker") not in (None, "default"):
            raise ValueError(DEFAULT_ONLY_REFUSAL)


def admit(tasks, routing_cfg, delegation_cfg, *, parent_agent=None, home=None):
    """Whole-batch admission before credentials. Returns ``(policy, routing_cfg, tasks, pool_used)``.

    ``ValueError`` from the policy is its refusal and propagates; any other policy failure degrades
    to default-worker-only for this batch.
    """
    policy = _policy()
    if policy is not None:
        try:
            routing, admitted, used = policy.admit(
                copy.deepcopy(tasks), copy.deepcopy(routing_cfg), copy.deepcopy(delegation_cfg),
                parent=parent_origin(parent_agent), home=home)
            return policy, routing, admitted, used
        except ValueError:
            raise
        except Exception:
            logger.warning("worker policy admission failed; default-worker-only", exc_info=True)
    _default_only(tasks)
    return None, routing_cfg, tasks, None


def task_route(policy, task, base_cfg):
    """Per-task credentials config BEFORE resolution; never endpoints/keys from task JSON."""
    if policy is None:
        return copy.deepcopy(base_cfg)
    return policy.task_route(copy.deepcopy(task), copy.deepcopy(base_cfg))


def validate_credentials(policy, task, credentials) -> None:
    if policy is not None:
        policy.validate_credentials(copy.deepcopy(task), copy.deepcopy(credentials))


def child_routing(policy, task, routing_cfg):
    if policy is None or not isinstance(routing_cfg, dict):
        return routing_cfg
    return policy.child_routing(copy.deepcopy(task), copy.deepcopy(routing_cfg))


def configure_child(policy, child, task, *, parent_agent=None, home=None) -> None:
    """Pin the admitting policy on the child (lease owner) and let it shape the child."""
    child._worker_policy = policy
    if policy is not None:
        policy.configure_child(child, copy.deepcopy(task), parent=parent_origin(parent_agent), home=home)


def _pinned(agent):
    """The policy pinned by configure_child (instance attribute only; never an auto-mock attribute)."""
    try:
        return vars(agent).get("_worker_policy")
    except TypeError:
        return None


def _named_lane(agent) -> bool:
    lane = getattr(agent, "_delegate_worker", None)
    return isinstance(lane, str) and lane not in ("", "default")


def run_child(child, **kwargs):
    """Run the child's conversation inside its admitting policy's lease.

    A named lane whose lease owner vanished is refused, never run unpermitted.
    """
    policy = _pinned(child)
    if policy is None:
        if _named_lane(child):
            raise RuntimeError("worker lease owner unavailable; named worker refused")
        return child.run_conversation(**kwargs)
    return policy.run_child(child, **kwargs)


def refresh_effort(agent) -> None:
    policy = _pinned(agent) or _policy()
    if policy is not None:
        policy.refresh_effort(agent)


def parent_authorized(agent, home=None) -> bool:
    """Policy-defined parent freedom; False without a healthy policy."""
    policy = _policy()
    if policy is None:
        return False
    try:
        return policy.parent_authorized(parent_origin(agent), home) is True
    except Exception:
        logger.debug("worker policy parent check failed closed", exc_info=True)
        return False


def task_schema_properties() -> dict:
    """Extra delegate_task task-item properties the policy advertises (empty without a policy)."""
    policy = _policy()
    if policy is None:
        return {}
    try:
        props = policy.task_schema()
        return copy.deepcopy(props) if isinstance(props, dict) else {}
    except Exception:
        return {}


def resolve_routes(policy, tasks, routing_cfg, delegation_cfg, parent_agent):
    """A policy owns default-backend routing; never bypass a separately configured backend."""
    from dataclasses import replace
    from tools.delegate_tool_routes import route_for_config
    if delegation_cfg.get("backend", "default") not in (None, "", "default"):
        raise ValueError("Worker policy requires the default delegation backend; configure one routing owner.")
    routes = []
    for task in tasks:
        selected = task_route(policy, task, routing_cfg)
        route = route_for_config(selected, parent_agent)
        validate_credentials(policy, task, route.creds)
        routes.append(replace(route, routing_cfg=child_routing(policy, task, route.routing_cfg)))
    return routes
