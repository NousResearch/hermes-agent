from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ConstraintViolation(RuntimeError):
    pass


# Explicit first-party surface, not a wildcard alias for third-party browser tools.
NATIVE_BROWSER_TOOLS = frozenset({
    'browser_navigate', 'browser_snapshot', 'browser_click', 'browser_type',
    'browser_press', 'browser_scroll', 'browser_extract_items', 'browser_console',
    'browser_get_images', 'browser_back', 'browser_close', 'browser_tab',
})


def canonical_route_for_tool(name: str, *, runtime: str | None = None) -> str:
    from tools.effects import tool_contract
    _, metadata = tool_contract(name)
    routes = metadata.get('routes') or []
    if name in NATIVE_BROWSER_TOOLS and runtime == 'internal':
        return 'native_browser'
    if runtime and runtime != 'internal':
        return runtime
    if len(routes) == 1:
        return routes[0]
    return f'tool.{name}'


def normalize_route_constraints(constraints):
    return {k: [canonical_route_for_tool(p.removeprefix('tool.'), runtime='internal')
                if p.removeprefix('tool.') in NATIVE_BROWSER_TOOLS else p for p in v]
            if k.endswith(('allowed_routes', 'forbidden_routes')) and isinstance(v, list) else v
            for k, v in constraints.items()}


def require_allowed_route(route: str, constraints: Mapping[str, Any]) -> None:
    """Prune prohibited routes before dispatch, discovery, or credential probes."""
    def matches(pattern: str) -> bool:
        return route == pattern or route.startswith(pattern + ".")
    forbidden = constraints.get("forbidden_routes", [])
    allowed = constraints.get("allowed_routes")
    for patterns in (forbidden, allowed):
        if patterns is not None and (not isinstance(patterns, list) or len(patterns) > 64
                or any(not isinstance(p, str) or not p or len(p) > 256 for p in patterns)):
            raise ConstraintViolation("Invalid bounded route constraints")
    if route == 'native_browser':
        normalized = normalize_route_constraints(constraints)
        forbidden = normalized.get('forbidden_routes', [])
        allowed = normalized.get('allowed_routes')
    if any(matches(p) for p in forbidden) or (
        allowed is not None and not any(matches(p) for p in allowed)
    ):
        raise ConstraintViolation(f"Route forbidden by task constraints: {route}")


class BrowserBackend(str, Enum):
    INTERNAL = "internal"
    AGENT_BROWSER = "agent-browser"
    BROWSER_EXEC = "browser-exec"
    LIGHTPANDA = "lightpanda"


@dataclass(frozen=True, slots=True)
class BrowserRoutingContext:
    requires_auth: bool = False
    requires_visible_state: bool = False
    public_read_only: bool = False
    heavy_adaptive_flow: bool = False
    headless_ok: bool = False
    bound_to_internal: bool = False


@dataclass(frozen=True, slots=True)
class BrowserRoutingPolicy:
    enabled: bool = True
    internal_only_when_disabled: bool = True

    def choose(self, ctx: BrowserRoutingContext) -> BrowserBackend:
        # Once bound, fail closed. Recovery is the internal runtime's job;
        # never silently move a task to a browser with different state.
        if ctx.bound_to_internal:
            return BrowserBackend.INTERNAL
        if not self.enabled and self.internal_only_when_disabled:
            return BrowserBackend.INTERNAL
        if ctx.requires_auth or ctx.requires_visible_state:
            return BrowserBackend.INTERNAL
        if ctx.heavy_adaptive_flow:
            return BrowserBackend.BROWSER_EXEC
        if ctx.public_read_only and ctx.headless_ok:
            return BrowserBackend.LIGHTPANDA
        if ctx.public_read_only:
            return BrowserBackend.AGENT_BROWSER
        return BrowserBackend.INTERNAL
