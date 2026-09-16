from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ConstraintViolation(RuntimeError):
    pass


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
