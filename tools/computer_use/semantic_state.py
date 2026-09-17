"""Shadow semantic GUI state (Phase 1A of #112734, parent RFC #112639).

Builds an internal ``GuiStateV0`` from one capture's parsed AX/UIA/AT-SPI elements and
nothing else. Shadow mode: the state is never inserted into the prompt, never replaces a
capture, never authorizes an action, and creates no persistent semantic IDs — it exists only
so the diff experiment can measure element-identity stability and reconciliation cost.

The cua-driver element token stays snapshot-scoped on purpose (section E of the issue):
``SemanticElement.token`` records the current revision's binding only and is excluded from
the identity evidence key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import UIElement

# Geometry grid: bounds are bucketed into cells so 1-2px jitter between captures does not
# churn identity. A 32-cell grid keeps a button-sized element (~100px on a 1024px capture)
# in ~3 cells — coarse enough to absorb jitter, fine enough to separate siblings.
_GEOM_GRID = 32


@dataclass(frozen=True)
class SemanticElement:
    """One element's semantic evidence. Identity evidence = role, name, rel_geom, parent,
    app/window; ``token`` and raw bounds are revision-local facts, not identity."""

    role: str
    name: str
    rel_geom: Tuple[int, int, int, int]  # grid-bucketed (cx, cy, cw, ch)
    parent: str  # evidence key of the smallest containing element, "" for top-level
    app: str
    window_id: int
    token: Optional[str] = None  # snapshot-scoped driver token; never part of identity
    raw_bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
    flags: Tuple[Tuple[str, str], ...] = ()  # sorted (key, str(value)) state flags

    def evidence_key(self) -> Tuple[Any, ...]:
        return (self.role, self.name, self.rel_geom, self.parent, self.app, self.window_id)


@dataclass(frozen=True)
class GuiStateV0:
    """Internal semantic snapshot of one capture. ``revision`` is a per-session monotonic
    counter owned by the shadow observer, not a driver fact."""

    revision: int
    target: str  # app/window identity the elements were captured for
    elements: Tuple[SemanticElement, ...] = ()
    captured_at: float = 0.0


def _grid_geom(bounds: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in bounds)
    if width > 0 and height > 0:
        return (x * _GEOM_GRID // width, y * _GEOM_GRID // height,
                w * _GEOM_GRID // width, h * _GEOM_GRID // height)
    return (x // 64, y // 64, w // 64, h // 64)  # degenerate dims: coarse pixel buckets


def _contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ow > 0 and oh > 0 and ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih


def _state_flags(attributes: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    # Only boolean-ish interaction state survives; free-text values are identity noise.
    flags = []
    for key in ("focused", "selected", "enabled", "checked", "expanded"):
        value = attributes.get(key)
        if isinstance(value, bool):
            flags.append((key, str(value)))
    return tuple(sorted(flags))


def build_state(elements: List[UIElement], *, revision: int, target: str,
                width: int, height: int, captured_at: float = 0.0) -> GuiStateV0:
    """Project parsed driver elements into a shadow ``GuiStateV0``. Pure function."""
    raws = []
    for e in elements:
        bounds = tuple(int(v) for v in (e.bounds or (0, 0, 0, 0)))
        attrs = e.attributes if isinstance(e.attributes, dict) else {}
        raws.append((e, bounds, _grid_geom(bounds, width, height), _state_flags(attrs)))
    # Parent = smallest strictly-containing element (deterministic; ties broken by role/name).
    parents: List[str] = []
    for i, (e, bounds, _, _) in enumerate(raws):
        best: Optional[Tuple[Tuple[Any, ...], str]] = None
        for j, (other, obounds, _, _) in enumerate(raws):
            if i == j or not _contains(obounds, bounds) or obounds == bounds:
                continue
            area = obounds[2] * obounds[3]
            key = (area, other.role or "", other.label or "")
            if best is None or key < best[0]:
                best = (key, f"{other.role}|{other.label}|{other.app}|{other.window_id}")
        parents.append(best[1] if best else "")
    return GuiStateV0(
        revision=revision, target=target, captured_at=captured_at,
        elements=tuple(
            SemanticElement(
                role=e.role or "", name=e.label or "", rel_geom=geom, parent=parents[i],
                app=e.app or "", window_id=int(e.window_id or 0),
                token=e.element_token, raw_bounds=bounds, flags=flags,
            )
            for i, (e, bounds, geom, flags) in enumerate(raws)
        ),
    )
