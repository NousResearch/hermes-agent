"""Code-owned typed pivot registry for companyintel.

The registry owns automatic fact-to-pivot expansion. Workers remain separate
and are selected by the ``worker`` field; this module performs no network I/O.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PivotSpec:
    pivot_type: str
    worker: str
    priority: int
    budget_cost: int
    max_attempts: int = 3
    source_class: str = "public_search"


_REGISTRY: dict[str, tuple[PivotSpec, ...]] = {
    "domain": (
        PivotSpec("site_inventory", "inventory", 100, 1),
        PivotSpec("exact_search", "public_search", 80, 1),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "url": (
        PivotSpec("exact_search", "public_search", 80, 1),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "phone": (
        PivotSpec("exact_search", "public_search", 95, 1),
        PivotSpec("maps", "maps_search", 90, 2),
        PivotSpec("marketplaces", "marketplace_search", 85, 2),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "email": (
        PivotSpec("exact_search", "public_search", 92, 1),
        PivotSpec("marketplaces", "marketplace_search", 80, 2),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "address": (
        PivotSpec("exact_search", "public_search", 88, 1),
        PivotSpec("maps", "maps_search", 90, 2),
        PivotSpec("documents", "document_search", 75, 1),
    ),
    "brand": (
        PivotSpec("exact_search", "public_search", 78, 1),
        PivotSpec("marketplaces", "marketplace_search", 72, 2),
    ),
}


def expand_pivots(node_type: str, _value: str = "") -> tuple[PivotSpec, ...]:
    """Return deterministic pivot specs for a normalized node type."""
    return _REGISTRY.get(node_type, (PivotSpec("exact_search", "public_search", 70, 1),))


def get_pivot(pivot_type: str) -> PivotSpec | None:
    for specs in _REGISTRY.values():
        for spec in specs:
            if spec.pivot_type == pivot_type:
                return spec
    return None
