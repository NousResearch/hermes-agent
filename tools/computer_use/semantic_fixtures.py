"""Deterministic semantic fixtures for #112734 §F / Phase 1A.

Each fixture is a small, fully deterministic capture sequence with a known state
transition and ground-truth element bindings — the reproducible microbenchmark the
issue asks for below OSWorld. Fixtures build synthetic AX trees (no Bot Screen, no
driver), so they run in CI; the same transition shapes drive a real Bot Desktop
later.

Ground truth is explicit: ``expected_binding`` maps each NEW element id to the OLD
element id it really is, or None when the matcher must leave it unbound. Cases the
issue calls out: a relabeled control (Render → Export, Working… → Complete) keeps
its identity across the name change; a canvas carries no semantic identity to
fabricate; a 5s render stall is a deterministic latency stand-in, not a wall-clock
sleep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from tools.computer_use.backend import UIElement

W, H = 800, 600


@dataclass(frozen=True)
class FixtureElement:
    """One capture element plus its stable fixture identity (``app/role/name``-style)."""
    fid: str
    element: UIElement


@dataclass(frozen=True)
class FixtureTransition:
    """One known before → after transition with ground-truth bindings."""
    name: str
    old: Tuple[FixtureElement, ...]
    new: Tuple[FixtureElement, ...]
    expected_binding: Mapping[str, Optional[str]]  # new fid -> old fid, or None
    expect_ambiguity: Tuple[str, ...] = ()  # new fids that must surface Ambiguity


@dataclass(frozen=True)
class SemanticFixture:
    name: str
    description: str
    transitions: Tuple[FixtureTransition, ...] = ()
    stall_ms: int = 0  # deterministic stall the transition models (issue §F example)


def _el(fid: str, index: int, role: str, label: str, bounds: Tuple[int, int, int, int],
        token: Optional[str] = None, app: str = "FixtureApp",
        window_id: int = 7, **flags) -> FixtureElement:
    return FixtureElement(
        fid=fid,
        element=UIElement(index=index, role=role, label=label, bounds=bounds, app=app,
                          pid=4242, window_id=window_id, attributes=dict(flags),
                          element_token=token),
    )


def _window(fid: str = "win") -> FixtureElement:
    return _el(fid, 1, "AXWindow", "Fixture", (0, 0, W, H), token="tok-win")


def form_fixture() -> SemanticFixture:
    """Settings form: focusing the search field changes flags, nothing else moves."""
    old = (
        _window("form:win"),
        _el("form:search", 2, "AXTextField", "Search settings", (100, 100, 300, 30), focused=False),
        _el("form:submit", 3, "AXButton", "Submit", (100, 150, 120, 30)),
        _el("form:sync", 4, "AXCheckBox", "Sync", (100, 200, 120, 20), checked=False),
    )
    new = (
        _window("form:win"),
        _el("form:search", 2, "AXTextField", "Search settings", (100, 100, 300, 30), focused=True),
        _el("form:submit", 3, "AXButton", "Submit", (100, 150, 120, 30)),
        _el("form:sync", 4, "AXCheckBox", "Sync", (100, 200, 120, 20), checked=False),
    )
    return SemanticFixture(
        name="form",
        description="form field focus flips; all identities stable",
        transitions=(FixtureTransition(
            name="focus-search",
            old=old, new=new,
            expected_binding={f.fid: f.fid for f in new},
        ),),
    )


def modal_fixture() -> SemanticFixture:
    """Dialog opens over the form, then dismisses: added/removed, nothing re-bound."""
    base = (
        _window("modal:win"),
        _el("modal:search", 2, "AXTextField", "Search settings", (100, 100, 300, 30)),
        _el("modal:submit", 3, "AXButton", "Submit", (100, 150, 120, 30)),
    )
    opened = base + (
        _el("modal:dialog", 4, "AXDialog", "Clear browsing data", (200, 150, 400, 200)),
        _el("modal:cancel", 5, "AXButton", "Cancel", (220, 300, 100, 30)),
        _el("modal:confirm", 6, "AXButton", "Clear data", (340, 300, 100, 30)),
    )
    return SemanticFixture(
        name="modal",
        description="dialog + buttons appear and dismiss",
        transitions=(
            FixtureTransition(
                name="dialog-opens",
                old=base, new=opened,
                expected_binding={**{f.fid: f.fid for f in base},
                                  "modal:dialog": None, "modal:cancel": None, "modal:confirm": None},
            ),
            FixtureTransition(
                name="dialog-dismisses",
                old=opened, new=base,
                expected_binding={f.fid: f.fid for f in base},
            ),
        ),
    )


def reordering_list_fixture() -> SemanticFixture:
    """Sortable list: items reorder by drag; identity follows the label, not geometry."""
    def items(order: Sequence[str]) -> Tuple[FixtureElement, ...]:
        return tuple(
            _el(f"list:{name}", i + 1, "AXStaticText", name, (50, 100 + i * 40, 200, 24))
            for i, name in enumerate(order)
        )
    old, new = items(("Alpha", "Beta", "Gamma")), items(("Gamma", "Alpha", "Beta"))
    return SemanticFixture(
        name="reordering-list",
        description="drag reorder; identity survives geometry change",
        transitions=(FixtureTransition(
            name="reorder",
            old=old, new=new,
            expected_binding={f.fid: f.fid for f in new},
        ),),
    )


def delayed_render_fixture() -> SemanticFixture:
    """Render button stalls deterministically, then relabels to Export with results.

    The 5s stall models the issue §F example. Ground truth treats Export as the same
    control relabeled (role + parent + geometry agree; only the name changed) — the
    matcher binds it at 0.5 and records the name change, which is exactly how the
    shadow experiment should represent "the button still exists but says Export now".
    The results list is absent until the final revision.
    """
    idle = (
        _window("render:win"),
        _el("render:button", 2, "AXButton", "Render", (100, 100, 120, 30)),
        _el("render:status", 3, "AXStaticText", "Idle", (100, 150, 200, 24)),
    )
    stalled = (
        _window("render:win"),
        _el("render:button", 2, "AXButton", "Render", (100, 100, 120, 30)),
        _el("render:status", 3, "AXStaticText", "Working…", (100, 150, 200, 24)),
    )
    done = (
        _window("render:win"),
        _el("render:export", 2, "AXButton", "Export", (100, 100, 120, 30)),
        _el("render:status", 3, "AXStaticText", "Complete", (100, 150, 200, 24)),
        _el("render:row1", 4, "AXStaticText", "result-1", (100, 200, 200, 24)),
        _el("render:row2", 5, "AXStaticText", "result-2", (100, 240, 200, 24)),
    )
    return SemanticFixture(
        name="delayed-render",
        description="deterministic stall, then control replacement + late content",
        stall_ms=5000,
        transitions=(
            FixtureTransition(
                name="stall-begins",
                old=idle, new=stalled,
                expected_binding={f.fid: f.fid for f in stalled},
            ),
            FixtureTransition(
                name="render-completes",
                old=stalled, new=done,
                expected_binding={
                    "render:win": "render:win",
                    "render:export": "render:button",  # same control relabeled
                    "render:status": "render:status",
                    "render:row1": None,  # late content
                    "render:row2": None,
                },
            ),
        ),
    )


def canvas_fixture() -> SemanticFixture:
    """Content-free canvas beside a labeled button.

    The canvas carries no name/label semantics; the matcher must not fabricate an
    identity from geometry alone. Ground truth binds the same canvas across a small
    geometry shift (role + app + parent agree) but never confuses it with the button.
    """
    old = (
        _window("canvas:win"),
        _el("canvas:draw", 2, "AXImage", "", (300, 100, 400, 300)),
        _el("canvas:save", 3, "AXButton", "Save", (100, 100, 120, 30)),
    )
    new = (
        _window("canvas:win"),
        _el("canvas:draw", 2, "AXImage", "", (302, 102, 400, 300)),  # 2px jitter
        _el("canvas:save", 3, "AXButton", "Save", (100, 100, 120, 30)),
    )
    return SemanticFixture(
        name="canvas",
        description="unlabeled canvas must not gain a fabricated identity",
        transitions=(FixtureTransition(
            name="jitter",
            old=old, new=new,
            expected_binding={f.fid: f.fid for f in new},
        ),),
    )


def all_fixtures() -> List[SemanticFixture]:
    return [form_fixture(), modal_fixture(), reordering_list_fixture(),
            delayed_render_fixture(), canvas_fixture()]


def element_list(els: Sequence[FixtureElement]) -> List[UIElement]:
    """Project a fixture state to the driver-element list ``build_state`` consumes."""
    return [f.element for f in els]


def fixture_ids(els: Sequence[FixtureElement]) -> Tuple[str, ...]:
    return tuple(f.fid for f in els)
