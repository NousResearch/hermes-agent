"""System-One decision lane substrate for computer use (#113850).

Reframes bounded GUI steps from generation into typed decisions so cheap
backends can answer without a frontier-model round trip. Backend order (``ladder="default"``):

    deterministic rules -> local semantic reranker -> fast aux model
    -> Jev (only when TYPESAFE_API_KEY / JEV_API_KEY is set) -> abstain (fail open)

``ladder="jev_first"`` (RFC #112639 / jev-ultrafast demos) prefers Jev when a key
is configured:

    rules -> Jev -> reranker -> aux -> abstain

Rules, reranker, aux, and Jev stages ship in ``decision_stages``; callers may
override any stage via the same ``(state, candidates) -> Decision | None`` shape
(None = abstain).
A stage that cannot run must abstain, never guess: the caller falls back to
the current planner. Approval and safety paths are untouched.

Decisions are recorded as replayable packets (state refs, candidates,
scores, chosen action, latency, verifier outcome). Packets never carry
screenshots or secrets — only element refs and labels the caller already had.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

ActionKind = Literal["click", "type", "key", "scroll", "wait", "done", "escalate"]
LadderMode = Literal["default", "jev_first"]

ACTION_KINDS: tuple[str, ...] = ("click", "type", "key", "scroll", "wait", "done", "escalate")

CONFIDENCE_THRESHOLD = 0.75

_JEV_KEY_ENV_VARS = ("TYPESAFE_API_KEY", "JEV_API_KEY")


@dataclass(frozen=True)
class ElementCandidate:
    """One currently visible semantic element. Refs and labels only."""

    ref: str
    label: str = ""
    role: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class SemanticState:
    """Minimal screen summary the lane reasons over. No pixels, no secrets."""

    elements: tuple[ElementCandidate, ...] = ()
    busy: bool = False
    goal_hint: str = ""


@dataclass(frozen=True)
class Decision:
    action: str
    target_ref: str | None = None
    needs_vision: bool = False
    needs_generation: bool = False
    done: bool = False
    confidence: float = 0.0
    backend: str = "rules"

    def __post_init__(self) -> None:
        if self.action not in ACTION_KINDS:
            raise ValueError(f"unknown action: {self.action!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence!r}")


@dataclass(frozen=True)
class DecisionPacket:
    """Replayable record of one lane run. Serializable; A/B-able across backends."""

    state_summary: dict[str, Any]
    candidates: list[dict[str, Any]]
    scores: dict[str, float]
    chosen_action: dict[str, Any] | None
    chosen_backend: str | None
    latency_s: float
    verifier_outcome: str  # "accept", "veto", or "skipped"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionPacket":
        return cls(**data)


Stage = Callable[[SemanticState, tuple[ElementCandidate, ...]], "Decision | None"]
Verifier = Callable[[SemanticState, "Decision"], bool]


def rules_stage(state: SemanticState, candidates: tuple[ElementCandidate, ...]) -> Decision | None:
    """Deterministic first stage. Abstains unless exactly one rule fires."""
    live = [c for c in candidates if c.enabled]
    if not live:
        return Decision(action="escalate", confidence=1.0, backend="rules")
    if state.busy:
        return Decision(action="wait", confidence=0.9, backend="rules")
    hint = (state.goal_hint or "").strip().lower()
    if hint:
        exact = [c for c in live if c.label.strip().lower() == hint]
        if len(exact) == 1:
            return Decision(
                action="click", target_ref=exact[0].ref, confidence=0.9, backend="rules")
    return None


def jev_available() -> bool:
    """True when a Jev call could be attempted (key present)."""
    return any(os.environ.get(name, "").strip() for name in _JEV_KEY_ENV_VARS)


def resolve_ladder_mode() -> LadderMode:
    """Resolve System-One backend order from ``HERMES_CU_SYSTEM_ONE_LADDER``."""
    raw = os.environ.get("HERMES_CU_SYSTEM_ONE_LADDER", "").strip().lower()
    if raw == "jev_first":
        return "jev_first"
    return "default"


def _stage_order(
    *,
    ladder: LadderMode,
    reranker: Stage | None,
    aux: Stage | None,
    jev: Stage | None,
) -> list[tuple[str, Stage | None, bool]]:
    jev_enabled = jev_available() and jev is not None
    if ladder == "jev_first" and jev_enabled:
        return [
            ("rules", rules_stage, True),
            ("jev", jev, True),
            ("reranker", reranker, reranker is not None),
            ("aux", aux, aux is not None),
        ]
    return [
        ("rules", rules_stage, True),
        ("reranker", reranker, reranker is not None),
        ("aux", aux, aux is not None),
        ("jev", jev, jev_enabled),
    ]


def default_verifier(state: SemanticState, decision: Decision) -> bool:
    """Reject decisions whose target ref is missing or incompatible with the action."""
    if decision.action in {"wait", "done", "escalate", "scroll", "key"}:
        return True
    if not decision.target_ref:
        return False
    live = {c.ref: c for c in state.elements if c.enabled}
    cand = live.get(decision.target_ref)
    if cand is None:
        return False
    role = (cand.role or "").lower()
    if decision.action == "type":
        return any(tok in role for tok in ("text", "search", "area", "field"))
    if decision.action == "click":
        return True
    return True


def run_decision_lane(
    state: SemanticState,
    candidates: tuple[ElementCandidate, ...] | list[ElementCandidate],
    *,
    reranker: Stage | None = None,
    aux: Stage | None = None,
    jev: Stage | None = None,
    verifier: Verifier | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ladder: LadderMode | None = None,
) -> tuple[Decision | None, DecisionPacket]:
    """Run the backend order; return (decision, packet). None = fail open.

    The Jev stage runs only when ``jev_available()`` and a ``jev`` callable
    is injected. Use ``ladder="jev_first"`` (or env) for jev-ultrafast demos.
    """
    cands = tuple(candidates)
    started = time.monotonic()
    scores: dict[str, float] = {}
    mode = ladder or resolve_ladder_mode()
    stages = _stage_order(ladder=mode, reranker=reranker, aux=aux, jev=jev)
    chosen: Decision | None = None
    chosen_backend: str | None = None
    for name, stage, enabled in stages:
        if not enabled or stage is None:
            continue
        try:
            decision = stage(state, cands)
        except Exception:
            continue  # a broken stage abstains, never blocks the lane
        if decision is None:
            continue
        scores[name] = decision.confidence
        if decision.confidence >= confidence_threshold:
            chosen, chosen_backend = decision, name
            break
    verdict = "skipped"
    check = verifier or default_verifier
    if chosen is not None:
        verify_state = state if state.elements else SemanticState(
            elements=cands, busy=state.busy, goal_hint=state.goal_hint)
        try:
            ok = check(verify_state, chosen)
        except Exception:
            ok = False
        verdict = "accept" if ok else "veto"
        if not ok:
            chosen, chosen_backend = None, None
    packet = DecisionPacket(
        state_summary={
            "busy": state.busy,
            "goal_hint": state.goal_hint,
            "element_count": len(cands),
        },
        candidates=[{"ref": c.ref, "label": c.label, "role": c.role} for c in cands],
        scores=scores,
        chosen_action={
            "action": chosen.action,
            "target_ref": chosen.target_ref,
            "needs_vision": chosen.needs_vision,
            "needs_generation": chosen.needs_generation,
            "done": chosen.done,
            "confidence": chosen.confidence,
            "backend": chosen.backend,
        } if chosen is not None else None,
        chosen_backend=chosen_backend,
        latency_s=time.monotonic() - started,
        verifier_outcome=verdict,
    )
    return chosen, packet
