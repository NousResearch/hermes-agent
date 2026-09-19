"""Shadow delta between consecutive ``GuiStateV0`` snapshots (Phase 1A of #112734).

Matching is deterministic and conservative: an exact evidence-key hit binds at confidence
1.0; otherwise a candidate needs the same role AND name, and when several old elements
qualify the match is recorded as ``Ambiguity`` and NOT bound (section E of the issue:
surface ambiguity, never silently mis-bind). The driver token is expected to change every
revision, so token rebinds are not reported as changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tools.computer_use.semantic_state import GuiStateV0, SemanticElement

# Fields compared when a matched pair is checked for changes (token excluded: it is
# snapshot-scoped by design, so a rebind across revisions is normal, not a change).
_COMPARE_FIELDS = ("name", "rel_geom", "parent", "flags")


@dataclass(frozen=True)
class ElementChange:
    element: SemanticElement  # the NEW revision's element
    changes: Tuple[Tuple[str, Tuple[str, str]], ...]  # ((field, (old, new)), ...)
    confidence: float


@dataclass(frozen=True)
class Ambiguity:
    """A new element with several plausible old identities — surfaced, never bound."""
    element: SemanticElement
    candidate_keys: Tuple[Tuple[str, str, str], ...]  # (role, name, parent) per candidate
    reason: str


@dataclass(frozen=True)
class StateDelta:
    old_revision: int
    new_revision: int
    added: Tuple[SemanticElement, ...] = ()
    removed: Tuple[SemanticElement, ...] = ()
    changed: Tuple[ElementChange, ...] = ()
    matched: int = 0
    ambiguous: Tuple[Ambiguity, ...] = ()
    mean_confidence: float = 1.0

    @property
    def changed_element_ratio(self) -> float:
        denom = max(len(self.added) + self.matched + len(self.removed), 1)
        return (len(self.added) + len(self.removed) + len(self.changed)) / denom

    @property
    def identity_retention(self) -> float:
        denom = self.matched + len(self.removed)
        return self.matched / denom if denom else 1.0


def _field_str(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def diff_states(old: GuiStateV0, new: GuiStateV0) -> StateDelta:
    """Diff two consecutive shadow states. Deterministic; never raises on odd input."""
    old_by_key: Dict[Tuple[object, ...], List[SemanticElement]] = {}
    for e in old.elements:
        old_by_key.setdefault(e.evidence_key(), []).append(e)

    matched_pairs: List[Tuple[SemanticElement, SemanticElement, float]] = []
    added: List[SemanticElement] = []
    ambiguous: List[Ambiguity] = []
    used_old: List[bool] = [False] * len(old.elements)
    old_index = {id(e): i for i, e in enumerate(old.elements)}

    for ne in new.elements:
        candidates = [oe for oe in old_by_key.get(ne.evidence_key(), []) if not used_old[old_index[id(oe)]]]
        if len(candidates) == 1:
            used_old[old_index[id(candidates[0])]] = True
            matched_pairs.append((candidates[0], ne, 1.0))
            continue
        if len(candidates) > 1:
            # Identical evidence keys (e.g. twin unnamed buttons): bind the first for
            # continuity but surface the ambiguity so nothing is silently mis-bound.
            used_old[old_index[id(candidates[0])]] = True
            matched_pairs.append((candidates[0], ne, 0.5))
            ambiguous.append(Ambiguity(
                element=ne,
                candidate_keys=tuple((c.role, c.name, c.parent) for c in candidates),
                reason=f"{len(candidates)} old elements share the evidence key",
            ))
            continue
        # Fuzzy pass: same role AND name required; parent or geometry must also agree.
        fuzzy = [oe for i, oe in enumerate(old.elements)
                 if not used_old[i] and oe.role == ne.role and oe.name == ne.name
                 and (oe.parent == ne.parent or oe.rel_geom == ne.rel_geom)]
        if len(fuzzy) == 1:
            oe = fuzzy[0]
            used_old[old_index[id(oe)]] = True
            confidence = 0.5 + 0.25 * (oe.parent == ne.parent) + 0.25 * (oe.rel_geom == ne.rel_geom)
            matched_pairs.append((oe, ne, confidence))
        elif len(fuzzy) > 1:
            ambiguous.append(Ambiguity(
                element=ne,
                candidate_keys=tuple((c.role, c.name, c.parent) for c in fuzzy),
                reason=f"{len(fuzzy)} fuzzy candidates share role+name",
            ))
            added.append(ne)  # unbound: counted as added, ambiguity recorded alongside
        else:
            added.append(ne)

    removed = [oe for i, oe in enumerate(old.elements) if not used_old[i]]

    changed: List[ElementChange] = []
    for oe, ne, confidence in matched_pairs:
        field_diffs = []
        for f in _COMPARE_FIELDS:
            ov, nv = getattr(oe, f), getattr(ne, f)
            if ov != nv:
                field_diffs.append((f, (_field_str(ov), _field_str(nv))))
        if field_diffs:
            changed.append(ElementChange(element=ne, changes=tuple(field_diffs), confidence=confidence))

    confidences = [c for _, _, c in matched_pairs]
    return StateDelta(
        old_revision=old.revision, new_revision=new.revision,
        added=tuple(added), removed=tuple(removed), changed=tuple(changed),
        matched=len(matched_pairs), ambiguous=tuple(ambiguous),
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else 1.0,
    )


def _state_json(state: GuiStateV0) -> str:
    return json.dumps({
        "revision": state.revision, "target": state.target,
        "elements": [list(e.evidence_key()) + [list(e.flags)] for e in state.elements],
    }, sort_keys=True, default=str)


def full_observation_bytes(state: GuiStateV0) -> int:
    """Byte size of the full shadow state — the denominator the delta is measured against."""
    return len(_state_json(state).encode("utf-8"))


def delta_bytes(delta: StateDelta) -> int:
    """Byte size of the delta projection — what a delta-based observation would carry."""
    return len(json.dumps({
        "from": delta.old_revision, "to": delta.new_revision,
        "added": [list(e.evidence_key()) for e in delta.added],
        "removed": [list(e.evidence_key()) for e in delta.removed],
        "changed": [(list(c.element.evidence_key()), list(c.changes)) for c in delta.changed],
        "ambiguous": len(delta.ambiguous),
    }, sort_keys=True, default=str).encode("utf-8"))
