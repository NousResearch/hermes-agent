"""Incorrect semantic-match measurement against fixture ground truth (#112734 §J).

``diff_states`` decides bindings; fixtures know the right answers. This module audits
one against the other and reports the "incorrect semantic match rate" metric from the
issue's metrics table. Wrong bindings, missed bindings, and bound-but-should-be-added
elements all count as incorrect; a surfaced Ambiguity where the ground truth expects
one counts as correct handling, not a match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from tools.computer_use.semantic_state import GuiStateV0, SemanticElement, build_state
from tools.computer_use.state_diff import StateDelta, diff_states
from tools.computer_use.semantic_fixtures import (
    FixtureTransition, SemanticFixture, W, H, element_list, fixture_ids,
)


@dataclass(frozen=True)
class MatchQuality:
    """Audit of one fixture transition: bindings vs ground truth."""
    fixture: str
    transition: str
    total: int  # new elements audited
    correct: int
    incorrect: int
    incorrect_details: Tuple[str, ...] = ()
    ambiguity_expected: int = 0
    ambiguity_surfaced: int = 0

    @property
    def incorrect_rate(self) -> float:
        return self.incorrect / self.total if self.total else 0.0


def _fid_map(state: GuiStateV0, fids: Sequence[str]) -> Dict[SemanticElement, str]:
    # build_state preserves input order, so position aligns elements to fixture ids.
    assert len(state.elements) == len(fids), "fixture ids must align with built elements"
    return dict(zip(state.elements, fids))


def measure_transition(fixture: SemanticFixture, transition: FixtureTransition,
                       rev_old: int = 1, rev_new: int = 2) -> MatchQuality:
    """Run one known transition through build + diff and audit every binding."""
    target = f"FixtureApp/{fixture.name}"
    old_state = build_state(element_list(transition.old), revision=rev_old, target=target,
                            width=W, height=H)
    new_state = build_state(element_list(transition.new), revision=rev_new, target=target,
                            width=W, height=H)
    delta = diff_states(old_state, new_state)
    return audit_delta(fixture.name, transition.name, delta,
                       old_state, new_state,
                       fixture_ids(transition.old), fixture_ids(transition.new),
                       transition.expected_binding, transition.expect_ambiguity)


def audit_delta(fixture: str, transition: str, delta: StateDelta,
                old_state: GuiStateV0, new_state: GuiStateV0,
                old_fids: Sequence[str], new_fids: Sequence[str],
                expected_binding: Mapping[str, Optional[str]],
                expect_ambiguity: Sequence[str] = ()) -> MatchQuality:
    """Audit a ``StateDelta`` against ground truth bindings.

    ``expected_binding`` maps each new fid to the old fid it truly is, or None when the
    matcher must leave it unbound. Every new element is exactly one of: correctly bound,
    incorrectly bound, correctly unbound, incorrectly unbound.
    """
    old_by_el = _fid_map(old_state, old_fids)
    new_by_el = _fid_map(new_state, new_fids)
    bound_new: Dict[SemanticElement, SemanticElement] = {ne: oe for oe, ne, _ in delta.matched_pairs}
    ambiguous_fids = {new_by_el[a.element] for a in delta.ambiguous if a.element in new_by_el}

    correct, incorrect = 0, 0
    details: List[str] = []
    for ne in new_state.elements:
        new_fid = new_by_el[ne]
        expected_old = expected_binding.get(new_fid)
        actual_old = bound_new.get(ne)
        if expected_old is None:
            if actual_old is None:
                correct += 1
            else:
                incorrect += 1
                details.append(f"{new_fid}: bound to {old_by_el.get(actual_old)} "
                               f"but ground truth says unbound")
        elif actual_old is None:
            incorrect += 1
            details.append(f"{new_fid}: expected bound to {expected_old} but left unbound")
        elif old_by_el.get(actual_old) == expected_old:
            correct += 1
        else:
            incorrect += 1
            details.append(f"{new_fid}: bound to {old_by_el.get(actual_old)} "
                           f"but ground truth is {expected_old}")

    expected_amb = set(expect_ambiguity)
    surfaced = len(expected_amb & ambiguous_fids)
    missing = expected_amb - ambiguous_fids
    for fid in sorted(missing):
        incorrect += 1
        details.append(f"{fid}: expected surfaced ambiguity, none recorded")

    return MatchQuality(
        fixture=fixture, transition=transition, total=len(new_state.elements),
        correct=correct, incorrect=incorrect, incorrect_details=tuple(details),
        ambiguity_expected=len(expected_amb), ambiguity_surfaced=surfaced,
    )


@dataclass
class AggregateMatchReport:
    """§J aggregate view: incorrect-match rate and ambiguity handling."""
    transitions: int
    elements: int
    incorrect: int
    incorrect_rate: float
    ambiguity_expected: int
    ambiguity_surfaced: int
    per_transition: List[MatchQuality] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join([
            f"semantic match audit: {self.transitions} transitions, {self.elements} elements",
            f"  incorrect match rate: {self.incorrect_rate:.1%} ({self.incorrect}/{self.elements})",
            f"  ambiguity: {self.ambiguity_surfaced}/{self.ambiguity_expected} surfaced",
        ])


def aggregate(qualities: Sequence[MatchQuality]) -> AggregateMatchReport:
    qualities = list(qualities)
    elements = sum(q.total for q in qualities)
    incorrect = sum(q.incorrect for q in qualities)
    return AggregateMatchReport(
        transitions=len(qualities),
        elements=elements,
        incorrect=incorrect,
        incorrect_rate=(incorrect / elements) if elements else 0.0,
        ambiguity_expected=sum(q.ambiguity_expected for q in qualities),
        ambiguity_surfaced=sum(q.ambiguity_surfaced for q in qualities),
        per_transition=qualities,
    )


def audit_all_fixtures(fixtures: Sequence[SemanticFixture]) -> AggregateMatchReport:
    """Run every transition of every fixture and aggregate — the §J measurement."""
    qualities = [measure_transition(fx, t) for fx in fixtures for t in fx.transitions]
    return aggregate(qualities)
