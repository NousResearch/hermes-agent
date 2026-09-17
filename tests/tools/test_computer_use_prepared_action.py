"""Tests for tools/computer_use/prepared_action.py (RFC #112639, P2: dependency tracking + PreparedAction).

Red-on-base: this module does not exist on main, so every test here fails at
import time until the implementation lands.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import pytest


@pytest.fixture()
def graph():
    from tools.computer_use.prepared_action import DependencyGraph
    return DependencyGraph()


def _fact(kind: str, key: str):
    from tools.computer_use.prepared_action import FactRef
    return FactRef(kind, key)


def _prepare(graph, **kw):
    from tools.computer_use.prepared_action import prepare
    base = dict(node_id="n1", action="click", args={"element": 3},
                dependencies=[_fact("grounding", "export-btn")])
    base.update(kw)
    return prepare(graph=graph, **base)


# ---------------------------------------------------------------------------
# DependencyGraph: selective invalidation
# ---------------------------------------------------------------------------

class TestDependencyGraph:
    def test_registered_node_is_valid(self, graph):
        _prepare(graph)
        assert graph.is_valid("n1")

    def test_unrelated_fact_change_keeps_node_valid(self, graph):
        _prepare(graph)
        affected = graph.invalidate("app_state", "blender:rendering")
        assert affected == frozenset()
        assert graph.is_valid("n1")

    def test_depended_fact_change_invalidates_only_dependents(self, graph):
        _prepare(graph, node_id="n1")
        _prepare(graph, node_id="n2", dependencies=[_fact("app_state", "blender:rendering")])
        affected = graph.invalidate("grounding", "export-btn")
        assert affected == frozenset({"n1"})
        assert not graph.is_valid("n1")
        assert graph.is_valid("n2")

    def test_invalid_reasons_name_the_changed_fact(self, graph):
        _prepare(graph)
        graph.invalidate("grounding", "export-btn")
        reasons = graph.invalid_reasons("n1")
        assert any("grounding" in r and "export-btn" in r for r in reasons)

    def test_invalidation_cascades_to_descendants(self, graph):
        _prepare(graph, node_id="p1", dependencies=[_fact("ui", "dialog")])
        _prepare(graph, node_id="c1", dependencies=[], parents=["p1"])
        _prepare(graph, node_id="g1", dependencies=[], parents=["c1"])
        affected = graph.invalidate("ui", "dialog")
        assert affected == frozenset({"p1", "c1", "g1"})
        assert not graph.is_valid("g1")

    def test_unknown_node_is_not_valid(self, graph):
        assert not graph.is_valid("nope")
        assert graph.invalid_reasons("nope")

    def test_reregister_refreshes_pinned_versions(self, graph):
        _prepare(graph)
        graph.invalidate("grounding", "export-btn")
        assert not graph.is_valid("n1")
        _prepare(graph)  # re-prepare after the world changed: valid again
        assert graph.is_valid("n1")


# ---------------------------------------------------------------------------
# prepare(): read-only preparation, grounding happens once
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_ground_runs_exactly_once_across_prepare_and_commit(self, graph):
        calls: List[Tuple[str, Dict[str, Any]]] = []

        def ground(action: str, args: Dict[str, Any]) -> Dict[str, Any]:
            calls.append((action, dict(args)))
            return {"coordinate": (10, 20)}

        prepared = _prepare(graph, ground=ground)
        assert prepared.prepared_payload[1] == {"coordinate": (10, 20)}

        dispatched: List[Tuple[str, Mapping[str, Any]]] = []
        outcome = prepared.commit(graph, dispatch=lambda a, kw: dispatched.append((a, kw)) or {"ok": True})
        assert outcome.committed
        assert dispatched == [("click", {"coordinate": (10, 20)})]
        assert len(calls) == 1  # commit reuses the prepared payload; no re-grounding

    def test_prepare_with_no_ground_passes_args_through(self, graph):
        # Grounding is optional: without a grounder the raw args are the payload,
        # and preparation still records no side effects anywhere.
        dispatched: List[Tuple[str, Mapping[str, Any]]] = []
        prepared = _prepare(graph)
        outcome = prepared.commit(graph, dispatch=lambda a, kw: dispatched.append((a, kw)) or {"ok": True})
        assert outcome.committed
        assert dispatched == [("click", {"element": 3})]

    def test_state_rev_pins_registration_revision(self, graph):
        p1 = _prepare(graph, node_id="p1")
        graph.invalidate("grounding", "export-btn")
        p2 = _prepare(graph, node_id="p2", dependencies=[])
        assert p2.state_rev > p1.state_rev
        assert p2.state_rev == graph.rev

    def test_unknown_risk_class_rejected(self, graph):
        with pytest.raises(ValueError):
            _prepare(graph, risk_class="whatever")


# ---------------------------------------------------------------------------
# PreparedAction.validate / commit
# ---------------------------------------------------------------------------

class TestValidateCommit:
    def test_commit_refused_after_invalidation_without_touching_dispatch(self, graph):
        prepared = _prepare(graph)
        graph.invalidate("grounding", "export-btn")
        assert not prepared.validate(graph).ok

        def boom(action: str, args: Mapping[str, Any]):
            raise AssertionError("dispatch must not run for an invalidated action")

        outcome = prepared.commit(graph, dispatch=boom)
        assert not outcome.committed
        assert not outcome.ok
        assert "refused" in outcome.reason

    def test_commit_runs_verifier_over_result(self, graph):
        from tools.computer_use.prepared_action import verifier_result_ok
        prepared = _prepare(graph, verifier=verifier_result_ok())
        outcome = prepared.commit(graph, dispatch=lambda a, kw: '{"ok": true}')
        assert outcome.committed and outcome.ok
        assert outcome.verdict is not None and outcome.verdict.ok

    def test_commit_reports_failed_verification(self, graph):
        from tools.computer_use.prepared_action import verifier_result_ok
        prepared = _prepare(graph, verifier=verifier_result_ok())
        outcome = prepared.commit(graph, dispatch=lambda a, kw: '{"error": "click missed"}')
        assert outcome.committed  # the action ran through the real path...
        assert not outcome.ok     # ...but the postcondition did not hold
        assert outcome.verdict is not None and not outcome.verdict.ok


# ---------------------------------------------------------------------------
# VerifierSpec: compiled, deterministic
# ---------------------------------------------------------------------------

class TestVerifiers:
    def test_verifiers_are_deterministic(self, graph):
        from tools.computer_use.prepared_action import verifier_result_ok, verifier_no_error
        for spec in (verifier_result_ok(), verifier_no_error()):
            v1 = spec.verify({"ok": True})
            v2 = spec.verify({"ok": True})
            assert v1 == v2 and v1.ok

    def test_verifier_handles_json_string_results(self, graph):
        from tools.computer_use.prepared_action import verifier_result_ok, verifier_no_error
        assert verifier_result_ok().verify('{"ok": true}').ok
        assert not verifier_no_error().verify('{"error": "denied"}').ok

    def test_verifier_combinators(self, graph):
        from tools.computer_use.prepared_action import (
            verifier_all, verifier_any, verifier_no_error, verifier_result_ok,
        )
        both = verifier_all(verifier_result_ok(), verifier_no_error())
        assert both.verify({"ok": True}).ok
        assert not both.verify({"ok": True, "error": "x"}).ok
        either = verifier_any(verifier_result_ok(), verifier_no_error())
        assert either.verify({"ok": True, "error": "x"}).ok
        assert not either.verify({"error": "x"}).ok

    def test_verifier_names_are_stable(self, graph):
        from tools.computer_use.prepared_action import verifier_all, verifier_result_ok
        assert verifier_all(verifier_result_ok()).name.startswith("all(")
