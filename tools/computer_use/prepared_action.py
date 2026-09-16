"""Dependency tracking + ``PreparedAction`` (RFC #112639, P2).

Implements the prepare → validate → commit split from section C and the dynamic
dependency graph from section B: preparation is read-only and may happen early
(in parallel, speculatively), while commit stays late, short, and routed through
the existing computer_use authorization/dispatch path.

``state_rev`` is a per-process monotonic counter owned by ``DependencyGraph``.
It is NOT a lease epoch: real lease-epoch semantics are blocked on #108914, and
nothing here invents them. When #108914 lands, its epoch can pin
``PreparedAction.state_rev`` the same way ``ExecutionRevision.control_epoch``
does today.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, Mapping, NamedTuple, Optional, Tuple

# Fact kinds a prepared action may depend on (RFC #112639 section B, suggested
# dependency types). ``FactRef`` is just (kind, key); the graph never interprets keys.
UI_FACT = "ui"               # semantic UI fact (element still exists / still enabled)
GROUNDING = "grounding"      # visual grounding result (token -> coordinates binding)
APP_STATE = "app_state"      # application state (render done, dialog open)
EXT_STATE = "ext_state"      # file / process / network state
MODEL_ASSUMPTION = "model"   # model-generated assumption
APPROVAL = "approval"        # user approval grant
PRIOR_RESULT = "result"      # result of a previous action
RESOURCE = "resource"        # resource availability

_RISK_CLASSES = frozenset({"read_only", "low", "standard", "high"})


class FactRef(NamedTuple):
    kind: str
    key: str


class ValidationResult(NamedTuple):
    ok: bool
    reason: str = ""


class VerifierVerdict(NamedTuple):
    ok: bool
    name: str
    detail: str = ""


class CommitOutcome(NamedTuple):
    ok: bool            # action ran AND the verifier (if any) passed
    committed: bool     # the dispatcher actually ran
    result: Any = None
    verdict: Optional[VerifierVerdict] = None
    reason: str = ""


class DependencyGraph:
    """Tracks fact versions and which prepared nodes depend on them.

    Selective invalidation: bumping one fact invalidates exactly the nodes that
    (transitively) depend on it. Everything else keeps its prepared work.
    """

    def __init__(self) -> None:
        self._rev = 0
        self._fact_versions: Dict[FactRef, int] = {}
        self._nodes: Dict[str, Dict[FactRef, int]] = {}
        self._parents: Dict[str, FrozenSet[str]] = {}

    @property
    def rev(self) -> int:
        return self._rev

    def _bump(self) -> int:
        self._rev += 1
        return self._rev

    def register(self, node_id: str, deps: Iterable[FactRef],
                 *, parents: Iterable[str] = ()) -> int:
        """Pin a node's dependencies at their current versions. Returns the graph revision."""
        self._nodes[node_id] = {d: self._fact_versions.get(d, 0) for d in deps}
        self._parents[node_id] = frozenset(parents)
        return self._bump()

    def invalidate(self, kind: str, key: str) -> FrozenSet[str]:
        """A fact changed: bump its version, return every node now invalid (direct + descendants)."""
        ref = FactRef(kind, key)
        self._fact_versions[ref] = self._fact_versions.get(ref, 0) + 1
        self._bump()
        direct = {nid for nid, deps in self._nodes.items() if ref in deps}
        affected = set(direct)
        children: Dict[str, set] = {}
        for nid, ps in self._parents.items():
            for p in ps:
                children.setdefault(p, set()).add(nid)
        queue = list(direct)
        while queue:
            for child in children.get(queue.pop(), ()):
                if child not in affected:
                    affected.add(child)
                    queue.append(child)
        return frozenset(affected)

    def is_valid(self, node_id: str) -> bool:
        return not self.invalid_reasons(node_id)

    def invalid_reasons(self, node_id: str) -> Tuple[str, ...]:
        pinned = self._nodes.get(node_id)
        if pinned is None:
            return ("unknown_node",)
        reasons = [
            f"fact_changed:{d.kind}:{d.key}"
            for d, ver in pinned.items()
            if self._fact_versions.get(d, 0) != ver
        ]
        reasons.extend(
            f"parent_invalid:{p}"
            for p in self._parents.get(node_id, ())
            if not self.is_valid(p)
        )
        return tuple(reasons)


@dataclass(frozen=True)
class VerifierSpec:
    """A compiled postcondition check: a pure predicate over the commit result.

    "Compiled" means it is built once from a plain function and needs no model
    call at verify time; "deterministic" means the same result always yields the
    same verdict. ``check`` receives the raw commit result (JSON string or mapping).
    """
    name: str
    check: Callable[[Any], bool]

    def verify(self, result: Any) -> VerifierVerdict:
        try:
            ok = bool(self.check(result))
        except Exception as e:  # a verifier must never crash the commit path
            return VerifierVerdict(False, self.name, f"verifier raised: {e}")
        return VerifierVerdict(ok, self.name, "" if ok else "postcondition failed")


def _as_mapping(result: Any) -> Mapping[str, Any]:
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return result if isinstance(result, Mapping) else {}


def verifier_result_ok() -> VerifierSpec:
    """The commit reported success: truthy ``ok``, or no ``error`` key at all."""
    def check(result: Any) -> bool:
        m = _as_mapping(result)
        return bool(m.get("ok", "error" not in m))
    return VerifierSpec("result_ok", check)


def verifier_no_error() -> VerifierSpec:
    def check(result: Any) -> bool:
        return not _as_mapping(result).get("error")
    return VerifierSpec("no_error", check)


def verifier_all(*specs: VerifierSpec) -> VerifierSpec:
    name = "all(" + ",".join(s.name for s in specs) + ")"
    def check(result: Any) -> bool:
        return all(s.check(result) for s in specs)
    return VerifierSpec(name, check)


def verifier_any(*specs: VerifierSpec) -> VerifierSpec:
    name = "any(" + ",".join(s.name for s in specs) + ")"
    def check(result: Any) -> bool:
        return any(s.check(result) for s in specs)
    return VerifierSpec(name, check)


@dataclass(frozen=True)
class PreparedAction:
    """One compiled future action, ready to commit (RFC #112639 section N).

    ``prepared_payload`` is the fully resolved ``(action, args)`` pair: any
    snapshot-bound references were already grounded during ``prepare()``, so
    commit performs no perception. ``commit()`` re-validates first and routes
    through the caller's dispatcher — in production
    ``tools.computer_use.tool.handle_computer_use``, which re-runs hard blocks
    and the approval gate per call, so a prepared action never caches an
    approval decision.
    """
    node_id: str
    state_rev: int
    prepared_payload: Tuple[str, Mapping[str, Any]]
    dependencies: Tuple[FactRef, ...]
    resource_set: Tuple[str, ...]
    risk_class: str
    verifier: Optional[VerifierSpec] = None
    parents: Tuple[str, ...] = ()

    def validate(self, graph: DependencyGraph) -> ValidationResult:
        reasons = graph.invalid_reasons(self.node_id)
        if reasons:
            return ValidationResult(False, ";".join(reasons))
        return ValidationResult(True)

    def commit(self, graph: DependencyGraph,
               dispatch: Callable[[str, Mapping[str, Any]], Any]) -> CommitOutcome:
        """Re-validate, then run through the existing computer_use path. An
        invalidated action is refused without touching ``dispatch`` — no side
        effects, no approval consumed."""
        verdict0 = self.validate(graph)
        if not verdict0.ok:
            return CommitOutcome(False, False, None, None, f"refused: {verdict0.reason}")
        action, args = self.prepared_payload
        result = dispatch(action, args)
        verdict = self.verifier.verify(result) if self.verifier else None
        ok = verdict.ok if verdict else True
        return CommitOutcome(ok, True, result, verdict,
                             "" if ok else f"verifier failed: {verdict.name if verdict else ''}")


def prepare(*, node_id: str, graph: DependencyGraph, action: str,
            args: Mapping[str, Any], dependencies: Iterable[FactRef],
            parents: Iterable[str] = (), resource_set: Iterable[str] = (),
            risk_class: str = "standard",
            ground: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
            verifier: Optional[VerifierSpec] = None) -> PreparedAction:
    """Read-only preparation: resolve/compile the action now, commit later.

    ``ground`` is the only world access preparation gets — a pure resolver such
    as "element token -> coordinates from the latest capture". ``prepare()``
    takes no dispatcher or backend handle, so preparation cannot cause side
    effects by construction; the expensive work (grounding, payload compile,
    verifier compile) happens once here, never again at commit time.
    """
    if risk_class not in _RISK_CLASSES:
        raise ValueError(f"unknown risk_class {risk_class!r}; expected one of {sorted(_RISK_CLASSES)}")
    if not action or not action.strip():
        raise ValueError("action must be a non-empty computer_use action name")
    compiled = ground(action, dict(args)) if ground is not None else dict(args)
    deps = tuple(dependencies)
    rev = graph.register(node_id, deps, parents=parents)
    return PreparedAction(
        node_id=node_id,
        state_rev=rev,
        prepared_payload=(action.strip().lower(), compiled),
        dependencies=deps,
        resource_set=tuple(resource_set),
        risk_class=risk_class,
        verifier=verifier,
        parents=tuple(parents),
    )
