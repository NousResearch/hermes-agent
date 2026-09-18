"""Operation-scoped compilation admission; resource authority stays in its owners.

Repeatability prose is an optimization hint, never mutation authority. This
transient projection uses the existing mutation ledger and structural signatures.
"""
from dataclasses import dataclass, field
from enum import Enum, IntEnum

from tools.effects import WRITE_EFFECTS, tool_contract, unwrap_call
from workstation.batch_detection import call_key, structural_signature
from workstation.routing import canonical_route_for_tool


class ExecutionMode(str, Enum):
    ADAPTIVE = 'ADAPTIVE'
    COMPILED = 'COMPILED'
    ROUTINE = 'ROUTINE'
    HUMAN = 'HUMAN'


class CompilationDecision(str, Enum):
    ALLOW_ADAPTIVE = 'ALLOW_ADAPTIVE'
    SUGGEST_COMPILE = 'SUGGEST_COMPILE'
    REQUIRE_COMPILE = 'REQUIRE_COMPILE'
    REQUIRE_HUMAN = 'REQUIRE_HUMAN'


class EvidenceStrength(IntEnum):
    TOOL_ACK_ONLY = 0
    SAME_SESSION_SEMANTIC_OBSERVATION = 1
    SEMANTIC_PERSISTED_READBACK = 2
    INDEPENDENT_PERSISTED_READBACK = 3


@dataclass
class CompilationCandidate:
    pattern_id: str
    operation_fingerprint: str
    route: str
    target_family: str
    occurrences: int = 0
    successful_occurrences: int = 0
    verifier_capability: str = 'unknown'
    confidence: float = 0.0
    blast_radius: int = 0
    compile_status: str = 'DISCOVERED'
    metrics: dict = field(default_factory=dict)


def decisions_for_calls(agent, calls):
    """Preview each call in emission order; refused calls never count as dispatch.

    The third distinct equivalent mutation is gated. Previous dispatched effects
    (including uncertain ones) count; reads and exact duplicate calls do not.
    No browser-name exemption: stateful actions have their own signatures too.
    """
    counts = dict(getattr(agent, '_work_mutation_shapes', {}))
    evidence = getattr(agent, '_work_mutation_evidence', {})
    seen = set(evidence) | set(getattr(agent, '_work_completed_mutations', {}))
    candidates = getattr(agent, '_work_compilation_candidates', {})
    decisions = []
    for call in calls:
        name, args = unwrap_call(call)
        effect, contract = tool_contract(name)
        decision = CompilationDecision.ALLOW_ADAPTIVE
        key = call_key(name, args)
        if effect in WRITE_EFFECTS and key not in evidence and getattr(agent, 'session_id', None):
            from workstation.artifacts import ArtifactStore
            owner = getattr(agent, '_conversation_root_id', lambda: None)() or agent.session_id
            store = ArtifactStore()
            ref = f'artifact://tasks/{owner}/mutation_{key}.json'
            if store.resolve_ref(ref):
                record = store.read_json(ref)
                if record.get('status') == 'uncertain':
                    evidence[key] = record
        if evidence.get(key, {}).get('status') == 'uncertain':
            decision = CompilationDecision.REQUIRE_HUMAN
        elif name != 'work_execute' and 'work_execute' in getattr(agent, 'valid_tool_names', ()) and effect in WRITE_EFFECTS:
            signature = structural_signature(name, args)
            if key not in seen:
                seen.add(key)
                counts[signature] = counts.get(signature, 0) + 1
            count = counts.get(signature, 0)
            route = canonical_route_for_tool(name)
            target = contract.get('mutation_target') or {}
            candidate = CompilationCandidate(signature, signature, route,
                str(target.get('kind') or name), count,
                sum(r.get('status') == 'executed_unverified' and r.get('operation_fingerprint') == signature for r in evidence.values()),
                contract.get('default_verifier', 'unknown'), min(1.0, count / 3), count)
            candidates[signature] = candidate
            if count >= 3:
                decision = CompilationDecision.REQUIRE_COMPILE
            elif count >= 2:
                decision = CompilationDecision.SUGGEST_COMPILE
        decisions.append(decision)
    agent._work_compilation_candidates = candidates
    agent._work_mutation_evidence = evidence
    return decisions
