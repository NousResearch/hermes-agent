"""Operation-scoped compilation admission; resource authority stays in its owners.

Repeatability prose is an optimization hint, never mutation authority. This
transient projection uses the existing mutation ledger, structural signatures,
and semantic operation fingerprints.
"""
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import json
import re
from typing import Any

from tools.effects import WRITE_EFFECTS, tool_contract, tool_effect, unwrap_call
from workstation.batch_detection import call_key, structural_signature
from workstation.recipes import digest, sanitize
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
    executed_occurrences: int = 0
    verified_successes: int = 0
    verifier_capability: str = 'unknown'
    confidence: float = 0.0
    blast_radius: int = 0
    compile_status: str = 'DISCOVERED'
    metrics: dict = field(default_factory=dict)
    is_homogeneous: bool = True
    semantic_fingerprint: str | None = None

    @property
    def successful_occurrences(self) -> int:
        """Semantic verified successes only; executed_unverified is never verified success."""
        return self.verified_successes

    @successful_occurrences.setter
    def successful_occurrences(self, value: int) -> None:
        self.verified_successes = value


_TRANSIENT_REF = re.compile(r"^@e\d+$")


def semantic_target_family(name: str, args: dict, *, contract: dict | None = None, agent: Any = None) -> str | None:
    """Extract or resolve the concrete semantic target family for a tool call.

    Returns None if semantic target family cannot be established with positive evidence
    (i.e. unknown / insufficient semantic identity).
    Never treats transient renderer refs (@eN) or arbitrary page text as semantic family.
    """
    if not isinstance(args, dict):
        return None

    # 1. Explicit target family in args
    explicit_family = args.get('target_family') or args.get('operation_family') or args.get('family')
    if explicit_family and isinstance(explicit_family, str):
        return explicit_family.strip()

    # 2. Semantic anchor in args (e.g. from durable trace or explicit call)
    anchor = args.get('semantic_anchor')
    if isinstance(anchor, dict):
        atype = str(anchor.get('type') or '').strip()
        avalue = str(anchor.get('value') or '').strip()
        if atype and avalue and avalue != '[REACQUIRE]':
            return f"{atype}:{avalue}"

    # 3. Semantic target in args (e.g. {role, name, testid, tag})
    starget = args.get('semantic_target') or args.get('target_descriptor') or args.get('target')
    if isinstance(starget, dict):
        testid = str(starget.get('testid') or starget.get('data-testid') or starget.get('data-test') or '').strip()
        role = str(starget.get('role') or '').strip()
        name_attr = str(starget.get('name') or starget.get('label') or '').strip()
        tag = str(starget.get('tag') or '').strip()
        if testid:
            return f"testid:{testid}"
        if role and name_attr:
            return f"role_name:{role}:{name_attr}"
        if role:
            return f"role:{role}"
        if tag:
            return f"tag:{tag}"
    elif isinstance(starget, str) and starget.strip() and not _TRANSIENT_REF.match(starget.strip()):
        return starget.strip()

    # 4. Resolve via agent's latest browser elements or trace if ref is provided
    ref = args.get('ref')
    if ref and isinstance(ref, str) and agent is not None:
        traces = getattr(agent, '_work_procedure_trace', [])
        if traces:
            for tr in reversed(traces):
                if tr.get('arguments', {}).get('ref') == ref and tr.get('semantic_anchor'):
                    anc = tr['semantic_anchor']
                    if isinstance(anc, dict) and anc.get('value') and anc.get('value') != '[REACQUIRE]':
                        return f"{anc.get('type', 'anchor')}:{anc['value']}"

    # 5. Non-browser tools: check contract mutation_target or known toolset
    if contract is None:
        _, contract = tool_contract(name)
    mutation_target = contract.get('mutation_target') or {}
    if mutation_target.get('kind'):
        provider = mutation_target.get('provider')
        kind = mutation_target['kind']
        return f"{provider}.{kind}" if provider else str(kind)
    if mutation_target.get('family'):
        return str(mutation_target['family'])

    # Filesystem tools: write_file, patch
    if name in {'write_file', 'patch'}:
        return 'filesystem.file'

    # Kanban tools
    if name.startswith('kanban_'):
        return f'kanban.{name.removeprefix("kanban_")}'

    # For browser tools with only an ephemeral ref, return None (insufficient semantic identity)
    if name.startswith('browser_'):
        return None

    # Fallback for non-browser domain tools: the tool name itself is the semantic operation/family
    return name


def semantic_operation_fingerprint(
    name: str,
    args: dict,
    *,
    contract: dict | None = None,
    route: str | None = None,
    scope: dict | None = None,
    target_family: str | None = None,
    version: int | str = 1,
    agent: Any = None,
) -> str | None:
    """Derive stable semantic operation fingerprint across stable dimensions.

    Uses:
    - canonical operation
    - canonical route/provider
    - semantic target/contract family
    - effect contract
    - relevant scope
    - version

    Returns None if semantic target family cannot be established with positive evidence.
    Never includes tokens, cookies, credentials, sensitive text, WebContents IDs,
    tab IDs, refs @eN, or transient renderer IDs.
    """
    if contract is None:
        _, contract = tool_contract(name)

    family = target_family or semantic_target_family(name, args, contract=contract, agent=agent)
    if not family:
        return None  # unknown / insufficient semantic identity

    canonical_op = name.removeprefix('browser_') if name.startswith('browser_') else name
    canonical_route = route or canonical_route_for_tool(name)
    effect = tool_effect(name).value

    payload = {
        'version': str(version),
        'canonical_op': canonical_op,
        'canonical_route': canonical_route,
        'target_family': family,
        'effect': effect,
        'scope': sanitize(scope or {}),
    }
    return f"sem_op_{digest(payload)[:24]}"


def decisions_for_calls(agent, calls):
    """Preview each call in emission order; refused calls never count as dispatch.

    The third distinct equivalent mutation of the SAME semantic family is gated with REQUIRE_COMPILE.
    Previous dispatched effects (including uncertain ones) count; reads and exact duplicate calls do not.
    Structural repetition without positive semantic homogeneity evidence may SUGGEST_COMPILE,
    but never REQUIRE_COMPILE.
    """
    struct_counts = dict(getattr(agent, '_work_mutation_shapes', {}))
    sem_counts = dict(getattr(agent, '_work_mutation_sem_families', {}))
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
            route = canonical_route_for_tool(name)
            sem_family = semantic_target_family(name, args, contract=contract, agent=agent)
            sem_fp = semantic_operation_fingerprint(
                name, args, contract=contract, route=route, target_family=sem_family, agent=agent
            ) if sem_family else None

            if sem_fp and sem_fp not in sem_counts:
                evidence_count = sum(
                    1 for r in evidence.values()
                    if r.get('operation_fingerprint') == sem_fp or r.get('semantic_fingerprint') == sem_fp
                )
                if evidence_count:
                    sem_counts[sem_fp] = evidence_count
                elif not name.startswith('browser_'):
                    sem_counts[sem_fp] = struct_counts.get(signature, 0)

            if key not in seen:
                seen.add(key)
                struct_counts[signature] = struct_counts.get(signature, 0) + 1
                if sem_fp:
                    sem_counts[sem_fp] = sem_counts.get(sem_fp, 0) + 1

            struct_count = struct_counts.get(signature, 0)
            sem_count = sem_counts.get(sem_fp, 0) if sem_fp else 0

            # Occurrence metrics: executed_occurrences vs verified_successes
            executed_occ = sum(
                1 for r in evidence.values()
                if (r.get('operation_fingerprint') in {signature, sem_fp} or r.get('structural_signature') == signature)
                and r.get('status') in {'executed_unverified', 'verified'}
            )
            verified_succ = sum(
                1 for r in evidence.values()
                if (r.get('operation_fingerprint') in {signature, sem_fp} or r.get('structural_signature') == signature)
                and r.get('status') == 'verified'
            )

            candidate_key = sem_fp or signature
            candidate = CompilationCandidate(
                pattern_id=signature,
                operation_fingerprint=sem_fp or signature,
                route=route,
                target_family=sem_family or str(contract.get('mutation_target', {}).get('kind') or name),
                occurrences=struct_count,
                executed_occurrences=executed_occ,
                verified_successes=verified_succ,
                verifier_capability=contract.get('default_verifier', 'unknown'),
                confidence=min(1.0, (sem_count or struct_count) / 3),
                blast_radius=struct_count,
                compile_status='DISCOVERED',
                metrics={},
                is_homogeneous=bool(sem_fp),
                semantic_fingerprint=sem_fp,
            )
            candidates[candidate_key] = candidate

            if (sem_fp and sem_count >= 3) or (not name.startswith('browser_') and struct_count >= 3):
                decision = CompilationDecision.REQUIRE_COMPILE
            elif (sem_fp and sem_count >= 2) or (not name.startswith('browser_') and struct_count >= 2):
                decision = CompilationDecision.SUGGEST_COMPILE
            elif struct_count >= 2:
                # Structural repetition suggests compilation/learning, but never forces it
                decision = CompilationDecision.SUGGEST_COMPILE
            else:
                decision = CompilationDecision.ALLOW_ADAPTIVE

        decisions.append(decision)

    agent._work_compilation_candidates = candidates
    agent._work_mutation_evidence = evidence
    return decisions

