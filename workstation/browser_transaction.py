"""Owner-declared UI transaction admission and semantic evidence strength.

Planner phase labels never grant exemptions. A recognized contract is registry
metadata supplied by the tool owner for one explicit operation, not arbitrary JS.
"""
from tools.effects import READ_EFFECTS, tool_contract, tool_effect
from workstation.execution_policy import EvidenceStrength


def transaction_contract(request, graph):
    contract_id = request.get('transaction_contract')
    if not contract_id:
        return None  # Existing strict verifier admission remains unchanged.
    if request.get('kind') != 'browser_transaction' or not request.get('preflight'):
        raise ValueError('Recognized browser transaction requires read-only semantic preflight')
    scope = request.get('recipe_scope') or {}
    if scope.get('route') != 'native_browser' or not scope.get('host') or not scope.get('path_family'):
        raise ValueError('Browser transaction requires exact host/path scope')
    nodes = sum(graph.values(), [])
    commits = []
    interactions = []
    for node in nodes:
        phase = node.get('transaction_phase')
        effect, metadata = tool_contract(node['tool'])
        if effect in READ_EFFECTS:
            if phase not in {'PREPARE', 'VERIFY'}:
                raise ValueError('Transaction reads must declare PREPARE/VERIFY')
            continue
        declarations = metadata.get('browser_transactions') or {}
        declaration = declarations.get(contract_id)
        if not isinstance(declaration, dict) or phase not in {'INTERACT', 'COMMIT'} or declaration.get('phase') != phase:
            raise ValueError('Unrecognized owner transaction phase; strict mutation admission required')
        operation_field = declaration.get('operation_field')
        if not operation_field or node.get('args', {}).get(operation_field) != declaration.get('operation'):
            raise ValueError('Transaction operation differs from owner contract')
        if not node.get('semantic_anchor'):
            raise ValueError('Transaction mutation requires reacquirable semantic anchor')
        if phase == 'COMMIT':
            commits.append(node)
        else:
            interactions.append(node)
    if not commits:
        raise ValueError('Browser transaction requires a COMMIT boundary')
    required = EvidenceStrength.SEMANTIC_PERSISTED_READBACK if request.get('mutation_target', {}).get('scope') == 'external' else EvidenceStrength.SAME_SESSION_SEMANTIC_OBSERVATION
    for mutation in [*commits, *interactions]:
        verifiers = [v for v in nodes if mutation['id'] in v.get('verifies', [])]
        minimum = required if mutation in commits else EvidenceStrength.SAME_SESSION_SEMANTIC_OBSERVATION
        if not any(v.get('transaction_phase') == 'VERIFY' and tool_effect(v['tool']) in READ_EFFECTS and v.get('expect') and evidence_strength(v) >= minimum for v in verifiers):
            raise ValueError('Transaction requires semantic verification at the effect boundary')
    return {'id': contract_id, 'interactions': {n['id'] for n in interactions}, 'minimum_commit_evidence': required}


def evidence_strength(node):
    """Strength comes from owner metadata and executed read mode, never a label."""
    _, metadata = tool_contract(node['tool'])
    strength = metadata.get('evidence_strength', 1 if node['tool'] == 'browser_snapshot' else 0)
    if isinstance(strength, int) and 0 <= strength <= 3:
        return EvidenceStrength(strength)
    return EvidenceStrength.TOOL_ACK_ONLY
