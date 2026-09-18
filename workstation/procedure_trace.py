"""Bounded adaptive observations; artifacts/journal/memory retain ownership."""
import re
import json
from tools.effects import tool_effect
from workstation.recipes import sanitize, digest
from workstation.routing import canonical_route_for_tool

_TRANSIENT = {'ref', 'node_id', 'nodeId', 'tab_id', 'tabId', 'webContentsId', 'webcontents_id'}


def durable_arguments(value):
    value = sanitize(value)
    if isinstance(value, dict):
        return {k: ('$item.text' if k == 'text' else durable_arguments(v)) for k, v in value.items() if k not in _TRANSIENT}
    if isinstance(value, list):
        return [durable_arguments(v) for v in value[:64]]
    if isinstance(value, str):
        return '[REACQUIRE]' if re.search(r'@e\d+\b', value) else value[:1024]
    return value


def record_trace(agent, name, args, raw, *, duration_ms=None):
    from workstation.artifacts import ArtifactStore
    from workstation.batch_detection import structural_signature
    from workstation.execution_policy import semantic_operation_fingerprint
    from agent.tool_guardrails import classify_tool_failure
    from workstation.journal import ExecutionJournal
    from workstation.contracts import ExecutionEventKind
    owner = getattr(agent, '_canonical_work_task_id', None) or agent._conversation_root_id() or agent.session_id
    artifacts = ArtifactStore()
    output = artifacts.store(owner, 'trace_result_' + digest(sanitize(raw)) + '.json', sanitize(raw))
    arguments = durable_arguments(args)
    try:
        decoded = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        decoded = None
    selected_runtime = 'internal' if isinstance(decoded, dict) and decoded.get('runtime') == 'electron-chromium' else None
    traces = getattr(agent, '_work_procedure_trace', [])
    before_ref = traces[-1]['after_state_ref'] if traces else None
    if selected_runtime == 'internal' and args.get('ref') and not arguments.get('semantic_anchor') and before_ref:
        from workstation.routines import semantic_browser_elements
        before = artifacts.read_json(before_ref)
        elements = semantic_browser_elements(before) if isinstance(before, dict) else []
        element = next((e for e in elements if e.get('ref') == args['ref']), None)
        if element:
            if element.get('testid'):
                arguments['semantic_anchor'] = {'type': 'testid', 'value': element['testid']}
            elif element.get('name'):
                arguments['semantic_anchor'] = {'type': 'name', 'value': f"{element.get('role', 'element')}:{element['name']}"}
            elif element.get('label') and sum(e.get('role') == element.get('role') and e.get('label') == element['label'] for e in elements) == 1:
                arguments['semantic_anchor'] = {'type': 'role_name', 'value': element['role'] + ':' + element['label']}
    sem_fp = semantic_operation_fingerprint(name, args)
    op_fp = sem_fp or structural_signature(name, args)
    record = {'tool': name, 'action': name.removeprefix('browser_'),
        'route': canonical_route_for_tool(name, runtime=selected_runtime), 'operation_fingerprint': op_fp,
        'semantic_fingerprint': sem_fp,
        'arguments': arguments, 'semantic_anchor': arguments.get('semantic_anchor'),
        'before_state_ref': before_ref, 'after_state_ref': output.ref,
        'outcome': 'failed' if classify_tool_failure(name, raw if isinstance(raw, str) else json.dumps(raw))[0] else 'executed_unverified',
        'effect': tool_effect(name).value, 'duration_ms': duration_ms,
        'task_id': getattr(agent, '_canonical_work_task_id', None),
        'run_id': getattr(agent, '_canonical_work_run_id', None),
        'provider_usage': sanitize(getattr(agent, '_current_provider_usage', None)),
        'replayable': not any(k in args for k in _TRANSIENT) or bool(arguments.get('semantic_anchor'))}
    if len(traces) >= 64:
        agent._work_procedure_trace_truncated = True
        return
    traces.append(record)
    agent._work_procedure_trace = traces
    ref = artifacts.store(owner, 'trace_' + digest(record) + '.json', record)
    ExecutionJournal(owner, agent.session_id).record(ExecutionEventKind.ACTION,
        'adaptive observation captured; semantic verification required', metadata={'trace_ref': ref.ref,
        'operation_fingerprint': record['operation_fingerprint'], 'run_id': record['run_id']})


def candidate_steps(traces):
    """Only semantic actions can enter existing promotion; never arbitrary code."""
    steps = []
    for trace in traces:
        if trace['outcome'] != 'executed_unverified' or not trace['replayable']:
            return []
        anchor = trace.get('semantic_anchor') or {}
        action = trace['action']
        if action in {'snapshot', 'extract_items'}:
            continue
        if action not in {'navigate', 'click', 'type', 'press', 'scroll'}:
            return []
        if action != 'navigate' and not anchor:
            return []
        if anchor and (anchor.get('type') not in {'role_name', 'testid', 'text'} or anchor.get('value') == '[REACQUIRE]'):
            return []
        steps.append({'action': action, 'target': anchor.get('value', trace['arguments'].get('url', '')),
            'anchor_type': anchor.get('type', 'selector'), 'fallback_anchors': [anchor] if anchor else [],
            'value': trace['arguments'].get('text', trace['arguments'].get('key', ''))})
    return steps


def learn_verified_trace(memory, traces, outcome, contract, *, scope, fingerprint):
    """Only acceptance-backed observations seed a versioned routine candidate."""
    from workstation.routines import RoutinePromotionService
    steps = candidate_steps(traces)
    if not steps:
        return None
    candidate = RoutinePromotionService(memory).experience_candidate(outcome, contract,
        site=scope.get('host', scope['route']), steps=steps, repeatable=True)
    if candidate:
        candidate.scope = sanitize(scope)
        candidate.capability_fingerprint = fingerprint
        candidate.preconditions = list(scope.get('preconditions', []))
        return memory.update_procedure(candidate)
    return None


def trace_compatibility(traces):
    """Derive exact host/path and schema fingerprint from executed native work."""
    from urllib.parse import urlparse
    from workstation.recipes import recipe_fingerprint
    native = [t for t in traces if t['route'] == 'native_browser']
    urls = [t['arguments'].get('url') for t in native if t['action'] == 'navigate']
    if not urls or not candidate_steps(native):
        return {}
    parsed = [urlparse(url) for url in urls]
    if any(p.scheme not in {'http', 'https'} or p.hostname != parsed[0].hostname or p.path != parsed[0].path for p in parsed):
        return {}
    scope = {'route': 'native_browser', 'host': parsed[0].hostname, 'path_family': parsed[0].path or '/'}
    graph = {'setup': [], 'fan_out': [{'tool': t['tool'], 'args': t['arguments']} for t in native], 'finalize': []}
    return {'scope': scope, 'fingerprint': recipe_fingerprint(graph, scope),
            'preconditions': [json.dumps({'url': urls[0], 'wall_detected': False}, sort_keys=True)]}
