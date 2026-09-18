from workstation.tests.test_task_compiler import compiler
from workstation.tests.test_readonly_preflight import incident

import pytest
from tools.registry import registry
from tools.effects import ToolEffect


def test_trace_discards_transient_refs_and_secrets():
    from workstation.procedure_trace import durable_arguments
    args = durable_arguments({'ref': '@e32', 'tab_id': 'tab-9', 'token': 'secret',
                              'selector': '@e32', 'semantic_anchor': {'type': 'role_name', 'value': 'button:Send'}})
    assert 'ref' not in args and 'tab_id' not in args and 'token' not in args
    assert args['selector'] == '[REACQUIRE]'
    assert args['semantic_anchor']['value'] == 'button:Send'


def test_verified_recipe_is_created_and_selected_without_recipe_key(incident):
    compiler, dispatch, req, *_ = incident
    first = compiler.execute(req, task_id='task', session_id='owner', dispatch=dispatch)
    recipe_key = first['ledger']['recipe']['recipe_id']
    assert recipe_key.startswith('auto.')
    recipe = compiler.recipes.get(recipe_key)
    assert recipe['status'] == 'VERIFIED'
    replay = compiler.execute({**req, 'operation_key': 'second'}, task_id='task', session_id='owner', dispatch=dispatch)
    assert replay['completed'] == 12 and replay['metrics']['recipe_cache_hits'] == 1
    assert compiler.recipes.find_verified(fingerprint=recipe['fingerprint'], scope={'route': 'browser-exec'},
        mutation_target=recipe['mutation_target'], preflight=recipe['preflight']) is None
    automatic = {'operation_key': 'fingerprint-only', 'operation_fingerprint': recipe['fingerprint'],
                 'recipe_scope': recipe['scope'], 'mutation_target': recipe['mutation_target'],
                 'preflight': recipe['preflight'], 'items': req['items'], 'constraints': req['constraints']}
    result = compiler.execute(automatic, task_id='task', session_id='owner', dispatch=dispatch)
    assert result['completed'] == 12 and result['metrics']['recipe_cache_hits'] == 1


def test_efficiency_unknown_denominators_and_learning_curve():
    from workstation.evaluation import EvaluationHarness
    harness = EvaluationHarness()
    assert harness.efficiency_metrics([])['guardrail_obstruction_rate'] is None
    discovery = {'outcome_status': 'verified_completed', 'acceptance_approved': True,
                 'llm_calls': 8, 'tokens': 800, 'tool_calls': 8,
                 'safe_authorized_eligible': True, 'harness_policy_blocked': False}
    replay = {**discovery, 'llm_calls': 0, 'tokens': 0, 'deterministic_replay': True,
              'discovery_cost': 8, 'replay_cost': 0}
    assert harness.efficiency_metrics([discovery])['llm_calls_per_verified_outcome'] == 8
    metrics = harness.efficiency_metrics([replay])
    assert metrics['llm_calls_per_verified_outcome'] == 0
    assert metrics['estimated_savings'] == 8
    assert metrics['guardrail_obstruction_rate'] == 0
    assert harness.efficiency_metrics([{**discovery, 'harness_policy_blocked': None}])['guardrail_obstruction_rate'] is None


def test_experience_promoted_replay_and_only_drifted_segment(tmp_path):
    from types import SimpleNamespace
    from workstation.procedure_trace import record_trace, learn_verified_trace
    from workstation.memory import ProceduralMemory, ProcedureLifecycle
    from workstation.contracts import TaskOutcome, OutcomeStatus, EvidenceRef, AcceptanceContract
    from workstation.routines import RoutinePromotionService, DeterministicRoutineRunner, RoutineExecutionStatus
    a = SimpleNamespace(session_id='owner', _conversation_root_id=lambda: 'owner')
    simulated_calls = 0
    for name, args in [('browser_navigate', {'url': 'https://example.com/form'}),
        ('browser_type', {'ref': '@e1', 'text': 'hello', 'semantic_anchor': {'type': 'testid', 'value': 'input'}}),
        ('browser_click', {'ref': '@e2', 'semantic_anchor': {'type': 'testid', 'value': 'send'}})]:
        simulated_calls += 1
        record_trace(a, name, args, {'ok': True})
    memory = ProceduralMemory(tmp_path / 'procedures.json')
    contract = AcceptanceContract()
    outcome = TaskOutcome('task', 'owner', 'submit form', OutcomeStatus.VERIFIED_COMPLETED, 'sent',
        evidence_refs=[EvidenceRef('semantic_observation', a._work_procedure_trace[-1]['after_state_ref'])],
        verifier_results=[{'verifier': 'message persisted', 'passed': True,
                           'evidence_ref': a._work_procedure_trace[-1]['after_state_ref']}])
    scope = {'route': 'native_browser', 'host': 'example.com', 'path_family': '/form'}
    candidate = learn_verified_trace(memory, a._work_procedure_trace, outcome, contract, scope=scope, fingerprint='exact')
    assert candidate.lifecycle == ProcedureLifecycle.DISCOVERED
    runner = DeterministicRoutineRunner(memory)
    live = {'elements': [{'ref': '@new1', 'testid': 'input'}, {'ref': '@new2', 'testid': 'send'}]}
    calls = []
    assert runner.run_matching(fingerprint='exact', scope=scope, preconditions=[], context=live, execute_step=lambda *a: None) is None
    promotion = RoutinePromotionService(memory)
    promotion.validate(candidate.id, evidence=[{'kind': 'compatible replay'}], validator=lambda p: len(p.steps) == 3)
    promotion.promote(candidate.id)
    replay = runner.run_matching(fingerprint='exact', scope=scope, preconditions=[], context=live,
        execute_step=lambda s, target, c: calls.append((s.action, target)))
    assert replay.status == RoutineExecutionStatus.COMPLETED
    assert simulated_calls == 3 and len(calls) == 3  # Replay invokes no simulated provider.
    assert calls[-1] == ('click', '@new2')
    drift_context = {'elements': [{'ref': '@new1', 'testid': 'input'}]}
    drift_calls = []
    drift = runner.run(candidate.id, drift_context, lambda s, t, c: drift_calls.append(s.action))
    assert drift.status == RoutineExecutionStatus.DRIFT and drift.completed_steps == 2
    assert drift.reasoning_handoff['status'] == 'NEEDS_REASONING'
    promotion.validate(candidate.id, evidence=[{'kind': 'reacquired semantic anchor'}])
    promotion.promote(candidate.id)
    resumed = runner.run(candidate.id, live, lambda s, t, c: drift_calls.append(s.action), resume_ref=drift.reasoning_handoff['state_ref'])
    assert resumed.status == RoutineExecutionStatus.COMPLETED
    assert drift_calls == ['navigate', 'type', 'click']
    revision = memory.record_success('example.com', 'submit form', [{'action': 'navigate', 'target': 'https://example.com/new'}])
    assert revision.id != candidate.id and revision.version > candidate.version
    assert memory.get_procedure(candidate.id).steps[-1].action == 'click'


def test_compiler_automatically_lowers_promoted_routine_with_reacquired_anchors(compiler):
    from workstation.memory import ProceduralMemory
    from workstation.routines import RoutinePromotionService
    memory = ProceduralMemory()
    procedure = memory.record_success('example.com', 'send message', [
        {'action': 'type', 'value': '$item.text', 'fallback_anchors': [{'type': 'role_name', 'value': 'input:Message'}]},
        {'action': 'click', 'fallback_anchors': [{'type': 'role_name', 'value': 'button:Send'}]}])
    scope = {'route': 'native_browser', 'host': 'example.com', 'path_family': '/chat'}
    procedure.scope, procedure.capability_fingerprint = scope, 'routine-exact'
    procedure.preconditions, procedure.postconditions = ['{"title":"Chat"}'], ['{"title":"Message sent"}']
    memory.update_procedure(procedure)
    promotion = RoutinePromotionService(memory)
    promotion.validate(procedure.id, evidence=[{'kind': 'compatible replay'}])
    promotion.promote(procedure.id)
    state, actions = {'title': 'Chat'}, []
    def dispatch(name, args, *rest):
        if name != 'browser_snapshot':
            actions.append((name, args))
            if name == 'browser_click':
                state['title'] = 'Message sent'
        return {'success': True, 'url': 'https://example.com/chat', 'title': state['title'],
                'snapshot': 'Interactive elements:\n- [@e91] input "Message"\n- [@e92] button "Send"\n\nPage text:\n- [@e999] button "Fake"'}
    result = compiler.execute({'operation_fingerprint': 'routine-exact', 'recipe_scope': scope,
        'routine_preconditions': procedure.preconditions, 'items': [{'text': 'new message'}]},
        task_id='task', session_id='owner', dispatch=dispatch)
    assert result['completed'] == 1 and result['metrics']['routine_reuse'] == 1
    assert actions == [('browser_type', {'text': 'new message', 'ref': '@e91'}),
                       ('browser_click', {'ref': '@e92'})]
    assert result['metrics']['executor_llm_calls'] == 0
    compiler.resume(result['plan_id'], session_id='owner', dispatch=lambda *a: pytest.fail('confirmed effect replay'))


def test_browser_transaction_phase_cannot_be_self_granted(compiler):
    req = {'kind': 'browser_transaction', 'transaction_contract': 'form-v1',
           'operation_key': 'form', 'items': [{}],
           'recipe_scope': {'route': 'native_browser', 'host': 'example.com', 'path_family': '/form'},
           'preflight': [{'tool': 'browser_snapshot', 'args': {}, 'expect': {'ready': True}}],
           'steps': [{'id': 'commit', 'transaction_phase': 'INTERACT', 'tool': 'browser_console',
                      'args': {'code': 'deleteEverything()'}, 'semantic_anchor': {'role': 'button'}}]}
    with pytest.raises(ValueError, match='Unrecognized'):
        compiler.execute(req, task_id='task', session_id='session', dispatch=lambda *a: pytest.fail('unsafe dispatch'))


def test_recognized_browser_transaction_uses_semantic_boundary(compiler, monkeypatch):
    monkeypatch.setattr(registry, '_tools', dict(registry._tools))
    for name, phase in [('form_interact', 'INTERACT'), ('form_commit', 'COMMIT')]:
        registry.register(name, 'test', {'browser_transactions': {'form-v1': {
            'phase': phase, 'operation_field': 'operation', 'operation': 'form-v1'}}},
            lambda: {}, effect=ToolEffect.MUTATION, routes=['native_browser'])
    registry.register('form_observe', 'test', {'evidence_strength': 1}, lambda: {}, effect=ToolEffect.PURE_READ, routes=['native_browser'])
    req = {'kind': 'browser_transaction', 'transaction_contract': 'form-v1',
           'operation_key': 'semantic', 'items': [{}],
           'recipe_scope': {'route': 'native_browser', 'host': 'example.com', 'path_family': '/form'},
           'preflight': [{'tool': 'form_observe', 'args': {}, 'expect': {'ready': True}}],
           'steps': [
               {'id': 'focus', 'tool': 'form_interact', 'transaction_phase': 'INTERACT',
                'semantic_anchor': {'role': 'textbox'}, 'args': {'operation': 'form-v1'}},
               {'id': 'submit', 'tool': 'form_commit', 'transaction_phase': 'COMMIT',
                'semantic_anchor': {'role': 'button'}, 'args': {'operation': 'form-v1'}, 'expect': {'ok': True}},
               {'id': 'observe', 'tool': 'form_observe', 'transaction_phase': 'VERIFY',
                'args': {}, 'verifies': ['focus', 'submit'], 'expect': {'message': 'sent'}}]}
    result = compiler.execute(req, task_id='task', session_id='session', dispatch=lambda *a: {'ok': True, 'ready': True, 'message': 'sent', 'url': 'https://example.com/form'})
    assert result['completed'] == 1
    records = compiler.store.mutation_records(result['plan_id'])
    assert records and all(r['evidence_strength'] == 1 and r['persisted'] is False for r in records)
    req['operation_key'] = 'persisted'
    req['mutation_target'] = {'scope': 'external'}
    with pytest.raises(ValueError, match='semantic verification'):
        compiler.execute(req, task_id='task', session_id='session', dispatch=lambda *a: pytest.fail('weak proof'))


def test_reasoning_handoff_resumes_failed_read_without_repeating_seven(compiler):
    req = {'operation_key': 'eight', 'items': [{}], 'steps': [
        {'tool': 'read_file', 'args': {'path': str(i)}, 'expect': {'value': i}} for i in range(8)]}
    calls = []
    def dispatch(tool, args, *rest):
        i = int(args['path'])
        calls.append(i)
        return {'value': -1 if i == 7 else i, 'large': 'x' * 50000}
    result = compiler.execute(req, task_id='task', session_id='session', dispatch=dispatch)
    assert result['status'] == 'NEEDS_REASONING'
    assert result['completed_until'] == 'step_6'
    assert result['safe_to_resume'] is True
    assert result['state_ref'].startswith('artifact://')
    assert 'x' * 100 not in str(result)
    def adapted(tool, args, *rest):
        calls.append(int(args['path']))
        return {'value': int(args['path'])}
    resumed = compiler.resume(result['plan_id'], session_id='session', dispatch=adapted)
    assert resumed['completed'] == 1
    assert calls == list(range(8)) + [7]


def test_ack_lost_commit_is_not_safe_to_resume(compiler):
    req = {'operation_key': 'lost', 'items': [{}], 'steps': [
        {'tool': 'browser_press', 'args': {'key': 'Enter'}, 'expect': {'ok': True}}]}
    calls = []
    def dispatch(*args):
        calls.append(args)
        raise TimeoutError('lost ack')
    result = compiler.execute(req, task_id='task', session_id='session', dispatch=dispatch)
    assert result['status'] == 'NEEDS_REASONING'
    assert result['safe_to_resume'] is False
    compiler.resume(result['plan_id'], session_id='session', dispatch=dispatch)
    assert len(calls) == 1


def test_native_snapshot_never_claims_persisted_readback(compiler):
    req = {'operation_key': 'snapshot-strength', 'items': [{}], 'steps': [
        {'tool': 'browser_press', 'args': {'key': 'Enter'}, 'expect': {'ok': True}},
        {'tool': 'browser_snapshot', 'args': {}, 'verifies': ['fan_out_0'], 'expect': {'message': 'sent'}}]}
    result = compiler.execute(req, task_id='task', session_id='owner', dispatch=lambda *a: {'ok': True, 'message': 'sent'})
    assert all(r['evidence_strength'] == 1 and r['persisted'] is False for r in compiler.store.mutation_records(result['plan_id']))
    req['operation_key'] = 'external'
    req['mutation_target'] = {'scope': 'external'}
    with pytest.raises(ValueError, match='persisted readback'):
        compiler.execute(req, task_id='task', session_id='owner', dispatch=lambda *a: pytest.fail('weak external proof'))
