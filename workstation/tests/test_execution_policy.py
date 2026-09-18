import json
from types import SimpleNamespace

from tools.effects import ToolEffect, tool_effect
from workstation.batch_detection import record_mutation
from workstation.tests.test_durable_hardening import call


def agent():
    return SimpleNamespace(valid_tool_names={'work_execute'}, _work_repeatability_hint=True,
                           session_id='owner', _conversation_root_id=lambda: 'owner')


def test_repeatability_hint_does_not_block_stateful_browser():
    from workstation.execution_policy import CompilationDecision, decisions_for_calls
    a = agent()
    calls = [call(n, args) for n, args in [
        ('browser_navigate', {'url': 'https://example.com'}),
        ('browser_snapshot', {}), ('browser_type', {'ref': '@e32', 'text': 'hello'}),
        ('browser_press', {'key': 'Enter'}), ('browser_snapshot', {})]]
    assert all(d != CompilationDecision.REQUIRE_COMPILE for d in decisions_for_calls(a, calls))


def test_threshold_is_operation_scoped():
    from workstation.execution_policy import CompilationDecision, decisions_for_calls
    a = agent()
    calls = [call('write_file', {'path': f'{i}.json', 'content': 'x'}) for i in range(4)]
    assert decisions_for_calls(a, calls) == [CompilationDecision.ALLOW_ADAPTIVE,
        CompilationDecision.SUGGEST_COMPILE, CompilationDecision.REQUIRE_COMPILE,
        CompilationDecision.REQUIRE_COMPILE]
    for i in range(2):
        record_mutation(a, 'write_file', {'path': f'{i}.json', 'content': 'x'}, {'ok': True})
    assert decisions_for_calls(a, [calls[2]]) == [CompilationDecision.REQUIRE_COMPILE]
    assert decisions_for_calls(a, [call('browser_click', {'ref': '@e1'})]) == [CompilationDecision.ALLOW_ADAPTIVE]


def test_effects_have_one_authority():
    from agent.tool_guardrails import ToolCallGuardrailController
    guard = ToolCallGuardrailController()
    for name, expected in [('browser_snapshot', ToolEffect.PURE_READ),
                           ('browser_extract_items', ToolEffect.DISCOVERY),
                           ('browser_console', ToolEffect.MUTATION),
                           ('browser_type', ToolEffect.MUTATION)]:
        assert tool_effect(name) == expected
        assert guard._is_idempotent(name) == (expected in {ToolEffect.PURE_READ, ToolEffect.DISCOVERY})


def test_native_route_normalization_is_runtime_scoped():
    import pytest
    from workstation.routing import canonical_route_for_tool, require_allowed_route, ConstraintViolation
    assert canonical_route_for_tool('browser_type', runtime='internal') == 'native_browser'
    require_allowed_route('native_browser', {'allowed_routes': ['browser_type', 'browser_click']})
    with pytest.raises(ConstraintViolation):
        require_allowed_route('browser-exec', {'allowed_routes': ['browser_type']})
    assert canonical_route_for_tool('browser_unknown', runtime='internal') != 'native_browser'


def test_uncertain_adaptive_effect_requires_human():
    from workstation.execution_policy import CompilationDecision, decisions_for_calls
    a = agent()
    record_mutation(a, 'browser_press', {'key': 'Enter'}, {'error': 'connection lost'})
    assert decisions_for_calls(a, [call('browser_press', {'key': 'Enter'})]) == [CompilationDecision.REQUIRE_HUMAN]


def test_dispatched_adaptive_effect_survives_restart():
    from workstation.batch_detection import prepare_mutation
    from workstation.execution_policy import CompilationDecision, decisions_for_calls
    a = agent()
    prepare_mutation(a, 'browser_press', {'key': 'Enter'})
    restarted = agent()
    assert decisions_for_calls(restarted, [call('browser_press', {'key': 'Enter'})]) == [CompilationDecision.REQUIRE_HUMAN]


def test_final_arguments_cannot_bypass_dispatch_threshold():
    import pytest
    from workstation.batch_detection import prepare_mutation
    a = agent()
    for i in range(2):
        args = {'path': f'{i}.json', 'content': 'x'}
        prepare_mutation(a, 'write_file', args)
        record_mutation(a, 'write_file', args, {'ok': True})
    with pytest.raises(RuntimeError, match='REQUIRE_COMPILE'):
        prepare_mutation(a, 'write_file', {'path': 'middleware-rewritten.json', 'content': 'x'})
    assert sum(a._work_mutation_shapes.values()) == 2


def test_heterogeneous_stateful_browser_never_requires_compile():
    from workstation.execution_policy import CompilationDecision, decisions_for_calls
    a = agent()
    calls = [
        call('browser_navigate', {'url': 'https://example.com/login'}),
        call('browser_snapshot', {}),
        call('browser_type', {'ref': '@e1', 'text': 'user_a', 'semantic_anchor': {'type': 'testid', 'value': 'username'}}),
        call('browser_snapshot', {}),
        call('browser_click', {'ref': '@e2', 'semantic_anchor': {'type': 'testid', 'value': 'next_btn'}}),
        call('browser_snapshot', {}),
        call('browser_type', {'ref': '@e3', 'text': 'pass_123', 'semantic_anchor': {'type': 'testid', 'value': 'password'}}),
        call('browser_snapshot', {}),
        call('browser_click', {'ref': '@e4', 'semantic_anchor': {'type': 'testid', 'value': 'submit_btn'}}),
        call('browser_snapshot', {}),
        call('browser_type', {'ref': '@e5', 'text': 'search_term', 'semantic_anchor': {'type': 'testid', 'value': 'search_box'}}),
    ]
    decisions = decisions_for_calls(a, calls)
    assert all(d != CompilationDecision.REQUIRE_COMPILE for d in decisions)


def test_homogeneous_fanout_requires_compile_on_third_mutation():
    from workstation.execution_policy import CompilationDecision, decisions_for_calls
    a = agent()
    calls = [
        call('browser_type', {'ref': '@e1', 'text': 'val1', 'target_family': 'row_amount_field'}),
        call('browser_type', {'ref': '@e2', 'text': 'val2', 'target_family': 'row_amount_field'}),
        call('browser_type', {'ref': '@e3', 'text': 'val3', 'target_family': 'row_amount_field'}),
    ]
    decisions = decisions_for_calls(a, calls)
    assert decisions[0] == CompilationDecision.ALLOW_ADAPTIVE
    assert decisions[1] == CompilationDecision.SUGGEST_COMPILE
    assert decisions[2] == CompilationDecision.REQUIRE_COMPILE


def test_unknown_semantic_family_does_not_fabricate_homogeneity():
    from workstation.execution_policy import CompilationDecision, decisions_for_calls
    a = agent()
    calls = [
        call('browser_type', {'ref': '@e1', 'text': 'val1'}),
        call('browser_type', {'ref': '@e2', 'text': 'val2'}),
        call('browser_type', {'ref': '@e3', 'text': 'val3'}),
    ]
    decisions = decisions_for_calls(a, calls)
    assert decisions[2] != CompilationDecision.REQUIRE_COMPILE


def test_executed_unverified_does_not_count_as_verified_success():
    from workstation.execution_policy import decisions_for_calls
    a = agent()
    record_mutation(a, 'write_file', {'path': 'test.json', 'content': 'x'}, {'ok': True})
    decisions_for_calls(a, [call('write_file', {'path': 'test2.json', 'content': 'y'})])
    candidates = getattr(a, '_work_compilation_candidates', {})
    for cand in candidates.values():
        assert cand.executed_occurrences >= 1
        assert cand.verified_successes == 0

