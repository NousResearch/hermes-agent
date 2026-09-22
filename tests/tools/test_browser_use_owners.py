"""Task-owner contracts for the Harness carrier (#100865); no real browsers."""
import threading
from pathlib import Path

import pytest
from tools import browser_use_cli as bu


def registry(monkeypatch, tmp_path):
    assert hasattr(bu, '_HarnessOwnerRegistry'), 'persistent Harness has no task-owner lifecycle'
    from tools import browser_tool
    from tools import browser_tool_lifecycle as lifecycle
    monkeypatch.setattr(browser_tool, '_socket_safe_tmpdir', lambda: str(tmp_path))
    monkeypatch.setattr(lifecycle, '_start_browser_cleanup_thread', lambda: None)
    stops = []
    monkeypatch.setattr(bu, '_stop_harness_resource', lambda r: stops.append(r.key) or True)
    return bu._HarnessOwnerRegistry(), stops


def test_browser_exec_persists_shared_owners_until_agent_hard_close(monkeypatch, tmp_path):
    import json
    import sys
    import run_agent
    from agent.client_lifecycle import ClientLifecycleMixin
    from tools import browser_tool, browser_tool_lifecycle as lifecycle
    from tools.process_registry import process_registry
    monkeypatch.setattr(browser_tool, '_socket_safe_tmpdir', lambda: str(tmp_path))
    monkeypatch.setattr(lifecycle, '_start_browser_cleanup_thread', lambda: None)
    monkeypatch.setattr(bu, '_route_backend', lambda *args: None)
    monkeypatch.setattr(bu, '_attach_vault_supervisor', lambda *args: None)
    monkeypatch.setattr(bu, '_base_subprocess_env', lambda: {})
    # Real child process and entrypoint; no Harness/browser install or network.
    monkeypatch.setattr(bu, '_find_cli', lambda: [sys.executable, '-c',
        'import json,os,sys;sys.stdin.read();print(json.dumps(dict(os.environ)))'])
    stops = []
    monkeypatch.setattr(bu, '_stop_harness_resource', lambda resource: stops.append(resource.key) or True, raising=False)
    if hasattr(bu, '_HarnessOwnerRegistry'):
        monkeypatch.setattr(bu, '_harness_owners', bu._HarnessOwnerRegistry())
    first = json.loads(bu.browser_exec('print(1)', session='shared', task_id='cron:run-A'))
    second = json.loads(bu.browser_exec('print(2)', session='shared', task_id='sibling'))
    assert first['success'] and second['success']
    first_env, second_env = json.loads(first['output']), json.loads(second['output'])
    assert first_env.get('BH_RUNTIME_DIR'), 'browser_exec did not register a managed Harness runtime'
    assert first_env['BH_RUNTIME_DIR'] == second_env['BH_RUNTIME_DIR']
    assert stops == []  # normal tool return is not a hard owner boundary
    monkeypatch.setattr(process_registry, 'list_sessions', lambda: [])
    monkeypatch.setattr(run_agent, 'cleanup_vm', lambda task: None)
    monkeypatch.setattr(run_agent, 'cleanup_browser', lambda task: None)
    agent = ClientLifecycleMixin()
    agent._process_owner_task_ids = {'cron:run-A'}
    agent._close_task_resources('session-A')
    assert stops == []
    sibling = ClientLifecycleMixin()
    sibling._process_owner_task_ids = {'sibling'}
    sibling._close_task_resources('session-B')
    assert len(stops) == 1
    late = json.loads(bu.browser_exec('print(3)', session='shared', task_id='sibling'))
    assert 'closed' in late['error']


def test_owners_persist_until_final_close_and_timeout_can_retry(monkeypatch, tmp_path):
    r, stops = registry(monkeypatch, tmp_path)
    for owner in ('cron:run-A', 'sibling'):
        with r.operation(owner, env={'BU_NAME': 'same'}) as resource:
            key = resource.key
            resource.started = True  # model a successfully launched CLI
    r.release('cron:run-A')
    assert stops == []
    monkeypatch.setattr(bu, '_stop_harness_resource', lambda resource: False)
    r.release('sibling')
    assert key in r._resources
    assert Path(resource.runtime).exists()
    monkeypatch.setattr(bu, '_stop_harness_resource', lambda resource: stops.append(resource.key) or True)
    gate = r.release('sibling')  # retained by the owning agent until it is retired
    assert gate is not None
    assert stops == [key]
    assert key not in r._resources
    r.release('sibling')
    assert stops == [key]
    with pytest.raises(RuntimeError, match='closed'):
        with r.operation('sibling', env={'BU_NAME': 'same'}):
            pytest.fail('closed owner resurrected')


def test_reservation_precedes_operation_lock_and_close_drains(monkeypatch, tmp_path):
    r, stops = registry(monkeypatch, tmp_path)
    entered = threading.Event()
    done = threading.Event()
    errors = []
    with r.operation('first', env={'BU_NAME': 'same'}) as resource:
        resource.started = True
        def queued():
            try:
                with r.operation('second', env={'BU_NAME': 'same'}):
                    entered.set()
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()
        thread = threading.Thread(target=queued)
        thread.start()
        # Condition not sleep: reservation itself signals metadata transitions.
        with r._changed:
            assert r._changed.wait_for(lambda: resource.in_flight == 2, timeout=5)
        assert not entered.is_set()
        r.release('first')
        r.release('second')
        assert stops == []
    assert done.wait(5)
    thread.join(5)
    assert not errors
    assert entered.is_set()
    assert stops == [resource.key]


def test_cleanup_fences_only_its_runtime_and_releases_after_error(monkeypatch, tmp_path):
    r, _ = registry(monkeypatch, tmp_path)
    runtime = str(tmp_path / 'being-reaped')
    done = threading.Event()
    errors = []

    def unrelated():
        try:
            with r.operation('other', env={'BH_RUNTIME_DIR': str(tmp_path / 'other')}):
                pass
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=unrelated)
    try:
        with pytest.raises(ValueError, match='reaper failed'):
            with r.guard_cleanup(runtime) as allowed:
                assert allowed
                thread.start()
                assert done.wait(2), 'cleanup blocks unrelated endpoint reservations'
                with pytest.raises(RuntimeError, match='cleanup'):
                    with r.operation('same', env={'BH_RUNTIME_DIR': runtime}):
                        pytest.fail('reservation raced cleanup')
                raise ValueError('reaper failed')
    finally:
        thread.join(5)
    assert not errors
    with r.operation('same', env={'BH_RUNTIME_DIR': runtime}):
        pass


def test_first_dispatch_after_hard_close_is_refused(monkeypatch, tmp_path):
    import json
    import sys
    import run_agent
    from agent.client_lifecycle import ClientLifecycleMixin
    from tools.process_registry import process_registry
    r, _ = registry(monkeypatch, tmp_path)
    monkeypatch.setattr(bu, '_harness_owners', r)
    monkeypatch.setattr(bu, '_base_subprocess_env', lambda: {})
    monkeypatch.setattr(bu, '_attach_vault_supervisor', lambda *a: None)
    monkeypatch.setattr(bu, '_find_cli', lambda: [sys.executable, '-c', 'print("late spawn")'])
    monkeypatch.setattr(process_registry, 'list_sessions', lambda: [])
    monkeypatch.setattr(run_agent, 'cleanup_vm', lambda task: None)
    monkeypatch.setattr(run_agent, 'cleanup_browser', lambda task: None)
    routed, resume = threading.Event(), threading.Event()
    def route(*args):
        routed.set()
        assert resume.wait(5)
    monkeypatch.setattr(bu, '_route_backend', route)
    results = []
    thread = threading.Thread(target=lambda: results.append(json.loads(bu.browser_exec('print(1)', task_id='first-turn'))))
    agent = ClientLifecycleMixin()
    agent._process_owner_task_ids = {'first-turn'}
    thread.start()
    try:
        assert routed.wait(5)
        agent._close_task_resources('session')
    finally:
        resume.set()
        thread.join(5)
    assert results and 'closed' in results[0].get('error', ''), results
    assert not r._resources


def test_profile_scope_returns_to_original_owner(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    @contextmanager
    def hermes_home_scope(home):
        token = set_hermes_home_override(home)
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    r, stops = registry(monkeypatch, tmp_path)
    a, b = tmp_path / 'home-a', tmp_path / 'home-b'
    with hermes_home_scope(a):
        with r.operation('same-task', env={'BU_NAME': 'same'}) as first:
            first.started = True
    with hermes_home_scope(b):
        with r.operation('same-task', env={'BU_NAME': 'same'}) as second:
            second.started = True
        assert second.key != first.key
        gate = r.release('same-task')
        assert stops == [second.key]
    with hermes_home_scope(a):
        with r.operation('same-task', env={'BU_NAME': 'same'}) as again:
            assert again is first
        gate_a = r.release('same-task')
        assert stops == [second.key, first.key]
    assert gate is not None and gate_a is not None


def test_pre_spawn_failure_drops_registry_without_deleting_ambiguous_files(monkeypatch, tmp_path):
    import json
    import run_agent
    from agent.client_lifecycle import ClientLifecycleMixin
    from tools import browser_tool, browser_tool_lifecycle as lifecycle
    monkeypatch.setattr(browser_tool, '_socket_safe_tmpdir', lambda: str(tmp_path))
    monkeypatch.setattr(lifecycle, '_start_browser_cleanup_thread', lambda: None)
    monkeypatch.setattr(bu, '_route_backend', lambda *a: None)
    monkeypatch.setattr(bu, '_attach_vault_supervisor', lambda *a: None)
    monkeypatch.setattr(bu, '_base_subprocess_env', lambda: {})
    monkeypatch.setattr(bu, '_find_cli', lambda: [str(tmp_path / 'missing.exe')])
    r = bu._HarnessOwnerRegistry()
    monkeypatch.setattr(bu, '_harness_owners', r)
    result = json.loads(bu.browser_exec('print(1)', task_id='never-spawned'))
    assert 'Failed to launch' in result['error']
    resource = next(iter(r._resources.values()))
    assert not resource.started
    r.release('never-spawned')
    assert not r._resources
    assert Path(resource.runtime).exists()  # not evidence of death; do not delete


@pytest.mark.parametrize('override', ['BH_RUNTIME_DIR', 'BH_TMP_DIR'])
def test_explicit_runtime_is_external(monkeypatch, tmp_path, override):
    r, stops = registry(monkeypatch, tmp_path)
    external = tmp_path / 'external'
    env = {override: str(external), 'BU_NAME': 'alpha'}
    before = dict(env)
    with r.operation('task', env=env) as first:
        assert not first.managed
    with r.operation('other', env={override: str(external), 'BU_NAME': 'beta'}) as second:
        assert first.key == second.key  # v0.1.9 bare bu stem, not BU_NAME
    r.release('task')
    r.release('other')
    assert env == before
    assert not external.exists()
    assert stops == []


@pytest.mark.parametrize('lost_ack', [False, True])
def test_shutdown_ack_is_not_exit_and_missing_endpoint_retries(monkeypatch, tmp_path, lost_ack):
    from tools import browser_tool_lifecycle as lifecycle
    import gateway.status as status
    from types import SimpleNamespace
    import sys
    r, _ = registry(monkeypatch, tmp_path)
    with r.operation('task', env={'BU_NAME': 'alpha'}) as resource:
        runtime = Path(resource.runtime)
        (runtime / 'bu.pid').write_text('12345')
    monkeypatch.setattr(lifecycle, 'BROWSER_PID_FILE_SETTLE_SECONDS', 0)
    assert hasattr(bu, '_stop_harness_resource')
    # Import optional adapter only after an assertion meaningful on the base.
    from tools import browser_use_ipc as ipc
    alive = [True]
    monkeypatch.setattr(status, '_pid_exists', lambda pid: alive[0])
    monkeypatch.setattr(status, 'get_process_start_time', lambda pid: 77 if alive[0] else None)
    monkeypatch.setattr(lifecycle, '_verify_reapable_browser_daemon', lambda *a, **kw: True)
    endpoint = Path(resource.key)
    endpoint.write_text('{"port":1234,"token":"unit-only"}')
    calls = []
    def request(evidence, meta, deadline):
        calls.append(meta)
        if meta == 'ping':
            return {'pong': True, 'pid': 12345}
        endpoint.unlink()  # upstream tears down IPC before completing shutdown
        if lost_ack:
            raise TimeoutError('response lost after shutdown was delivered')
        return {'ok': True}
    monkeypatch.setattr(ipc, '_request', request)
    assert not ipc.stop_managed_harness(resource, timeout=.01)
    assert resource.evidence is not None
    assert runtime.exists()
    assert (runtime / 'bu.pid').exists()
    alive[0] = False
    assert ipc.stop_managed_harness(resource, timeout=.01)
    assert calls == ['ping', 'shutdown']
    assert not runtime.exists()


def test_harness_live_process_owner_never_uses_idle_escape_hatch(monkeypatch, tmp_path):
    from tools import browser_tool_lifecycle as lifecycle
    from unittest.mock import Mock
    runtime = tmp_path / 'agent-browser-hermes_bh_123_test'
    runtime.mkdir()
    (runtime / 'bu.pid').write_text('456')
    monkeypatch.setattr(lifecycle, '_owner_pid_alive', lambda *args: (123, True))
    monkeypatch.setattr(lifecycle, '_socket_dir_idle_seconds', lambda *args: 999999)
    verify = Mock(return_value=True)
    monkeypatch.setattr(lifecycle, '_verify_reapable_browser_daemon', verify)
    assert lifecycle._reap_socket_dir(str(runtime), 'hermes_bh_123_test', set()) is False
    assert runtime.exists()
    verify.assert_not_called()
