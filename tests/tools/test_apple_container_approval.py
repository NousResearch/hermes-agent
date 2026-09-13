"""Apple isolation must preserve unattended execution and explicit deny rules."""
import json

import pytest

from tools.environments.apple_container_provider import apple_container_has_host_access


@pytest.mark.parametrize('extra_args,volumes,guarded', [
    ([], [], False),
    (['--network', 'none', '--tmpfs', '/scratch'], [], False),
    (['--mount', 'type=bind,source=/tmp,target=/mnt'], [], True),
    (['--mount=type=bind,source=/tmp,target=/mnt'], [], True),
    (['-v/tmp:/mnt'], [], True),
    (['-iv', '/tmp:/mnt'], [], True),
    (['--label', '--volume=/tmp:/mnt'], [], True),
    (['--ssh'], [], True),
    ([], ['/tmp:/mnt:ro'], True),
])
def test_unattended_code_reaches_backend_only_when_isolated(monkeypatch, tmp_path, extra_args, volumes, guarded):
    import tools.code_execution_tool as code_tool
    import tools.terminal_tool as terminal
    from tools.approval import check_all_command_guards

    (tmp_path / 'config.yaml').write_text('approvals:\n  cron_mode: deny\n  deny:\n    - "echo forbidden"\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_CRON_SESSION', '1')
    monkeypatch.setenv('TERMINAL_ENV', 'apple_container')
    monkeypatch.setenv('TERMINAL_APPLE_CONTAINER_EXTRA_ARGS', json.dumps(extra_args))
    monkeypatch.setenv('TERMINAL_APPLE_CONTAINER_VOLUMES', json.dumps(volumes))
    called = []
    def remote(*args, **kwargs):
        called.append(args)
        return json.dumps({'status': 'success', 'output': 'executed'})
    monkeypatch.setattr(code_tool, '_execute_remote', remote)
    config = terminal._get_env_config()
    assert apple_container_has_host_access(config) is guarded
    result = json.loads(code_tool.execute_code('print(42)', task_id='approval-test'))
    assert bool(called) is not guarded
    assert (result.get('status') == 'success') is not guarded
    # Operator intent is enforced even when the container fast path applies.
    verdict = check_all_command_guards('echo forbidden', 'apple_container', has_host_access=guarded)
    assert verdict['approved'] is False
    assert 'approvals.deny' in verdict['message']


def test_registry_backend_cannot_be_shadowed_or_lost_on_plugin_reset():
    from agent import terminal_env_registry as registry
    from tools.terminal_tool_config import _is_container_backend
    from tools.environments.apple_container_provider import AppleContainerProvider

    with pytest.raises(ValueError):
        registry.register_provider(AppleContainerProvider())
    registry._reset_for_tests()
    assert isinstance(registry.get_provider('apple_container'), AppleContainerProvider)
    assert _is_container_backend('apple_container')
    assert registry.provider_flag('apple_container', 'is_remote') is True
