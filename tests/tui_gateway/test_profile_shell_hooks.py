"""Profile-bound agent construction must arm real shell policy."""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def test_agent_build_arms_only_consented_profile_policy(tmp_path, monkeypatch):
    from agent import shell_hooks
    from hermes_cli import plugins
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'os-home'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'os-home')
    monkeypatch.setenv('HERMES_IGNORE_RULES', '1')
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
    monkeypatch.delenv('HERMES_ACCEPT_HOOKS', raising=False)
    monkeypatch.setattr(server, '_hermes_home', tmp_path)
    monkeypatch.setattr(server, '_get_db', lambda: None)
    # Only provider resolution is a fixture; registration, agent, hook process,
    # consent, manager selection and public dispatch are the real implementation.
    monkeypatch.setattr(server, '_resolve_agent_model_runtime', lambda *_a: (
        'fixture-model', {'provider': 'openai-compat', 'base_url': 'http://127.0.0.1:18019/v1', 'api_key': 'fixture-key'}))
    plugins._reset_plugin_managers_for_tests()
    shell_hooks.reset_for_tests()
    try:
        for label, consent in [('alpha', True), ('beta', True), ('unapproved', False)]:
            home = tmp_path / label
            home.mkdir()
            script = home / 'guard.py'
            script.write_text('import json\nprint(json.dumps({"decision":"block","reason":' + repr(label) + '}))\n', encoding='utf-8')
            quote_command = subprocess.list2cmdline if os.name == 'nt' else shlex.join
            cfg = {'hooks_auto_accept': consent, 'hooks': {'pre_tool_call': [{
                'command': quote_command([sys.executable, str(script)]), 'matcher': 'write_file', 'fail_closed': True}]},
                'toolsets': ['file'], 'memory': {'memory_enabled': False, 'user_profile_enabled': False}}
            (home / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        alpha_manager = None
        beta_manager = None
        beta_callbacks = None
        for label in ['alpha', 'beta', 'alpha', 'unapproved']:
            token = set_hermes_home_override(tmp_path / label)
            try:
                agent = server._make_agent(label, label, context_cwd_is_launch_artifact=False)
                assert agent is not None
                block = plugins.get_pre_tool_call_block_message('write_file', {'path': str(tmp_path / 'protected'), 'content': 'x'})
                assert block == (None if label == 'unapproved' else label)
                assert plugins.get_pre_tool_call_block_message('read_file', {'path': str(tmp_path / 'protected')}) is None
                if label == 'beta':
                    beta_manager = plugins.get_plugin_manager()
                    beta_callbacks = list(beta_manager._hooks['pre_tool_call'])
                if label == 'alpha' and alpha_manager is None:
                    alpha_manager = plugins.get_plugin_manager()
                elif label == 'alpha':
                    assert alpha_manager is not None and beta_manager is not None
                    assert plugins.get_plugin_manager() is alpha_manager
                    home = tmp_path / label
                    # Consumer fixture, not the installed Atlas policy: exercise command replacement,
                    # both denial paths and the public manager dispatch without creating a card.
                    script = home / 'guard-v4.py'
                    script.write_text(
                        'import json, sys\np = json.load(sys.stdin)["tool_input"]\n'
                        'ok = p.get("Padrao") == "v4" and p.get("Parar") == "complete"\n'
                        'print(json.dumps({} if ok else {"decision":"block", "reason":"v4-denied"}))\n',
                        encoding='utf-8',
                    )
                    cfg = yaml.safe_load((home / 'config.yaml').read_text())
                    cfg['hooks']['pre_tool_call'][0]['command'] = quote_command([sys.executable, str(script)])
                    (home / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
                    def plugin(**kwargs):
                        if kwargs['args'].get('plugin_denied'):
                            return {'action': 'block', 'message': 'plugin-denied'}
                    alpha_manager._hooks['pre_tool_call'].append(plugin)
                    for _ in range(2):
                        rebuilt = server._make_agent(label, label, context_cwd_is_launch_artifact=False)
                        assert rebuilt is not None
                        rebuilt.close()
                        assert plugins.get_plugin_manager() is alpha_manager
                        assert len(alpha_manager._hooks['pre_tool_call']) == 2
                        valid = {'Padrao': 'v4', 'Parar': 'complete'}
                        assert plugins.get_pre_tool_call_block_message('write_file', valid) is None
                        for invalid in [{'Padrao': 'v3', 'Parar': 'complete'}, {'Padrao': 'v4'}]:
                            assert plugins.get_pre_tool_call_block_message('write_file', invalid) == 'v4-denied'
                        assert plugins.get_pre_tool_call_block_message('write_file', {
                            **valid, 'plugin_denied': True,
                        }) == 'plugin-denied'
                        assert beta_manager._hooks['pre_tool_call'] == beta_callbacks
                    # A new consumer must not disarm V4 when a second command lacks
                    # consent, including after the real force-loader clears callbacks.
                    marker = home / 'unapproved-ran'
                    extra = home / 'unapproved.py'
                    extra.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
                    cfg['hooks_auto_accept'] = False
                    cfg['hooks']['pre_tool_call'].append({
                        'command': quote_command([sys.executable, str(extra)]),
                        'matcher': 'write_file', 'fail_closed': True,
                    })
                    (home / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
                    for force in [False, True]:
                        if force:
                            alpha_manager.discover_and_load(force=True)
                        rebuilt = server._make_agent(label, label, context_cwd_is_launch_artifact=False)
                        rebuilt.close()
                        assert plugins.get_pre_tool_call_block_message('write_file', {
                            'Padrao': 'v4', 'Parar': 'complete',
                        }) is None
                        assert plugins.get_pre_tool_call_block_message('write_file', {
                            'Padrao': 'v3', 'Parar': 'complete',
                        }) == 'v4-denied'
                        assert not marker.exists()
                        assert beta_manager._hooks['pre_tool_call'] == beta_callbacks
                agent.close()
            finally:
                reset_hermes_home_override(token)
    finally:
        plugins._reset_plugin_managers_for_tests()
        shell_hooks.reset_for_tests()
