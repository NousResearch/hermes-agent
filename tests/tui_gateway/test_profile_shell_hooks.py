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
        for label in ['alpha', 'beta', 'alpha', 'unapproved']:
            token = set_hermes_home_override(tmp_path / label)
            try:
                agent = server._make_agent(label, label, context_cwd_is_launch_artifact=False)
                assert agent is not None
                block = plugins.get_pre_tool_call_block_message('write_file', {'path': str(tmp_path / 'protected'), 'content': 'x'})
                assert block == (None if label == 'unapproved' else label)
                assert plugins.get_pre_tool_call_block_message('read_file', {'path': str(tmp_path / 'protected')}) is None
            finally:
                reset_hermes_home_override(token)
    finally:
        plugins._reset_plugin_managers_for_tests()
        shell_hooks.reset_for_tests()
