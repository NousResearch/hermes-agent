"""Profile-bound agent construction must arm real shell policy."""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import hermes_yaml as yaml


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


def test_agent_build_arms_profile_outbound_webhooks(tmp_path, monkeypatch):
    """hooks.outbound must arm on the TUI/dashboard agent-build surface too:
    entrypoints (cli/gateway/serve startup) that never construct agents here
    would otherwise leave Desktop sessions emitting zero webhook events."""
    from agent import outbound_webhooks
    from hermes_cli import plugins
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'os-home'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'os-home')
    monkeypatch.setattr(server, '_hermes_home', tmp_path)
    monkeypatch.setattr(server, '_get_db', lambda: None)
    # Same fixture boundary as the shell-policy test: only provider resolution
    # is mocked; registration, manager selection and the agent build are real.
    monkeypatch.setattr(server, '_resolve_agent_model_runtime', lambda *_a: (
        'fixture-model', {'provider': 'openai-compat', 'base_url': 'http://127.0.0.1:18019/v1', 'api_key': 'fixture-key'}))
    plugins._reset_plugin_managers_for_tests()
    outbound_webhooks.reset_for_tests()
    url = 'http://127.0.0.1:1/never-delivered'
    try:
        home = tmp_path / 'wired'
        home.mkdir()
        cfg = {'hooks': {'outbound': [{
            'name': 'probe', 'url': url, 'events': ['on_session_start', 'post_tool_call']}]},
            'toolsets': ['file'], 'memory': {'memory_enabled': False, 'user_profile_enabled': False}}
        (home / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        token = set_hermes_home_override(home)
        try:
            agent = server._make_agent('wired', 'wired', context_cwd_is_launch_artifact=False)
            assert agent is not None
            manager = plugins.get_plugin_manager()
            assert manager._hooks.get('on_session_start'), 'on_session_start callback not wired'
            assert manager._hooks.get('post_tool_call'), 'post_tool_call callback not wired'
            # Idempotence: a second agent build for the same home adds no dupes.
            before = len(manager._hooks['on_session_start'])
            agent2 = server._make_agent('wired2', 'wired2', context_cwd_is_launch_artifact=False)
            assert agent2 is not None
            assert len(manager._hooks['on_session_start']) == before
        finally:
            reset_hermes_home_override(token)
    finally:
        plugins._reset_plugin_managers_for_tests()
        outbound_webhooks.reset_for_tests()
