"""Restart notices compare saved choices with real process-consumed plugin config."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RPC_BOOT = """
import json
from hermes_cli.plugins import discover_plugins, get_plugin_manager
from tui_gateway import server
import sys
from functools import partial
print = partial(print, file=sys.__stdout__)

def rpc(action='list', **params):
    response = server.handle_request({'id': 'restart-test', 'method': 'plugins.manage',
                                      'params': {'action': action, **params}})
    assert 'result' in response, response
    return response['result']

def row(result, name='restart-probe'):
    return next(p for p in result['plugins'] if p['name'] == name)
"""
BOOT = 'from hermes_cli.plugins import discover_plugins\ndiscover_plugins()\n' + RPC_BOOT


def run_backend(home, script):
    env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'LANG') if key in os.environ}
    env.update(HOME=str(home.parent), USERPROFILE=str(home.parent), HERMES_HOME=str(home),
               PYTHONUTF8='1', PATH=str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', ''))
    proc = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(next(line.removeprefix('RESULT:') for line in proc.stdout.splitlines()
                           if line.startswith('RESULT:')))


def native_home(tmp_path, enabled):
    home = tmp_path / 'runtime'
    plugin = home / 'plugins' / 'restart-probe'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: restart-probe\nversion: "1.0"\n', encoding='utf-8')
    (plugin / '__init__.py').write_text(
        'def register(ctx):\n    ctx.register_hook("on_session_end", lambda **kwargs: None)\n',
        encoding='utf-8')
    (home / 'config.yaml').write_text(json.dumps({'plugins': {
        'enabled': ['restart-probe'] if enabled else [], 'disabled': []}}), encoding='utf-8')
    return home


@pytest.mark.parametrize('initially_enabled', [False, True])
@pytest.mark.parametrize('startup', ['native', 'dashboard'])
def test_native_notice_survives_saved_config_and_new_client_until_fresh_backend(tmp_path, initially_enabled, startup):
    home = native_home(tmp_path, initially_enabled)
    dashboard = home / 'plugins' / 'restart-probe' / 'dashboard'
    dashboard.mkdir()
    (dashboard / 'manifest.json').write_text(json.dumps({'name': 'restart-probe', 'api': 'api.py'}),
                                            encoding='utf-8')
    (dashboard / 'api.py').write_text(
        'from fastapi import APIRouter\nrouter = APIRouter()\n'
        '@router.get("/probe")\ndef probe():\n    return {"alive": True}\n', encoding='utf-8')
    boot = BOOT if startup == 'native' else 'from hermes_cli import web_server\n' + RPC_BOOT
    loaded_expr = ("get_plugin_manager()._plugins['restart-probe'].enabled" if startup == 'native' else
                   "any(getattr(r, 'path', '') == '/api/plugins/restart-probe/probe' for r in web_server.app.routes)")
    observed = run_backend(home, boot + f"""
initial = rpc()
changed = rpc('toggle', name='restart-probe', enable={not initially_enabled!r})
# New requests model a renderer reload: no client-side latch or baseline is retained.
refreshed = rpc()
noop = rpc('toggle', name='restart-probe', enable={not initially_enabled!r})
reverted = rpc('toggle', name='restart-probe', enable={initially_enabled!r})
rpc('toggle', name='restart-probe', enable={not initially_enabled!r})
loaded = {loaded_expr}
discover_plugins(force=True)
rediscovered = rpc()
print('RESULT:' + json.dumps(dict(initial=initial, changed=changed, refreshed=refreshed,
                                 noop=noop, reverted=reverted, loaded=loaded, rediscovered=rediscovered)))
""")
    assert observed['initial'].get('restart_required') is False
    assert observed['initial']['plugins']
    for state in ('changed', 'refreshed', 'noop'):
        assert observed[state]['restart_required'] is True
        plugin = (observed[state]['plugin'] if state != 'refreshed' else
                  next(p for p in observed[state]['plugins'] if p['name'] == 'restart-probe'))
        assert plugin['restart_required'] is True
        assert plugin['applies_on'] == 'backend_restart'
        assert (plugin['status'] == 'enabled') is not initially_enabled
    assert observed['noop']['unchanged'] is True
    assert observed['rediscovered']['restart_required'] is True  # Discovery cannot remount API routes.
    # A cold web import has no native manager yet. The first toggle's toolset
    # refresh discovers native hooks AFTER saving, so reverting then disagrees
    # with that native configuration even though it matches mounted routes.
    assert observed['reverted']['restart_required'] is (startup == 'dashboard')
    assert observed['loaded'] is initially_enabled  # Saving did not remount routes or reload existing hooks.

    restarted = run_backend(home, boot + "print('RESULT:' + json.dumps(rpc()))")
    assert restarted['restart_required'] is False
    plugin = next(p for p in restarted['plugins'] if p['name'] == 'restart-probe')
    assert plugin['restart_required'] is False
    assert (plugin['status'] == 'enabled') is not initially_enabled


def test_portable_changes_require_restart_without_claiming_another_profiles_runtime(tmp_path):
    import shutil

    home = native_home(tmp_path, False)
    unknown = run_backend(home, RPC_BOOT + "print('RESULT:' + json.dumps(rpc()))")
    assert unknown['restart_required'] is None
    assert next(p for p in unknown['plugins'] if p['name'] == 'restart-probe')['restart_required'] is None
    portable = home / 'plugins' / 'portable-probe'
    portable.mkdir()
    from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1
    (portable / 'plugin.json').write_text(
        json.dumps({'$schema': PLUGIN_SCHEMA_V1, 'name': 'portable-probe'}), encoding='utf-8')
    other = home / 'profiles' / 'other'
    shutil.copytree(home / 'plugins', other / 'plugins')
    shutil.copy2(home / 'config.yaml', other / 'config.yaml')
    observed = run_backend(home, BOOT + """
changed = rpc('toggle', name='portable-probe', enable=True)
refreshed = rpc()
foreign = rpc('toggle', profile='other', name='restart-probe', enable=True)
# A scoped session cannot vouch for another profile's API-route runtime.
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_cli.profiles import get_profile_dir
token = set_hermes_home_override(get_profile_dir('other'))
try:
    discover_plugins()
finally:
    reset_hermes_home_override(token)
foreign_refreshed = rpc(profile='other')
own = rpc()
print('RESULT:' + json.dumps(dict(changed=changed, refreshed=refreshed,
    foreign=foreign, foreign_refreshed=foreign_refreshed, own=own)))
""")
    assert observed['changed']['plugin']['portable'] is True
    assert observed['changed']['plugin']['applies_on'] == 'backend_restart'
    assert observed['changed']['plugin']['restart_required'] is True
    for state in ('changed', 'refreshed', 'own'):
        assert observed[state]['restart_required'] is True
    foreign_portable = next(p for p in observed['foreign_refreshed']['plugins'] if p['name'] == 'portable-probe')
    assert foreign_portable['restart_required'] is None
    assert observed['foreign']['restart_required'] is None
    assert observed['foreign']['plugin']['restart_required'] is None
    foreign_native = next(p for p in observed['foreign_refreshed']['plugins'] if p['name'] == 'restart-probe')
    assert foreign_native['restart_required'] is None
    assert foreign_native['status'] == 'enabled'
    assert observed['foreign_refreshed']['restart_required'] is None


@pytest.mark.parametrize('initially_enabled', [False, True])
def test_portable_skill_and_mcp_stay_cached_across_save_and_new_session(tmp_path, initially_enabled):
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1

    home = tmp_path / 'runtime'
    plugin = home / 'plugins' / 'portable-probe'
    skill = plugin / 'skills' / 'probe'
    skill.mkdir(parents=True)
    (plugin / 'plugin.json').write_text(json.dumps({
        '$schema': PLUGIN_SCHEMA_V1, 'name': 'portable-probe'}), encoding='utf-8')
    (skill / 'SKILL.md').write_text(
        '---\nname: probe\ndescription: Portable activation probe.\n---\nPortable skill body.\n',
        encoding='utf-8')
    mcp_server = plugin / 'probe_server.py'
    mcp_server.write_text(
        'from mcp.server import MCPServer\n'
        'mcp = MCPServer("portable-probe")\n'
        '@mcp.tool()\ndef activation_probe() -> str:\n    return "portable MCP body"\n'
        'mcp.run(transport="stdio")\n', encoding='utf-8')
    (plugin / 'mcp.json').write_text(json.dumps({
        '$schema': MCP_SCHEMA_V1,
        'mcpServers': {'worker': {'type': 'stdio', 'command': Path(sys.executable).name,
                                  'args': [str(mcp_server)]}}}), encoding='utf-8')
    (home / 'config.yaml').write_text(json.dumps({'plugins': {
        'enabled': ['portable-probe'] if initially_enabled else [], 'disabled': []}}), encoding='utf-8')
    probe = BOOT + """
from unittest.mock import patch
from tools.skills_tool import skills_list, skill_view
from tools.mcp_tool_config import _load_mcp_config
from tools.mcp_tool_discovery import discover_mcp_tools, get_mcp_status
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
from tools.mcp_tool_schema import mcp_prefixed_tool_name
from tools.registry import registry
namespace = get_plugin_manager()._plugins['portable-probe'].manifest.skill_namespace
skill_name, mcp_name = namespace + ':probe', namespace + '__worker'
tool_name = mcp_prefixed_tool_name(mcp_name, 'activation_probe')

def new_session():
    # Only suppress unrelated provider prewarming. Real RPC creation, ownership
    # publication and its plugin discovery run; consumers below run without mocks.
    with patch.object(server, '_schedule_agent_build'):
        response = server.handle_request({'id': 'create', 'method': 'session.create', 'params': {}})
    assert 'result' in response, response
    return response['result']['session_id']

def components(sid):
    skills = json.loads(skills_list(task_id=sid))
    assert skills['success'], skills
    viewed = json.loads(skill_view(skill_name, task_id=sid))
    configured = _load_mcp_config()
    discover_mcp_tools(allowed_mcp_names=[mcp_name])
    return dict(skill_listed=any(s['name'] == skill_name for s in skills['skills']),
                skill_served=viewed.get('success', False),
                mcp_configured=mcp_name in configured,
                mcp_connected=any(s['name'] == mcp_name and s['connected'] for s in get_mcp_status()),
                mcp_registered=tool_name in registry.get_all_tool_names())
"""
    observed = run_backend(home, probe + f"""
try:
    old_sid = new_session()
    initial = components(old_sid)
    initial_notice = rpc()
    changed = rpc('toggle', name='portable-probe', enable={not initially_enabled!r})
    saved = components(old_sid)
    new_sid = new_session()
    assert old_sid != new_sid
    new = components(new_sid)
    refreshed = rpc()
    noop = rpc('toggle', name='portable-probe', enable={not initially_enabled!r})
    reverted = rpc('toggle', name='portable-probe', enable={initially_enabled!r})
    rpc('toggle', name='portable-probe', enable={not initially_enabled!r})
    print('RESULT:' + json.dumps(dict(initial=initial, saved=saved, new=new,
        initial_notice=initial_notice, changed=changed, refreshed=refreshed, noop=noop, reverted=reverted)))
finally:
    shutdown_mcp_servers()
""")
    # Real skill serving and MCP stdio handshake/registry stay on the first discovery.
    for stage in ('initial', 'saved', 'new'):
        assert observed[stage] == dict.fromkeys(
            ('skill_listed', 'skill_served', 'mcp_configured', 'mcp_connected', 'mcp_registered'),
            initially_enabled)
    assert observed['initial_notice']['restart_required'] is False
    for stage in ('changed', 'refreshed', 'noop'):
        assert observed[stage]['restart_required'] is True
    assert observed['changed']['plugin']['applies_on'] == 'backend_restart'
    assert (observed['changed']['plugin']['status'] == 'enabled') is not initially_enabled
    assert observed['noop']['unchanged'] is True
    assert observed['reverted']['restart_required'] is False

    restarted = run_backend(home, probe + """
try:
    print('RESULT:' + json.dumps(dict(components=components(new_session()), notice=rpc())))
finally:
    shutdown_mcp_servers()
""")
    assert restarted['components'] == dict.fromkeys(observed['initial'], not initially_enabled)
    assert restarted['notice']['restart_required'] is False
