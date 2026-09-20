"""Minimal WSL host routing from #114565 onto Bot Screen (#108914).

The device is a real stdio MCP subprocess test double, NOT a Windows desktop.
Only WSL interop discovery/translation is substituted; config, resolver, contract,
backend, MCP handshake, enumeration, capture and teardown execute normally.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.computer_use import cua_backend as cb, cua_backend_driver as driver
from tools.computer_use import tool
from tools.computer_use.cua_backend_daemon import _EmbeddedCuaDaemon
from tools.computer_use_tool import registry

_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nGNgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
_SERVER = '''
import json, sys, time
from pathlib import Path
spec = json.loads(Path(__file__).with_suffix('.json').read_text())
args = sys.argv[1:]
with Path(__file__).with_suffix('.log').open('a') as log:
    log.write(json.dumps(args) + '\\n')
if args[0] == 'manifest':
    print(json.dumps(spec['manifest']))
elif args[0] in ('--version', '--help', 'check-update', 'doctor'):
    print(json.dumps({'ok': True, 'update_available': False}))
elif args[0] in ('serve', 'status', 'stop'):
    socket = args[args.index('--socket') + 1]
    assert socket.startswith('\\\\\\\\.\\\\pipe\\\\'), socket
    stop = Path(__file__).with_name(socket.split('-')[-1] + '.stop')
    if args[0] == 'stop': stop.touch()
    if args[0] == 'serve':
        while not stop.exists(): time.sleep(0.05)
else:
    assert args[0] == 'mcp', args
    names = ['list_windows', 'get_desktop_state', 'get_config', 'set_config',
             'start_session', 'end_session', 'set_agent_cursor_enabled']
    for line in sys.stdin:
        msg = json.loads(line)
        if 'id' not in msg: continue
        method = msg['method']
        if method == 'initialize':
            result = {'protocolVersion': msg['params']['protocolVersion'], 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'device-test-double', 'version': '1.0'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': n, 'inputSchema': {'type': 'object'}} for n in names]}
        else:
            name = msg['params']['name']
            payload = {'capture_scope': 'desktop'}
            if name == 'list_windows':
                payload = {'windows': [{'app_name': spec['label'], 'title': spec['label'],
                                        'pid': 123, 'window_id': 42, 'width': 8, 'height': 8}]}
            result = {'content': [{'type': 'text', 'text': json.dumps(payload)}], 'structuredContent': payload}
            if name == 'get_desktop_state':
                result['content'].append({'type': 'image', 'data': spec['png'], 'mimeType': 'image/png'})
        print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)
'''


@pytest.fixture
def interop(tmp_path, monkeypatch):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    monkeypatch.setenv('PATH', str(bindir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr('hermes_constants.is_wsl', lambda: True)
    monkeypatch.delenv('HERMES_CUA_DRIVER_CMD', raising=False)
    win = r"C:\Users\Alice O'Neil\AppData\Local\Programs\Cua\cua-driver\bin\cua-driver.exe"
    to_posix = lambda p: str(tmp_path / 'custom-mount' / 'c' / Path(*PureWindowsPath(p).parts[1:]))
    host = Path(to_posix(win))
    guest = bindir / 'cua-driver'
    for exe, label, command in [(host, 'HOST', win), (guest, 'GUEST', str(guest))]:
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text(f'#!{sys.executable}\n' + _SERVER)
        exe.chmod(0o755)
        exe.with_suffix('.json').write_text(json.dumps({'label': label, 'png': _PNG, 'manifest': {
            'binary_version': '0.28.2', 'mcp_invocation': {'command': command, 'args': ['mcp']},
            'subcommands': [{'name': n, 'args': [{'name': a} for a in sorted(args)]}
                            for n, args in driver._CUA_DRIVER_RUNTIME_CONTRACT_ARGS.items()]}}))
    for name in ['powershell.exe', 'wslpath']:
        p = bindir / name
        p.write_text('#!/bin/sh\nexit 1\n')
        p.chmod(0o755)
    real_run = cb._run_quiet
    calls = []

    def run(argv, **kwargs):
        if Path(argv[0]).name == 'powershell.exe':
            calls.append(argv)
            return SimpleNamespace(returncode=0, stdout=json.dumps([
                r"C:\Users\Alice O'Neil\AppData\Local", r"C:\Users\Alice O'Neil"]))
        if Path(argv[0]).name == 'wslpath':
            out = to_posix(argv[2]) if argv[1] == '-u' else r'\\wsl.localhost\Ubuntu\manifest.yaml'
            return SimpleNamespace(returncode=0, stdout=out)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(cb, '_run_quiet', run)
    if discovery := getattr(driver, '_wsl_windows_install_paths', None):
        discovery.cache_clear()
    tool.reset_backend_for_tests()
    yield host, guest, calls
    tool.reset_backend_for_tests()
    if discovery := getattr(driver, '_wsl_windows_install_paths', None):
        discovery.cache_clear()


@pytest.mark.linux_only
@pytest.mark.parametrize('permission', ['standard', 'unrestricted'])
def test_profile_selection_reaches_mcp_windows_and_capture(interop, tmp_path, monkeypatch, permission):
    host, guest, calls = interop
    # Keep approval behavior outside this routing test. No input is dispatched.
    monkeypatch.setattr(tool, '_cua_permission_mode', lambda sid: permission)
    homes = [tmp_path / 'windows-profile', tmp_path / 'guest-profile']
    for home, target in zip(homes, ['windows', 'linux']):
        home.mkdir()
        (home / 'config.yaml').write_text(f'computer_use:\n  target: {target}\n')
    for home, expected, binary in [(homes[0], 'HOST', host), (homes[1], 'GUEST', guest), (homes[0], 'HOST', host)]:
        # Private-daemon IPC is a Windows-only part of this test double.
        monkeypatch.setattr(tool, '_cua_permission_mode', lambda sid, home=home:
                            permission if home == homes[0] else 'standard')
        token = set_hermes_home_override(str(home))
        try:
            from tools.computer_use.permissions import computer_use_status
            assert driver.resolve_cua_driver_cmd() == str(binary)
            assert computer_use_status()['ready'] is True
            result = json.loads(registry.dispatch('computer_use', {'action': 'list_windows'}, session_id='same'))
            assert result['windows'][0]['app_name'] == expected, result
            backend = tool._get_backend('same')
            frame = backend.capture(mode='vision', app='screen')
            assert (frame.width, frame.height, frame.png_b64) == (8, 8, _PNG)
        finally:
            reset_hermes_home_override(token)
    assert len(calls) == 1  # successful root discovery is cached across status/runtime probes
    launches = [json.loads(line) for line in host.with_suffix('.log').read_text().splitlines()]
    assert any(args[0] == 'mcp' for args in launches)
    if permission == 'unrestricted':
        assert any(args[0] == 'serve' and args[args.index('--socket') + 1].startswith(r'\\.\pipe')
                   for args in launches)


@pytest.mark.linux_only
@pytest.mark.parametrize('failure', ['missing', 'timeout', 'json', 'partial'])
def test_discovery_failure_retries_and_selection_never_crosses_targets(interop, tmp_path, monkeypatch, failure):
    host, guest, _ = interop
    from hermes_cli.config import save_config
    save_config({'computer_use': {'target': 'windows'}})
    real_run, real_which = cb._run_quiet, driver.shutil.which
    with monkeypatch.context() as m:
        if failure == 'missing':
            m.setattr(driver.shutil, 'which', lambda name: None if name == 'powershell.exe' else real_which(name))
        else:
            def failed(argv, **kw):
                if Path(argv[0]).name == 'powershell.exe':
                    if failure == 'timeout':
                        raise driver.subprocess.TimeoutExpired(argv, 5)
                    return SimpleNamespace(returncode=0, stdout='["C:\\\\Users"]' if failure == 'partial' else '!')
                return real_run(argv, **kw)
            m.setattr(cb, '_run_quiet', failed)
        assert driver.resolve_cua_driver_cmd() is None  # the working Linux executable is not a fallback
    assert driver.resolve_cua_driver_cmd() == str(host)
    assert driver.resolve_cua_driver_cmd(str(guest)) is None
    alias = tmp_path / 'host-alias'
    alias.symlink_to(host)
    assert driver.is_windows_driver(str(alias))
    manifest = tmp_path / 'capability.yaml'
    manifest.write_text('version: 3\n')
    daemon = _EmbeddedCuaDaemon(str(alias), 'bounded', str(manifest))
    args = daemon._serve_args()
    assert daemon.socket_path.startswith(r'\\.\pipe')
    assert args[args.index('--capability-manifest') + 1] == r'\\wsl.localhost\Ubuntu\manifest.yaml'
    monkeypatch.setattr('tools.bot_desktop.runtime.published_env', lambda: {'DISPLAY': ':20'})
    with pytest.raises(ValueError, match='active Linux Bot Screen'):
        driver.resolve_cua_driver_cmd()


@pytest.mark.linux_only
@pytest.mark.parametrize("failure", ["nonzero", "empty", "multiline", "missing", "timeout"])
def test_manifest_translation_failure_is_a_retryable_tool_error(interop, tmp_path, monkeypatch, failure):
    """The existing tool boundary must refuse startup, not drop the manifest or cache a broken backend."""
    from hermes_cli.config import save_config

    host, _, _ = interop
    manifest = tmp_path / "capabilities.yaml"
    manifest.write_text("version: 3\n")
    save_config({"computer_use": {"target": "windows", "permission_mode": "bounded",
                                  "capability_manifest": str(manifest)}})
    real_run = cb._run_quiet
    translations = []

    def failed_translation(argv, **kwargs):
        if argv[:2] == ["wslpath", "-w"]:
            translations.append(argv)
            if failure == "missing":
                raise FileNotFoundError("wslpath is unavailable")
            if failure == "timeout":
                raise driver.subprocess.TimeoutExpired(argv, 3.0)
            output = {"nonzero": "ignored", "empty": "", "multiline": "first\nsecond"}[failure]
            return SimpleNamespace(returncode=1 if failure == "nonzero" else 0, stdout=output)
        return real_run(argv, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(cb, "_run_quiet", failed_translation)
        result = json.loads(registry.dispatch("computer_use", {"action": "capture", "app": "screen"},
                                              session_id="manifest-error"))
    assert translations == [["wslpath", "-w", str(manifest)]]
    assert "computer_use backend unavailable:" in result["error"], result
    message = "wslpath" if failure in {"missing", "timeout"} else "Cannot translate the capability manifest"
    assert message in result["error"], result
    assert not tool._backends
    launches = [json.loads(line) for line in host.with_suffix(".log").read_text().splitlines()]
    assert not any(args[0] in {"serve", "mcp"} for args in launches)

    # Retrying through the same public entry point succeeds once translation recovers.
    result = json.loads(registry.dispatch("computer_use", {"action": "list_windows"}, session_id="manifest-error"))
    assert result["windows"][0]["app_name"] == "HOST", result
    launches = [json.loads(line) for line in host.with_suffix(".log").read_text().splitlines()]
    serve = [args for args in launches if args[0] == "serve"]
    assert len(serve) == 1
    args = serve[0]
    assert args[args.index("--capability-manifest") + 1] == r"\\wsl.localhost\Ubuntu\manifest.yaml"
    assert "--approve-capability-manifest" in args
