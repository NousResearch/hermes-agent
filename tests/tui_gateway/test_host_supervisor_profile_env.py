"""Positive control: same-profile compute host runs Hermes, not terminal code."""
import json
import os
from pathlib import Path
import sys
import queue
from unittest.mock import patch
import pytest


@pytest.mark.parametrize('multiplex', [False, True])
def test_same_profile_compute_host_keeps_operator_injected_turn_credentials(tmp_path, multiplex):
    home = tmp_path / 'profile'
    home.mkdir()
    (home / 'config.yaml').write_text('{}\n', encoding='utf-8')
    (home / '.env').write_text('', encoding='utf-8')
    keys = ['HASS_TOKEN', 'MODAL_TOKEN_ID', 'TELEGRAM_BOT_TOKEN', 'OPENAI_API_KEY']
    seed = {'PATH':'/usr/bin:/bin','HOME':str(tmp_path),'USER':'synthetic-audit',
            'HERMES_HOME':str(home),'PYTEST_CURRENT_TEST':'reconciliation','LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
            **{key:'fake-'+key.lower() for key in keys}}
    code = ('import json,os,sys; print(json.dumps({"type":"hello","seen":'
            + '{n:os.environ.get(n) for n in ' + repr(keys) + '}}),flush=True);sys.stdin.readline()')
    with patch.dict(os.environ, seed, clear=True), patch.object(Path,'home',return_value=tmp_path):
        from agent import secret_scope as ss
        ss.set_multiplex_active(multiplex)
        token = ss.set_secret_scope({key: seed[key] for key in keys}, profile_home=str(home))
        from tui_gateway.host_supervisor import HostSupervisor
        assert Path(sys.modules['tui_gateway.host_supervisor'].__file__).resolve().is_relative_to(Path.cwd())
        sup = HostSupervisor(registry_path=tmp_path/'host.json',argv=[sys.executable,'-I','-c',code],
                             cwd=tmp_path,expected_build_sha='unknown',autostart=False)
        try:
            sup.start()
            observed = sup._hello['seen']
            assert observed == {key:'fake-'+key.lower() for key in keys}
        finally:
            sup.shutdown()
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)


def test_foreign_host_respawn_keeps_captured_owner_and_refreshes_credentials(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    for home in (source, target):
        home.mkdir()
        (home / 'config.yaml').write_text('{}\n', encoding='utf-8')
    keys = ['HASS_TOKEN', 'MODAL_TOKEN_ID', 'TELEGRAM_BOT_TOKEN',
            'OPENAI_API_KEY', 'SOURCE_LOGIN', 'BENIGN_CONTROL', 'HERMES_HOME']
    source_values = {name: 'fake-source' for name in keys[:5]}
    (source / '.env').write_text(''.join(f'{k}={v}\n' for k, v in source_values.items()), encoding='utf-8')
    (target / '.env').write_text('HASS_TOKEN=fake-target\nOPENAI_API_KEY=fake-target\n', encoding='utf-8')
    seed = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path), 'HERMES_HOME': str(source),
            'BENIGN_CONTROL': 'keep', 'PYTEST_CURRENT_TEST': 'reconciliation', **source_values}
    code = ('import json,os,sys; print(json.dumps({"type":"hello"}),flush=True); '
            'print(json.dumps({"type":"rpc","message":'
            + '{n:os.environ.get(n) for n in ' + repr(keys) + '}}),flush=True); sys.stdin.readline()')
    seen = queue.Queue()
    with patch.dict(os.environ, seed, clear=True), patch.object(Path, 'home', return_value=tmp_path):
        from agent import secret_scope as ss
        from hermes_constants import pin_process_hermes_home
        from hermes_cli import env_loader
        from tui_gateway.host_supervisor import HostSupervisor
        ss.set_multiplex_active(False)
        pin_process_hermes_home(source)
        env_loader.reset_secret_source_cache()
        sup = HostSupervisor(registry_path=tmp_path / 'host.json',
                             argv=[sys.executable, '-I', '-c', code], cwd=tmp_path,
                             expected_build_sha='unknown', expected_hermes_home=str(target),
                             env={'HASS_TOKEN': 'fake-extra-source', 'SOURCE_LOGIN': 'fake-extra-source'},
                             rpc_sink=seen.put, autostart=False)
        try:
            sup.start()
            expected = dict.fromkeys(keys)
            expected.update(HASS_TOKEN='fake-target', OPENAI_API_KEY='fake-target',
                            BENIGN_CONTROL='keep', HERMES_HOME=str(target))
            assert seen.get(timeout=15) == expected
            old_pid = sup.pid
            (target / '.env').write_text('OPENAI_API_KEY=fake-rotated\n', encoding='utf-8')
            # A different caller's active home must not retarget the respawn.
            os.environ['HERMES_HOME'] = str(source)
            sup._proc.kill()
            expected.update(HASS_TOKEN=None, OPENAI_API_KEY='fake-rotated')
            assert seen.get(timeout=15) == expected
            assert sup.pid != old_pid
        finally:
            sup.shutdown()
            pin_process_hermes_home(None)
            env_loader.reset_secret_source_cache()
