"""Native skill writers serialize commits and batch rollback across processes."""

import contextvars
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
BEFORE = ('---\nname: mutation-probe\ndescription: Use when checking concurrent skill edits.\n'
          '---\nAlpha original.\nBeta original.\n')
CHILD = r'''
import json,sys,time
from pathlib import Path
import tools.skill_manager_tool as smt
import model_tools
from hermes_cli.plugins import get_pre_tool_call_directive
home,role,mode=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
target=home/'skills/mutation-probe/SKILL.md'
get_pre_tool_call_directive('skills_list', {})
if role=='a':
    original=smt.atomic_write_text
    paused=False
    def pause_at_commit(path,content,**kwargs):
        global paused
        if Path(path)==target and not paused:
            paused=True
            (home/'a-at-commit').write_text('ready')
            deadline=time.monotonic()+12
            while not (home/'release-a').exists():
                if time.monotonic()>deadline: raise RuntimeError('writer_release_timeout')
                time.sleep(.01)
        return original(path,content,**kwargs)
    smt.atomic_write_text=pause_at_commit
    args={'action':'patch','name':'mutation-probe','old_string':'Alpha original.','new_string':'Alpha changed.'}
    if mode=='batch_failure':
        args={'operations':[args,{'action':'write_file','name':'mutation-probe',
                                 'file_path':'invalid/file.md','file_content':'rejected'}]}
else:
    args={'action':'patch','name':'mutation-probe','old_string':'Beta original.','new_string':'Beta changed.'}
(home/(role+'-started')).write_text('ready')
result=model_tools.handle_function_call('skill_manage',args,task_id='lock-regression',session_id='isolated')
(home/(role+'-result.json')).write_text(result if isinstance(result,str) else json.dumps(result))
'''


def wait_for(path, process, timeout=12):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f'child exited before {path.name}: {stdout}\n{stderr}')
        if time.monotonic() >= deadline:
            raise AssertionError(f'timed out waiting for {path.name}')
        time.sleep(0.01)


@pytest.mark.parametrize('mode', ['single', 'batch_failure'])
def test_registry_process_writers_preserve_updates_during_commit_and_rollback(tmp_path, mode):
    home = tmp_path / 'profile'
    target = home / 'skills/mutation-probe/SKILL.md'
    target.parent.mkdir(parents=True)
    target.write_text(BEFORE)
    (home / 'config.yaml').write_text('{}\n')
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HERMES_HOME': str(home),
           'HERMES_TEST_ISOLATION': str(home), 'PYTHONPATH': str(ROOT),
           'PYTHONDONTWRITEBYTECODE': '1', 'TZ': 'UTC', 'LANG': 'C.UTF-8'}
    processes = []
    try:
        first = subprocess.Popen([sys.executable, '-B', '-c', CHILD, str(home), 'a', mode],
                                 cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(first)
        wait_for(home / 'a-at-commit', first)
        second = subprocess.Popen([sys.executable, '-B', '-c', CHILD, str(home), 'b', mode],
                                  cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(second)
        wait_for(home / 'b-started', second)
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline and not (home / 'b-result.json').exists():
            time.sleep(0.01)
        overlapped = (home / 'b-result.json').exists()
        (home / 'release-a').write_text('continue')
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, (stdout, stderr)
        first_result = json.loads((home / 'a-result.json').read_text())
        assert first_result['success'] is (mode == 'single')
        assert json.loads((home / 'b-result.json').read_text())['success'] is True
        actual = target.read_text()
        assert 'Beta changed.' in actual, 'Concurrent native update was overwritten.'
        assert ('Alpha changed.' in actual) is (mode == 'single')
        assert not overlapped, 'The second native writer committed before the first transaction ended.'
    finally:
        (home / 'release-a').touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()


def test_preimage_rejection_and_nested_native_dispatch(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.skill_mutation_lock import SkillMutationError, skill_mutation_lock
    import model_tools
    home = tmp_path / 'profile'
    target = home / 'skills/mutation-probe/SKILL.md'
    target.parent.mkdir(parents=True)
    target.write_text(BEFORE)
    monkeypatch.setenv('HERMES_HOME', str(home))
    token = set_hermes_home_override(home)
    try:
        with skill_mutation_lock(expected={target: BEFORE.encode()}, timeout=0.1):
            with skill_mutation_lock(timeout=0):
                result = model_tools.handle_function_call('skill_manage', {
                    'operations': [{'name': 'mutation-probe', 'action': 'patch',
                                    'old_string': 'Alpha original.', 'new_string': 'Alpha changed.'}]})
        assert json.loads(result)['success'] is True
        after = target.read_bytes()
        with pytest.raises(SkillMutationError, match='preimage'):
            with skill_mutation_lock(expected={target: BEFORE.encode()}):
                target.write_text('This must never be written.')
        with pytest.raises(SkillMutationError, match='preimage'):
            with skill_mutation_lock(expected={target: None}):
                pass
        assert target.read_bytes() == after
        absent = home / 'skills/new-probe/SKILL.md'
        with skill_mutation_lock(expected={absent: None}):
            assert not absent.exists()
    finally:
        reset_hermes_home_override(token)


def test_copied_context_cannot_bypass_lock_and_other_profile_is_independent(tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.skill_mutation_lock import SkillMutationError, skill_mutation_lock
    first_home, second_home = tmp_path / 'first', tmp_path / 'second'
    token = set_hermes_home_override(first_home)

    def other_profile():
        other_token = set_hermes_home_override(second_home)
        try:
            with skill_mutation_lock(timeout=0.1):
                return True
        finally:
            reset_hermes_home_override(other_token)

    def same_profile():
        with skill_mutation_lock(timeout=0.05):
            return 'incorrectly_entered'

    try:
        with skill_mutation_lock(), ThreadPoolExecutor(max_workers=2) as pool:
            copied = contextvars.copy_context()
            blocked = pool.submit(copied.run, same_profile)
            separate = pool.submit(other_profile)
            with pytest.raises(SkillMutationError, match='busy'):
                blocked.result(timeout=2)
            assert separate.result(timeout=2) is True
        assert (first_home / '.skill-mutation.lock').is_file()
        assert (second_home / '.skill-mutation.lock').is_file()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='POSIX process inheritance')
def test_fork_cannot_inherit_reentrancy_or_release_parent_lock(tmp_path):
    script = r'''
import os,select
from tools.skill_mutation_lock import SkillMutationError,skill_mutation_lock
reader,writer=os.pipe()
pid=None
try:
    with skill_mutation_lock():
        pid=os.fork()
        if pid==0:
            os.close(reader)
            outcome=b'entered'
            try:
                with skill_mutation_lock(timeout=.05): pass
            except SkillMutationError as exc:
                outcome=b'blocked' if str(exc)=='skill_mutation_busy' else b'error'
        else:
            os.close(writer)
            assert select.select([reader],[],[],3)[0], 'child cleanup did not finish'
            assert os.read(reader,64)==b'blocked', 'child inherited reentrancy or lost lock cleanup'
    if pid==0:
        os.write(writer,outcome)
        os.close(writer)
        os._exit(0)
    _,status=os.waitpid(pid,0)
    assert status==0
    with skill_mutation_lock(timeout=0): pass
except BaseException:
    if pid==0:
        os.write(writer,b'error')
        os._exit(1)
    raise
'''
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HERMES_HOME': str(tmp_path / 'profile'),
           'HERMES_TEST_ISOLATION': str(tmp_path), 'PYTHONPATH': str(ROOT),
           'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, (result.stdout, result.stderr)
