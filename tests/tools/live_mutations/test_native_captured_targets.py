"""Write actual project documents through the real task-scoped local file backend."""
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def local_files(tmp_path):
    home, workspace = tmp_path / 'home', tmp_path / 'workspace'
    home.mkdir()
    workspace.mkdir()
    previous = os.environ.get('HERMES_HOME')
    os.environ['HERMES_HOME'] = str(home)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(home)
    from hermes_cli.config import atomic_config_write

    atomic_config_write(home / 'config.yaml', {'lsp': {'enabled': False}, 'terminal': {'backend': 'local'}})
    from tools import file_tools, terminal_tool

    task = uuid4().hex
    terminal_tool.register_task_env_overrides(task, {'env_type': 'local', 'cwd': str(workspace)})
    key = terminal_tool._resolve_container_task_id(task)
    operations = file_tools._get_file_ops(task)
    assert operations.env.is_local
    try:
        yield file_tools, terminal_tool, operations, task, home, workspace
    finally:
        operations.env.cleanup()
        with terminal_tool._env_lock:
            terminal_tool._active_environments.pop(key, None)
            terminal_tool._last_activity.pop(key, None)
        file_tools.clear_file_ops_cache(key)
        terminal_tool.clear_task_env_overrides(task)
        reset_hermes_home_override(token)
        if previous is None:
            os.environ.pop('HERMES_HOME')
        else:
            os.environ['HERMES_HOME'] = previous


@pytest.mark.parametrize('mode', ['replace', 'patch'])
def test_task_relative_patch_carries_its_full_write_baseline(local_files, mode):
    tools, _terminal, _operations, task, _home, workspace = local_files
    root = Path(__file__).resolve().parents[3]
    content = (root / 'README.md').read_text(encoding='utf-8')
    name = 'project-document.md'
    target = workspace / name
    written = json.loads(tools.write_file_tool(name, content, task_id=task))
    assert not written.get('error'), written
    assert Path(written['resolved_path']).resolve() == target.resolve()
    assert target.read_text(encoding='utf-8') == content
    original = next(line for line in content.splitlines() if line and line.upper() != line and content.count(line) == 1)
    replacement = original.upper()
    assert original != replacement
    if mode == 'replace':
        patched = json.loads(tools.patch_tool(mode=mode, path=name, old_string=original,
                                             new_string=replacement, task_id=task))
    else:
        patch = f'*** Begin Patch\n*** Update File: {name}\n@@\n-{original}\n+{replacement}\n*** End Patch\n'
        patched = json.loads(tools.patch_tool(mode=mode, patch=patch, task_id=task))
    assert not patched.get('error'), patched
    assert Path(patched['resolved_path']).resolve() == target.resolve()
    assert target.read_text(encoding='utf-8') == content.replace(original, replacement, 1)
    overwritten = json.loads(tools.write_file_tool(name, content, task_id=task))
    assert not overwritten.get('error'), overwritten
    assert target.read_text(encoding='utf-8') == content
    externally_changed = content + '\n' + content.splitlines()[0] + '\n'
    target.write_text(externally_changed, encoding='utf-8')
    refused = json.loads(tools.write_file_tool(name, content, task_id=task))
    assert refused.get('error'), refused
    assert target.read_text(encoding='utf-8') == externally_changed


def test_captured_identity_survives_a_real_session_cwd_change(local_files):
    tools, terminal, operations, task, home, workspace = local_files
    root = Path(__file__).resolve().parents[3]
    content = (root / 'README.md').read_text(encoding='utf-8')
    name = 'project-document.md'
    targets = tools._capture_mutation_targets([name], task)
    captured = targets[name]
    assert Path(captured.backend_path).resolve() == (workspace / name).resolve()
    other = workspace.parent / 'other-workspace'
    other.mkdir()
    terminal.register_task_env_overrides(task, {'env_type': 'local', 'cwd': str(other)})
    assert tools._write_precheck_error([name], [name], task, False, {name: captured.backend_path}) is None
    written = operations.write_file(captured.backend_path, content)
    assert not written.error, written
    assert (workspace / name).read_text(encoding='utf-8') == content
    assert not (other / name).exists()
    protected = tools._write_precheck_error([name], [name], task, False, {name: str(home / 'config.yaml')})
    assert protected and 'security-sensitive configuration' in protected
