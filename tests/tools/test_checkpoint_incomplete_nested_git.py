"""Real-Git checkpoint contracts for incomplete nested repository metadata."""
import subprocess

from tools import checkpoint_manager as cm


def test_incomplete_nested_git_directory_does_not_block_parent_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, 'CHECKPOINT_BASE', tmp_path / 'checkpoints')
    work = tmp_path / 'work'
    work.mkdir()
    nested = work / 'g2-observer-pilot'
    nested.mkdir()
    for args in (['init', '-q'], ['config', 'user.name', 'Test'],
                 ['config', 'user.email', 'test@example.invalid']):
        subprocess.run(['git', *args], cwd=nested, check=True, capture_output=True)
    (nested / 'notes.txt').write_text('uncaptured nested project\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=nested, check=True, capture_output=True)
    subprocess.run(['git', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'seed'],
                   cwd=nested, check=True, capture_output=True)
    (work / 'parent.txt').write_text('before\n', encoding='utf-8')
    manager = cm.CheckpointManager(enabled=True)
    assert manager.ensure_checkpoint(str(work), 'healthy nested repository')

    # Preserve the original metadata outside the worktree, then reproduce the
    # observed partial metadata directory on an already indexed gitlink.
    (nested / '.git').rename(tmp_path / 'saved-git-metadata')
    (nested / '.git' / 'objects').mkdir(parents=True)
    (work / 'parent.txt').write_text('changed\n', encoding='utf-8')
    manager.new_turn()
    assert manager.ensure_checkpoint(str(work), 'incomplete nested metadata')
    entry = manager.list_checkpoints(str(work))[0]
    assert 'nested git repos not captured: g2-observer-pilot' in entry['reason']
    ok, tree, error = cm._run_git(['ls-tree', '-r', '-z', entry['hash']], cm._store_path(), str(work))
    assert ok, error
    assert 'parent.txt' in tree
    assert '160000 commit ' in tree
    assert '/.git/' not in tree
    assert (nested / '.git' / 'objects').is_dir()
    result = manager.restore(str(work), entry['hash'], file_path='g2-observer-pilot/notes.txt')
    assert result['success'] is False
    assert result['nested_repositories'] == ['g2-observer-pilot']
