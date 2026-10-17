"""Exact skill previews, human revisions and dependent atomic approval in isolated homes."""
import json
from pathlib import Path

import gateway.run  # Bootstrap dependencies before the per-test real-home I/O guard.
import pytest

from tools import skill_manager_tool as sm, write_approval as wa
from tools import skill_review as review

ORIGINAL = '---\nname: sample\ndescription: Use when testing skill review.\n---\n\nKeep the original rule.\n'
REPLACEMENT = ORIGINAL.replace('original rule', 'revised rule')


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes'))
    home = tmp_path / '.hermes'
    home.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'skills': {'write_approval': False}}))
    assert json.loads(sm.skill_manage('create', 'sample', content=ORIGINAL))['success']
    (home / 'config.yaml').write_text(json.dumps({'skills': {'write_approval': True}}))
    return home


def stage(ops):
    result = json.loads(sm.skill_manage('', '', operations=ops))
    assert result['success'] and result['staged']
    return review.inspect(result['pending_id'])


@pytest.mark.parametrize('kind', ['patch', 'chain', 'edit', 'remove', 'create', 'delete'])
def test_preview_and_independent_approval_apply_exact_files(home, kind):
    root = home / 'skills/sample'
    (root / 'references').mkdir()
    supporting = root / 'references/guide.md'
    supporting.write_text('Old guide.\n')
    if kind == 'create':
        ops = [{'action': 'create', 'name': 'new-sample', 'content': ORIGINAL.replace('name: sample', 'name: new-sample')},
               {'action': 'write_file', 'name': 'new-sample', 'file_path': 'references/guide.md', 'file_content': 'New guide.\n'}]
    elif kind == 'delete':
        ops = [{'action': 'delete', 'name': 'sample'}]
    elif kind == 'remove':
        ops = [{'action': 'remove_file', 'name': 'sample', 'file_path': 'references/guide.md'}]
    else:
        ops = [{'action': 'patch', 'name': 'sample', 'old_string': 'original rule', 'new_string': 'revised rule'}]
        if kind == 'chain':
            ops.append({'action': 'patch', 'name': 'sample', 'old_string': 'revised rule', 'new_string': 'final rule'})
        if kind == 'edit':
            ops.append({'action': 'write_file', 'name': 'sample', 'file_path': 'references/guide.md', 'file_content': 'New guide.\n'})
    record = stage(ops)
    files = record['skill_review']['files']
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert all(Path(f['path']).is_absolute() for f in files)
    if kind == 'edit':
        record = review.revise(record['id'], 1, 0, REPLACEMENT + '\nHuman addition.\n')
        assert record['revision'] == 2 and record['skill_review']['approved'] == []
        files = record['skill_review']['files']
    for i in range(len(files)):
        outcome = review.decide(record['id'], record['revision'], i, 'approve')
        if i < len(files) - 1:
            assert not outcome['finished']
            assert {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
            assert not (home / 'skills/new-sample').exists()
    assert outcome['finished'] and wa.get_pending('skills', record['id']) is None
    for file in files:
        path = Path(file['path'])
        assert not path.exists() if file['after'] is None else path.read_text() == file['after']


@pytest.mark.parametrize('conflict', ['source', 'payload', 'invalid_edit', 'stale_revision', 'reject', 'sibling_failure', 'preview_error'])
def test_review_refuses_changed_sources_and_keeps_atomic_edits_safe(home, conflict):
    root = home / 'skills/sample'
    ops = [{'action': 'patch', 'name': 'sample', 'old_string': 'original rule', 'new_string': 'revised rule'},
           {'action': 'write_file', 'name': 'sample', 'file_path': 'references/guide.md', 'file_content': 'Guide.\n'}]
    if conflict == 'preview_error':
        ops[0]['old_string'] = 'not originally present'
        record = stage(ops)
        assert record.get('review_issue') and not record.get('skill_review')
        (root / 'SKILL.md').write_text(ORIGINAL + '\nnot originally present\n')
        assert not review.inspect(record['id']).get('skill_review')
        with pytest.raises(ValueError):
            review.decide(record['id'], 1, 0, 'approve')
        assert not (root / 'references/guide.md').exists()
        return
    record = stage(ops)
    assert review.decide(record['id'], 1, 0, 'approve')['finished'] is False
    # health: allow HX011 -- Parameterized conflict injection; branches alter distinct approval inputs and share the refusal assertions.
    if conflict == 'source':
        (root / 'SKILL.md').write_text(ORIGINAL + '\nConcurrent change.\n')
    elif conflict == 'payload':
        current = wa.get_pending('skills', record['id'])
        current['payload']['operations'][0]['new_string'] = 'Unreviewed text'
        review.save(current)
    elif conflict == 'invalid_edit':
        with pytest.raises(ValueError):
            review.revise(record['id'], 1, 0, 'Missing frontmatter')
        assert wa.get_pending('skills', record['id'])['skill_review']['approved'] == [0]
        return
    elif conflict == 'stale_revision':
        edited = review.revise(record['id'], 1, 0, REPLACEMENT)
        assert edited['skill_review']['approved'] == []
    elif conflict == 'reject':
        assert review.decide(record['id'], 1, 1, 'reject')['finished']
        assert (root / 'SKILL.md').read_text() == ORIGINAL
        assert not (root / 'references/guide.md').exists()
        return
    elif conflict == 'sibling_failure':
        # Existing apply validation fails only after the first op; it must roll back.
        record = review.revise(record['id'], 1, 1, 'Guide.\n')
        from unittest.mock import patch
        with patch.object(sm, '_security_scan_skill', side_effect=[None, 'Test scan refusal.']):
            review.decide(record['id'], 2, 0, 'approve')
            with pytest.raises(ValueError, match='batch aborted'):
                review.decide(record['id'], 2, 1, 'approve')
        assert (root / 'SKILL.md').read_text() == ORIGINAL
        assert not (root / 'references/guide.md').exists()
        return
    before = (root / 'SKILL.md').read_bytes()
    with pytest.raises(ValueError):
        review.decide(record['id'], 1, 1, 'approve')
    assert (root / 'SKILL.md').read_bytes() == before
    assert not (root / 'references/guide.md').exists()
