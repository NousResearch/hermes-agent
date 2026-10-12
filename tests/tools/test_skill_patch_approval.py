"""Exact-anchor approval replay and interactive patch compatibility for #132822."""
import json
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.parametrize('shape', ['flat', 'batch'])
@pytest.mark.parametrize('anchor,replace_all,allowed', [
    ('Reviewed anchor.   ', False, False),
    ('  Reviewed anchor.', False, False),
    ('Reviewed  anchor.', False, False),
    ('Reviewed\tanchor.', False, False),
    ('Reviewed anchor.', False, True),
    ('Repeated anchor.', True, True),
    ('Repeated anchor.', False, False),
])
@pytest.mark.parametrize('named_launch', [False, True])
def test_approval_matches_exact_preview_without_changing_interactive_matching(
    tmp_path, monkeypatch, shape, anchor, replace_all, allowed, named_launch,
):
    from tools import skill_manager_tool, write_approval as wa  # registers skill_manage
    from tools.registry import registry
    from hermes_cli.write_approval_commands import handle_pending_subcommand

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    root = tmp_path / '.hermes'
    launch = root / 'profiles' / 'launch' if named_launch else root
    launch.mkdir(parents=True)
    (launch / 'config.yaml').write_text('{}', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(launch))
    skills = {}
    for profile in ('a', 'b', 'a'):
        home = root / 'profiles' / profile
        home.mkdir(parents=True, exist_ok=True)
        (home / 'config.yaml').write_text('{}', encoding='utf-8')
        token = set_hermes_home_override(home)
        try:
            if profile in skills:
                assert skills[profile][0].read_bytes() == skills[profile][1]
                continue
            skill = home / 'skills' / 'approval-probe'
            (skill / 'references').mkdir(parents=True)
            main = skill / 'SKILL.md'
            main.write_text('---\nname: approval-probe\ndescription: Use when reviewing a patch.\n---\n'
                            '# Review\n\n## When to Use\nReview a change.\n\n## Steps\nRead the anchor.\n', encoding='utf-8')
            target = skill / 'references' / 'review.md'
            target.write_text('Reviewed anchor.\nRepeated anchor.\nRepeated anchor.\nUnrelated prose.\n', encoding='utf-8')
            before = {p: p.read_bytes() for p in (main, target)}
            op = {'action': 'patch', 'name': 'approval-probe', 'file_path': 'references/review.md',
                  'old_string': anchor, 'new_string': 'Approved replacement.', 'replace_all': replace_all}
            payload = op if shape == 'flat' else {'action': 'batch', 'operations': [
                {'action': 'write_file', 'name': 'approval-probe', 'file_path': 'references/early.md',
                 'file_content': 'Must roll back on a later refusal.'}, op]}
            record = wa.stage_write(wa.SKILLS, payload, summary='Review supporting-file patch.', origin='foreground')
            preview = wa.skill_pending_diff(wa.get_pending(wa.SKILLS, record['id']))
            result = handle_pending_subcommand(wa.SKILLS, ['approve', record['id']])
            if allowed:
                assert 'Approved 1 skills write' in result, result
                assert 'patch would fail' not in preview and '+Approved replacement.' in preview
                assert target.read_text().count('Approved replacement.') == (2 if replace_all else 1)
                assert wa.get_pending(wa.SKILLS, record['id']) is None
            else:
                assert 'Approved 0 skills write' in result, result
                assert 'patch would fail' in preview, preview
                assert all(p.read_bytes() == data for p, data in before.items())
                assert not (skill / 'references' / 'early.md').exists()
                assert wa.get_pending(wa.SKILLS, record['id']) is not None
            assert main.read_bytes() == before[main]
            target.write_bytes(before[target])
            interactive = json.loads(registry.dispatch('skill_manage', {'operations': [op]}))
            assert interactive['success'] == (allowed or anchor != 'Repeated anchor.'), interactive
            if interactive['success']:
                assert 'Approved replacement.' in target.read_text()
            skills[profile] = (target, target.read_bytes())
        finally:
            reset_hermes_home_override(token)
    assert not (launch / 'skills').exists()
