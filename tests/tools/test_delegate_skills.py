import sys
sys.path.insert(0, '/root/hermes-agent')

def test_load_skill_content_returns_none_for_unknown():
    from tools.delegate_tool import _load_skill_content
    assert _load_skill_content('this-skill-does-not-exist-xyz') is None

def test_load_skill_content_finds_existing_skill():
    from pathlib import Path
    from tools.delegate_tool import _load_skill_content
    skills_dir = Path('/root/hermes-agent/skills')
    skill_names = [p.parent.name for p in skills_dir.rglob('SKILL.md')]
    if not skill_names:
        return
    content = _load_skill_content(skill_names[0])
    assert content is not None and len(content) > 0

def test_skill_injected_in_prompt(monkeypatch):
    import tools.delegate_tool as dt
    monkeypatch.setattr(dt, '_load_skill_content', lambda name: f'SKILL_CONTENT_{name}')
    from tools.delegate_tool import _build_child_system_prompt
    prompt = _build_child_system_prompt('do something', None, skills=['my-skill'])
    assert 'SKILL_CONTENT_my-skill' in prompt

def test_missing_skill_silently_skipped(monkeypatch):
    import tools.delegate_tool as dt
    monkeypatch.setattr(dt, '_load_skill_content', lambda name: None)
    from tools.delegate_tool import _build_child_system_prompt
    prompt = _build_child_system_prompt('goal', None, skills=['nonexistent'])
    assert 'SKILL_CONTENT' not in prompt

def test_no_skills_does_not_crash():
    from tools.delegate_tool import _build_child_system_prompt
    p1 = _build_child_system_prompt('goal', None, skills=None)
    p2 = _build_child_system_prompt('goal', None, skills=[])
    assert 'goal' in p1
