from io import StringIO
from pathlib import Path
from tools.skill_usage import _iter_skill_mds, _read_skill_name


def test_curator_reads_only_the_name_header():
    class HeaderOnly:
        def open(self, **kwargs):
            class Reader(StringIO):
                def read(self, size=-1):
                    assert 0 < size <= 4000
                    return super().read(size)
            return Reader('---\nname: declared-name\n---\n' + 'large body' * 1000)
    assert _read_skill_name(HeaderOnly(), 'fallback') == 'declared-name'


def test_curator_prunes_excluded_trees_without_losing_visible_names(tmp_path, monkeypatch):
    visible = tmp_path / 'ops' / 'visible'
    visible.mkdir(parents=True)
    (visible / 'SKILL.md').write_text('---\nname: declared-name\n---\nBody')
    excluded = tmp_path / 'node_modules' / 'ignored'
    excluded.mkdir(parents=True)
    (excluded / 'SKILL.md').write_text('---\nname: ignored\n---')
    original = Path.open
    def checked_open(path, *args, **kwargs):
        assert 'node_modules' not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', checked_open)
    assert list(_iter_skill_mds(tmp_path, local_only=False)) == [('declared-name', visible / 'SKILL.md')]
    from tools import skill_usage
    monkeypatch.setattr(skill_usage, '_skills_dir', lambda: tmp_path)
    monkeypatch.setattr(skill_usage, '_read_hub_installed_names', lambda: set())
    monkeypatch.setattr(skill_usage, '_read_bundled_manifest_names', lambda: set())
    monkeypatch.setattr(skill_usage, 'load_usage', lambda: {})
    def unexpected_lookup(name):
        raise AssertionError('known-local scan must not re-resolve every name')
    monkeypatch.setattr(skill_usage, '_find_skill_dir', unexpected_lookup)
    assert skill_usage.list_unmanaged_skill_names() == ['declared-name']
