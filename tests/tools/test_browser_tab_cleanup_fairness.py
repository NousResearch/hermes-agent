"""Failed low-rowid targets cannot starve later bounded cleanup."""
from tools.browser_tab_ownership import OwnershipRegistry


def test_failed_targets_do_not_starve_later_targets(tmp_path):
    registry = OwnershipRegistry(tmp_path / 'tabs.sqlite')
    token = registry.admit('owner', 'run', 'browser', 'daemon')
    for target in ('failed-a', 'failed-b', 'ready'):
        registry.record_created(token, target)
    registry.finish(token)
    registry.retire('owner', 'run', evidence='fixture:finished')
    live = {'failed-a', 'failed-b', 'ready'}
    def close(target):
        if target.startswith('failed'):
            raise TimeoutError('held')
        live.remove(target)
    registry.cleanup('browser', lambda: set(live), close, limit=2)
    assert 'ready' in live
    registry.cleanup('browser', lambda: set(live), close, limit=2)
    assert live == {'failed-a', 'failed-b'}
    assert {row['target'] for row in registry.pending('browser', limit=2)} == live
