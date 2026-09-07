"""Supporting native lifetime checks, no Telegram/model calls or real profiles.

Exercise the actual methods_groups lifecycle and native config/path resolution.
Service construction is intercepted; the stop-timeout arm owns a real bounded thread.
This is not full configured-transport or real-interface acceptance.
"""
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / '.hermes'
    home = root / 'rooms' / 'isolated'
    home.mkdir(parents=True)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    from tui_gateway import methods_groups as module
    from tui_gateway import hosted_room_service as services

    created = []

    class Service:
        def __init__(self, server, *, db_path, profiles_root):
            self.db_path = db_path
            self.profiles_root = profiles_root
            self.starts = 0
            self.stops = 0
            self.can_stop = True
            created.append(self)

        def start(self):
            self.starts += 1

        def stop(self, *, timeout):
            self.stops += 1
            return self.can_stop

    monkeypatch.setattr(module, '_bound_server', SimpleNamespace())
    monkeypatch.setattr(module, '_service', None)
    monkeypatch.setattr(module, '_transport', None)
    monkeypatch.setattr(services, 'HostedRoomService', Service)
    return SimpleNamespace(root=root, home=home, module=module, created=created)


def test_native_absent_binding_owns_one_service_with_separate_profile_root(env):
    one = env.module.start_hosted_room_service()
    two = env.module.start_hosted_room_service()
    assert one is two
    assert env.created == [one]
    assert one.profiles_root == env.root
    assert one.db_path.parent == env.home
    assert env.module._transport is None
    assert env.module.stop_hosted_room_service(timeout=0)
    assert env.module._service is None


def test_native_live_companion_prevents_service_forgetting_until_cessation(env, monkeypatch):
    service = env.module.start_hosted_room_service()
    entered, release = threading.Event(), threading.Event()

    def work():
        entered.set()
        assert release.wait(10), 'test-owned worker was not released'

    worker = threading.Thread(target=work, name='native-lifetime-proof')
    worker.start()
    assert entered.wait(5)

    class Companion:
        def stop(self, *, timeout):
            worker.join(timeout)
            return not worker.is_alive()

    companion = Companion()
    monkeypatch.setattr(env.module, '_transport', companion)
    try:
        assert not env.module.stop_hosted_room_service(timeout=0)
        assert worker.is_alive()
        assert env.module._transport is companion
        assert env.module._service is service
        assert service.stops == 0
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert env.module.stop_hosted_room_service(timeout=0)
    assert env.module._transport is None
    assert env.module._service is None


def test_native_db_swap_does_not_replace_nonstopped_service(env, monkeypatch):
    from gateway import hosted_rooms
    old = env.module.start_hosted_room_service()
    old.can_stop = False
    monkeypatch.setattr(hosted_rooms, 'default_db_path', lambda: env.home / 'replacement.db')
    with pytest.raises(RuntimeError, match='was not replaced'):
        env.module.start_hosted_room_service()
    assert env.module._service is old
    assert env.created == [old]
    assert old.stops == 1
    old.can_stop = True
    assert env.module.stop_hosted_room_service(timeout=0)


@pytest.mark.parametrize('config', [
    'gateway: []\n',
    'gateway:\n  hosted_rooms: invalid\n',
    'gateway:\n  hosted_rooms:\n    telegram: []\n',
    'gateway:\n  hosted_rooms:\n    telegram:\n      binding_file: 17\n',
])
def test_native_malformed_binding_does_not_start_or_stop_any_worker(env, config):
    (env.home / 'config.yaml').write_text(config)
    with pytest.raises(ValueError, match='gateway.hosted_rooms.telegram.binding_file'):
        env.module.start_hosted_room_service()
    assert not env.created
    assert env.module._service is None
    assert env.module._transport is None



def test_member_ingress_config_does_not_claim_another_storage_transport(env, monkeypatch):
    from hermes_constants import get_hermes_home
    from plugins.platforms.telegram.hosted_room_transport import hosted_room_binding_path
    import json

    ingress_home = env.root / 'profiles' / 'alpha'
    ingress_home.mkdir(parents=True)
    binding_file = env.root / 'ingress-only.json'
    binding_file.write_text(json.dumps({'enabled': True}))
    (ingress_home / 'config.yaml').write_text(json.dumps({
        'gateway': {'hosted_rooms': {'telegram': {'binding_file': str(binding_file)}}}}))
    monkeypatch.setenv('HERMES_HOME', str(ingress_home))
    assert hosted_room_binding_path() == binding_file
    # The ingress document is deliberately incomplete for a consumer. The storage
    # owner has no transport configured, so a member gateway must not load it.
    service = env.module.start_hosted_room_service()
    assert service.db_path.parent == env.root
    assert env.module._transport is None
    assert get_hermes_home() == ingress_home
    assert env.module.stop_hosted_room_service(timeout=0)

    (env.root / 'config.yaml').write_text('gateway: []\n')
    with pytest.raises(ValueError, match='binding_file'):
        env.module.start_hosted_room_service()
    assert get_hermes_home() == ingress_home
    assert env.module._service is None
