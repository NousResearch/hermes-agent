import threading
from types import SimpleNamespace

from tui_gateway import server


class Peer:
    def __init__(self, frames, accepted=True):
        self.frames, self.accepted = frames, accepted

    def write(self, frame):
        if isinstance(self.accepted, Exception):
            raise self.accepted
        self.frames.append(frame['params']['type'])
        return self.accepted


def _skills_watch(monkeypatch, signature, *, peers=(), stdio=False):
    monkeypatch.setattr(server, '_live_transports', set(peers))
    monkeypatch.setattr(server, '_stdio_is_rpc_channel', stdio)
    monkeypatch.setattr(server, '_change_checked_at', {})
    monkeypatch.setattr(server, '_change_sigs', {})
    monkeypatch.setattr(server, '_change_broadcast_at', {})
    monkeypatch.setattr(server, '_CHANGE_WATCHES', {'skills.changed': (2.0, signature, dict)})


def test_without_a_peer_room_wake_keeps_running_but_skills_scan_does_not(monkeypatch):
    reached = threading.Event()
    scans = []
    workers = []

    def thread(target, **kwargs):
        def run():
            try:
                target()
            except StopIteration:
                pass

        worker = threading.Thread(target=run, daemon=True)
        workers.append(worker)
        return worker

    ticks = iter((10.0,))
    monkeypatch.setattr(server, 'threading', SimpleNamespace(Thread=thread))
    monkeypatch.setattr(server, 'time', SimpleNamespace(sleep=lambda _: next(ticks), monotonic=lambda: 10.0))
    monkeypatch.setattr(server, '_live_transports', set())
    monkeypatch.setattr(server, '_stdio_is_rpc_channel', False)
    monkeypatch.setattr(server, '_skin_watcher_started', False)
    monkeypatch.setattr(server, '_note_skin_broadcast', lambda: None)
    monkeypatch.setattr(server, '_broadcast_skin_if_changed', lambda: None)
    monkeypatch.setattr(server, '_change_checked_at', {})
    monkeypatch.setattr(server, '_change_sigs', {})
    monkeypatch.setattr(server, '_CHANGE_WATCHES', {
        'skills.changed': (2.0, lambda: scans.append('skills'), dict),
        'groups.changed': (0.5, lambda: scans.append('groups'), dict),
    })
    monkeypatch.setattr(server, '_wake_room_drivers', reached.set)
    server._ensure_skin_watcher()
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive()
    assert reached.is_set()
    assert scans == ['groups']


def test_reconnect_receives_skill_change_scanned_during_last_disconnect(monkeypatch):
    frames = []
    peer = Peer(frames)
    phase = 0

    def skill_signature():
        if phase == 1:
            server.unregister_live_transport(peer)
        return int(phase > 0)

    _skills_watch(monkeypatch, skill_signature, peers=[peer])
    server._broadcast_watched_changes(now=0.0)
    phase = 1
    server._broadcast_watched_changes(now=3.0)
    assert not frames
    phase = 2
    server.register_live_transport(peer)
    server._broadcast_watched_changes(now=6.0)
    assert frames == ['skills.changed']


def test_deferred_first_scan_invalidates_the_catalog_once(monkeypatch):
    frames = []
    scanned = []
    first_peer, reader, next_peer = Peer(frames), Peer(frames), Peer(frames)
    revision = 0

    def signature():
        scanned.append(revision)
        if reader in server._live_transports:
            server.unregister_live_transport(reader)
        return revision

    _skills_watch(monkeypatch, signature)
    server.register_live_transport(first_peer)
    server.unregister_live_transport(first_peer)
    server._broadcast_watched_changes(now=0.0)
    assert not scanned

    server.register_live_transport(reader)
    # The catalogue could have been read before the first watcher sample.
    revision = 1
    server._broadcast_watched_changes(now=3.0)
    assert scanned == [1]
    assert not frames

    server.register_live_transport(next_peer)
    server._broadcast_watched_changes(now=6.0)
    server._broadcast_watched_changes(now=9.0)
    assert frames == ['skills.changed']


def test_skill_change_waits_for_a_peer_that_accepts_the_write(monkeypatch):
    refused, wedged, accepted = [], [], []
    peers = [Peer(refused, accepted=False), Peer(wedged, accepted=RuntimeError('wedged'))]
    revision = 0
    _skills_watch(monkeypatch, lambda: revision, peers=peers)
    server._broadcast_watched_changes(now=0.0)
    revision = 1
    server._broadcast_watched_changes(now=3.0)
    assert refused == ['skills.changed'] and not wedged

    server.register_live_transport(Peer(accepted))
    server._broadcast_watched_changes(now=6.0)
    server._broadcast_watched_changes(now=9.0)
    assert accepted == ['skills.changed']


def test_global_broadcast_reports_whether_any_peer_accepted(monkeypatch):
    peers = {Peer([], accepted=False), Peer([], accepted=RuntimeError('wedged'))}
    monkeypatch.setattr(server, '_stdio_is_rpc_channel', False)
    monkeypatch.setattr(server, '_live_transports', peers)
    assert server._broadcast_global_event('skills.changed', {}) is False
    peers.add(Peer([]))
    assert server._broadcast_global_event('skills.changed', {}) is True


def test_stdio_seeds_silently_then_emits_one_skill_change(monkeypatch):
    emitted = []
    revision = 0
    _skills_watch(monkeypatch, lambda: revision, stdio=True)
    monkeypatch.setattr(server, '_emit', lambda event, sid, payload=None: emitted.append(event) or True)
    server._broadcast_watched_changes(now=0.0)
    assert not emitted
    revision = 1
    server._broadcast_watched_changes(now=3.0)
    server._broadcast_watched_changes(now=6.0)
    assert emitted == ['skills.changed']


def test_watcher_started_with_a_peer_sends_no_spurious_skill_change(monkeypatch):
    frames = []
    _skills_watch(monkeypatch, lambda: 0, peers=[Peer(frames)])
    for now in (0.0, 3.0, 6.0):
        server._broadcast_watched_changes(now=now)
    assert not frames
