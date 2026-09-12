"""Probe and work outages share bounded, restart-durable alternate turns."""
import io
import urllib.error

from gateway import hosted_room_peer as peer, hosted_rooms as rooms
from gateway import hosted_room_work_records as work
from hermes_cli import urllib_security
from tests.tui_gateway.test_hosted_room_replication import pair, SECRET  # noqa: F401
from tests.tui_gateway.test_work_record_delivery_fairness import successor_copying
from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR
from tui_gateway.hosted_room_replication import HostedRoomReplicationPublisher


def test_offline_preferred_probe_cannot_starve_recovered_work_alternate(pair, monkeypatch):
    successor_copying(pair, monkeypatch, 'unavailable')
    underlying = urllib_security.open_credentialed_url
    offline_probes = []

    def transport(request, *, timeout):
        token = request.get_header('Authorization').removeprefix('HermesRoom ')
        claims = peer.decode_room_grant(SECRET, token, permission='replicate')
        if request.full_url.endswith('/capabilities') and claims['member_id'] == 'reviewer':
            offline_probes.append(True)
            raise urllib.error.HTTPError(request.full_url, 503, 'offline', {}, io.BytesIO(b'{}'))
        return underlying(request, timeout=timeout)

    monkeypatch.setattr(urllib_security, 'open_credentialed_url', transport)
    pub = HostedRoomReplicationPublisher(pair.source)
    # Give both routes real attempts, leaving an anchored record after transient loss.
    for member in ('reviewer', 'z-other', 'reviewer', 'z-other'):
        route = pub._load_route(('room', member))
        pub._checkpoint(route)
        pub._publish_locked(route)
    assert pair.attempts == ['z-other']
    pair.recovered = True
    pub = HostedRoomReplicationPublisher(pair.source)
    before = len(offline_probes)
    for turn in range(6):
        rooms.append_event(pair.source, room_id='room', event_id=f'review-busy-{turn}', kind='message.user',
                           actor={'kind': 'user', 'id': 'owner'}, payload={'text': 'new'},
                           authority_gateway_id=SUCCESSOR, authority_epoch=2)
        pub._publish_one(('room', 'z-other' if turn % 2 == 0 else 'reviewer'))
    observed = next(r for r in pub.status('room')['work_records'] if r['producer_epoch'] == 2)['status']
    print('AUTOMATIC WORK', observed, 'OFFLINE PROBES', len(offline_probes) - before,
          'WORK ATTEMPTS', pair.attempts)
    # Positive control: the unmodified alternate transport/auth/ingest now succeeds.
    pub._publish_locked(pub._load_route(('room', 'z-other')))
    assert next(r for r in pub.status('room')['work_records'] if r['producer_epoch'] == 2)['status'] == 'acked'
    assert observed == 'acked', 'route selection must give the recovered alternate a turn'

def test_quiet_transient_work_failures_rotate_without_new_history(pair, monkeypatch):
    successor_copying(pair, monkeypatch, 'unavailable')
    pub = HostedRoomReplicationPublisher(pair.source)
    for member in ('reviewer', 'z-other', 'reviewer', 'z-other'):
        route = pub._load_route(('room', member))
        pub._checkpoint(route)
        pub._publish_locked(route)
    before = len(pair.attempts)
    # Even repeatedly asking for one key may not pin automatic retries there.
    for _ in range(4):
        pub._publish_one(('room', 'reviewer'))
    assert set(pair.attempts[before:]) == {'reviewer', 'z-other'}
    assert all(a != b for a, b in zip(pair.attempts[before:], pair.attempts[before + 1:]))


def test_probe_outage_remains_retryable_with_exact_generation_cache(pair, monkeypatch):
    import json
    from tests.tui_gateway.test_hosted_room_replication import TARGET
    successor_copying(pair, monkeypatch, 'unavailable')
    underlying = urllib_security.open_credentialed_url
    recovered = False
    probes = []

    def transport(request, *, timeout):
        token = request.get_header('Authorization').removeprefix('HermesRoom ')
        claims = peer.decode_room_grant(SECRET, token, permission='replicate')
        if claims['member_id'] == 'reviewer':
            if request.full_url.endswith('/capabilities'):
                probes.append(True)
                if not recovered:
                    raise urllib.error.HTTPError(request.full_url, 503, 'offline', {}, io.BytesIO(b'{}'))
            if recovered and request.full_url.endswith('/work-records'):
                record = json.loads(request.data)['record']
                ack = work.ingest(pair.target, record=record, token=token, secret=SECRET,
                    target_install_id=TARGET, target_profile=claims['target_profile'])
                return io.BytesIO(json.dumps(ack).encode())
        return underlying(request, timeout=timeout)

    monkeypatch.setattr(urllib_security, 'open_credentialed_url', transport)
    pub = HostedRoomReplicationPublisher(pair.source)
    for member in ('reviewer', 'z-other', 'reviewer', 'z-other'):
        route = pub._load_route(('room', member))
        pub._checkpoint(route)
        pub._publish_locked(route)
    with rooms._transaction(pair.source) as conn:
        frozen = conn.execute(f'SELECT record_json FROM {work.PENDING_TABLE} WHERE producer_epoch=2').fetchone()[0]
    recovered = True
    pub = HostedRoomReplicationPublisher(pair.source)
    for _ in range(4):
        pub._publish_one(('room', 'z-other'))
    assert next(r for r in pub.status()['work_records'] if r['producer_epoch'] == 2)['status'] == 'acked'
    assert len(probes) == 3  # two offline attempts, one successful negotiation
    with rooms._transaction(pair.target) as conn:
        assert conn.execute(f'SELECT record_json FROM {work.TARGET_TABLE} WHERE producer_epoch=2').fetchone()[0] == frozen
