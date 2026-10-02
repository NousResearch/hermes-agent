"""Freeze negotiated output consent before a peer attempt can be sent."""
import hashlib
import json

from gateway import hosted_rooms
from gateway.hosted_room_peer import GatewayRoomCatalog
from gateway.hosted_room_peer_output import OUTPUT_FEATURE, output_contract


def consent_key(scope):
    coordinates = {key: scope[key] for key in ('room_id', 'authority_gateway_id', 'authority_epoch',
        'member_id', 'task_id', 'execution_generation')}
    return 'group.peer-output.v1.' + hashlib.sha256(json.dumps(coordinates, sort_keys=True).encode()).hexdigest()


def stored_consent(db_path, scope):
    with hosted_rooms._transaction(db_path) as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (consent_key(scope),)).fetchone()
    return json.loads(row[0]) if row else None


def task_output(db_path, binding, task, route, client, *, negotiate=True):
    if not getattr(client, 'proof_install_id', None):
        return None
    generation = int(task.get('execution_generation') or 0) + (task.get('status') == 'queued')
    scope = dict(room_id=binding.room_id, authority_gateway_id=binding.gateway_id,
                 authority_epoch=binding.authority_epoch, member_id=route.member_id,
                 task_id=task['identity'].task_id, execution_generation=generation)
    saved = stored_consent(db_path, scope)
    target = dict(target_install_id=route.target_install_id, target_profile=route.target_profile,
                  home_install_id=route.home_install_id, capability_digest=route.capability_digest,
                  execution_policy_digest=route.execution_policy_digest, cancellation_scope_id=route.cancellation_scope_id,
                  trace_id=route.trace_id)
    if saved is not None:
        if saved['target'] != target:
            raise ValueError('peer output recipient changed')
        return output_contract(saved['contract'])
    if task.get('status') == 'queued' and not negotiate:
        return None
    contract = None
    # Existing/uncertain attempts predate this consent and retain their original wire shape.
    if task.get('status') == 'queued' and getattr(client, 'proof_install_id', None):
        response = client.probe(grant=route.grant, features=OUTPUT_FEATURE)
        live = GatewayRoomCatalog.from_mapping(response.get('catalog'))
        if (live.installation_id != route.target_install_id or live.catalog_digest != route.capability_digest
                or any(response.get(key) != value for key, value in scope.items()
                       if key not in {'task_id', 'execution_generation'})):
            raise ValueError('peer output capability scope changed')
        contract = output_contract(response.get('document_output'))
    encoded = json.dumps({'target': target, 'contract': contract, 'dispatched': False}, sort_keys=True, separators=(',', ':'))
    with hosted_rooms._transaction(db_path, immediate=True) as conn:
        conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)', (consent_key(scope), encoded))
        if conn.execute('SELECT value FROM state_meta WHERE key=?', (consent_key(scope),)).fetchone()[0] != encoded:
            raise ValueError('peer output consent changed')
    return contract


def mark_dispatched(db_path, dispatch):
    if db_path is None:
        raise ValueError('peer output requires durable source intent')
    key = consent_key(dispatch.as_mapping())
    with hosted_rooms._transaction(db_path, immediate=True) as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        saved = json.loads(row[0]) if row else None
        if (saved is None or saved['contract'] != dispatch.document_output
                or any(getattr(dispatch, name) != value for name, value in saved['target'].items())):
            raise ValueError('peer output consent unavailable')
        if saved.get('dispatch') not in (None, dispatch.as_mapping()):
            raise ValueError('peer output dispatch changed')
        saved['dispatch'] = dispatch.as_mapping()
        saved['dispatched'] = True
        conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                     (json.dumps(saved, sort_keys=True, separators=(',', ':')), key))


class OutputPending(RuntimeError):
    retryable = True


class PeerOutputSource:
    """Existing durable obligations own work; bounded futures keep I/O off the policy lock."""
    def __init__(self, service, scope, manifest):
        from gateway.hosted_room_artifacts import RoomArtifactError
        from tui_gateway.hosted_room_driver import HostedRoomBinding
        self.service, self.scope, self.manifest = service, scope, manifest
        self.generation = getattr(service, "_peer_output_generation", 0)
        saved = stored_consent(service.db_path, scope.as_mapping())
        if saved is None or output_contract(saved['contract']) is None:
            raise RoomArtifactError('Peer output has no frozen consent')
        target = saved['target']
        if (target['target_install_id'] != scope.target_install_id or target['target_profile'] != scope.target_profile
                or target['home_install_id'] != scope.home_install_id
                or service._owned_authority(scope.room_id) != (scope.authority_gateway_id, scope.authority_epoch)):
            raise RoomArtifactError('Peer output owner changed')
        key = scope.room_id, scope.member_id
        with service._policy_lock:
            route, client = service.peer_routes.get(key), service.peer_clients.get(key)
            if route is None or client is None or any(getattr(route, name) != value for name, value in target.items()):
                raise RoomArtifactError('Peer output route changed')
            self.route = route
            self.client = service._track_peer_client(HostedRoomBinding(scope.room_id, scope.authority_gateway_id,
                                                                       scope.authority_epoch), key, route, client)
            self.endpoint = client.base_url
        if not hasattr(service, '_peer_output_io'):
            from collections import OrderedDict
            service._peer_output_io = OrderedDict()

    def _start(self, operation, fields):
        from concurrent.futures import Future
        from agent.memory_provider import spawn_context_thread
        body = dict(scope=self.scope.as_mapping(), manifest_digest=self.manifest['manifest_digest'] if self.manifest else None,
                    operation=operation, **fields)
        if (getattr(self.service, '_peer_output_stopping', False)
                or self.generation != getattr(self.service, '_peer_output_generation', 0)):
            raise OutputPending('Peer output owner is stopping')
        route_key = (self.route.trace_id, self.route.cancellation_scope_id, hashlib.sha256(self.route.grant.encode()).hexdigest())
        key = (self.scope.key, self.endpoint, json.dumps(body, sort_keys=True), route_key, self.generation)
        table = self.service._peer_output_io
        if key in table:
            table.move_to_end(key)
            return key, table[key]
        if sum(not f.done() for f in table.values()) >= 4:
            return key, None
        while len(table) >= 16:
            victim = next((k for k, f in table.items() if f.done()), None)
            if victim is None:
                return key, None
            table.pop(victim)
        future = table[key] = Future()
        def execute():
            try:
                if (getattr(self.service, '_peer_output_stopping', False)
                        or self.generation != getattr(self.service, '_peer_output_generation', 0)):
                    raise OutputPending('Peer output owner is stopping')
                result = self.client.output_request(grant=self.route.grant, **body)
                if operation == 'read':
                    item = next(item for item in self.manifest['items'] if item['artifact_id'] == fields['artifact_id'])
                    if (set(result) != {'metadata', 'data_base64'} or result['metadata'] != item
                            or not isinstance(result['data_base64'], str)
                            or len(result['data_base64']) != ((item['size'] + 2) // 3) * 4):
                        raise ValueError('peer output response differs from frozen manifest')
                else:
                    keys = ('acknowledged', 'changed') if operation == 'ack' else ('discarded', 'removed')
                    if (set(result) != set(keys) or result[keys[0]] is not True
                            or type(result[keys[1]]) is not int or not 0 <= result[keys[1]] <= 8):
                        raise ValueError('peer output disposition response is invalid')
                future.set_result(result)
            except Exception as error:
                future.set_exception(error)
            finally:
                if self.generation == getattr(self.service, '_peer_output_generation', 0):
                    self.service.runtime.wakeup()
        spawn_context_thread(execute, name='peer-output-io', daemon=True).start()
        return key, future

    def _call(self, operation, **fields):
        import copy
        row = self.service._obligation(self.scope.room_id, self.scope.task_id, self.scope.execution_generation)
        expected_op = 'ack' if operation == 'read' else operation
        encoded = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
        if (row is None or row['operation'] != expected_op or row['state'] != 'pending'
                or row['scope_json'] != encoded(self.scope.as_mapping())
                or row['manifest_json'] != (encoded(self.manifest) if self.manifest is not None else None)):
            raise ValueError('peer output obligation changed')
        key, future = self._start(operation, fields)
        if future is None or not future.done():
            raise OutputPending('Peer output transfer is pending')
        try:
            return copy.deepcopy(future.result())
        except Exception:
            self.service._peer_output_io.pop(key, None)
            raise

    def read(self, scope, artifact_id):
        import base64
        from gateway.hosted_room_peer_output import output_manifest
        if scope != self.scope:
            raise ValueError('peer output scope changed')
        for item in output_manifest(self.manifest):
            self._start('read', {'artifact_id': item['artifact_id']})
        value = self._call('read', artifact_id=artifact_id)
        return value['metadata'], base64.b64decode(value['data_base64'], validate=True)

    def acknowledge(self, scope, artifact_ids, *, message_event_id):
        if scope != self.scope:
            raise ValueError('peer output scope changed')
        value = self._call('ack', artifact_ids=list(artifact_ids), message_event_id=message_event_id)
        if set(value) != {'acknowledged', 'changed'} or value['acknowledged'] is not True or type(value['changed']) is not int:
            raise ValueError('peer output ACK unconfirmed')
        self._release_reads()
        return value['changed']

    def discard_durably(self, scope):
        if scope != self.scope:
            raise ValueError('peer output scope changed')
        value = self._call('discard')
        if set(value) != {'discarded', 'removed'} or value['discarded'] is not True or type(value['removed']) is not int:
            raise ValueError('peer output discard unconfirmed')
        self._release_reads()
        return value['removed']

    def _release_reads(self):
        table = self.service._peer_output_io
        for key in list(table):
            if key[0] == self.scope.key and json.loads(key[2])['operation'] == 'read' and table[key].done():
                table.pop(key)


def remember_unreceived_discard(conn, binding, task):
    """Called only inside the writer that consumes the exact nonadmission proof."""
    scope = dict(room_id=binding.room_id, authority_gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, member_id=task['payload'].get('target_member_id', task['payload']['target_profile']),
        task_id=task['identity'].task_id, execution_generation=task['execution_generation'])
    key = consent_key(scope)
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if row:
        value = json.loads(row[0])
        value['unreceived_cancel_generation'] = task['cancel_generation'] + 1
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(value, sort_keys=True, separators=(',', ':')), key))
