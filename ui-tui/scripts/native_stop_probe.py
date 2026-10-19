"""Loopback Stop exercise and observation-only recording of real RPC replies."""
import asyncio
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import time


class StopPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        try:
            self.respond()
        except (BrokenPipeError, ConnectionResetError):
            # Interrupt closes the real inference stream before its final chunk.
            self.server.disconnected.set()

    def respond(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        self.server.requests.append(messages)
        latest = next((m.get('content') for m in reversed(messages) if m['role'] == 'user'), '')
        asking = bool(messages) and latest != 'AFTER_STOP' and not any(m['role'] == 'tool' for m in messages)
        message = {'role': 'assistant', 'content': 'FRESH_AFTER_STOP' if latest == 'AFTER_STOP' else 'STOP_CONTROL_FINISHED'}
        if asking:
            message = {'role': 'assistant', 'tool_calls': [{
                'index': 0, 'id': 'owned-stop-control', 'type': 'function', 'function': {
                    'name': 'terminal', 'arguments': json.dumps({'command': self.server.command})}}]}
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
        self.end_headers()
        if body.get('stream'):
            def emit(delta, finish=None):
                frame = {'id': 'local', 'object': 'chat.completion.chunk', 'created': 1,
                         'model': 'local', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
                self.wfile.flush()
            if asking:
                emit({'role': 'assistant', 'content': 'STOP_RUNNING'})
                self.server.blocked.set()
                assert self.server.release.wait(30), 'native Stop never released the model boundary'
            emit(message)
            emit({}, 'tool_calls' if asking else 'stop')
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps({'id': 'local', 'choices': [{'index': 0, 'message': message,
                'finish_reason': 'tool_calls' if asking else 'stop'}]}).encode())
        self.wfile.flush()


async def created_session_id(grant, rpc, connect):
    g = grant()
    async with connect(g['url'], subprotocols=g['protocols']) as ws:
        async with asyncio.timeout(15):
            while True:
                inventory = await rpc(ws, 'session.list')
                if inventory['sessions']:
                    assert len(inventory['sessions']) == 1, inventory
                    sid = inventory['sessions'][0]['session_id']
                    info = await rpc(ws, 'session.info', session_id=sid)
                    assert info['source'] == 'tui', info
                    return sid
                await asyncio.sleep(.05)


def exercise(kind, model, first, sid, grant, rpc, connect, home, receipts, destination):
    def wait(predicate, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            assert first[0].poll() is None, bytes(first[2]).decode(errors='replace')
            time.sleep(.05)
        raise AssertionError('native Stop condition timeout')

    wait(lambda: model.blocked.is_set() and b'STOP_RUNNING' in first[2])
    control = home / 'native-control-receipts.jsonl'
    if kind in ('stop', 'stop-launcher'):
        os.write(first[1], b'\x03')
        wait(lambda: control.exists())
        wire = [json.loads(line) for line in control.read_text().splitlines()]
        (destination / 'native-interrupt.json').write_text(json.dumps(wire, indent=2))
        receipts['native_interrupt_acknowledged'] = any(
            row['request']['params']['session_id'] == sid and 'result' in row['reply']
            and row['reply']['result']['execution_generation'] == row['request']['params']['execution_generation']
            for row in wire)
        assert receipts['native_interrupt_acknowledged']
    model.release.set()

    async def snapshot():
        g = grant()
        async with connect(g['url'], subprotocols=g['protocols']) as ws:
            async with asyncio.timeout(15):
                while True:
                    result = await rpc(ws, 'session.resume', session_id=sid)
                    if not result['running']:
                        return result
                    await asyncio.sleep(.05)
    settled = asyncio.run(snapshot())
    (destination / 'stopped-snapshot.json').write_text(json.dumps(settled, indent=2))
    effect = home / 'stop-effect'
    receipts['post_stop_tool_suppressed'] = not effect.exists()
    assert effect.exists() == (kind == 'stop-control')
    os.write(first[1], b'AFTER_STOP')
    wait(lambda: b'AFTER_STOP' in first[2])
    os.write(first[1], b'\r')
    wait(lambda: b'FRESH_AFTER_STOP' in first[2])
    receipts['fresh_after_stop_rendered'] = True
    final = asyncio.run(snapshot())
    (destination / 'fresh-snapshot.json').write_text(json.dumps(final, indent=2))
    (destination / 'stop.pty').write_bytes(first[2])
    receipts['fresh_after_stop_persisted'] = any(
        row.get('content') == 'FRESH_AFTER_STOP' for row in final['messages'])
    receipts['prepared_journal_cleared'] = not list((home / 'tui-pending-inputs').glob('*.json'))
    assert receipts['fresh_after_stop_persisted'] and receipts['prepared_journal_cleared']


def daemon():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from gateway.session_controls import AuthorityConnection
    original = AuthorityConnection.dispatch
    async def observed(self, request):
        reply = await original(self, request)
        if request.get('method') == 'session.interrupt':
            with (Path(os.environ['HERMES_HOME']) / 'native-control-receipts.jsonl').open('a') as output:
                output.write(json.dumps({'request': request, 'reply': reply}) + '\n')
        return reply
    AuthorityConnection.dispatch = observed
    sys.argv = ['gateway.run']
    runpy.run_module('gateway.run', run_name='__main__')


if __name__ == '__main__':
    daemon()
