"""Disposable ordinary-daemon + real Ink PTY probe (no renderer injection)."""
import asyncio
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import signal
import shlex
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.gateway.fixtures.shared_authority_peer import ModelPeer
from websockets.asyncio.client import connect


def run():
    receipts = {}
    kind = sys.argv[2] if len(sys.argv) > 2 else 'text'
    startup = kind in ('startup', 'launcher')
    expected = b'LOCAL_ACK' if kind == 'text' else (kind.upper() + '_FINISHED').encode()
    with tempfile.TemporaryDirectory(prefix='ink-native-authority-') as temp:
        base = Path(temp)
        home = base / 'state'
        home.mkdir(mode=0o700)
        user = base / 'user'
        user.mkdir()
        peer_class = ModelPeer
        stopping = kind in ('stop', 'stop-control', 'stop-launcher')
        if stopping:
            from native_stop_probe import StopPeer
            peer_class = StopPeer
        if kind in ('approval', 'clarify'):
            sys.path.insert(0, str(ROOT / 'tests/gateway/fixtures'))
            if kind == 'approval':
                from authority_controls_peer import ModelPeer as peer_class
            else:
                from authority_clarify_peer import ModelPeer as peer_class
        target = base / 'owned-removal'
        target.mkdir()
        (target / 'owned.txt').write_text('fixture')
        model = ThreadingHTTPServer(('127.0.0.1', 0), peer_class)
        model.command = 'rm -r -- ' + shlex.quote(str(target))
        model.requests, model.metadata_requests = [], []
        model.blocked, model.release, model.disconnected = threading.Event(), threading.Event(), threading.Event()
        if stopping:
            model.command = 'touch -- ' + shlex.quote(str(home / 'stop-effect'))
        threading.Thread(target=model.serve_forever, daemon=True).start()
        model_url = f'http://127.0.0.1:{model.server_port}/v1'
        (home / 'config.yaml').write_text(json.dumps({
            'gateway': {'multiplex_profiles': False},
            'approvals': {'mode': 'manual', 'timeout': 60},
            'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': model_url},
            'auxiliary': {'title_generation': {'enabled': False}},
        }))
        env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
        env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
                   PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', HERMES_PYTHON=sys.executable,
                   HERMES_PYTHON_SRC_ROOT=str(ROOT), TERM='xterm-256color',
                   OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=model_url)
        children = []
        daemon = None
        ensured_pid = None

        def grant():
            result = subprocess.run([sys.executable, str(ROOT / 'ui-tui/scripts/gateway_bootstrap.py')],
                                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)

        async def rpc(ws, method, **params):
            await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
            async with asyncio.timeout(15):
                while True:
                    reply = json.loads(await ws.recv())
                    if reply.get('id') == method:
                        assert 'result' in reply, reply
                        return reply['result']

        async def seed():
            g = grant()
            async with connect(g['url'], subprotocols=g['protocols']) as ws:
                description = await rpc(ws, 'runtime.describe')
                source = 'tui' if 'tui' in description['session_create']['sources'] else 'cli'
                result = await rpc(ws, 'session.create', request_id='native-seed', source=source)
                receipts['seed_source'] = source
                receipts['session_id'] = result['session_id']
                return result['session_id']

        def launch(name, sid=None, query=None):
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 36, 120, 0, 0))
            launch_env = dict(env)
            launch_env['HERMES_TUI_ACTIVE_SESSION_FILE'] = str(base / (name + '.json'))
            if sid:
                launch_env['HERMES_TUI_RESUME'] = sid
            if query:
                launch_env['HERMES_TUI_QUERY'] = query
            argv = ['node', str(ROOT / 'ui-tui/dist/entry.js')]
            if kind in ('launcher', 'stop-launcher'):
                # Exercise the actual parser and Python launcher, using the
                # supported prebuilt path so shared dependencies stay read-only.
                launch_env.update(HERMES_TUI_DIR=str(ROOT / 'ui-tui'), TMPDIR=str(base))
                argv = [sys.executable, '-m', 'hermes_cli.main', '--tui', 'chat', '-q', 'WS_SHARED']
            proc = subprocess.Popen(argv, cwd=ROOT,
                                    env=launch_env, stdin=slave, stdout=slave, stderr=slave,
                                    start_new_session=True)
            os.close(slave)
            children.append((proc, master))
            output = bytearray()
            def drain():
                while proc.poll() is None:
                    if select.select([master], [], [], .1)[0]:
                        try:
                            output.extend(os.read(master, 65536))
                        except OSError:
                            break
            threading.Thread(target=drain, daemon=True).start()
            return proc, master, output

        try:
            if startup:
                first = launch('normal-local', query='WS_SHARED')
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    if b'LOCAL_ACK' in first[2]:
                        break
                    assert first[0].poll() is None, bytes(first[2]).decode(errors='replace')
                    time.sleep(.1)
                (Path(sys.argv[1]) / 'startup.pty').write_bytes(first[2])
                from gateway.status import get_running_pid_identity_strict
                identity = get_running_pid_identity_strict(home / 'gateway.pid')
                assert identity is not None, bytes(first[2]).decode(errors='replace')[-8000:]
                ensured_pid = identity[0]
                async def created_session():
                    g = grant()
                    async with connect(g['url'], subprotocols=g['protocols']) as ws:
                        inventory = await rpc(ws, 'session.list')
                        for row in inventory['sessions']:
                            row['policy'] = await rpc(ws, 'session.info', session_id=row['session_id'])
                        return inventory
                inventory = asyncio.run(created_session())
                (Path(sys.argv[1]) / 'created-sessions.json').write_text(json.dumps(inventory, indent=2))
                receipts['canonical_owner_started_by_ink'] = True
                receipts['python_launcher'] = kind == 'launcher'
                receipts['fresh_tui_created'] = len(inventory['sessions']) == 1 and inventory['sessions'][0]['policy']['source'] == 'tui'
                receipts['fresh_reply_rendered'] = b'LOCAL_ACK' in first[2]
                receipts['prepared_journal_cleared'] = not list((home / 'tui-pending-inputs').glob('*.json'))
                assert receipts['fresh_tui_created'] and receipts['fresh_reply_rendered']
                assert receipts['prepared_journal_cleared']
                return
            log = (base / 'daemon.log').open('w+')
            daemon_argv = [sys.executable, '-m', 'gateway.run']
            if stopping:
                daemon_argv = [sys.executable, str(ROOT / 'ui-tui/scripts/native_stop_probe.py')]
            daemon = subprocess.Popen(daemon_argv, cwd=ROOT, env=env,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                try:
                    g = grant()
                    break
                except AssertionError:
                    assert daemon.poll() is None, 'daemon exited'
                    time.sleep(.1)
            else:
                raise AssertionError('daemon bootstrap readiness timeout')
            receipts['instance_id'] = g['instance_id']
            sid = None if kind == 'stop-launcher' else asyncio.run(seed())
            first = launch('first', sid, 'WS_SHARED')
            if kind == 'stop-launcher':
                from native_stop_probe import created_session_id
                sid = asyncio.run(created_session_id(grant, rpc, connect))
                receipts.update(session_id=sid, python_launcher=True)
            if stopping:
                from native_stop_probe import exercise
                exercise(kind, model, first, sid, grant, rpc, connect, home, receipts, Path(sys.argv[1]))
                receipts['daemon_survived_stop'] = daemon.poll() is None
                return
            second = launch('second', sid)
            if kind != 'text':
                card = b'Allow once' if kind == 'approval' else b'green'
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline and (card not in first[2] or card not in second[2]):
                    time.sleep(.1)
                receipts['two_native_prompt_cards'] = card in first[2] and card in second[2]
                for name, child in [('first', first), ('second', second)]:
                    (Path(sys.argv[1]) / (name + '.pty')).write_bytes(child[2])
                assert receipts['two_native_prompt_cards'], bytes(first[2]).decode(errors='replace')[-8000:]
                assert target.exists()
                first[0].terminate()
                second[0].terminate()
                first[0].wait(timeout=5)
                second[0].wait(timeout=5)
                first = launch('answering', sid)
                second = launch('observer', sid)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and card not in first[2]:
                    time.sleep(.1)
                assert card in first[2], bytes(first[2]).decode(errors='replace')[-8000:]
                receipts['restored_native_prompt_card'] = True
                if kind == 'approval':
                    os.write(first[1], b'1')
                else:
                    os.write(first[1], b'2')
            deadline = time.monotonic() + 35
            while time.monotonic() < deadline and (not model.requests or expected not in first[2]):
                assert first[0].poll() is None, bytes(first[2]).decode(errors='replace')
                time.sleep(.1)
            receipts['prepared_journal_cleared'] = not list((home / 'tui-pending-inputs').glob('*.json'))
            assert receipts['prepared_journal_cleared'], 'canonical ACK failed to retire prepared input'
            receipts['model_requests'] = len(model.requests)
            receipts['first_rendered_reply'] = expected in first[2]
            receipts['second_rendered_reply'] = expected in second[2]
            for name, child in [('first', first), ('second', second)]:
                (Path(sys.argv[1]) / (name + '.pty')).write_bytes(child[2])
            assert receipts['first_rendered_reply'], bytes(first[2]).decode(errors='replace')[-12000:]
            assert receipts['second_rendered_reply'], bytes(second[2]).decode(errors='replace')[-12000:]
            first[0].terminate()
            second[0].terminate()
            first[0].wait(timeout=5)
            second[0].wait(timeout=5)
            # A reconnect must recover committed history, not win a race with
            # the last live delta. Wait on the real authority's settled snapshot.
            async def settled_snapshot():
                g = grant()
                async with connect(g['url'], subprotocols=g['protocols']) as ws:
                    async with asyncio.timeout(15):
                        while True:
                            snapshot = await rpc(ws, 'session.resume', session_id=sid)
                            if not snapshot['running']:
                                return snapshot
                            await asyncio.sleep(.05)
            snapshot = asyncio.run(settled_snapshot())
            (Path(sys.argv[1]) / 'settled-snapshot.json').write_text(json.dumps(snapshot, indent=2))
            receipts['settled_history_has_reply'] = expected.decode() in json.dumps(snapshot['messages'])
            assert receipts['settled_history_has_reply']
            third = launch('reconnected', sid)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and expected not in third[2]:
                time.sleep(.1)
            (Path(sys.argv[1]) / 'reconnected.pty').write_bytes(third[2])
            receipts['reconnect_rendered_reply'] = expected in third[2]
            receipts['daemon_survived_detach'] = daemon.poll() is None
            if kind == 'approval':
                receipts['owned_effect_after_native_consent'] = not target.exists()
                assert not target.exists()
            if kind == 'clarify':
                tool_replies = [json.loads(message['content']) for messages in model.requests
                                for message in messages if message['role'] == 'tool']
                receipts['native_answer_on_model_wire'] = any(
                    response.get('user_response') == 'green'
                    for reply in tool_replies for response in reply.get('responses', []))
                (Path(sys.argv[1]) / 'model-tool-replies.json').write_text(json.dumps(tool_replies, indent=2))
                assert receipts['native_answer_on_model_wire']
            assert receipts['reconnect_rendered_reply']
        finally:
            model.release.set()
            if 'first' in locals():
                (Path(sys.argv[1]) / 'last-first.pty').write_bytes(first[2])
            for proc, master in children:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
                os.close(master)
            if daemon and daemon.poll() is None:
                daemon.send_signal(signal.SIGINT)
                try:
                    daemon.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    daemon.kill()
                    daemon.wait(timeout=5)
            if startup and not ensured_pid:
                from gateway.status import get_running_pid_identity_strict
                identity = get_running_pid_identity_strict(home / 'gateway.pid')
                ensured_pid = identity[0] if identity else None
            if ensured_pid:
                import psutil
                owned = psutil.Process(ensured_pid)
                owned.send_signal(signal.SIGINT)
                try:
                    owned.wait(timeout=20)
                except psutil.TimeoutExpired:
                    owned.kill()
            model.shutdown()
            model.server_close()
            (Path(sys.argv[1]) / 'receipt.json').write_text(json.dumps(receipts, indent=2))
            print(json.dumps(receipts), flush=True)


if __name__ == '__main__':
    Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)
    run()
