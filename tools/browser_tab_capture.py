"""Opt-in Harness capture. Unknown owners are retained; no inferred retirement.

Only synchronous Harness requests are covered, not arbitrary Python CDP clients.
"""
from contextvars import ContextVar
import os
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import urlopen

from tools.browser_tab_ownership import OwnershipBusy, OwnershipRegistry


def pinned_websocket(env):
    ws = env.get('BU_CDP_WS')
    if not ws:
        endpoint = env.get('BU_CDP_URL', '')
        if urlsplit(endpoint).scheme not in ('http', 'https'):
            raise OwnershipBusy('ownership requires an explicit CDP endpoint')
        with urlopen(endpoint.rstrip('/') + '/json/version', timeout=5) as response:
            ws = json.load(response).get('webSocketDebuggerUrl', '')
    parsed = urlsplit(ws)
    if parsed.scheme not in ('ws', 'wss') or not parsed.path.startswith('/devtools/browser/') or not parsed.path.removeprefix('/devtools/browser/'):
        raise OwnershipBusy('CDP endpoint has no immutable browser identity')
    return ws


def _explicit_error(reply):
    return isinstance(reply, dict) and isinstance(reply.get('error'), str) and bool(reply['error'])


def _confirmed_reply(req, reply):
    if not isinstance(reply, dict):
        return False
    if 'error' in reply:
        return _explicit_error(reply)
    if req.get('method'):
        return isinstance(reply.get('result'), dict)
    meta = req.get('meta')
    if meta == 'ping':
        pid = reply.get('pid')
        return reply.get('pong') is True and type(pid) is int and 0 < pid < (1 << 31)
    if meta == 'set_session':
        return bool(req.get('session_id')) and reply.get('session_id') == req['session_id']
    if meta == 'session':
        return 'session_id' in reply
    if meta == 'drain_events':
        return isinstance(reply.get('events'), list)
    if meta == 'pending_dialog':
        return 'dialog' in reply
    if meta == 'current_tab':
        return isinstance(reply.get('targetId'), str) and bool(reply['targetId'])
    if meta == 'connection_status':
        return all(key in reply for key in ('target_id', 'session_id', 'page'))
    if meta == 'shutdown':
        return reply.get('ok') is True
    return False


def install_capture(helpers, registry, token):
    original = helpers._send
    response: ContextVar[list | None] = ContextVar('harness_capture_response', default=None)
    ipc = getattr(helpers, 'ipc', None)
    if ipc is not None:
        original_request = ipc.request

        def observed_request(*args, **kwargs):
            reply = original_request(*args, **kwargs)
            evidence = response.get()
            if evidence is not None:
                evidence.append(reply)
            return reply

        # _send raises for daemon error envelopes: inspect wire evidence,
        # never an exception's class/message, to distinguish it from uncertainty.
        ipc.request = observed_request

    def captured(req, *args, **kwargs):
        registry.request_started(token)
        evidence = []
        context = response.set(evidence)
        try:
            try:
                reply = original(req, *args, **kwargs)
            except BaseException:
                if evidence and _explicit_error(evidence[-1]):
                    registry.request_finished(token)
                else:
                    registry.quarantine(token)
                raise
            if not _confirmed_reply(req, reply):
                registry.quarantine(token)
                raise ValueError('Harness request has no confirmed reply')
            if req.get('method') == 'Target.createTarget' and not _explicit_error(reply):
                try:
                    registry.record_created(token, reply['result'].get('targetId'))
                except BaseException:
                    registry.quarantine(token)
                    raise
            registry.request_finished(token)
            return reply
        finally:
            response.reset(context)

    helpers._send = captured


def _verify_daemon(helpers, registry, token):
    """Linux proof of the actual named daemon's pinned launch endpoint; fail closed."""
    call = registry.call(token)
    pid = int(helpers._send({'meta': 'ping'})['pid'])
    proc = Path('/proc') / str(pid)
    before = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    environ = dict(item.split(b'=', 1) for item in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in item)
    after = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    if before != after or environ.get(b'BU_CDP_WS') != call['browser'].encode() or environ.get(b'BU_NAME') != call['daemon'].encode():
        raise OwnershipBusy('daemon browser identity does not match admission')
    registry.bind_daemon(token, pid, before)


def _daemon_created_current(helpers, registry, token, current, live):
    """Adopt only an exact dedicated-creation receipt from the verified daemon.

    connection_status alone is attachment, NOT creation proof. Named Harness
    daemons create a startup target outside helpers._send; their per-daemon
    log records the returned ID and is truncated at daemon startup. Recheck
    the immutable daemon identity after reading that receipt. Unknown/older
    Harness implementations remain on the explicit create-and-capture path.
    """
    target = current.get('target_id')
    if target not in live or not current.get('session_id'):
        return None
    call = registry.call(token)
    if not call.get('daemon_pid') or not call.get('daemon_start'):
        return None
    with registry._db() as db:
        pending = db.execute('SELECT 1 FROM targets t JOIN calls c ON c.token=t.call_token '
                             'WHERE t.browser=? AND t.target=? AND t.owner=? AND t.generation=? '
                             'AND c.daemon=? AND t.pending_close=1',
                             (call['browser'], target, call['owner'], call['generation'], call['daemon'])).fetchone()
    if pending:
        return None  # Already captured and released; never re-adopt it under this call.
    receipt = f"named daemon {call['daemon']}: created dedicated tab ({target})"
    try:
        from browser_harness import _ipc
        path = _ipc.log_path(call['daemon'])
        # Bound the compatibility read; fail closed rather than scanning an
        # unbounded log or accepting a substring from another log message.
        with Path(path).open('rb') as stream:
            data = stream.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024 or receipt.encode() not in data.splitlines():
            return None
        _verify_daemon(helpers, registry, token)
        # Current target info is browser-pinned; only adopt an untouched startup
        # blank. Never retroactively claim an existing authenticated/foreign page.
        info = helpers.cdp('Target.getTargetInfo', targetId=target)['targetInfo']
        if info.get('type') != 'page' or info.get('url') != 'about:blank':
            return None
        registry.record_created(token, target)
        return target
    except (OSError, KeyError, ImportError):
        return None


def _current_status(helpers):
    try:
        return helpers._send({'meta': 'connection_status'})
    except RuntimeError as exc:
        if str(exc) != 'cdp_disconnected':
            raise
        return {}  # definitive no-current-page reply; never attachment proof


def begin_exec(path, token):
    from browser_harness import helpers
    registry = OwnershipRegistry(path)
    import os
    client_start = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    with registry._db() as db:
        db.execute('UPDATE calls SET client_pid=?, client_start=? WHERE token=?',
                   (os.getpid(), client_start, token))
    install_capture(helpers, registry, token)
    try:
        _verify_daemon(helpers, registry, token)
        live = {t['targetId'] for t in helpers.cdp('Target.getTargets')['targetInfos'] if t.get('type') == 'page'}
        owned = [target for target in registry.owned_for_daemon(token) if target in live]
        current = _current_status(helpers)
        if current.get('target_id') in owned and current.get('session_id'):
            # Retain the verified mission tab and its existing flattened session.
            # Reattaching every exec loses a user's selected second owned tab
            # and accumulates enabled CDP sessions on the same renderer.
            return
        # Capture a proved daemon startup/recovery creation even when an older
        # mission target survives a daemon restart. Otherwise reattaching the
        # continuation would leave the newly created blank unregistered again.
        startup_target = _daemon_created_current(helpers, registry, token, current, live)
        if startup_target and not owned:
            return
        target = owned[0] if owned else helpers.cdp('Target.createTarget', url='about:blank', background=True)['targetId']
        # Avoid switch_tab's Runtime.evaluate against the previously attached unknown tab.
        sid = helpers.cdp('Target.attachToTarget', targetId=target, flatten=True)['sessionId']
        helpers._send({'meta': 'set_session', 'session_id': sid, 'target_id': target})
    except BaseException:
        registry.quarantine(token)
        raise


def prepare_exec(env, code, home, owner):
    ws = pinned_websocket(env)
    home = Path(home).resolve()
    identity = json.dumps([str(home), *owner, env.get('BU_NAME', 'default'), ws])
    daemon = 'ht-' + hashlib.sha256(identity.encode()).hexdigest()[:32]
    registry = OwnershipRegistry(home / 'browser_tabs.sqlite')
    token = registry.admit(*owner, ws, daemon)
    env['BU_NAME'] = daemon
    env['BU_CDP_WS'] = ws
    env.pop('BU_CDP_URL', None)
    capture_root = str(Path(__file__).resolve().parent.parent)
    env['PYTHONPATH'] = os.pathsep.join(path for path in (env.get('PYTHONPATH'), capture_root) if path)
    preamble = ('from tools.browser_tab_capture import begin_exec as _hermes_begin_exec\n'
                f'_hermes_begin_exec({str(registry.path)!r}, {token!r})\n'
                'del _hermes_begin_exec\n')
    # Read the actual daemon current target AFTER user code, including on a
    # Python exception. Never infer it from creation order or a site's origin.
    wrapped = (preamble + 'try:\n'
               + f'    exec(compile({code!r}, "<browser_exec>", "exec"), globals())\n'
               + 'finally:\n'
               + '    from tools.browser_tab_capture import end_exec as _hermes_end_exec\n'
               + f'    _hermes_end_exec({str(registry.path)!r}, {token!r})\n'
               + '    del _hermes_end_exec\n')
    return SimpleNamespace(registry=registry, token=token), wrapped


def end_exec(path, token):
    from browser_harness import helpers
    registry = OwnershipRegistry(path)
    try:
        _verify_daemon(helpers, registry, token)
        status = _current_status(helpers)
        if status.get('session_id') and status.get('target_id') in registry.owned_for_daemon(token):
            registry.publish_current(token, status['target_id'])
        # A known foreign/closed current page is not uncertain CDP work. Leave
        # this call's handoff empty (vault refuses), without fencing navigation.
    except BaseException:
        registry.quarantine(token)
        raise


def unresolved_owner(task_id):
    # Deliberately UNKNOWN: caller task IDs do not prove semantic task lifetime.
    # Stable routing for callers without a verified native lifecycle mapping.
    return ('unknown:' + (str(task_id) if task_id else 'unscoped'), 'unresolved')
