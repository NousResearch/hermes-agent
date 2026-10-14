"""Conservative resource retirement from read-only canonical authority.

No semantic completion is inferred from an answer, idle time or process death.
"""
import contextlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time

log = logging.getLogger(__name__)


@contextlib.contextmanager
def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=0.2)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA query_only=ON')
        yield db
    finally:
        db.close()


class Authority:
    def __init__(self, home):
        self.home = Path(home).resolve()

    def _sessions(self):
        from hermes_state import SessionDB
        return SessionDB(db_path=self.home / 'state.db', read_only=True)

    def owner(self, task_id, session_id):
        task = str(task_id or '')
        if task.startswith('cron:') and len(task.split(':')) == 3:
            _, job, execution = task.split(':')
            if job and execution:
                return ('cron:' + job, execution)
        try:
            # Exact durable child mapping takes precedence over inherited worker context.
            with readonly(self.home / 'state.db') as db:
                rows = db.execute("SELECT delegation_id, task_json FROM async_delegations "
                                  "WHERE EXISTS (SELECT 1 FROM json_each(task_json, '$.child_task_ids') WHERE value=?) LIMIT 2", (task,)).fetchall()
            if len(rows) == 1:
                return ('child:' + rows[0]['delegation_id'], task)
            if rows:
                return ('unknown:' + task, 'unresolved')
        except Exception:
            # A native child must never fall back to its parent's session/worker owner.
            return ('unknown:' + task, 'unresolved')
        try:
            from agent.delegation_context import is_dispatcher_owned_worker_context, is_delegated_child_context
            if is_delegated_child_context() and task:
                return ('child-task:' + task, task)
            if task.startswith(('sa_', 'sa-', 'subagent-')):
                return ('unknown:' + task, 'unresolved')
            if (not is_delegated_child_context() and is_dispatcher_owned_worker_context()
                    and task and os.environ.get('HERMES_KANBAN_TASK') == task):
                from hermes_cli.kanban_db import kanban_db_path
                run = int(os.environ['HERMES_KANBAN_RUN_ID'])
                return ('kanban:' + json.dumps([str(kanban_db_path().resolve()), task]), str(run))
        except Exception:
            return ('unknown:' + task, 'unresolved')
        if session_id:
            try:
                db = self._sessions()
                try:
                    chain = db.get_compression_lineage(session_id)
                    if chain:
                        return ('session:' + chain[0], chain[0])
                finally:
                    db.close()
            except Exception:
                pass
        return ('unknown:' + (task or 'unscoped'), 'unresolved')

    def evidence(self, owner, generation):
        try:
            if owner.startswith(('child:', 'child-task:')):
                with readonly(self.home / 'browser_tabs.sqlite') as db:
                    if db.execute('SELECT 1 FROM native_completions WHERE owner=? AND generation=?',
                                  (owner, generation)).fetchone():
                        return 'native-child:' + generation
                return None
            if owner.startswith('cron:'):
                with readonly(self.home / 'cron' / 'executions.db') as db:
                    row = db.execute('SELECT * FROM executions WHERE id=? AND job_id=?',
                                     (generation, owner[5:])).fetchone()
                if row and row['status'] in ('completed', 'failed') and not row['handoff_pending'] and not dict(row).get('effect_in_flight', 0):
                    return 'executions:' + generation + ':' + row['status']
            elif owner.startswith('kanban:'):
                path, task = json.loads(owner[7:])
                with readonly(path) as db:
                    row = db.execute('SELECT t.status AS card, r.status, r.outcome, r.ended_at, '
                                     'r.worker_pid, r.worker_started_at FROM tasks t '
                                     'JOIN task_runs r ON r.task_id=t.id WHERE t.id=? AND r.id=?',
                                     (task, int(generation))).fetchone()
                completed = row and row['card'] in ('done', 'archived') and row['status'] == 'done' and row['outcome'] == 'completed'
                cancelled = row and row['card'] == 'archived' and row['status'] == 'reclaimed' and row['outcome'] == 'reclaimed'
                if (completed or cancelled) and row['ended_at'] is not None:
                    if row['worker_pid'] is not None and not process_dead(row['worker_pid'], row['worker_started_at']):
                        return None
                    return 'kanban:' + task + ':' + generation + ':' + row['outcome']
            elif owner.startswith('session:'):
                db = self._sessions()
                try:
                    chain = db.get_compression_lineage(owner[8:])
                    if not chain or chain[0] != generation:
                        return None
                    tip = db.get_session(chain[-1])
                    if tip and tip.get('ended_at') is not None and tip.get('end_reason') in ('new_session', 'session_reset'):
                        return 'sessions:' + tip['id'] + ':' + tip['end_reason']
                finally:
                    db.close()
            # Child delivery/final text is not native termination proof. Retain.
        except Exception:
            return None
        return None

    def actor_finished(self, actor):
        # Exact actor keys use the same authority encoding, never process-parent inference.
        try:
            decoded = json.loads(actor)
            if len(decoded) == 5 and decoded[0] == 'process':
                _, identifier, started_at, pid, fingerprint = decoded
                import re
                if not re.fullmatch(r'proc_[\w]+', identifier):
                    return False
                receipt = json.loads((self.home / 'logs' / 'process-results' / (identifier + '.json')).read_text())
                return (receipt.get('id') == identifier and receipt.get('started_at') == started_at
                        and process_dead(pid, fingerprint))
            owner, generation = json.loads(actor)
            with readonly(self.home / 'browser_tabs.sqlite') as db:
                if db.execute("SELECT 1 FROM calls WHERE owner=? AND generation=? AND state IN ('active','quarantined') LIMIT 1",
                              (owner, generation)).fetchone():
                    return False
            return self.evidence(owner, generation) is not None and not owner.startswith('session:')
        except Exception:
            return False


def process_identity(pid):
    from gateway.drain_control import current_instantiation_epoch
    from gateway.status import get_process_start_time
    start = get_process_start_time(int(pid))
    if start is None:
        raise OSError('process identity unavailable')
    return f'{current_instantiation_epoch()}|{start}'


def register_process_hold(home, browser, target, process):
    """Trusted callers register an exact target BEFORE handing it to a host process."""
    from tools.browser_tab_ownership import OwnershipBusy, OwnershipRegistry
    if process.exited or process.pid_scope != 'host' or not process.pid:
        raise OwnershipBusy('a live host process identity is required')
    actor = json.dumps(['process', process.id, process.started_at, process.pid, process_identity(process.pid)])
    OwnershipRegistry(Path(home) / 'browser_tabs.sqlite').hold(browser, target, actor)
    return actor


def process_dead(pid, start):
    if not pid or not start or start == 'unverified':
        return False
    try:
        stat = (Path('/proc') / str(int(pid)) / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return True
        current = process_identity(pid) if '|' in str(start) else stat[19]
        return current != str(start)
    except FileNotFoundError:
        return Path('/proc/self/stat').exists()
    except (OSError, ValueError, IndexError):
        return False


class PinnedBrowser:
    def __init__(self, ws):
        from urllib.parse import urlsplit
        parsed = urlsplit(ws)
        if parsed.scheme not in ('ws', 'wss') or not parsed.path.startswith('/devtools/browser/'):
            raise ValueError('immutable browser websocket required')
        from websockets.sync.client import connect
        self.socket = connect(ws, open_timeout=1, close_timeout=1, proxy=None)
        self.sequence = 0

    def command(self, method, **params):
        self.sequence += 1
        self.socket.send(json.dumps(dict(id=self.sequence, method=method, params=params)))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            reply = json.loads(self.socket.recv(timeout=max(.01, deadline - time.monotonic())))
            if reply.get('id') == self.sequence:
                if 'error' in reply:
                    raise RuntimeError('CDP rejected cleanup')
                return reply['result']
        raise TimeoutError('CDP cleanup response')

    def targets(self):
        return {t['targetId'] for t in self.command('Target.getTargets')['targetInfos']}

    def close_target(self, target):
        self.command('Target.closeTarget', targetId=target)


def enabled(home):
    try:
        from hermes_cli.config_effective import load_user_config_effective
        config = load_user_config_effective(Path(home) / 'config.yaml', fail_closed=True)
        return config.get('browser', {}).get('tab_cleanup_enabled') is True
    except Exception:
        return False


def native_child_finished(task_id, home=None):
    """Called only after the complete native child runner has stopped, not after a turn."""
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    path = Path(home) / 'browser_tabs.sqlite'
    if not path.exists() or not task_id:
        return
    from tools.browser_tab_ownership import OwnershipRegistry
    registry = OwnershipRegistry(path)
    with registry._db() as db:
        db.execute('INSERT OR IGNORE INTO native_completions VALUES (?,?)',
                   ('child-task:' + str(task_id), str(task_id)))
        db.execute("INSERT OR IGNORE INTO native_completions SELECT owner,generation FROM owners "
                   "WHERE generation=? AND (owner LIKE 'child:%' OR owner LIKE 'child-task:%')", (str(task_id),))
        db.execute("INSERT OR IGNORE INTO native_completions SELECT json_extract(actor,'$[0]'),json_extract(actor,'$[1]') FROM holds "
                   "WHERE json_valid(actor) AND json_extract(actor,'$[1]')=? AND json_extract(actor,'$[0]') LIKE 'child:%'", (str(task_id),))


def close_owned(home, task_id=None, session_id=None, connect=PinnedBrowser, limit=32):
    """Explicit owner-scoped close; never ends a conversation or starts a Harness."""
    from tools.browser_tab_ownership import OwnershipBusy, OwnershipRegistry
    if not enabled(home):
        return {'success': False, 'reason': 'ownership_disabled', 'closed': 0, 'remaining': 0}
    owner = Authority(home).owner(task_id, session_id)
    if owner[0].startswith('unknown:') or owner[1] == 'unresolved':
        return {'success': False, 'reason': 'owner_unresolved', 'closed': 0, 'remaining': 0}
    path = Path(home) / 'browser_tabs.sqlite'
    if not path.exists():
        return {'success': True, 'closed': 0, 'remaining': 0}
    registry = OwnershipRegistry(path)
    try:
        rows = registry.request_owner_close(*owner)
    except OwnershipBusy:
        return {'success': False, 'reason': 'owner_busy', 'closed': 0, 'remaining': None}
    for ws in dict.fromkeys(row['browser'] for row in rows):
        browser = None
        try:
            browser = connect(ws)
            def close(target):
                if not enabled(home):
                    raise OwnershipBusy('cleanup disabled')
                browser.close_target(target)
                deadline = time.monotonic() + 0.5
                while target in browser.targets() and time.monotonic() < deadline:
                    time.sleep(0.02)
            registry.cleanup(ws, browser.targets, close, limit=limit, owner_scope=owner)
        except Exception as exc:
            log.debug('Explicit browser close held: %s', type(exc).__name__)
        finally:
            if browser:
                browser.socket.close()
    with registry._db() as db:
        remaining = db.execute('SELECT COUNT(*) FROM targets WHERE owner=? AND generation=? AND pending_close=1', owner).fetchone()[0]
    return {'success': remaining == 0, 'closed': len(rows) - remaining,
            'remaining': remaining, 'reason': 'pending' if remaining else 'closed'}


def _batch(db, key, query, limit):
    """Bounded round-robin; retained unknown rows cannot starve later work."""
    db.execute('CREATE TABLE IF NOT EXISTS sweep_cursor (id INTEGER PRIMARY KEY, position INTEGER)')
    cursor = db.execute('SELECT position FROM sweep_cursor WHERE id=?', (key,)).fetchone()
    position = cursor[0] if cursor else 0
    sql = 'SELECT * FROM (' + query + ') WHERE position>? ORDER BY position LIMIT ?'
    rows = list(db.execute(sql, (position, limit)))
    if not rows:
        rows = list(db.execute(sql, (0, limit)))
    db.execute('INSERT OR REPLACE INTO sweep_cursor VALUES (?,?)', (key, rows[-1]['position'] if rows else 0))
    return rows


def sweep(home, connect=PinnedBrowser, limit=8):
    if not enabled(home):
        return
    from tools.browser_tab_ownership import OwnershipRegistry
    path = Path(home) / 'browser_tabs.sqlite'
    if not path.exists():
        return
    registry = OwnershipRegistry(path)
    authority = Authority(home)
    with registry._db() as db:
        calls = _batch(db, 2, "SELECT rowid AS position,* FROM calls WHERE state IN ('active','quarantined')", limit)
        owners = _batch(db, 1, 'SELECT rowid AS position,owner,generation FROM owners WHERE terminal=0', limit)
        holds = _batch(db, 3, 'SELECT rowid AS position,* FROM holds', limit)
    for call in calls:
        if process_dead(call['daemon_pid'], call['daemon_start']) and process_dead(call['client_pid'], call['client_start']):
            with registry._db() as db:
                db.execute("UPDATE calls SET state='drained' WHERE token=? AND state IN ('active','quarantined')", (call['token'],))
    for hold in holds:
        if authority.actor_finished(hold['actor']):
            registry.release(hold['browser'], hold['target'], hold['actor'])
    for _, owner, generation in owners:
        evidence = authority.evidence(owner, generation)
        if evidence:
            registry.retire(owner, generation, evidence=evidence)
    with registry._db() as db:
        browsers = [r['browser'] for r in _batch(db, 4, 'SELECT MIN(t.rowid) AS position,t.browser FROM targets t JOIN owners o USING(owner,generation) WHERE o.terminal=1 OR t.pending_close=1 GROUP BY t.browser', 2)]
    for ws in browsers:
        browser = None
        try:
            browser = connect(ws)
            def close_if_enabled(target):
                if not enabled(home):
                    raise RuntimeError('cleanup disabled')
                browser.close_target(target)
            registry.cleanup(ws, browser.targets, close_if_enabled, limit=limit)
        except Exception as exc:
            log.debug('Browser tab sweep held: %s', type(exc).__name__)
        finally:
            if browser:
                browser.socket.close()


_handles = {}
_lock = threading.Lock()


def ensure_scheduled(home):
    from agent.periodic_scheduler import schedule
    key = str(Path(home).resolve())
    with _lock:
        if key not in _handles:
            _handles[key] = schedule(lambda: sweep(key), 60)


def startup(home):
    if enabled(home) and (Path(home) / 'browser_tabs.sqlite').exists():
        ensure_scheduled(home)


def after_call(home):
    try:
        sweep(home)
    except Exception as exc:
        log.debug('Browser tab lifecycle held: %s', type(exc).__name__)
