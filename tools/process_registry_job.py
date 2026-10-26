"""Linux per-job subreaper: kernel-owned children, existing native exit evidence.

The executable is stdlib-only. It keeps the job's recorded PID alive until
waitpid proves that no child remains, including orphans that called setsid.
It never treats a snapshot or the primary command's exit as that proof.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def job_receipt_path(session_id):
    from tools.process_registry import _checkpoint_path
    return _checkpoint_path().parent / 'logs' / 'process-output' / (session_id + '.scope.exit')


def read_job_receipt(session):
    get = session.get if isinstance(session, dict) else lambda name, default=None: getattr(session, name, default)
    native = get('native_job_scope') is True
    if not native:
        return None
    sid = get('session_id') or get('id')
    pid, birth = get('pid'), get('host_start_time')
    if (not isinstance(sid, str) or not sid.startswith('proc_')
            or not sid[5:] or any(not (value.isalnum() or value == '_') for value in sid[5:])):
        return None
    try:
        row = json.loads(job_receipt_path(sid).read_text(encoding='utf-8-sig'))
        identity = (all(type(value) is int and value > 0 for value in (pid, birth))
                    and type(row.get('pid')) is int and type(row.get('birth')) is int
                    and row['pid'] == pid and row['birth'] == birth)
        # A successful native unit Stop proves an empty scope, never permission
        # to signal a PID. Only a live helper's positive identity grants that.
        unit = (isinstance(row.get('unit'), str) and row['unit'].startswith('hermes-worker-')
                and row.get('quiescent') is True and get('systemd_unit', '') in ('', row['unit']))
        if (type(row.get('version')) is int and row['version'] == 1
                and (identity or unit)
                and type(row.get('quiescent')) is bool
                and (row.get('exit_code') is None or type(row['exit_code']) is int)):
            return row
    except (OSError, ValueError, AttributeError):
        pass
    return None


def checkpoint_retains_receipt(path):
    """A scope verdict remains needed until its existing checkpoint is retired."""
    from tools.process_registry import _checkpoint_path
    try:
        entries = json.loads(_checkpoint_path().read_text(encoding='utf-8-sig'))
        return any(row['session_id'] + '.scope.exit' == path.name for row in entries)
    except FileNotFoundError:
        return False
    except (OSError, ValueError, TypeError, KeyError):
        return True


def job_quiescent(session):
    row = read_job_receipt(session)
    return bool(row and row['quiescent'] and type(row['exit_code']) is int)


def inherit_job_scope(entries, previous):
    """Older cached writers cannot erase a runtime-captured job capability."""
    known = {row.get('session_id', row.get('id')) for row in previous if row.get('native_job_scope') is True}
    inherited = set()
    for row in entries:
        sid = row.get('session_id', row.get('id'))
        if sid in known:
            row['native_job_scope'] = True
            inherited.add(sid)
    return inherited


def job_exit_result(session, exit_code, reason, source):
    if not session.native_job_scope:
        return exit_code, reason, source
    row = read_job_receipt(session)
    if row and type(row['exit_code']) is int:
        exit_code = row['exit_code']
    if not job_quiescent(session):
        session._scope_stop_pending = True
        reason, source = 'lost', 'native_job_lost'
    elif reason == 'lost':
        reason, source = 'exited', ''
    return exit_code, reason, source


def finish_detached_scope(registry, session):
    if session.native_job_scope and job_quiescent(session):
        registry._finish_exited(session, read_job_receipt(session)['exit_code'])
    else:
        session._scope_stop_pending = True
        registry._close_reused_detached(session)  # No collected status and no PID signal.
    return session


def record_scope_quiescence(session):
    """Retain an affirmative native unit Stop in the same bound exit receipt."""
    get = session.get if isinstance(session, dict) else lambda name, default=None: getattr(session, name, default)
    unit = get('systemd_unit')
    if not isinstance(unit, str) or not unit.startswith('hermes-worker-'):
        raise ValueError('native unit binding missing')
    from hermes_dir_acl import make_user_dir
    from utils import atomic_json_write
    target = job_receipt_path(get('session_id') or get('id'))
    make_user_dir(target.parent, mode=0o700, parents=True, exist_ok=True)
    atomic_json_write(target, dict(version=1, pid=get('pid'), birth=get('host_start_time'), unit=unit,
                                  exit_code=get('exit_code') if type(get('exit_code')) is int else -15,
                                  quiescent=True), mode=0o600)
    if isinstance(session, dict):
        session['native_job_scope'] = True
    else:
        session.native_job_scope = True


def record_stopped_unit(registry, entry):
    """Output snapshots are imported only after affirmative native unit cleanup."""
    systemd_unit = entry['systemd_unit']
    record_scope_quiescence(entry)
    with registry._lock:
        cached = registry._finished.get(entry['session_id'])
    if cached is None:
        from tools.process_registry_results import load_completed_results
        cached = load_completed_results(owns_entry=lambda row: row['id'] == entry['session_id'],
                                        strict=True).get(entry['session_id'])
        if cached is not None:
            with registry._lock:
                registry._finished.setdefault(cached.id, cached)
    with registry._lock:
        if cached is not None and cached.systemd_unit in ('', systemd_unit):
            cached.native_job_scope = True
            cached.systemd_unit, cached._scope_stop_pending = '', False
            from tools.process_registry_results import save_completed_result
            if not save_completed_result(cached):
                raise OSError('native scope result is not durable')


def job_argv(session, argv, grace):
    from hermes_dir_acl import make_user_dir
    target = job_receipt_path(session.id)
    make_user_dir(target.parent, mode=0o700, parents=True, exist_ok=True)
    session.native_job_scope = True
    return [sys.executable, '-I', '-S', str(Path(__file__).resolve()), str(target), str(grace), '--', *argv]


def request_job_stop(registry, session):
    """Signal only a ready, birth-verified native helper; never force-kill it."""
    if job_quiescent(session):
        return True
    if session.native_job_scope is not True:
        return False
    if (not read_job_receipt(session)
            or not registry._host_pid_is_ours(session.pid, session.host_start_time)):
        return False
    try:
        os.kill(session.pid, signal.SIGTERM)
    except OSError:
        return job_quiescent(session)
    # This is the existing native Stop grace, not a new scheduler or retry owner.
    end = time.monotonic() + max(1.0, 2 * registry._daemon_term_grace_seconds() + 1.0)
    while time.monotonic() < end:
        if job_quiescent(session):
            return True
        time.sleep(.02)
    return job_quiescent(session)


def signal_job_stop(registry, session, consume_output):
    if job_quiescent(session):
        return None
    if not request_job_stop(registry, session):
        return registry._refuse_backend_stop(session, consume_output)
    session._completion_event.wait(1.0)  # Existing reader retains its final output.
    return None


def _main():
    import ctypes
    target, grace = Path(sys.argv[1]), max(0.0, float(sys.argv[2]))
    argv = sys.argv[4:]
    if sys.argv[3] != '--' or not argv:
        raise ValueError('native job argv missing')
    pid = os.getpid()
    stat = Path('/proc/self/stat').read_text(encoding='utf-8')
    birth = int(stat[stat.rindex(')') + 2:].split()[19])
    code, stopped = None, False

    def receipt(quiescent):
        row = dict(version=1, pid=pid, birth=birth, exit_code=code, quiescent=quiescent)
        temporary = target.with_name(target.name + '.' + str(pid) + '.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(row, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for name in ('SIGTERM', 'SIGINT', 'SIGHUP'):
        signal.signal(getattr(signal, name), stop)
    # Set containment before the command can create even its first descendant.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        code = 125
        print('Native Linux job scope unavailable; command was not started.', file=sys.stderr)
        receipt(True)  # No command/child was created.
        return code
    receipt(False)
    try:
        child = subprocess.Popen(argv)
    except OSError:
        code = 126
        receipt(True)
        raise
    sent = {}
    while True:
        # Only this process reaps its children, so a PID returned by the kernel
        # cannot be reused by an unrelated process before the signal below.
        while True:
            try:
                found, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                if code is None:
                    raise RuntimeError('native job lost primary status')
                receipt(True)
                return code if code >= 0 else 128 - code
            if not found:
                break
            sent.pop(found, None)
            if found == child.pid:
                code = child.returncode = os.waitstatus_to_exitcode(status)
                receipt(False)
        if stopped:
            # This is kernel parentage, not a process-name/PGID scan or a ledger.
            direct = Path('/proc/self/task/' + str(pid) + '/children').read_text(encoding='utf-8').split()
            now = time.monotonic()
            for value in direct:
                owned = int(value)
                try:
                    if owned not in sent:
                        os.kill(owned, signal.SIGTERM)
                        sent[owned] = now
                    elif grace > 0 and now - sent[owned] >= grace:
                        os.kill(owned, signal.SIGKILL)  # windows-footgun: ok - Linux-only subreaper executable
                except ProcessLookupError:
                    pass
        time.sleep(.01)


if __name__ == '__main__':
    raise SystemExit(_main())
