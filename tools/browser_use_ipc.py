"""No-spawn shutdown for the Browser Harness v0.1.9 IPC contract.

Newline JSON ping/shutdown also exists in v0.1.13. Unknown replies fail closed.
Never invokes admin.restart_daemon/--reload (they unlink evidence on failure),
never sends OS signals, and never interprets an acknowledgement as process exit.
Only Hermes-managed private runtimes are eligible. Browser/provider ownership
remains with the existing backend; this confirms the Harness process only.
"""
import json
import os
import shutil
import socket
import time
from pathlib import Path


def _stamp(path):
    st = Path(path).stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _request(evidence, meta, deadline):
    """v0.1.9 _ipc.connect/request framing, without package import side effects."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Harness shutdown deadline")
    token = evidence['token']
    if os.name == 'nt':
        conn = socket.create_connection(('127.0.0.1', evidence['port']), timeout=remaining)
    else:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(remaining)
        try:
            conn.connect(evidence['endpoint'])
        except BaseException:
            conn.close()
            raise
    with conn:
        req = {'meta': meta}
        if token is not None:
            req['token'] = token
        conn.settimeout(max(.001, deadline - time.monotonic()))
        conn.sendall((json.dumps(req) + '\n').encode())
        data = b''
        while not data.endswith(b'\n'):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Harness shutdown deadline')
            conn.settimeout(remaining)
            chunk = conn.recv(4096)
            if not chunk:
                raise OSError('Incomplete Harness response')
            data += chunk
            if len(data) > 65536:
                raise ValueError('Oversized Harness response')
        return json.loads(data)


def _snapshot(resource):
    from gateway.status import get_process_start_time
    from tools.browser_tool_lifecycle import _settled_pid_record, _verify_reapable_browser_daemon
    runtime = Path(resource.runtime)
    marker = runtime / (resource.identity + '.owner_pid')
    if runtime.is_symlink() or marker.read_text(encoding='utf-8') != str(os.getpid()):
        raise ValueError('Managed runtime ownership changed')
    record = _settled_pid_record(str(runtime / 'bu.pid'))
    if record is None:
        raise ValueError('Unsettled Harness PID')
    pid, text, stat = record
    start = get_process_start_time(pid)
    if start is None or not _verify_reapable_browser_daemon(
            pid, str(runtime), resource.identity, expected_start=start):
        raise ValueError('Unverified Harness identity')
    endpoint = Path(resource.key)
    stamp = _stamp(endpoint)
    port, token = None, None
    if os.name == 'nt':
        data = json.loads(endpoint.read_text(encoding='utf-8'))
        port, token = data['port'], data['token']
        if type(port) is not int or not 0 < port < 65536 or not isinstance(token, str) or not token:
            raise ValueError('Invalid Harness port record')
    if _stamp(endpoint) != stamp:
        raise ValueError('Harness endpoint replaced')
    directory = runtime.stat()
    return dict(pid=pid, start=start, pid_text=text, pid_stat=stat,
                directory=(directory.st_dev, directory.st_ino), marker=_stamp(marker),
                endpoint=str(endpoint), endpoint_stamp=stamp, port=port, token=token, ack=False, sent=False)


def _unchanged(resource, evidence, runtime=None):
    from tools.browser_tool_lifecycle import _pid_file_still_matches
    runtime = Path(runtime or resource.runtime)
    st = runtime.stat()
    if runtime.is_symlink() or (st.st_dev, st.st_ino) != evidence['directory']:
        return False
    marker = runtime / (resource.identity + '.owner_pid')
    if _stamp(marker) != evidence['marker']:
        return False
    pid_file = runtime / 'bu.pid'
    if pid_file.exists():
        if not _pid_file_still_matches(str(pid_file), evidence['pid_text'], evidence['pid_stat']):
            return False
    elif not evidence['sent']:
        return False
    endpoint = runtime / Path(evidence['endpoint']).name
    if endpoint.exists():
        if _stamp(endpoint) != evidence['endpoint_stamp']:
            return False
    elif not evidence['sent']:
        return False
    return True


def _remove_stopped_runtime(resource, evidence):
    if not _unchanged(resource, evidence):
        return False
    runtime = Path(resource.runtime)
    claim = runtime.with_name(f'.hermes-browser-reap-{os.getpid()}-{time.time_ns()}')
    runtime.rename(claim)
    # Directory/record replacement between check and rename must not be deleted.
    if not _unchanged(resource, evidence, claim):
        return False
    shutil.rmtree(claim)
    return True


def stop_managed_harness(resource, timeout=5.0):
    from gateway.status import _pid_exists, get_process_start_time
    from tools.browser_tool_lifecycle import _verify_reapable_browser_daemon
    if not resource.managed:
        return False
    deadline = time.monotonic() + timeout
    try:
        evidence = resource.evidence
        if evidence is None:
            evidence = resource.evidence = _snapshot(resource)
        if not _unchanged(resource, evidence):
            return False
        pid, start = evidence['pid'], evidence['start']
        if not _pid_exists(pid):
            # A successful prior request may have removed IPC long before exit.
            return evidence['sent'] and _remove_stopped_runtime(resource, evidence)
        if get_process_start_time(pid) != start:
            return False  # recycled/unknown: don't touch the new generation
        if not evidence['ack']:
            reply = _request(evidence, 'ping', deadline)
            if not isinstance(reply, dict) or reply.get('pong') is not True or type(reply.get('pid')) is not int or reply['pid'] != pid:
                return False
            if not _unchanged(resource, evidence) or not _verify_reapable_browser_daemon(
                    pid, resource.runtime, resource.identity, expected_start=start):
                return False
            evidence['sent'] = True  # verified request may reach daemon even if ACK is lost
            reply = _request(evidence, 'shutdown', deadline)
            if not isinstance(reply, dict) or reply.get('ok') is not True or reply.get('error'):
                return False
            evidence['ack'] = True
        while time.monotonic() < deadline:
            if not _pid_exists(pid):
                return _remove_stopped_runtime(resource, evidence)
            if get_process_start_time(pid) != start:
                return False
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        resource.last_error = type(exc).__name__
    return False
