"""File admission is bounded independently of long-running turns."""
import io
import threading
from types import SimpleNamespace

from tui_gateway.compute_host import ComputeHost


def test_file_open_succeeds_while_turn_executor_is_saturated(monkeypatch, tmp_path):
    from tui_gateway import server
    from tools import terminal_tool
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    token = set_terminal_scope({'TERMINAL_ENV': 'test_remote'})
    from hermes_cli.web_server_files import _agent_path_scope
    with _agent_path_scope(tmp_path):
        key = terminal_tool._qualify_task_key('session:durable')
    monkeypatch.setitem(terminal_tool._active_environments, key,
                        SimpleNamespace(cwd='/workspace', _hermes_backend_name='test_remote'))
    monkeypatch.setitem(server._sessions, 'live', {'session_key': 'durable', 'profile_home': str(tmp_path)})
    (tmp_path / 'config.yaml').write_text('terminal:\n  backend: test_remote\n')
    host = ComputeHost(stdout=io.StringIO(), max_workers=1, heartbeat_secs=0)
    entered, release, replied = threading.Event(), threading.Event(), threading.Event()
    replies = []
    def turn():
        entered.set()
        release.wait(10)
    def emit(frame):
        replies.append(frame)
        replied.set()
    monkeypatch.setattr(host, 'emit', emit)
    future = host._executor.submit(turn)
    try:
        assert entered.wait(5)
        host.handle_frame({'type': 'terminal.file', 'operation': 'open', 'sid': 'live',
                           'session_key': 'durable', 'profile_home': str(tmp_path), 'backend': 'test_remote'})
        assert replied.wait(3), 'file open was starved by an unrelated turn'
        assert replies[0]['type'] == 'terminal.file.ack', replies
        assert replies[0]['result']['cwd'] == '/workspace'
    finally:
        release.set()
        future.result(timeout=5)
        host.close()
        reset_terminal_scope(token)


def test_file_executor_submit_failure_releases_admission(monkeypatch):
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    def failed(*args, **kwargs):
        raise RuntimeError('executor closed concurrently')
    executor = getattr(host, '_file_executor', host._executor)
    monkeypatch.setattr(executor, 'submit', failed)
    try:
        host.handle_frame({'type': 'terminal.file', 'operation': 'open'})
        assert all(host._file_slots.acquire(blocking=False) for _ in range(4))
        assert not host._file_slots.acquire(blocking=False)
    finally:
        host.close()


def test_file_close_cancels_queued_work_and_releases_all_slots(monkeypatch):
    import concurrent.futures
    from tui_gateway import terminal_files
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)
    host._file_executor.shutdown()
    host._file_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    entered, release = threading.Event(), threading.Event()
    def read(*args):
        entered.set()
        release.wait(10)
        return {'cwd': '/workspace'}
    monkeypatch.setattr(terminal_files, 'handle_file_request', read)
    submitted = []
    submit = host._file_executor.submit
    def capture(*args, **kwargs):
        future = submit(*args, **kwargs)
        submitted.append(future)
        return future
    monkeypatch.setattr(host._file_executor, 'submit', capture)
    try:
        host.handle_frame({'type': 'terminal.file'})
        assert entered.wait(5)
        for _ in range(3):
            host.handle_frame({'type': 'terminal.file'})
        host.handle_frame({'type': 'terminal.file', 'request_id': 'excess'})
        assert len(submitted) == 4
        host.close()
        assert all(f.cancelled() for f in submitted[1:])
        assert all(host._file_slots.acquire(blocking=False) for _ in range(3))
        assert not host._file_slots.acquire(blocking=False)
        finished = threading.Event()
        submitted[0].add_done_callback(lambda future: finished.set())
        release.set()
        assert finished.wait(5)
        assert host._file_slots.acquire(blocking=False)
    finally:
        release.set()
        host._file_executor.shutdown(wait=True)
        host.close()
