"""A background dev server's listening port shows up wherever its process is listed."""

import shlex
import sys
import time

import pytest


@pytest.mark.platforms("posix")
def test_list_reports_the_port_a_background_server_listens_on(tmp_path):
    import tools.process_registry as module
    import tools.listening_ports as ports_module

    module._SYSTEMD_SCOPE_AVAILABLE = False
    registry = module.ProcessRegistry()
    # A shell wrapping the real server, as `npm run dev` would: the port belongs to a descendant.
    server = ("import socket,sys,time; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen();"
              "print(s.getsockname()[1], flush=True); time.sleep(60)")
    session = registry.spawn_local(f"sh -c {shlex.quote(f'{shlex.quote(sys.executable)} -c {shlex.quote(server)}')}",
                                   cwd=str(tmp_path))
    try:
        deadline = time.monotonic() + 15
        while not session.output_buffer.strip() and time.monotonic() < deadline:
            time.sleep(0.05)
        bound = int(session.output_buffer.split()[0])
        ports_module._cache.clear()
        [entry] = [e for e in registry.list_sessions() if e["session_id"] == session.id]
        assert entry["ports"] == [bound]

        registry.kill_process(session.id)
        deadline = time.monotonic() + 15
        while not session.exited and time.monotonic() < deadline:
            time.sleep(0.05)
        [entry] = [e for e in registry.list_sessions() if e["session_id"] == session.id]
        assert "ports" not in entry  # a finished process advertises nothing
    finally:
        registry.kill_process(session.id)
