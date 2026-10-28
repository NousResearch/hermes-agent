"""Reviewer regression: final original bytes arrive between EOF read and status poll."""
import builtins
from pathlib import Path
import shlex
import sys

import pytest

from tests.waiting import wait_until
from tools import process_registry as registry_module
from tools import process_registry_chat_execution as execution
from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.platforms('linux')


def test_foreground_keeps_tail_published_between_read_and_exit_observation(tmp_path, monkeypatch):
    registry = registry_module.ProcessRegistry()
    monkeypatch.setattr(registry_module, 'process_registry', registry)
    ready, release = tmp_path / 'ready', tmp_path / 'release'
    payload = f'''from pathlib import Path
import time
print('prefix',flush=True)
Path({str(ready)!r}).touch()
end=time.monotonic()+15
while not Path({str(release)!r}).exists() and time.monotonic()<end: time.sleep(.01)
print('tail',flush=True)
raise SystemExit(7)
'''
    command = shlex.join([sys.executable, '-c', payload])
    session = registry._new_session(command, 'owner', 'owner', 'route', str(tmp_path))
    env = LocalEnvironment(cwd=str(tmp_path))
    handle = env._run_bash(command, native_session=session)
    try:
        wait_until(ready.exists, timeout=5)
        class EmptyReadBoundary:
            def __init__(self, stream):
                self.stream = stream
                self.released = False
            def __getattr__(self, name):
                return getattr(self.stream, name)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def read(self, count):
                chunk = self.stream.read(count)
                if not chunk and not self.released:
                    self.released = True
                    release.touch()
                    assert session._completion_event.wait(10)
                return chunk
        monkeypatch.setattr(execution, 'open', lambda *args, **kwargs: EmptyReadBoundary(builtins.open(*args, **kwargs)), raising=False)
        inline_output = ''.join(handle.stdout)
        assert handle.poll() == session.exit_code == 7
        assert session.output_buffer == 'prefix\ntail\n'
        assert inline_output == session.output_buffer
    finally:
        release.touch()
        registry.kill_process(session.id)
        handle.stdout.close()
        env.cleanup()
