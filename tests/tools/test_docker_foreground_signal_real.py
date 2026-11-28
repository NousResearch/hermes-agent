"""Trace actual namespace signal delivery, not the host docker-exec client's exit."""

import json
import os
import re
import shlex
import subprocess
import threading
import uuid

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from tests.tools import test_docker_backend_real as real_docker
from tests.tools.test_docker_backend_real import _container_id, _run
from tests.waiting import wait_until
from tools import interrupt
from tools.environments.docker import DockerEnvironment
from tools.process_registry import process_registry

pytestmark = pytest.mark.skipif(not os.getenv("HERMES_TEST_DOCKER_IMAGE"),
                                reason="HERMES_TEST_DOCKER_IMAGE not set")
docker_task = real_docker.docker_task


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("term_mode", ["default", "cleanup", "ignore"])
def test_stop_signals_owned_group_before_waiting_for_grace(docker_task, term_mode, monkeypatch):
    _run("true", docker_task)
    cid = _container_id(docker_task)

    def control(*args):
        return subprocess.run(["docker", "exec", cid, *args], capture_output=True, text=True,
                              timeout=10, check=False)

    if control("sh", "-c", "command -v strace && command -v python3").returncode:
        pytest.skip("signal-delivery contract image needs strace and python3")
    root = f"/tmp/{uuid.uuid4().hex}"
    script = f'''import json,os,signal,time
def write(kind):
 with open({root + '.writes'!r},'a') as out:
  out.write(json.dumps(dict(t=time.time(),kind=kind,pid=os.getpid()))+'\\n')
def cleanup(signum,frame):
 write('cleanup')
 raise SystemExit(0)
mode={term_mode!r}
if mode=='cleanup': signal.signal(signal.SIGTERM,cleanup)
if mode=='ignore': signal.signal(signal.SIGTERM,signal.SIG_IGN)
while True:
 write('counter')
 time.sleep(.05)
'''
    assert control("sh", "-c", f"printf %s {shlex.quote(script)} > {root}.py").returncode == 0
    original = DockerEnvironment._run_bash_untracked

    def traced(self, command, **kwargs):
        if root + ".py" in command:
            command = f"strace -ff -ttt -e trace=none -e signal=TERM,KILL -o {root}.trace bash -c {shlex.quote(command)}"
        return original(self, command, **kwargs)

    monkeypatch.setattr(DockerEnvironment, "_run_bash_untracked", traced)
    result, errors = {}, []

    def worker():
        try:
            result.update(_run(f"python3 {root}.py", docker_task, timeout=60))
        except BaseException as exc:
            errors.append(exc)
        finally:
            interrupt.clear_current_thread_interrupt()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        wait_until(lambda: len(control("cat", root + ".writes").stdout.splitlines()) >= 3,
                   what="non-idempotent counter running")
        agent = InterruptControlMixin()
        agent._execution_thread_id = thread.ident
        agent._tool_worker_threads, agent._tool_worker_threads_lock = {thread.ident}, threading.Lock()
        agent._active_children, agent._active_children_lock, agent.quiet_mode = [], threading.Lock(), True
        runner = GatewayRunner(GatewayConfig())
        runner._session_state("key").turn.agent = agent
        runner._interrupt_running_turn("key", interrupt_reason="Stop requested", invalidation_reason="stop_command")
        thread.join(timeout=15)
        assert not thread.is_alive() and not errors, errors
        assert result["exit_code"] == 130 and "session_id" not in result, result
        traces = control("sh", "-c", f"cat {root}.trace.*").stdout
        delivered = sorted(float(t) for t in re.findall(r"(\d+\.\d+) --- SIGTERM", traces))
        assert delivered, traces
        rows = [json.loads(line) for line in control("cat", root + ".writes").stdout.splitlines()]
        late = [r for r in rows if r["t"] > delivered[0]]
        print(json.dumps(dict(mode=term_mode, term_deliveries=delivered, writes_after_first_term=late,
                              trace=traces)), flush=True)
        if term_mode == "default":
            assert not late, "non-trapping counter wrote after TERM delivery to its owned group"
        elif term_mode == "cleanup":
            assert any(row["kind"] == "cleanup" for row in late), "TERM handler may write during grace"
        else:
            assert late and "killed by SIGKILL" in traces, "TERM-ignoring work must be killed after grace"
        assert not process_registry.running_owned_by(docker_task)
        for row in rows[-1:]:
            assert control("sh", "-c", f"test ! -e /proc/{row['pid']}/stat").returncode == 0
    finally:
        interrupt.set_interrupt(True, thread_id=thread.ident)
        thread.join(timeout=15)
        interrupt.set_interrupt(False, thread_id=thread.ident)
