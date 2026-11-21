"""Cancellation must stop a Docker foreground group, not adopt it after a cleanup sweep."""

import os
import subprocess
import threading
import uuid

import pytest

from agent.interrupt_control import InterruptControlMixin
from agent.tool_executor import _interrupt_worker_tids
from gateway import run as gateway_run
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from tests.tools import test_docker_backend_real as real_docker
from tests.tools.test_docker_backend_real import _container_id, _env_key, _run
from tests.waiting import wait_until
from tools import interrupt, terminal_tool, terminal_tool_background
from tools.environments import base
from tools.process_registry import process_registry

pytestmark = pytest.mark.skipif(not os.getenv("HERMES_TEST_DOCKER_IMAGE"),
                                reason="HERMES_TEST_DOCKER_IMAGE not set")
docker_task = real_docker.docker_task


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("producer", ["stop", "reset", "shutdown", "interrupt", "worker_cancel",
                                    "stop_after_yield", "stop_during_handoff"])
def test_cancel_terminates_namespace_before_foreground_returns(docker_task, producer, monkeypatch):
    root = f"/tmp/{uuid.uuid4().hex}"
    result, errors, reaped_counts = {}, [], []
    armed, paused, resume, reaped = (threading.Event() for _ in range(4))
    adopting, finish_adoption = threading.Event(), threading.Event()
    original_check = base.is_interrupted
    original_reap = gateway_run._reap_gateway_turn_processes
    original_adopt = terminal_tool_background.adopt_foreground

    def adopt(*args, **kwargs):
        if producer == "stop_during_handoff":
            adopting.set()
            assert finish_adoption.wait(10)
        return original_adopt(*args, **kwargs)

    def check():
        if armed.is_set():
            paused.set()
            assert resume.wait(10), "cancel producer did not release the foreground poll"
        return original_check()

    def reap(*args, **kwargs):
        try:
            count = original_reap(*args, **kwargs)
            reaped_counts.append(count)
            return count
        finally:
            reaped.set()

    def worker():
        try:
            result.update(_run(f"printf '1\\n' > {root}.writes; touch {root}.ready; "
                               f"while ! test -e {root}.release; do sleep 0.05; done; "
                               f"printf '2\\n' >> {root}.writes; touch {root}.finished", docker_task, timeout=60))
        except BaseException as exc:
            errors.append(exc)
        finally:
            interrupt.clear_current_thread_interrupt()

    monkeypatch.setattr(base, "is_interrupted", check)
    monkeypatch.setattr(gateway_run, "_reap_gateway_turn_processes", reap)
    monkeypatch.setattr(terminal_tool_background, "adopt_foreground", adopt)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    cid = wait_until(lambda: _container_id(docker_task), timeout=30, what="Docker environment")

    def control(*args):
        return subprocess.run(["docker", "exec", cid, *args], capture_output=True, text=True,
                              timeout=10, check=False)

    try:
        wait_until(lambda: control("test", "-e", root + ".ready").returncode == 0, what="running command")
        env = terminal_tool._active_environments[_env_key(docker_task)]
        record = next(iter(env._foreground_commands.values()))
        group = record.pid()
        assert group is not None and group > 1
        agent = InterruptControlMixin()
        agent._execution_thread_id = thread.ident
        agent._tool_worker_threads, agent._tool_worker_threads_lock = {thread.ident}, threading.Lock()
        agent._active_children, agent._active_children_lock, agent.quiet_mode = [], threading.Lock(), True
        agent._gateway_turn_process_task_id = docker_task
        agent._gateway_turn_process_baseline = process_registry.snapshot_running_ids(docker_task)
        runner = GatewayRunner(GatewayConfig())
        runner._session_state("key").turn.agent = agent
        armed.set()
        assert paused.wait(10), "foreground poll did not reach the cancel boundary"
        def cancel_turn():
            if producer == "stop_during_handoff":
                interrupt.request_yield(thread.ident)
                resume.set()
                assert adopting.wait(10)
            runner._interrupt_running_turn("key", interrupt_reason=producer, invalidation_reason=producer)
            assert reaped.wait(10)
            assert reaped_counts == [0], "the pre-adoption sweep must complete before the tool resumes"
            finish_adoption.set()

        def stop_after_yield():
            interrupt.request_yield(thread.ident)
            agent.hard_interrupt("Stop requested")

        producers = {
            "stop": cancel_turn,
            "reset": cancel_turn,
            "stop_during_handoff": cancel_turn,
            "shutdown": lambda: runner._interrupt_running_agents("Gateway shutting down"),
            "worker_cancel": lambda: _interrupt_worker_tids(agent, [thread.ident]),
            "interrupt": lambda: agent.interrupt("cancel current work"),
            "stop_after_yield": stop_after_yield,
        }
        producers[producer]()
        resume.set()
        thread.join(timeout=15)
        assert not thread.is_alive() and not errors, errors
        live_group = control("bash", "-c", 'for f in /proc/[0-9]*/stat; do '
                             'read -r s < "$f" 2>/dev/null || continue; s=${s##*) }; set -- $s; '
                             f'[ "$3" = "{group}" ] && [ "$1" != Z ] && exit 1; done; exit 0')
        assert live_group.returncode == 0, "cancelled Docker workload is still running in its namespace"
        assert result["exit_code"] == 130 and "[Command interrupted]" in result["output"], result
        assert "session_id" not in result, "cancellation must not create a background job"
        assert control("cat", root + ".writes").stdout == "1\n"
        assert control("test", "-e", root + ".finished").returncode != 0
        assert not process_registry.running_owned_by(docker_task)
    finally:
        finish_adoption.set()
        resume.set()
        interrupt.set_interrupt(False, thread_id=thread.ident)
        control("touch", root + ".release")
        thread.join(timeout=10)
        if result.get("session_id"):
            process_registry.kill_process(result["session_id"])
