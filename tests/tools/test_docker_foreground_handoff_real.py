"""Foreground handoff uses the real Docker backend fixture and opt-in image gate."""

import asyncio
import os
import shlex
import subprocess
import threading
import uuid

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.tools import test_docker_backend_real as real_docker
from tests.tools.test_docker_backend_real import _container_id, _run
from tests.waiting import wait_until
from tools import interrupt
from tools.process_registry import process_registry

pytestmark = pytest.mark.skipif(not os.getenv("HERMES_TEST_DOCKER_IMAGE"),
                                reason="HERMES_TEST_DOCKER_IMAGE not set")
docker_task = real_docker.docker_task


class _Agent(InterruptControlMixin):
    _supports_active_turn_redirect = True
    _executing_tools = True
    _interrupt_requested = False


@pytest.mark.parametrize("signal_kind", ["yield", "attachment", "kill", "graceful-kill", "stale-kill"])
def test_foreground_handoff_keeps_one_process_and_real_completion(docker_task, signal_kind):
    ready, release, writes = (f"/tmp/{uuid.uuid4().hex}" for _ in range(3))
    result = {}

    def worker():
        trap = f"trap 'printf cleaned > {ready}.cleaned; exit 130' TERM; " if signal_kind == "graceful-kill" else ""
        try:
            result.update(_run(trap + f"printf 'started\\n'; printf '1\\n' >> {writes}; touch {ready}; "
                               f"while ! test -e {release}; do sleep 0.05; done; printf 'done\\n'; exit 7",
                               docker_task, timeout=60))
        finally:
            interrupt.clear_current_thread_interrupt()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    sid = sentinel = None
    cid = wait_until(lambda: _container_id(docker_task), timeout=30, what="Docker environment")

    def control(*args):
        return subprocess.run(["docker", "exec", cid, *args], capture_output=True, text=True,
                              timeout=10, check=False)

    try:
        wait_until(lambda: control("test", "-e", ready).returncode == 0, what="running sandbox command")
        if signal_kind in {"yield", "kill", "graceful-kill", "stale-kill"}:
            interrupt.request_yield(thread.ident)
        else:
            agent = _Agent()
            agent._tool_worker_threads = {thread.ident}
            agent._tool_worker_threads_lock = threading.Lock()
            runner = GatewayRunner(GatewayConfig())
            runner._session_state("key").turn.agent = agent
            source = SessionSource(platform=Platform.SLACK, chat_id="lab", user_id="requester")
            event = MessageEvent(text="Read the attachment", source=source, message_type=MessageType.DOCUMENT,
                                 media_urls=["/cached/note.txt"], media_types=["text/plain"])
            outcome = asyncio.run(runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent))
            assert outcome.redirected and not agent._interrupt_requested
            assert "/cached/note.txt" in agent._pending_steer
        thread.join(timeout=10)
        assert not thread.is_alive(), "inbound input must release the foreground wait"
        assert result["status"] == "yielded_to_background", result
        sid = result["session_id"]
        assert result["notify_on_complete"] is True
        session = process_registry.get(sid)
        assert session.pid_scope == "sandbox"
        assert control("kill", "-0", str(result["pid"])).returncode == 0
        assert process_registry.poll(sid)["status"] == "running"
        if signal_kind.endswith("kill"):
            sentinel = control("bash", "-c", "set -m; sleep 1000 >/dev/null 2>&1 & echo $!").stdout.strip()
            if signal_kind == "stale-kill":
                record = session.env_ref._foreground_commands[id(session.process)]
                published = control("cat", record.directory + "/pid").stdout
                control("bash", "-c", f"printf '%s' {sentinel} > {record.directory}/pid")
                try:
                    record.kill()
                    alive = control("bash", "-c", f'read -r s < /proc/{sentinel}/stat || exit 1; '
                                    's=${s##*) }; set -- $s; test "$1" != Z')
                    assert alive.returncode == 0, "a stale ownership record must not kill another process"
                finally:
                    control("bash", "-c", f"printf '%s' {shlex.quote(published)} > {record.directory}/pid")
            killed = process_registry.kill_process(sid)
            assert killed["status"] == "killed", killed
            group = result["pid"]
            live_group = control("bash", "-c", 'for f in /proc/[0-9]*/stat; do '
                                 'read -r s < "$f" 2>/dev/null || continue; s=${s##*) }; set -- $s; '
                                 f'[ "$3" = "{group}" ] && [ "$1" != Z ] && exit 1; done; exit 0')
            assert live_group.returncode == 0
            assert control("kill", "-0", sentinel).returncode == 0
            if signal_kind == "graceful-kill":
                assert control("test", "-f", ready + ".cleaned").returncode == 0
            return
        control("touch", release)
        wait_until(lambda: session.exited, what="managed sandbox completion")
        assert session.exit_code == 7
        assert "started" in session.output_buffer and "done" in session.output_buffer
        assert control("cat", writes).stdout == "1\n"
        completion = process_registry.completion_queue.get(timeout=10)
        assert completion["session_id"] == sid and completion["exit_code"] == 7
        assert "done" in completion["output"]
        assert process_registry.wait(sid, timeout=10)["exit_code"] == 7
    finally:
        interrupt.set_interrupt(False, thread_id=thread.ident)
        interrupt.consume_yield(thread.ident)
        control("touch", release)
        if sentinel:
            control("kill", "-9", sentinel)
        thread.join(timeout=10)
        if sid and not process_registry.get(sid).exited:
            process_registry.kill_process(sid)
