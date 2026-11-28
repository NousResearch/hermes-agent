"""Zero TERM grace requests termination without escalating an uncooperative group."""

import os
import shlex
import uuid

import pytest

from tests.tools import test_docker_backend_real as real_docker
from tests.waiting import wait_until
from tools import terminal_tool

pytestmark = pytest.mark.skipif(not os.getenv("HERMES_TEST_DOCKER_IMAGE"),
                                reason="HERMES_TEST_DOCKER_IMAGE not set")
docker_task = real_docker.docker_task


@pytest.mark.platforms("linux")
def test_zero_grace_delivers_term_without_killing_survivors(docker_task):
    real_docker._run("true", docker_task)
    env = terminal_tool._active_environments[real_docker._env_key(docker_task)]
    root = f"/tmp/{uuid.uuid4().hex}"
    script = ("import pathlib,signal; "
              f"signal.signal(signal.SIGTERM,lambda *_:pathlib.Path({root + '.term'!r}).touch()); "
              f"pathlib.Path({root + '.ready'!r}).touch(); "
              "exec('while True: signal.pause()')")
    proc = env._run_bash("exec python3 -c " + shlex.quote(script))
    record = env._foreground_commands[id(proc)]
    record.term_grace = 0
    pid = None
    try:
        wait_until(lambda: record.control(f"test -e {root}.ready")["returncode"] == 0,
                   what="TERM handler installed in the namespace")
        pid = record.pid()
        record.kill()
        alive = record.control(f"read -r s < /proc/{pid}/stat || exit 1; s=${{s##*) }}; "
                               'set -- $s; [ "$1" != Z ]')
        assert alive["returncode"] == 0, "zero grace escalated TERM to KILL"
        wait_until(lambda: record.control(f"test -e {root}.term")["returncode"] == 0,
                   what="TERM delivered without escalation")
    finally:
        if pid is not None:
            record.control(f"kill -KILL -- -{pid} 2>/dev/null || :")
        proc.wait(timeout=10)
        env._foreground_commands.pop(id(proc), None)
