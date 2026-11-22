"""Foreground tracking keeps login-profile functions and aliases in the session snapshot."""

import os
import subprocess

import pytest

from tests.waiting import wait_until

from tests.tools import test_docker_backend_real as real_docker
from tools import terminal_tool

pytestmark = pytest.mark.skipif(not os.getenv("HERMES_TEST_DOCKER_IMAGE"),
                                reason="HERMES_TEST_DOCKER_IMAGE not set")
docker_task = real_docker.docker_task


@pytest.mark.platforms("linux")
def test_snapshot_keeps_login_profile_functions_and_aliases(docker_task):
    real_docker._run("true", docker_task)
    env = terminal_tool._active_environments[real_docker._env_key(docker_task)]
    profile = ("hermes_profile_fn() { echo fn-from-profile; }\n"
               "shopt -s expand_aliases\nalias hermes_profile_alias='echo alias-from-profile'\n")
    subprocess.run(["docker", "exec", "-i", env._container_id, "bash", "-c",
                    "cat > /etc/profile.d/zz-hermes-test.sh"],
                   input=profile, text=True, check=True, timeout=30)
    env.init_session()
    assert env._snapshot_ready

    result = real_docker._run("hermes_profile_fn; hermes_profile_alias", docker_task)

    assert result["exit_code"] == 0, result
    assert "fn-from-profile" in result["output"]
    assert "alias-from-profile" in result["output"]


@pytest.mark.platforms("linux")
def test_login_fallback_commands_stay_owned_and_stoppable(docker_task):
    real_docker._run("true", docker_task)
    env = terminal_tool._active_environments[real_docker._env_key(docker_task)]

    def survivors():
        probe = ("for f in /proc/[0-9]*/cmdline; do tr '\\0' ' ' < \"$f\" 2>/dev/null; echo; done "
                 "| grep -c '^sleep 3017' || :")
        out = subprocess.run(["docker", "exec", env._container_id, "bash", "-c", probe],
                             capture_output=True, text=True, timeout=30, check=False).stdout
        return int(out.strip() or 0)

    proc = env._run_bash("sleep 3017", login=True)
    record = env._foreground_commands[id(proc)]
    wait_until(lambda: survivors() == 1, timeout=30)
    record.pid()

    env._kill_process(proc)

    wait_until(lambda: survivors() == 0, timeout=30)
