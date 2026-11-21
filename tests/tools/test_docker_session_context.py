"""Docker exec must resolve turn identity at spawn, not at container creation."""

import asyncio
from contextvars import Context
import json
import os
import subprocess

import pytest

from gateway import session_context as session
from tools.environments import docker


@pytest.fixture(params=[False, True])
def exec_shell(monkeypatch, request):
    env = object.__new__(docker.DockerEnvironment)
    env._docker_exe = "docker"
    env._container_id = "shared-container"
    env._profile_scoped_passthrough = request.param
    env._init_env_args = ["-e", "HERMES_SESSION_USER_ID"]
    env._init_env_values = {"HERMES_SESSION_USER_ID": "cached-stale"}
    env._init_unset_passthrough_names = ("HERMES_SESSION_USER_ID",)
    monkeypatch.setattr(env, "_build_runtime_env_args_with_unsets", lambda: (
        ["-e", "HERMES_SESSION_USER_ID"], ("HERMES_SESSION_USER_ID",),
        {"HERMES_SESSION_USER_ID": "configured-stale"}))
    calls = []

    def spawn(argv, stdin_data, **kwargs):
        calls.append((argv, kwargs.get("env")))
        # Emulate only Docker's env transfer; run the actual constructed bash script.
        child_env = {"PATH": os.environ["PATH"], "HERMES_SESSION_USER_ID": "baked-stale"}
        client_env = kwargs.get("env") or os.environ
        i = 2
        while argv[i] != env._container_id:
            if argv[i] == "-e":
                name = argv[i + 1]
                if name in client_env:
                    child_env[name] = client_env[name]
                else:
                    child_env.pop(name, None)
                i += 2
            else:
                i += 1
        return subprocess.Popen(argv[i + 1:], env=child_env, stdout=subprocess.PIPE, text=True)

    monkeypatch.setattr(docker, "_popen_bash", spawn)
    monkeypatch.setattr(session, "_session_context_engaged", False)
    for name in session._VAR_MAP:
        monkeypatch.delenv(name, raising=False)
    yield env, calls


def _read(env, login=False):
    command = "python3 -c 'import os,json; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(\"HERMES_SESSION_\")}))'"
    proc = env._run_bash(command, login=login, stdin_data="")
    output, _ = proc.communicate(timeout=10)
    assert proc.returncode == 0
    return json.loads(output)


@pytest.mark.platforms("posix")
def test_docker_exec_resolves_concurrent_turns_and_cleared_context(exec_shell, monkeypatch):
    env, calls = exec_shell
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "process-stale")

    async def turn(user):
        session.set_session_vars(platform="telegram", user_id=user)
        await asyncio.sleep(0)
        for login in (True, False):
            observed = _read(env, login)
            assert observed["HERMES_SESSION_USER_ID"] == user
            assert observed["HERMES_SESSION_PLATFORM"] == "telegram"
        session.clear_session_vars([])
        assert _read(env).get("HERMES_SESSION_USER_ID", "") == ""

    async def run():
        await asyncio.gather(turn("user-a"), turn("user-b"))

    Context().run(asyncio.run, run())
    assert Context().run(_read, env).get("HERMES_SESSION_USER_ID", "") == ""
    for argv, _ in calls:
        assert not any(value in " ".join(argv) for value in ("user-a", "user-b", "process-stale"))


@pytest.mark.platforms("posix")
def test_docker_exec_preserves_unengaged_cli_mirror(exec_shell, monkeypatch):
    env, _ = exec_shell
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "cli-user")
    assert Context().run(_read, env)["HERMES_SESSION_USER_ID"] == "cli-user"
