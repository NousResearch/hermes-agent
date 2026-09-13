"""The model must be told when the backend replaced its container/sandbox
mid-command (ported from lobehub/lobehub#19329): silent recovery leaves the
agent assuming background processes and unsynced files survived."""

import json

from tools.environments import docker as docker_env
from tools.environments.local import LocalEnvironment
from tools.terminal_tool_result import finalize_foreground_result


def test_execute_folds_recreation_flag_once():
    """A pending recreation mark surfaces as ``environment_recreated: True``
    on the next execute() result and is consumed — the following command
    reports a clean result. Exercises the real BaseEnvironment.execute path."""
    env = LocalEnvironment(cwd=".", timeout=30)
    try:
        env._mark_recreated()
        result = env.execute("echo hi")
        assert result["returncode"] == 0
        assert result.get("environment_recreated") is True

        result2 = env.execute("echo again")
        assert "environment_recreated" not in result2
    finally:
        env.cleanup()


def test_docker_recovery_marks_pending_and_finalizer_warns(monkeypatch):
    """Docker's out-of-band recovery sets the pending mark, and the tool-layer
    finalizer turns the flag into a model-facing warning — omitted entirely
    when the backend never flagged a recreation."""
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env._container_id = "old"
    env._labels = {}
    env._image = ""
    monkeypatch.setattr(
        docker_env.DockerEnvironment, "_find_reusable_container",
        lambda self, *a: ("newcid", "running"))
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
    assert env._recreate_container() is True
    assert getattr(env, "_recreated_notice_pending", False) is True

    common: dict = dict(
        command="echo hi", env=env, env_type="docker", effective_task_id="t",
        task_id="t", session_id="s", session_key="k", workdir=None,
        command_cwd=None, approval_note=None)

    flagged = json.loads(finalize_foreground_result(
        result={"output": "hi", "returncode": 0, "environment_recreated": True}, **common))
    assert "recreated" in flagged.get("environment_recreated", "")
    assert "may be lost" in flagged["environment_recreated"]

    clean = json.loads(finalize_foreground_result(
        result={"output": "hi", "returncode": 0}, **common))
    assert "environment_recreated" not in clean
