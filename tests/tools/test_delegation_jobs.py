import json
import os
from pathlib import Path

import pytest


class _FakeParent:
    session_id = "parent-session"
    _delegate_depth = 0
    _subagent_id = None
    tool_progress_callback = None
    enabled_toolsets = ["terminal", "file", "web", "delegation"]
    valid_tool_names = []


class _FakePopen:
    calls = []
    _next_pid = 41000

    def __init__(self, argv, **kwargs):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = 0
        self.argv = argv
        self.kwargs = kwargs
        type(self).calls.append((argv, kwargs, self))

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


@pytest.fixture(autouse=True)
def _clean_fake_popen():
    _FakePopen.calls.clear()
    yield
    _FakePopen.calls.clear()


def test_start_jobs_builds_profile_subprocess_request(monkeypatch, tmp_path):
    from tools import delegation_jobs

    hermes_home = tmp_path / "hermes"
    (hermes_home / "profiles" / "coder").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_DELEGATE_COMMAND", "/bin/hermes-test")
    monkeypatch.setattr(delegation_jobs.subprocess, "Popen", _FakePopen)

    result = delegation_jobs.start_jobs(
        [
            {
                "goal": "check profile routing",
                "context": "ctx",
                "toolsets": ["terminal"],
                "profile": "coder",
                "role": "leaf",
                "model": "test-model",
                "provider": "test-provider",
                "skills": ["cluster-topology"],
                "resume": "sess-previous",
                "pass_session_id": True,
                "system_prompt": "child prompt",
                "max_iterations": 7,
            }
        ],
        parent_agent=_FakeParent(),
        cfg={"allowed_profiles": ["coder"], "default_profile": "", "job_retention_hours": 24},
    )

    assert result["status"] == "running"
    job_id = result["job_ids"][0]
    argv, kwargs, proc = _FakePopen.calls[0]
    assert argv[:4] == ["/bin/hermes-test", "--profile", "coder", "chat"]
    assert "--delegate-request" in argv
    assert "--delegate-output" in argv
    assert kwargs["env"]["HERMES_HOME"] == str(hermes_home / "profiles" / "coder")
    assert kwargs["env"]["HERMES_DELEGATE_JOB_ID"] == job_id
    assert kwargs["env"]["HERMES_DELEGATE_APPROVAL_MODE"] == "deny"

    request_path = Path(argv[argv.index("--delegate-request") + 1])
    request = json.loads(request_path.read_text())
    assert request["job_id"] == job_id
    assert request["profile"] == "coder"
    assert request["goal"] == "check profile routing"
    assert request["toolsets"] == ["terminal"]
    assert request["skills"] == ["cluster-topology"]
    assert request["resume"] == "sess-previous"
    assert request["pass_session_id"] is True


def test_start_jobs_bridges_managed_tool_gateway_auth(monkeypatch, tmp_path):
    from tools import delegation_jobs

    hermes_home = tmp_path / "hermes"
    (hermes_home / "profiles" / "coder").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_DELEGATE_COMMAND", "/bin/hermes-test")
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_MANAGED_TOOLS_ENABLED", raising=False)
    monkeypatch.setattr(delegation_jobs.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr("tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr("tools.managed_tool_gateway.read_nous_access_token", lambda: "nous-token")

    delegation_jobs.start_jobs(
        [{"goal": "web research", "profile": "coder", "toolsets": ["web"]}],
        parent_agent=_FakeParent(),
        cfg={"allowed_profiles": ["coder"], "default_profile": "", "job_retention_hours": 24},
    )

    env = _FakePopen.calls[0][1]["env"]
    assert env["TOOL_GATEWAY_USER_TOKEN"] == "nous-token"
    assert env["HERMES_MANAGED_TOOLS_ENABLED"] == "1"


def test_status_prefers_failed_when_process_crashes_after_completed_output():
    from tools import delegation_jobs

    assert delegation_jobs._normalize_status_from_output({"status": "completed"}, -6) == "failed"


def test_start_jobs_rejects_disallowed_profile(monkeypatch, tmp_path):
    from tools import delegation_jobs

    hermes_home = tmp_path / "hermes"
    (hermes_home / "profiles" / "coder").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(ValueError, match="not allowed"):
        delegation_jobs.start_jobs(
            [{"goal": "x", "profile": "coder"}],
            parent_agent=_FakeParent(),
            cfg={"allowed_profiles": ["other"]},
        )


def test_delegate_task_async_default_returns_job_handles(monkeypatch):
    import tools.delegate_tool as delegate_tool

    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {
            "async_default": True,
            "max_iterations": 5,
            "max_concurrent_children": 3,
            "max_spawn_depth": 1,
            "orchestrator_enabled": True,
        },
    )
    monkeypatch.setattr(
        "tools.delegation_jobs.start_jobs",
        lambda tasks, **kwargs: {"status": "running", "job_ids": ["job-test"], "jobs": []},
    )

    raw = delegate_tool.delegate_task(goal="async please", parent_agent=_FakeParent())
    result = json.loads(raw)
    assert result["status"] == "running"
    assert result["job_ids"] == ["job-test"]


def test_delegate_task_profile_override_forces_async(monkeypatch):
    import tools.delegate_tool as delegate_tool

    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {
            "async_default": False,
            "max_iterations": 5,
            "max_concurrent_children": 3,
            "max_spawn_depth": 1,
            "orchestrator_enabled": True,
        },
    )
    seen = {}

    def _start_jobs(tasks, **kwargs):
        seen["tasks"] = tasks
        seen["kwargs"] = kwargs
        return {"status": "running", "job_ids": ["job-profile"], "jobs": []}

    monkeypatch.setattr("tools.delegation_jobs.start_jobs", _start_jobs)

    raw = delegate_tool.delegate_task(
        goal="profile child",
        profile="coder",
        parent_agent=_FakeParent(),
    )
    result = json.loads(raw)
    assert result["job_ids"] == ["job-profile"]
    assert seen["kwargs"]["top_profile"] == "coder"


def test_delegate_task_async_passes_session_controls_to_job_request(monkeypatch):
    import tools.delegate_tool as delegate_tool

    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {
            "async_default": True,
            "max_iterations": 5,
            "max_concurrent_children": 3,
            "max_spawn_depth": 1,
            "orchestrator_enabled": True,
        },
    )
    seen = {}

    def _start_jobs(tasks, **kwargs):
        seen["tasks"] = tasks
        return {"status": "running", "job_ids": ["job-controls"], "jobs": []}

    monkeypatch.setattr("tools.delegation_jobs.start_jobs", _start_jobs)

    raw = delegate_tool.delegate_task(
        goal="use session controls",
        skills="cluster-topology,python-dev",
        resume="sess-previous",
        pass_session_id=True,
        parent_agent=_FakeParent(),
    )

    result = json.loads(raw)
    assert result["job_ids"] == ["job-controls"]
    assert seen["tasks"][0]["skills"] == ["cluster-topology", "python-dev"]
    assert seen["tasks"][0]["resume"] == "sess-previous"
    assert seen["tasks"][0]["pass_session_id"] is True


def test_delegate_result_missing_job(monkeypatch, tmp_path):
    from tools import delegation_jobs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    result = delegation_jobs.get_result("job-missing")
    assert result["status"] == "missing"
