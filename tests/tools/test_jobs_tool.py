"""Tests for tools/jobs_tool.py — the owner-scoped background "jobs" tool.

Coverage: start/list, owner isolation (foreign ids are indistinguishable from
unknown ids), output with wait+timeout → status "running" (never an error),
output after completion, kill, byte truncation with head/tail marker, the
per-owner concurrency cap, notify_on_complete wiring, and registry/toolset
exposure.
"""

import json
import sys
import time
import uuid

import pytest

import tools.jobs_tool as jobs_tool
from tools.jobs_tool import _handle_jobs, truncate_output_bytes
from tools.process_registry import process_registry


def _py(command_body: str) -> str:
    """Build a portable `python -c "<body>"` command string for bash."""
    return f'"{sys.executable}" -c "{command_body}"'


def _new_task() -> str:
    """Unique owner task id per test so tests never see each other's jobs."""
    return f"jobs-test-{uuid.uuid4().hex[:10]}"


@pytest.fixture()
def owner() -> dict:
    """A fresh (session_key, task_id) owner identity for one test."""
    return {"session_key": "sess-a", "task_id": _new_task()}


@pytest.fixture(autouse=True)
def _cleanup_jobs():
    """Kill any jobs this test file spawned (avoids stray processes)."""
    created = []
    yield created
    for job_id in created:
        try:
            session = process_registry.get(job_id)
        except Exception:
            session = None
        if session is not None and not session.exited:
            try:
                process_registry.kill_process(job_id, source="test-cleanup")
            except Exception:
                pass


def _start(owner: dict, command: str, **extra) -> dict:
    """Run jobs start and return the parsed result."""
    args = {"action": "start", "command": command, **extra}
    raw = _handle_jobs(args, **owner)
    return json.loads(raw)


def _list(owner: dict) -> dict:
    return json.loads(_handle_jobs({"action": "list"}, **owner))


def _output(owner: dict, job_id: str, **extra) -> dict:
    return json.loads(_handle_jobs({"action": "output", "job_id": job_id, **extra}, **owner))


def _kill(owner: dict, job_id: str) -> dict:
    return json.loads(_handle_jobs({"action": "kill", "job_id": job_id}, **owner))


def _wait_until(predicate, timeout: float = 15.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# start / list
# ---------------------------------------------------------------------------

def test_start_returns_job_id_and_list_shows_it(owner, _cleanup_jobs):
    result = _start(owner, _py("import time; time.sleep(30)"))
    assert result["status"] == "started"
    job_id = result["job_id"]
    assert job_id.startswith("proc_")
    assert result["pid"]
    _cleanup_jobs.append(job_id)

    listed = _list(owner)
    ids = [j["job_id"] for j in listed["jobs"]]
    assert job_id in ids
    entry = next(j for j in listed["jobs"] if j["job_id"] == job_id)
    assert entry["status"] == "running"
    assert entry["command"].startswith('"')

    # notify_on_complete must be wired so the completion notification lands in
    # the existing background-process channel (requirement: async completion).
    session = process_registry.get(job_id)
    assert session.notify_on_complete is True


def test_start_rejects_missing_command(owner):
    result = json.loads(_handle_jobs({"action": "start"}, **owner))
    assert "error" in result


def test_unknown_action(owner):
    result = json.loads(_handle_jobs({"action": "frobnicate"}, **owner))
    assert "error" in result
    assert "frobnicate" in result["error"]


# ---------------------------------------------------------------------------
# Owner isolation
# ---------------------------------------------------------------------------

def test_list_isolation_between_sessions(_cleanup_jobs):
    owner_a = {"session_key": "sess-a", "task_id": _new_task()}
    owner_b = {"session_key": "sess-b", "task_id": _new_task()}
    started = _start(owner_a, _py("import time; time.sleep(30)"))
    _cleanup_jobs.append(started["job_id"])

    ids_b = [j["job_id"] for j in _list(owner_b)["jobs"]]
    assert started["job_id"] not in ids_b
    ids_a = [j["job_id"] for j in _list(owner_a)["jobs"]]
    assert started["job_id"] in ids_a


def test_output_and_kill_refuse_foreign_job(_cleanup_jobs):
    owner_a = {"session_key": "sess-a", "task_id": _new_task()}
    owner_b = {"session_key": "sess-b", "task_id": _new_task()}
    started = _start(owner_a, _py("import time; time.sleep(30)"))
    _cleanup_jobs.append(started["job_id"])
    job_id = started["job_id"]

    # Foreign output: same error SHAPE as an unknown id — no existence leak.
    # (Both are "No job with ID <id>" errors; a leak would say something like
    # "job belongs to another session".)
    foreign_output = _output(owner_b, job_id, wait=True, timeout_ms=500)
    assert foreign_output["error"] == f"No job with ID {job_id}"
    unknown = _output(owner_b, "proc_doesnotexist", wait=True, timeout_ms=500)
    assert unknown["error"] == "No job with ID proc_doesnotexist"
    assert foreign_output["error"].startswith("No job with ID")
    assert unknown["error"].startswith("No job with ID")
    assert "another session" not in foreign_output["error"]
    assert "not yours" not in foreign_output["error"]

    # Foreign kill is refused; the job is still alive afterwards.
    foreign_kill = _kill(owner_b, job_id)
    assert "error" in foreign_kill
    session = process_registry.get(job_id)
    assert session is not None and not session.exited

    # The owner can still kill it.
    killed = _kill(owner_a, job_id)
    assert killed["status"] == "killed"


# ---------------------------------------------------------------------------
# output: wait + timeout → "running", not an error
# ---------------------------------------------------------------------------

def test_output_wait_timeout_returns_running_not_error(owner, _cleanup_jobs):
    started = _start(owner, _py("import time; time.sleep(30)"))
    _cleanup_jobs.append(started["job_id"])
    job_id = started["job_id"]

    t0 = time.monotonic()
    result = _output(owner, job_id, wait=True, timeout_ms=800)
    elapsed = time.monotonic() - t0

    assert "error" not in result
    assert result["status"] == "running"
    assert elapsed < 10  # returned promptly; did not block for the full 30s
    assert "note" in result  # wait note explains the job is still running

    # The job is still alive after a timed-out wait.
    session = process_registry.get(job_id)
    assert session is not None and not session.exited


def test_output_non_blocking_shows_running_without_wait(owner, _cleanup_jobs):
    started = _start(owner, _py("import time; time.sleep(30)"))
    _cleanup_jobs.append(started["job_id"])

    t0 = time.monotonic()
    result = _output(owner, started["job_id"])
    assert time.monotonic() - t0 < 5
    assert result["status"] == "running"


def test_output_after_completion_returns_output_and_exit_code(owner, _cleanup_jobs):
    started = _start(owner, _py("print('JOB-OUTPUT-MARKER-42'); import time; time.sleep(0.2)"))
    _cleanup_jobs.append(started["job_id"])
    job_id = started["job_id"]

    result = _output(owner, job_id, wait=True, timeout_ms=15000)
    assert "error" not in result
    assert result["status"] == "exited"
    assert "JOB-OUTPUT-MARKER-42" in result["output"]
    assert result["exit_code"] == 0


def test_kill_terminates_running_job(owner, _cleanup_jobs):
    started = _start(owner, _py("import time; time.sleep(60)"))
    _cleanup_jobs.append(started["job_id"])
    job_id = started["job_id"]

    killed = _kill(owner, job_id)
    assert killed["status"] == "killed"

    assert _wait_until(lambda: process_registry.get(job_id).exited)
    listed = _list(owner)
    entry = next(j for j in listed["jobs"] if j["job_id"] == job_id)
    assert entry["status"] == "exited"


def test_kill_already_exited_is_not_an_error(owner, _cleanup_jobs):
    started = _start(owner, _py("print('done')"))
    _cleanup_jobs.append(started["job_id"])
    job_id = started["job_id"]

    # Let it finish, then kill again.
    _output(owner, job_id, wait=True, timeout_ms=15000)
    result = _kill(owner, job_id)
    assert "error" not in result
    assert result["status"] == "already_exited"


# ---------------------------------------------------------------------------
# Byte truncation (head + tail marker)
# ---------------------------------------------------------------------------

def test_truncate_output_bytes_keeps_head_and_tail():
    text = "A" * 1000 + "B" * 1000
    out, truncated = truncate_output_bytes(text, 600)
    assert truncated is True
    assert len(out.encode("utf-8")) <= 600
    assert "[output truncated: showing head and tail]" in out
    assert out.startswith("A" * 200)
    assert out.endswith("B" * 200)


def test_truncate_output_bytes_passthrough_when_small():
    text = "short"
    out, truncated = truncate_output_bytes(text, 1000)
    assert truncated is False
    assert out == text


def test_truncate_output_bytes_multibyte_safe():
    # 1000 multibyte chars; a naive byte cut would split a char. Must decode
    # without raising and stay within the byte bound.
    text = "é" * 1000
    out, truncated = truncate_output_bytes(text, 700)
    assert truncated is True
    assert isinstance(out, str)
    assert len(out.encode("utf-8", errors="replace")) <= 700
    assert "[output truncated: showing head and tail]" in out


def test_output_action_truncates_by_bytes(owner, _cleanup_jobs):
    started = _start(owner, _py("print('X' * 5000)"))
    _cleanup_jobs.append(started["job_id"])
    job_id = started["job_id"]

    result = _output(owner, job_id, wait=True, timeout_ms=15000)
    assert result["status"] == "exited"
    full_len = len(result["output"])

    result = _output(owner, job_id, max_output_bytes=1024)
    assert result["truncated"] is True
    assert result["output_limit_bytes"] == 1024
    assert len(result["output"].encode("utf-8")) <= 1024
    assert "[output truncated: showing head and tail]" in result["output"]
    assert full_len > len(result["output"])


def test_output_action_no_truncation_when_under_limit(owner, _cleanup_jobs):
    started = _start(owner, _py("print('tiny')"))
    _cleanup_jobs.append(started["job_id"])
    job_id = started["job_id"]

    result = _output(owner, job_id, wait=True, timeout_ms=15000)
    assert result["status"] == "exited"
    assert result.get("truncated") is None or result["truncated"] is False
    assert "tiny" in result["output"]


# ---------------------------------------------------------------------------
# Per-owner concurrency cap
# ---------------------------------------------------------------------------

def test_max_concurrent_per_session_cap(owner, _cleanup_jobs, monkeypatch):
    monkeypatch.setattr(jobs_tool, "max_concurrent_jobs_per_session", lambda: 2)

    j1 = _start(owner, _py("import time; time.sleep(30)"))
    j2 = _start(owner, _py("import time; time.sleep(30)"))
    _cleanup_jobs.extend([j1["job_id"], j2["job_id"]])
    assert j1["status"] == "started" and j2["status"] == "started"

    j3 = _start(owner, _py("import time; time.sleep(30)"))
    assert "error" in j3
    assert "limit" in j3["error"]

    # Freeing a slot lets a new job start.
    _kill(owner, j1["job_id"])
    assert _wait_until(lambda: process_registry.get(j1["job_id"]).exited)
    j3 = _start(owner, _py("import time; time.sleep(30)"))
    _cleanup_jobs.append(j3["job_id"])
    assert j3["status"] == "started"


def test_cap_counts_only_running_jobs(owner, _cleanup_jobs, monkeypatch):
    monkeypatch.setattr(jobs_tool, "max_concurrent_jobs_per_session", lambda: 1)

    j1 = _start(owner, _py("print('fast')"))
    _cleanup_jobs.append(j1["job_id"])
    _output(owner, j1["job_id"], wait=True, timeout_ms=15000)  # finished

    # Finished job no longer counts toward the cap.
    j2 = _start(owner, _py("import time; time.sleep(30)"))
    _cleanup_jobs.append(j2["job_id"])
    assert j2["status"] == "started"


# ---------------------------------------------------------------------------
# Registration / exposure
# ---------------------------------------------------------------------------

def test_jobs_tool_registered_in_terminal_toolset():
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    entry = registry.get_entry("jobs")
    assert entry is not None
    assert entry.toolset == "terminal"
    assert entry.schema["name"] == "jobs"
    assert entry.schema["parameters"]["properties"]["action"]["enum"] == [
        "start", "list", "output", "kill",
    ]


def test_jobs_tool_resolves_from_terminal_toolset():
    from toolsets import resolve_toolset

    assert "jobs" in resolve_toolset("terminal")


def test_jobs_schema_includes_guidance():
    from tools.registry import registry

    description = registry.get_entry("jobs").schema["description"]
    # No busy-poll guidance must be visible to the model.
    assert "busy-poll" in description
    assert "running" in description
