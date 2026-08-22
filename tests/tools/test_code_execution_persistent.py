import contextvars
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.code_execution_kernel import (
    KernelDiedError,
    KernelRegistry,
    KernelStartupTimeout,
    PersistentPythonKernel,
    kernel_registry,
)
from tools.code_execution_tool import (
    _execute_code_handler,
    build_execute_code_schema,
    dispose_code_execution_sessions,
    execute_code,
)


@pytest.fixture(autouse=True)
def _clean_kernels(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    kernel_registry.close_all()
    yield
    kernel_registry.close_all()


@pytest.fixture
def config():
    return {
        "mode": "project",
        "kernel_mode": "session",
        "timeout": 10,
        "max_tool_calls": 10,
        "kernel_idle_seconds": 60,
        "max_live_kernels": 8,
    }


def _run(code, scope, config, **kwargs):
    with patch("tools.code_execution_tool._load_config", return_value=config), patch(
        "tools.approval.check_execute_code_guard", return_value={"approved": True}
    ):
        return json.loads(
            execute_code(
                code,
                task_id=f"task-{scope}",
                enabled_tools=[],
                execution_session_id=scope,
                **kwargs,
            )
        )


def test_state_and_last_expression_persist(config):
    first = _run("value = 41", "conversation-a", config)
    second = _run("value + 1", "conversation-a", config)

    assert first["status"] == "success"
    assert first["kernel_reused"] is False
    assert second["status"] == "success"
    assert second["output"].strip() == "42"
    assert second["kernel_reused"] is True
    assert second["persistent"] is True


def test_schema_exposes_reset_and_describes_configured_lifetime():
    persistent = build_execute_code_schema(mode="project", kernel_mode="session")
    per_call = build_execute_code_schema(mode="project", kernel_mode="per_call")

    persistent_description = persistent["description"]
    persistent_code = persistent["parameters"]["properties"]["code"]["description"]
    per_call_description = per_call["description"]
    per_call_code = per_call["parameters"]["properties"]["code"]["description"]

    assert persistent["parameters"]["properties"]["reset"]["type"] == "boolean"
    assert persistent_description.startswith("Run one Python cell")
    assert "conversation-scoped persistent kernel" in persistent_description
    assert "Work incrementally" in persistent_description
    assert "reuse prior names" in persistent_description
    assert "fix and rerun only the failing step" in persistent_description
    assert "Prior top-level names" in persistent_code
    assert "fresh process" in per_call_description
    assert "no prior Python state" in per_call_description
    assert "Prior top-level names" not in per_call_code


def test_registry_handler_forwards_session_scope_and_reset():
    with patch("tools.code_execution_tool.execute_code", return_value="{}") as mocked:
        result = _execute_code_handler(
            {"code": "1 + 1", "reset": True},
            task_id="task",
            session_id="segment",
            code_execution_session_id="conversation:root",
            enabled_tools=["read_file"],
        )

    assert result == "{}"
    assert mocked.call_args.kwargs["execution_session_id"] == "conversation:root"
    assert mocked.call_args.kwargs["reset"] is True


def test_agent_scope_survives_compression_but_isolates_subagents():
    from run_agent import AIAgent

    parent = SimpleNamespace(
        session_id="compressed-segment",
        platform="cli",
        _conversation_root_id=lambda: "conversation-root",
    )
    child = SimpleNamespace(
        session_id="child-session",
        platform="subagent",
        _conversation_root_id=lambda: "conversation-root",
    )

    assert AIAgent._code_execution_scope_id(parent) == "conversation:conversation-root"
    assert AIAgent._code_execution_scope_id(child) == "subagent:child-session"


def test_sessions_are_isolated_and_reset_is_scoped(config):
    _run("value = 'alpha'", "conversation-a", config)
    isolated = _run("'value' in globals()", "conversation-b", config)
    reset = _run("'value' in globals()", "conversation-a", config, reset=True)

    assert isolated["output"].strip() == "False"
    assert reset["output"].strip() == "False"
    assert reset["kernel_reused"] is False


def test_exception_does_not_destroy_kernel_state(config):
    failed = _run(
        "items = [1]; items.append(2); raise ValueError('bad')",
        "errors",
        config,
    )
    recovered = _run("items", "errors", config)

    assert failed["status"] == "error"
    assert "ValueError: bad" in failed["error"]
    assert recovered["status"] == "success"
    assert recovered["output"].strip() == "[1, 2]"
    assert recovered["kernel_reused"] is True


def test_user_tracebacks_hide_runner_frames(config):
    result = _run(
        "def fail():\n    raise ValueError('boom')\nfail()",
        "traceback",
        config,
    )

    assert result["status"] == "error"
    assert "ValueError: boom" in result["error"]
    assert 'File "<cell>"' in result["error"]
    assert "code_execution_kernel.py" not in result["error"]
    assert "_runner_eval" not in result["error"]


def test_syntax_errors_point_to_the_cell_without_runner_frames(config):
    result = _run("if True print('no')", "syntax", config)

    assert result["status"] == "error"
    assert "SyntaxError" in result["error"]
    assert 'File "<cell>"' in result["error"]
    assert "code_execution_kernel.py" not in result["error"]
    assert "_runner_eval" not in result["error"]


def test_input_fails_immediately_without_destroying_state(config):
    started = time.monotonic()
    result = _run("value = 41\ninput('prompt: ')", "input", config)
    recovered = _run("value + 1", "input", config)

    assert time.monotonic() - started < 2
    assert result["status"] == "error"
    assert "input() is unavailable" in result["error"]
    assert recovered["output"].strip() == "42"
    assert recovered["kernel_reused"] is True


def test_top_level_await_and_imports_persist(config):
    _run(
        "import asyncio\nasync def answer():\n    await asyncio.sleep(0)\n    return 9",
        "await",
        config,
    )
    result = _run("await answer()", "await", config)

    assert result["status"] == "success"
    assert result["output"].strip() == "9"


def test_project_cwd_changes_without_resetting_state(config, tmp_path):
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    with patch.dict("os.environ", {"TERMINAL_CWD": str(first_cwd)}):
        _run("value = 42", "moving-cwd", config)
    with patch.dict("os.environ", {"TERMINAL_CWD": str(second_cwd)}):
        result = _run("import os; (os.getcwd(), value)", "moving-cwd", config)

    assert str(second_cwd) in result["output"]
    assert "42" in result["output"]
    assert result["kernel_reused"] is True


def test_project_cwd_modules_are_importable(config, tmp_path):
    (tmp_path / "project_module.py").write_text("VALUE = 42\n", encoding="utf-8")
    with patch.dict("os.environ", {"TERMINAL_CWD": str(tmp_path)}):
        result = _run(
            "import project_module\nproject_module.VALUE",
            "project-import",
            config,
        )

    assert result["status"] == "success"
    assert result["output"].strip() == "42"


def test_per_call_mode_preserves_legacy_fresh_process_behavior(config):
    per_call = {**config, "kernel_mode": "per_call"}
    _run("value = 41", "per-call", per_call)
    result = _run("print('value' in globals())", "per-call", per_call)

    assert result["status"] == "success"
    assert result["output"].strip() == "False"
    assert result["persistent"] is False


def test_cached_hermes_tools_reconnects_with_fresh_turn_context(config):
    turn_value = contextvars.ContextVar("turn_value", default="missing")

    def dispatch(_name, _args, task_id=None):
        return json.dumps({"content": turn_value.get()})

    code = "from hermes_tools import read_file\nprint(read_file('x')['content'])"
    with patch("model_tools.handle_function_call", side_effect=dispatch):
        token = turn_value.set("first-turn")
        try:
            first = _run(code, "rpc", config)
        finally:
            turn_value.reset(token)
        token = turn_value.set("second-turn")
        try:
            second = _run(code, "rpc", config)
        finally:
            turn_value.reset(token)

    assert first["output"].strip() == "first-turn"
    assert second["output"].strip() == "second-turn"
    assert first["tool_calls_made"] == 1
    assert second["tool_calls_made"] == 1


def test_kernel_crash_is_discarded_and_next_call_recovers(config):
    crashed = _run("import os; os._exit(7)", "crash", config)
    recovered = _run("6 * 7", "crash", config)

    assert crashed["status"] == "error"
    assert "kernel stopped unexpectedly" in crashed["error"].lower()
    assert recovered["status"] == "success"
    assert recovered["output"].strip() == "42"
    assert recovered["kernel_reused"] is False


def test_persistent_kernel_captures_subprocess_stdout(config):
    result = _run(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', \"print('child-output')\"])",
        "subprocess-output",
        config,
    )

    assert result["status"] == "success"
    assert "child-output" in result["output"]


def test_persistent_output_keeps_head_tail_and_metadata(config):
    result = _run("print('x' * 60000)\n'final-value'", "large-output", config)

    assert result["status"] == "success"
    assert result["stdout_truncated"] is True
    assert result["stdout_bytes_total"] > result["stdout_bytes_captured"]
    assert "final-value" in result["output"]


def test_persistent_output_sanitizes_ansi_and_carriage_returns(config):
    result = _run(
        "print('\\x1b[31mred\\x1b[0m\\rreplacement')",
        "sanitized-output",
        config,
    )

    assert result["status"] == "success"
    assert "\x1b" not in result["output"]
    assert "\r" not in result["output"]
    assert result["output"].splitlines() == ["red", "replacement"]


def test_timeout_resets_kernel(config):
    timed = {**config, "timeout": 2}
    _run("value = 41", "timeout", timed)
    result = _run("while True:\n    pass", "timeout", timed)
    recovered = _run("'value' in globals()", "timeout", timed)

    assert result["status"] == "timeout"
    assert "kernel was reset" in result["error"]
    assert recovered["output"].strip() == "False"
    assert recovered["kernel_reused"] is False


def test_interrupt_resets_kernel(config, tmp_path):
    from tools.interrupt import set_interrupt

    marker = tmp_path / "started"
    result = {}

    def run_cell():
        result.update(
            _run(
                f"import pathlib, time\npathlib.Path({str(marker)!r}).touch()\n"
                "value = 41\ntime.sleep(30)",
                "interrupt",
                {**config, "timeout": 60},
            )
        )

    thread = threading.Thread(target=run_cell)
    thread.start()
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists()

    set_interrupt(True, thread.ident)
    thread.join(timeout=3)
    set_interrupt(False, thread.ident)
    recovered = _run("'value' in globals()", "interrupt", config)

    assert thread.is_alive() is False
    assert result["status"] == "interrupted"
    assert "kernel reset" in result["output"]
    assert recovered["output"].strip() == "False"
    assert recovered["kernel_reused"] is False


def test_queued_interrupt_does_not_kill_the_running_cell(config, tmp_path):
    from tools.interrupt import set_interrupt

    marker = tmp_path / "running"
    first = {}
    second = {}

    def run_first():
        first.update(
            _run(
                f"import pathlib, time\npathlib.Path({str(marker)!r}).touch()\n"
                "shared = 42\ntime.sleep(1)",
                "queued-interrupt",
                config,
            )
        )

    def run_second():
        second.update(_run("shared + 1", "queued-interrupt", config))

    first_thread = threading.Thread(target=run_first)
    first_thread.start()
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists()
    second_thread = threading.Thread(target=run_second)
    second_thread.start()
    time.sleep(0.1)
    set_interrupt(True, second_thread.ident)
    second_thread.join(timeout=2)
    set_interrupt(False, second_thread.ident)

    assert second_thread.is_alive() is False
    assert second["status"] == "interrupted"
    assert "kernel reset" not in second["output"]

    first_thread.join(timeout=3)
    recovered = _run("shared", "queued-interrupt", config)
    assert first["status"] == "success"
    assert recovered["output"].strip() == "42"
    assert recovered["kernel_reused"] is True


def test_profile_context_is_part_of_kernel_identity(config, tmp_path):
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    token = set_hermes_home_override(home_a)
    try:
        _run("value = 'profile-a'", "same-conversation", config)
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(home_b)
    try:
        profile_b = _run("'value' in globals()", "same-conversation", config)
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(home_a)
    try:
        profile_a = _run("value", "same-conversation", config)
    finally:
        reset_hermes_home_override(token)

    assert profile_b["output"].strip() == "False"
    assert profile_a["output"].strip() == "'profile-a'"


def test_explicit_disposal_closes_session_kernel(config):
    _run("value = 1", "dispose", config)
    assert kernel_registry.size() == 1

    dispose_code_execution_sessions("dispose")

    assert kernel_registry.size() == 0


def test_agent_hard_close_disposes_owned_kernel(config):
    from run_agent import AIAgent

    _run("value = 1", "conversation:hard-close", config)
    agent = object.__new__(AIAgent)
    agent.session_id = "hard-close"
    agent.platform = "cli"
    agent._session_db = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()

    agent.close()
    agent.close()

    assert kernel_registry.size() == 0


def test_agent_close_does_not_dispose_another_session(config):
    from run_agent import AIAgent

    _run("value = 'first'", "conversation:first", config)
    _run("value = 'second'", "conversation:second", config)
    agent = object.__new__(AIAgent)
    agent.session_id = "first"
    agent.platform = "cli"
    agent._session_db = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()

    agent.close()
    first = _run("'value' in globals()", "conversation:first", config)
    second = _run("value", "conversation:second", config)

    assert first["output"].strip() == "False"
    assert first["kernel_reused"] is False
    assert second["output"].strip() == "'second'"
    assert second["kernel_reused"] is True


def test_reset_racing_active_execution_does_not_discard_new_kernel(config, tmp_path):
    marker = tmp_path / "old-running"
    old = {}

    def run_old():
        old.update(
            _run(
                f"import pathlib, time\npathlib.Path({str(marker)!r}).touch()\n"
                "old_value = 1\ntime.sleep(30)",
                "reset-race",
                {**config, "timeout": 60},
            )
        )

    thread = threading.Thread(target=run_old)
    thread.start()
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists()

    reset = _run("new_value = 42", "reset-race", config, reset=True)
    thread.join(timeout=3)
    recovered = _run("new_value", "reset-race", config)

    assert thread.is_alive() is False
    assert old["status"] == "error"
    assert reset["status"] == "success"
    assert recovered["output"].strip() == "42"
    assert recovered["kernel_reused"] is True


def test_disposal_stops_an_active_cell_without_waiting_for_its_timeout(config):
    result = {}

    def run_cell():
        result.update(_run("import time; time.sleep(30)", "active-close", config))

    thread = threading.Thread(target=run_cell)
    thread.start()
    deadline = time.monotonic() + 5
    while kernel_registry.size() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert kernel_registry.size() == 1

    dispose_code_execution_sessions("active-close")
    thread.join(timeout=3)

    assert thread.is_alive() is False
    assert result["status"] == "error"
    assert kernel_registry.size() == 0


def test_disposal_kills_subprocesses_started_by_the_kernel(config):
    import psutil

    result = _run(
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "child.pid",
        "child-process",
        config,
    )
    child_pid = int(result["output"].strip())
    assert psutil.pid_exists(child_pid)

    dispose_code_execution_sessions("child-process")
    deadline = time.monotonic() + 3
    while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert psutil.pid_exists(child_pid) is False


@pytest.mark.linux_only
def test_kernel_crash_kills_its_existing_process_group(config, tmp_path):
    import psutil

    pid_file = tmp_path / "child.pid"
    code = (
        "import os, pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "os._exit(9)"
    )
    result = _run(code, "crash-child", config)
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3
    while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert result["status"] == "error"
    assert psutil.pid_exists(child_pid) is False


class _FakeKernel:
    def __init__(self):
        self.alive = True
        self.closed = False

    def close(self):
        self.closed = True
        self.alive = False


def test_registry_coalesces_concurrent_startup():
    registry = KernelRegistry()
    created = []
    factory_started = threading.Event()
    allow_factory = threading.Event()
    results = []

    def factory():
        kernel = _FakeKernel()
        created.append(kernel)
        factory_started.set()
        assert allow_factory.wait(timeout=3)
        return kernel

    def acquire():
        results.append(
            registry.acquire(
                ("shared",),
                scope_id="shared",
                profile_key="profile",
                factory=factory,
                max_live=8,
                idle_seconds=60,
                deadline=time.monotonic() + 5,
            )
        )

    first = threading.Thread(target=acquire)
    second = threading.Thread(target=acquire)
    first.start()
    assert factory_started.wait(timeout=2)
    second.start()
    allow_factory.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert first.is_alive() is False
    assert second.is_alive() is False
    assert len(created) == 1
    assert len(results) == 2
    assert sorted(reused for _, reused in results) == [False, True]
    registry.close_all()


def test_registry_startup_does_not_block_other_scopes():
    registry = KernelRegistry()
    first_started = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()

    def first_factory():
        first_started.set()
        assert release_first.wait(timeout=3)
        return _FakeKernel()

    def acquire_first():
        registry.acquire(
            ("first",),
            scope_id="first",
            profile_key="profile",
            factory=first_factory,
            max_live=8,
            idle_seconds=60,
        )

    def acquire_second():
        registry.acquire(
            ("second",),
            scope_id="second",
            profile_key="profile",
            factory=_FakeKernel,
            max_live=8,
            idle_seconds=60,
        )
        second_finished.set()

    first = threading.Thread(target=acquire_first)
    second = threading.Thread(target=acquire_second)
    first.start()
    assert first_started.wait(timeout=2)
    second.start()
    assert second_finished.wait(timeout=1)
    release_first.set()
    first.join(timeout=3)
    second.join(timeout=3)
    registry.close_all()


def test_dispose_invalidates_inflight_startup():
    registry = KernelRegistry()
    factory_started = threading.Event()
    allow_factory = threading.Event()
    created = _FakeKernel()
    error = []

    def factory():
        factory_started.set()
        assert allow_factory.wait(timeout=3)
        return created

    def acquire():
        try:
            registry.acquire(
                ("pending",),
                scope_id="pending",
                profile_key="profile",
                factory=factory,
                max_live=8,
                idle_seconds=60,
            )
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=acquire)
    thread.start()
    assert factory_started.wait(timeout=2)
    registry.dispose_scope("pending", "profile")
    allow_factory.set()
    thread.join(timeout=3)

    assert thread.is_alive() is False
    assert len(error) == 1
    assert isinstance(error[0], KernelDiedError)
    assert created.closed is True
    assert registry.size() == 0


def test_startup_timeout_is_structured_and_leaves_no_registry_entry(config):
    class SlowKernel:
        def __init__(self, *_args, deadline, **_kwargs):
            time.sleep(max(0, deadline - time.monotonic()) + 0.01)
            raise KernelStartupTimeout("slow startup")

    started = time.monotonic()
    with patch("tools.code_execution_kernel.PersistentPythonKernel", SlowKernel):
        result = _run(
            "42",
            "startup-timeout",
            {**config, "timeout": 0.2},
        )
    elapsed = time.monotonic() - started
    recovered = _run("42", "startup-timeout", config)

    assert elapsed < 1
    assert result["status"] == "timeout"
    assert "did not start within" in result["error"]
    assert recovered["status"] == "success"
    assert recovered["output"].strip() == "42"
    assert recovered["kernel_reused"] is False


def test_failed_startup_terminates_process_and_removes_staging_dir(tmp_path):
    real_popen = subprocess.Popen
    real_mkdtemp = tempfile.mkdtemp
    processes = []
    staging_dirs = []

    def tracked_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def tracked_mkdtemp(*, prefix):
        path = real_mkdtemp(prefix=prefix, dir=tmp_path)
        staging_dirs.append(path)
        return path

    with patch(
        "tools.code_execution_kernel.subprocess.Popen",
        side_effect=tracked_popen,
    ), patch(
        "tools.code_execution_kernel.tempfile.mkdtemp",
        side_effect=tracked_mkdtemp,
    ), patch.object(
        PersistentPythonKernel,
        "_wait_until_ready",
        side_effect=KernelStartupTimeout("forced timeout"),
    ):
        with pytest.raises(KernelStartupTimeout):
            PersistentPythonKernel(
                sys.executable,
                str(tmp_path),
                os.environ.copy(),
                "",
                deadline=time.monotonic() + 1,
            )

    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert len(staging_dirs) == 1
    assert os.path.exists(staging_dirs[0]) is False


def test_registry_lru_bound_and_idle_reaping():
    registry = KernelRegistry()
    first = _FakeKernel()
    second = _FakeKernel()
    registry.acquire(
        ("first",),
        scope_id="a",
        profile_key="p",
        factory=lambda: first,
        max_live=1,
        idle_seconds=1,
    )
    registry.release(("first",))
    registry.acquire(
        ("second",),
        scope_id="b",
        profile_key="p",
        factory=lambda: second,
        max_live=1,
        idle_seconds=1,
    )
    registry.release(("second",))

    assert first.closed is True
    assert registry.size() == 1
    assert registry.reap_idle(now=time.monotonic() + 2) == 1
    assert second.closed is True
    assert registry.size() == 0
